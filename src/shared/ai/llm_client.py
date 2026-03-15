"""Unified LLM client using LiteLLM for provider abstraction.

This module provides a unified interface for text generation and embeddings
that works with multiple LLM providers (Ollama, OpenAI, Anthropic, etc.).

Usage:
    from shared.ai.llm_client import LLMClient, EmbeddingClient

    # Text generation
    llm = LLMClient(model="ollama/qwen2.5-coder:3b")
    response = llm.generate("Explain this code...")

    # Embeddings
    embed = EmbeddingClient(model="ollama/qwen3-embedding:0.6b")
    vector = embed.embed("Some text to embed")

    # Cleanup on shutdown (optional but recommended)
    import atexit
    atexit.register(cleanup_llm_sessions)
"""

from __future__ import annotations

import asyncio
import atexit
import contextlib
import logging
import random
import re
import signal
import time
import warnings
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from shared.config import ModelConfig

import aiohttp
import httpx
import litellm
from litellm import aembedding, completion, embedding
from litellm.llms.custom_httpx.aiohttp_handler import BaseLLMAIOHTTPHandler

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Disable LiteLLM telemetry and debug info
litellm.telemetry = False
litellm.suppress_debug_info = True

# Suppress Pydantic 2.12+ serialization warnings from LiteLLM
# See: https://github.com/BerriAI/litellm/issues/11759
warnings.filterwarnings("ignore", message="Pydantic serializer warnings")

# Suppress "coroutine 'ollama_aembeddings' was never awaited" warnings from LiteLLM.
# LiteLLM creates async coroutines internally even for sync embedding calls,
# and these aren't properly cleaned up on Ctrl+C or shutdown.
# See: https://github.com/BerriAI/litellm/issues/13220
warnings.filterwarnings("ignore", message="coroutine .* was never awaited", category=RuntimeWarning)

# Note: Python 3.14 may show "_UnixSelectorEventLoop has no attribute '_ssock'"
# errors during shutdown. These are harmless GC cleanup messages from event loops
# created internally by LiteLLM/httpx that weren't fully initialized.
# See: https://github.com/BerriAI/litellm/issues/13220

# =============================================================================
# SYNC CONNECTION POOL (for completion/embedding calls)
# =============================================================================
# Configure httpx client for sync calls with connection pooling
# This reduces overhead from creating new connections for each request
# See: https://docs.litellm.ai/docs/providers/openai
_httpx_client = httpx.Client(
    limits=httpx.Limits(
        max_connections=100,
        max_keepalive_connections=20,
    ),
    # No client-side read timeout — Ollama queues requests internally and may
    # take arbitrarily long when saturated.  Keep a connect timeout so we
    # detect unreachable servers quickly.
    timeout=httpx.Timeout(None, connect=30.0),
)
litellm.client_session = _httpx_client


# =============================================================================
# ASYNC CONNECTION POOL MANAGEMENT
# =============================================================================

# Global aiohttp session for async connection pooling
_aiohttp_session: aiohttp.ClientSession | None = None
_session_lock: asyncio.Lock | None = None  # Created lazily to avoid event loop issues


async def _get_or_create_session_async() -> aiohttp.ClientSession:
    """Get or create the global aiohttp session with connection pooling.

    This session is reused across all async LiteLLM requests for better performance.
    The session uses keep-alive connections and a connection pool.

    Note: Only call this from async context. Sync LiteLLM calls use httpx internally.

    Returns:
        Configured aiohttp.ClientSession instance.
    """
    global _aiohttp_session, _session_lock

    # Create lock lazily to avoid "no running event loop" error at import time
    if _session_lock is None:
        _session_lock = asyncio.Lock()

    async with _session_lock:
        if _aiohttp_session is None or _aiohttp_session.closed:
            connector = aiohttp.TCPConnector(
                limit=100,  # Total connection pool size
                limit_per_host=20,  # Per-host connection limit
                ttl_dns_cache=300,  # DNS cache TTL in seconds
                keepalive_timeout=60,  # Keep connections alive for 60s
                enable_cleanup_closed=True,  # Clean up closed connections
            )
            timeout = aiohttp.ClientTimeout(
                total=None,  # No client-side read timeout (same as httpx)
                connect=30,  # Connection timeout
            )
            _aiohttp_session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
            )
            # Configure LiteLLM to use our managed session for async calls
            # This may fail on newer LiteLLM versions with changed API
            with contextlib.suppress(TypeError):
                litellm.base_llm_aiohttp_handler = BaseLLMAIOHTTPHandler(
                    client_session=_aiohttp_session
                )

    return _aiohttp_session


def cleanup_llm_sessions() -> None:
    """Close global HTTP sessions (sync httpx and async aiohttp).

    Call this function during application shutdown to cleanly close
    all HTTP connections. Can be registered with atexit:

        import atexit
        atexit.register(cleanup_llm_sessions)
    """
    global _aiohttp_session, _httpx_client

    # Close sync httpx client
    if _httpx_client is not None:
        try:
            _httpx_client.close()
        except Exception:
            logger.debug("Failed to close httpx client during cleanup", exc_info=True)

    # Close async aiohttp session
    if _aiohttp_session is not None and not _aiohttp_session.closed:
        # For sync cleanup, we need to run in event loop
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(_aiohttp_session.close())
        except RuntimeError:
            # No running loop, create one for cleanup
            try:
                asyncio.run(_aiohttp_session.close())
            except Exception:
                logger.debug("Failed to close aiohttp session during cleanup", exc_info=True)
        _aiohttp_session = None


async def cleanup_llm_sessions_async() -> None:
    """Async version of session cleanup.

    Use this in async applications for proper cleanup:

        await cleanup_llm_sessions_async()
    """
    global _aiohttp_session

    if _aiohttp_session is not None and not _aiohttp_session.closed:
        await _aiohttp_session.close()
        _aiohttp_session = None


# Register cleanup on interpreter exit
atexit.register(cleanup_llm_sessions)


def _sigint_handler(_signum: int, _frame: Any) -> None:
    """Handle SIGINT (Ctrl+C) with proper cleanup before exit.

    This ensures HTTP sessions are closed before Python starts
    garbage collection, avoiding "coroutine was never awaited" warnings.
    """
    # Restore default handler so subsequent SIGINTs (e.g. during thread
    # shutdown joins) don't re-enter this handler and crash the shutdown.
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    cleanup_llm_sessions()
    # Raise KeyboardInterrupt to let the application handle it normally
    raise KeyboardInterrupt


# Install SIGINT handler only if we're in the main thread
# (signal handlers can only be installed from the main thread)
try:
    if hasattr(signal, "SIGINT"):
        signal.signal(signal.SIGINT, _sigint_handler)
except ValueError:
    # Not in main thread, skip signal handler installation
    pass


class LLMError(Exception):
    """Raised when LLM operations fail."""

    pass


# =============================================================================
# RETRY HELPERS
# =============================================================================


def _is_retryable_error(e: Exception) -> bool:
    """Check if an error is retryable (connection/timeout issues)."""
    error_str = str(e).lower()
    retryable_patterns = [
        "connection",
        "timeout",
        "temporarily unavailable",
        "service unavailable",
        "rate limit",
        "too many requests",
        "503",
        "502",
        "504",
        "429",
    ]
    return any(pattern in error_str for pattern in retryable_patterns)


def _retry_with_backoff(
    fn: Callable[[], T],
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    operation: str = "LLM request",
) -> T:
    """Execute function with exponential backoff retry.

    Args:
        fn: Function to execute.
        max_retries: Maximum number of retry attempts.
        base_delay: Initial delay between retries in seconds.
        max_delay: Maximum delay between retries.
        operation: Description for error messages.

    Returns:
        Result from successful function call.

    Raises:
        LLMError: If all retries are exhausted.
    """
    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as e:
            last_error = e

            # Don't retry on non-retryable errors
            if not _is_retryable_error(e):
                raise LLMError(f"{operation} failed: {e}") from e

            # Don't sleep after last attempt
            if attempt < max_retries:
                # Exponential backoff with jitter
                delay = min(base_delay * (2**attempt) + random.uniform(0, 1), max_delay)
                time.sleep(delay)

    raise LLMError(f"{operation} failed after {max_retries + 1} attempts: {last_error}") from last_error


async def _retry_with_backoff_async(
    fn: Callable[[], Awaitable[T]],
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    operation: str = "LLM request",
) -> T:
    """Execute async function with exponential backoff retry.

    Args:
        fn: Async function to execute.
        max_retries: Maximum number of retry attempts.
        base_delay: Initial delay between retries in seconds.
        max_delay: Maximum delay between retries.
        operation: Description for error messages.

    Returns:
        Result from successful function call.

    Raises:
        LLMError: If all retries are exhausted.
    """
    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            return await fn()
        except Exception as e:
            last_error = e

            # Don't retry on non-retryable errors
            if not _is_retryable_error(e):
                raise LLMError(f"{operation} failed: {e}") from e

            # Don't sleep after last attempt
            if attempt < max_retries:
                # Exponential backoff with jitter
                delay = min(base_delay * (2**attempt) + random.uniform(0, 1), max_delay)
                await asyncio.sleep(delay)

    raise LLMError(f"{operation} failed after {max_retries + 1} attempts: {last_error}") from last_error


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class LLMConfig:
    """Configuration for LLM client."""

    # Model identifier (format: provider/model or just model for OpenAI)
    # Examples:
    #   - "ollama/qwen3.5:4b"
    #   - "gpt-4o-mini"
    #   - "claude-3-haiku-20240307"
    model: str = "ollama/qwen3.5:4b"

    # API configuration
    api_base: str | None = None  # For Ollama: "http://localhost:11434"
    api_key: str | None = None  # For cloud providers

    # Generation settings — Qwen3.5 "Precise Coding Tasks" preset
    # huggingface.co/Qwen/Qwen3.5-4B
    temperature: float = 0.6
    max_tokens: int = 512
    timeout: int = 180  # 3 minutes to handle queue wait with many workers

    # Retry settings
    max_retries: int = 3

    @classmethod
    def from_ollama_url(cls, url: str, model: str) -> LLMConfig:
        """Create config for Ollama provider.

        Args:
            url: Ollama server URL (e.g., "http://localhost:11434")
            model: Model name (e.g., "qwen2.5-coder:3b")

        Returns:
            LLMConfig configured for Ollama.
        """
        return cls(
            model=f"ollama/{model}",
            api_base=url,
        )


@dataclass
class EmbeddingConfig:
    """Configuration for embedding client."""

    # Model identifier
    # Examples:
    #   - "ollama/qwen3-embedding:0.6b"
    #   - "text-embedding-3-small"
    model: str = "ollama/qwen3-embedding:0.6b"

    # API configuration
    api_base: str | None = None
    api_key: str | None = None

    # Vector settings
    dimensions: int = 1024

    # Request settings
    timeout: int = 30
    max_retries: int = 3

    @classmethod
    def from_ollama_url(cls, url: str, model: str, dimensions: int = 1024) -> EmbeddingConfig:
        """Create config for Ollama embedding provider.

        Args:
            url: Ollama server URL
            model: Embedding model name
            dimensions: Expected embedding dimensions

        Returns:
            EmbeddingConfig configured for Ollama.
        """
        return cls(
            model=f"ollama/{model}",
            api_base=url,
            dimensions=dimensions,
        )


# =============================================================================
# LLM CLIENT (TEXT GENERATION)
# =============================================================================


class LLMClient:
    """Unified LLM client for text generation using LiteLLM.

    Supports multiple providers through LiteLLM:
    - Ollama: model="ollama/qwen2.5-coder:3b", api_base="http://localhost:11434"
    - OpenAI: model="gpt-4o-mini", api_key="sk-..."
    - Anthropic: model="claude-3-haiku-20240307", api_key="sk-ant-..."
    - And many more (see LiteLLM docs)
    """

    def __init__(
        self,
        model: str,
        api_base: str | None = None,
        api_key: str | None = None,
        max_retries: int = 3,
        model_name: str | None = None,
    ):
        """Initialize LLM client.

        Args:
            model: LiteLLM model identifier (e.g., "ollama/qwen3:4b", "openai/default").
            api_base: API base URL (required for Ollama, optional for cloud).
            api_key: API key (required for cloud providers).
            max_retries: Maximum retry attempts for transient failures.
            model_name: Original model name for thinking model detection.
                When the LiteLLM identifier differs from the actual model
                (e.g., "openai/Qwen3.5-4B-Q4_K_M" for llamacpp), pass the
                real name here so thinking model detection works correctly.

        Prefer ``from_model_config()`` when a ``ModelConfig`` is available —
        it handles all provider-specific translation automatically.
        """
        self.model = model
        self.api_base = api_base
        self.api_key = api_key
        self.max_retries = max_retries
        self.model_name = model_name

        # Extract provider from model string
        self.provider = model.split("/")[0] if "/" in model else "openai"

        # Pre-compute thinking model flag once (instead of every generate call).
        # Uses model_name (real name) if available, otherwise the LiteLLM identifier.
        _raw = model_name or model
        _base = _raw.split("/")[-1].lower()
        self._is_thinking_model = self._check_is_thinking_model(_base)

        # Compiled regex for stripping <think> blocks (only used for thinking models)
        self._think_re = re.compile(r"<think>.*?</think>", re.DOTALL)

    @staticmethod
    def _check_is_thinking_model(base: str) -> bool:
        """Check if a model base name is a thinking model.

        qwen3 uses thinking by default and needs think=False.
        qwen3.5+ has reasoning disabled by default — NOT a thinking model.
        Instruct variants (e.g., Qwen3-4B-Instruct-2507-4bit) have thinking
        disabled at the model level — NOT a thinking model.
        """
        # Check exclusions first (qwen3.5 matches "qwen3" prefix but is NOT a thinking model)
        if any(base.startswith(ntm) for ntm in LLMClient.NON_THINKING_MODELS):
            return False
        # Instruct models have thinking disabled at the model level
        if "instruct" in base:
            return False
        return any(base.startswith(tm) for tm in LLMClient.THINKING_MODELS)

    def _strip_think_tags(self, text: str) -> str:
        """Strip <think>...</think> blocks and leaked thinking from output.

        Thinking models (Qwen3, DeepSeek-R1) may still emit <think> tags even
        when thinking is disabled via API params — e.g. when the inference
        server doesn't honor ``chat_template_kwargs``.  Stripping these blocks
        prevents reasoning text from polluting downstream parsers (variable
        scoring, summarization) and avoids wasting the tight token budget on
        hidden reasoning.

        Also handles "leaked thinking" where the model outputs reasoning
        followed by a lone ``</think>`` (without ``<think>``).  This happens
        on Ollama when the template hardcodes the ``<think>`` primer but
        ``think=false`` puts everything into the content field.
        """
        text = self._think_re.sub("", text).strip()
        # Strip leaked thinking (everything before a lone </think>)
        if "</think>" in text:
            text = text.split("</think>", 1)[1].strip()
        return text

    def _extract_content(self, message: Any, use_model: str) -> str:
        """Extract text content from an LLM response message.

        Handles thinking models where the server may split output into
        ``content`` (actual answer) and ``reasoning_content`` (thinking).
        Falls back to ``reasoning_content`` when ``content`` is empty —
        this covers cases where the server ignores enable_thinking=False
        (e.g., llama-server with Qwen3.5 GGUF chat templates that enable
        thinking by default).

        Args:
            message: The response message object from LiteLLM.
            use_model: Model identifier (for error messages).

        Returns:
            Extracted text with <think> tags stripped if present.

        Raises:
            LLMError: If no usable content is found.
        """
        content = message.content

        # Fallback to reasoning/thinking content for ANY model when content is empty.
        # Servers may split output into separate fields depending on provider:
        # - llama-server: reasoning_content (via LiteLLM mapping)
        # - Ollama: thinking (native /api/chat field, mapped by LiteLLM)
        # This covers cases where the GGUF chat template enables thinking by
        # default and the model spends its entire token budget on reasoning,
        # returning empty content.
        if not content:
            # Try reasoning_content (llama-server) then thinking (Ollama)
            reasoning = getattr(message, "reasoning_content", None)
            if not (isinstance(reasoning, str) and reasoning.strip()):
                reasoning = getattr(message, "thinking", None)
            if isinstance(reasoning, str) and reasoning.strip():
                logger.warning(
                    "Model '%s' returned empty content but has reasoning/thinking "
                    "(%d chars) — using it as fallback. "
                    "Consider enabling thinking suppression for this model.",
                    use_model, len(reasoning),
                )
                content = reasoning

        if not content:
            raise LLMError(f"Empty response from model '{use_model}'")

        text = content.strip()
        # Always strip thinking content — covers both matched <think>...</think>
        # pairs and leaked thinking (content before a lone </think> tag).
        text = self._strip_think_tags(text)
        return text

    @classmethod
    def from_model_config(cls, config: ModelConfig) -> LLMClient:
        """Create client from a ModelConfig.

        This is the preferred constructor — ModelConfig handles all
        provider-specific translation (LiteLLM identifier, API base URL,
        thinking model detection).

        Args:
            config: Model configuration from magaldi config.
        """
        return cls(
            model=config.get_litellm_model(),
            api_base=config.get_api_base(),
            api_key=config.api_key,
            model_name=config.name,
        )

    @classmethod
    def from_config(cls, config: LLMConfig) -> LLMClient:
        """Create client from LLMConfig."""
        return cls(
            model=config.model,
            api_base=config.api_base,
            api_key=config.api_key,
            max_retries=config.max_retries,
        )

    @classmethod
    def from_ollama(cls, url: str, model: str, max_retries: int = 3) -> LLMClient:
        """Create client for Ollama provider.

        Args:
            url: Ollama server URL
            model: Model name without provider prefix
            max_retries: Maximum retry attempts for transient failures.

        Returns:
            LLMClient configured for Ollama.
        """
        return cls(
            model=f"ollama/{model}",
            api_base=url,
            max_retries=max_retries,
        )

    def _build_kwargs(
        self,
        use_model: str,
        messages: list[dict[str, str]],
        temperature: float,
        top_p: float,
        max_tokens: int,
        timeout: int,
        num_ctx: int | None,
        top_k: int | None = None,
        min_p: float | None = None,
        presence_penalty: float | None = None,
        repetition_penalty: float | None = None,
    ) -> dict[str, Any]:
        """Build the kwargs dict for litellm.completion().

        Centralises auth, thinking-model suppression, and num_ctx handling
        so that generate() and generate_from_messages() stay thin.
        """
        # For local providers (Ollama), disable client-side timeout.
        # Ollama queues requests internally and may take arbitrarily long
        # when saturated — client-side timeouts cause spurious failures.
        # LiteLLM clamps None/0 to 600s, so we use a large sentinel instead.
        _is_ollama = use_model.startswith(("ollama/", "ollama_chat/"))
        effective_timeout: float = 86400.0 if _is_ollama else float(timeout)

        kwargs: dict[str, Any] = {
            "model": use_model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "timeout": effective_timeout,
        }

        # Optional sampling parameters (from model-specific configs)
        # For Ollama, presence_penalty must go through extra_body — LiteLLM's
        # ollama_chat provider does not support it as a top-level parameter.
        if presence_penalty is not None and presence_penalty != 0.0:
            if _is_ollama:
                kwargs.setdefault("extra_body", {})
                kwargs["extra_body"]["presence_penalty"] = presence_penalty
            else:
                kwargs["presence_penalty"] = presence_penalty
        if repetition_penalty is not None and repetition_penalty != 1.0:
            # LiteLLM passes repetition_penalty via extra_body for Ollama
            kwargs.setdefault("extra_body", {})
            kwargs["extra_body"]["repetition_penalty"] = repetition_penalty
        if top_k is not None or min_p is not None:
            kwargs.setdefault("extra_body", {})
            if top_k is not None:
                kwargs["extra_body"]["top_k"] = top_k
            if min_p is not None:
                kwargs["extra_body"]["min_p"] = min_p

        # Auth
        if self.api_base:
            kwargs["api_base"] = self.api_base
        if self.api_key:
            kwargs["api_key"] = self.api_key
        elif use_model.startswith("openai/") and self.api_base:
            # Local OpenAI-compatible servers don't need auth
            # but LiteLLM requires an API key for openai/ prefix
            kwargs["api_key"] = "not-needed"

        # Helper to check if model is a local server (not cloud)
        _is_local = use_model.startswith(("openai/", "lm_studio/")) and self.api_base
        # llamacpp/vllm-mlx both use openai/ prefix with a local api_base.
        # LM Studio has its own lm_studio/ prefix.
        _is_llamacpp = use_model.startswith("openai/") and self.api_base

        # Disable thinking mode for local and thinking models.
        # GGUF chat templates on llama-server may enable thinking by default
        # even for models like Qwen3.5 that Ollama serves without thinking.
        # Always send enable_thinking=False for local servers — it's harmless
        # for models that don't support thinking (param is ignored), and
        # prevents wasting the token budget on reasoning_content.
        if self._is_thinking_model and use_model.startswith("ollama_chat/"):
            # Native /api/chat supports "think" parameter directly
            kwargs["think"] = False
            # Also prepend /no_think to system message as a fallback.
            # Ollama's Qwen3 template hardcodes <think> primer regardless of
            # think=false, so the model still enters thinking mode and wastes
            # the entire token budget on reasoning, returning empty content.
            # /no_think is a soft-switch that Qwen3 honors at the prompt level.
            msgs = kwargs["messages"]
            if msgs and msgs[0].get("role") == "system":
                content = msgs[0]["content"]
                if "/no_think" not in content:
                    msgs = [{"role": "system", "content": f"/no_think\n{content}"}] + msgs[1:]
                    kwargs["messages"] = msgs
            else:
                msgs = [{"role": "system", "content": "/no_think"}] + msgs
                kwargs["messages"] = msgs
        elif _is_llamacpp:
            # llama-server / vllm-mlx (openai/ prefix + local api_base).
            # Qwen3.5 GGUF templates enable thinking by default on llama-server
            # even though Ollama's Qwen3.5 Modelfile has it disabled.
            kwargs["extra_body"] = kwargs.get("extra_body", {})
            # Strategy 1: chat_template_kwargs.enable_thinking=False
            # Tells llama-server to render the template with thinking off.
            kwargs["extra_body"]["chat_template_kwargs"] = {"enable_thinking": False}
            # Strategy 2: reasoning_format=none (llama-server specific)
            # Qwen3.5 templates output <think>\n\n</think> even with
            # enable_thinking=false, and the model still reasons into
            # reasoning_content. With format=none, any thinking stays in
            # message.content where _strip_think_tags() removes it.
            # NOTE: LM Studio does NOT support this param and will error.
            kwargs["extra_body"]["reasoning_format"] = "none"
            # Strategy 3: /no_think in system message
            # Qwen3 models honor /no_think as a chat template directive.
            # Qwen3.5 ignores it but it's harmless.
            msgs = kwargs["messages"]
            if msgs and msgs[0].get("role") == "system":
                content = msgs[0]["content"]
                if "/no_think" not in content:
                    msgs = [{"role": "system", "content": f"/no_think\n{content}"}] + msgs[1:]
                    kwargs["messages"] = msgs
            else:
                msgs = [{"role": "system", "content": "/no_think"}] + msgs
                kwargs["messages"] = msgs
        elif _is_local:
            # LM Studio and other local OpenAI-compatible servers.
            # LM Studio does NOT support chat_template_kwargs or
            # reasoning_format — sending them causes request errors.
            # Use /no_think in system message as a safe fallback.
            msgs = kwargs["messages"]
            if msgs and msgs[0].get("role") == "system":
                content = msgs[0]["content"]
                if "/no_think" not in content:
                    msgs = [{"role": "system", "content": f"/no_think\n{content}"}] + msgs[1:]
                    kwargs["messages"] = msgs
            else:
                msgs = [{"role": "system", "content": "/no_think"}] + msgs
                kwargs["messages"] = msgs
        elif self._is_thinking_model:
            # Cloud thinking models — disable via extra_body
            kwargs["extra_body"] = kwargs.get("extra_body", {})
            kwargs["extra_body"]["chat_template_kwargs"] = {"enable_thinking": False}

        # Context window for local providers
        # Note: Only Ollama supports per-request context sizing via num_ctx.
        # LM Studio / llama.cpp servers set context at model load time —
        # sending n_ctx in extra_body causes them to reject requests that
        # exceed the specified value, even if the model was loaded with a
        # larger context.
        if num_ctx and use_model.startswith(("ollama/", "ollama_chat/")):
            kwargs["num_ctx"] = num_ctx

        return kwargs

    def verify_model(self) -> bool:
        """Check if model is available.

        Returns:
            True if model is accessible.
        """
        try:
            # Try a minimal completion to verify
            self.generate("test", max_tokens=1, timeout=10)
            return True
        except Exception:
            logger.debug("LLM health check failed for model %s", self.model, exc_info=True)
            return False

    # Models that use thinking/reasoning tags by default.
    # qwen3 thinks by default but qwen3.5+ has reasoning disabled by default.
    THINKING_MODELS = ("qwen3", "deepseek-r1", "deepseek-coder-v2", "nemotron")

    # Models that are NOT thinking models despite matching a THINKING_MODELS prefix.
    # qwen3.5+ has reasoning disabled by default — no think=False needed.
    NON_THINKING_MODELS = ("qwen3.5", "qwen3.6", "qwen3.7", "qwen3.8", "qwen3.9",
                           "qwen3-embedding", "qwen3-coder")

    def generate(
        self,
        prompt: str,
        temperature: float = 0.2,
        top_p: float = 0.95,
        max_tokens: int = 512,
        timeout: int = 60,
        model: str | None = None,
        num_ctx: int | None = None,
        top_k: int | None = None,
        min_p: float | None = None,
        presence_penalty: float | None = None,
        repetition_penalty: float | None = None,
    ) -> str:
        """Generate text completion.

        Args:
            prompt: The prompt to send to the model.
            temperature: Sampling temperature (0.0 to 1.0).
            top_p: Nucleus sampling parameter (0.0 to 1.0).
            max_tokens: Maximum tokens to generate.
            timeout: Request timeout in seconds.
            model: Optional model override.
            num_ctx: Context window size for local providers (Ollama, llama.cpp).
                     For Ollama, this triggers use of tiered model aliases to avoid
                     reload overhead when context size changes.
            top_k: Top-k sampling parameter.
            min_p: Min-p sampling parameter.
            presence_penalty: Presence penalty (0.0 to 2.0).
            repetition_penalty: Repetition penalty.

        Returns:
            Generated text.

        Raises:
            LLMError: If generation fails after retries.
        """
        use_model = model or self.model

        # For Ollama, resolve to tiered model alias based on num_ctx
        if use_model.startswith(("ollama/", "ollama_chat/")) and num_ctx:
            from shared.ai.ollama_models import resolve_ollama_model
            use_model, num_ctx = resolve_ollama_model(use_model, self.api_base, num_ctx)

        def _do_generate() -> str:
            kwargs = self._build_kwargs(use_model, [{"role": "user", "content": prompt}],
                                        temperature, top_p, max_tokens, timeout, num_ctx,
                                        top_k, min_p, presence_penalty, repetition_penalty)

            try:
                response = completion(**kwargs)
            except Exception as e:
                debug_kwargs = {k: v for k, v in kwargs.items() if k != "messages"}
                msg_summary = [(m.get("role"), len(m.get("content", ""))) for m in kwargs.get("messages", [])]
                logger.warning(
                    "LiteLLM completion failed: %s | model=%s msgs=%s kwargs=%s",
                    e, use_model, msg_summary, debug_kwargs,
                )
                raise
            return self._extract_content(response.choices[0].message, use_model)

        return _retry_with_backoff(
            _do_generate,
            max_retries=self.max_retries,
            operation=f"LLM generation ({use_model})",
        )

    def generate_from_messages(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        top_p: float = 0.95,
        max_tokens: int = 512,
        timeout: int = 60,
        model: str | None = None,
        num_ctx: int | None = None,
        top_k: int | None = None,
        min_p: float | None = None,
        presence_penalty: float | None = None,
        repetition_penalty: float | None = None,
    ) -> str:
        """Generate text completion from messages (system + user).

        This method is optimized for Ollama's KV cache prefix caching.
        The system message is static and gets cached, while the user
        message contains variable content.

        Args:
            messages: List of message dicts with 'role' and 'content'.
                      Typically [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
            temperature: Sampling temperature (0.0 to 1.0).
            top_p: Nucleus sampling parameter (0.0 to 1.0).
            max_tokens: Maximum tokens to generate.
            timeout: Request timeout in seconds.
            model: Optional model override.
            num_ctx: Context window size for local providers (Ollama, llama.cpp).
                     For Ollama, this triggers use of tiered model aliases to avoid
                     reload overhead when context size changes.
            top_k: Top-k sampling parameter.
            min_p: Min-p sampling parameter.
            presence_penalty: Presence penalty (0.0 to 2.0).
            repetition_penalty: Repetition penalty.

        Returns:
            Generated text.

        Raises:
            LLMError: If generation fails after retries.
        """
        use_model = model or self.model

        # For Ollama, resolve to tiered model alias based on num_ctx
        if use_model.startswith(("ollama/", "ollama_chat/")) and num_ctx:
            from shared.ai.ollama_models import resolve_ollama_model
            use_model, num_ctx = resolve_ollama_model(use_model, self.api_base, num_ctx)

        def _do_generate() -> str:
            kwargs = self._build_kwargs(use_model, messages,
                                        temperature, top_p, max_tokens, timeout, num_ctx,
                                        top_k, min_p, presence_penalty, repetition_penalty)

            try:
                response = completion(**kwargs)
            except Exception as e:
                # Log the kwargs that caused the failure (omit message content for brevity)
                debug_kwargs = {k: v for k, v in kwargs.items() if k != "messages"}
                msg_summary = [(m.get("role"), len(m.get("content", ""))) for m in kwargs.get("messages", [])]
                logger.warning(
                    "LiteLLM completion failed: %s | model=%s msgs=%s kwargs=%s",
                    e, use_model, msg_summary, debug_kwargs,
                )
                raise
            return self._extract_content(response.choices[0].message, use_model)

        return _retry_with_backoff(
            _do_generate,
            max_retries=self.max_retries,
            operation=f"LLM generation ({use_model})",
        )


# =============================================================================
# EMBEDDING CLIENT
# =============================================================================


class EmbeddingClient:
    """Unified embedding client using LiteLLM.

    Supports multiple providers:
    - Ollama: model="ollama/qwen3-embedding:0.6b", api_base="http://localhost:11434"
    - OpenAI: model="text-embedding-3-small", api_key="sk-..."
    - And many more (see LiteLLM docs)
    """

    def __init__(
        self,
        model: str,
        api_base: str | None = None,
        api_key: str | None = None,
        dimensions: int = 1024,
        max_retries: int = 3,
    ):
        """Initialize embedding client.

        Args:
            model: Model identifier
            api_base: API base URL (required for Ollama)
            api_key: API key (required for cloud providers)
            dimensions: Expected embedding dimensions
            max_retries: Maximum retry attempts for transient failures.
        """
        self.model = model
        self.api_base = api_base
        self.api_key = api_key
        self.dimensions = dimensions
        self.max_retries = max_retries

        # Extract provider from model string
        self.provider = model.split("/")[0] if "/" in model else "openai"

    @classmethod
    def from_model_config(cls, config: ModelConfig) -> EmbeddingClient:
        """Create client from a ModelConfig.

        Preferred constructor — ModelConfig handles provider-specific translation.

        Args:
            config: Model configuration (must have ``dimensions`` set).
        """
        return cls(
            model=config.get_litellm_model(),
            api_base=config.get_api_base(),
            api_key=config.api_key,
            dimensions=config.dimensions or 1024,
        )

    @classmethod
    def from_config(cls, config: EmbeddingConfig) -> EmbeddingClient:
        """Create client from EmbeddingConfig."""
        return cls(
            model=config.model,
            api_base=config.api_base,
            api_key=config.api_key,
            dimensions=config.dimensions,
            max_retries=config.max_retries,
        )

    @classmethod
    def from_ollama(cls, url: str, model: str, dimensions: int = 1024, max_retries: int = 3) -> EmbeddingClient:
        """Create client for Ollama provider.

        Args:
            url: Ollama server URL
            model: Model name without provider prefix
            dimensions: Expected embedding dimensions
            max_retries: Maximum retry attempts for transient failures.

        Returns:
            EmbeddingClient configured for Ollama.
        """
        return cls(
            model=f"ollama/{model}",
            api_base=url,
            dimensions=dimensions,
            max_retries=max_retries,
        )

    def verify_model(self) -> bool:
        """Check if embedding model is available.

        Returns:
            True if model is accessible.
        """
        try:
            self.embed("test", timeout=10)
            return True
        except Exception:
            logger.debug("Embedding health check failed for model %s", self.model, exc_info=True)
            return False

    def embed(self, text: str, timeout: int = 30) -> list[float]:
        """Generate embedding for single text.

        Args:
            text: Text to embed.
            timeout: Request timeout in seconds.

        Returns:
            Embedding vector as list of floats.

        Raises:
            LLMError: If embedding generation fails after retries.
        """
        def _do_embed() -> list[float]:
            kwargs: dict[str, Any] = {
                "model": self.model,
                "input": [text],
                "timeout": timeout,
            }

            if self.api_base:
                kwargs["api_base"] = self.api_base

            # Add api_key if provided, or use dummy key for OpenAI-compatible local servers
            if self.api_key:
                kwargs["api_key"] = self.api_key
            elif self.model.startswith("openai/") and self.api_base:
                # Local OpenAI-compatible servers (llama.cpp) don't need auth
                # but LiteLLM requires an API key for openai/ prefix
                kwargs["api_key"] = "not-needed"

            response = embedding(**kwargs)

            if not response.data:
                raise LLMError("No embedding returned")

            return response.data[0]["embedding"]  # type: ignore[no-any-return]

        return _retry_with_backoff(
            _do_embed,
            max_retries=self.max_retries,
            operation=f"Embedding generation ({self.model})",
        )

    async def embed_async(self, text: str, timeout: int = 30) -> list[float]:
        """Generate embedding for single text (async version).

        Use this method when calling from an async context (e.g., FastAPI routes).

        Args:
            text: Text to embed.
            timeout: Request timeout in seconds.

        Returns:
            Embedding vector as list of floats.

        Raises:
            LLMError: If embedding generation fails after retries.
        """
        # Ensure connection pool is initialized (async-safe)
        await _get_or_create_session_async()

        async def _do_embed() -> list[float]:
            kwargs: dict[str, Any] = {
                "model": self.model,
                "input": [text],
                "timeout": timeout,
            }

            if self.api_base:
                kwargs["api_base"] = self.api_base

            # Add api_key if provided, or use dummy key for OpenAI-compatible local servers
            if self.api_key:
                kwargs["api_key"] = self.api_key
            elif self.model.startswith("openai/") and self.api_base:
                # Local OpenAI-compatible servers (llama.cpp) don't need auth
                # but LiteLLM requires an API key for openai/ prefix
                kwargs["api_key"] = "not-needed"

            response = await aembedding(**kwargs)

            if not response.data:
                raise LLMError("No embedding returned")

            return response.data[0]["embedding"]  # type: ignore[no-any-return]

        return await _retry_with_backoff_async(
            _do_embed,
            max_retries=self.max_retries,
            operation=f"Embedding generation ({self.model})",
        )

    def embed_batch(self, texts: list[str], timeout: int = 60) -> list[list[float]]:
        """Generate embeddings for batch of texts.

        Args:
            texts: List of texts to embed.
            timeout: Request timeout in seconds.

        Returns:
            List of embedding vectors.

        Raises:
            LLMError: If embedding generation fails after retries.
        """
        if not texts:
            return []

        def _do_embed_batch() -> list[list[float]]:
            kwargs: dict[str, Any] = {
                "model": self.model,
                "input": texts,
                "timeout": timeout,
            }

            if self.api_base:
                kwargs["api_base"] = self.api_base

            # Add api_key if provided, or use dummy key for OpenAI-compatible local servers
            if self.api_key:
                kwargs["api_key"] = self.api_key
            elif self.model.startswith("openai/") and self.api_base:
                # Local OpenAI-compatible servers (llama.cpp) don't need auth
                # but LiteLLM requires an API key for openai/ prefix
                kwargs["api_key"] = "not-needed"

            response = embedding(**kwargs)

            return [item["embedding"] for item in response.data]

        return _retry_with_backoff(
            _do_embed_batch,
            max_retries=self.max_retries,
            operation=f"Batch embedding generation ({self.model})",
        )


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================


def create_llm_client_from_config(
    ollama_url: str,
    model: str,
) -> LLMClient:
    """Create LLM client from legacy Ollama config.

    This is a convenience function for migrating existing code.

    Args:
        ollama_url: Ollama server URL
        model: Model name

    Returns:
        LLMClient configured for Ollama.
    """
    return LLMClient.from_ollama(ollama_url, model)


def create_embedding_client_from_config(
    ollama_url: str,
    model: str,
    dimensions: int = 1024,
) -> EmbeddingClient:
    """Create embedding client from legacy Ollama config.

    Args:
        ollama_url: Ollama server URL
        model: Embedding model name
        dimensions: Expected embedding dimensions

    Returns:
        EmbeddingClient configured for Ollama.
    """
    return EmbeddingClient.from_ollama(ollama_url, model, dimensions)
