"""Unified LLM client using LiteLLM for provider abstraction.

This module provides a unified interface for text generation and embeddings
that works with multiple LLM providers (Ollama, OpenAI, Anthropic, etc.).

Usage:
    from shared.ai.llm_client import LLMClient, EmbeddingClient

    # Text generation
    llm = LLMClient(model="ollama/qwen2.5-coder:3b")
    response = llm.generate("Explain this code...")

    # Embeddings
    embed = EmbeddingClient(model="ollama/snowflake-arctic-embed2")
    vector = embed.embed("Some text to embed")

    # Cleanup on shutdown (optional but recommended)
    import atexit
    atexit.register(cleanup_llm_sessions)
"""

from __future__ import annotations

import asyncio
import atexit
import os
import random
import time
import warnings
from dataclasses import dataclass
from typing import Any, Callable, TypeVar

import aiohttp
import httpx
import litellm
from litellm import aembedding, completion, embedding
from litellm.llms.custom_httpx.aiohttp_handler import BaseLLMAIOHTTPHandler

# Workaround for LiteLLM bug #11657: Ollama embeddings leak aiohttp sessions
# causing "Too many open files" errors. Disabling aiohttp transport forces
# LiteLLM to use httpx which has proper connection pooling.
# See: https://github.com/BerriAI/litellm/issues/11657
os.environ.setdefault("DISABLE_AIOHTTP_TRANSPORT", "True")

T = TypeVar("T")

# Disable LiteLLM telemetry and debug info
litellm.telemetry = False
litellm.suppress_debug_info = True

# Suppress Pydantic 2.12+ serialization warnings from LiteLLM
# See: https://github.com/BerriAI/litellm/issues/11759
warnings.filterwarnings("ignore", message="Pydantic serializer warnings")

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
    timeout=httpx.Timeout(300.0, connect=30.0),
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
                total=300,  # Total request timeout
                connect=30,  # Connection timeout
            )
            _aiohttp_session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
            )
            # Configure LiteLLM to use our managed session for async calls
            # This may fail on newer LiteLLM versions with changed API
            try:
                litellm.base_llm_aiohttp_handler = BaseLLMAIOHTTPHandler(
                    client_session=_aiohttp_session
                )
            except TypeError:
                # Newer LiteLLM version with different API - use default handling
                pass

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
            pass  # Best effort cleanup

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
                pass  # Best effort cleanup
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
    fn: Callable[[], T],
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
    #   - "ollama/qwen3:4b-instruct"
    #   - "gpt-4o-mini"
    #   - "claude-3-haiku-20240307"
    model: str = "ollama/qwen3:4b-instruct"

    # API configuration
    api_base: str | None = None  # For Ollama: "http://localhost:11434"
    api_key: str | None = None  # For cloud providers

    # Generation settings (based on arxiv.org/html/2507.03160v2)
    temperature: float = 0.2
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
    #   - "ollama/snowflake-arctic-embed2"
    #   - "text-embedding-3-small"
    model: str = "ollama/snowflake-arctic-embed2"

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
    ):
        """Initialize LLM client.

        Args:
            model: Model identifier (e.g., "ollama/qwen2.5-coder:3b", "gpt-4o-mini")
            api_base: API base URL (required for Ollama, optional for cloud)
            api_key: API key (required for cloud providers)
            max_retries: Maximum retry attempts for transient failures.
        """
        self.model = model
        self.api_base = api_base
        self.api_key = api_key
        self.max_retries = max_retries

        # Extract provider from model string
        self.provider = model.split("/")[0] if "/" in model else "openai"

    @classmethod
    def from_config(cls, config: LLMConfig) -> LLMClient:
        """Create client from config."""
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
            return False

    # Models that use thinking/reasoning tags by default
    THINKING_MODELS = ("qwen3", "deepseek-r1", "deepseek-coder-v2", "nemotron")

    def generate(
        self,
        prompt: str,
        temperature: float = 0.2,
        top_p: float = 0.95,
        max_tokens: int = 512,
        timeout: int = 60,
        model: str | None = None,
        num_ctx: int | None = None,
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
                     Used to optimize KV cache sizing for better performance.

        Returns:
            Generated text.

        Raises:
            LLMError: If generation fails after retries.
        """
        use_model = model or self.model

        # Check if this is a thinking model that needs think=false
        model_name = use_model.split("/")[-1] if "/" in use_model else use_model
        is_thinking_model = any(model_name.startswith(tm) for tm in self.THINKING_MODELS)

        def _do_generate() -> str:
            # Build kwargs for litellm
            kwargs: dict[str, Any] = {
                "model": use_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
                "timeout": timeout,
            }

            # Add api_base for custom endpoints (Ollama, llama.cpp, etc.)
            if self.api_base:
                kwargs["api_base"] = self.api_base

            # Add api_key if provided, or use dummy key for OpenAI-compatible local servers
            if self.api_key:
                kwargs["api_key"] = self.api_key
            elif use_model.startswith("openai/") and self.api_base:
                # Local OpenAI-compatible servers (llama.cpp) don't need auth
                # but LiteLLM requires an API key for openai/ prefix
                kwargs["api_key"] = "not-needed"

            # Disable thinking mode for models that support it
            # LiteLLM added think parameter support in PR #15465 (Sept 2025)
            if is_thinking_model and use_model.startswith("ollama/"):
                kwargs["think"] = False

            # Add num_ctx for local providers (KV cache optimization)
            if num_ctx:
                if use_model.startswith("ollama/"):
                    kwargs["num_ctx"] = num_ctx
                elif use_model.startswith("openai/") and self.api_base:
                    # OpenAI-compatible local servers (llama.cpp, LM Studio, LocalAI)
                    kwargs["extra_body"] = kwargs.get("extra_body", {})
                    kwargs["extra_body"]["n_ctx"] = num_ctx

            response = completion(**kwargs)

            # Extract text from response
            content = response.choices[0].message.content
            if content is None:
                raise LLMError(f"Empty response from model '{use_model}'")

            return content.strip()

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
                     Used to optimize KV cache sizing for better performance.

        Returns:
            Generated text.

        Raises:
            LLMError: If generation fails after retries.
        """
        use_model = model or self.model

        # Check if this is a thinking model that needs think=false
        model_name = use_model.split("/")[-1] if "/" in use_model else use_model
        is_thinking_model = any(model_name.startswith(tm) for tm in self.THINKING_MODELS)

        def _do_generate() -> str:
            kwargs: dict[str, Any] = {
                "model": use_model,
                "messages": messages,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
                "timeout": timeout,
            }

            if self.api_base:
                kwargs["api_base"] = self.api_base

            if self.api_key:
                kwargs["api_key"] = self.api_key
            elif use_model.startswith("openai/") and self.api_base:
                kwargs["api_key"] = "not-needed"

            if is_thinking_model and use_model.startswith("ollama/"):
                kwargs["think"] = False

            # Add num_ctx for local providers (KV cache optimization)
            if num_ctx:
                if use_model.startswith("ollama/"):
                    kwargs["num_ctx"] = num_ctx
                elif use_model.startswith("openai/") and self.api_base:
                    # OpenAI-compatible local servers (llama.cpp, LM Studio, LocalAI)
                    kwargs["extra_body"] = kwargs.get("extra_body", {})
                    kwargs["extra_body"]["n_ctx"] = num_ctx

            response = completion(**kwargs)

            content = response.choices[0].message.content
            if content is None:
                raise LLMError(f"Empty response from model '{use_model}'")

            return content.strip()

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
    - Ollama: model="ollama/snowflake-arctic-embed2", api_base="http://localhost:11434"
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
    def from_config(cls, config: EmbeddingConfig) -> EmbeddingClient:
        """Create client from config."""
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

            return response.data[0]["embedding"]

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

            return response.data[0]["embedding"]

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
