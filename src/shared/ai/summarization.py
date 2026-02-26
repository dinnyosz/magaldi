"""Phase 5: Summarization - Generate summaries using LLM.

This module handles:
1. LLM client for text generation (via LiteLLM)
2. Hierarchical job processing
3. Summary storage

Prompt templates and building functions are in shared.ai.prompts.

Supports multiple LLM providers through LiteLLM:
- Ollama (local)
- OpenAI
- Anthropic
- And many more
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol

from magaldi_core.code_parser import CodeElement

# Import the new LLM client
from shared.ai.llm_client import LLMClient, LLMError

# Import prompt templates and builders from prompts module
from shared.ai.prompts import (
    LINE_THRESHOLDS,
    PROMPTS,
    SENTENCE_RANGES,
    SYSTEM_PROMPTS,
    USER_PROMPTS,
    build_messages,
    build_prompt,
    clean_summary,
    format_sentence_range,
    get_sentence_range,
    get_size_tier,
    truncate_code,
)

# Re-export for backwards compatibility
__all__ = [
    "LINE_THRESHOLDS",
    "PROMPTS",
    "SENTENCE_RANGES",
    "SYSTEM_PROMPTS",
    "USER_PROMPTS",
    "SummarizationConfig",
    "SummarizationError",
    "SummarizationLLMClient",
    "SummarizationResult",
    "build_messages",
    "build_prompt",
    "clean_summary",
    "format_sentence_range",
    "generate_summary",
    "get_sentence_range",
    "get_size_tier",
    "process_summarization_job",
    "truncate_code",
    "update_dependencies_after_completion",
]


class SummarizationError(Exception):
    """Raised when summarization fails."""

    pass


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class SummarizationConfig:
    """Configuration for summarization."""

    # LLM settings (supports any LiteLLM provider)
    ollama_url: str = "http://localhost:11434"  # For Ollama provider
    model: str = "qwen3:4b-instruct"
    provider: str = "ollama"  # ollama, openai, anthropic, etc.
    api_key: str | None = None  # For cloud providers

    # Generation settings (based on arxiv.org/html/2507.03160v2)
    temperature: float = 0.2
    top_p: float = 0.95
    max_tokens: int = 512
    timeout: int = 180  # 3 minutes to handle queue wait with many workers

    # Retry settings
    max_retries: int = 3
    retry_delay: float = 5.0

    # Code handling
    max_code_tokens: int = 4000


@dataclass
class SummarizationResult:
    """Result of summarization phase."""

    scope: str
    repository: str
    username: str

    # Counts by level
    files_summarized: int = 0
    classes_summarized: int = 0
    functions_summarized: int = 0
    variables_summarized: int = 0

    # Status
    total_jobs: int = 0
    completed_jobs: int = 0
    failed_jobs: int = 0

    # Errors
    errors: list[str] = field(default_factory=list)


# =============================================================================
# LLM CLIENT FOR SUMMARIZATION
# =============================================================================


class SummarizationLLMClient:
    """Client for LLM text generation used in summarization.

    Supports multiple providers through LiteLLM:
    - Ollama (local)
    - llamacpp (llama.cpp server with continuous batching)
    - OpenAI
    - Anthropic
    - And many more
    """

    def __init__(
        self,
        url: str,
        model: str,
        provider: str = "ollama",
        api_key: str | None = None,
    ):
        """Initialize LLM client.

        Args:
            url: API base URL (for Ollama: "http://localhost:11434",
                 for llamacpp: "http://localhost:8080/v1")
            model: Model name (e.g., "qwen2.5-coder:3b", "gpt-4o-mini")
            provider: LLM provider (ollama, llamacpp, openai, anthropic, etc.)
            api_key: API key for cloud providers
        """
        self.url = url.rstrip("/") if url else ""
        self.model = model
        self.provider = provider
        self.api_key = api_key

        # Build full model identifier for LiteLLM
        if provider == "ollama":
            full_model = f"ollama/{model}"
            api_base = url
        elif provider == "llamacpp":
            # llama.cpp server exposes OpenAI-compatible API
            full_model = f"openai/{model}"
            api_base = url  # Should include /v1 suffix
        elif provider == "openai":
            full_model = model
            api_base = None
        else:
            full_model = f"{provider}/{model}"
            api_base = None

        self._client = LLMClient(
            model=full_model,
            api_base=api_base,
            api_key=api_key,
        )

    def verify_model(self) -> bool:
        """Check if model is available."""
        return self._client.verify_model()  # type: ignore[no-any-return]

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
        """Generate completion from LLM.

        Args:
            prompt: The prompt to send to the model.
            temperature: Sampling temperature.
            top_p: Nucleus sampling parameter.
            max_tokens: Maximum tokens to generate.
            timeout: Request timeout in seconds.
            model: Optional model override (uses default if not specified).
            num_ctx: Context window size for Ollama models.

        Returns:
            Generated text.

        Raises:
            ValueError: If response is empty or contains an error.
        """
        # Build model identifier for override if provided
        use_model = None
        if model:
            if self.provider == "ollama":
                use_model = f"ollama/{model}"
            elif self.provider == "llamacpp":
                # llama.cpp uses OpenAI-compatible API
                use_model = f"openai/{model}"
            elif self.provider == "openai":
                use_model = model
            else:
                use_model = f"{self.provider}/{model}"

        try:
            return self._client.generate(  # type: ignore[no-any-return]
                prompt=prompt,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                timeout=timeout,
                model=use_model,
                num_ctx=num_ctx,
            )
        except LLMError as e:
            raise ValueError(str(e)) from e

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
        """Generate completion from messages (optimized for prefix caching).

        This method uses system + user messages to maximize Ollama's KV cache
        reuse. The system message (static instructions) gets cached, while
        the user message contains variable content.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            temperature: Sampling temperature.
            top_p: Nucleus sampling parameter.
            max_tokens: Maximum tokens to generate.
            timeout: Request timeout in seconds.
            model: Optional model override.
            num_ctx: Context window size for Ollama models.

        Returns:
            Generated text.

        Raises:
            ValueError: If response is empty or contains an error.
        """
        use_model = None
        if model:
            if self.provider == "ollama":
                use_model = f"ollama/{model}"
            elif self.provider == "llamacpp":
                use_model = f"openai/{model}"
            elif self.provider == "openai":
                use_model = model
            else:
                use_model = f"{self.provider}/{model}"

        try:
            return self._client.generate_from_messages(  # type: ignore[no-any-return]
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                timeout=timeout,
                model=use_model,
                num_ctx=num_ctx,
            )
        except LLMError as e:
            raise ValueError(str(e)) from e


# =============================================================================
# REPOSITORY PROTOCOLS
# =============================================================================


class JobRepository(Protocol):
    """Interface for summarization job storage.

    All methods require scope, repository, and username for isolated queues.
    """

    def add_job(
        self,
        element_id: str,
        scope: str,
        repository: str,
        username: str,
        level: int,
        parent_id: str | None,
        dependencies_met: bool,
        priority: int = 0,
    ) -> None:
        """Add a summarization job to user's queue."""
        ...

    def get_job(
        self, element_id: str, scope: str, repository: str, username: str
    ) -> dict[str, Any] | None:
        """Get job by element ID from user's queue."""
        ...

    def claim_pending_jobs(
        self, worker_id: str, scope: str, repository: str, username: str, batch_size: int
    ) -> list[dict[str, Any]]:
        """Claim pending jobs from user's queue."""
        ...

    def mark_completed(
        self, element_id: str, scope: str, repository: str, username: str
    ) -> None:
        """Mark job as completed in user's queue."""
        ...

    def mark_failed(
        self, element_id: str, scope: str, repository: str, username: str, error_message: str
    ) -> None:
        """Mark job as failed in user's queue."""
        ...

    def unlock_dependencies(
        self, parent_element_id: str, scope: str, repository: str, username: str
    ) -> int:
        """Unlock jobs that depend on this element. Returns count unlocked."""
        ...


class SummaryStore(Protocol):
    """Interface for element and summary storage."""

    def store_element(self, element: CodeElement) -> None:
        """Store a code element."""
        ...

    def get_element(self, element_id: str) -> CodeElement | None:
        """Get element by ID."""
        ...

    def store_summary(self, element_id: str, summary: str) -> None:
        """Store summary for an element."""
        ...

    def get_summary(self, element_id: str) -> str | None:
        """Get summary for an element."""
        ...

    def get_parent_summaries(self, element: CodeElement) -> dict[str, str]:
        """Get summaries from parent elements."""
        ...


# =============================================================================
# IN-MEMORY IMPLEMENTATIONS
# =============================================================================


class InMemoryJobRepository:
    """In-memory implementation of job repository for testing.

    Note: This simple implementation stores all jobs globally.
    The scope/repository/username parameters are accepted for interface compatibility.
    """

    def __init__(self) -> None:
        self._jobs: dict[str, dict[str, Any]] = {}

    def add_job(
        self,
        element_id: str,
        scope: str,
        repository: str,
        username: str,
        level: int,
        parent_id: str | None,
        dependencies_met: bool,
        priority: int = 0,
    ) -> None:
        self._jobs[element_id] = {
            "element_id": element_id,
            "scope": scope,
            "repository": repository,
            "username": username,
            "level": level,
            "parent_id": parent_id,
            "dependencies_met": dependencies_met,
            "status": "pending",
            "worker_id": None,
            "claimed_at": None,
            "completed_at": None,
            "error_message": None,
            "priority": priority if priority else (100 - level),
        }

    def get_job(
        self, element_id: str, _scope: str, _repository: str, _username: str
    ) -> dict[str, Any] | None:
        return self._jobs.get(element_id)

    def claim_pending_jobs(
        self, worker_id: str, _scope: str, _repository: str, _username: str, batch_size: int
    ) -> list[dict[str, Any]]:
        # Find pending jobs with dependencies met, sorted by level then priority
        available = [
            job
            for job in self._jobs.values()
            if job["status"] == "pending" and job["dependencies_met"]
        ]
        available.sort(key=lambda j: (j["level"], -j["priority"]))

        claimed = available[:batch_size]
        for job in claimed:
            job["status"] = "running"
            job["worker_id"] = worker_id
            job["claimed_at"] = datetime.now()

        return claimed

    def mark_completed(
        self, element_id: str, _scope: str, _repository: str, _username: str
    ) -> None:
        if element_id in self._jobs:
            self._jobs[element_id]["status"] = "completed"
            self._jobs[element_id]["completed_at"] = datetime.now()

    def mark_failed(
        self, element_id: str, _scope: str, _repository: str, _username: str, error_message: str
    ) -> None:
        if element_id in self._jobs:
            self._jobs[element_id]["status"] = "failed"
            self._jobs[element_id]["error_message"] = error_message
            self._jobs[element_id]["completed_at"] = datetime.now()

    def unlock_dependencies(
        self, parent_element_id: str, _scope: str, _repository: str, _username: str
    ) -> int:
        count = 0
        for job in self._jobs.values():
            if (
                job["parent_id"] == parent_element_id
                and job["status"] == "pending"
                and not job["dependencies_met"]
            ):
                job["dependencies_met"] = True
                count += 1
        return count


class InMemorySummaryStore:
    """In-memory implementation of summary store for testing."""

    def __init__(self) -> None:
        self._elements: dict[str, CodeElement] = {}
        self._summaries: dict[str, str] = {}

    def store_element(self, element: CodeElement) -> None:
        self._elements[element.element_id] = element

    def get_element(self, element_id: str) -> CodeElement | None:
        return self._elements.get(element_id)

    def store_summary(self, element_id: str, summary: str) -> None:
        self._summaries[element_id] = summary

    def get_summary(self, element_id: str) -> str | None:
        return self._summaries.get(element_id)

    def get_parent_summaries(self, element: CodeElement) -> dict[str, str]:
        summaries: dict[str, str] = {}

        # For non-file elements, find file summary
        if element.level > 0:
            # Find file element for this path
            for eid, elem in self._elements.items():
                if (
                    elem.scope == element.scope
                    and elem.repository == element.repository
                    and elem.username == element.username
                    and elem.relative_path == element.relative_path
                    and elem.element_type == "file"
                ):
                    file_summary = self.get_summary(eid)
                    if file_summary:
                        summaries["file"] = file_summary
                    break

        # For methods, find class summary
        if element.parent_id and element.element_type in ("method", "function"):
            parent = self.get_element(element.parent_id)
            if parent and parent.element_type == "class":
                class_summary = self.get_summary(element.parent_id)
                if class_summary:
                    summaries["class"] = class_summary

        return summaries


# =============================================================================
# DEPENDENCY RESOLUTION
# =============================================================================


def update_dependencies_after_completion(
    element_id: str,
    scope: str,
    repository: str,
    username: str,
    job_repo: JobRepository,
) -> int:
    """Mark dependent jobs as ready when parent completes.

    Args:
        element_id: Completed element ID.
        scope: Repository scope.
        repository: Repository name.
        username: User who owns the jobs.
        job_repo: Job repository.

    Returns:
        Count of jobs unlocked.
    """
    return job_repo.unlock_dependencies(element_id, scope, repository, username)


# =============================================================================
# SUMMARY GENERATION
# =============================================================================


def generate_summary(
    element: CodeElement,
    summary_store: SummaryStore,
    llm_client: SummarizationLLMClient,
    config: SummarizationConfig,
) -> str:
    """Generate summary for a single element.

    Uses message-based format (system + user) optimized for Ollama's KV cache
    prefix caching. The system message contains static instructions that get
    cached, while the user message has variable content with shared context
    (file_summary, class_summary) at the top for maximum cache reuse.

    Args:
        element: Code element to summarize.
        summary_store: Summary store for parent context.
        llm_client: LLM client for text generation.
        config: Summarization config.

    Returns:
        Generated summary.
    """
    # Get parent summaries for context
    parent_summaries = summary_store.get_parent_summaries(element)

    # Build messages optimized for prefix caching:
    # - System message: static instructions (cached per element type)
    # - User message: shared context first (file/class), then element-specific
    messages = build_messages(element, parent_summaries, config.max_code_tokens)

    # Generate summary using message-based API
    raw_summary = llm_client.generate_from_messages(
        messages=messages,
        temperature=config.temperature,
        top_p=config.top_p,
        max_tokens=config.max_tokens,
        timeout=config.timeout,
    )

    # Clean and return
    return clean_summary(raw_summary)  # type: ignore[no-any-return]


def process_summarization_job(
    element_id: str,
    scope: str,
    repository: str,
    username: str,
    job_repo: JobRepository,
    summary_store: SummaryStore,
    llm_client: SummarizationLLMClient,
    config: SummarizationConfig,
) -> bool:
    """Process a single summarization job.

    Args:
        element_id: Element ID to summarize.
        scope: Repository scope.
        repository: Repository name.
        username: User who owns this job.
        job_repo: Job repository.
        summary_store: Summary store.
        llm_client: LLM client for text generation.
        config: Summarization config.

    Returns:
        True if successful, False otherwise.
    """
    try:
        # Get element
        element = summary_store.get_element(element_id)
        if element is None:
            job_repo.mark_failed(
                element_id, scope, repository, username,
                f"Element not found: {element_id}"
            )
            return False

        # Generate summary
        summary = generate_summary(element, summary_store, llm_client, config)

        # Store summary
        summary_store.store_summary(element_id, summary)

        # Mark job completed
        job_repo.mark_completed(element_id, scope, repository, username)

        # Unlock dependent jobs
        update_dependencies_after_completion(
            element_id, scope, repository, username, job_repo
        )

        return True

    except Exception as e:
        job_repo.mark_failed(element_id, scope, repository, username, str(e))
        return False
