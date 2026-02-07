"""Phase 6: Embedding - Generate vector embeddings using LLM providers.

This module handles:
1. LLM embedding client (via LiteLLM)
2. Context building with hierarchical enrichment
3. Vector generation and validation
4. Embedding storage

Supports multiple embedding providers through LiteLLM:
- Ollama (local)
- OpenAI
- And many more
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol

from magaldi_core.code_parser import CodeElement

# Import the new embedding client
from shared.ai.llm_client import EmbeddingClient, LLMError


class EmbeddingError(Exception):
    """Raised when embedding generation fails."""

    pass


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation."""

    # LLM settings (supports any LiteLLM provider)
    ollama_url: str = "http://localhost:11434"  # For Ollama provider
    model: str = "qwen3-embedding:0.6b"
    provider: str = "ollama"  # ollama, openai, etc.
    api_key: str | None = None  # For cloud providers

    # Vector settings
    dimensions: int = 1024
    max_context: int = 32768

    # Worker settings
    batch_size: int = 20
    es_batch_size: int = 100

    # Retry settings
    max_retries: int = 3
    retry_delay: float = 2.0
    timeout: int = 30


@dataclass
class EmbeddingResult:
    """Result of embedding phase."""

    scope: str
    repository: str
    username: str

    # Counts
    total_elements: int = 0
    embedded_elements: int = 0
    failed_elements: int = 0

    # Errors
    errors: list[str] = field(default_factory=list)


# =============================================================================
# EMBEDDING CLIENT
# =============================================================================


class CodeEmbeddingClient:
    """Client for LLM embedding generation.

    Supports multiple providers through LiteLLM:
    - Ollama (local)
    - llamacpp (llama.cpp server with --embedding flag)
    - OpenAI
    - And many more
    """

    def __init__(
        self,
        url: str,
        model: str,
        provider: str = "ollama",
        api_key: str | None = None,
        dimensions: int = 1024,
    ):
        """Initialize embedding client.

        Args:
            url: API base URL (for Ollama: "http://localhost:11434",
                 for llamacpp: "http://localhost:8080/v1")
            model: Model name (e.g., "qwen3-embedding:0.6b", "text-embedding-3-small")
            provider: Embedding provider (ollama, llamacpp, openai, etc.)
            api_key: API key for cloud providers
            dimensions: Expected embedding dimensions
        """
        self.url = url.rstrip("/") if url else ""
        self.model = model
        self.provider = provider
        self.api_key = api_key
        self.dimensions = dimensions

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

        self._client = EmbeddingClient(
            model=full_model,
            api_base=api_base,
            api_key=api_key,
            dimensions=dimensions,
        )

    def verify_model(self) -> bool:
        """Check if embedding model is available."""
        return self._client.verify_model()

    def embed_single(self, text: str, timeout: int = 30) -> list[float]:
        """Generate embedding for single text.

        Args:
            text: Text to embed.
            timeout: Request timeout in seconds.

        Returns:
            Embedding vector as list of floats.

        Raises:
            EmbeddingError: If embedding generation fails.
        """
        try:
            return self._client.embed(text, timeout=timeout)
        except LLMError as e:
            raise EmbeddingError(str(e)) from e

    async def embed_single_async(self, text: str, timeout: int = 30) -> list[float]:
        """Generate embedding for single text (async version).

        Use this method when calling from an async context (e.g., FastAPI routes).

        Args:
            text: Text to embed.
            timeout: Request timeout in seconds.

        Returns:
            Embedding vector as list of floats.

        Raises:
            EmbeddingError: If embedding generation fails.
        """
        try:
            return await self._client.embed_async(text, timeout=timeout)
        except LLMError as e:
            raise EmbeddingError(str(e)) from e

    def embed_batch(self, texts: list[str], timeout: int = 60) -> list[list[float]]:
        """Generate embeddings for batch of texts.

        Args:
            texts: List of texts to embed.
            timeout: Request timeout in seconds.

        Returns:
            List of embedding vectors.

        Raises:
            EmbeddingError: If embedding generation fails.
        """
        try:
            return self._client.embed_batch(texts, timeout=timeout)
        except LLMError as e:
            raise EmbeddingError(str(e)) from e


# =============================================================================
# REPOSITORY PROTOCOLS
# =============================================================================


class EmbeddingJobRepository(Protocol):
    """Interface for embedding job storage.

    All methods require scope, repository, and username for isolated queues.
    """

    def add_job(
        self, element_id: str, scope: str, repository: str, username: str
    ) -> None:
        """Add an embedding job to user's queue."""
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


class EmbeddingStore(Protocol):
    """Interface for element and embedding storage."""

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

    def store_embedding(self, element_id: str, embedding: list[float]) -> None:
        """Store embedding vector for an element."""
        ...

    def get_embedding(self, element_id: str) -> list[float] | None:
        """Get embedding vector for an element."""
        ...

    def get_file_summary(self, element: CodeElement) -> str | None:
        """Get file summary for context enrichment."""
        ...

    def get_class_summary(self, element: CodeElement) -> str | None:
        """Get class summary for context enrichment."""
        ...


# =============================================================================
# IN-MEMORY IMPLEMENTATIONS
# =============================================================================


class InMemoryEmbeddingJobRepository:
    """In-memory implementation of job repository for testing.

    Note: This simple implementation stores all jobs globally.
    The scope/repository/username parameters are accepted for interface compatibility.
    """

    def __init__(self) -> None:
        self._jobs: dict[str, dict[str, Any]] = {}

    def add_job(
        self, element_id: str, scope: str, repository: str, username: str
    ) -> None:
        self._jobs[element_id] = {
            "element_id": element_id,
            "scope": scope,
            "repository": repository,
            "username": username,
            "status": "pending",
            "worker_id": None,
            "claimed_at": None,
            "completed_at": None,
            "error_message": None,
        }

    def get_job(
        self, element_id: str, scope: str, repository: str, username: str
    ) -> dict[str, Any] | None:
        return self._jobs.get(element_id)

    def claim_pending_jobs(
        self, worker_id: str, scope: str, repository: str, username: str, batch_size: int
    ) -> list[dict[str, Any]]:
        available = [
            job for job in self._jobs.values() if job["status"] == "pending"
        ]

        claimed = available[:batch_size]
        for job in claimed:
            job["status"] = "running"
            job["worker_id"] = worker_id
            job["claimed_at"] = datetime.now()

        return claimed

    def mark_completed(
        self, element_id: str, scope: str, repository: str, username: str
    ) -> None:
        if element_id in self._jobs:
            self._jobs[element_id]["status"] = "completed"
            self._jobs[element_id]["completed_at"] = datetime.now()

    def mark_failed(
        self, element_id: str, scope: str, repository: str, username: str, error_message: str
    ) -> None:
        if element_id in self._jobs:
            self._jobs[element_id]["status"] = "failed"
            self._jobs[element_id]["error_message"] = error_message
            self._jobs[element_id]["completed_at"] = datetime.now()


class InMemoryEmbeddingStore:
    """In-memory implementation of embedding store for testing."""

    def __init__(self) -> None:
        self._elements: dict[str, CodeElement] = {}
        self._summaries: dict[str, str] = {}
        self._embeddings: dict[str, list[float]] = {}

    def store_element(self, element: CodeElement) -> None:
        self._elements[element.element_id] = element

    def get_element(self, element_id: str) -> CodeElement | None:
        return self._elements.get(element_id)

    def store_summary(self, element_id: str, summary: str) -> None:
        self._summaries[element_id] = summary

    def get_summary(self, element_id: str) -> str | None:
        return self._summaries.get(element_id)

    def store_embedding(self, element_id: str, embedding: list[float]) -> None:
        self._embeddings[element_id] = embedding

    def get_embedding(self, element_id: str) -> list[float] | None:
        return self._embeddings.get(element_id)

    def get_file_summary(self, element: CodeElement) -> str | None:
        # Find file element for this path
        for eid, elem in self._elements.items():
            if (
                elem.scope == element.scope
                and elem.repository == element.repository
                and elem.username == element.username
                and elem.relative_path == element.relative_path
                and elem.element_type == "file"
            ):
                return self.get_summary(eid)
        return None

    def get_class_summary(self, element: CodeElement) -> str | None:
        if not element.parent_id:
            return None

        parent = self.get_element(element.parent_id)
        if parent and parent.element_type == "class":
            return self.get_summary(element.parent_id)
        return None


# =============================================================================
# TOKEN ESTIMATION
# =============================================================================


CODE_LANGUAGES = {"python", "javascript", "typescript", "tsx", "php", "rust"}


def estimate_tokens(text: str, chars_per_token: int = 2) -> int:
    """Estimate token count (rough approximation).

    Args:
        text: Input text.
        chars_per_token: Average characters per token. Code languages average ~3,
            markdown/yaml/text with T5/sentencepiece tokenizers average ~2.

    Returns:
        Estimated token count.
    """
    if not text:
        return 0
    return len(text) // chars_per_token


def chars_per_token_for_language(language: str | None) -> int:
    """Return the appropriate chars-per-token ratio for a language.

    Code languages (Python, JS, etc.) average ~3 chars/token.
    Non-code content (markdown, yaml, text) averages ~2 chars/token.
    """
    if language and language in CODE_LANGUAGES:
        return 3
    return 2


def validate_context_length(
    text: str, max_tokens: int = 8000, chars_per_token: int = 2
) -> str:
    """Ensure text fits within model context.

    Args:
        text: Input text.
        max_tokens: Maximum allowed tokens.
        chars_per_token: Average characters per token for estimation.

    Returns:
        Original or truncated text.
    """
    estimated = estimate_tokens(text, chars_per_token)

    if estimated <= max_tokens:
        return text

    # Truncate line by line
    lines = text.split("\n")
    truncated = []
    token_count = 0

    for line in lines:
        line_tokens = estimate_tokens(line, chars_per_token)
        if token_count + line_tokens <= max_tokens * 0.9:
            truncated.append(line)
            token_count += line_tokens
        else:
            truncated.append("... (context truncated)")
            break

    return "\n".join(truncated)


# =============================================================================
# VECTOR OPERATIONS
# =============================================================================


def normalize_vector(vector: list[float]) -> list[float]:
    """Normalize vector to unit length for cosine similarity.

    Args:
        vector: Input vector.

    Returns:
        Normalized vector.
    """
    magnitude = math.sqrt(sum(x * x for x in vector))

    if magnitude == 0:
        return vector

    return [x / magnitude for x in vector]


def validate_vector(vector: list[float], expected_dims: int) -> bool:
    """Validate vector dimensions and values.

    Args:
        vector: Vector to validate.
        expected_dims: Expected dimensions.

    Returns:
        True if valid.
    """
    if len(vector) != expected_dims:
        return False

    for v in vector:
        if math.isnan(v) or math.isinf(v):
            return False

    return True


# =============================================================================
# EMBEDDING TEXT BUILDING
# =============================================================================


def _format_calls_text(calls: list[Any]) -> str | None:
    """Format calls into a deduplicated comma-separated string.

    Args:
        calls: List of Call objects with .name and .receiver attributes.

    Returns:
        Formatted string like "Flask, app.run, setup" or None if no calls.
    """
    seen: set[str] = set()
    call_names: list[str] = []
    for call in calls:
        key = f"{call.receiver}.{call.name}" if call.receiver else call.name
        if key not in seen:
            seen.add(key)
            call_names.append(key)
    return ", ".join(call_names) if call_names else None


def build_summary_embedding_text(
    element: CodeElement,
    embedding_store: EmbeddingStore,
    max_tokens: int = 8000,
) -> str:
    """Build text for summary embedding (metadata + summary + context).

    This builds enriched text with hierarchical context from parent summaries,
    optimized for semantic search based on what the code does.

    Args:
        element: Code element to embed.
        embedding_store: Store for getting summaries.
        max_tokens: Maximum context tokens.

    Returns:
        Formatted embedding text.
    """
    parts: list[str] = []

    # Always include file path
    parts.append(f"File: {element.relative_path}")

    if element.element_type == "file":
        parts.append(f"Language: {element.language}")
        summary = embedding_store.get_summary(element.element_id)
        parts.append(f"Summary: {summary or 'No summary available'}")

        # Top-level calls (e.g., app = Flask(__name__), setup(), register_routes())
        if element.calls:
            calls_text = _format_calls_text(element.calls)
            if calls_text:
                parts.append(f"Calls: {calls_text}")

    elif element.element_type == "class":
        # Include file context
        file_summary = embedding_store.get_file_summary(element)
        if file_summary:
            parts.append(f"File context: {file_summary}")

        parts.append(f"Class: {element.name}")

        summary = embedding_store.get_summary(element.element_id)
        parts.append(f"Summary: {summary or 'No summary available'}")

        if element.docstring:
            parts.append(f"Docstring: {element.docstring[:500]}")

    elif element.element_type in ("function", "method"):
        # Include file context
        file_summary = embedding_store.get_file_summary(element)
        if file_summary:
            parts.append(f"File context: {file_summary}")

        # Include class context for methods
        class_summary = embedding_store.get_class_summary(element)
        if class_summary:
            parts.append(f"Class context: {class_summary}")

        parts.append(f"Function: {element.name}")

        summary = embedding_store.get_summary(element.element_id)
        parts.append(f"Summary: {summary or 'No summary available'}")

        if element.signature:
            parts.append(f"Signature: {element.signature}")

        if element.docstring:
            docstring = element.docstring[:500]
            if len(element.docstring) > 500:
                docstring += "..."
            parts.append(f"Docstring: {docstring}")

        # Outbound call names improve caller→callee embedding similarity
        # by +7.3% MRR (validated via passport embedding benchmark)
        if element.calls:
            calls_text = _format_calls_text(element.calls)
            if calls_text:
                parts.append(f"Calls: {calls_text}")

    elif element.element_type in ("variable", "constant"):
        # Variables and constants get hierarchical context
        file_summary = embedding_store.get_file_summary(element)
        if file_summary:
            parts.append(f"File context: {file_summary}")

        class_summary = embedding_store.get_class_summary(element)
        if class_summary:
            parts.append(f"Class context: {class_summary}")

        parts.append(f"Name: {element.name}")
        parts.append(f"Type: {element.element_type}")

        summary = embedding_store.get_summary(element.element_id)
        parts.append(f"Summary: {summary or 'No summary available'}")

    else:
        # Default for other types
        parts.append(f"Name: {element.name}")
        summary = embedding_store.get_summary(element.element_id)
        if summary:
            parts.append(f"Summary: {summary}")

    text = "\n".join(parts)
    cpt = chars_per_token_for_language(element.language)
    return validate_context_length(text, max_tokens, chars_per_token=cpt)


# Backwards compatibility alias
build_embedding_text = build_summary_embedding_text


def build_code_embedding_text(
    element: CodeElement,
    max_tokens: int = 8000,
) -> str:
    """Build text for code embedding (raw code with minimal context).

    This builds text focused on the actual code implementation,
    optimized for finding similar code patterns and implementations.

    Args:
        element: Code element to embed.
        max_tokens: Maximum context tokens.

    Returns:
        Formatted text for code embedding.
    """
    parts: list[str] = []

    # Add minimal context (file path, element type)
    parts.append(f"# {element.element_type}: {element.name}")
    parts.append(f"# File: {element.relative_path}")

    # Add signature if available
    if element.signature:
        parts.append(f"# Signature: {element.signature}")

    # Add the raw code
    if element.raw_code:
        parts.append(element.raw_code)

    text = "\n".join(parts)
    cpt = chars_per_token_for_language(element.language)
    return validate_context_length(text, max_tokens, chars_per_token=cpt)


# =============================================================================
# EMBEDDING GENERATION
# =============================================================================


def process_embedding_job(
    element_id: str,
    scope: str,
    repository: str,
    username: str,
    job_repo: EmbeddingJobRepository,
    embedding_store: EmbeddingStore,
    embed_client: CodeEmbeddingClient,
    config: EmbeddingConfig,
) -> bool:
    """Process a single embedding job.

    Args:
        element_id: Element ID to embed.
        scope: Repository scope.
        repository: Repository name.
        username: User who owns this job.
        job_repo: Job repository.
        embedding_store: Embedding store.
        embed_client: Embedding client.
        config: Embedding config.

    Returns:
        True if successful, False otherwise.
    """
    try:
        # Get element
        element = embedding_store.get_element(element_id)
        if element is None:
            job_repo.mark_failed(
                element_id, scope, repository, username,
                f"Element not found: {element_id}"
            )
            return False

        # Build embedding text with context
        text = build_embedding_text(element, embedding_store, config.max_context)

        # Generate embedding
        embedding = embed_client.embed_single(text, timeout=config.timeout)

        # Validate dimensions
        if not validate_vector(embedding, config.dimensions):
            job_repo.mark_failed(
                element_id, scope, repository, username,
                f"Invalid embedding: expected {config.dimensions} dims, "
                f"got {len(embedding)}",
            )
            return False

        # Normalize for cosine similarity
        embedding = normalize_vector(embedding)

        # Store embedding
        embedding_store.store_embedding(element_id, embedding)

        # Mark job completed
        job_repo.mark_completed(element_id, scope, repository, username)

        return True

    except Exception as e:
        job_repo.mark_failed(element_id, scope, repository, username, str(e))
        return False
