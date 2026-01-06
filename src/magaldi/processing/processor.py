"""Unified element processor - atomic summarize -> embed -> index flow.

Processes elements level-by-level:
- Level 0: Files
- Level 1: Classes
- Level 2: Functions/Methods
- Level 3: Variables

Only indexes to ES after full processing, ensuring ES presence = completion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

from magaldi.db.elasticsearch import ElasticsearchRepository
from magaldi.embedding.embedding import (
    EmbeddingConfig,
    OllamaEmbedClient,
    build_embedding_text,
    normalize_vector,
    validate_vector,
)
from magaldi.parser.code_parser import CodeElement, ParsedFile
from magaldi.summarization.summarization import (
    OllamaClient,
    SummarizationConfig,
    build_prompt,
    clean_summary,
)


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class ProcessingConfig:
    """Configuration for unified processing."""

    summarize_model: str = "qwen2.5-coder:7b"
    embed_model: str = "snowflake-arctic-embed2"
    ollama_url: str = "http://localhost:11434"
    skip_ai: bool = False

    # Summarization settings
    summarize_temperature: float = 0.3
    summarize_max_tokens: int = 256
    summarize_timeout: int = 60
    max_code_tokens: int = 4000

    # Embedding settings
    embed_dimensions: int = 1024
    embed_max_context: int = 8192
    embed_timeout: int = 30


@dataclass
class ProcessingResult:
    """Result of unified processing."""

    scope: str
    repository: str
    username: str

    # Counts
    elements_processed: int = 0
    elements_skipped: int = 0  # Already in ES
    elements_failed: int = 0

    # By type
    summarized: int = 0
    embedded: int = 0
    indexed: int = 0

    # Errors
    errors: list[str] = field(default_factory=list)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def should_embed(element: CodeElement) -> bool:
    """Determine if element should be embedded.

    Args:
        element: Code element to check.

    Returns:
        True if element should be embedded.
    """
    # Files, classes, functions, and methods always get embedded
    if element.element_type in ("file", "class", "function", "method"):
        return True

    # Variables only if UPPER_CASE constants or have docstrings
    if element.element_type == "variable":
        if element.name.isupper():
            return True
        if element.docstring:
            return True

    return False


# =============================================================================
# INTERNAL STORE ADAPTER
# =============================================================================


class _SummaryCache:
    """In-memory cache that acts as EmbeddingStore for build_embedding_text.

    This adapter allows us to use build_embedding_text without requiring
    elements to be stored in ES first.
    """

    def __init__(self) -> None:
        self._elements: dict[str, CodeElement] = {}
        self._summaries: dict[str, str] = {}

    def add_element(self, element: CodeElement) -> None:
        """Add element to cache."""
        self._elements[element.element_id] = element

    def add_summary(self, element_id: str, summary: str) -> None:
        """Add summary to cache."""
        self._summaries[element_id] = summary

    def get_element(self, element_id: str) -> CodeElement | None:
        """Get element from cache."""
        return self._elements.get(element_id)

    def get_summary(self, element_id: str) -> str | None:
        """Get summary from cache."""
        return self._summaries.get(element_id)

    def get_file_summary(self, element: CodeElement) -> str | None:
        """Get file summary for an element."""
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
        """Get class summary for an element (via parent_id)."""
        if element.parent_id:
            parent = self.get_element(element.parent_id)
            if parent and parent.element_type == "class":
                return self.get_summary(element.parent_id)
        return None

    def get_parent_summaries(self, element: CodeElement) -> dict[str, str]:
        """Get parent summaries for context."""
        summaries: dict[str, str] = {}

        # Get file summary
        file_summary = self.get_file_summary(element)
        if file_summary:
            summaries["file"] = file_summary

        # Get class summary if method
        if element.element_type == "method":
            class_summary = self.get_class_summary(element)
            if class_summary:
                summaries["class"] = class_summary

        return summaries


# =============================================================================
# ELEMENT PROCESSING HELPERS
# =============================================================================


def _summarize_element(
    element: CodeElement,
    summary_cache: _SummaryCache,
    ollama: OllamaClient,
    config: ProcessingConfig,
) -> str:
    """Generate summary for an element.

    Args:
        element: Element to summarize.
        summary_cache: Cache with parent summaries.
        ollama: Ollama client for LLM.
        config: Processing configuration.

    Returns:
        Generated summary.
    """
    # Get parent summaries for context
    parent_summaries = summary_cache.get_parent_summaries(element)

    # Build prompt with context
    prompt = build_prompt(element, parent_summaries, config.max_code_tokens)

    # Generate with Ollama
    raw_summary = ollama.generate(
        prompt=prompt,
        temperature=config.summarize_temperature,
        max_tokens=config.summarize_max_tokens,
        timeout=config.summarize_timeout,
    )

    # Clean and return
    return clean_summary(raw_summary)


def _embed_element(
    element: CodeElement,
    summary_cache: _SummaryCache,
    ollama_embed: OllamaEmbedClient,
    config: ProcessingConfig,
) -> list[float]:
    """Generate embedding for an element.

    Args:
        element: Element to embed.
        summary_cache: Cache with summaries for context.
        ollama_embed: Ollama embedding client.
        config: Processing configuration.

    Returns:
        Embedding vector.

    Raises:
        ValueError: If embedding validation fails.
    """
    # Build enriched text for embedding
    text = build_embedding_text(element, summary_cache, config.embed_max_context)

    # Generate embedding
    embedding = ollama_embed.embed_single(text, timeout=config.embed_timeout)

    # Validate dimensions
    if not validate_vector(embedding, config.embed_dimensions):
        raise ValueError(
            f"Invalid embedding: expected {config.embed_dimensions} dims, "
            f"got {len(embedding)}"
        )

    # Normalize for cosine similarity
    return normalize_vector(embedding)


def _index_element(
    element: CodeElement,
    summary: str,
    embedding: list[float] | None,
    es_repo: ElasticsearchRepository,
    file_hash: str | None = None,
) -> bool:
    """Index element to Elasticsearch with summary and embedding.

    Args:
        element: Element to index.
        summary: Generated summary.
        embedding: Embedding vector (or None if not embedded).
        es_repo: Elasticsearch repository.
        file_hash: File hash for file-level elements.

    Returns:
        True on success.
    """
    # Index the element
    es_repo.index_element(element, indexed_at=datetime.now(), file_hash=file_hash)

    # Store summary
    es_repo.store_summary(element.element_id, summary)

    # Store embedding if present
    if embedding is not None:
        es_repo.store_embedding(element.element_id, embedding)

    return True


# =============================================================================
# MAIN PROCESSING FUNCTION
# =============================================================================


def process_elements(
    parsed_files: list[ParsedFile],
    scope: str,
    repository: str,
    username: str,
    es_repo: ElasticsearchRepository,
    config: ProcessingConfig | None = None,
    on_progress: Callable[[int, int, str], None] | None = None,
    file_hashes: dict[str, str] | None = None,
) -> ProcessingResult:
    """Process elements: summarize -> embed -> index (atomic per element).

    Processes elements level-by-level (0->1->2->3) to ensure parent
    summaries exist when processing children.

    Args:
        parsed_files: List of parsed files from Phase 3.
        scope: Repository scope.
        repository: Repository name.
        username: Username/branch.
        es_repo: Elasticsearch repository for indexing.
        config: Processing configuration.
        on_progress: Optional callback(completed, total, element_name).
        file_hashes: Optional dict mapping relative_path to file hash.

    Returns:
        ProcessingResult with counts and errors.
    """
    if config is None:
        config = ProcessingConfig()

    result = ProcessingResult(scope=scope, repository=repository, username=username)

    # Collect all elements and organize by level
    all_elements: list[CodeElement] = []
    for pf in parsed_files:
        all_elements.extend(pf.elements)

    # Group by level for hierarchical processing
    elements_by_level: dict[int, list[CodeElement]] = {}
    for elem in all_elements:
        level = elem.level
        if level not in elements_by_level:
            elements_by_level[level] = []
        elements_by_level[level].append(elem)

    # Get all element IDs to check which already exist in ES
    all_element_ids = [e.element_id for e in all_elements]
    existing_ids = es_repo.get_existing_element_ids(all_element_ids)

    # Summary cache for hierarchical context
    summary_cache = _SummaryCache()

    # Populate cache with all elements (for parent lookup)
    for elem in all_elements:
        summary_cache.add_element(elem)

    # Initialize Ollama clients (only if not skipping AI)
    ollama: OllamaClient | None = None
    ollama_embed: OllamaEmbedClient | None = None

    if not config.skip_ai:
        ollama = OllamaClient(config.ollama_url, config.summarize_model)
        ollama_embed = OllamaEmbedClient(config.ollama_url, config.embed_model)

    total = len(all_elements)
    processed_count = 0

    # Process level by level (0, 1, 2, 3)
    for level in sorted(elements_by_level.keys()):
        for element in elements_by_level[level]:
            processed_count += 1

            # Skip if already in ES (fully processed)
            if element.element_id in existing_ids:
                result.elements_skipped += 1
                if on_progress:
                    on_progress(processed_count, total, f"[skip] {element.name}")
                continue

            try:
                # Step 1: Summarize
                if config.skip_ai:
                    summary = f"{element.element_type.title()}: {element.name}"
                else:
                    summary = _summarize_element(
                        element, summary_cache, ollama, config
                    )
                    result.summarized += 1

                # Cache summary for children
                summary_cache.add_summary(element.element_id, summary)

                # Step 2: Embed (if applicable)
                embedding: list[float] | None = None
                if should_embed(element):
                    if config.skip_ai:
                        # Generate dummy embedding for testing
                        embedding = [0.0] * config.embed_dimensions
                    else:
                        embedding = _embed_element(
                            element, summary_cache, ollama_embed, config
                        )
                        result.embedded += 1

                # Step 3: Index to ES (only after summarize+embed complete)
                file_hash = None
                if element.element_type == "file" and file_hashes:
                    file_hash = file_hashes.get(element.relative_path)

                _index_element(element, summary, embedding, es_repo, file_hash)
                result.indexed += 1
                result.elements_processed += 1

                if on_progress:
                    on_progress(processed_count, total, element.name)

            except Exception as e:
                result.elements_failed += 1
                error_msg = f"Failed to process {element.element_id}: {e}"
                result.errors.append(error_msg)

                if on_progress:
                    on_progress(processed_count, total, f"[error] {element.name}")

    return result
