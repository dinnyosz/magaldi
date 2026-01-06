"""Unified element processor - atomic summarize -> embed -> index flow.

Processes elements level-by-level:
- Level 0: Files
- Level 1: Classes
- Level 2: Functions/Methods
- Level 3: Variables

Only indexes to ES after full processing, ensuring ES presence = completion.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
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

    # Parallel processing
    num_workers: int = 4


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

    # Failed elements with errors
    failed_elements: list[tuple[str, str]] = field(default_factory=list)  # (element_id, error)


@dataclass
class TimingStats:
    """Thread-safe timing statistics."""

    _lock: threading.Lock = field(default_factory=threading.Lock)

    wall_times: list[float] = field(default_factory=list)
    summarize_times: list[float] = field(default_factory=list)
    embed_times: list[float] = field(default_factory=list)
    phase_start: float = 0.0

    # Per-type tracking: type -> list of wall times
    wall_times_by_type: dict[str, list[float]] = field(default_factory=dict)
    counts_by_type: dict[str, int] = field(default_factory=dict)  # completed counts
    totals_by_type: dict[str, int] = field(default_factory=dict)  # total counts

    def set_totals_by_type(self, totals: dict[str, int]) -> None:
        """Set total element counts by type."""
        with self._lock:
            self.totals_by_type = dict(totals)
            # Initialize counts
            for t in totals:
                if t not in self.counts_by_type:
                    self.counts_by_type[t] = 0
                if t not in self.wall_times_by_type:
                    self.wall_times_by_type[t] = []

    def record(self, wall_time: float, summarize_time: float, embed_time: float, element_type: str = "") -> None:
        with self._lock:
            self.wall_times.append(wall_time)
            self.summarize_times.append(summarize_time)
            self.embed_times.append(embed_time)
            if element_type:
                if element_type not in self.wall_times_by_type:
                    self.wall_times_by_type[element_type] = []
                self.wall_times_by_type[element_type].append(wall_time)
                self.counts_by_type[element_type] = self.counts_by_type.get(element_type, 0) + 1

    @property
    def avg_wall_time(self) -> float:
        with self._lock:
            return sum(self.wall_times) / len(self.wall_times) if self.wall_times else 0.0

    @property
    def avg_summarize_time(self) -> float:
        with self._lock:
            return sum(self.summarize_times) / len(self.summarize_times) if self.summarize_times else 0.0

    @property
    def avg_embed_time(self) -> float:
        with self._lock:
            return sum(self.embed_times) / len(self.embed_times) if self.embed_times else 0.0

    @property
    def elapsed(self) -> float:
        return time.time() - self.phase_start

    def get_type_stats(self) -> dict[str, tuple[int, int, float]]:
        """Get per-type stats: type -> (completed, total, avg_time)."""
        with self._lock:
            result = {}
            for t in self.totals_by_type:
                completed = self.counts_by_type.get(t, 0)
                total = self.totals_by_type.get(t, 0)
                times = self.wall_times_by_type.get(t, [])
                avg = sum(times) / len(times) if times else 0.0
                result[t] = (completed, total, avg)
            return result

    def eta_seconds(self, completed: int, total: int) -> float | None:
        """Calculate ETA based on per-type averages."""
        with self._lock:
            if completed == 0:
                return None
            # Calculate ETA using per-type averages (inline to avoid deadlock)
            eta = 0.0
            for t in self.totals_by_type:
                done = self.counts_by_type.get(t, 0)
                tot = self.totals_by_type.get(t, 0)
                times = self.wall_times_by_type.get(t, [])
                avg = sum(times) / len(times) if times else 0.0
                remaining = tot - done
                if remaining > 0 and avg > 0:
                    eta += remaining * avg
            return eta if eta > 0 else None


@dataclass
class WorkerStatus:
    """Track what each worker is doing."""

    _lock: threading.Lock = field(default_factory=threading.Lock)
    _status: dict[int, tuple[str, str]] = field(default_factory=dict)  # worker_id -> (element_name, stage)

    def set(self, worker_id: int, element_name: str, stage: str) -> None:
        with self._lock:
            self._status[worker_id] = (element_name, stage)

    def clear(self, worker_id: int) -> None:
        with self._lock:
            self._status.pop(worker_id, None)

    def get_all(self) -> dict[int, tuple[str, str]]:
        with self._lock:
            return dict(self._status)


@dataclass
class ProgressState:
    """Combined state for display updates."""

    total: int
    completed: int
    skipped: int
    failed: int
    timing: TimingStats
    workers: WorkerStatus


@dataclass
class ProcessedElement:
    """Result from processing a single element."""

    element_id: str
    success: bool
    wall_time: float
    summarize_time: float
    embed_time: float
    error: str | None = None


class DependencyTracker:
    """Track element dependencies for parallel processing.

    Rules:
    - Level 0 (files): Always ready
    - Level 1 (classes): Ready when parent file done
    - Level 2 (methods/functions): Ready when parent class done (or file if no class)
    - Level 3 (variables): Ready when parent done
    """

    def __init__(self, elements: list[CodeElement]) -> None:
        self._lock = threading.Lock()
        self._elements = {e.element_id: e for e in elements}
        self._completed: set[str] = set()
        self._in_progress: set[str] = set()

        # Build parent lookup: element_id -> parent_element_id
        self._parents: dict[str, str | None] = {}
        for e in elements:
            self._parents[e.element_id] = e.parent_id

    def get_ready_elements(self, max_count: int = 10) -> list[CodeElement]:
        """Get elements ready for processing (parent done, not started)."""
        with self._lock:
            ready = []
            for eid, element in self._elements.items():
                if eid in self._completed or eid in self._in_progress:
                    continue

                parent_id = self._parents.get(eid)
                if parent_id is None or parent_id in self._completed:
                    ready.append(element)
                    if len(ready) >= max_count:
                        break

            # Mark as in-progress
            for e in ready:
                self._in_progress.add(e.element_id)

            return ready

    def mark_complete(self, element_id: str) -> None:
        """Mark element as completed."""
        with self._lock:
            self._in_progress.discard(element_id)
            self._completed.add(element_id)

    def mark_failed(self, element_id: str) -> None:
        """Mark element as failed (won't block children)."""
        with self._lock:
            self._in_progress.discard(element_id)
            self._completed.add(element_id)  # Treat as done so children can proceed

    def is_complete(self) -> bool:
        """Check if all elements are processed."""
        with self._lock:
            return len(self._completed) == len(self._elements)

    def pending_count(self) -> int:
        """Count elements not yet completed."""
        with self._lock:
            return len(self._elements) - len(self._completed)


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
    elements to be stored in ES first. Thread-safe for parallel processing.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._elements: dict[str, CodeElement] = {}
        self._summaries: dict[str, str] = {}

    def add_element(self, element: CodeElement) -> None:
        """Add element to cache."""
        self._elements[element.element_id] = element

    def add_summary(self, element_id: str, summary: str) -> None:
        """Add summary to cache."""
        with self._lock:
            self._summaries[element_id] = summary

    def get_element(self, element_id: str) -> CodeElement | None:
        """Get element from cache."""
        return self._elements.get(element_id)

    def get_summary(self, element_id: str) -> str | None:
        """Get summary from cache."""
        with self._lock:
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


def _process_single_element(
    element: CodeElement,
    summary_cache: _SummaryCache,
    ollama: OllamaClient | None,
    ollama_embed: OllamaEmbedClient | None,
    config: ProcessingConfig,
    file_hashes: dict[str, str] | None,
    es_repo: ElasticsearchRepository,
    worker_id: int,
    worker_status: WorkerStatus,
    on_status_change: Callable[[], None] | None = None,
) -> ProcessedElement:
    """Process a single element: summarize -> embed -> index.

    Args:
        element: Element to process.
        summary_cache: Cache for summaries.
        ollama: Ollama client for summarization (None if skip_ai).
        ollama_embed: Ollama client for embeddings (None if skip_ai).
        config: Processing configuration.
        file_hashes: Optional dict mapping relative_path to file hash.
        es_repo: Elasticsearch repository for indexing.
        worker_id: Worker thread ID.
        worker_status: Status tracker for workers.
        on_status_change: Optional callback when worker status changes.

    Returns:
        ProcessedElement with timing info and success/error status.
    """
    start_wall = time.time()
    summarize_time = 0.0
    embed_time = 0.0

    # Build hierarchical display name: file.py → Class → method
    def build_display_name() -> str:
        parts = []
        # Add filename (shortened)
        filename = element.relative_path.split("/")[-1] if "/" in element.relative_path else element.relative_path
        if element.element_type != "file":
            parts.append(filename)
        # Add parent class if method
        if element.parent_id:
            parent = summary_cache.get_element(element.parent_id)
            if parent and parent.element_type == "class":
                parts.append(parent.name)
        # Add element name
        parts.append(element.name)
        return " → ".join(parts)

    display_name = build_display_name()

    def update_status(stage: str) -> None:
        worker_status.set(worker_id, display_name, stage)
        if on_status_change:
            on_status_change()

    try:
        # Step 1: Summarize
        update_status("summarizing")
        if config.skip_ai:
            summary = f"{element.element_type.title()}: {element.name}"
        else:
            api_start = time.time()
            summary = _summarize_element(element, summary_cache, ollama, config)
            summarize_time = time.time() - api_start

        # Cache summary for children
        summary_cache.add_summary(element.element_id, summary)

        # Step 2: Embed (if applicable)
        update_status("embedding")
        embedding: list[float] | None = None
        if should_embed(element):
            if config.skip_ai:
                # Generate dummy embedding for testing
                embedding = [0.0] * config.embed_dimensions
            else:
                api_start = time.time()
                embedding = _embed_element(element, summary_cache, ollama_embed, config)
                embed_time = time.time() - api_start

        # Step 3: Index to ES (only after summarize+embed complete)
        update_status("indexing")
        file_hash = None
        if element.element_type == "file" and file_hashes:
            file_hash = file_hashes.get(element.relative_path)

        _index_element(element, summary, embedding, es_repo, file_hash)

        worker_status.clear(worker_id)
        wall_time = time.time() - start_wall

        return ProcessedElement(
            element_id=element.element_id,
            success=True,
            wall_time=wall_time,
            summarize_time=summarize_time,
            embed_time=embed_time,
        )

    except Exception as e:
        worker_status.clear(worker_id)
        wall_time = time.time() - start_wall
        return ProcessedElement(
            element_id=element.element_id,
            success=False,
            wall_time=wall_time,
            summarize_time=summarize_time,
            embed_time=embed_time,
            error=str(e),
        )


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
    on_progress: Callable[[ProgressState], None] | None = None,
    file_hashes: dict[str, str] | None = None,
    on_status_change: Callable[[], None] | None = None,
    worker_status: WorkerStatus | None = None,
    timing_stats: TimingStats | None = None,
) -> ProcessingResult:
    """Process elements: summarize -> embed -> index (atomic per element).

    Uses DependencyTracker and ThreadPoolExecutor for parallel processing
    while respecting parent-child dependencies.

    Args:
        parsed_files: List of parsed files from Phase 3.
        scope: Repository scope.
        repository: Repository name.
        username: Username/branch.
        es_repo: Elasticsearch repository for indexing.
        config: Processing configuration.
        on_progress: Optional callback(ProgressState) for progress updates.
        file_hashes: Optional dict mapping relative_path to file hash.
        on_status_change: Optional callback when any worker status changes.
        worker_status: Optional shared WorkerStatus (created if not provided).
        timing_stats: Optional shared TimingStats (created if not provided).

    Returns:
        ProcessingResult with counts and errors.
    """
    if config is None:
        config = ProcessingConfig()

    result = ProcessingResult(scope=scope, repository=repository, username=username)

    # Handle deletions first - remove old elements for files being reprocessed
    # This ensures modified files get their old elements removed before new ones are indexed
    for pf in parsed_files:
        if pf.elements:
            es_repo.delete_by_file(
                scope,
                repository,
                username,
                pf.file_info.relative_path,
            )

    # Collect all elements
    all_elements: list[CodeElement] = []
    for pf in parsed_files:
        all_elements.extend(pf.elements)

    if not all_elements:
        return result

    # Get all element IDs to check which already exist in ES
    all_element_ids = [e.element_id for e in all_elements]
    existing_ids = es_repo.get_existing_element_ids(all_element_ids)

    # Filter out already-existing elements and count skipped
    elements_to_process = []
    for elem in all_elements:
        if elem.element_id in existing_ids:
            result.elements_skipped += 1
        else:
            elements_to_process.append(elem)

    total = len(all_elements)

    if not elements_to_process:
        # All elements already exist
        return result

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

    # Initialize tracking structures (use provided or create new)
    dependency_tracker = DependencyTracker(elements_to_process)
    if timing_stats is None:
        timing_stats = TimingStats()
    timing_stats.phase_start = time.time()
    if worker_status is None:
        worker_status = WorkerStatus()

    # Count elements by type for per-type ETA
    totals_by_type: dict[str, int] = {}
    for elem in elements_to_process:
        totals_by_type[elem.element_type] = totals_by_type.get(elem.element_type, 0) + 1
    timing_stats.set_totals_by_type(totals_by_type)

    # Track completed/failed counts for progress
    completed_count = result.elements_skipped  # Start with skipped count
    failed_count = 0

    # Worker ID pool - reuse IDs 0 to num_workers-1
    available_worker_ids: list[int] = list(range(config.num_workers))
    worker_id_lock = threading.Lock()

    def acquire_worker_id() -> int:
        """Get an available worker ID from the pool."""
        with worker_id_lock:
            if available_worker_ids:
                return available_worker_ids.pop(0)
            return 0  # Fallback

    def release_worker_id(wid: int) -> None:
        """Return a worker ID to the pool."""
        with worker_id_lock:
            if wid not in available_worker_ids:
                available_worker_ids.append(wid)

    def process_wrapper(element: CodeElement) -> tuple[ProcessedElement, int]:
        """Wrapper to assign worker ID and call _process_single_element."""
        wid = acquire_worker_id()
        proc_result = _process_single_element(
            element=element,
            summary_cache=summary_cache,
            ollama=ollama,
            ollama_embed=ollama_embed,
            config=config,
            file_hashes=file_hashes,
            es_repo=es_repo,
            worker_id=wid,
            worker_status=worker_status,
            on_status_change=on_status_change,
        )
        return (proc_result, wid)

    # Process elements in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=config.num_workers) as executor:
        # Map of future -> element for tracking
        future_to_element: dict = {}

        while not dependency_tracker.is_complete():
            # Get elements that are ready (parents completed)
            ready_elements = dependency_tracker.get_ready_elements(
                max_count=config.num_workers * 2
            )

            # Submit new tasks for ready elements
            for element in ready_elements:
                future = executor.submit(process_wrapper, element)
                future_to_element[future] = element

            if not future_to_element:
                # No futures pending and not complete - shouldn't happen
                # but break to avoid infinite loop
                break

            # Wait for at least one to complete
            done, _ = wait(future_to_element.keys(), return_when=FIRST_COMPLETED)

            for future in done:
                element = future_to_element.pop(future)
                processed, worker_id = future.result()

                # Release worker ID back to pool
                release_worker_id(worker_id)

                # Record timing with element type
                timing_stats.record(processed.wall_time, processed.summarize_time, processed.embed_time, element.element_type)

                if processed.success:
                    dependency_tracker.mark_complete(element.element_id)
                    result.elements_processed += 1
                    result.indexed += 1
                    if not config.skip_ai:
                        result.summarized += 1
                        if should_embed(element):
                            result.embedded += 1
                    completed_count += 1
                else:
                    dependency_tracker.mark_failed(element.element_id)
                    result.elements_failed += 1
                    failed_count += 1
                    error_msg = f"Failed to process {element.element_id}: {processed.error}"
                    result.errors.append(error_msg)
                    result.failed_elements.append((element.element_id, processed.error or "Unknown error"))

                # Report progress
                if on_progress:
                    progress_state = ProgressState(
                        total=total,
                        completed=completed_count,
                        skipped=result.elements_skipped,
                        failed=failed_count,
                        timing=timing_stats,
                        workers=worker_status,
                    )
                    on_progress(progress_state)

    return result
