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
from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from shared.config import MagaldiConfig

from shared.config import ModelConfig
from shared.db.elasticsearch import ElasticsearchRepository
from shared.db.redis import RedisSummarizationJobRepository, RedisEmbeddingJobRepository
from shared.ai.embedding import (
    EmbeddingConfig,
    CodeEmbeddingClient,
    build_summary_embedding_text,
    build_code_embedding_text,
    normalize_vector,
    validate_vector,
)
from magaldi_core.code_parser import CodeElement, ParsedFile
from shared.ai.summarization import (
    SummarizationLLMClient,
    SummarizationConfig,
    build_prompt,
    clean_summary,
)
from shared.ai.context_size import compute_element_num_ctx


def _get_model_display_name(model_config: ModelConfig, num_ctx: int) -> str:
    """Get the display name for a model, including tier suffix for Ollama models.

    Args:
        model_config: Model configuration.
        num_ctx: Context size being used.

    Returns:
        Model name with tier suffix for Ollama models (e.g., "qwen3:4b-instruct-4k"),
        or the original name for other providers.
    """
    if model_config.provider == "ollama":
        from shared.ai.ollama_models import get_tiered_model_name

        return get_tiered_model_name(model_config.name, num_ctx)
    return model_config.name


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class ProcessingConfig:
    """Configuration for unified processing.

    Uses ModelConfig objects that encapsulate name, provider, url, api_key.
    """

    # Model configurations (encapsulate name + provider + url + api_key)
    # Note: Each llamacpp model needs its own server instance on a different port
    summarize_model: ModelConfig = field(default_factory=lambda: ModelConfig(
        name="qwen3:4b-instruct", provider="llamacpp", url="http://localhost:8080"
    ))
    summarize_model_small: ModelConfig = field(default_factory=lambda: ModelConfig(
        name="qwen3:1.7b", provider="llamacpp", url="http://localhost:8081"
    ))
    embed_model: ModelConfig = field(default_factory=lambda: ModelConfig(
        name="snowflake-arctic-embed2", provider="ollama", url="http://localhost:11434", dimensions=1024
    ))

    skip_ai: bool = False

    # Summarization settings (based on arxiv.org/html/2507.03160v2)
    summarize_temperature: float = 0.2
    summarize_max_tokens: int = 512
    summarize_timeout: int = 180  # 3 minutes to handle queue wait with many workers
    max_code_tokens: int = 4000

    # Embedding settings
    embed_dimensions: int = 1024
    embed_max_context: int = 8192
    embed_timeout: int = 120  # 2 minutes for embedding batches

    # Parallel processing
    num_workers: int = 4

    # Context sizes per element type (for LLM num_ctx parameter)
    context_sizes: dict[str, int] = field(default_factory=dict)

    def get_model_for_element_type(self, element_type: str) -> "ModelConfig":
        """Get the appropriate model config for an element type.

        Uses small model for functions, methods, variables, constants.
        Uses main model for files, classes.
        """
        if element_type in ("function", "method", "variable", "constant"):
            return self.summarize_model_small
        return self.summarize_model


@dataclass
class ProcessingResult:
    """Result of unified processing."""

    scope: str
    repository: str
    username: str

    # Counts
    elements_processed: int = 0
    elements_skipped: int = 0  # Already in ES with same content
    elements_deleted: int = 0  # Stale elements removed (in ES but not in code)
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
    """Thread-safe timing statistics using running totals."""

    _lock: threading.Lock = field(default_factory=threading.Lock)
    phase_start: float = 0.0

    # Per-type tracking
    total_summarize_by_type: dict[str, float] = field(default_factory=dict)
    total_embed_by_type: dict[str, float] = field(default_factory=dict)  # Legacy: total embed time (summary + code)
    total_summary_embed_by_type: dict[str, float] = field(default_factory=dict)  # Summary embedding time
    total_code_embed_by_type: dict[str, float] = field(default_factory=dict)  # Code embedding time
    summarize_counts_by_type: dict[str, int] = field(default_factory=dict)  # count of summarized
    embed_counts_by_type: dict[str, int] = field(default_factory=dict)  # count of embedded (legacy)
    summary_embed_counts_by_type: dict[str, int] = field(default_factory=dict)  # count of summary embeds
    code_embed_counts_by_type: dict[str, int] = field(default_factory=dict)  # count of code embeds
    totals_by_type: dict[str, int] = field(default_factory=dict)  # total element counts

    def set_totals_by_type(self, totals: dict[str, int]) -> None:
        """Set total element counts by type."""
        with self._lock:
            self.totals_by_type = dict(totals)
            # Initialize per-type tracking
            for t in totals:
                if t not in self.summarize_counts_by_type:
                    self.summarize_counts_by_type[t] = 0
                if t not in self.embed_counts_by_type:
                    self.embed_counts_by_type[t] = 0
                if t not in self.summary_embed_counts_by_type:
                    self.summary_embed_counts_by_type[t] = 0
                if t not in self.code_embed_counts_by_type:
                    self.code_embed_counts_by_type[t] = 0
                if t not in self.total_summarize_by_type:
                    self.total_summarize_by_type[t] = 0.0
                if t not in self.total_embed_by_type:
                    self.total_embed_by_type[t] = 0.0
                if t not in self.total_summary_embed_by_type:
                    self.total_summary_embed_by_type[t] = 0.0
                if t not in self.total_code_embed_by_type:
                    self.total_code_embed_by_type[t] = 0.0

    def record(
        self,
        wall_time: float,
        summarize_time: float,
        embed_time: float,
        element_type: str = "",
        was_embedded: bool = True,
        summary_embed_time: float = 0.0,
        code_embed_time: float = 0.0,
    ) -> None:
        """Record timing for a completed element.

        Args:
            wall_time: Total wall clock time.
            summarize_time: Time spent on LLM summarization.
            embed_time: Total embedding time (legacy, for backwards compat).
            element_type: Type of element (file, class, function, etc).
            was_embedded: Whether the element was embedded.
            summary_embed_time: Time spent on summary embedding.
            code_embed_time: Time spent on code embedding.
        """
        with self._lock:
            if element_type:
                if element_type not in self.total_summarize_by_type:
                    self.total_summarize_by_type[element_type] = 0.0
                if element_type not in self.total_embed_by_type:
                    self.total_embed_by_type[element_type] = 0.0
                if element_type not in self.total_summary_embed_by_type:
                    self.total_summary_embed_by_type[element_type] = 0.0
                if element_type not in self.total_code_embed_by_type:
                    self.total_code_embed_by_type[element_type] = 0.0
                # Always record summarize time (every element is summarized)
                self.total_summarize_by_type[element_type] += summarize_time
                self.summarize_counts_by_type[element_type] = self.summarize_counts_by_type.get(element_type, 0) + 1
                # Only record embed time if element was actually embedded
                if was_embedded and embed_time > 0:
                    self.total_embed_by_type[element_type] += embed_time
                    self.embed_counts_by_type[element_type] = self.embed_counts_by_type.get(element_type, 0) + 1
                    # Track dual embedding times separately
                    if summary_embed_time > 0:
                        self.total_summary_embed_by_type[element_type] += summary_embed_time
                        self.summary_embed_counts_by_type[element_type] = self.summary_embed_counts_by_type.get(element_type, 0) + 1
                    if code_embed_time > 0:
                        self.total_code_embed_by_type[element_type] += code_embed_time
                        self.code_embed_counts_by_type[element_type] = self.code_embed_counts_by_type.get(element_type, 0) + 1

    @property
    def total_summarize_count(self) -> int:
        """Total number of elements summarized."""
        with self._lock:
            return sum(self.summarize_counts_by_type.values())

    @property
    def total_embed_count(self) -> int:
        """Total number of elements embedded."""
        with self._lock:
            return sum(self.embed_counts_by_type.values())

    @property
    def avg_summarize_time(self) -> float:
        """Global average summarize time = sum(all type totals) / sum(all summarize counts)."""
        with self._lock:
            total_time = sum(self.total_summarize_by_type.values())
            total_count = sum(self.summarize_counts_by_type.values())
            return total_time / total_count if total_count > 0 else 0.0

    @property
    def avg_embed_time(self) -> float:
        """Global average embed time = sum(all type totals) / sum(all embed counts)."""
        with self._lock:
            total_time = sum(self.total_embed_by_type.values())
            total_count = sum(self.embed_counts_by_type.values())
            return total_time / total_count if total_count > 0 else 0.0

    @property
    def avg_summary_embed_time(self) -> float:
        """Global average summary embedding time."""
        with self._lock:
            total_time = sum(self.total_summary_embed_by_type.values())
            total_count = sum(self.summary_embed_counts_by_type.values())
            return total_time / total_count if total_count > 0 else 0.0

    @property
    def avg_code_embed_time(self) -> float:
        """Global average code embedding time."""
        with self._lock:
            total_time = sum(self.total_code_embed_by_type.values())
            total_count = sum(self.code_embed_counts_by_type.values())
            return total_time / total_count if total_count > 0 else 0.0

    @property
    def elapsed(self) -> float:
        return time.time() - self.phase_start

    def get_type_stats(self) -> dict[str, tuple[int, int, float, float, float]]:
        """Get per-type stats: type -> (completed, total, avg_api, avg_summ, avg_embed)."""
        with self._lock:
            result = {}
            for t in self.totals_by_type:
                completed = self.summarize_counts_by_type.get(t, 0)  # Use summarize count as "completed"
                total = self.totals_by_type.get(t, 0)
                total_summ = self.total_summarize_by_type.get(t, 0.0)
                total_embed = self.total_embed_by_type.get(t, 0.0)
                summ_count = self.summarize_counts_by_type.get(t, 0)
                embed_count = self.embed_counts_by_type.get(t, 0)
                avg_summ = total_summ / summ_count if summ_count > 0 else 0.0
                avg_embed = total_embed / embed_count if embed_count > 0 else 0.0
                avg_api = avg_summ + avg_embed  # Use API time as "wall" for ETA
                result[t] = (completed, total, avg_api, avg_summ, avg_embed)
            return result

    def eta_seconds(self, completed: int, total: int, num_workers: int = 1) -> float | None:
        """Calculate ETA based on per-type API time averages.

        Args:
            completed: Number of elements completed.
            total: Total number of elements.
            num_workers: Number of parallel workers (divides total work time).

        Returns:
            Estimated seconds remaining, or None if cannot estimate.
        """
        with self._lock:
            if completed == 0:
                return None

            # Global average API time as fallback
            total_api_time = sum(self.total_summarize_by_type.values()) + sum(self.total_embed_by_type.values())
            total_count = sum(self.summarize_counts_by_type.values())
            global_avg = total_api_time / total_count if total_count > 0 else 0.0

            # Calculate total remaining work time using per-type averages
            total_work_time = 0.0
            for t in self.totals_by_type:
                done = self.summarize_counts_by_type.get(t, 0)
                tot = self.totals_by_type.get(t, 0)
                if done > 0:
                    type_total = self.total_summarize_by_type.get(t, 0.0) + self.total_embed_by_type.get(t, 0.0)
                    avg = type_total / done
                else:
                    avg = global_avg
                remaining = tot - done
                if remaining > 0 and avg > 0:
                    total_work_time += remaining * avg

            if total_work_time <= 0:
                return None

            # Divide by number of workers to get wall clock time
            # (parallel workers process elements concurrently)
            eta = total_work_time / max(num_workers, 1)
            return eta


@dataclass
class WorkerStatus:
    """Track what each worker is doing."""

    _lock: threading.Lock = field(default_factory=threading.Lock)
    _status: dict[int, tuple[str, str, str, str, float]] = field(default_factory=dict)  # worker_id -> (element_name, stage, model, ctx_size, start_time)

    def set(self, worker_id: int, element_name: str, stage: str, model: str = "", ctx_size: str = "", start_time: float = 0.0) -> None:
        with self._lock:
            self._status[worker_id] = (element_name, stage, model, ctx_size, start_time)

    def clear(self, worker_id: int) -> None:
        with self._lock:
            self._status.pop(worker_id, None)

    def get_all(self) -> dict[int, tuple[str, str, str, str, float]]:
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
    num_workers: int = 1
    recent_errors: list[tuple[str, str]] = field(default_factory=list)  # (element_name, error)


@dataclass
class ProcessedElement:
    """Result from processing a single element."""

    element_id: str
    success: bool
    wall_time: float
    summarize_time: float
    embed_time: float  # Total embed time (summary + code)
    summary_embed_time: float = 0.0  # Time for summary embedding
    code_embed_time: float = 0.0  # Time for code embedding
    error: str | None = None


class DependencyTracker:
    """Track element dependencies for parallel processing.

    Rules:
    - Level 0 (files): Always ready
    - Level 1 (classes): Ready when parent file done
    - Level 2 (methods/functions): Ready when parent class done (or file if no class)
    - Level 3 (variables): Ready when parent done

    Tier batching: All workers process the same context tier together to minimize
    model reloads. Only moves to next tier when current tier has no ready elements.

    Dynamic worker scaling: Smaller context tiers allow more parallel workers since
    they use less GPU memory. Larger tiers are limited to prevent OOM/slowdowns.
    """

    # Standard context tiers (must match ollama_models.CONTEXT_TIERS)
    CONTEXT_TIERS = [2048, 4096, 8192, 16384, 32768]

    # Max concurrent workers per tier (inversely proportional to context size)
    # Smaller contexts = more parallelism, larger contexts = less to avoid GPU saturation
    # Tuned for M4 Pro 48GB - adjust for different hardware
    TIER_MAX_WORKERS = {
        2048: 12,  # Small context - max parallelism
        4096: 8,   # Medium-small
        8192: 4,   # Medium
        16384: 2,  # Large - limited parallelism
        32768: 1,  # Very large - sequential to avoid OOM
    }

    def __init__(
        self, elements: list[CodeElement], context_sizes: dict[str, int] | None = None
    ) -> None:
        self._lock = threading.Lock()
        self._elements = {e.element_id: e for e in elements}
        self._completed: set[str] = set()
        self._in_progress: set[str] = set()
        self._context_sizes = context_sizes or {}
        self._current_tier: int | None = None  # Current tier all workers use

        # Build parent lookup: element_id -> parent_element_id
        self._parents: dict[str, str | None] = {}
        for e in elements:
            self._parents[e.element_id] = e.parent_id

    def _get_tier(self, element_id: str) -> int:
        """Get the context tier for an element (snaps to standard tiers)."""
        ctx = self._context_sizes.get(element_id, 2048)
        # Find the tier this context size belongs to
        for tier in self.CONTEXT_TIERS:
            if ctx <= tier:
                return tier
        return self.CONTEXT_TIERS[-1]  # Max tier

    def _get_all_ready(self) -> list[CodeElement]:
        """Get all ready elements (parent done, not started). Must hold lock."""
        ready = []
        for eid, element in self._elements.items():
            if eid in self._completed or eid in self._in_progress:
                continue

            parent_id = self._parents.get(eid)
            # Element is ready if:
            # - No parent (level 0)
            # - Parent completed
            # - Parent not in elements_to_process (was skipped/unchanged)
            parent_ready = (
                parent_id is None
                or parent_id in self._completed
                or parent_id not in self._elements
            )
            if parent_ready:
                ready.append(element)
        return ready

    def get_ready_elements(self, max_count: int = 10) -> list[CodeElement]:
        """Get elements ready for processing (parent done, not started).

        Uses tier batching: all workers process the same context tier together.
        Only moves to next tier when current tier has no ready elements.
        This minimizes Ollama model reloads.

        Dynamic worker scaling: Returns fewer elements for larger tiers to limit
        concurrent GPU memory usage (see TIER_MAX_WORKERS).
        """
        with self._lock:
            ready = self._get_all_ready()
            if not ready:
                return []

            # Group ready elements by tier
            by_tier: dict[int, list[CodeElement]] = {}
            for elem in ready:
                tier = self._get_tier(elem.element_id)
                by_tier.setdefault(tier, []).append(elem)

            # Determine which tier to use
            if self._current_tier is not None and self._current_tier in by_tier:
                # Continue with current tier if it has ready elements
                tier = self._current_tier
            else:
                # Move to smallest tier that has ready elements
                tier = min(by_tier.keys())
                self._current_tier = tier

            # Apply tier-specific worker limit (dynamic scaling)
            # Count how many of this tier are already in progress
            tier_limit = self.TIER_MAX_WORKERS.get(tier, 4)
            tier_in_progress = sum(
                1 for eid in self._in_progress if self._get_tier(eid) == tier
            )
            # Only return enough to reach tier limit, not add to it
            slots_available = max(0, tier_limit - tier_in_progress)
            effective_limit = min(max_count, slots_available)

            # Get elements from current tier only
            tier_ready = by_tier[tier][:effective_limit]

            # Mark as in-progress
            for e in tier_ready:
                self._in_progress.add(e.element_id)

            return tier_ready

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

    def get_current_tier(self) -> int | None:
        """Get the current context tier being processed."""
        with self._lock:
            return self._current_tier

    def get_current_max_workers(self) -> int:
        """Get max workers for the current tier (for status display)."""
        with self._lock:
            if self._current_tier is None:
                return self.TIER_MAX_WORKERS.get(self.CONTEXT_TIERS[0], 8)
            return self.TIER_MAX_WORKERS.get(self._current_tier, 4)

    def get_tier_stats(self) -> dict[int, tuple[int, int]]:
        """Get (ready, total) counts per tier for pending elements."""
        with self._lock:
            ready = self._get_all_ready()
            ready_ids = {e.element_id for e in ready}

            stats: dict[int, tuple[int, int]] = {}
            for eid in self._elements:
                if eid in self._completed:
                    continue
                tier = self._get_tier(eid)
                r, t = stats.get(tier, (0, 0))
                is_ready = 1 if eid in ready_ids else 0
                stats[tier] = (r + is_ready, t + 1)
            return stats


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

    # Constants (UPPER_CASE module-level) always get embedded
    if element.element_type == "constant":
        return True

    # Variables (class variables, etc.) - embed if they have docstrings or usages
    if element.element_type == "variable":
        if element.docstring:
            return True
        if element.context_usages:  # Has usages = likely important
            return True

    return False


# =============================================================================
# REDIS JOB TRACKER
# =============================================================================


class RedisJobTracker:
    """Track processing jobs in Redis for dashboard monitoring.

    This writes job status to Redis so the dashboard can show queue activity
    during synchronous processing.
    """

    def __init__(
        self,
        config: "MagaldiConfig",
        scope: str,
        repository: str,
        username: str,
    ) -> None:
        self._scope = scope
        self._repository = repository
        self._username = username
        self._sum_repo = RedisSummarizationJobRepository(config)
        self._emb_repo = RedisEmbeddingJobRepository(config)
        self._lock = threading.Lock()

    def clear_queues(self) -> None:
        """Clear all Redis queue keys for this scope/repository/username."""
        client = self._sum_repo._get_client()

        # Keys to delete for summarization and embedding
        keys_to_delete = [
            f"magaldi:summarization:jobs:{self._scope}:{self._repository}:{self._username}",
            f"magaldi:summarization:running:{self._scope}:{self._repository}:{self._username}",
            f"magaldi:summarization:queue:{self._scope}:{self._repository}:{self._username}",
            f"magaldi:embedding:jobs:{self._scope}:{self._repository}:{self._username}",
            f"magaldi:embedding:running:{self._scope}:{self._repository}:{self._username}",
            f"magaldi:embedding:queue:{self._scope}:{self._repository}:{self._username}",
        ]

        for key in keys_to_delete:
            client.delete(key)

    def add_pending_jobs(self, elements: list["CodeElement"]) -> None:
        """Add all elements as pending jobs to Redis."""
        for element in elements:
            # Add summarization job (all elements get summarized)
            self._sum_repo.add_job(
                element_id=element.element_id,
                scope=self._scope,
                repository=self._repository,
                username=self._username,
                level=element.level,
                parent_id=element.parent_id,
                dependencies_met=True,  # We handle dependencies in processor
                priority=100 - element.level,
            )
            # Add embedding job (only for embeddable elements)
            if should_embed(element):
                self._emb_repo.add_job(
                    element_id=element.element_id,
                    scope=self._scope,
                    repository=self._repository,
                    username=self._username,
                )

    def mark_running(self, element_id: str, was_embedded: bool = True) -> None:
        """Mark element as running in Redis."""
        with self._lock:
            # Update job status to running and add to running set
            client = self._sum_repo._get_client()
            jobs_key = f"magaldi:summarization:jobs:{self._scope}:{self._repository}:{self._username}"
            running_key = f"magaldi:summarization:running:{self._scope}:{self._repository}:{self._username}"

            # Update status in job hash
            import json
            job_data = client.hget(jobs_key, element_id)
            if job_data:
                job = json.loads(job_data)
                job["status"] = "running"
                client.hset(jobs_key, element_id, json.dumps(job))
                client.sadd(running_key, element_id)

            if was_embedded:
                emb_jobs_key = f"magaldi:embedding:jobs:{self._scope}:{self._repository}:{self._username}"
                emb_running_key = f"magaldi:embedding:running:{self._scope}:{self._repository}:{self._username}"
                emb_data = client.hget(emb_jobs_key, element_id)
                if emb_data:
                    emb_job = json.loads(emb_data)
                    emb_job["status"] = "running"
                    client.hset(emb_jobs_key, element_id, json.dumps(emb_job))
                    client.sadd(emb_running_key, element_id)

    def mark_completed(self, element_id: str, was_embedded: bool = True) -> None:
        """Mark element as completed in Redis."""
        with self._lock:
            self._sum_repo.mark_completed(
                element_id, self._scope, self._repository, self._username
            )
            if was_embedded:
                self._emb_repo.mark_completed(
                    element_id, self._scope, self._repository, self._username
                )

    def mark_failed(self, element_id: str, error: str, was_embedded: bool = True) -> None:
        """Mark element as failed in Redis."""
        with self._lock:
            self._sum_repo.mark_failed(
                element_id, self._scope, self._repository, self._username, error
            )
            if was_embedded:
                self._emb_repo.mark_failed(
                    element_id, self._scope, self._repository, self._username, error
                )

    def close(self) -> None:
        """Close Redis connections."""
        self._sum_repo.close()
        self._emb_repo.close()


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
    llm_client: SummarizationLLMClient,
    config: ProcessingConfig,
) -> str:
    """Generate summary for an element.

    Args:
        element: Element to summarize.
        summary_cache: Cache with parent summaries.
        llm_client: LLM client for text generation.
        config: Processing configuration.

    Returns:
        Generated summary.
    """
    # Get parent summaries for context
    parent_summaries = summary_cache.get_parent_summaries(element)

    # Build prompt with context
    prompt = build_prompt(element, parent_summaries, config.max_code_tokens)

    # Generate summary (select model based on element type)
    model_config = config.get_model_for_element_type(element.element_type)
    # Compute per-element context size for optimal KV cache efficiency
    num_ctx = compute_element_num_ctx(
        element.element_type,
        len(element.raw_code or ""),
    )
    raw_summary = llm_client.generate(
        prompt=prompt,
        temperature=config.summarize_temperature,
        max_tokens=config.summarize_max_tokens,
        timeout=config.summarize_timeout,
        model=model_config.name,
        num_ctx=num_ctx,
    )

    # Clean and return
    return clean_summary(raw_summary)


def _embed_element(
    element: CodeElement,
    summary_cache: _SummaryCache,
    embed_client: CodeEmbeddingClient,
    config: ProcessingConfig,
    on_stage_change: Callable[[str], None] | None = None,
) -> tuple[list[float], list[float], float, float]:
    """Generate both embeddings for an element.

    Args:
        element: Element to embed.
        summary_cache: Cache with summaries for context.
        embed_client: Embedding client.
        config: Processing configuration.
        on_stage_change: Optional callback to update status stage.

    Returns:
        Tuple of (summary_embedding, code_embedding, summary_embed_time, code_embed_time).

    Raises:
        ValueError: If embedding validation fails.
    """
    import time as _time

    # Summary embedding (existing logic)
    if on_stage_change:
        on_stage_change("summ_embed")
    summary_text = build_summary_embedding_text(element, summary_cache, config.embed_max_context)
    summary_start = _time.time()
    summary_embedding = embed_client.embed_single(summary_text, timeout=config.embed_timeout)
    summary_embed_time = _time.time() - summary_start

    # Validate dimensions
    if not validate_vector(summary_embedding, config.embed_dimensions):
        raise ValueError(
            f"Invalid summary embedding: expected {config.embed_dimensions} dims, "
            f"got {len(summary_embedding)}"
        )
    summary_embedding = normalize_vector(summary_embedding)

    # Code embedding (new)
    if on_stage_change:
        on_stage_change("code_embed")
    code_text = build_code_embedding_text(element, config.embed_max_context)
    code_start = _time.time()
    code_embedding = embed_client.embed_single(code_text, timeout=config.embed_timeout)
    code_embed_time = _time.time() - code_start

    # Validate dimensions
    if not validate_vector(code_embedding, config.embed_dimensions):
        raise ValueError(
            f"Invalid code embedding: expected {config.embed_dimensions} dims, "
            f"got {len(code_embedding)}"
        )
    code_embedding = normalize_vector(code_embedding)

    return summary_embedding, code_embedding, summary_embed_time, code_embed_time


def _index_element(
    element: CodeElement,
    summary: str,
    summary_embedding: list[float] | None,
    code_embedding: list[float] | None,
    es_repo: ElasticsearchRepository,
    file_hash: str | None = None,
    element_count: int | None = None,
) -> bool:
    """Index element to Elasticsearch with summary and both embeddings.

    Args:
        element: Element to index.
        summary: Generated summary.
        summary_embedding: Summary embedding vector (or None if not embedded).
        code_embedding: Code embedding vector (or None if not embedded).
        es_repo: Elasticsearch repository.
        file_hash: File hash for all elements.
        element_count: Total element count in file (only for file-level elements).

    Returns:
        True on success.
    """
    # Index the element
    es_repo.index_element(element, indexed_at=datetime.now(), file_hash=file_hash, element_count=element_count)

    # Store summary
    es_repo.store_summary(element.element_id, summary)

    # Store embeddings if present (using type-specific methods)
    if summary_embedding is not None:
        es_repo.store_summary_embedding(element.element_id, summary_embedding)
    if code_embedding is not None:
        es_repo.store_code_embedding(element.element_id, code_embedding)

    # Store imports for file elements
    if element.element_type == "file" and element.imports:
        imports_data = [
            {"name": imp.name, "module": imp.module, "alias": imp.alias, "line": imp.line}
            for imp in element.imports
        ]
        es_repo.store_imports(element.element_id, imports_data)

    # Store calls for function/method elements
    if element.element_type in ("function", "method") and element.calls:
        calls_data = [
            {"name": call.name, "receiver": call.receiver, "line": call.line, "resolved_id": call.resolved_id}
            for call in element.calls
        ]
        es_repo.store_calls(element.element_id, calls_data)

    return True


def _process_single_element(
    element: CodeElement,
    summary_cache: _SummaryCache,
    llm_client: SummarizationLLMClient | None,
    embed_client: CodeEmbeddingClient | None,
    config: ProcessingConfig,
    file_hashes: dict[str, str] | None,
    element_counts: dict[str, int] | None,
    es_repo: ElasticsearchRepository,
    worker_id: int,
    worker_status: WorkerStatus,
    on_status_change: Callable[[], None] | None = None,
) -> ProcessedElement:
    """Process a single element: summarize -> embed -> index.

    Args:
        element: Element to process.
        summary_cache: Cache for summaries.
        llm_client: LLM client for summarization (None if skip_ai).
        embed_client: Embedding client (None if skip_ai).
        config: Processing configuration.
        file_hashes: Optional dict mapping relative_path to file hash.
        element_counts: Optional dict mapping relative_path to element count.
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
    summary_embed_time = 0.0
    code_embed_time = 0.0

    # Build hierarchical display name: [type] .../path/file.py → Class → method
    def build_display_name(max_path_len: int = 40) -> str:
        parts = []
        # Add path (truncated from left if too long)
        path = element.relative_path
        if len(path) > max_path_len:
            path = "..." + path[-(max_path_len - 3):]
        if element.element_type != "file":
            parts.append(path)
        # Add parent class if method
        if element.parent_id:
            parent = summary_cache.get_element(element.parent_id)
            if parent and parent.element_type == "class":
                parts.append(parent.name)
        # Add element name
        parts.append(element.name)
        # Prefix with element type (use angle brackets to avoid Rich markup interpretation)
        return f"<{element.element_type}> " + " → ".join(parts)

    display_name = build_display_name()
    # Get model for this element type
    element_model = config.get_model_for_element_type(element.element_type)

    # Track current stage start time for elapsed display
    stage_start_time = time.time()

    def update_status(stage: str, model: str = "", ctx_size: str = "") -> None:
        nonlocal stage_start_time
        stage_start_time = time.time()
        worker_status.set(worker_id, display_name, stage, model, ctx_size, stage_start_time)
        if on_status_change:
            on_status_change()

    try:
        # Step 1: Summarize
        # Compute context tier for display
        num_ctx = compute_element_num_ctx(
            element.element_type,
            len(element.raw_code or ""),
        )
        # Format tier compactly: 2048 -> "2K", 32768 -> "32K"
        ctx_display = f"{num_ctx // 1024}K" if num_ctx >= 1024 else str(num_ctx)
        # Display tiered model name for Ollama (e.g., "qwen3:4b-instruct-4k")
        model_display = _get_model_display_name(element_model, num_ctx)
        update_status("summarizing", model_display, ctx_display)
        if config.skip_ai:
            summary = f"{element.element_type.title()}: {element.name}"
        else:
            api_start = time.time()
            summary = _summarize_element(element, summary_cache, llm_client, config)
            summarize_time = time.time() - api_start

        # Cache summary for children
        summary_cache.add_summary(element.element_id, summary)

        # Step 2: Embed (if applicable) - generate both summary and code embeddings
        summary_embedding: list[float] | None = None
        code_embedding: list[float] | None = None
        if should_embed(element):
            if config.skip_ai:
                # Generate dummy embeddings for testing
                update_status("summ_embed", config.embed_model.name, "-")
                summary_embedding = [0.0] * config.embed_dimensions
                update_status("code_embed", config.embed_model.name, "-")
                code_embedding = [0.0] * config.embed_dimensions
            else:
                # Generate both embeddings (returns tuple with timing)
                # Pass callback to update status between embedding phases
                def on_embed_stage(stage: str) -> None:
                    update_status(stage, config.embed_model.name, "-")
                summary_embedding, code_embedding, summary_embed_time, code_embed_time = _embed_element(
                    element, summary_cache, embed_client, config, on_embed_stage
                )
                embed_time = summary_embed_time + code_embed_time

        # Step 3: Index to ES (only after summarize+embed complete)
        update_status("indexing")
        # Store file_hash on ALL elements (not just file elements) for change detection
        # This allows us to delete all elements by file_hash if needed
        file_hash = file_hashes.get(element.relative_path) if file_hashes else None
        # Store element_count only on FILE elements for completeness verification
        element_count = None
        if element.element_type == "file" and element_counts:
            element_count = element_counts.get(element.relative_path)

        _index_element(element, summary, summary_embedding, code_embedding, es_repo, file_hash, element_count)

        worker_status.clear(worker_id)
        wall_time = time.time() - start_wall

        return ProcessedElement(
            element_id=element.element_id,
            success=True,
            wall_time=wall_time,
            summarize_time=summarize_time,
            embed_time=embed_time,
            summary_embed_time=summary_embed_time,
            code_embed_time=code_embed_time,
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
            summary_embed_time=summary_embed_time,
            code_embed_time=code_embed_time,
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
    magaldi_config: "MagaldiConfig | None" = None,
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
        magaldi_config: Optional Magaldi config for Redis job tracking.

    Returns:
        ProcessingResult with counts and errors.
    """
    if config is None:
        config = ProcessingConfig()

    result = ProcessingResult(scope=scope, repository=repository, username=username)

    # Collect all elements and compute element counts per file
    all_elements: list[CodeElement] = []
    element_counts: dict[str, int] = {}
    for pf in parsed_files:
        all_elements.extend(pf.elements)
        element_counts[pf.file_info.relative_path] = len(pf.elements)

    if not all_elements:
        return result

    # Smart delete: only remove stale elements (those no longer in code)
    # For each file, compare existing ES elements with newly parsed elements
    new_element_ids = {e.element_id for e in all_elements}
    stale_element_ids: list[str] = []

    for pf in parsed_files:
        existing_ids = es_repo.get_element_ids_by_file(
            scope, repository, username, pf.file_info.relative_path
        )
        # Stale = in ES but not in new code
        stale_ids = existing_ids - new_element_ids
        stale_element_ids.extend(stale_ids)

    # Delete only stale elements (e.g., a variable that was removed from code)
    if stale_element_ids:
        es_repo.delete_elements(stale_element_ids)
        result.elements_deleted = len(stale_element_ids)

    # Get content hashes for change detection
    # Unchanged elements stay in ES - no need to re-process or re-index
    all_element_ids = list(new_element_ids)
    existing_hashes = es_repo.get_element_content_hashes(all_element_ids)

    # Filter: only process elements that are new or have changed content
    elements_to_process = []
    for elem in all_elements:
        existing_hash = existing_hashes.get(elem.element_id)
        if existing_hash is not None and existing_hash == elem.content_hash:
            # Element exists in ES with same content - skip entirely
            result.elements_skipped += 1
        else:
            # Element is new OR content changed - needs processing
            elements_to_process.append(elem)

    total = len(all_elements)

    if not elements_to_process:
        # All elements unchanged - nothing to do
        return result

    # Summary cache for hierarchical context
    summary_cache = _SummaryCache()

    # Populate cache with all elements (for parent lookup)
    for elem in all_elements:
        summary_cache.add_element(elem)

    # Initialize LLM clients (only if not skipping AI)
    llm_client: SummarizationLLMClient | None = None
    embed_client: CodeEmbeddingClient | None = None

    if not config.skip_ai:
        summarize_cfg = config.summarize_model
        llm_client = SummarizationLLMClient(
            url=summarize_cfg.get_api_base() or "",
            model=summarize_cfg.name,
            provider=summarize_cfg.provider,
            api_key=summarize_cfg.api_key,
        )
        embed_cfg = config.embed_model
        embed_client = CodeEmbeddingClient(
            url=embed_cfg.get_api_base() or "",
            model=embed_cfg.name,
            provider=embed_cfg.provider,
            api_key=embed_cfg.api_key,
        )

    # Pre-compute context sizes for all elements (for tier batching)
    # This enables DependencyTracker to group elements by context tier
    element_context_sizes: dict[str, int] = {}
    for elem in elements_to_process:
        char_count = len(elem.raw_code or "")
        element_context_sizes[elem.element_id] = compute_element_num_ctx(
            elem.element_type, char_count
        )

    # Initialize tracking structures (use provided or create new)
    # Pass per-element context sizes for tier batching (minimizes model reloads)
    dependency_tracker = DependencyTracker(elements_to_process, element_context_sizes)
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
    recent_errors: list[tuple[str, str]] = []  # Track recent errors for display

    # Initialize Redis job tracker if config provided
    redis_tracker: RedisJobTracker | None = None
    if magaldi_config is not None:
        try:
            redis_tracker = RedisJobTracker(magaldi_config, scope, repository, username)
            redis_tracker.clear_queues()  # Clear stale data before adding new jobs
            redis_tracker.add_pending_jobs(elements_to_process)
        except Exception:
            # Redis unavailable - continue without tracking
            redis_tracker = None

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

    def process_wrapper(element: CodeElement) -> ProcessedElement:
        """Wrapper to assign worker ID and call _process_single_element."""
        wid = acquire_worker_id()
        # Mark as running in Redis before processing
        if redis_tracker:
            try:
                redis_tracker.mark_running(element.element_id, should_embed(element))
            except Exception:
                pass
        try:
            return _process_single_element(
                element=element,
                summary_cache=summary_cache,
                llm_client=llm_client,
                embed_client=embed_client,
                config=config,
                file_hashes=file_hashes,
                element_counts=element_counts,
                es_repo=es_repo,
                worker_id=wid,
                worker_status=worker_status,
                on_status_change=on_status_change,
            )
        finally:
            release_worker_id(wid)

    # Process elements in parallel using ThreadPoolExecutor
    # Use max possible workers - tier batching in DependencyTracker limits concurrency
    # dynamically based on context size (smaller context = more workers)
    max_workers = max(DependencyTracker.TIER_MAX_WORKERS.values())
    executor = ThreadPoolExecutor(max_workers=max_workers)
    future_to_element: dict = {}

    try:
        while not dependency_tracker.is_complete():
            # Get elements that are ready (parents completed)
            # DependencyTracker applies tier-specific limits internally
            ready_elements = dependency_tracker.get_ready_elements(
                max_count=max_workers * 2
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
                processed = future.result()

                # Record timing with element type (including dual embedding times)
                timing_stats.record(
                    processed.wall_time,
                    processed.summarize_time,
                    processed.embed_time,
                    element.element_type,
                    was_embedded=should_embed(element),
                    summary_embed_time=processed.summary_embed_time,
                    code_embed_time=processed.code_embed_time,
                )

                was_embedded = should_embed(element)
                if processed.success:
                    dependency_tracker.mark_complete(element.element_id)
                    result.elements_processed += 1
                    result.indexed += 1
                    if not config.skip_ai:
                        result.summarized += 1
                        if was_embedded:
                            result.embedded += 1
                    completed_count += 1
                    # Update Redis job status
                    if redis_tracker:
                        try:
                            redis_tracker.mark_completed(element.element_id, was_embedded)
                        except Exception:
                            pass  # Don't fail processing if Redis update fails
                else:
                    dependency_tracker.mark_failed(element.element_id)
                    result.elements_failed += 1
                    failed_count += 1
                    error_msg = f"Failed to process {element.element_id}: {processed.error}"
                    result.errors.append(error_msg)
                    result.failed_elements.append((element.element_id, processed.error or "Unknown error"))
                    # Add to recent errors for display (keep last 3)
                    short_name = f"{element.element_type}:{element.name}"
                    recent_errors.append((short_name, processed.error or "Unknown error"))
                    if len(recent_errors) > 3:
                        recent_errors.pop(0)
                    # Update Redis job status
                    if redis_tracker:
                        try:
                            redis_tracker.mark_failed(element.element_id, processed.error or "Unknown", was_embedded)
                        except Exception:
                            pass

                # Report progress
                if on_progress:
                    progress_state = ProgressState(
                        total=total,
                        completed=completed_count,
                        skipped=result.elements_skipped,
                        failed=failed_count,
                        timing=timing_stats,
                        workers=worker_status,
                        num_workers=config.num_workers,
                        recent_errors=list(recent_errors),
                    )
                    on_progress(progress_state)

    except KeyboardInterrupt:
        # Graceful shutdown on Ctrl+C - cancel pending futures and stop executor
        for future in future_to_element:
            future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        # Clear worker status display first (so Live display is clean)
        for wid in range(config.num_workers):
            worker_status.clear(wid)
        # Clean up Redis tracker
        if redis_tracker:
            try:
                redis_tracker.close()
            except Exception:
                pass
        # Re-raise so CLI can handle the wait message and exit
        raise
    else:
        # Normal completion - shutdown and wait
        executor.shutdown(wait=True)

    # Clean up Redis tracker
    if redis_tracker:
        try:
            redis_tracker.close()
        except Exception:
            pass

    return result
