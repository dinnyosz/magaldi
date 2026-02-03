"""Unified element processor - atomic summarize -> embed -> index flow.

Processes elements level-by-level:
- Level 0: Files
- Level 1: Classes
- Level 2: Functions/Methods
- Level 3: Variables

Only indexes to ES after full processing, ensuring ES presence = completion.
"""

from __future__ import annotations

import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, TYPE_CHECKING

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
from shared.ai.context_size import compute_element_num_ctx, CONTEXT_TIERS, TIER_TIMEOUTS, TIER_SCALING_EXPONENT
from shared.throttling import ThroughputTracker, compute_throttle_decision, ThrottleDecision


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

    # Parallel processing (0 = use default of 8 workers)
    num_workers: int = 0

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

    # Per-(type, tier) tracking for more accurate ETA
    total_summarize_by_type_tier: dict[tuple[str, int], float] = field(default_factory=dict)
    total_base_by_type_tier: dict[tuple[str, int], float] = field(default_factory=dict)  # base time (throughput) for ETA
    summarize_counts_by_type_tier: dict[tuple[str, int], int] = field(default_factory=dict)
    totals_by_type_tier: dict[tuple[str, int], int] = field(default_factory=dict)

    # Throughput tracker for throttling (5 min window for slow AI tasks)
    throughput_tracker: ThroughputTracker = field(
        default_factory=lambda: ThroughputTracker(window_seconds=300.0)
    )

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

    def set_totals_by_type_tier(self, totals: dict[tuple[str, int], int]) -> None:
        """Set total element counts by (type, tier) for accurate ETA."""
        with self._lock:
            self.totals_by_type_tier = dict(totals)
            # Initialize per-(type, tier) tracking
            for key in totals:
                if key not in self.summarize_counts_by_type_tier:
                    self.summarize_counts_by_type_tier[key] = 0
                if key not in self.total_summarize_by_type_tier:
                    self.total_summarize_by_type_tier[key] = 0.0
                if key not in self.total_base_by_type_tier:
                    self.total_base_by_type_tier[key] = 0.0

    def record(
        self,
        wall_time: float,
        summarize_time: float,
        embed_time: float,
        element_type: str = "",
        was_embedded: bool = True,
        summary_embed_time: float = 0.0,
        code_embed_time: float = 0.0,
        tier: int = 0,
        avg_workers: float = 1.0,
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
            tier: Context tier (2048, 4096, etc) for per-tier ETA tracking.
            avg_workers: Average workers during this task (for throughput calculation).
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

                # Track per-(type, tier) timing for accurate ETA
                if tier > 0:
                    type_tier_key = (element_type, tier)
                    if type_tier_key not in self.total_summarize_by_type_tier:
                        self.total_summarize_by_type_tier[type_tier_key] = 0.0
                    if type_tier_key not in self.total_base_by_type_tier:
                        self.total_base_by_type_tier[type_tier_key] = 0.0
                    self.total_summarize_by_type_tier[type_tier_key] += summarize_time
                    # Track base_time (throughput) = wall_time / workers for ETA
                    base_time = wall_time / max(avg_workers, 1.0)
                    self.total_base_by_type_tier[type_tier_key] += base_time
                    self.summarize_counts_by_type_tier[type_tier_key] = (
                        self.summarize_counts_by_type_tier.get(type_tier_key, 0) + 1
                    )

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
        """Get per-type stats: type -> (completed, total, avg_base, avg_summ, avg_embed)."""
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
                # Calculate avg wall time from type_tier data
                type_wall_total = sum(
                    self.total_base_by_type_tier.get((t, tr), 0.0)
                    for tr in set(tr for (typ, tr) in self.total_base_by_type_tier if typ == t)
                )
                type_count = sum(
                    self.summarize_counts_by_type_tier.get((t, tr), 0)
                    for tr in set(tr for (typ, tr) in self.summarize_counts_by_type_tier if typ == t)
                )
                avg_wall = type_wall_total / type_count if type_count > 0 else 0.0
                result[t] = (completed, total, avg_wall, avg_summ, avg_embed)
            return result

    def record_task_runtime(self, runtime: float, concurrent_workers: int = 1) -> None:
        """Record a completed task's runtime for throttling decisions.

        Args:
            runtime: Task wall-clock runtime in seconds.
            concurrent_workers: Number of workers active when task completed.
        """
        self.throughput_tracker.record_completion(runtime, concurrent_workers)

    def get_throughput_stats(self) -> tuple[float, float, int]:
        """Get throughput statistics for throttling (backwards compatible).

        Returns:
            Tuple of (throughput_per_sec, avg_runtime, completion_count).
        """
        return self.throughput_tracker.get_stats()

    def get_throughput_stats_with_concurrency(self) -> tuple[float, float, int, float, float]:
        """Get throughput statistics with concurrency context.

        Returns:
            Tuple of (throughput, avg_runtime, count, avg_concurrency, avg_base_time).
        """
        return self.throughput_tracker.get_stats_with_concurrency()

    def _get_avg_for_type_tier(
        self,
        element_type: str,
        tier: int,
        global_avg: float,
    ) -> float:
        """Get average processing time for (type, tier) with smart fallback.

        Fallback order:
        1. Exact (type, tier) match
        2. Same type, closest tier with data
        3. Same model group (large/small), closest tier with data
        4. Global average

        Must hold _lock when calling.
        """
        avg, _ = self._get_avg_for_type_tier_with_fallback(element_type, tier, global_avg)
        return avg

    def _get_avg_for_type_tier_with_fallback(
        self,
        element_type: str,
        tier: int,
        global_avg: float,
    ) -> tuple[float, bool]:
        """Get average processing time for (type, tier) with smart fallback.

        Returns:
            Tuple of (avg_time, is_fallback) where is_fallback is True if
            the value was estimated from a different (type, tier).

        Must hold _lock when calling.
        """
        type_tier_key = (element_type, tier)

        # 1. Exact match - use wall_time for accurate throughput-based ETA
        if type_tier_key in self.summarize_counts_by_type_tier:
            count = self.summarize_counts_by_type_tier[type_tier_key]
            if count > 0:
                total_time = self.total_base_by_type_tier.get(type_tier_key, 0.0)
                return total_time / count, False

        # 2. Same type, find closest tier(s) and average
        same_type_tiers = [
            (t, tr) for (t, tr) in self.summarize_counts_by_type_tier
            if t == element_type and self.summarize_counts_by_type_tier[(t, tr)] > 0
        ]
        if same_type_tiers:
            # Find the minimum tier distance
            min_distance = min(abs(tr - tier) for (t, tr) in same_type_tiers)
            # Get all items at that closest distance and average them
            closest_items = [(t, tr) for (t, tr) in same_type_tiers if abs(tr - tier) == min_distance]
            total_time = sum(self.total_base_by_type_tier.get(key, 0.0) for key in closest_items)
            total_count = sum(self.summarize_counts_by_type_tier[key] for key in closest_items)
            base_avg = total_time / total_count
            # Scale by tier ratio (use first closest tier for ratio)
            # Use empirically-derived exponent for sub-linear scaling
            closest_tier = closest_items[0][1]
            tier_ratio = (tier / closest_tier) ** TIER_SCALING_EXPONENT if closest_tier > 0 else 1.0
            return base_avg * tier_ratio, True

        # 3. Same model group (large: file/class/interface/type_alias, small: function/method/variable)
        large_types = {"file", "class", "interface", "type_alias"}
        small_types = {"function", "method", "variable", "constant"}
        if element_type in large_types:
            model_types = large_types
        elif element_type in small_types:
            model_types = small_types
        else:
            model_types = set()

        same_model_tiers = [
            (t, tr) for (t, tr) in self.summarize_counts_by_type_tier
            if t in model_types and self.summarize_counts_by_type_tier[(t, tr)] > 0
        ]
        if same_model_tiers:
            # Find the minimum tier distance
            min_distance = min(abs(tr - tier) for (t, tr) in same_model_tiers)
            # Get all items at that closest distance and average them
            closest_items = [(t, tr) for (t, tr) in same_model_tiers if abs(tr - tier) == min_distance]
            total_time = sum(self.total_base_by_type_tier.get(key, 0.0) for key in closest_items)
            total_count = sum(self.summarize_counts_by_type_tier[key] for key in closest_items)
            base_avg = total_time / total_count
            # Scale by tier ratio (use first closest tier for ratio)
            # Use empirically-derived exponent for sub-linear scaling
            closest_tier = closest_items[0][1]
            tier_ratio = (tier / closest_tier) ** TIER_SCALING_EXPONENT if closest_tier > 0 else 1.0
            return base_avg * tier_ratio, True

        # 4. Fall back to per-type average (ignoring tier) - use wall_time from type_tier
        # Sum all wall_time for this type across all tiers
        type_wall_total = sum(
            self.total_base_by_type_tier.get((element_type, tr), 0.0)
            for tr in set(tr for (t, tr) in self.total_base_by_type_tier if t == element_type)
        )
        type_count = sum(
            self.summarize_counts_by_type_tier.get((element_type, tr), 0)
            for tr in set(tr for (t, tr) in self.summarize_counts_by_type_tier if t == element_type)
        )
        if type_count > 0:
            return type_wall_total / type_count, True

        # 5. Fall back to same model group average (any type, any tier in same group)
        # This is better than global average which mixes large/small models
        model_wall_total = sum(
            self.total_base_by_type_tier.get((t, tr), 0.0)
            for (t, tr) in self.total_base_by_type_tier if t in model_types
        )
        model_count = sum(
            self.summarize_counts_by_type_tier.get((t, tr), 0)
            for (t, tr) in self.total_base_by_type_tier if t in model_types
        )
        if model_count > 0:
            base_avg = model_wall_total / model_count
            # Scale by tier ratio vs average tier in model group
            avg_tier = sum(
                tr * self.summarize_counts_by_type_tier.get((t, tr), 0)
                for (t, tr) in self.summarize_counts_by_type_tier if t in model_types
            ) / model_count if model_count > 0 else tier
            tier_ratio = tier / avg_tier if avg_tier > 0 else 1.0
            return base_avg * tier_ratio, True

        # 6. Cross-model fallback - use OTHER model group with scaling
        # Small model (~1.7B) is roughly 2x faster than large model (~4B) for same context
        other_model_types = small_types if element_type in large_types else large_types
        other_wall_total = sum(
            self.total_base_by_type_tier.get((t, tr), 0.0)
            for (t, tr) in self.total_base_by_type_tier if t in other_model_types
        )
        other_count = sum(
            self.summarize_counts_by_type_tier.get((t, tr), 0)
            for (t, tr) in self.total_base_by_type_tier if t in other_model_types
        )
        if other_count > 0:
            base_avg = other_wall_total / other_count
            # Scale by tier ratio
            avg_tier = sum(
                tr * self.summarize_counts_by_type_tier.get((t, tr), 0)
                for (t, tr) in self.summarize_counts_by_type_tier if t in other_model_types
            ) / other_count if other_count > 0 else tier
            tier_ratio = tier / avg_tier if avg_tier > 0 else 1.0
            # Apply model scaling: small model ~2x faster than large
            model_scale = 0.5 if element_type in small_types else 2.0
            return base_avg * tier_ratio * model_scale, True

        # 7. No data at all - return 0 (display will show "-")
        return 0.0, True

    def eta_seconds(self, completed: int, total: int, num_workers: int = 1) -> float | None:
        """Calculate ETA based on per-(type, tier) API time averages.

        Uses tier-aware averages for more accurate estimates, with smart
        fallback to similar tiers/types when exact data isn't available.

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

            # Global average wall time as ultimate fallback
            total_wall_time = sum(self.total_base_by_type_tier.values())
            total_count = sum(self.summarize_counts_by_type_tier.values())
            global_avg = total_wall_time / total_count if total_count > 0 else 0.0

            total_work_time = 0.0

            # If we have tier-level tracking, use it for more accurate ETA
            if self.totals_by_type_tier:
                for (element_type, tier), tot in self.totals_by_type_tier.items():
                    done = self.summarize_counts_by_type_tier.get((element_type, tier), 0)
                    remaining = tot - done
                    if remaining > 0:
                        avg = self._get_avg_for_type_tier(element_type, tier, global_avg)
                        if avg > 0:
                            total_work_time += remaining * avg
            else:
                # Fall back to type-only tracking (backwards compatibility)
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

            # base_time is already throughput-normalized (wall_time / workers at record time)
            # so total_work_time is already the ETA - no further division needed
            return total_work_time

    def get_eta_breakdown(self, num_workers: int = 1) -> list[tuple[str, int, int, int, float]]:
        """Get ETA breakdown per (type, tier) for display.

        Returns:
            List of (type, tier, remaining, total, eta_seconds) tuples,
            sorted by remaining ETA descending.
        """
        with self._lock:
            if not self.totals_by_type_tier:
                return []

            # Global average wall time as fallback
            total_wall_time = sum(self.total_base_by_type_tier.values())
            total_count = sum(self.summarize_counts_by_type_tier.values())
            global_avg = total_wall_time / total_count if total_count > 0 else 0.0

            breakdown = []
            for (element_type, tier), tot in self.totals_by_type_tier.items():
                done = self.summarize_counts_by_type_tier.get((element_type, tier), 0)
                remaining = tot - done
                if remaining > 0:
                    avg = self._get_avg_for_type_tier(element_type, tier, global_avg)
                    # avg is already throughput-normalized, so work_time IS the ETA
                    eta = remaining * avg if avg > 0 else 0.0
                    breakdown.append((element_type, tier, remaining, tot, eta))

            # Sort by ETA descending (largest remaining time first)
            breakdown.sort(key=lambda x: x[4], reverse=True)
            return breakdown

    def get_eta_breakdown_with_avg(self, num_workers: int = 1) -> list[tuple[str, int, float, bool, int, int]]:
        """Get average time per item for each (type, tier) combination.

        Returns:
            List of (type, tier, avg_seconds, is_fallback, done, total) tuples,
            sorted by hierarchy then tier descending.
            is_fallback is True if the avg was estimated from a different (type, tier).
        """
        with self._lock:
            if not self.totals_by_type_tier:
                return []

            # Global average wall time as fallback
            total_wall_time = sum(self.total_base_by_type_tier.values())
            total_count = sum(self.summarize_counts_by_type_tier.values())
            global_avg = total_wall_time / total_count if total_count > 0 else 0.0

            breakdown = []
            for (element_type, tier), tot in self.totals_by_type_tier.items():
                avg, is_fallback = self._get_avg_for_type_tier_with_fallback(element_type, tier, global_avg)
                done = self.summarize_counts_by_type_tier.get((element_type, tier), 0)
                # Include all items, even those with no timing data yet (avg=0)
                breakdown.append((element_type, tier, avg, is_fallback, done, tot))

            # Sort by hierarchy (file → class/interface/type_alias → function → method → variable), then tier descending
            type_order = {"file": 0, "class": 1, "interface": 1, "type_alias": 1, "function": 2, "method": 3, "variable": 4, "constant": 5}
            breakdown.sort(key=lambda x: (type_order.get(x[0], 99), -x[1]))
            return breakdown


@dataclass
class WorkerStatus:
    """Track what each worker is doing."""

    _lock: threading.Lock = field(default_factory=threading.Lock)
    # worker_id -> (element_name, stage, model, ctx_size, element_start_time)
    # element_start_time is set once when element starts and never reset
    _status: dict[int, tuple[str, str, str, str, float]] = field(default_factory=dict)

    def set(self, worker_id: int, element_name: str, stage: str, model: str = "", ctx_size: str = "", start_time: float = 0.0) -> None:
        """Set worker status. If worker already has an entry, keeps original start_time."""
        with self._lock:
            existing = self._status.get(worker_id)
            if existing is not None:
                # Keep original element start time - don't reset on stage change
                # This is critical for throttling: we want total element time, not stage time
                _, _, _, _, original_start = existing
                self._status[worker_id] = (element_name, stage, model, ctx_size, original_start)
            else:
                self._status[worker_id] = (element_name, stage, model, ctx_size, start_time)

    def clear(self, worker_id: int) -> None:
        with self._lock:
            self._status.pop(worker_id, None)

    def get_all(self) -> dict[int, tuple[str, str, str, str, float]]:
        with self._lock:
            return dict(self._status)

    def get_max_active_runtime(self) -> float:
        """Get the max runtime of currently running workers.

        Returns:
            Maximum runtime in seconds of any active worker, or 0.0 if no workers active.
        """
        now = time.time()
        max_runtime = 0.0
        with self._lock:
            for _, (_, _, _, _, start_time) in self._status.items():
                if start_time > 0:
                    runtime = now - start_time
                    max_runtime = max(max_runtime, runtime)
        return max_runtime

    def get_runtime_stats(self) -> tuple[float, float, int]:
        """Get runtime statistics for active workers.

        Returns:
            Tuple of (max_runtime, avg_runtime, active_count).
        """
        now = time.time()
        runtimes = []
        with self._lock:
            for _, (_, _, _, _, start_time) in self._status.items():
                if start_time > 0:
                    runtimes.append(now - start_time)
        if not runtimes:
            return 0.0, 0.0, 0
        return max(runtimes), sum(runtimes) / len(runtimes), len(runtimes)

    def active_count(self) -> int:
        """Get the number of currently active workers."""
        with self._lock:
            return len(self._status)


@dataclass
class ParallelismStats:
    """Statistics about worker parallelism."""

    max_possible: int  # Total worker pool size
    tier_limit: int    # Current tier's max workers
    running: int       # Workers currently processing
    current_tier: int | None = None  # Current context tier (e.g., 2048, 4096)

    # Throttling info
    throttle_decision: ThrottleDecision | None = None  # Current throttle decision
    tier_changing: bool = False  # True if waiting for tier change

    @property
    def throttled(self) -> int:
        """Workers held back due to tier limits."""
        return self.max_possible - self.tier_limit

    @property
    def idle(self) -> int:
        """Workers within tier limit but waiting for work."""
        return max(0, self.tier_limit - self.running)

    @property
    def effective_limit(self) -> int:
        """Effective worker limit after throttling."""
        if self.throttle_decision and self.throttle_decision.should_throttle:
            return self.throttle_decision.recommended_workers
        return self.tier_limit


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
    parallelism: ParallelismStats | None = None


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

    Batching strategy (model+tier based):
    - Batch by (model, context_tier) to minimize Ollama reloads
    - Priority: same model+tier > same model+different tier > different model
    - Drains current tasks when model changes
    - Warmup limits to 1 task for model/tier transitions
    """

    # Default worker count when not specified by user
    DEFAULT_WORKERS = 8

    def __init__(
        self,
        elements: list[CodeElement],
        context_sizes: dict[str, int] | None = None,
        max_num_workers: int | None = None,
        timeout: float = 180.0,  # Default timeout for throttle calculations
    ) -> None:
        # RLock for reentrant calls (get_parallelism_stats -> get_current_max_workers)
        self._lock = threading.RLock()
        self._elements = {e.element_id: e for e in elements}
        self._completed: set[str] = set()
        self._in_progress: set[str] = set()
        self._context_sizes = context_sizes or {}
        self._current_model: str | None = None  # Current model key ("large" or "small")
        self._current_tier: int | None = None  # Current context tier all workers use
        self._previous_tier: int | None = None  # Previous tier (to detect changes)
        self._tier_changing: bool = False  # True when switching model/tier (warmup period)
        self._warmup_task_id: str | None = None  # ID of warmup task (first task of new tier)
        # Use provided workers or default
        self._max_num_workers = max_num_workers if max_num_workers else self.DEFAULT_WORKERS
        self._timeout = timeout  # Timeout for throttle calculations

        # Build parent lookup: element_id -> parent_element_id
        self._parents: dict[str, str | None] = {}
        for e in elements:
            self._parents[e.element_id] = e.parent_id

    def _get_level(self, element: CodeElement) -> int:
        """Get hierarchy level from element type.

        Level 0: files
        Level 1: classes
        Level 2: functions, methods
        Level 3: variables, constants
        """
        level_map = {
            "file": 0,
            "class": 1,
            "function": 2,
            "method": 2,
            "variable": 3,
            "constant": 3,
        }
        return level_map.get(element.element_type, 2)

    def _get_model_key(self, element: CodeElement) -> str:
        """Get model key for an element (for batching by model).

        Returns 'large' for files/classes/interfaces/type_aliases, 'small' for functions/methods/variables.
        This matches ProcessingConfig.get_model_for_element_type().
        """
        if element.element_type in ("function", "method", "variable", "constant"):
            return "small"
        return "large"

    def _get_tier(self, element_id: str) -> int:
        """Get the context tier for an element (snaps to standard tiers)."""
        ctx = self._context_sizes.get(element_id, 2048)
        # Find the tier this context size belongs to
        for tier in CONTEXT_TIERS:
            if ctx <= tier:
                return tier
        return CONTEXT_TIERS[-1]  # Max tier

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

    def get_ready_elements(
        self,
        max_count: int = 10,
        throttle_limit: int | None = None,
    ) -> list[CodeElement]:
        """Get elements ready for processing (parent done, not started).

        Ordering priority:
        1. By hierarchy level (files → classes → methods) to respect parent deps
        2. Within level, by model+tier to minimize Ollama reloads
        3. Prefer current model+tier, then same model different tier, then switch

        Throttling:
        - Runtime-based throttle_limit reduces workers when approaching timeout
        - Tier warmup limits to 1 task during model/tier transitions

        Args:
            max_count: Maximum elements to return.
            throttle_limit: If set, reduces worker limit for runtime throttling.
        """
        with self._lock:
            ready = self._get_all_ready()
            if not ready:
                return []

            # First, group by hierarchy level to ensure parents processed before children
            by_level: dict[int, list[CodeElement]] = {}
            for elem in ready:
                level = self._get_level(elem)
                by_level.setdefault(level, []).append(elem)

            # Pick the lowest level with ready elements (files before classes before methods)
            target_level = min(by_level.keys())
            level_ready = by_level[target_level]

            # Within the level, group by (model, tier) for optimal batching
            by_model_tier: dict[tuple[str, int], list[CodeElement]] = {}
            for elem in level_ready:
                model_key = self._get_model_key(elem)
                tier = self._get_tier(elem.element_id)
                by_model_tier.setdefault((model_key, tier), []).append(elem)

            # Sort each batch by context size (largest first - finish big ones first)
            for key in by_model_tier:
                by_model_tier[key].sort(
                    key=lambda e: self._context_sizes.get(e.element_id, 0),
                    reverse=True
                )

            # Determine which (model, tier) to use with priority:
            # 1. Current model + current tier (if available at this level)
            # 2. Current model + largest available tier (at this level)
            # 3. Different model + largest available tier (at this level)
            # Rationale: finish big tasks first to avoid long tail at the end
            selected_model = None
            selected_tier = None

            current_key = (self._current_model, self._current_tier)
            if self._current_model is not None and current_key in by_model_tier:
                # Priority 1: same model + same tier
                selected_model = self._current_model
                selected_tier = self._current_tier
            elif self._current_model is not None:
                # Priority 2: same model + different tier (largest first)
                same_model_tiers = [
                    tier for (model, tier) in by_model_tier.keys()
                    if model == self._current_model
                ]
                if same_model_tiers:
                    selected_model = self._current_model
                    selected_tier = max(same_model_tiers)

            if selected_model is None:
                # Priority 3: different model, pick largest tier overall
                # Group by model first, then pick model with largest max tier
                models_with_max_tier = {}
                for (model, tier) in by_model_tier.keys():
                    if model not in models_with_max_tier:
                        models_with_max_tier[model] = tier
                    else:
                        models_with_max_tier[model] = max(models_with_max_tier[model], tier)

                # Pick model with largest maximum tier (start big)
                selected_model = max(models_with_max_tier.keys(), key=lambda m: models_with_max_tier[m])
                selected_tier = models_with_max_tier[selected_model]

            # Check if model or tier is changing
            model_changing = self._current_model is not None and selected_model != self._current_model
            tier_changing = self._current_tier is not None and selected_tier != self._current_tier

            # Drain on model OR tier change: wait for current tasks to finish before switching
            # This ensures proper warmup and ramp-up for each new tier
            if (model_changing or tier_changing) and len(self._in_progress) > 0:
                return []  # Drain: wait for current tasks to finish

            # Set tier_changing flag for warmup (model change or tier change)
            if model_changing or tier_changing or self._current_model is None:
                self._tier_changing = True

            self._current_model = selected_model
            self._previous_tier = self._current_tier
            self._current_tier = selected_tier

            # Worker limit: start with configured max, apply throttling
            worker_limit = self._max_num_workers

            # Apply runtime-based throttle limit if set
            if throttle_limit is not None:
                worker_limit = min(worker_limit, throttle_limit)

            # When tier is changing, wait for old tasks to complete first
            # then start 1 task to load new model before ramping up
            if self._tier_changing:
                # If any tasks still running, wait for them to finish
                if len(self._in_progress) > 0:
                    return []  # Don't start new tier until old tier drains
                worker_limit = 1

            # Ensure at least 1 worker always
            worker_limit = max(1, worker_limit)

            # When throttling, count ALL in-progress to limit total concurrency
            # Otherwise, count only current (model, tier) for normal batched operation
            if throttle_limit is not None:
                total_in_progress = len(self._in_progress)
                slots_available = max(0, worker_limit - total_in_progress)
            else:
                batch_in_progress = sum(
                    1 for eid in self._in_progress
                    if self._get_tier(eid) == selected_tier
                    and self._get_model_key(self._elements[eid]) == selected_model
                )
                slots_available = max(0, worker_limit - batch_in_progress)
            effective_limit = min(max_count, slots_available)

            # During warmup, only return 1 element maximum
            if self._tier_changing:
                effective_limit = 1

            # Get elements from current (model, tier) batch
            batch_key = (selected_model, selected_tier)
            batch_ready = by_model_tier[batch_key][:effective_limit]

            # Mark as in-progress
            for e in batch_ready:
                self._in_progress.add(e.element_id)
                # Track warmup task (first task dispatched when tier_changing)
                if self._tier_changing and self._warmup_task_id is None:
                    self._warmup_task_id = e.element_id

            return batch_ready

    def is_tier_changing(self) -> bool:
        """Check if we're in startup warmup (first task loading model)."""
        with self._lock:
            return self._tier_changing

    def get_current_tier_timeout(self) -> float:
        """Get timeout for the current tier (scales with context size)."""
        with self._lock:
            if self._current_tier and self._current_tier in TIER_TIMEOUTS:
                return float(TIER_TIMEOUTS[self._current_tier])
            return self._timeout  # Fallback to default

    def get_current_model(self) -> str | None:
        """Get the current model key being used ('large' or 'small')."""
        with self._lock:
            return self._current_model

    def mark_complete(self, element_id: str) -> None:
        """Mark element as completed."""
        with self._lock:
            self._in_progress.discard(element_id)
            self._completed.add(element_id)
            # Clear warmup flag only when the warmup task itself completes
            # This ensures the model is fully loaded before ramping up
            if self._tier_changing and element_id == self._warmup_task_id:
                self._tier_changing = False
                self._warmup_task_id = None

    def mark_failed(self, element_id: str) -> None:
        """Mark element as failed (won't block children)."""
        with self._lock:
            self._in_progress.discard(element_id)
            self._completed.add(element_id)  # Treat as done so children can proceed
            # Clear warmup flag if warmup task failed (so processing can continue)
            if self._tier_changing and element_id == self._warmup_task_id:
                self._tier_changing = False
                self._warmup_task_id = None

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
        """Get max workers (for status display)."""
        with self._lock:
            return self._max_num_workers

    def get_parallelism_stats(
        self,
        max_possible: int,
        throttle_decision: ThrottleDecision | None = None,
    ) -> "ParallelismStats":
        """Get current parallelism statistics for display."""
        with self._lock:
            tier_limit = self.get_current_max_workers()
            running = len(self._in_progress)
            return ParallelismStats(
                max_possible=max_possible,
                tier_limit=tier_limit,
                running=running,
                current_tier=self._current_tier,
                throttle_decision=throttle_decision,
                tier_changing=self._tier_changing,
            )

    def compute_throttle_decision(
        self,
        current_max_runtime: float,
        active_workers: int,
        throughput: float = 0.0,
        avg_runtime: float = 0.0,
        completion_count: int = 0,
        avg_concurrency: float = 0.0,
        avg_base_time: float = 0.0,
    ) -> ThrottleDecision:
        """Compute throttle decision based on base_time (normalized by concurrency).

        Key insight: runtime scales linearly with concurrent workers (GPU contention).
        base_time = runtime / workers is the normalized per-worker cost.

        Args:
            current_max_runtime: Max runtime of currently active workers.
            active_workers: Number of currently active workers.
            throughput: Actual completions per second.
            avg_runtime: Average completion time from recent completions.
            completion_count: Number of completions in tracking window.
            avg_concurrency: Average workers active at task start.
            avg_base_time: Average of (runtime/workers) from completions.

        Returns:
            ThrottleDecision with recommended action.
        """
        with self._lock:
            return compute_throttle_decision(
                current_max_runtime=current_max_runtime,
                tier_timeout=self.get_current_tier_timeout(),
                base_workers=self._max_num_workers,
                active_workers=active_workers,
                throughput=throughput,
                avg_runtime=avg_runtime,
                completion_count=completion_count,
                avg_concurrency=avg_concurrency,
                avg_base_time=avg_base_time,
            )

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
    # All code elements get embedded (imports are tracked but not embedded)
    if element.element_type in (
        "file", "class", "interface", "type_alias", "trait", "enum",
        "function", "method", "constant", "variable"
    ):
        return True

    return False


# Element types that should NOT go through AI processing (summarization, embedding)
# These are tracked/stored but don't need AI-generated summaries
_NON_AI_ELEMENT_TYPES = frozenset({"import"})


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
            {
                "name": call.name,
                "receiver": call.receiver,
                "line": call.line,
                "resolved_id": call.resolved_id,
                "category": call.category,
            }
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
    def build_display_name(max_path_len: int = 60) -> str:
        parts = []
        # Add path (truncated from left if too long)
        path = element.relative_path
        if len(path) > max_path_len:
            path = "..." + path[-(max_path_len - 3):]
        if element.element_type == "file":
            # For file elements, show the path as the name
            parts.append(path)
        else:
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
                # Skip embeddings entirely - don't generate zero vectors
                # (ES rejects dense_vectors with zero magnitude)
                update_status("summ_embed", config.embed_model.name, "-")
                update_status("code_embed", config.embed_model.name, "-")
                # Leave embeddings as None
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

    # Deduplicate parsed_files by path (keep first occurrence)
    seen_paths: set[str] = set()
    unique_parsed_files: list[ParsedFile] = []
    for pf in parsed_files:
        if pf.file_info.relative_path not in seen_paths:
            seen_paths.add(pf.file_info.relative_path)
            unique_parsed_files.append(pf)

    parsed_files = unique_parsed_files

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

    # Get content hashes and summary state for change detection
    # Only skip if content unchanged AND summary exists (handles interrupted runs)
    all_element_ids = list(new_element_ids)
    existing_states = es_repo.get_element_processing_state(all_element_ids)

    # RELOCATED ELEMENT MATCHING (must happen BEFORE deleting stale elements!)
    # When an element's line number changes, its ID changes but content_hash stays same.
    # We find these "relocated" elements by matching content_hash from stale → new elements.
    # IMPORTANT: Search per-file to avoid cross-file false matches (same code in different files).
    elements_not_found_by_id = [
        elem for elem in all_elements
        if elem.element_id not in existing_states and elem.content_hash
    ]
    relocated_states: dict[str, dict] = {}  # content_hash -> state (includes actual data)
    relocated_old_ids: set[str] = set()  # old element IDs that were relocated (don't delete)
    if elements_not_found_by_id:
        # Group by file path for per-file searching
        by_file: dict[str, list[str]] = {}
        for elem in elements_not_found_by_id:
            if elem.content_hash:
                by_file.setdefault(elem.relative_path, []).append(elem.content_hash)

        # Search each file separately to avoid cross-file false matches
        for rel_path, hashes in by_file.items():
            unique_hashes = list(set(hashes))
            file_relocated = es_repo.find_elements_by_content_hash(
                scope, repository, username, unique_hashes, relative_path=rel_path
            )
            relocated_states.update(file_relocated)
            # Track old element IDs that matched - these are relocated, not deleted
            for state in file_relocated.values():
                old_id = state.get("old_element_id")
                if old_id:
                    relocated_old_ids.add(old_id)

    # Now delete stale elements, EXCLUDING relocated ones (they'll be updated, not deleted)
    truly_stale_ids = [eid for eid in stale_element_ids if eid not in relocated_old_ids]
    if truly_stale_ids:
        es_repo.delete_elements(truly_stale_ids)
        result.elements_deleted = len(truly_stale_ids)

    # Filter: only process elements that are new, changed, or missing summary
    # Also track which files had skipped elements (need file_hash update)
    elements_to_process = []
    skipped_by_file: dict[str, int] = {}  # relative_path -> count of skipped elements
    skipped_with_summary = 0
    skipped_no_summary = 0
    new_elements = 0
    state_none_count = 0
    state_found_count = 0
    relocated_copied = 0
    for elem in all_elements:
        # Skip import elements - they're stored but don't need AI processing
        if elem.element_type in _NON_AI_ELEMENT_TYPES:
            continue

        state = existing_states.get(elem.element_id)
        is_relocated = False

        # Fallback: check if element was found by content_hash (relocated element)
        if state is None and elem.content_hash:
            relocated_data = relocated_states.get(elem.content_hash)
            if relocated_data is not None:
                is_relocated = True
                state = relocated_data

        if state is not None:
            state_found_count += 1
            content_unchanged = state.get("content_hash") == elem.content_hash
            has_summary = state.get("has_summary", False)
            has_summary_embedding = state.get("has_summary_embedding", False)
            has_code_embedding = state.get("has_code_embedding", False)

            # Check if element is fully processed
            # For embeddable elements, require both embeddings
            is_embeddable = should_embed(elem)
            is_fully_processed = has_summary and (
                not is_embeddable or (has_summary_embedding and has_code_embedding)
            )

            if content_unchanged and is_fully_processed:
                if is_relocated:
                    # Relocated element: copy data to new element with new ID
                    # This preserves the summary while updating line numbers
                    file_hash = file_hashes.get(elem.relative_path) if file_hashes else None
                    element_count = None
                    if elem.element_type == "file" and element_counts:
                        element_count = element_counts.get(elem.relative_path)
                    _index_element(
                        elem,
                        state.get("summary"),
                        state.get("summary_embedding"),
                        state.get("code_embedding"),
                        es_repo,
                        file_hash,
                        element_count,
                    )
                    # Delete the old element now that we've indexed the new one
                    old_element_id = state.get("old_element_id")
                    if old_element_id:
                        es_repo.delete_elements([old_element_id])
                    relocated_copied += 1
                    result.elements_skipped += 1
                    skipped_by_file[elem.relative_path] = skipped_by_file.get(elem.relative_path, 0) + 1
                    skipped_with_summary += 1
                    continue
                else:
                    # Element exists with same ID and content - skip entirely
                    result.elements_skipped += 1
                    skipped_by_file[elem.relative_path] = skipped_by_file.get(elem.relative_path, 0) + 1
                    skipped_with_summary += 1
                    continue
            elif content_unchanged and not is_fully_processed:
                skipped_no_summary += 1
        else:
            state_none_count += 1
            new_elements += 1
        # Element is new OR content changed OR missing summary/embeddings - needs processing
        elements_to_process.append(elem)

    total = len(all_elements)

    # Count elements per file to identify files where ALL elements were skipped
    elements_per_file: dict[str, int] = {}
    for elem in all_elements:
        elements_per_file[elem.relative_path] = elements_per_file.get(elem.relative_path, 0) + 1

    # Find files where ALL elements were skipped and update their file_hash in ES
    # This prevents files from appearing "modified" on every run when only
    # file metadata changed but element content stayed the same
    files_to_update: dict[str, str] = {}
    if file_hashes and skipped_by_file:
        for rel_path, skipped_count in skipped_by_file.items():
            if skipped_count == elements_per_file.get(rel_path, 0):
                # All elements in this file were skipped - update file_hash
                if rel_path in file_hashes:
                    files_to_update[rel_path] = file_hashes[rel_path]

        if files_to_update:
            # Pass element_counts so FILE elements also get updated element_count
            es_repo.update_file_hashes(
                scope, repository, username, files_to_update, elements_per_file
            )

    if not elements_to_process:
        # All elements unchanged - fire progress callback showing 100% complete
        if on_progress:
            # Show as complete with all skipped
            if timing_stats is None:
                timing_stats = TimingStats()
            if worker_status is None:
                worker_status = WorkerStatus()
            progress_state = ProgressState(
                total=total,
                completed=total,  # All done (skipped counts as done)
                skipped=result.elements_skipped,
                failed=0,
                timing=timing_stats,
                workers=worker_status,
                num_workers=config.num_workers if config.num_workers > 0 else 8,
                recent_errors=[],
                parallelism=None,
            )
            on_progress(progress_state)
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
    # Pass num_workers as upper limit (0 or None = use tier defaults)
    max_num_workers = config.num_workers if config.num_workers > 0 else None
    dependency_tracker = DependencyTracker(
        elements_to_process,
        element_context_sizes,
        max_num_workers=max_num_workers,
        timeout=config.summarize_timeout,  # For throttle calculations
    )
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

    # Count elements by (type, tier) for tier-aware ETA
    totals_by_type_tier: dict[tuple[str, int], int] = {}
    for elem in elements_to_process:
        ctx_size = element_context_sizes.get(elem.element_id, 2048)
        # Snap to standard tier
        tier = 2048
        for t in CONTEXT_TIERS:
            if ctx_size <= t:
                tier = t
                break
        else:
            tier = CONTEXT_TIERS[-1]
        key = (elem.element_type, tier)
        totals_by_type_tier[key] = totals_by_type_tier.get(key, 0) + 1
    timing_stats.set_totals_by_type_tier(totals_by_type_tier)

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

    # Worker ID pool - use configured num_workers or default
    max_workers = config.num_workers if config.num_workers > 0 else DependencyTracker.DEFAULT_WORKERS
    available_worker_ids: list[int] = list(range(max_workers))
    worker_id_lock = threading.Lock()

    def acquire_worker_id() -> int:
        """Get an available worker ID from the pool."""
        with worker_id_lock:
            if available_worker_ids:
                return available_worker_ids.pop(0)
            return 0  # Fallback

    def release_worker_id(wid: int) -> None:
        """Return a worker ID to the pool (sorted to prefer low IDs)."""
        import bisect
        with worker_id_lock:
            if wid not in available_worker_ids:
                bisect.insort(available_worker_ids, wid)

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
    # DependencyTracker manages concurrency with tier batching and runtime-based throttling
    executor = ThreadPoolExecutor(max_workers=max_workers)
    future_to_element: dict = {}

    # Track allowed workers at task start for averaging with end value
    future_to_allowed_at_start: dict[Any, int] = {}

    # Track current throttle decision for display
    current_throttle: ThrottleDecision | None = None
    # Interval for wait() timeout - allows periodic display updates
    HISTORY_RECORD_INTERVAL = 2.0  # Refresh display every 2 seconds

    try:
        while not dependency_tracker.is_complete():
            # Get current max from active workers (for emergency throttling)
            current_max_runtime = worker_status.get_max_active_runtime()

            # Get current time and active worker count for throttle decisions
            now = time.time()
            active_workers = worker_status.active_count()

            # Get throughput-based stats for adaptive throttling (with concurrency context)
            throughput, avg_runtime, completion_count, avg_concurrency, high_load_avg = (
                timing_stats.get_throughput_stats_with_concurrency()
            )
            current_throttle = dependency_tracker.compute_throttle_decision(
                current_max_runtime, active_workers, throughput, avg_runtime, completion_count,
                avg_concurrency, high_load_avg
            )

            # Always use recommended_workers as the limit (includes ramp-up logic)
            # Even when not throttling, we ramp up gradually to avoid overwhelming
            throttle_limit = current_throttle.recommended_workers

            # Get elements that are ready (parents completed)
            # DependencyTracker applies tier-specific limits internally
            # Pass throttle_limit to further reduce concurrency if needed
            ready_elements = dependency_tracker.get_ready_elements(
                max_count=max_workers * 2,
                throttle_limit=throttle_limit,
            )

            # Submit new tasks for ready elements
            # Track allowed workers at start for throughput calculation
            for element in ready_elements:
                future = executor.submit(process_wrapper, element)
                future_to_element[future] = element
                future_to_allowed_at_start[future] = throttle_limit

            if not future_to_element:
                # No futures pending and not complete - shouldn't happen
                # Store diagnostic info in result for the CLI to display
                result.errors.append(
                    f"Processing stalled: no ready elements. "
                    f"pending={dependency_tracker.pending_count()}, "
                    f"tier_changing={dependency_tracker.is_tier_changing()}"
                )
                break

            # Wait for at least one to complete, or timeout for periodic updates
            done, _ = wait(future_to_element.keys(), timeout=HISTORY_RECORD_INTERVAL, return_when=FIRST_COMPLETED)

            # If timeout with no completions, just update display
            # NOTE: We no longer record to history here because fresh_active
            # (current count) gives wrong base_time for tasks that started earlier.
            if not done:
                fresh_current_max = worker_status.get_max_active_runtime()
                fresh_active = worker_status.active_count()

                if on_progress:
                    fresh_throughput, fresh_avg, fresh_count, fresh_avg_conc, fresh_high_load = (
                        timing_stats.get_throughput_stats_with_concurrency()
                    )
                    fresh_throttle = dependency_tracker.compute_throttle_decision(
                        fresh_current_max, fresh_active, fresh_throughput, fresh_avg, fresh_count,
                        fresh_avg_conc, fresh_high_load
                    )
                    progress_state = ProgressState(
                        total=total,
                        completed=completed_count,
                        skipped=result.elements_skipped,
                        failed=failed_count,
                        timing=timing_stats,
                        workers=worker_status,
                        num_workers=max_workers,
                        recent_errors=list(recent_errors),
                        parallelism=dependency_tracker.get_parallelism_stats(
                            max_workers, fresh_throttle
                        ),
                    )
                    on_progress(progress_state)
                continue

            for future in done:
                element = future_to_element.pop(future)
                processed = future.result()

                # Get element's tier for accurate ETA tracking
                element_tier = dependency_tracker._get_tier(element.element_id)

                # Use allowed workers for throughput calculation (average of start and end)
                # This is more stable than actual counts which fluctuate during ramp-up
                allowed_at_start = future_to_allowed_at_start.pop(future, throttle_limit)
                allowed_at_end = throttle_limit
                avg_workers = min(allowed_at_start, allowed_at_end)

                # Record timing with element type, tier, and avg_workers (for throughput)
                timing_stats.record(
                    processed.wall_time,
                    processed.summarize_time,
                    processed.embed_time,
                    element.element_type,
                    was_embedded=should_embed(element),
                    summary_embed_time=processed.summary_embed_time,
                    code_embed_time=processed.code_embed_time,
                    tier=element_tier,
                    avg_workers=avg_workers,
                )
                timing_stats.record_task_runtime(processed.wall_time, avg_workers)

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

                # Report progress with throttle info
                if on_progress:
                    # Refresh throttle decision with current throughput values for accurate display
                    fresh_current_max = worker_status.get_max_active_runtime()
                    fresh_throughput, fresh_avg, fresh_count, fresh_avg_conc, fresh_high_load = (
                        timing_stats.get_throughput_stats_with_concurrency()
                    )
                    fresh_active = worker_status.active_count()
                    fresh_throttle = dependency_tracker.compute_throttle_decision(
                        fresh_current_max, fresh_active, fresh_throughput, fresh_avg, fresh_count,
                        fresh_avg_conc, fresh_high_load
                    )
                    progress_state = ProgressState(
                        total=total,
                        completed=completed_count,
                        skipped=result.elements_skipped,
                        failed=failed_count,
                        timing=timing_stats,
                        workers=worker_status,
                        num_workers=max_workers,
                        recent_errors=list(recent_errors),
                        parallelism=dependency_tracker.get_parallelism_stats(
                            max_workers, fresh_throttle
                        ),
                    )
                    on_progress(progress_state)

    except KeyboardInterrupt:
        # Graceful shutdown on Ctrl+C - cancel pending futures and stop executor
        for future in future_to_element:
            future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        # Clear worker status display first (so Live display is clean)
        for wid in range(max_workers):
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
