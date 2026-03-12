"""Timing statistics for element processing.

Thread-safe timing tracking with per-type and per-tier ETA estimation.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

from shared.ai.context_size import TIER_SCALING_EXPONENT, is_handcrafted_tier
from shared.ai.prompts import OUTPUT_TOKEN_BUDGETS
from shared.throttling import ThroughputTracker

# Types that always use the small model (regardless of tier)
_ALWAYS_SMALL_TYPES = frozenset({"function", "method", "variable", "constant"})

# Tier threshold at or below which all types use the small model
_SMALL_MODEL_TIER_THRESHOLD = 1024

# Default ETA for handcrafted elements (embed + index only, no LLM)
_HANDCRAFTED_DEFAULT_ETA = 0.1


def _uses_small_model(element_type: str, tier: int) -> bool:
    """Determine if a (type, tier) combo uses the small model.

    Must mirror ProcessingConfig.get_model_for_element_type() logic.
    Handcrafted tier (no LLM) is treated as "small" for grouping purposes.
    """
    if is_handcrafted_tier(tier):
        return True  # No model needed, group with small for fallback
    if element_type in _ALWAYS_SMALL_TYPES:
        return True
    return tier <= _SMALL_MODEL_TIER_THRESHOLD


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

    # Tier accuracy tracking: per-(type, tier) input overflow detection
    tier_overflow_counts: dict[tuple[str, int], int] = field(default_factory=dict)
    tier_headroom_sum: dict[tuple[str, int], float] = field(default_factory=dict)
    tier_headroom_min: dict[tuple[str, int], float] = field(default_factory=dict)
    tier_sample_counts: dict[tuple[str, int], int] = field(default_factory=dict)

    # Output token tracking: per-type response token stats
    output_tokens_sum: dict[str, int] = field(default_factory=dict)
    output_tokens_max: dict[str, int] = field(default_factory=dict)
    output_sample_counts: dict[str, int] = field(default_factory=dict)

    # Input token tracking: per-type prompt token stats (total input to LLM)
    input_tokens_sum: dict[str, int] = field(default_factory=dict)
    input_sample_counts: dict[str, int] = field(default_factory=dict)

    # Per-model token tracking: model_name -> {input_sum, output_sum, count}
    model_input_tokens: dict[str, int] = field(default_factory=dict)
    model_output_tokens: dict[str, int] = field(default_factory=dict)
    model_sample_counts: dict[str, int] = field(default_factory=dict)

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
        prompt_tokens: int = 0,
        response_tokens: int = 0,
        assigned_tier: int = 0,
        model_name: str = "",
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
            prompt_tokens: Estimated tokens in the full prompt.
            response_tokens: Estimated tokens in the LLM response.
            assigned_tier: Context tier assigned to this element.
            model_name: Display name of the model used for summarization.
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
                # tier >= 0 to include HANDCRAFTED_TIER (0); tier < 0 means "not set"
                if tier >= 0:
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

                # Track tier accuracy metrics (input overflow / headroom)
                if assigned_tier > 0 and prompt_tokens > 0 and element_type:
                    tt_key = (element_type, assigned_tier)
                    headroom = 1.0 - (prompt_tokens / assigned_tier)
                    self.tier_sample_counts[tt_key] = self.tier_sample_counts.get(tt_key, 0) + 1
                    self.tier_headroom_sum[tt_key] = self.tier_headroom_sum.get(tt_key, 0.0) + headroom
                    if tt_key not in self.tier_headroom_min or headroom < self.tier_headroom_min[tt_key]:
                        self.tier_headroom_min[tt_key] = headroom
                    if prompt_tokens > assigned_tier:
                        self.tier_overflow_counts[tt_key] = self.tier_overflow_counts.get(tt_key, 0) + 1

                # Track output token usage per type
                if response_tokens > 0 and element_type:
                    self.output_sample_counts[element_type] = self.output_sample_counts.get(element_type, 0) + 1
                    self.output_tokens_sum[element_type] = self.output_tokens_sum.get(element_type, 0) + response_tokens
                    if element_type not in self.output_tokens_max or response_tokens > self.output_tokens_max[element_type]:
                        self.output_tokens_max[element_type] = response_tokens

                # Track input token usage per type
                if prompt_tokens > 0 and element_type:
                    self.input_sample_counts[element_type] = self.input_sample_counts.get(element_type, 0) + 1
                    self.input_tokens_sum[element_type] = self.input_tokens_sum.get(element_type, 0) + prompt_tokens

                # Track per-model token usage (only when model actually used)
                if model_name and (prompt_tokens > 0 or response_tokens > 0):
                    self.model_input_tokens[model_name] = self.model_input_tokens.get(model_name, 0) + prompt_tokens
                    self.model_output_tokens[model_name] = self.model_output_tokens.get(model_name, 0) + response_tokens
                    self.model_sample_counts[model_name] = self.model_sample_counts.get(model_name, 0) + 1

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
                    for tr in {tr for (typ, tr) in self.total_base_by_type_tier if typ == t}
                )
                type_count = sum(
                    self.summarize_counts_by_type_tier.get((t, tr), 0)
                    for tr in {tr for (typ, tr) in self.summarize_counts_by_type_tier if typ == t}
                )
                avg_wall = type_wall_total / type_count if type_count > 0 else 0.0
                result[t] = (completed, total, avg_wall, avg_summ, avg_embed)
            return result

    def get_tier_accuracy_summary(self) -> dict[str, list | dict]:
        """Get tier accuracy metrics for end-of-phase display.

        Returns:
            Dict with keys:
                "input": list of (type, tier, count, overflows, avg_headroom_pct, worst_headroom_pct)
                "output": list of (type, avg_tokens, max_tokens, budget)
                "has_issues": True if any overflows or tight headroom detected
        """

        with self._lock:
            # Input overflow summary
            input_rows: list[tuple[str, int, int, int, float, float]] = []
            has_issues = False

            for (elem_type, tier), count in sorted(self.tier_sample_counts.items()):
                overflows = self.tier_overflow_counts.get((elem_type, tier), 0)
                avg_headroom = self.tier_headroom_sum.get((elem_type, tier), 0.0) / count
                worst_headroom = self.tier_headroom_min.get((elem_type, tier), 1.0)
                avg_pct = avg_headroom * 100
                worst_pct = worst_headroom * 100

                if overflows > 0 or worst_pct < 10:
                    has_issues = True
                    input_rows.append((elem_type, tier, count, overflows, avg_pct, worst_pct))

            # Output token summary — budgets from OUTPUT_TOKEN_BUDGETS (shared.ai.prompts)
            output_rows: list[tuple[str, int, int, int]] = []
            for elem_type in sorted(self.output_sample_counts.keys()):
                count = self.output_sample_counts[elem_type]
                avg_tokens = self.output_tokens_sum[elem_type] // count
                max_tokens = self.output_tokens_max[elem_type]
                budget = OUTPUT_TOKEN_BUDGETS.get(elem_type, 200)
                if max_tokens > budget:
                    has_issues = True
                    output_rows.append((elem_type, avg_tokens, max_tokens, budget))

            return {
                "input": input_rows,
                "output": output_rows,
                "has_issues": has_issues,  # type: ignore[dict-item]
            }

    def get_token_usage_summary(self) -> dict[str, Any]:
        """Get per-type and per-model token usage (input + output) for display.

        Returns:
            Dict with keys:
                "by_type": {type: {"input": total_input, "output": total_output, "count": count}}
                "by_model": {model: {"input": total_input, "output": total_output, "count": count}}
                "totals": {"input": total_input, "output": total_output, "count": count}
        """
        with self._lock:
            by_type: dict[str, dict[str, int]] = {}

            # Merge input and output token data by type
            all_types = set(self.input_sample_counts.keys()) | set(self.output_sample_counts.keys())
            for elem_type in sorted(all_types):
                input_tokens = self.input_tokens_sum.get(elem_type, 0)
                output_tokens = self.output_tokens_sum.get(elem_type, 0)
                count = max(
                    self.input_sample_counts.get(elem_type, 0),
                    self.output_sample_counts.get(elem_type, 0),
                )
                by_type[elem_type] = {
                    "input": input_tokens,
                    "output": output_tokens,
                    "count": count,
                }

            # Per-model token usage
            by_model: dict[str, dict[str, int]] = {}
            for model in sorted(self.model_sample_counts.keys()):
                by_model[model] = {
                    "input": self.model_input_tokens.get(model, 0),
                    "output": self.model_output_tokens.get(model, 0),
                    "count": self.model_sample_counts.get(model, 0),
                }

            total_input = sum(self.input_tokens_sum.values())
            total_output = sum(self.output_tokens_sum.values())
            total_count = sum(
                max(self.input_sample_counts.get(t, 0), self.output_sample_counts.get(t, 0))
                for t in all_types
            )

            return {
                "by_type": by_type,
                "by_model": by_model,
                "totals": {
                    "input": total_input,
                    "output": total_output,
                    "count": total_count,
                },
            }

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
        return self.throughput_tracker.get_stats()  # type: ignore[no-any-return]

    def get_throughput_stats_with_concurrency(self) -> tuple[float, float, int, float, float]:
        """Get throughput statistics with concurrency context.

        Returns:
            Tuple of (throughput, avg_runtime, count, avg_concurrency, avg_base_time).
        """
        return self.throughput_tracker.get_stats_with_concurrency()  # type: ignore[no-any-return]

    def get_peak_concurrency(self) -> int | None:
        """Get concurrency level with peak throughput, or None if insufficient data."""
        return self.throughput_tracker.get_peak_concurrency()  # type: ignore[no-any-return]

    def get_all_throughput_levels(self) -> dict[int, tuple[float, int]]:
        """Get throughput data for all concurrency levels.

        Returns:
            Dict of level -> (throughput_per_sec, sample_count).
        """
        return self.throughput_tracker._throughput_by_level.get_all_levels()  # type: ignore[no-any-return]

    def get_exploration_target(self, max_level: int, remaining: int | None = None) -> int | None:
        """Get a neighbor of the peak that needs more data.

        Args:
            max_level: Upper bound for exploration (typically base_workers).
            remaining: Number of elements left to process, or None if unknown.
                When provided, skips exploration if the budget is too small.

        Returns:
            Level to explore, or None if no exploration needed.
        """
        return self.throughput_tracker.get_exploration_target(max_level, remaining)  # type: ignore[no-any-return]

    def reset_throughput(self) -> None:
        """Reset throughput tracker history.

        Call this on tier/model changes since historical data from a different
        tier isn't relevant for throttling decisions.
        """
        self.throughput_tracker.reset()

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
        _global_avg: float,
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

        # 1b. Handcrafted tier: use seeded default when no data yet.
        # Handcrafted elements skip LLM and only do embed+index (~0.1s).
        # Don't fall through to LLM-based tier fallbacks which would overestimate.
        if is_handcrafted_tier(tier):
            return _HANDCRAFTED_DEFAULT_ETA, True

        # 2. Same type, closest tier — but only tiers using the same model.
        # file@1024 (small model) must NOT extrapolate from file@16384 (large model),
        # because the ~2x model speed difference would make the ETA way off.
        is_small = _uses_small_model(element_type, tier)
        same_type_same_model = [
            (t, tr) for (t, tr) in self.summarize_counts_by_type_tier
            if t == element_type
            and _uses_small_model(t, tr) == is_small
            and self.summarize_counts_by_type_tier[(t, tr)] > 0
        ]
        if same_type_same_model:
            # Find the minimum tier distance
            min_distance = min(abs(tr - tier) for (t, tr) in same_type_same_model)
            # Get all items at that closest distance and average them
            closest_items = [(t, tr) for (t, tr) in same_type_same_model if abs(tr - tier) == min_distance]
            total_time = sum(self.total_base_by_type_tier.get(key, 0.0) for key in closest_items)
            total_count = sum(self.summarize_counts_by_type_tier[key] for key in closest_items)
            base_avg = total_time / total_count
            # Scale by tier ratio (use first closest tier for ratio)
            # Use empirically-derived exponent for sub-linear scaling
            closest_tier = closest_items[0][1]
            tier_ratio = (tier / closest_tier) ** TIER_SCALING_EXPONENT if closest_tier > 0 else 1.0
            return base_avg * tier_ratio, True

        # 3. Same model group — match by actual model used, not just type
        # A file@1024 uses the small model, so it should look at other small-model
        # (type, tier) combos for fallback, not at class@4096 which uses the large model.

        same_model_tiers = [
            (t, tr) for (t, tr) in self.summarize_counts_by_type_tier
            if _uses_small_model(t, tr) == is_small
            and self.summarize_counts_by_type_tier[(t, tr)] > 0
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

        # 4. Fall back to per-type average — only same-model tiers
        same_type_any_tier = [
            (t, tr) for (t, tr) in self.total_base_by_type_tier
            if t == element_type and _uses_small_model(t, tr) == is_small
        ]
        type_wall_total = sum(
            self.total_base_by_type_tier.get(key, 0.0) for key in same_type_any_tier
        )
        type_count = sum(
            self.summarize_counts_by_type_tier.get(key, 0) for key in same_type_any_tier
        )
        if type_count > 0:
            return type_wall_total / type_count, True

        # 5. Fall back to same model group average (any type, any tier in same group)
        # This is better than global average which mixes large/small models
        same_model_all = [
            (t, tr) for (t, tr) in self.total_base_by_type_tier
            if _uses_small_model(t, tr) == is_small
        ]
        model_wall_total = sum(
            self.total_base_by_type_tier.get(key, 0.0) for key in same_model_all
        )
        model_count = sum(
            self.summarize_counts_by_type_tier.get(key, 0) for key in same_model_all
        )
        if model_count > 0:
            base_avg = model_wall_total / model_count
            # Scale by tier ratio vs average tier in model group
            avg_tier = sum(
                tr * self.summarize_counts_by_type_tier.get((t, tr), 0)
                for (t, tr) in same_model_all
            ) / model_count if model_count > 0 else tier
            tier_ratio = tier / avg_tier if avg_tier > 0 else 1.0
            return base_avg * tier_ratio, True

        # 6. Cross-model fallback - use OTHER model group with scaling
        # Small model (~1.7B) is roughly 2x faster than large model (~4B) for same context
        other_model_all = [
            (t, tr) for (t, tr) in self.total_base_by_type_tier
            if _uses_small_model(t, tr) != is_small
        ]
        other_wall_total = sum(
            self.total_base_by_type_tier.get(key, 0.0) for key in other_model_all
        )
        other_count = sum(
            self.summarize_counts_by_type_tier.get(key, 0) for key in other_model_all
        )
        if other_count > 0:
            base_avg = other_wall_total / other_count
            # Scale by tier ratio
            avg_tier = sum(
                tr * self.summarize_counts_by_type_tier.get((t, tr), 0)
                for (t, tr) in other_model_all
            ) / other_count if other_count > 0 else tier
            tier_ratio = tier / avg_tier if avg_tier > 0 else 1.0
            # Apply model scaling: small model ~2x faster than large
            model_scale = 0.5 if is_small else 2.0
            return base_avg * tier_ratio * model_scale, True

        # 7. No data at all - return 0 (display will show "-")
        return 0.0, True

    def eta_seconds(self, completed: int, _total: int, _num_workers: int = 1) -> float | None:
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

            # Elapsed-rate fallback: when items are near-instant (skip_ai),
            # base_time ≈ 0 but wall-clock throughput reflects real overhead
            # (indexing I/O, thread scheduling, futures management).
            # Use max(tier_eta, elapsed_eta) so the ETA never underestimates.
            elapsed_eta = 0.0
            if self.phase_start > 0:
                total_done = sum(self.summarize_counts_by_type_tier.values())
                total_remaining = (
                    sum(self.totals_by_type_tier.values()) - total_done
                )
                elapsed = self.elapsed
                if total_done > 0 and total_remaining > 0 and elapsed > 0:
                    elapsed_eta = (elapsed / total_done) * total_remaining

            if total_work_time <= 0 and elapsed_eta <= 0:
                return None

            # Use whichever is larger: tier-based or elapsed-rate.
            # For LLM workloads, tier-based dominates (correct).
            # For skip_ai, elapsed-rate dominates (correct).
            return max(total_work_time, elapsed_eta)

    def get_eta_breakdown(self, _num_workers: int = 1) -> list[tuple[str, int, int, int, float]]:
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

            # Elapsed-rate fallback: global wall-clock rate when tier avg ≈ 0
            # (skip_ai: base_time ≈ 0 but real throughput reflects I/O overhead)
            elapsed_rate = 0.0
            if self.phase_start > 0:
                elapsed = self.elapsed
                elapsed_rate = elapsed / total_count if total_count > 0 and elapsed > 0 else 0.0

            breakdown = []
            for (element_type, tier), tot in self.totals_by_type_tier.items():
                done = self.summarize_counts_by_type_tier.get((element_type, tier), 0)
                remaining = tot - done
                if remaining > 0:
                    avg = self._get_avg_for_type_tier(element_type, tier, global_avg)
                    # Use max(tier_avg, elapsed_rate) so ETA never underestimates
                    avg = max(avg, elapsed_rate)
                    eta = remaining * avg if avg > 0 else 0.0
                    breakdown.append((element_type, tier, remaining, tot, eta))

            # Sort by ETA descending (largest remaining time first)
            breakdown.sort(key=lambda x: x[4], reverse=True)
            return breakdown

    def get_eta_breakdown_with_avg(self, _num_workers: int = 1) -> list[tuple[str, int, float, bool, int, int]]:
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

            # Elapsed-rate fallback: global wall-clock rate when tier avg ≈ 0
            elapsed_rate = 0.0
            if self.phase_start > 0:
                elapsed = self.elapsed
                elapsed_rate = elapsed / total_count if total_count > 0 and elapsed > 0 else 0.0

            breakdown = []
            for (element_type, tier), tot in self.totals_by_type_tier.items():
                avg, is_fallback = self._get_avg_for_type_tier_with_fallback(element_type, tier, global_avg)
                # Use max(tier_avg, elapsed_rate) so ETA never underestimates.
                # Only mark as fallback when the original was already a fallback or
                # when base_time is near-zero (skip_ai: no LLM, wall_time ≈ 0).
                # Don't mark real LLM measurements as fallback just because
                # elapsed_rate > base_time (common with parallelism overhead).
                if elapsed_rate > avg:
                    if is_fallback or avg < 0.01:
                        is_fallback = True
                    avg = elapsed_rate
                done = self.summarize_counts_by_type_tier.get((element_type, tier), 0)
                # Include all items, even those with no timing data yet (avg=0)
                breakdown.append((element_type, tier, avg, is_fallback, done, tot))

            # Sort by hierarchy (file → class/interface/type_alias → function → method → variable → import), then tier descending
            type_order = {"file": 0, "class": 1, "interface": 1, "type_alias": 1, "function": 2, "method": 3, "variable": 4, "constant": 5, "import": 6}
            breakdown.sort(key=lambda x: (type_order.get(x[0], 99), -x[1]))
            return breakdown
