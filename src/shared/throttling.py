"""Runtime-aware throttling for parallel processing.

Tracks max runtimes of active workers and historical windows to make
intelligent throttling decisions that maximize throughput.

Two complementary strategies:
1. Formula-based safety: max_workers = (timeout * margin) / base_time
   Prevents timeouts but uses made-up constants.
2. Peak throughput tracking: find the concurrency level where tasks/sec peaks.
   Optimizes toward the actual sweet spot where GPU utilization is maximized
   without contention degrading total throughput.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from threading import Lock


def _log_throttle(message: str) -> None:
    """Debug logging - disabled by default."""
    pass


# Safety margin for throttling - use 65% of timeout to leave 35% headroom
# for variance in task runtimes when running concurrently.
# Example: if base_time=1s and timeout=10s, theoretical max is 10 workers,
# but we limit to 10 * 0.65 = 6 workers for safety.
THROTTLE_SAFETY_MARGIN = 0.65

# Ramp-up factor for gradual scaling. When increasing workers, only move
# this fraction of the way from current to target. This prevents overwhelming
# the system during warmup when base_time estimates are optimistic.
# Example: current=2, target=32 → 2 + (32-2) * 0.25 = 9 workers
# Note: scaling DOWN is always instant (we're seeing slowness, react fast).
RAMP_UP_FACTOR = 0.25

# Maximum workers to add per ramp-up. Even if 25% of gap is larger,
# cap the increment to avoid overwhelming the system when base_time
# estimates are too optimistic (e.g., first few tasks were easy/small).
MAX_RAMP_INCREMENT = 1

# Threshold for holding ramp-up. If any active task has been running longer
# than this fraction of timeout, don't ramp up - wait for tasks to complete
# and provide feedback. This prevents ramping blindly based on stale data.
# 30% gives time for contention to show before we add more workers.
RAMP_HOLD_THRESHOLD = 0.30

# Exploration: when peak is confident but nearby levels lack data, temporarily
# target an under-explored level to collect samples and verify the peak is real.
# Significance threshold: max(EXPLORE_MIN_SAMPLES, level * EXPLORE_SAMPLES_PER_LEVEL).
# At level 2 → 10 (min floor), level 5 → 10, level 6 → 12, level 10 → 20.
# Higher concurrency has more variance, so we need more data to trust it.
# Exploration range: ±(max_level // 3) around the peak (clamped to 1..max_level).
# Priority: scan outward from peak (upward first, then downward).
EXPLORE_MIN_SAMPLES = 10
EXPLORE_SAMPLES_PER_LEVEL = 2

# Budget-aware exploration: skip if remaining elements < samples_needed * multiplier.
# With x3, we only explore if there's enough runway for the exploration data
# PLUS meaningful processing at the (possibly new) optimal level afterward.
EXPLORE_BUDGET_MULTIPLIER = 3

# Scoring weights for exploration target selection (sum to 1.0).
# Each candidate level gets a weighted score across 4 dimensions.
EXPLORE_WEIGHT_COMPLETION = 0.40  # Prefer levels with partial data (cheaper to finish)
EXPLORE_WEIGHT_PROXIMITY = 0.25  # Prefer levels near the peak
EXPLORE_WEIGHT_TREND = 0.25  # Prefer direction where base times improve
EXPLORE_WEIGHT_COST = 0.10  # Prefer levels needing fewer total samples

# Minimum explored levels needed for a reliable trend signal.
# Below this, trend component is zeroed out (no directional bias).
EXPLORE_TREND_MIN_LEVELS = 3

# Ramp cooldown scales with context tier: tier // 512 seconds.
# 2k→4s, 4k→8s, 8k→16s, 16k→32s, 32k→64s
# Larger contexts take longer to process, so we wait longer to see impact.
def get_ramp_cooldown(tier: int) -> float:
    """Get ramp cooldown in seconds based on context tier."""
    return tier // 512


@dataclass
class ThrottleDecision:
    """Result of throttling analysis."""

    should_throttle: bool
    current_max: float  # Max runtime of active workers
    historical_max: float  # Max from historical windows
    completed_avg: float  # Average runtime from completions (used for throttling)
    recommended_workers: int  # Suggested worker count
    reason: str
    completion_count: int = 0  # Number of completions used for completed_avg
    peak_concurrency: int | None = None  # Concurrency level with peak throughput (if available)
    exploration_target: int | None = None  # Level being explored (if any)
    # Per-level throughput data: level -> (throughput_per_sec, sample_count)
    all_levels: dict[int, tuple[float, int]] | None = None


class ThroughputTracker:
    """Tracks completion throughput with concurrency context for throttling.

    Records (runtime, concurrent_workers) for each completion to understand
    how task duration relates to system load. This allows smarter throttling:
    - If tasks are slow at high concurrency but fast at low → GPU contention
    - If tasks are slow regardless of concurrency → inherently slow tasks

    The key insight: a 50s task with 8 workers is different from 50s with 1 worker.
    """

    def __init__(self, window_seconds: float = 10.0):
        """Initialize throughput tracker.

        Args:
            window_seconds: Time window for measuring throughput (default 10s)
        """
        self.window_seconds = window_seconds
        # Store (timestamp, runtime, concurrent_workers) for each completion
        self.completions: deque[tuple[float, float, int]] = deque()
        self._lock = Lock()
        # Track throughput at each concurrency level for peak detection
        self._throughput_by_level = ThroughputByLevel(window_seconds=window_seconds)

    def record_completion(self, runtime: float, concurrent_workers: float = 1.0) -> None:
        """Record a completed task with concurrency context.

        Args:
            runtime: Task runtime in seconds
            concurrent_workers: Average workers active during task (start + end / 2)
        """
        now = time.time()
        base_time = runtime / max(concurrent_workers, 1.0)
        _log_throttle(
            f"RECORD: wall={runtime:.1f}s workers={concurrent_workers:.1f} → base={base_time:.1f}s"
        )
        with self._lock:
            self.completions.append((now, runtime, concurrent_workers))
            # Prune old entries outside window
            cutoff = now - self.window_seconds
            while self.completions and self.completions[0][0] < cutoff:
                self.completions.popleft()
        # Also record for per-level peak throughput tracking
        self._throughput_by_level.record(round(concurrent_workers), runtime)

    def get_stats(self) -> tuple[float, float, int]:
        """Get throughput statistics (backwards compatible).

        Returns:
            Tuple of (throughput_per_sec, avg_runtime, completion_count)
        """
        now = time.time()
        with self._lock:
            # Prune old entries
            cutoff = now - self.window_seconds
            while self.completions and self.completions[0][0] < cutoff:
                self.completions.popleft()

            if not self.completions:
                return 0.0, 0.0, 0

            count = len(self.completions)
            total_runtime = sum(r for _, r, _ in self.completions)
            avg_runtime = total_runtime / count

            # Calculate actual time span of completions
            oldest = self.completions[0][0]
            time_span = now - oldest
            if time_span < 1.0:
                time_span = 1.0  # Avoid division issues for very short spans

            throughput = count / time_span
            return throughput, avg_runtime, count

    def get_stats_with_concurrency(self) -> tuple[float, float, int, float, float]:
        """Get throughput statistics with concurrency-normalized base time.

        The key insight: runtime scales linearly with concurrent workers (GPU contention).
        So we normalize: base_time = runtime / workers

        Example: 10 workers each taking 10s means base_time = 1s per task.
        If timeout is 7s, max_workers = 7/1 = 7.

        Returns:
            Tuple of (throughput, avg_runtime, count, avg_concurrency, avg_base_time)
            - avg_concurrency: average workers active at task start
            - avg_base_time: average of (runtime / workers) for each completion.
              This is the normalized per-worker cost. Returns 0 if no data.
        """
        now = time.time()
        with self._lock:
            # Prune old entries
            cutoff = now - self.window_seconds
            while self.completions and self.completions[0][0] < cutoff:
                self.completions.popleft()

            if not self.completions:
                return 0.0, 0.0, 0, 0.0, 0.0

            count = len(self.completions)
            total_runtime = sum(r for _, r, _ in self.completions)
            total_concurrency = sum(c for _, _, c in self.completions)
            avg_runtime = total_runtime / count
            avg_concurrency = total_concurrency / count

            # Calculate actual time span of completions
            oldest = self.completions[0][0]
            time_span = now - oldest
            if time_span < 1.0:
                time_span = 1.0
            throughput = count / time_span

            # Calculate avg base_time = runtime / workers for each completion
            # This normalizes for concurrency: a 50s task with 8 workers = 6.25s base
            # Workers is avg of start+end counts; for warmup (0), treat as 1
            base_times = [r / max(c, 1.0) for _, r, c in self.completions]
            avg_base_time = sum(base_times) / len(base_times) if base_times else 0.0

            return throughput, avg_runtime, count, avg_concurrency, avg_base_time

    def reset(self) -> None:
        """Clear all history."""
        with self._lock:
            self.completions.clear()
        self._throughput_by_level.reset()

    def get_peak_concurrency(self) -> int | None:
        """Get the concurrency level with peak throughput.

        Returns:
            The concurrency level (int) where throughput peaked, or None
            if insufficient data (< 2 levels with enough samples).
        """
        result = self._throughput_by_level.get_peak_level()
        return result[0] if result else None

    def get_exploration_target(self, max_level: int, remaining: int | None = None) -> int | None:
        """Get a level that needs more data.

        If the peak is confident but any level within ±(max_level // 3) of
        the peak lacks samples, returns the nearest under-explored level
        (scanning outward from peak: upward first, then downward).

        Args:
            max_level: Upper bound for exploration (typically base_workers).
                Also determines radius: max(3, max_level // 3).
            remaining: Number of elements left to process, or None if unknown.
                When provided, skips exploration if the budget is too small.

        Returns:
            Level to explore, or None if all levels in range are significant.
        """
        return self._throughput_by_level.get_exploration_target(max_level, remaining)


# Keep old name as alias for compatibility
RuntimeHistory = ThroughputTracker


@dataclass
class TimeoutEvent:
    """Record of a timeout event for debugging."""

    element_id: str
    element_type: str
    tier: int
    workers_active: int
    avg_runtime: float
    max_runtime: float
    timeout_limit: float
    timestamp: float = field(default_factory=time.time)

    def to_log_line(self) -> str:
        """Format as a log line for debugging."""
        return (
            f"[TIMEOUT] element={self.element_type}:{self.element_id.split(':')[-2]}, "
            f"tier={self.tier}, workers={self.workers_active}, "
            f"avg={self.avg_runtime:.1f}s, max={self.max_runtime:.1f}s, "
            f"limit={self.timeout_limit:.0f}s"
        )


class ThroughputByLevel:
    """Track throughput at each concurrency level to find the peak.

    The key insight: there's a sweet spot for concurrency where throughput
    (tasks/second) is maximized. Below it, GPU sits idle. Above it,
    contention causes each task to take longer, reducing total throughput.

    Records completions bucketed by concurrency level. For each level,
    computes actual throughput = completions_in_window / window_duration.
    The level with highest throughput is the target.

    Requires data at >= 2 levels (each with level-proportional samples:
    max(EXPLORE_MIN_SAMPLES, level * EXPLORE_SAMPLES_PER_LEVEL)) to detect a peak.
    With only 1 level, there's no slope to compare — returns None.
    """

    def __init__(self, window_seconds: float = 300.0):
        """Initialize throughput-by-level tracker.

        Data is kept for the entire tier lifetime (no time-based pruning).
        Call reset() on tier/model changes to clear stale data.

        Args:
            window_seconds: Unused (kept for backward compat). Data is never pruned by time.
        """
        # level -> deque of (timestamp, runtime) for all completions in this tier
        self._levels: dict[int, deque[tuple[float, float]]] = {}
        self._lock = Lock()

    def record(self, concurrency: int, runtime: float) -> None:
        """Record a completion at this concurrency level.

        Args:
            concurrency: Number of concurrent workers when task completed.
            runtime: Task runtime in seconds.
        """
        now = time.time()
        level = max(1, concurrency)  # Clamp to at least 1
        with self._lock:
            if level not in self._levels:
                self._levels[level] = deque()
            self._levels[level].append((now, runtime))

    def _compute_throughput(self, dq: deque[tuple[float, float]], now: float) -> float:
        """Compute throughput for a level's data (caller must hold lock).

        Throughput = completions / time_span, where time_span is from
        the oldest completion to now.
        """
        if not dq:
            return 0.0
        count = len(dq)
        oldest = dq[0][0]
        time_span = now - oldest
        if time_span < 1.0:
            time_span = 1.0  # Avoid division issues for near-instant completions
        return count / time_span

    def get_peak_level(self) -> tuple[int, float] | None:
        """Find concurrency level with lowest base time (best per-worker cost).

        Base time = runtime / concurrency_level. The level with the lowest
        average base time is the sweet spot: GPU handles that parallelism
        most efficiently.

        Returns:
            (level, avg_base_time) for the peak, or None if insufficient data.
            Requires >= 2 levels with sufficient samples each
            (level-proportional: max(EXPLORE_MIN_SAMPLES, level * EXPLORE_SAMPLES_PER_LEVEL)).
        """
        with self._lock:
            # Collect levels with sufficient samples (level-proportional threshold)
            qualified: list[tuple[int, float]] = []
            for level, dq in self._levels.items():
                if len(dq) >= self._min_samples_for_level(level):
                    avg_base = sum(rt / level for _, rt in dq) / len(dq)
                    qualified.append((level, avg_base))

            # Need >= 2 qualified levels to detect a peak
            if len(qualified) < 2:
                return None

            # Find level with lowest base time (= best throughput)
            best_level, best_bt = min(qualified, key=lambda x: x[1])
            return (best_level, best_bt)

    def get_all_levels(self) -> dict[int, tuple[float, int]]:
        """Get average base time (per-worker cost) for all levels (for display).

        Base time = runtime / concurrency_level. This normalizes across levels
        so they're directly comparable: a 6s task at level 3 = 2.0s base,
        same as a 2s task at level 1.

        Data is kept for the entire tier lifetime — reset() clears it on
        tier/model changes.

        Returns:
            Dict of level -> (avg_base_time_seconds, sample_count).
        """
        with self._lock:
            result = {}
            for level, dq in self._levels.items():
                if dq:
                    avg_base = sum(rt / level for _, rt in dq) / len(dq)
                    result[level] = (avg_base, len(dq))
            return result

    @staticmethod
    def _min_samples_for_level(level: int) -> int:
        """Minimum samples needed to trust a level.

        Higher concurrency → more variance → need more samples.
        Returns max(EXPLORE_MIN_SAMPLES, level * EXPLORE_SAMPLES_PER_LEVEL).
        Floor of 10 ensures even low levels get enough data.
        """
        return max(EXPLORE_MIN_SAMPLES, level * EXPLORE_SAMPLES_PER_LEVEL)

    def _compute_trend_locked(self) -> float:
        """Compute slope of base_time vs level (caller must hold lock).

        Uses simple linear regression across explored levels.
        Negative slope: base_time improves (decreases) at higher levels → explore up.
        Positive slope: base_time worsens (increases) at higher levels → explore down.
        Returns 0.0 if fewer than EXPLORE_TREND_MIN_LEVELS explored levels.
        """
        points: list[tuple[int, float]] = []
        for level, dq in self._levels.items():
            if len(dq) >= self._min_samples_for_level(level):
                avg_base = sum(rt / level for _, rt in dq) / len(dq)
                points.append((level, avg_base))

        if len(points) < EXPLORE_TREND_MIN_LEVELS:
            return 0.0

        n = len(points)
        sum_x = sum(x for x, _ in points)
        sum_y = sum(y for _, y in points)
        sum_xy = sum(x * y for x, y in points)
        sum_x2 = sum(x * x for x, _ in points)

        denom = n * sum_x2 - sum_x * sum_x
        if denom == 0:
            return 0.0

        return (n * sum_xy - sum_x * sum_y) / denom

    def get_exploration_target(self, max_level: int, remaining: int | None = None) -> int | None:
        """Find the most valuable under-explored level to collect data at.

        Scores candidate levels within ±(max_level // 3) of the peak by:
        - Completion: prefer levels with partial data (cheaper to finish)
        - Proximity: prefer levels near the peak (refine understanding)
        - Trend: prefer direction where base times are improving
        - Cost: prefer levels needing fewer total samples

        Budget-aware: skips candidates when remaining elements can't cover
        the sample cost (samples_needed * EXPLORE_BUDGET_MULTIPLIER).

        Args:
            max_level: Upper bound for exploration (typically base_workers).
                Also determines radius: max(3, max_level // 3).
            remaining: Number of elements left to process, or None if unknown.
                When provided, skips candidates whose sample cost exceeds
                the remaining budget.

        Returns:
            Level to explore, or None if all levels in range are significant.
        """
        peak_result = self.get_peak_level()
        if peak_result is None:
            return None

        peak, _ = peak_result
        radius = max(3, max_level // 3)

        with self._lock:
            # Peak must be confident before we explore from it
            peak_count = len(self._levels.get(peak, deque()))
            if peak_count < self._min_samples_for_level(peak):
                return None

            # Compute trend once for all candidates
            slope = self._compute_trend_locked()

            # Bounds for candidate range
            upper = min(peak + radius, max_level)
            lower = max(peak - radius, 1)
            max_possible_cost = self._min_samples_for_level(max_level)

            best_score = -1.0
            best_candidate = None

            # Build candidate list: scan outward from peak (upward first)
            # so ties break toward higher concurrency (more potential).
            candidates: list[int] = []
            for offset in range(1, radius + 1):
                up = peak + offset
                if up <= upper:
                    candidates.append(up)
                down = peak - offset
                if down >= lower:
                    candidates.append(down)

            for candidate in candidates:

                needed = self._min_samples_for_level(candidate)
                count = len(self._levels.get(candidate, deque()))

                if count >= needed:
                    continue  # Already explored

                # Budget filter (hard gate)
                samples_left = needed - count
                if remaining is not None and remaining < samples_left * EXPLORE_BUDGET_MULTIPLIER:
                    continue

                # A) Completion score: partial progress = cheaper to finish
                completion_score = count / needed if needed > 0 else 0.0

                # B) Proximity score: closer to peak = more informative
                distance = abs(candidate - peak)
                proximity_score = 1.0 - (distance / radius) if radius > 0 else 1.0

                # C) Trend score: favor the direction where base times improve
                if slope == 0.0:
                    trend_score = 0.5  # Neutral — no directional preference
                elif slope < 0:
                    # Base time decreasing at higher levels → explore upward
                    trend_score = 1.0 if candidate > peak else 0.0
                else:
                    # Base time increasing at higher levels → explore downward
                    trend_score = 1.0 if candidate < peak else 0.0

                # D) Cost score: fewer remaining samples = cheaper
                cost_score = (
                    1.0 - (samples_left / max_possible_cost)
                    if max_possible_cost > 0
                    else 0.5
                )

                score = (
                    EXPLORE_WEIGHT_COMPLETION * completion_score
                    + EXPLORE_WEIGHT_PROXIMITY * proximity_score
                    + EXPLORE_WEIGHT_TREND * trend_score
                    + EXPLORE_WEIGHT_COST * cost_score
                )

                if score > best_score:
                    best_score = score
                    best_candidate = candidate

        return best_candidate

    def reset(self) -> None:
        """Clear all history."""
        with self._lock:
            self._levels.clear()


def compute_throttle_decision(
    current_max_runtime: float,
    tier_timeout: float,
    base_workers: int,
    active_workers: int,
    throughput: float = 0.0,
    avg_runtime: float = 0.0,
    completion_count: int = 0,
    avg_concurrency: float = 0.0,
    avg_base_time: float = 0.0,
    post_warmup: bool = False,
    peak_concurrency: int | None = None,
    exploration_target: int | None = None,
) -> ThrottleDecision:
    """Determine if throttling should be applied.

    Uses two complementary strategies:
    1. Peak throughput: if we have data at multiple concurrency levels, target
       the level where actual throughput (tasks/sec) peaked. This optimizes
       toward the real sweet spot rather than using made-up constants.
    2. Formula safety cap: max_workers = (timeout * safety_margin) / base_time.
       Prevents timeouts. The peak can never exceed this cap.

    The recommended workers = min(peak_level, formula_limit) when peak data
    is available, otherwise falls back to formula only.

    When exploration_target is set, it temporarily overrides the peak as the
    optimization target — allowing the system to ramp beyond the current peak
    to collect data at a neighboring level.

    Args:
        current_max_runtime: Max runtime of currently active workers
        tier_timeout: Timeout for this tier (e.g., 180s for summarize)
        base_workers: Original max workers
        active_workers: Currently active worker count
        throughput: Actual completions per second (for display)
        avg_runtime: Average completion time (all conditions)
        completion_count: Number of completions in window
        avg_concurrency: Average workers active at task start
        avg_base_time: Average of (runtime/workers) from completions - normalized cost
        post_warmup: If True, just exited warmup - force gradual ramp from 1 worker
            instead of using historical data for FRESH START. Historical data may
            be from a different tier/model and shouldn't be trusted for initial count.
        peak_concurrency: Concurrency level with highest observed throughput, or None
            if insufficient data. When set, this becomes the primary optimization
            target, capped by the formula-based safety limit.
        exploration_target: Level to explore (neighbor of peak lacking data), or None.
            When set, overrides peak_concurrency as the optimization target.

    Returns:
        ThrottleDecision with recommended action
    """
    # Emergency check uses RAW max_runtime - is any task about to timeout?
    # Use 60% threshold - more aggressive than safety margin (65%) to react before it's too late
    raw_ratio = current_max_runtime / tier_timeout if tier_timeout > 0 else 0.0
    if raw_ratio >= 0.60:
        _log_throttle(
            f"EMERGENCY: max_runtime={current_max_runtime:.1f}s ({raw_ratio:.0%} of {tier_timeout}s timeout) "
            f"active={active_workers} → forcing 1 worker"
        )
        return ThrottleDecision(
            should_throttle=True,
            current_max=current_max_runtime,
            historical_max=0,
            completed_avg=current_max_runtime,
            recommended_workers=1,
            reason="Emergency (>60% timeout)",
            completion_count=completion_count,
        )

    # Normalize max_runtime by current workers for hold threshold comparison
    # This puts it on the same scale as base_time (per-worker cost)
    normalized_max = current_max_runtime / max(active_workers, 1)
    max_ratio = normalized_max / tier_timeout if tier_timeout > 0 else 0.0

    # Check if any task is taking too long - if so, hold at current level
    # Uses raw max (like emergency) but with lower threshold
    hold_threshold = tier_timeout * RAMP_HOLD_THRESHOLD
    should_hold = current_max_runtime > hold_threshold and active_workers > 0

    # Use the larger of historical avg_base_time or normalized_max for worker calculation.
    # - Historical: average from completed tasks
    # - Normalized: current max / workers (reflects current conditions)
    # Using max() is conservative: if current tasks are slower, account for that.
    if avg_base_time > 0:
        effective_base_time = max(avg_base_time, normalized_max)
        if normalized_max > avg_base_time:
            _log_throttle(
                f"BASE TIME: using normalized_max={normalized_max:.1f}s > historical={avg_base_time:.1f}s"
            )
    else:
        # No completion history yet - check if we should hold or ramp
        if should_hold:
            # Tasks running long, hold at current level until we get feedback
            _log_throttle(
                f"NO DATA HOLD: max={current_max_runtime:.1f}s ({current_max_runtime/tier_timeout:.0%} of {tier_timeout}s) "
                f"> {RAMP_HOLD_THRESHOLD:.0%} threshold, holding at {active_workers}"
            )
            return ThrottleDecision(
                should_throttle=False,
                current_max=current_max_runtime,
                historical_max=0,
                completed_avg=avg_runtime,
                recommended_workers=active_workers,
                reason=f"No data, holding (>{RAMP_HOLD_THRESHOLD:.0%} timeout)",
                completion_count=completion_count,
            )
        elif active_workers > 0:
            # Tasks running fast, safe to ramp
            delta = base_workers - active_workers
            increment = min(max(1, int(delta * RAMP_UP_FACTOR)), MAX_RAMP_INCREMENT)
            ramped = active_workers + increment
            ramped = min(ramped, base_workers)
            _log_throttle(
                f"NO DATA RAMP: active={active_workers} normalized_max={normalized_max:.1f}s "
                f"< {hold_threshold:.0f}s threshold → ramped to {ramped} (+{increment})"
            )
            return ThrottleDecision(
                should_throttle=False,
                current_max=current_max_runtime,
                historical_max=0,
                completed_avg=avg_runtime,
                recommended_workers=ramped,
                reason=f"No data, ramped from {active_workers}",
                completion_count=completion_count,
            )
        else:
            # Fresh start with no data - start with 1 worker for warmup
            _log_throttle(f"NO DATA FRESH: starting at 1 worker")
            return ThrottleDecision(
                should_throttle=False,
                current_max=current_max_runtime,
                historical_max=0,
                completed_avg=avg_runtime,
                recommended_workers=1,
                reason="No data (fresh)",
                completion_count=completion_count,
            )

    # Calculate optimal workers: (timeout * safety_margin) / base_time
    # This formula acts as a SAFETY CAP to prevent timeouts
    effective_timeout = tier_timeout * THROTTLE_SAFETY_MARGIN
    formula_optimal = int(effective_timeout / effective_base_time)
    formula_optimal = max(1, min(formula_optimal, base_workers))  # Clamp to [1, base_workers]

    # Apply peak throughput optimization: if we have data showing which
    # concurrency level produces the best throughput, use that as the
    # primary target — but never exceed the formula's safety cap.
    # Exploration target overrides peak when set (temporarily ramp beyond
    # current peak to collect data at a neighboring level).
    if exploration_target is not None:
        optimal = min(exploration_target, formula_optimal)
        optimal = max(1, min(optimal, base_workers))  # Re-clamp
        _log_throttle(
            f"EXPLORE: target@{exploration_target} (peak@{peak_concurrency}) formula@{formula_optimal} → {optimal}"
        )
    elif peak_concurrency is not None:
        optimal = min(peak_concurrency, formula_optimal)
        optimal = max(1, min(optimal, base_workers))  # Re-clamp
        if peak_concurrency < formula_optimal:
            _log_throttle(
                f"PEAK THROUGHPUT: peak@{peak_concurrency} < formula@{formula_optimal} → using peak"
            )
    else:
        optimal = formula_optimal

    # Determine target: either optimal (if throttling) or base_workers (if not)
    target_workers = optimal if optimal < base_workers else base_workers
    should_throttle = optimal < base_workers

    # Apply ramp-up logic: when INCREASING workers, move gradually (25% of gap)
    # When DECREASING, apply immediately (we're seeing slowness, react fast)
    # Only ramp if we have active workers (not starting fresh)
    if target_workers > active_workers and active_workers > 0:
        # Check if we should hold due to long-running tasks
        if should_hold:
            # Tasks running long, hold at current level until we get fresh feedback
            effective_workers = active_workers
            reason_suffix = f", holding (>{RAMP_HOLD_THRESHOLD:.0%} timeout)"
            _log_throttle(
                f"RAMP HOLD: max={current_max_runtime:.1f}s ({current_max_runtime/tier_timeout:.0%} of {tier_timeout}s) "
                f"> {RAMP_HOLD_THRESHOLD:.0%} threshold, holding at {active_workers} (target={target_workers})"
            )
        else:
            # Scaling UP - ramp gradually to avoid overwhelming during warmup
            # Cap increment to MAX_RAMP_INCREMENT to handle optimistic base_time estimates
            delta = target_workers - active_workers
            increment = min(max(1, int(delta * RAMP_UP_FACTOR)), MAX_RAMP_INCREMENT)
            ramped = active_workers + increment
            ramped = min(ramped, target_workers)  # Don't exceed target
            effective_workers = ramped
            reason_suffix = f", ramped from {active_workers}"
            _log_throttle(
                f"RAMP UP: base_time={effective_base_time:.1f}s target={target_workers} "
                f"active={active_workers} normalized_max={normalized_max:.1f}s < {hold_threshold:.0f}s → ramped to {effective_workers} (+{increment})"
            )
    else:
        # Scaling DOWN, steady, or starting fresh - apply immediately
        effective_workers = target_workers
        reason_suffix = ""
        if target_workers < active_workers:
            _log_throttle(
                f"SCALE DOWN: base_time={effective_base_time:.1f}s target={target_workers} "
                f"active={active_workers} → immediate drop to {effective_workers}"
            )
        elif active_workers == 0:
            if post_warmup:
                # Just came out of warmup - start at 1 and ramp gradually
                # Historical data may be from different tier, don't trust it for initial count
                effective_workers = 1
                reason_suffix = ", post-warmup gradual ramp"
                _log_throttle(
                    f"POST WARMUP: base_time={effective_base_time:.1f}s target={target_workers} "
                    f"→ starting at 1 worker (gradual ramp)"
                )
            else:
                _log_throttle(
                    f"FRESH START: base_time={effective_base_time:.1f}s → starting at {effective_workers} workers"
                )

    if should_throttle:
        peak_info = f",peak@{peak_concurrency}" if peak_concurrency is not None else ""
        explore_info = f",explore@{exploration_target}" if exploration_target is not None else ""
        return ThrottleDecision(
            should_throttle=True,
            current_max=current_max_runtime,
            historical_max=0,
            completed_avg=effective_base_time,
            recommended_workers=effective_workers,
            reason=f"Throttle ({effective_workers}={int(effective_timeout)}s/{effective_base_time:.1f}s base{peak_info}{explore_info}{reason_suffix})",
            completion_count=completion_count,
            peak_concurrency=peak_concurrency,
            exploration_target=exploration_target,
        )
    else:
        return ThrottleDecision(
            should_throttle=False,
            current_max=current_max_runtime,
            historical_max=0,
            completed_avg=effective_base_time if effective_base_time > 0 else avg_runtime,
            recommended_workers=effective_workers,
            reason=f"Normal{reason_suffix}" if reason_suffix else "Normal",
            completion_count=completion_count,
            peak_concurrency=peak_concurrency,
            exploration_target=exploration_target,
        )


def _lerp_color(t: float) -> str:
    """Interpolate from green (t=0, best) to red (t=1, worst).

    Uses green → yellow → red gradient via RGB interpolation.

    Args:
        t: Value in [0, 1] where 0 = best (green), 1 = worst (red).

    Returns:
        Hex color string like ``#rrggbb``.
    """
    t = max(0.0, min(1.0, t))
    # Green (0,200,0) → Yellow (220,200,0) → Red (220,0,0)
    if t < 0.5:
        # Green → Yellow
        p = t * 2  # 0→1 over first half
        r = int(220 * p)
        g = 200
    else:
        # Yellow → Red
        p = (t - 0.5) * 2  # 0→1 over second half
        r = 220
        g = int(200 * (1 - p))
    return f"#{r:02x}{g:02x}00"


def _text_color_for_bg(bg_hex: str) -> str:
    """Return 'black' or 'white' for readable text on the given background.

    Uses relative luminance (ITU-R BT.709) to decide contrast.
    Light backgrounds (green, yellow) → black text.
    Dark backgrounds (red) → white text.
    """
    r = int(bg_hex[1:3], 16)
    g = int(bg_hex[3:5], 16)
    b = int(bg_hex[5:7], 16)
    # Perceived luminance (0-255 scale)
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "black" if luminance > 128 else "white"


_LEVEL_COL_WIDTH = 6  # Fixed width per level column (chars)
_LEVELS_PER_ROW = 16  # Max levels before wrapping to a new line


def _confidence_style(count: int) -> str:
    """Return a Rich style modifier based on sample count (data confidence).

    1 sample: dim (uncertain), 2: normal, 3+: bold (confident).
    """
    if count <= 1:
        return "dim"
    if count <= 2:
        return ""
    return "bold"


def _build_levels_row(
    levels_range: range,
    all_levels: dict[int, tuple[float, int]],
    min_bt: float,
    bt_range: float,
    peak_concurrency: int | None,
    exploration_target: int | None = None,
) -> object:
    """Build a single Rich Table for a chunk of levels (three rows: number + time + count).

    Text confidence varies by sample count: dim (1), normal (2), bold (3+).
    Color-coded background: green (best) → red (worst).
    Peak level is underlined. Exploration target is marked with "?" prefix.

    Args:
        levels_range: Range of level numbers to display in this row.
        all_levels: Full dict of level -> (avg_base_time_seconds, sample_count).
        min_bt: Minimum base time across ALL levels (for consistent color scaling).
        bt_range: max_bt - min_bt across ALL levels.
        peak_concurrency: The level identified as peak, or None.
        exploration_target: Level being explored (neighbor of peak), or None.

    Returns:
        A rich.table.Table renderable.
    """
    from rich.table import Table
    from rich.text import Text

    table = Table(
        show_header=False,
        show_edge=False,
        show_lines=False,
        box=None,
        padding=(0, 0),
        expand=False,
    )

    # Add columns: leading indent + one per level in this chunk
    table.add_column(width=2, no_wrap=True)  # indent
    for _ in levels_range:
        table.add_column(width=_LEVEL_COL_WIDTH, no_wrap=True, justify="center")

    row1_cells: list[Text] = [Text("")]  # indent (level number)
    row2_cells: list[Text] = [Text("")]  # indent (base time)
    row3_cells: list[Text] = [Text("")]  # indent (sample count)

    for level in levels_range:
        is_peak = peak_concurrency is not None and level == peak_concurrency
        is_explore = exploration_target is not None and level == exploration_target
        if level in all_levels:
            avg_bt, count = all_levels[level]
            t = (avg_bt - min_bt) / bt_range if bt_range > 0 else 0.0
            color = _lerp_color(t)
            fg = _text_color_for_bg(color)
            bg = f"on {color}"
            conf = _confidence_style(count)
            # Combine: "bold black on #00c800" or "dim white on #dc0000"
            parts = [s for s in [conf, fg, bg] if s]
            style = " ".join(parts)
            if is_peak:
                style = f"underline {style}"

            # Mark exploration target with "?" prefix
            level_label = f"?{level}" if is_explore else str(level)
            level_str = level_label.center(_LEVEL_COL_WIDTH)
            row1_cells.append(Text(level_str, style=style))

            bt_str = f"{avg_bt:.1f}s" if avg_bt < 100 else f"{avg_bt:.0f}s"
            bt_padded = bt_str.center(_LEVEL_COL_WIDTH)
            row2_cells.append(Text(bt_padded, style=style))

            count_str = str(count).center(_LEVEL_COL_WIDTH)
            row3_cells.append(Text(count_str, style=style))
        else:
            # Mark exploration target even when no data yet
            level_label = f"?{level}" if is_explore else str(level)
            level_str = level_label.center(_LEVEL_COL_WIDTH)
            row1_cells.append(Text(level_str, style="dim"))
            row2_cells.append(Text("···".center(_LEVEL_COL_WIDTH), style="dim"))
            row3_cells.append(Text(" ".center(_LEVEL_COL_WIDTH), style="dim"))

    table.add_row(*row1_cells)
    table.add_row(*row2_cells)
    table.add_row(*row3_cells)

    return table


def _build_levels_table(
    all_levels: dict[int, tuple[float, int]],
    peak_concurrency: int | None = None,
    max_workers: int = 0,
    exploration_target: int | None = None,
) -> object:
    """Build color-graded level blocks, wrapping every 16 levels.

    Each level is a fixed-width column with two rows:
    - Row 1: level number (centered)
    - Row 2: avg base time, e.g. "1.3s" (centered)

    Columns are color-coded on a green→red gradient based on base time.
    Levels without data show dim placeholders. The peak level is highlighted.
    Exploration target is marked with "?" prefix.
    When there are more than 16 levels, they wrap onto new lines.

    Args:
        all_levels: Dict of level -> (avg_base_time_seconds, sample_count). Must be non-empty.
        peak_concurrency: The level identified as peak, or None.
        max_workers: Total worker slots (1..max_workers). If 0, uses max level in data.
        exploration_target: Level being explored, or None.

    Returns:
        A Rich renderable (Table for ≤16 levels, Group of Tables for >16).
    """
    from rich.console import Group

    max_level = max_workers if max_workers > 0 else max(all_levels.keys())

    # Find min/max base times for color scaling (consistent across all chunks)
    base_times = {lvl: bt for lvl, (bt, _) in all_levels.items()}
    min_bt = min(base_times.values())
    max_bt = max(base_times.values())
    bt_range = max_bt - min_bt

    # Split levels into chunks of _LEVELS_PER_ROW
    chunks: list[range] = []
    for start in range(1, max_level + 1, _LEVELS_PER_ROW):
        end = min(start + _LEVELS_PER_ROW, max_level + 1)
        chunks.append(range(start, end))

    if len(chunks) == 1:
        return _build_levels_row(chunks[0], all_levels, min_bt, bt_range, peak_concurrency, exploration_target)

    tables = [
        _build_levels_row(chunk, all_levels, min_bt, bt_range, peak_concurrency, exploration_target)
        for chunk in chunks
    ]
    return Group(*tables)


def format_throughput_levels(
    all_levels: dict[int, tuple[float, int]] | None,
    peak_concurrency: int | None = None,
    max_workers: int = 0,
    exploration_target: int | None = None,
) -> object | str:
    """Build a color-graded level table (Rich renderable).

    Each level is a fixed-width block, two rows tall:
    - Row 1: level number (centered)
    - Row 2: avg base time (centered)
    Color-coded green (best) → red (worst). Peak level highlighted.
    Exploration target marked with "?" prefix.

    Args:
        all_levels: Dict of level -> (avg_base_time_seconds, sample_count), or None.
        peak_concurrency: The level identified as peak, or None.
        max_workers: Total worker slots to display (1..max_workers). If 0, uses max level.
        exploration_target: Level being explored, or None.

    Returns:
        Rich Table renderable, or empty string if no data.
    """
    if not all_levels:
        return ""

    return _build_levels_table(all_levels, peak_concurrency, max_workers, exploration_target)


def build_throughput_levels_text(
    all_levels: dict[int, tuple[float, int]] | None,
    peak_concurrency: int | None = None,
    max_workers: int = 0,
    exploration_target: int | None = None,
) -> object | None:
    """Build a color-graded level table (Rich renderable).

    Each level is a fixed-width block, two rows tall:
    - Row 1: level number (centered)
    - Row 2: avg base time (centered)
    Color-coded green (best) → red (worst). Peak level highlighted.
    Exploration target marked with "?" prefix.

    Returns None if no data.

    Args:
        all_levels: Dict of level -> (avg_base_time_seconds, sample_count), or None.
        peak_concurrency: The level identified as peak, or None.
        max_workers: Total worker slots to display (1..max_workers). If 0, uses max level.
        exploration_target: Level being explored, or None.

    Returns:
        A Rich Table renderable, or None if no data.
    """
    if not all_levels:
        return None

    return _build_levels_table(all_levels, peak_concurrency, max_workers, exploration_target)
