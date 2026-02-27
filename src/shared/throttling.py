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

import math
import os
import time
from collections import deque
from dataclasses import dataclass, field
from threading import Lock

_throttle_log_initialized = False


def _log_throttle(message: str) -> None:
    """Debug logging - writes to /tmp/throttle.log.

    Truncates the file on the first call per process to avoid
    unbounded growth across runs.
    """
    global _throttle_log_initialized  # noqa: PLW0603
    mode = "a" if _throttle_log_initialized else "w"
    with open("/tmp/throttle.log", mode) as f:
        if not _throttle_log_initialized:
            f.write(f"{time.time():.3f} --- throttle log start (pid={os.getpid()}) ---\n")
            _throttle_log_initialized = True
        f.write(f"{time.time():.3f} {message}\n")


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
# 50% is conservative enough to avoid premature holds while still catching
# genuine slowdowns before emergency kicks in.
RAMP_HOLD_THRESHOLD = 0.50

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

# Exploration budget: at most this fraction of total elements is spent on
# warmup + GSS probing. The rest runs at the discovered peak.
# 0.5 = spend at most half exploring, half at the optimum.
EXPLORE_BUDGET_FRACTION = 0.5

# Signal-aware GSS: minimum samples to consider a level's partial signal.
# 3 is the minimum for a meaningful signal; 2 is too noisy, 1 is random.
GSS_PARTIAL_MIN_SAMPLES = 3

# Promising threshold: base_time <= best_known * this factor.
# 1.2 = within 20% of best. Generous enough to catch near-optimal levels.
GSS_PROMISING_THRESHOLD = 1.2

# Blacklist threshold: base_time > best_known * this factor.
# 1.5 = 50% worse than best. Conservative enough to avoid false positives
# from high-variance early samples.
GSS_BLACKLIST_THRESHOLD = 1.5

# Scoring weights for exploration target selection (sum to 1.0).
# Each candidate level gets a weighted score across 4 dimensions.
EXPLORE_WEIGHT_COMPLETION = 0.40  # Prefer levels with partial data (cheaper to finish)
EXPLORE_WEIGHT_PROXIMITY = 0.25  # Prefer levels near the peak
EXPLORE_WEIGHT_TREND = 0.25  # Prefer direction where base times improve
EXPLORE_WEIGHT_COST = 0.10  # Prefer levels needing fewer total samples

# Minimum explored levels needed for a reliable trend signal.
# Below this, trend component is zeroed out (no directional bias).
EXPLORE_TREND_MIN_LEVELS = 3

# Formula confidence curve: scales down formula_optimal when completion data is sparse.
# With few completions, base_time is unreliable and the formula produces aggressive
# worker counts. The confidence multiplier brings it down until data firms up.
# Logarithmic curve: count=1→0.30, count=5→0.65, count=10→0.80, count=20→1.0.
FORMULA_CONFIDENCE_MIN = 0.30  # Floor confidence at count=1
FORMULA_CONFIDENCE_FULL_AT = 25  # Full confidence at this many completions
FORMULA_CONFIDENCE_THRESHOLD = 0.95  # Snap to 1.0 above this


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
    gss_probe: int | None = None  # GSS probe target (if active)
    gss_lo: int | None = None  # GSS bracket lower bound
    gss_hi: int | None = None  # GSS bracket upper bound
    gss_signal: str | None = None  # Signal-aware action for display
    is_hold: bool = False  # True when ramp-up is held (>50% timeout)
    is_emergency: bool = False  # True when emergency brake fired (>80% timeout)
    # Per-level throughput data: level -> (throughput_per_sec, sample_count)
    all_levels: dict[int, tuple[float, int]] | None = None
    # Probability map: level -> P(best) for display
    prob_map_data: dict[int, float] | None = None
    exploration_status: str | None = None  # Lifecycle status for constant feedback


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
        self.completions: deque[tuple[float, float, float]] = deque()
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

        If the peak is confident but any level within ±radius of
        the peak lacks samples, returns the nearest under-explored level
        (scanning outward from peak: upward first, then downward).

        Args:
            max_level: Upper bound for exploration (typically base_workers).
                Also determines radius: min(5, max(3, max_level // 4)).
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

    def __init__(self, window_seconds: float = 300.0):  # noqa: ARG002
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

        Scores candidate levels within ±min(5, max(3, max_level // 4)) of the peak by:
        - Completion: prefer levels with partial data (cheaper to finish)
        - Proximity: prefer levels near the peak (refine understanding)
        - Trend: prefer direction where base times are improving
        - Cost: prefer levels needing fewer total samples

        Budget-aware: skips candidates when remaining elements can't cover
        the sample cost (samples_needed * EXPLORE_BUDGET_MULTIPLIER).

        Args:
            max_level: Upper bound for exploration (typically base_workers).
                Also determines radius: min(5, max(3, max_level // 4)).
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
        radius = min(5, max(3, max_level // 4))

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

            # Peak base_time for early bail-out of bad candidates
            peak_dq = self._levels.get(peak, deque())
            peak_bt = (sum(rt / peak for _, rt in peak_dq) / len(peak_dq)) if peak_dq else None

            for candidate in candidates:

                needed = self._min_samples_for_level(candidate)
                count = len(self._levels.get(candidate, deque()))

                if count >= needed:
                    continue  # Already explored

                # Early bail-out: if candidate already has some data and its
                # base_time is much worse than peak, don't waste more samples.
                # Uses GSS_BLACKLIST_THRESHOLD (1.5x) for consistency.
                # Min samples scales with level: higher concurrency = noisier.
                bail_min = max(GSS_PARTIAL_MIN_SAMPLES, candidate)
                if peak_bt is not None and count >= bail_min:
                    cand_dq = self._levels[candidate]
                    cand_bt = sum(rt / candidate for _, rt in cand_dq) / len(cand_dq)
                    if cand_bt > peak_bt * GSS_BLACKLIST_THRESHOLD:
                        continue  # Clearly worse — skip

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


PHI = (1 + math.sqrt(5)) / 2  # Golden ratio ≈ 1.618


class GoldenSectionSearch:
    """Golden section search for optimal concurrency level.

    Maintains a bracket [lo, hi] and narrows it by evaluating throughput
    at golden-ratio interior points. Converges in O(log_φ(N)) steps.

    Starts with full bracket [1, max_workers] and uses geometric probes
    for broad coverage, with signal-aware and probability map overrides.
    """

    def __init__(self, lo: int, hi: int):
        """Initialize with search bracket.

        Args:
            lo: Lower bound (inclusive), typically 1.
            hi: Upper bound (inclusive), typically max_workers.
        """
        self.lo = lo
        self.hi = hi
        self.converged = False
        self.best_level: int | None = None
        # Pending probe: the level we're currently collecting data for.
        # None means we need to decide which probe to request next.
        self._pending_probe: int | None = None
        # Last signal-aware action for display (e.g. "promote 7" or "skip [5,9]")
        self.last_signal: str | None = None
        # Scale convergence threshold with search space size.
        # ~30% of initial range, floored at 2 (tiny spaces still explore),
        # capped at 8 (huge spaces don't over-explore).
        self._initial_range = hi - lo
        self._min_bracket = max(2, min(8, round(self._initial_range * 0.3)))

    def get_next_probe(
        self,
        level_data: dict[int, float],
        partial_data: dict[int, tuple[float, int]] | None = None,
        prob_map: ProbabilityMap | None = None,
    ) -> int | None:
        """Get the next level to probe, or None if converged.

        Probe selection priority:
        1. Signal-aware override (partial data: promote/blacklist)
        2. Probability map (highest-probability unqualified level)
        3. Geometric (golden-ratio interior points)

        Args:
            level_data: Dict of level -> avg_base_time for levels with
                sufficient data (≥ EXPLORE_MIN_SAMPLES).
            partial_data: Dict of level -> (avg_base_time, sample_count) for
                levels with GSS_PARTIAL_MIN_SAMPLES..EXPLORE_MIN_SAMPLES-1
                samples. Used for signal-aware probe steering.
            prob_map: Probability distribution over levels. When active,
                steers probes toward highest-probability unqualified level.

        Returns:
            Level to collect data at, or None if search is complete.
        """
        self.last_signal = None  # Reset each call

        if self.converged:
            return None

        _log_throttle(
            f"GSS STATE: bracket=[{self.lo},{self.hi}] min_bracket={self._min_bracket} "
            f"level_data={sorted(level_data.keys())} "
            f"partial={sorted((partial_data or {}).keys())}"
        )

        # Probe stickiness: keep collecting at the current probe until it has
        # enough samples for meaningful signal. Scales with concurrency level
        # because higher parallelism = more variance per sample.
        # Without this, pmap/signal overrides bounce between levels after 1
        # sample, never accumulating enough data at any single level.
        if self._pending_probe is not None:
            pending = self._pending_probe
            if pending not in level_data:  # Not yet fully qualified
                count = 0
                if partial_data and pending in partial_data:
                    count = partial_data[pending][1]
                # Scale with level: at level 1, need 3; at level 13, need 13.
                # Same EXPLORE_SAMPLES_PER_LEVEL ratio as full qualification.
                min_sticky = max(GSS_PARTIAL_MIN_SAMPLES, pending * EXPLORE_SAMPLES_PER_LEVEL)
                if count < min_sticky:
                    self.last_signal = f"sticky {pending} ({count}/{min_sticky})"
                    return pending
            # Probe reached minimum samples or is qualified — release it
            self._pending_probe = None

        if self.hi - self.lo <= self._min_bracket:
            # Bracket narrow enough — pick best known level in [lo, hi]
            self._finalize(level_data, partial_data)
            return None

        # Step 0: Find best level across ALL data and protect it.
        # The bracket must always contain the best-known level.
        all_data: dict[int, float] = dict(level_data)
        if partial_data:
            for lvl, (bt, _cnt) in partial_data.items():
                if lvl not in all_data:
                    all_data[lvl] = bt
        _best_known: int | None = None
        if all_data:
            _best_known = min(all_data, key=all_data.get)  # type: ignore[arg-type]
            if _best_known < self.lo:
                old_lo = self.lo
                self.lo = _best_known
                _log_throttle(
                    f"GSS EXPAND lo: {old_lo}→{self.lo} "
                    f"(level {_best_known} bt={all_data[_best_known]:.2f} is best)"
                )
            elif _best_known > self.hi:
                old_hi = self.hi
                self.hi = _best_known
                _log_throttle(
                    f"GSS EXPAND hi: {old_hi}→{self.hi} "
                    f"(level {_best_known} bt={all_data[_best_known]:.2f} is best)"
                )

        # Step 1: Narrow bracket using any available data.
        # Loop until we can't narrow further — ensures bracket shrinks
        # every cycle, even when overrides would otherwise prevent it.
        # CRITICAL: Never narrow past the best-known level.
        narrowing_data = all_data

        # Scale fuzzy-match radius with current bracket width.
        # Small brackets (8): radius=2, large brackets (64): radius=6, cap at 6.
        _radius = max(2, min(6, (self.hi - self.lo) // 5))

        def _nearest_data(target: int, data: dict[int, float], radius: int = _radius) -> tuple[int, float] | None:
            """Find nearest level with data within ±radius of target."""
            if target in data:
                return (target, data[target])
            for offset in range(1, radius + 1):
                for candidate in [target - offset, target + offset]:
                    if candidate in data:
                        return (candidate, data[candidate])
            return None

        _max_narrow_iters = 20  # Safety: prevent infinite loop
        for _narrow_iter in range(_max_narrow_iters):
            if self.hi - self.lo <= self._min_bracket:
                break
            m1 = round(self.hi - (self.hi - self.lo) / PHI)
            m2 = round(self.lo + (self.hi - self.lo) / PHI)
            m1 = max(self.lo + 1, min(m1, self.hi - 1))
            m2 = max(self.lo + 1, min(m2, self.hi - 1))
            if m1 == m2:
                m2 = min(m1 + 1, self.hi - 1)

            d1 = _nearest_data(m1, narrowing_data)
            d2 = _nearest_data(m2, narrowing_data)

            if d1 is not None and d2 is not None:
                lvl1, bt1 = d1
                lvl2, bt2 = d2
                if lvl1 == lvl2:
                    break  # Both probes matched same level — need more data
                old_lo, old_hi = self.lo, self.hi
                if bt1 <= bt2:
                    # Lower region better — discard upper: narrow hi
                    new_hi = min(lvl2, m2)  # Use tighter of actual/geometric
                    # Never narrow below the best-known level
                    if _best_known is not None:
                        new_hi = max(new_hi, _best_known)
                    if new_hi >= self.hi:
                        break  # No progress
                    self.hi = new_hi
                else:
                    # Upper region better — discard lower: narrow lo
                    new_lo = max(lvl1, m1)  # Use tighter of actual/geometric
                    # Never narrow above the best-known level
                    if _best_known is not None:
                        new_lo = min(new_lo, _best_known)
                    if new_lo <= self.lo:
                        break  # No progress
                    self.lo = new_lo
                if self.lo != old_lo or self.hi != old_hi:
                    _log_throttle(
                        f"GSS NARROW: [{old_lo},{old_hi}]→[{self.lo},{self.hi}] "
                        f"(m1~{lvl1}:{bt1:.2f}, m2~{lvl2}:{bt2:.2f})"
                    )
            else:
                break  # No nearby data — need to collect it

        if self.hi - self.lo <= self._min_bracket:
            self._finalize(level_data, partial_data)
            return None

        # Recompute probes for current bracket
        m1 = round(self.hi - (self.hi - self.lo) / PHI)
        m2 = round(self.lo + (self.hi - self.lo) / PHI)
        m1 = max(self.lo + 1, min(m1, self.hi - 1))
        m2 = max(self.lo + 1, min(m2, self.hi - 1))
        if m1 == m2:
            m2 = min(m1 + 1, self.hi - 1)

        # Step 2: Signal-aware override within (possibly narrowed) bracket
        if partial_data:
            probe = self._signal_aware_probe(level_data, partial_data, m1, m2)
            if probe is not None:
                self._pending_probe = probe
                return probe

        # Step 3: Probability map override within bracket
        if prob_map is not None:
            qualified = set(level_data.keys())
            result = prob_map.get_best_unqualified(qualified, self.lo, self.hi)
            if result is not None:
                target, prob = result
                self.last_signal = f"pmap→{target} (P={prob:.2f})"
                _log_throttle(
                    f"GSS PMAP: steering to level {target} (P={prob:.3f})"
                )
                self._pending_probe = target
                return target

        # Step 4: Geometric probes for missing data
        have_m1 = m1 in narrowing_data
        have_m2 = m2 in narrowing_data

        if have_m1 and have_m2:
            # Both probes have data but narrowing couldn't make progress
            # (same-level fuzzy match, best_known protection, etc.).
            # Converge rather than endlessly re-probing known levels.
            _log_throttle(
                f"GSS STALL: both m1={m1} and m2={m2} have data but "
                f"narrowing stalled — forcing convergence"
            )
            self._finalize(level_data, partial_data)
            return None
        elif have_m1:
            if prob_map is not None:
                prob_map.record_geometric_probe()
            self._pending_probe = m2
            return m2
        elif have_m2:
            if prob_map is not None:
                prob_map.record_geometric_probe()
            self._pending_probe = m1
            return m1
        else:
            if prob_map is not None:
                prob_map.record_geometric_probe()
            self._pending_probe = m1
            return m1

    def _signal_aware_probe(
        self,
        level_data: dict[int, float],
        partial_data: dict[int, tuple[float, int]],
        m1: int,
        m2: int,
    ) -> int | None:
        """Override geometric probe using partial data signals.

        Three behaviors:
        1. If a promising partial level exists in [lo, hi], probe it instead
           of the geometric point (accelerates convergence toward likely optimum).
        2. If a geometric probe (m1 or m2) is blacklisted, redirect to the
           nearest non-blacklisted level.
        3. Returns None if no override needed (fall through to geometric).

        Args:
            level_data: Qualified levels with avg_base_time.
            partial_data: Partial levels with (avg_base_time, sample_count).
            m1: Lower geometric probe point.
            m2: Upper geometric probe point.

        Returns:
            Override probe level, or None to use default geometric.
        """
        # Find best known base_time as reference (from all qualified levels)
        if not level_data:
            return None  # No reference point to judge partial data

        best_bt = min(level_data.values())

        # Build blacklist: partial levels clearly worse than best
        blacklisted: set[int] = set()
        for lvl, (bt, _cnt) in partial_data.items():
            if self.lo <= lvl <= self.hi and bt > best_bt * GSS_BLACKLIST_THRESHOLD:
                blacklisted.add(lvl)

        if blacklisted:
            self.last_signal = f"skip {sorted(blacklisted)}"
            _log_throttle(
                f"GSS SIGNAL: blacklisted {sorted(blacklisted)} "
                f"(best_bt={best_bt:.2f}, threshold={best_bt * GSS_BLACKLIST_THRESHOLD:.2f})"
            )

        # Find promising partial levels: good base_time, not yet qualified
        promising: list[tuple[int, float, int]] = []  # (level, base_time, count)
        for lvl, (bt, cnt) in partial_data.items():
            if self.lo <= lvl <= self.hi and lvl not in blacklisted and bt <= best_bt * GSS_PROMISING_THRESHOLD:
                promising.append((lvl, bt, cnt))

        # Behavior 1: probe the most promising partial level
        # Prefer highest count (cheapest to promote), then lowest base_time
        if promising:
            promising.sort(key=lambda x: (-x[2], x[1]))
            target = promising[0][0]
            self.last_signal = f"promote {target} ({promising[0][2]}/{EXPLORE_MIN_SAMPLES})"
            _log_throttle(
                f"GSS SIGNAL: promoting level {target} "
                f"(bt={promising[0][1]:.2f}, cnt={promising[0][2]}, "
                f"best_bt={best_bt:.2f})"
            )
            return target

        # Behavior 2: redirect blacklisted geometric probes
        have_m1 = m1 in level_data
        have_m2 = m2 in level_data

        redirected = False
        if not have_m1 and m1 in blacklisted:
            new_m1 = self._nearest_non_blacklisted(m1, blacklisted)
            if new_m1 is not None and new_m1 not in level_data:
                self.last_signal = f"skip {m1}→{new_m1}"
                _log_throttle(
                    f"GSS SIGNAL: redirected probe {m1}→{new_m1} (blacklisted)"
                )
                return new_m1
            redirected = True

        if not have_m2 and m2 in blacklisted and not redirected:
            new_m2 = self._nearest_non_blacklisted(m2, blacklisted)
            if new_m2 is not None and new_m2 not in level_data:
                self.last_signal = f"skip {m2}→{new_m2}"
                _log_throttle(
                    f"GSS SIGNAL: redirected probe {m2}→{new_m2} (blacklisted)"
                )
                return new_m2

        # No override needed
        return None

    def _nearest_non_blacklisted(
        self, target: int, blacklisted: set[int]
    ) -> int | None:
        """Find nearest level to target within [lo, hi] not in blacklist.

        Searches outward from target in both directions simultaneously.

        Returns:
            Nearest non-blacklisted level, or None if all are blacklisted.
        """
        for offset in range(1, self.hi - self.lo + 1):
            for candidate in [target + offset, target - offset]:
                if self.lo <= candidate <= self.hi and candidate not in blacklisted:
                    return candidate
        return None

    def _finalize(
        self,
        level_data: dict[int, float],
        partial_data: dict[int, tuple[float, int]] | None = None,
    ) -> None:
        """Mark search as converged, pick best level in final bracket."""
        self.converged = True
        # Find best level in [lo, hi] range from qualified data
        candidates = {
            lvl: bt for lvl, bt in level_data.items()
            if self.lo <= lvl <= self.hi
        }
        if candidates:
            self.best_level = min(candidates, key=candidates.get)  # type: ignore[arg-type]
        elif partial_data:
            # No qualified data in bracket — use best partial level
            partial_candidates = {
                lvl: bt for lvl, (bt, _cnt) in partial_data.items()
                if self.lo <= lvl <= self.hi
            }
            if partial_candidates:
                self.best_level = min(partial_candidates, key=partial_candidates.get)  # type: ignore[arg-type]
            else:
                self.best_level = (self.lo + self.hi) // 2
        else:
            # Fallback: midpoint
            self.best_level = (self.lo + self.hi) // 2


class ProbabilityMap:
    """Independent per-level scores for probe steering.

    Each level has an independent score in [0.0, 1.0] indicating how likely
    it is to be the optimal concurrency.

    Two-pass scoring:
    1. Measured levels get direct scores: best time → 1.0, worst → 0.0.
    2. Unmeasured levels get Gaussian-weighted average from nearby measured
       levels, blended with neutral (0.5) by distance. This spreads signal
       to neighbors without compounding — each set_scores() call is a full
       replacement.

    Activation guard: only activates after sufficient geometric probes to
    prevent sequential walking from the low end.
    """

    # Minimum geometric probes before probability map activates
    ACTIVATION_MIN = 2

    def __init__(self, lo: int, hi: int):
        self.lo = lo
        self.hi = hi
        self._probs: dict[int, float] = {}  # level -> score
        self._sigma = max(2, (hi - lo) // 6)
        self._activation_threshold = max(self.ACTIVATION_MIN, self._sigma)
        self._geometric_probes = 0
        self._initialized = False

    @property
    def active(self) -> bool:
        """Whether the map has enough data to steer probes."""
        return self._geometric_probes >= self._activation_threshold

    def record_geometric_probe(self) -> None:
        """Increment geometric probe counter (call when GSS uses geometric)."""
        self._geometric_probes += 1

    def _ensure_initialized(self) -> None:
        """Lazy-init scores to 0.5 (neutral)."""
        if not self._initialized:
            self._probs = dict.fromkeys(range(self.lo, self.hi + 1), 0.5)
            self._initialized = True

    def set_scores(self, level_base_times: dict[int, float]) -> None:
        """Set scores from base_time data with Gaussian neighbor spread.

        Pass 1: Measured levels get direct scores (best→1.0, worst→0.0).
        Pass 2: Unmeasured levels get Gaussian-weighted average of nearby
        measured scores, blended with neutral (0.5). Far-away unmeasured
        levels stay near 0.5.

        Args:
            level_base_times: Dict of level -> avg_base_time for levels with data.
        """
        self._ensure_initialized()
        if len(level_base_times) < 2:
            return  # Need 2+ levels for meaningful comparison

        best_bt = min(level_base_times.values())
        worst_bt = max(level_base_times.values())
        bt_spread = worst_bt - best_bt

        # Pass 1: score measured levels directly
        measured_scores: dict[int, float] = {}
        for lvl in level_base_times:
            if bt_spread > 0:
                measured_scores[lvl] = 1.0 - (level_base_times[lvl] - best_bt) / bt_spread
            else:
                measured_scores[lvl] = 0.5
            self._probs[lvl] = measured_scores[lvl]

        # Pass 2: spread to unmeasured neighbors via Gaussian kernel
        sigma = self._sigma
        for lvl in range(self.lo, self.hi + 1):
            if lvl in measured_scores:
                continue  # Already scored directly

            # Gaussian-weighted average of nearby measured scores
            weight_sum = 0.0
            score_sum = 0.0
            for m_lvl, m_score in measured_scores.items():
                dist = abs(lvl - m_lvl)
                w = math.exp(-(dist ** 2) / (2 * sigma ** 2))
                weight_sum += w
                score_sum += w * m_score

            if weight_sum > 0:
                neighbor_score = score_sum / weight_sum
                # Blend with neutral — farther from any measured level = more neutral
                max_weight = max(
                    math.exp(-(abs(lvl - m) ** 2) / (2 * sigma ** 2))
                    for m in measured_scores
                )
                # max_weight is 1.0 when adjacent, decays with distance
                self._probs[lvl] = max_weight * neighbor_score + (1 - max_weight) * 0.5
            else:
                self._probs[lvl] = 0.5

    def get_best_unqualified(
        self, qualified_levels: set[int], bracket_lo: int, bracket_hi: int
    ) -> tuple[int, float] | None:
        """Return highest-probability unqualified level within bracket.

        Returns None if the map is inactive, all levels are qualified,
        or the probability distribution is flat (all within 10%).

        Args:
            qualified_levels: Levels already at EXPLORE_MIN_SAMPLES.
            bracket_lo: Current GSS bracket lower bound.
            bracket_hi: Current GSS bracket upper bound.

        Returns:
            (level, probability) or None if no override needed.
        """
        if not self.active:
            return None

        self._ensure_initialized()
        best_lvl = None
        best_prob = -1.0

        for lvl in range(bracket_lo, bracket_hi + 1):
            if lvl not in qualified_levels and self._probs.get(lvl, 0.5) > best_prob:
                best_prob = self._probs.get(lvl, 0.5)
                best_lvl = lvl

        if best_lvl is None:
            return None

        # Check if the map is "flat" — all unqualified levels within 10% of each other
        unqualified_probs = [
            self._probs.get(lvl, 0.5)
            for lvl in range(bracket_lo, bracket_hi + 1)
            if lvl not in qualified_levels
        ]
        if unqualified_probs and len(unqualified_probs) > 1:
            min_p = min(unqualified_probs)
            max_p = max(unqualified_probs)
            if max_p > 0 and (max_p - min_p) / max_p < 0.10:
                return None  # Map is flat, fall through to geometric

        return (best_lvl, best_prob)

    def get_probabilities(self) -> dict[int, float]:
        """Return current probability map (for display)."""
        self._ensure_initialized()
        return dict(self._probs)


@dataclass
class ExplorationState:
    """Result of exploration orchestration."""

    gss_probe: int | None = None
    effective_peak: int | None = None
    exploration_target: int | None = None
    warmup_override: int | None = None  # Force workers to this level (warmup phases)
    warmup_reason: str | None = None  # Reason string for display
    # Display fields
    gss_lo: int | None = None
    gss_hi: int | None = None
    gss_signal: str | None = None
    prob_map_data: dict[int, float] | None = None
    exploration_status: str | None = None  # Lifecycle status for constant feedback


def compute_effective_base_workers(base_workers: int, total_elements: int) -> int:
    """Cap base_workers so exploration costs at most half the total elements.

    Budget = total_elements / 2. Walk from level 1 to base_workers,
    accumulating the sample cost per level (_min_samples_for_level).
    Stop when the next level would overflow the budget.

    Args:
        base_workers: Original max workers (from config or tier).
        total_elements: Total elements to process in this tier/batch.

    Returns:
        Effective base_workers, capped to fit the exploration budget.
    """
    if base_workers <= 1:
        return base_workers

    budget = int(total_elements * EXPLORE_BUDGET_FRACTION)
    cost = 0
    cap = 0

    for level in range(1, base_workers + 1):
        level_cost = ThroughputByLevel._min_samples_for_level(level)
        if cost + level_cost > budget:
            break
        cost += level_cost
        cap = level

    return max(1, cap)


class ExplorationOrchestrator:
    """Manages GSS-based worker count exploration.

    Lifecycle: warmup A (level 1) → warmup B (high probe) → GSS → converge → monitor peak

    Shared between ThrottleContext (phases 4/7/8) and DependencyTracker (phase 5)
    to avoid duplicating the GSS orchestration logic.

    When total_elements is provided, base_workers is capped so that exploration
    (warmup + GSS) costs at most EXPLORE_BUDGET_FRACTION of total elements.
    The remaining elements run at the discovered peak for actual throughput.
    """

    def __init__(self, base_workers: int, total_elements: int | None = None):
        if total_elements is not None:
            effective = compute_effective_base_workers(base_workers, total_elements)
            if effective < base_workers:
                _log_throttle(
                    f"EXPLORE CAP: {base_workers}→{effective} workers "
                    f"(budget={int(total_elements * EXPLORE_BUDGET_FRACTION)} "
                    f"from {total_elements} elements)"
                )
            self.base_workers = effective
        else:
            self.base_workers = base_workers
        self._original_base_workers = base_workers
        self._total_elements = total_elements
        self._gss: GoldenSectionSearch | None = None
        self._prob_map: ProbabilityMap | None = None

    def reset(self) -> None:
        """Reset on tier change."""
        self._gss = None
        self._prob_map = None

    def orchestrate(
        self,
        all_levels: dict[int, tuple[float, int]],
        legacy_peak: int | None,
        throughput_tracker: ThroughputTracker,
        remaining: int | None = None,
    ) -> ExplorationState:
        """Run one cycle of the exploration state machine.

        Args:
            all_levels: Per-level throughput data: level -> (avg_base_time, sample_count).
            legacy_peak: Peak concurrency from ThroughputTracker (used as fallback).
            throughput_tracker: For legacy exploration target when GSS is not active.
            remaining: Elements left to process (for budget-aware exploration).

        Returns:
            ExplorationState with all fields populated for the caller.
        """
        state = ExplorationState()
        gss_probe = None

        # Skip exploration when budget is too small for meaningful data collection.
        # base_workers=1 means compute_effective_base_workers couldn't fit even
        # the minimum exploration cost — fall through to formula-based throttling.
        if self.base_workers <= 1 and self._total_elements is not None:
            state.exploration_status = f"Skip ({self._total_elements} elem)"
            return state

        # Build level_data: levels with enough samples for GSS decisions.
        level_data = {
            lvl: bt
            for lvl, (bt, cnt) in all_levels.items()
            if cnt >= EXPLORE_MIN_SAMPLES
        }

        # Partial data: levels with some signal but not yet qualified.
        partial_data = {
            lvl: (bt, cnt)
            for lvl, (bt, cnt) in all_levels.items()
            if GSS_PARTIAL_MIN_SAMPLES <= cnt < EXPLORE_MIN_SAMPLES
        }

        # Two-phase warmup before GSS:
        # Phase A: run at level 1 until EXPLORE_MIN_SAMPLES → solid low-end baseline
        # Phase B: run at high probe (~2/3 of max) until EXPLORE_MIN_SAMPLES → high-end reference
        warmup_high_probe = max(2, int(self.base_workers * 2 / 3))
        level1_count = all_levels.get(1, (0, 0))[1] if all_levels else 0
        high_probe_count = (
            all_levels.get(warmup_high_probe, (0, 0))[1] if all_levels else 0
        )

        if (
            self._gss is None
            and level1_count >= EXPLORE_MIN_SAMPLES
            and high_probe_count >= EXPLORE_MIN_SAMPLES
        ):
            self._gss = GoldenSectionSearch(lo=1, hi=self.base_workers)
            self._prob_map = ProbabilityMap(lo=1, hi=self.base_workers)
            _log_throttle(
                f"GSS INIT: bracket=[{self._gss.lo}, {self._gss.hi}] "
                f"(level 1={level1_count}, level {warmup_high_probe}={high_probe_count} samples)"
            )

        # Update probability map: direct score mapping from base_time data.
        if self._prob_map is not None and all_levels:
            level_base_times = {lvl: bt for lvl, (bt, _cnt) in all_levels.items()}
            self._prob_map.set_scores(level_base_times)

        if self._gss is not None and not self._gss.converged:
            gss_probe = self._gss.get_next_probe(
                level_data,
                partial_data=partial_data,
                prob_map=self._prob_map,
            )
            if self._gss.converged:
                # Boundary re-search: if best_level landed at/near the hi edge
                # and there's room above, extend and restart GSS.
                old_hi = self._gss.hi
                if (
                    self._gss.best_level is not None
                    and self._gss.best_level >= old_hi - 1
                    and old_hi < self.base_workers
                ):
                    new_lo = max(1, old_hi - 2)  # overlap slightly
                    _log_throttle(
                        f"GSS BOUNDARY: best={self._gss.best_level} hit hi={old_hi}, "
                        f"extending to [{new_lo}, {self.base_workers}]"
                    )
                    self._gss = GoldenSectionSearch(lo=new_lo, hi=self.base_workers)
                    self._prob_map = ProbabilityMap(lo=new_lo, hi=self.base_workers)
                    gss_probe = self._gss.get_next_probe(
                        level_data,
                        partial_data=partial_data,
                        prob_map=self._prob_map,
                    )
                else:
                    _log_throttle(
                        f"GSS CONVERGED: best_level={self._gss.best_level}"
                    )

        # Post-convergence peak update with hold logic
        if self._gss and self._gss.converged and level_data:
            actual_best = min(level_data, key=level_data.get)  # type: ignore[arg-type]
            if (
                self._gss.best_level is None
                or level_data.get(actual_best, float("inf"))
                < level_data.get(self._gss.best_level, float("inf"))
            ) and actual_best != self._gss.best_level:
                # Check sample count before switching peak
                min_hold = max(5, actual_best // 2)
                best_count = all_levels.get(actual_best, (0, 0))[1]
                if best_count >= min_hold:
                    _log_throttle(
                        f"PEAK UPDATE: {self._gss.best_level}→{actual_best} "
                        f"(bt={level_data[actual_best]:.2f} vs "
                        f"{level_data.get(self._gss.best_level, 0):.2f}, "
                        f"samples={best_count}≥{min_hold})"
                    )
                    self._gss.best_level = actual_best
                else:
                    _log_throttle(
                        f"PEAK HOLD: {actual_best} looks better "
                        f"(bt={level_data[actual_best]:.2f}) but only "
                        f"{best_count}/{min_hold} samples — keeping "
                        f"{self._gss.best_level}"
                    )

        effective_peak = (
            self._gss.best_level if (self._gss and self._gss.converged) else legacy_peak
        )

        # Only use legacy exploration when GSS is not active
        if self._gss is None or self._gss.converged:
            explore = throughput_tracker.get_exploration_target(
                self.base_workers, remaining
            )
        else:
            explore = None  # GSS handles exploration

        # Pre-GSS warmup: force specific levels to collect baseline data.
        if self._gss is None:
            l1_cnt = all_levels.get(1, (0, 0))[1] if all_levels else 0
            if l1_cnt < EXPLORE_MIN_SAMPLES:
                state.warmup_override = 1
                state.warmup_reason = (
                    f"Warmup A: level 1 ({l1_cnt}/{EXPLORE_MIN_SAMPLES})"
                )
                state.exploration_status = (
                    f"WarmA 1 ({l1_cnt}/{EXPLORE_MIN_SAMPLES})"
                )
            else:
                hi_cnt = (
                    all_levels.get(warmup_high_probe, (0, 0))[1]
                    if all_levels
                    else 0
                )
                state.warmup_override = warmup_high_probe
                state.warmup_reason = (
                    f"Warmup B: level {warmup_high_probe} "
                    f"({hi_cnt}/{EXPLORE_MIN_SAMPLES})"
                )
                state.exploration_status = (
                    f"WarmB {warmup_high_probe} ({hi_cnt}/{EXPLORE_MIN_SAMPLES})"
                )

        # Populate state
        state.gss_probe = gss_probe
        state.effective_peak = effective_peak
        state.exploration_target = explore
        if self._gss is not None:
            state.gss_lo = self._gss.lo
            state.gss_hi = self._gss.hi
            state.gss_signal = self._gss.last_signal
            # Set exploration_status for GSS phases
            if self._gss.converged:
                best = self._gss.best_level
                state.exploration_status = f"Peak@{best} \u2713"
            elif gss_probe is not None:
                state.exploration_status = (
                    f"GSS[{self._gss.lo},{self._gss.hi}]\u2192{gss_probe}"
                )
        state.prob_map_data = (
            self._prob_map.get_probabilities() if self._prob_map else None
        )

        return state


def _compute_formula_confidence(
    completion_count: int,
    peak_concurrency: int | None = None,
) -> float:
    """Compute confidence multiplier for formula-based worker ceiling.

    With thin data (few completions), the base_time estimate is unreliable
    and the formula produces aggressive worker counts. Returns a multiplier
    in [FORMULA_CONFIDENCE_MIN, 1.0] that scales down the formula ceiling
    during the early phase.

    Uses a logarithmic curve: rapid initial growth that flattens as we
    approach full confidence. This matches intuition: first few samples
    provide the most information about whether our estimate is reasonable.

    When peak_concurrency is detected (requires >=2 levels with sufficient
    samples each), returns 1.0 — peak data is empirical and more reliable
    than the formula estimate.

    Args:
        completion_count: Number of completions in the tracking window.
        peak_concurrency: If set, peak throughput data is available.

    Returns:
        Confidence multiplier in [FORMULA_CONFIDENCE_MIN, 1.0].
    """
    # Peak detected = empirical data, trust fully
    if peak_concurrency is not None:
        return 1.0

    # Defensive: should not reach formula path with count=0
    if completion_count <= 0:
        return FORMULA_CONFIDENCE_MIN

    if completion_count >= FORMULA_CONFIDENCE_FULL_AT:
        return 1.0

    # Logarithmic ramp: log(1)/log(25)=0, log(5)/log(25)=0.50, log(15)/log(25)=0.84
    t = math.log(completion_count) / math.log(FORMULA_CONFIDENCE_FULL_AT)
    confidence = FORMULA_CONFIDENCE_MIN + (1.0 - FORMULA_CONFIDENCE_MIN) * t

    if confidence >= FORMULA_CONFIDENCE_THRESHOLD:
        return 1.0

    return confidence



def compute_throttle_decision(
    current_max_runtime: float,
    tier_timeout: float,
    base_workers: int,
    active_workers: int,
    throughput: float = 0.0,  # noqa: ARG001
    avg_runtime: float = 0.0,
    completion_count: int = 0,
    avg_concurrency: float = 0.0,  # noqa: ARG001
    avg_base_time: float = 0.0,
    post_warmup: bool = False,
    peak_concurrency: int | None = None,
    exploration_target: int | None = None,
    gss_probe: int | None = None,
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
        gss_probe: Golden section search probe target, or None. When set, overrides
            peak and exploration targeting — jumps directly to the probe level to
            collect data for the search. Capped by the formula safety limit.

    Returns:
        ThrottleDecision with recommended action
    """
    _log_throttle(
        f"DECISION INPUT: active={active_workers} base={base_workers} max_rt={current_max_runtime:.1f}s "
        f"tier_timeout={tier_timeout:.0f}s avg_base={avg_base_time:.3f}s count={completion_count} "
        f"peak={peak_concurrency} explore={exploration_target} post_warmup={post_warmup}"
    )

    # Emergency check uses RAW max_runtime - is any task about to timeout?
    # Use 80% threshold - fires after hold (50%) has had time to stabilize,
    # but before the safety margin (65% effective timeout) is breached.
    raw_ratio = current_max_runtime / tier_timeout if tier_timeout > 0 else 0.0
    if raw_ratio >= 0.80:
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
            reason="Emergency (>80% timeout)",
            completion_count=completion_count,
            is_emergency=True,
        )

    # Normalize max_runtime by current workers for hold threshold comparison
    # This puts it on the same scale as base_time (per-worker cost)
    normalized_max = current_max_runtime / max(active_workers, 1)
    normalized_max / tier_timeout if tier_timeout > 0 else 0.0

    # Check if any task is taking too long - if so, hold at current level
    # Uses raw max (like emergency) but with lower threshold
    hold_threshold = tier_timeout * RAMP_HOLD_THRESHOLD
    should_hold = (
        current_max_runtime > hold_threshold
        and active_workers > 0
    )

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
                is_hold=True,
            )
        elif active_workers > 0:
            # Tasks running fast, safe to ramp +1
            cap = base_workers
            delta = cap - active_workers
            increment = min(MAX_RAMP_INCREMENT, max(0, delta))
            ramped = active_workers + increment
            ramped = min(ramped, cap)
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
            _log_throttle("NO DATA FRESH: starting at 1 worker")
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

    # Apply confidence discount: with thin data, scale down the formula ceiling.
    # Early completions have optimistic base_time (low concurrency = less contention).
    confidence = _compute_formula_confidence(completion_count, peak_concurrency)
    if confidence < 1.0:
        raw_formula = formula_optimal
        formula_optimal = max(1, int(formula_optimal * confidence))
        _log_throttle(
            f"CONFIDENCE: {confidence:.2f} ({completion_count} completions) "
            f"→ formula {raw_formula}→{formula_optimal}"
        )

    formula_optimal = max(1, min(formula_optimal, base_workers))  # Clamp to [1, base_workers]

    # Apply peak throughput optimization: if we have data showing which
    # concurrency level produces the best throughput, use that as the
    # primary target — but never exceed the formula's safety cap.
    # Priority: GSS probe > exploration target > peak concurrency > formula.
    if gss_probe is not None:
        # Golden section search: jump directly to probe level
        optimal = min(gss_probe, formula_optimal)
        optimal = max(1, min(optimal, base_workers))  # Re-clamp
        _log_throttle(
            f"GSS PROBE: target@{gss_probe} formula@{formula_optimal} → {optimal}"
        )
    elif exploration_target is not None:
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
            if gss_probe is not None:
                # GSS: jump directly to probe level (need data there ASAP)
                effective_workers = target_workers
                reason_suffix = f", GSS jump from {active_workers}"
                _log_throttle(
                    f"GSS JUMP: target={target_workers} active={active_workers} "
                    f"→ jumping to {effective_workers}"
                )
            else:
                # Scaling UP - ramp +1 per decision to avoid overwhelming the system
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

    _log_throttle(
        f"DECISION OUTPUT: recommended={effective_workers} throttle={should_throttle} "
        f"target={target_workers} should_hold={should_hold}{reason_suffix}"
    )

    if should_throttle:
        peak_info = f",peak@{peak_concurrency}" if peak_concurrency is not None else ""
        explore_info = f",explore@{exploration_target}" if exploration_target is not None else ""
        gss_info = f",GSS→{gss_probe}" if gss_probe is not None else ""
        return ThrottleDecision(
            should_throttle=True,
            current_max=current_max_runtime,
            historical_max=0,
            completed_avg=effective_base_time,
            recommended_workers=effective_workers,
            reason=f"Throttle ({effective_workers}={int(effective_timeout)}s/{effective_base_time:.1f}s base{peak_info}{explore_info}{gss_info}{reason_suffix})",
            completion_count=completion_count,
            peak_concurrency=peak_concurrency,
            exploration_target=exploration_target,
            gss_probe=gss_probe,
            is_hold=should_hold,
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
            gss_probe=gss_probe,
            is_hold=should_hold,
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


_LEVEL_COL_WIDTH = 10  # Fixed width per level column (chars)
_LEVELS_PER_ROW = 8  # Max levels before wrapping to a new line


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
    prob_map_data: dict[int, float] | None = None,
    prob_min: float = 0.0,
    prob_range: float = 0.0,
) -> object:
    """Build a single Rich Table for a chunk of levels (three rows: number + time + count).

    Row 1 (level number + probability): colored by probability (green=high, red=low).
    Row 2 (base time): colored by base_time performance (green=fast, red=slow).
    Row 3 (sample count): same coloring as row 2.
    Peak level is underlined. Exploration target is marked with "?" prefix.

    Args:
        levels_range: Range of level numbers to display in this row.
        all_levels: Full dict of level -> (avg_base_time_seconds, sample_count).
        min_bt: Minimum base time across ALL levels (for consistent color scaling).
        bt_range: max_bt - min_bt across ALL levels.
        peak_concurrency: The level identified as peak, or None.
        exploration_target: Level being explored (neighbor of peak), or None.
        prob_map_data: Dict of level -> P(best) for probability display, or None.
        prob_min: Minimum probability across ALL levels (for consistent color scaling).
        prob_range: max_prob - min_prob across ALL levels.

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

    row1_cells: list[Text] = [Text("")]  # indent (level number + probability)
    row2_cells: list[Text] = [Text("")]  # indent (base time)
    row3_cells: list[Text] = [Text("")]  # indent (sample count)

    for level in levels_range:
        is_peak = peak_concurrency is not None and level == peak_concurrency
        is_explore = exploration_target is not None and level == exploration_target

        # Build level label with probability
        level_label = f"?{level}" if is_explore else str(level)
        if prob_map_data and level in prob_map_data:
            prob = prob_map_data[level]
            level_label = f"{level_label} .{int(prob * 100):02d}"

        if level in all_levels:
            avg_bt, count = all_levels[level]

            # Row 1 style: colored by probability (high=green, low=red)
            if prob_map_data and level in prob_map_data and prob_range > 0:
                prob = prob_map_data[level]
                # Invert: high probability = green (t=0), low = red (t=1)
                prob_t = 1.0 - ((prob - prob_min) / prob_range)
                prob_color = _lerp_color(prob_t)
                prob_fg = _text_color_for_bg(prob_color)
                prob_style = f"{prob_fg} on {prob_color}"
            else:
                # Fallback: use base_time color for row 1 too
                t = (avg_bt - min_bt) / bt_range if bt_range > 0 else 0.0
                prob_color = _lerp_color(t)
                prob_fg = _text_color_for_bg(prob_color)
                conf = _confidence_style(count)
                parts = [s for s in [conf, prob_fg, f"on {prob_color}"] if s]
                prob_style = " ".join(parts)

            if is_peak:
                prob_style = f"underline {prob_style}"

            level_str = level_label.center(_LEVEL_COL_WIDTH)
            row1_cells.append(Text(level_str, style=prob_style))

            # Rows 2-3 style: colored by base_time (green=fast, red=slow)
            t = (avg_bt - min_bt) / bt_range if bt_range > 0 else 0.0
            color = _lerp_color(t)
            fg = _text_color_for_bg(color)
            bg = f"on {color}"
            conf = _confidence_style(count)
            parts = [s for s in [conf, fg, bg] if s]
            bt_style = " ".join(parts)
            if is_peak:
                bt_style = f"underline {bt_style}"

            bt_str = f"{avg_bt:.1f}s" if avg_bt < 100 else f"{avg_bt:.0f}s"
            bt_padded = bt_str.center(_LEVEL_COL_WIDTH)
            row2_cells.append(Text(bt_padded, style=bt_style))

            count_str = str(count).center(_LEVEL_COL_WIDTH)
            row3_cells.append(Text(count_str, style=bt_style))
        else:
            # No data for this level
            level_str = level_label.center(_LEVEL_COL_WIDTH)

            # Row 1: probability color even without base_time data
            if prob_map_data and level in prob_map_data and prob_range > 0:
                prob = prob_map_data[level]
                prob_t = 1.0 - ((prob - prob_min) / prob_range)
                prob_color = _lerp_color(prob_t)
                prob_fg = _text_color_for_bg(prob_color)
                row1_cells.append(Text(level_str, style=f"{prob_fg} on {prob_color}"))
            else:
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
    prob_map_data: dict[int, float] | None = None,
) -> object:
    """Build color-graded level blocks, wrapping every _LEVELS_PER_ROW levels.

    Each level is a fixed-width column with three rows:
    - Row 1: level number + probability (colored by probability)
    - Row 2: avg base time (colored by base_time)
    - Row 3: sample count (colored by base_time)

    Peak level is underlined. Exploration target is marked with "?" prefix.

    Args:
        all_levels: Dict of level -> (avg_base_time_seconds, sample_count). Must be non-empty.
        peak_concurrency: The level identified as peak, or None.
        max_workers: Total worker slots (1..max_workers). If 0, uses max level in data.
        exploration_target: Level being explored, or None.
        prob_map_data: Dict of level -> P(best) for probability display, or None.

    Returns:
        A Rich renderable (Table for single row, Group of Tables for multiple).
    """
    from rich.console import Group, RenderableType

    max_level = max_workers if max_workers > 0 else max(all_levels.keys())

    # Find min/max base times for color scaling (consistent across all chunks)
    base_times = {lvl: bt for lvl, (bt, _) in all_levels.items()}
    min_bt = min(base_times.values())
    max_bt = max(base_times.values())
    bt_range = max_bt - min_bt

    # Find min/max probability for color scaling (consistent across all chunks)
    p_min = 0.0
    p_range = 0.0
    if prob_map_data:
        all_probs = list(prob_map_data.values())
        if all_probs:
            p_min = min(all_probs)
            p_max = max(all_probs)
            p_range = p_max - p_min

    # Split levels into chunks of _LEVELS_PER_ROW
    chunks: list[range] = []
    for start in range(1, max_level + 1, _LEVELS_PER_ROW):
        end = min(start + _LEVELS_PER_ROW, max_level + 1)
        chunks.append(range(start, end))

    if len(chunks) == 1:
        return _build_levels_row(
            chunks[0], all_levels, min_bt, bt_range, peak_concurrency,
            exploration_target, prob_map_data, p_min, p_range,
        )

    from rich.text import Text

    tables: list[RenderableType] = []
    for i, chunk in enumerate(chunks):
        if i > 0:
            tables.append(Text(""))  # blank line between row groups
        tables.append(
            _build_levels_row(
                chunk, all_levels, min_bt, bt_range, peak_concurrency,
                exploration_target, prob_map_data, p_min, p_range,
            )  # type: ignore[misc]
        )
    return Group(*tables)


def format_throughput_levels(
    all_levels: dict[int, tuple[float, int]] | None,
    peak_concurrency: int | None = None,
    max_workers: int = 0,
    exploration_target: int | None = None,
    prob_map_data: dict[int, float] | None = None,
) -> object | str:
    """Build a color-graded level table (Rich renderable).

    Row 1: level number + probability (colored by probability).
    Row 2: avg base time (colored by base_time).
    Row 3: sample count (colored by base_time).

    Args:
        all_levels: Dict of level -> (avg_base_time_seconds, sample_count), or None.
        peak_concurrency: The level identified as peak, or None.
        max_workers: Total worker slots to display (1..max_workers). If 0, uses max level.
        exploration_target: Level being explored, or None.
        prob_map_data: Dict of level -> P(best) for probability display, or None.

    Returns:
        Rich Table renderable, or empty string if no data.
    """
    if not all_levels:
        return ""

    return _build_levels_table(all_levels, peak_concurrency, max_workers, exploration_target, prob_map_data)


def build_throughput_levels_text(
    all_levels: dict[int, tuple[float, int]] | None,
    peak_concurrency: int | None = None,
    max_workers: int = 0,
    exploration_target: int | None = None,
    prob_map_data: dict[int, float] | None = None,
) -> object | None:
    """Build a color-graded level table (Rich renderable).

    Row 1: level number + probability (colored by probability).
    Row 2: avg base time (colored by base_time).
    Row 3: sample count (colored by base_time).

    Returns None if no data.

    Args:
        all_levels: Dict of level -> (avg_base_time_seconds, sample_count), or None.
        peak_concurrency: The level identified as peak, or None.
        max_workers: Total worker slots to display (1..max_workers). If 0, uses max level.
        exploration_target: Level being explored, or None.
        prob_map_data: Dict of level -> P(best) for probability display, or None.

    Returns:
        A Rich Table renderable, or None if no data.
    """
    if not all_levels:
        return None

    return _build_levels_table(all_levels, peak_concurrency, max_workers, exploration_target, prob_map_data)
