"""Runtime-aware throttling for parallel processing.

Tracks max runtimes of active workers and historical windows to make
intelligent throttling decisions that prevent timeouts.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock

# Throttle log file for debugging
THROTTLE_LOG_PATH = Path("/tmp/magaldi_throttle.log")


def _log_throttle(message: str) -> None:
    """Append a timestamped message to the throttle log."""
    try:
        timestamp = time.strftime("%H:%M:%S")
        with open(THROTTLE_LOG_PATH, "a") as f:
            f.write(f"[{timestamp}] {message}\n")
    except Exception:
        pass  # Don't let logging failures affect throttling

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
MAX_RAMP_INCREMENT = 3


@dataclass
class ThrottleDecision:
    """Result of throttling analysis."""

    should_throttle: bool
    current_max: float  # Max runtime of active workers
    historical_max: float  # Max from historical windows
    completed_avg: float  # Average runtime from completions (used for throttling)
    recommended_workers: int  # Suggested worker count
    reason: str


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

    def record_completion(self, runtime: float, concurrent_workers: int = 1) -> None:
        """Record a completed task with concurrency context.

        Args:
            runtime: Task runtime in seconds
            concurrent_workers: Number of workers active when task completed
        """
        now = time.time()
        base_time = runtime / max(concurrent_workers, 1)
        _log_throttle(
            f"RECORD: wall={runtime:.1f}s workers={concurrent_workers} → base={base_time:.1f}s"
        )
        with self._lock:
            self.completions.append((now, runtime, concurrent_workers))
            # Prune old entries outside window
            cutoff = now - self.window_seconds
            while self.completions and self.completions[0][0] < cutoff:
                self.completions.popleft()

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
            # IMPORTANT: Only include completions where workers > 0 (exclude warmup tasks)
            # Warmup tasks run alone and don't reflect contention behavior
            base_times = [r / c for _, r, c in self.completions if c > 0]
            avg_base_time = sum(base_times) / len(base_times) if base_times else 0.0

            return throughput, avg_runtime, count, avg_concurrency, avg_base_time

    def reset(self) -> None:
        """Clear all history."""
        with self._lock:
            self.completions.clear()


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
        """Format as a log line for /tmp/magaldi_warmup.log."""
        return (
            f"[TIMEOUT] element={self.element_type}:{self.element_id.split(':')[-2]}, "
            f"tier={self.tier}, workers={self.workers_active}, "
            f"avg={self.avg_runtime:.1f}s, max={self.max_runtime:.1f}s, "
            f"limit={self.timeout_limit:.0f}s"
        )


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
) -> ThrottleDecision:
    """Determine if throttling should be applied.

    KEY INSIGHT: Runtime scales linearly with concurrent workers (GPU contention).
    So we use base_time = runtime / workers for throttling decisions.

    Example: 10 workers each taking 10s → base_time = 1s
    If timeout = 7s → max_workers = 7/1 = 7

    Formula: max_workers = (timeout * safety_margin) / base_time

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

    Returns:
        ThrottleDecision with recommended action
    """
    # Emergency check FIRST - if any active task is near timeout, throttle hard
    # This applies even without completion history
    # Use 60% threshold - more aggressive than safety margin (65%) to react before it's too late
    max_ratio = current_max_runtime / tier_timeout if tier_timeout > 0 else 0.0
    if max_ratio >= 0.60:
        _log_throttle(
            f"EMERGENCY: max_runtime={current_max_runtime:.1f}s ({max_ratio:.0%} of {tier_timeout}s timeout) "
            f"active={active_workers} → forcing 1 worker"
        )
        return ThrottleDecision(
            should_throttle=True,
            current_max=current_max_runtime,
            historical_max=0,
            completed_avg=current_max_runtime,
            recommended_workers=1,
            reason="Emergency (>60% timeout)",
        )

    # Use ONLY historical avg_base_time from actual completions.
    # We can't calculate base_time from current running tasks because we don't
    # know how many workers were active when each task STARTED (only current count).
    # Dividing current_max_runtime by active_workers gives wrong results when
    # workers have ramped up since the task started.
    # Need >= 3 completions for reliable avg_base_time.
    if completion_count >= 3 and avg_base_time > 0:
        effective_base_time = avg_base_time
    else:
        # No completion history yet - still apply ramp-up to avoid jumping to max
        # Target is base_workers but we ramp up gradually
        if active_workers > 0:
            delta = base_workers - active_workers
            increment = min(max(1, int(delta * RAMP_UP_FACTOR)), MAX_RAMP_INCREMENT)
            ramped = active_workers + increment
            ramped = min(ramped, base_workers)
            _log_throttle(
                f"NO DATA RAMP: active={active_workers} → ramped to {ramped} (+{increment}, max {MAX_RAMP_INCREMENT})"
            )
            return ThrottleDecision(
                should_throttle=False,
                current_max=current_max_runtime,
                historical_max=0,
                completed_avg=avg_runtime,
                recommended_workers=ramped,
                reason=f"No data, ramped from {active_workers}",
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
            )

    # Calculate optimal workers: (timeout * safety_margin) / base_time
    # Safety margin accounts for variance in task runtimes
    effective_timeout = tier_timeout * THROTTLE_SAFETY_MARGIN
    optimal = int(effective_timeout / effective_base_time)
    optimal = max(1, min(optimal, base_workers))  # Clamp to [1, base_workers]

    # Determine target: either optimal (if throttling) or base_workers (if not)
    target_workers = optimal if optimal < base_workers else base_workers
    should_throttle = optimal < base_workers

    # Apply ramp-up logic: when INCREASING workers, move gradually (25% of gap)
    # When DECREASING, apply immediately (we're seeing slowness, react fast)
    # Only ramp if we have active workers (not starting fresh)
    if target_workers > active_workers and active_workers > 0:
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
            f"active={active_workers} → ramped to {effective_workers} (+{increment}, max {MAX_RAMP_INCREMENT})"
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
            _log_throttle(
                f"FRESH START: base_time={effective_base_time:.1f}s → starting at {effective_workers} workers"
            )

    if should_throttle:
        return ThrottleDecision(
            should_throttle=True,
            current_max=current_max_runtime,
            historical_max=0,
            completed_avg=effective_base_time,
            recommended_workers=effective_workers,
            reason=f"Throttle ({effective_workers}={int(effective_timeout)}s/{effective_base_time:.1f}s base{reason_suffix})",
        )
    else:
        return ThrottleDecision(
            should_throttle=False,
            current_max=current_max_runtime,
            historical_max=0,
            completed_avg=effective_base_time if effective_base_time > 0 else avg_runtime,
            recommended_workers=effective_workers,
            reason=f"Normal{reason_suffix}" if reason_suffix else "Normal",
        )
