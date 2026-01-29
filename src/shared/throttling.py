"""Runtime-aware throttling for parallel processing.

Tracks max runtimes of active workers and historical windows to make
intelligent throttling decisions that prevent timeouts.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from threading import Lock

# Safety margin for throttling - use 70% of timeout to leave headroom
# for variance in task runtimes when running concurrently
THROTTLE_SAFETY_MARGIN = 0.7


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
            base_times = [r / max(c, 1) for _, r, c in self.completions]
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
    max_ratio = current_max_runtime / tier_timeout if tier_timeout > 0 else 0.0
    if max_ratio >= 0.80:
        return ThrottleDecision(
            should_throttle=True,
            current_max=current_max_runtime,
            historical_max=0,
            completed_avg=current_max_runtime,
            recommended_workers=1,
            reason="Emergency (>80% timeout)",
        )

    # Calculate base_time from current running tasks (normalized by concurrency)
    # base_time = runtime / workers - this is the per-worker cost
    current_base_time = 0.0
    if current_max_runtime > 0 and active_workers > 0:
        current_base_time = current_max_runtime / active_workers

    # Use the worst case: max of current base_time and historical avg_base_time
    # This ensures we react to slow tasks immediately (proactive) while also
    # learning from historical data
    if current_base_time > 0 and avg_base_time > 0:
        effective_base_time = max(current_base_time, avg_base_time)
    elif current_base_time > 0:
        effective_base_time = current_base_time
    elif completion_count >= 3 and avg_base_time > 0:
        effective_base_time = avg_base_time
    else:
        # No running tasks, no completion history - can't throttle yet
        return ThrottleDecision(
            should_throttle=False,
            current_max=current_max_runtime,
            historical_max=0,
            completed_avg=avg_runtime,
            recommended_workers=base_workers,
            reason="No data",
        )

    # Calculate optimal workers: (timeout * safety_margin) / base_time
    # Safety margin accounts for variance in task runtimes
    effective_timeout = tier_timeout * THROTTLE_SAFETY_MARGIN
    optimal = int(effective_timeout / effective_base_time)
    optimal = max(1, min(optimal, base_workers))  # Clamp to [1, base_workers]

    if optimal < base_workers:
        return ThrottleDecision(
            should_throttle=True,
            current_max=current_max_runtime,
            historical_max=0,
            completed_avg=effective_base_time,  # Now stores base_time for display
            recommended_workers=optimal,
            reason=f"Throttle ({optimal}={int(effective_timeout)}s/{effective_base_time:.1f}s base)",
        )

    # Normal operation - base_time is low enough to allow full workers
    return ThrottleDecision(
        should_throttle=False,
        current_max=current_max_runtime,
        historical_max=0,
        completed_avg=effective_base_time if effective_base_time > 0 else avg_runtime,
        recommended_workers=base_workers,
        reason="Normal",
    )
