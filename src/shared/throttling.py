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
        """Get throughput statistics with concurrency context.

        Returns:
            Tuple of (throughput, avg_runtime, count, avg_concurrency, high_load_avg_runtime)
            - avg_concurrency: average workers active at completion time
            - high_load_avg_runtime: avg runtime of tasks that completed under high load
              (>= 50% of max observed concurrency). Returns 0 if no high-load data.
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

            # Calculate avg runtime under high load (>= 50% of max concurrency)
            max_concurrency = max(c for _, _, c in self.completions)
            high_load_threshold = max(1, max_concurrency // 2)
            high_load_runtimes = [r for _, r, c in self.completions if c >= high_load_threshold]
            high_load_avg = sum(high_load_runtimes) / len(high_load_runtimes) if high_load_runtimes else 0.0

            return throughput, avg_runtime, count, avg_concurrency, high_load_avg

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
    high_load_avg_runtime: float = 0.0,
) -> ThrottleDecision:
    """Determine if throttling should be applied.

    Uses the formula: max_workers = timeout / avg_runtime
    This ensures all concurrent tasks can complete before timeout.

    IMPORTANT: Uses high_load_avg_runtime (avg runtime when many workers were active)
    instead of overall avg_runtime. This matters because:
    - A 50s task with 8 workers indicates GPU contention → throttle
    - A 50s task with 1 worker indicates slow task → don't throttle

    Also uses emergency throttling if any active task is near timeout.

    Args:
        current_max_runtime: Max runtime of currently active workers
        tier_timeout: Timeout for this tier (e.g., 180s for summarize)
        base_workers: Original max workers
        active_workers: Currently active worker count
        throughput: Actual completions per second (for display)
        avg_runtime: Average completion time (all conditions)
        completion_count: Number of completions in window
        avg_concurrency: Average workers active at completion time
        high_load_avg_runtime: Avg runtime of tasks completed under high load

    Returns:
        ThrottleDecision with recommended action
    """
    # No data yet - normal operation
    if completion_count < 3:
        return ThrottleDecision(
            should_throttle=False,
            current_max=current_max_runtime,
            historical_max=0,
            completed_avg=avg_runtime,
            recommended_workers=base_workers,
            reason="No data",
        )

    # Emergency check - if any active task is near timeout, throttle hard
    max_ratio = current_max_runtime / tier_timeout if tier_timeout > 0 else 0.0
    if max_ratio >= 0.80:
        return ThrottleDecision(
            should_throttle=True,
            current_max=current_max_runtime,
            historical_max=0,
            completed_avg=avg_runtime,
            recommended_workers=1,
            reason="Emergency (>80% timeout)",
        )

    # Use high-load avg runtime for throttling if available, otherwise fall back to overall avg
    # This ensures we throttle based on what happens under load, not during ramp-up
    effective_avg = high_load_avg_runtime if high_load_avg_runtime > 0 else avg_runtime

    # Calculate optimal workers: (timeout * safety_margin) / effective_avg
    # Safety margin accounts for variance in task runtimes under concurrency
    if effective_avg > 0:
        effective_timeout = tier_timeout * THROTTLE_SAFETY_MARGIN
        optimal = int(effective_timeout / effective_avg)
        optimal = max(1, min(optimal, base_workers))  # Clamp to [1, base_workers]

        if optimal < base_workers:
            return ThrottleDecision(
                should_throttle=True,
                current_max=current_max_runtime,
                historical_max=0,
                completed_avg=effective_avg,
                recommended_workers=optimal,
                reason=f"Throttle ({optimal}={int(effective_timeout)}s/{effective_avg:.0f}s)",
            )

    # Normal operation - avg is low enough to allow full workers
    return ThrottleDecision(
        should_throttle=False,
        current_max=current_max_runtime,
        historical_max=0,
        completed_avg=avg_runtime,
        recommended_workers=base_workers,
        reason="Normal",
    )
