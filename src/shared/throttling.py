"""Runtime-aware throttling for parallel processing.

Tracks max runtimes of active workers and historical windows to make
intelligent throttling decisions that prevent timeouts.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from threading import Lock


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
    """Tracks completion throughput for throttling decisions.

    Measures actual completions per second over a sliding window
    to detect when the system is saturated (more workers doesn't
    increase throughput).
    """

    def __init__(self, window_seconds: float = 10.0):
        """Initialize throughput tracker.

        Args:
            window_seconds: Time window for measuring throughput (default 10s)
        """
        self.window_seconds = window_seconds
        # Store (timestamp, runtime) for each completion
        self.completions: deque[tuple[float, float]] = deque()
        self._lock = Lock()

    def record_completion(self, runtime: float) -> None:
        """Record a completed task.

        Args:
            runtime: Task runtime in seconds
        """
        now = time.time()
        with self._lock:
            self.completions.append((now, runtime))
            # Prune old entries outside window
            cutoff = now - self.window_seconds
            while self.completions and self.completions[0][0] < cutoff:
                self.completions.popleft()

    def get_stats(self) -> tuple[float, float, int]:
        """Get throughput statistics.

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
            total_runtime = sum(r for _, r in self.completions)
            avg_runtime = total_runtime / count

            # Calculate actual time span of completions
            oldest = self.completions[0][0]
            time_span = now - oldest
            if time_span < 1.0:
                time_span = 1.0  # Avoid division issues for very short spans

            throughput = count / time_span
            return throughput, avg_runtime, count

    def reset(self) -> None:
        """Clear all history."""
        with self._lock:
            self.completions.clear()


# Keep old name as alias for compatibility
RuntimeHistory = ThroughputTracker


def compute_throttle_decision(
    current_max_runtime: float,
    tier_timeout: float,
    base_workers: int,
    active_workers: int,
    throughput: float = 0.0,
    avg_runtime: float = 0.0,
    completion_count: int = 0,
) -> ThrottleDecision:
    """Determine if throttling should be applied based on throughput.

    Compares actual throughput vs expected throughput:
        expected = active_workers / avg_runtime
        actual = completions / time_window

    If actual < expected * 0.7, system is saturated - reduce workers.
    Optimal workers ≈ actual_throughput * avg_runtime

    Also uses emergency throttling if any task approaches timeout.

    Args:
        current_max_runtime: Max runtime of currently active workers
        tier_timeout: Timeout for this tier (e.g., 180s for summarize)
        base_workers: Original max workers
        active_workers: Currently active worker count
        throughput: Actual completions per second
        avg_runtime: Average completion time
        completion_count: Number of completions in window

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

    # Emergency check - if any task is near timeout, throttle hard
    max_ratio = current_max_runtime / tier_timeout if current_max_runtime > 0 else 0.0
    if max_ratio >= 0.70:
        return ThrottleDecision(
            should_throttle=True,
            current_max=current_max_runtime,
            historical_max=0,
            completed_avg=avg_runtime,
            recommended_workers=1,
            reason="Emergency (near timeout)",
        )
    elif max_ratio >= 0.50:
        return ThrottleDecision(
            should_throttle=True,
            current_max=current_max_runtime,
            historical_max=0,
            completed_avg=avg_runtime,
            recommended_workers=min(2, base_workers),
            reason="Critical (>50% timeout)",
        )

    # Throughput-based throttling
    if avg_runtime > 0 and active_workers > 0:
        # Expected throughput if system scales linearly
        expected_throughput = active_workers / avg_runtime

        # If actual throughput is much lower than expected, system is saturated
        if expected_throughput > 0 and throughput < expected_throughput * 0.7:
            # Optimal workers = throughput * avg_runtime
            # This is roughly how many workers the system can actually handle
            optimal = max(1, int(throughput * avg_runtime * 1.2))  # 20% headroom
            optimal = min(optimal, base_workers)  # Don't exceed base

            efficiency = (throughput / expected_throughput) * 100
            return ThrottleDecision(
                should_throttle=True,
                current_max=current_max_runtime,
                historical_max=0,
                completed_avg=avg_runtime,
                recommended_workers=optimal,
                reason=f"Saturated ({efficiency:.0f}% eff)",
            )

    # Normal operation
    return ThrottleDecision(
        should_throttle=False,
        current_max=current_max_runtime,
        historical_max=0,
        completed_avg=avg_runtime,
        recommended_workers=base_workers,
        reason="Normal",
    )
