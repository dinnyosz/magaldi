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


class RuntimeHistory:
    """Tracks last N completion times for throttling decisions.

    Simple ring buffer of recent completion times. Average of last N
    completions is used for adaptive throttling.
    """

    def __init__(self, max_completions: int = 10):
        """Initialize runtime history tracker.

        Args:
            max_completions: Number of recent completions to track (default 10)
        """
        self.completions: deque[float] = deque(maxlen=max_completions)
        self._lock = Lock()

    def record_runtime(self, runtime: float) -> None:
        """Record a completed task's runtime.

        Args:
            runtime: Task runtime in seconds
        """
        with self._lock:
            self.completions.append(runtime)

    def get_historical_max(self) -> float:
        """Get the max runtime from recent completions.

        Returns:
            Maximum runtime from recent completions, or 0.0 if no data
        """
        with self._lock:
            if not self.completions:
                return 0.0
            return max(self.completions)

    def get_historical_avg(self) -> float:
        """Get the average runtime from recent completions.

        Returns:
            Average runtime from recent completions, or 0.0 if no data
        """
        with self._lock:
            if not self.completions:
                return 0.0
            return sum(self.completions) / len(self.completions)

    def get_historical_stats(self) -> tuple[float, float, int]:
        """Get max, average, and count from recent completions.

        Returns:
            Tuple of (max_runtime, avg_runtime, count)
        """
        with self._lock:
            if not self.completions:
                return 0.0, 0.0, 0
            return max(self.completions), sum(self.completions) / len(self.completions), len(self.completions)

    def reset(self) -> None:
        """Clear all history."""
        with self._lock:
            self.completions.clear()


def compute_throttle_decision(
    current_max_runtime: float,
    historical_max_runtime: float,
    tier_timeout: float,
    base_workers: int,
    completed_avg_runtime: float = 0.0,
    completed_count: int = 0,
) -> ThrottleDecision:
    """Determine if throttling should be applied based on completion times.

    Uses a simple adaptive formula:
        safe_workers = base_workers * (target_ratio / avg_ratio)

    Where target_ratio is the desired safety margin (30% of timeout).
    This naturally scales workers based on how fast/slow tasks complete.

    Also considers max runtime for emergency throttling when individual
    tasks approach timeout.

    Args:
        current_max_runtime: Max runtime of currently active workers
        historical_max_runtime: Max from historical 10s windows
        tier_timeout: Timeout for this tier (e.g., 180s for summarize)
        base_workers: Original max workers
        completed_avg_runtime: Average runtime from recent completions
        completed_count: Number of recent completions

    Returns:
        ThrottleDecision with recommended action
    """
    effective_max = max(current_max_runtime, historical_max_runtime)

    # No data yet - normal operation
    if effective_max == 0 and completed_avg_runtime == 0:
        return ThrottleDecision(
            should_throttle=False,
            current_max=0,
            historical_max=0,
            completed_avg=0,
            recommended_workers=max(1, base_workers),
            reason="No data",
        )

    # Target: keep average completion time at 30% of timeout
    # This leaves 70% headroom for spikes
    TARGET_RATIO = 0.30

    # Emergency max check - if any task is near timeout, throttle hard
    max_ratio = effective_max / tier_timeout if effective_max > 0 else 0.0
    if max_ratio >= 0.70:
        # Very close to timeout - absolute minimum
        return ThrottleDecision(
            should_throttle=True,
            current_max=current_max_runtime,
            historical_max=historical_max_runtime,
            completed_avg=completed_avg_runtime,
            recommended_workers=1,
            reason="Emergency (near timeout)",
        )
    elif max_ratio >= 0.50:
        # Getting dangerous - cap at 2
        return ThrottleDecision(
            should_throttle=True,
            current_max=current_max_runtime,
            historical_max=historical_max_runtime,
            completed_avg=completed_avg_runtime,
            recommended_workers=min(2, base_workers),
            reason="Critical (>50% timeout)",
        )

    # Adaptive throttling based on completion average
    if completed_avg_runtime > 0 and completed_count >= 3:
        avg_ratio = completed_avg_runtime / tier_timeout

        # Formula: safe_workers = base * (target / actual)
        # If avg is 60% of timeout and target is 30%, we get 0.5x workers
        # If avg is 15% of timeout and target is 30%, we get 2x workers (capped)
        if avg_ratio > TARGET_RATIO:
            scale = TARGET_RATIO / avg_ratio
            workers = max(1, int(base_workers * scale))

            # Determine severity for display
            if avg_ratio >= 0.50:
                reason = "High (avg >50%)"
            elif avg_ratio >= 0.40:
                reason = "Elevated (avg >40%)"
            else:
                reason = "Moderate (avg >30%)"

            return ThrottleDecision(
                should_throttle=True,
                current_max=current_max_runtime,
                historical_max=historical_max_runtime,
                completed_avg=completed_avg_runtime,
                recommended_workers=workers,
                reason=reason,
            )

    # Normal operation
    return ThrottleDecision(
        should_throttle=False,
        current_max=current_max_runtime,
        historical_max=historical_max_runtime,
        completed_avg=completed_avg_runtime,
        recommended_workers=max(1, base_workers),
        reason="Normal",
    )
