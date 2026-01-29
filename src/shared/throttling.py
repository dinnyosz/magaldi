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
class RuntimeWindow:
    """A time window of runtime data."""

    timestamp: float  # Window start time
    max_runtime: float  # Max runtime observed in this window
    total_runtime: float = 0.0  # Sum of all runtimes in this window
    count: int = 0  # Number of completions in this window


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
    """Tracks historical max runtimes for throttling decisions.

    Maintains a sliding window of runtime observations to detect
    sustained slow performance and prevent "false recovery" when
    a slow task completes but similar tasks remain slow.
    """

    def __init__(self, window_seconds: float = 10.0, history_windows: int = 6):
        """Initialize runtime history tracker.

        Args:
            window_seconds: Duration of each window (default 10s)
            history_windows: Number of windows to keep (default 6 = 60s total)
        """
        self.window_seconds = window_seconds
        self.history_windows = history_windows
        self.windows: deque[RuntimeWindow] = deque(maxlen=history_windows)
        self.current_window_start: float = 0
        self.current_window_max: float = 0
        self.current_window_total: float = 0
        self.current_window_count: int = 0
        self._lock = Lock()

    def record_runtime(self, runtime: float) -> None:
        """Record a completed task's runtime.

        Args:
            runtime: Task runtime in seconds
        """
        now = time.time()
        with self._lock:
            # Initialize window start if first record
            if self.current_window_start == 0:
                self.current_window_start = now

            # Check if we need to rotate to a new window
            if now - self.current_window_start >= self.window_seconds:
                # Save current window if it has data
                if self.current_window_count > 0:
                    self.windows.append(
                        RuntimeWindow(
                            timestamp=self.current_window_start,
                            max_runtime=self.current_window_max,
                            total_runtime=self.current_window_total,
                            count=self.current_window_count,
                        )
                    )
                # Start new window
                self.current_window_start = now
                self.current_window_max = runtime
                self.current_window_total = runtime
                self.current_window_count = 1
            else:
                # Update current window stats
                self.current_window_max = max(self.current_window_max, runtime)
                self.current_window_total += runtime
                self.current_window_count += 1

    def get_historical_max(self) -> float:
        """Get the max runtime across all historical windows.

        Returns:
            Maximum runtime from history, or 0.0 if no data
        """
        with self._lock:
            if not self.windows and self.current_window_max == 0:
                return 0.0
            historical_max = max((w.max_runtime for w in self.windows), default=0)
            return max(historical_max, self.current_window_max)

    def get_historical_avg(self) -> float:
        """Get the average runtime across all historical windows.

        Returns:
            Average runtime from history, or 0.0 if no data
        """
        with self._lock:
            total = self.current_window_total
            count = self.current_window_count
            for w in self.windows:
                total += w.total_runtime
                count += w.count
            if count == 0:
                return 0.0
            return total / count

    def get_historical_stats(self) -> tuple[float, float, int]:
        """Get max, average, and count from historical windows.

        Returns:
            Tuple of (max_runtime, avg_runtime, total_count)
        """
        with self._lock:
            if not self.windows and self.current_window_count == 0:
                return 0.0, 0.0, 0

            historical_max = max((w.max_runtime for w in self.windows), default=0)
            max_runtime = max(historical_max, self.current_window_max)

            total = self.current_window_total
            count = self.current_window_count
            for w in self.windows:
                total += w.total_runtime
                count += w.count

            avg_runtime = total / count if count > 0 else 0.0
            return max_runtime, avg_runtime, count

    def reset(self) -> None:
        """Clear all history."""
        with self._lock:
            self.windows.clear()
            self.current_window_start = 0
            self.current_window_max = 0
            self.current_window_total = 0
            self.current_window_count = 0


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
