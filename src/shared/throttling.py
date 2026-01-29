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


@dataclass
class ThrottleDecision:
    """Result of throttling analysis."""

    should_throttle: bool
    current_max: float  # Max runtime of active workers
    historical_max: float  # Max from historical windows
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
                if self.current_window_max > 0:
                    self.windows.append(
                        RuntimeWindow(
                            timestamp=self.current_window_start,
                            max_runtime=self.current_window_max,
                        )
                    )
                # Start new window
                self.current_window_start = now
                self.current_window_max = runtime
            else:
                # Update current window max
                self.current_window_max = max(self.current_window_max, runtime)

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

    def reset(self) -> None:
        """Clear all history."""
        with self._lock:
            self.windows.clear()
            self.current_window_start = 0
            self.current_window_max = 0


def compute_throttle_decision(
    current_max_runtime: float,
    historical_max_runtime: float,
    tier_timeout: float,
    base_workers: int,
    avg_runtime: float = 0.0,
    active_count: int = 0,
) -> ThrottleDecision:
    """Determine if throttling should be applied based on runtimes.

    Uses a graduated approach considering both max and average runtimes.
    When multiple workers are running hot (high average), throttling is
    more aggressive.

    Thresholds (based on effective ratio = max_ratio + pressure_boost):
    - >= 60% of timeout: Critical, reduce to 20% workers
    - >= 45% of timeout: High, reduce to 35% workers
    - >= 30% of timeout: Elevated, reduce to 50% workers
    - >= 20% of timeout: Moderate, reduce to 65% workers
    - >= 10% of timeout: Light, reduce to 80% workers
    - < 10%: Normal operation

    Pressure boost: When avg_ratio > 15% and multiple workers active,
    adds up to 20% to effective ratio (more aggressive throttling).

    Always ensures at least 1 worker.

    Args:
        current_max_runtime: Max runtime of currently active workers
        historical_max_runtime: Max from historical 10s windows
        tier_timeout: Timeout for this tier (e.g., 180s for summarize)
        base_workers: Original max workers
        avg_runtime: Average runtime of active workers (optional)
        active_count: Number of active workers (optional)

    Returns:
        ThrottleDecision with recommended action
    """
    # Use the higher of current or historical max
    effective_max = max(current_max_runtime, historical_max_runtime)

    # No data yet - normal operation
    if effective_max == 0:
        return ThrottleDecision(
            should_throttle=False,
            current_max=0,
            historical_max=0,
            recommended_workers=max(1, base_workers),
            reason="No data",
        )

    max_ratio = effective_max / tier_timeout
    avg_ratio = avg_runtime / tier_timeout if avg_runtime > 0 else 0.0

    # Calculate pressure boost based on average and worker count
    # If many workers are running hot (avg > 15% of timeout), increase effective ratio
    pressure_boost = 0.0
    if avg_ratio > 0.15 and active_count >= 3:
        # Scale boost by how hot the average is and how many workers
        # Max boost is 0.20 (20% added to ratio)
        avg_factor = min(1.0, (avg_ratio - 0.15) / 0.35)  # 0 at 15%, 1 at 50%
        count_factor = min(1.0, active_count / 10)  # 0 at 0, 1 at 10+ workers
        pressure_boost = 0.20 * avg_factor * count_factor

    effective_ratio = max_ratio + pressure_boost

    if effective_ratio >= 0.70:  # >= 70% of timeout
        # Extreme: absolute minimum of 1 worker
        return ThrottleDecision(
            should_throttle=True,
            current_max=current_max_runtime,
            historical_max=historical_max_runtime,
            recommended_workers=1,
            reason="Extreme throttling",
        )
    elif effective_ratio >= 0.55:  # >= 55% of timeout
        # Critical: max 2 workers
        workers = min(2, max(1, int(base_workers * 0.10)))
        return ThrottleDecision(
            should_throttle=True,
            current_max=current_max_runtime,
            historical_max=historical_max_runtime,
            recommended_workers=workers,
            reason="Critical throttling",
        )
    elif effective_ratio >= 0.40:  # >= 40% of timeout
        # High: max 4 workers or 15%
        workers = min(4, max(1, int(base_workers * 0.15)))
        return ThrottleDecision(
            should_throttle=True,
            current_max=current_max_runtime,
            historical_max=historical_max_runtime,
            recommended_workers=workers,
            reason="High throttling",
        )
    elif effective_ratio >= 0.30:  # >= 30% of timeout
        # Elevated: max 8 workers or 25%
        workers = min(8, max(1, int(base_workers * 0.25)))
        return ThrottleDecision(
            should_throttle=True,
            current_max=current_max_runtime,
            historical_max=historical_max_runtime,
            recommended_workers=workers,
            reason="Elevated throttling",
        )
    elif effective_ratio >= 0.20:  # >= 20% of timeout
        # Moderate: 40% workers
        workers = max(1, int(base_workers * 0.40))
        return ThrottleDecision(
            should_throttle=True,
            current_max=current_max_runtime,
            historical_max=historical_max_runtime,
            recommended_workers=workers,
            reason="Moderate throttling",
        )
    elif effective_ratio >= 0.10:  # >= 10% of timeout
        # Light: 60% workers
        workers = max(1, int(base_workers * 0.60))
        return ThrottleDecision(
            should_throttle=True,
            current_max=current_max_runtime,
            historical_max=historical_max_runtime,
            recommended_workers=workers,
            reason="Light throttling",
        )
    else:
        return ThrottleDecision(
            should_throttle=False,
            current_max=current_max_runtime,
            historical_max=historical_max_runtime,
            recommended_workers=max(1, base_workers),
            reason="Normal",
        )
