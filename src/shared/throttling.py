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
) -> ThrottleDecision:
    """Determine if throttling should be applied based on runtimes.

    Uses a graduated approach with more granular levels:
    - >= 80% of timeout: Critical, reduce to 25% workers
    - >= 65% of timeout: High, reduce to 40% workers
    - >= 50% of timeout: Elevated, reduce to 55% workers
    - >= 35% of timeout: Moderate, reduce to 70% workers
    - >= 20% of timeout: Light, reduce to 85% workers
    - < 20%: Normal operation

    Always ensures at least 1 worker.

    Args:
        current_max_runtime: Max runtime of currently active workers
        historical_max_runtime: Max from historical 10s windows
        tier_timeout: Timeout for this tier (e.g., 180s for summarize)
        base_workers: Original max workers

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

    ratio = effective_max / tier_timeout

    if ratio >= 0.8:  # >= 80% of timeout
        # Critical: reduce to 25% workers
        workers = max(1, int(base_workers * 0.25))
        return ThrottleDecision(
            should_throttle=True,
            current_max=current_max_runtime,
            historical_max=historical_max_runtime,
            recommended_workers=workers,
            reason="Critical throttling",
        )
    elif ratio >= 0.65:  # >= 65% of timeout
        # High: reduce to 40% workers
        workers = max(1, int(base_workers * 0.40))
        return ThrottleDecision(
            should_throttle=True,
            current_max=current_max_runtime,
            historical_max=historical_max_runtime,
            recommended_workers=workers,
            reason="High throttling",
        )
    elif ratio >= 0.50:  # >= 50% of timeout
        # Elevated: reduce to 55% workers
        workers = max(1, int(base_workers * 0.55))
        return ThrottleDecision(
            should_throttle=True,
            current_max=current_max_runtime,
            historical_max=historical_max_runtime,
            recommended_workers=workers,
            reason="Elevated throttling",
        )
    elif ratio >= 0.35:  # >= 35% of timeout
        # Moderate: reduce to 70% workers
        workers = max(1, int(base_workers * 0.70))
        return ThrottleDecision(
            should_throttle=True,
            current_max=current_max_runtime,
            historical_max=historical_max_runtime,
            recommended_workers=workers,
            reason="Moderate throttling",
        )
    elif ratio >= 0.20:  # >= 20% of timeout
        # Light: reduce to 85% workers
        workers = max(1, int(base_workers * 0.85))
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
