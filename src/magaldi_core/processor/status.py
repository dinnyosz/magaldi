"""Worker status and progress tracking for element processing.

Contains dataclasses for tracking worker activity and parallelism statistics.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from shared.throttling import ThrottleDecision

    from .timing import TimingStats


@dataclass
class WorkerStatus:
    """Track what each worker is doing."""

    _lock: threading.Lock = field(default_factory=threading.Lock)
    # worker_id -> (element_name, stage, model, ctx_size, element_start_time)
    # element_start_time is set once when element starts and never reset
    _status: dict[int, tuple[str, str, str, str, float]] = field(default_factory=dict)

    def set(self, worker_id: int, element_name: str, stage: str, model: str = "", ctx_size: str = "", start_time: float = 0.0) -> None:
        """Set worker status. If worker already has an entry, keeps original start_time."""
        with self._lock:
            existing = self._status.get(worker_id)
            if existing is not None:
                # Keep original element start time - don't reset on stage change
                # This is critical for throttling: we want total element time, not stage time
                _, _, _, _, original_start = existing
                self._status[worker_id] = (element_name, stage, model, ctx_size, original_start)
            else:
                self._status[worker_id] = (element_name, stage, model, ctx_size, start_time)

    def clear(self, worker_id: int) -> None:
        with self._lock:
            self._status.pop(worker_id, None)

    def get_all(self) -> dict[int, tuple[str, str, str, str, float]]:
        with self._lock:
            return dict(self._status)

    def get_max_active_runtime(self) -> float:
        """Get the max runtime of currently running workers.

        Returns:
            Maximum runtime in seconds of any active worker, or 0.0 if no workers active.
        """
        now = time.time()
        max_runtime = 0.0
        with self._lock:
            for _, (_, _, _, _, start_time) in self._status.items():
                if start_time > 0:
                    runtime = now - start_time
                    max_runtime = max(max_runtime, runtime)
        return max_runtime

    def get_runtime_stats(self) -> tuple[float, float, int]:
        """Get runtime statistics for active workers.

        Returns:
            Tuple of (max_runtime, avg_runtime, active_count).
        """
        now = time.time()
        runtimes = []
        with self._lock:
            for _, (_, _, _, _, start_time) in self._status.items():
                if start_time > 0:
                    runtimes.append(now - start_time)
        if not runtimes:
            return 0.0, 0.0, 0
        return max(runtimes), sum(runtimes) / len(runtimes), len(runtimes)

    def active_count(self) -> int:
        """Get the number of currently active workers."""
        with self._lock:
            return len(self._status)


@dataclass
class ParallelismStats:
    """Statistics about worker parallelism."""

    max_possible: int  # Total worker pool size
    tier_limit: int    # Current tier's max workers
    running: int       # Workers currently processing
    current_tier: int | None = None  # Current context tier (e.g., 2048, 4096)

    # Throttling info
    throttle_decision: ThrottleDecision | None = None  # Current throttle decision
    tier_changing: bool = False  # True if waiting for tier change

    @property
    def throttled(self) -> int:
        """Workers held back due to tier limits."""
        return self.max_possible - self.tier_limit

    @property
    def idle(self) -> int:
        """Workers within tier limit but waiting for work."""
        return max(0, self.tier_limit - self.running)

    @property
    def effective_limit(self) -> int:
        """Effective worker limit after throttling."""
        if self.throttle_decision and self.throttle_decision.should_throttle:
            return self.throttle_decision.recommended_workers  # type: ignore[no-any-return]
        return self.tier_limit


@dataclass
class ProgressState:
    """Combined state for display updates."""

    total: int  # Elements to process (excludes unchanged and non-AI)
    completed: int  # Elements processed so far
    skipped: int  # Elements unchanged (already processed in previous run)
    failed: int
    timing: TimingStats
    workers: WorkerStatus
    num_workers: int = 1
    recent_errors: list[tuple[str, str]] = field(default_factory=list)  # (element_name, error)
    parallelism: ParallelismStats | None = None
    total_found: int = 0  # All elements found from parsing
    non_ai_skipped: int = 0  # Elements that don't need AI (imports, etc.)
