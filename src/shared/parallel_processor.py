"""Generic throttled parallel processor with tier-based batching.

This module provides a reusable parallel processing engine that handles:
- Tier-based batching (grouping items by context size)
- Dynamic throttling based on runtime
- Gradual ramp-up and emergency throttle
- Progress tracking and display refresh

Used by: element summarization, feature processing, subfeature processing, glossary extraction.

Two APIs are provided:
1. ThrottledParallelProcessor - full class-based processor with built-in stats
2. run_throttled_tier() - lightweight function for integration with existing code
"""

from __future__ import annotations

import bisect
import threading
import time
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Generic, TypeVar

from shared.ai.context_size import TIER_MAX_WORKERS, TIER_TIMEOUTS, iter_by_tier
from shared.throttling import (
    ThrottleDecision,
    ThroughputTracker,
    compute_throttle_decision,
    get_ramp_cooldown,
)

if TYPE_CHECKING:
    pass

# Type variables for generic processing
T = TypeVar("T")  # Input item type
R = TypeVar("R")  # Result type


@dataclass
class ProcessingStats:
    """Thread-safe processing statistics with throttling support."""

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    start_time: float = 0.0
    completed: int = 0
    failed: int = 0
    throughput_tracker: ThroughputTracker = field(
        default_factory=lambda: ThroughputTracker(window_seconds=300.0)
    )

    # Per-tier tracking
    time_by_tier: dict[int, float] = field(default_factory=dict)
    count_by_tier: dict[int, int] = field(default_factory=dict)
    totals_by_tier: dict[int, int] = field(default_factory=dict)

    def start(self) -> None:
        """Mark processing start time."""
        self.start_time = time.time()

    @property
    def elapsed(self) -> float:
        """Elapsed wall time since start."""
        if self.start_time == 0:
            return 0.0
        return time.time() - self.start_time

    def set_totals_by_tier(self, totals: dict[int, int]) -> None:
        """Set total counts by tier for ETA calculation."""
        with self._lock:
            self.totals_by_tier = dict(totals)

    def record_completion(
        self, runtime: float, avg_workers: float, tier: int, success: bool
    ) -> None:
        """Record a completed task."""
        with self._lock:
            if success:
                self.completed += 1
            else:
                self.failed += 1

            # Record for throttling
            self.throughput_tracker.record_completion(runtime, avg_workers)

            # Track per-tier timing
            if tier > 0:
                self.time_by_tier[tier] = self.time_by_tier.get(tier, 0.0) + runtime
                self.count_by_tier[tier] = self.count_by_tier.get(tier, 0) + 1

    def get_throttle_stats(self) -> tuple[float, float, int, float, float]:
        """Get stats for throttle decision."""
        return self.throughput_tracker.get_stats_with_concurrency()


@dataclass
class WorkerStatus:
    """Thread-safe worker status tracking."""

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _status: dict[int, tuple[str, str, str, float]] = field(default_factory=dict)
    # worker_id -> (item_name, model, ctx_size, start_time)

    def set(self, worker_id: int, item_name: str, model: str, ctx_size: str) -> None:
        """Set worker status."""
        with self._lock:
            self._status[worker_id] = (item_name, model, ctx_size, time.time())

    def clear(self, worker_id: int) -> None:
        """Clear worker status."""
        with self._lock:
            self._status.pop(worker_id, None)

    def get_all(self) -> dict[int, tuple[str, str, str]]:
        """Get all worker statuses (without start_time for display)."""
        with self._lock:
            return {k: (v[0], v[1], v[2]) for k, v in self._status.items()}

    def active_count(self) -> int:
        """Get number of active workers."""
        with self._lock:
            return len(self._status)

    def get_max_active_runtime(self) -> float:
        """Get maximum runtime of currently active workers."""
        with self._lock:
            if not self._status:
                return 0.0
            now = time.time()
            return max(now - v[3] for v in self._status.values())


@dataclass
class ProcessingResult(Generic[R]):
    """Result of parallel processing."""

    completed: int = 0
    failed: int = 0
    results: list[R] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    elapsed: float = 0.0


@dataclass
class ProgressInfo:
    """Progress information passed to callbacks."""

    total: int
    completed: int
    failed: int
    elapsed: float
    stats: ProcessingStats
    workers: WorkerStatus
    num_workers: int
    tier: int
    throttle_allowed: int


class ThrottledParallelProcessor(Generic[T, R]):
    """Generic throttled parallel processor with tier-based batching.

    Usage:
        processor = ThrottledParallelProcessor(
            max_workers=8,
            tier_fn=lambda item: compute_tier(item),
            process_fn=lambda item, worker_id, set_status: process(item),
            on_progress=lambda info: update_display(info),
        )
        result = processor.process(items)
    """

    def __init__(
        self,
        max_workers: int,
        tier_fn: Callable[[T], int],
        process_fn: Callable[[T, int, Callable[[str, str, str], None]], R],
        on_progress: Callable[[ProgressInfo], None] | None = None,
        on_result: Callable[[T, R, bool], None] | None = None,
        get_runtime: Callable[[R], float] | None = None,
        is_success: Callable[[R], bool] | None = None,
        get_error: Callable[[R], str | None] | None = None,
    ):
        """Initialize the processor.

        Args:
            max_workers: Maximum workers (0 = auto based on tier limits)
            tier_fn: Function to get tier for an item
            process_fn: Function to process an item. Receives (item, worker_id, set_status_fn).
                        set_status_fn takes (item_name, model, ctx_size).
            on_progress: Called on each progress update
            on_result: Called when an item completes (item, result, success)
            get_runtime: Extract runtime from result (default: assumes result.wall_time)
            is_success: Check if result is successful (default: assumes result.success)
            get_error: Get error message from result (default: assumes result.error)
        """
        self.max_workers = max_workers if max_workers > 0 else max(TIER_MAX_WORKERS.values())
        self.tier_fn = tier_fn
        self.process_fn = process_fn
        self.on_progress = on_progress
        self.on_result = on_result

        # Default accessors for common result patterns
        self.get_runtime = get_runtime or (lambda r: getattr(r, "wall_time", 0.0))
        self.is_success = is_success or (lambda r: getattr(r, "success", True))
        self.get_error = get_error or (lambda r: getattr(r, "error", None))

        self.stats = ProcessingStats()
        self.workers = WorkerStatus()

        # Worker ID pool
        self._worker_id_lock = threading.Lock()
        self._available_worker_ids: list[int] = []

    def _acquire_worker_id(self) -> int:
        """Get an available worker ID (lowest first)."""
        with self._worker_id_lock:
            if self._available_worker_ids:
                # Pop from front to prefer lower IDs
                return self._available_worker_ids.pop(0)
            return 0

    def _release_worker_id(self, worker_id: int) -> None:
        """Return a worker ID to the pool (maintains sorted order)."""
        with self._worker_id_lock:
            if worker_id not in self._available_worker_ids:
                # Insert in sorted position to keep lower IDs at front
                bisect.insort(self._available_worker_ids, worker_id)

    def process(self, items: list[T]) -> ProcessingResult[R]:
        """Process items with throttled parallelism.

        Args:
            items: Items to process

        Returns:
            ProcessingResult with completed count, failures, and results
        """
        if not items:
            return ProcessingResult()

        self.stats.start()
        total = len(items)
        results: list[R] = []
        errors: list[str] = []
        results_lock = threading.Lock()

        # Group items by tier
        tier_groups = list(iter_by_tier(items, self.tier_fn))

        # Build tier counts for ETA
        tier_counts = {tier: len(tier_items) for tier, _, tier_items in tier_groups}
        self.stats.set_totals_by_tier(tier_counts)

        # Re-iterate (iter_by_tier is a generator)
        tier_groups = list(iter_by_tier(items, self.tier_fn))

        def make_set_status(worker_id: int) -> Callable[[str, str, str], None]:
            """Create a status setter for a worker."""
            def set_status(item_name: str, model: str, ctx_size: str) -> None:
                self.workers.set(worker_id, item_name, model, ctx_size)
            return set_status

        def process_wrapper(item: T) -> tuple[T, R, int]:
            """Wrapper that handles worker ID and status."""
            worker_id = self._acquire_worker_id()
            try:
                result = self.process_fn(item, worker_id, make_set_status(worker_id))
                return item, result, worker_id
            finally:
                self.workers.clear(worker_id)
                self._release_worker_id(worker_id)

        try:
            for tier, tier_max_workers, tier_items in tier_groups:
                effective_workers = min(self.max_workers, tier_max_workers)
                tier_timeout = TIER_TIMEOUTS.get(tier, 180)

                # Reset worker ID pool for this tier
                with self._worker_id_lock:
                    self._available_worker_ids.clear()
                    self._available_worker_ids.extend(range(effective_workers))

                executor = ThreadPoolExecutor(max_workers=effective_workers)
                futures: dict = {}
                future_to_allowed_at_start: dict = {}
                pending_items = list(tier_items)

                try:
                    while pending_items or futures:
                        # Get throttle decision
                        active_workers = len(futures)
                        current_max = self.workers.get_max_active_runtime()
                        throughput, avg_runtime, count, avg_conc, avg_base = (
                            self.stats.get_throttle_stats()
                        )
                        throttle = compute_throttle_decision(
                            current_max_runtime=current_max,
                            tier_timeout=tier_timeout,
                            base_workers=effective_workers,
                            active_workers=active_workers,
                            throughput=throughput,
                            avg_runtime=avg_runtime,
                            completion_count=count,
                            avg_concurrency=avg_conc,
                            avg_base_time=avg_base,
                        )
                        allowed_workers = throttle.recommended_workers

                        # Submit new tasks up to allowed limit
                        while pending_items and len(futures) < allowed_workers:
                            item = pending_items.pop(0)
                            future = executor.submit(process_wrapper, item)
                            futures[future] = item
                            future_to_allowed_at_start[future] = allowed_workers

                        if not futures:
                            break

                        # Wait for completion or timeout for display refresh
                        done, _ = wait(
                            futures.keys(), timeout=2.0, return_when=FIRST_COMPLETED
                        )

                        # Handle completed tasks
                        for future in done:
                            item = futures.pop(future)
                            # Use allowed workers (average of start and end)
                            allowed_at_start = future_to_allowed_at_start.pop(future, allowed_workers)
                            allowed_at_end = allowed_workers
                            avg_workers = min(allowed_at_start, allowed_at_end)

                            item, result, worker_id = future.result()

                            # Extract result info
                            runtime = self.get_runtime(result)
                            success = self.is_success(result)
                            error = self.get_error(result)

                            # Record completion
                            self.stats.record_completion(
                                runtime, avg_workers, tier, success
                            )

                            with results_lock:
                                results.append(result)
                                if not success and error:
                                    errors.append(error)

                            # Callback for custom handling
                            if self.on_result:
                                self.on_result(item, result, success)

                        # Update progress
                        if self.on_progress:
                            info = ProgressInfo(
                                total=total,
                                completed=self.stats.completed,
                                failed=self.stats.failed,
                                elapsed=self.stats.elapsed,
                                stats=self.stats,
                                workers=self.workers,
                                num_workers=effective_workers,
                                tier=tier,
                                throttle_allowed=allowed_workers,
                            )
                            self.on_progress(info)

                finally:
                    executor.shutdown(wait=True)

        except KeyboardInterrupt:
            # Clear worker status on interrupt
            for wid in range(self.max_workers):
                self.workers.clear(wid)
            raise

        return ProcessingResult(
            completed=self.stats.completed,
            failed=self.stats.failed,
            results=results,
            errors=errors,
            elapsed=self.stats.elapsed,
        )


# =============================================================================
# LIGHTWEIGHT FUNCTION API
# =============================================================================
# For easier integration with existing code that has custom stats/worker classes


@dataclass
class ThrottleDisplayInfo:
    """Info for displaying throttle status."""

    allowed_workers: int
    current_max: float
    avg_base_time: float
    completion_count: int  # Number of completions used to calculate avg_base_time
    peak_concurrency: int | None = None  # Concurrency level with peak throughput
    exploration_target: int | None = None  # Level being explored (neighbor of peak)
    # Per-level throughput data: level -> (throughput_per_sec, sample_count)
    all_levels: dict[int, tuple[float, int]] | None = None


@dataclass
class ThrottleContext:
    """Context for throttle calculations with warmup and cooldown.

    Mirrors the throttling logic from DependencyTracker (phase 5):
    - post_warmup: after warmup completes, forces gradual ramp from 1 worker
    - Cooldown timer: prevents ramp-up faster than the tier's cooldown period
    - Emergency scale-down: always immediate (no cooldown check)
    """

    tier_timeout: float
    base_workers: int
    throughput_tracker: ThroughputTracker
    tier: int = 2048  # Context tier for cooldown calculation

    # Warmup + cooldown state (managed internally)
    post_warmup: bool = False
    last_decision: ThrottleDecision | None = field(default=None, repr=False)
    _last_ramp_time: float = field(default=0.0, repr=False)
    _last_recommended_workers: int = field(default=0, repr=False)

    def get_throttle_decision(
        self, active_workers: int, current_max_runtime: float,
        remaining: int | None = None,
    ) -> ThrottleDisplayInfo:
        """Get throttle decision with cooldown gating.

        Always calls compute_throttle_decision (for emergency detection).
        Ramp-up is gated by cooldown timer — mirrors DependencyTracker pattern.

        Args:
            active_workers: Number of currently active workers.
            current_max_runtime: Max runtime of currently active workers.
            remaining: Elements left to process (for budget-aware exploration).
        """
        throughput, avg_runtime, count, avg_conc, avg_base = (
            self.throughput_tracker.get_stats_with_concurrency()
        )
        peak = self.throughput_tracker.get_peak_concurrency()
        explore = self.throughput_tracker.get_exploration_target(self.base_workers, remaining)
        all_levels = self.throughput_tracker._throughput_by_level.get_all_levels()
        throttle = compute_throttle_decision(
            current_max_runtime=current_max_runtime,
            tier_timeout=self.tier_timeout,
            base_workers=self.base_workers,
            active_workers=active_workers,
            throughput=throughput,
            avg_runtime=avg_runtime,
            completion_count=count,
            avg_concurrency=avg_conc,
            avg_base_time=avg_base,
            post_warmup=self.post_warmup,
            peak_concurrency=peak,
            exploration_target=explore,
        )
        self.post_warmup = False  # Only signal once
        # Attach per-level data for display (early returns in compute_throttle_decision
        # don't set these, so we always override from the source of truth)
        throttle.all_levels = all_levels if all_levels else None
        throttle.peak_concurrency = peak
        throttle.exploration_target = explore
        self.last_decision = throttle

        # Apply ramp cooldown (mirrors DependencyTracker.compute_throttle_decision)
        recommended = throttle.recommended_workers
        now = time.time()
        cooldown_seconds = get_ramp_cooldown(self.tier)
        cooldown_elapsed = (now - self._last_ramp_time) >= cooldown_seconds
        is_ramping = recommended > self._last_recommended_workers

        if is_ramping and not cooldown_elapsed:
            # Hold at current level during cooldown
            allowed_workers = self._last_recommended_workers
        else:
            allowed_workers = recommended
            if is_ramping and cooldown_elapsed:
                self._last_ramp_time = now
            self._last_recommended_workers = allowed_workers

        return ThrottleDisplayInfo(
            allowed_workers=allowed_workers,
            current_max=max(current_max_runtime, throttle.historical_max),
            avg_base_time=avg_base,
            completion_count=count,
            peak_concurrency=peak,
            exploration_target=explore,
            all_levels=all_levels if all_levels else None,
        )


def run_throttled_tier(
    items: list[T],
    tier: int,
    effective_workers: int,
    process_fn: Callable[[T], R],
    throttle_ctx: ThrottleContext,
    get_max_runtime: Callable[[], float],
    on_complete: Callable[[T, R, float, float], None],
    on_tick: Callable[[ThrottleDisplayInfo], None] | None = None,
    warmup: bool = True,
) -> None:
    """Run throttled processing for a single tier.

    This is a lightweight function for integrating throttling into existing code.
    It doesn't manage stats or worker status - the caller handles those.

    Args:
        items: Items to process in this tier
        tier: The context tier (for timeout lookup)
        effective_workers: Max workers for this tier
        process_fn: Function to process an item
        throttle_ctx: Context with throughput tracker and config
        get_max_runtime: Function to get current max active runtime
        on_complete: Called when item completes with (item, result, avg_workers, runtime)
        on_tick: Called on each loop iteration with ThrottleDisplayInfo (for progress refresh)
        warmup: If True, run the first item synchronously to get a baseline
            runtime before entering the throttled loop. Prevents aggressive
            ramp-up when no completion data exists yet.
    """
    if not items:
        return

    tier_timeout = TIER_TIMEOUTS.get(tier, 180)
    throttle_ctx.tier_timeout = tier_timeout
    throttle_ctx.base_workers = effective_workers
    throttle_ctx.tier = tier

    pending_items = list(items)

    # Warmup: run first item synchronously to get baseline runtime.
    # Without this, the throttler has no data and may ramp up too aggressively.
    if warmup and pending_items:
        warmup_item = pending_items.pop(0)
        warmup_start = time.time()
        warmup_result = process_fn(warmup_item)
        warmup_time = time.time() - warmup_start
        throttle_ctx.throughput_tracker.record_completion(warmup_time, 1.0)
        on_complete(warmup_item, warmup_result, 1.0, warmup_time)
        throttle_ctx.post_warmup = True

    if not pending_items:
        return

    executor = ThreadPoolExecutor(max_workers=effective_workers)
    futures: dict = {}
    future_to_allowed_at_start: dict = {}
    future_to_submit_time: dict = {}

    try:
        while pending_items or futures:
            # Get throttle decision (always runs for emergency detection)
            active_workers = len(futures)
            current_max = get_max_runtime()
            remaining = len(pending_items) + len(futures)
            throttle_info = throttle_ctx.get_throttle_decision(
                active_workers, current_max, remaining=remaining
            )
            allowed_workers = throttle_info.allowed_workers

            # Submit new tasks up to allowed limit
            while pending_items and len(futures) < allowed_workers:
                item = pending_items.pop(0)
                future = executor.submit(process_fn, item)
                futures[future] = item
                future_to_allowed_at_start[future] = allowed_workers
                future_to_submit_time[future] = time.time()

            if not futures:
                break

            # Wait for completion or timeout for display refresh
            done, _ = wait(futures.keys(), timeout=2.0, return_when=FIRST_COMPLETED)

            # Handle completed tasks
            for future in done:
                item = futures.pop(future)
                submit_time = future_to_submit_time.pop(future, time.time())
                runtime = time.time() - submit_time

                # Use allowed workers (max of start and end for peak concurrency)
                allowed_at_start = future_to_allowed_at_start.pop(future, allowed_workers)
                allowed_at_end = allowed_workers
                avg_workers = max(allowed_at_start, allowed_at_end)

                # Record in throughput tracker for throttle decisions
                throttle_ctx.throughput_tracker.record_completion(runtime, avg_workers)

                result = future.result()
                on_complete(item, result, avg_workers, runtime)

            # Tick for progress refresh
            if on_tick:
                on_tick(throttle_info)

    finally:
        executor.shutdown(wait=True)
