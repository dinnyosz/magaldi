"""Dependency tracking for parallel element processing.

Manages parent-child dependencies and batching strategy for optimal processing.
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

from shared.ai.context_size import CONTEXT_TIERS, get_tier_timeout
from shared.throttling import (
    ExplorationOrchestrator,
    ThrottleDecision,
    ThroughputTracker,
    compute_throttle_decision,
    get_ramp_cooldown,
)


def _log_warmup(message: str) -> None:
    """Debug logging - disabled by default."""
    pass


if TYPE_CHECKING:
    from magaldi_core.code_parser import CodeElement

    from .status import ParallelismStats


class DependencyTracker:
    """Track element dependencies for parallel processing.

    Rules:
    - Level 0 (files): Always ready
    - Level 1 (classes): Ready when parent file done
    - Level 2 (methods/functions): Ready when parent class done (or file if no class)
    - Level 3 (variables): Ready when parent done

    Batching strategy (model+tier based):
    - Batch by (model, context_tier) to minimize Ollama reloads
    - Priority: same model+tier > same model+different tier > different model
    - Drains current tasks when model changes
    - Warmup limits to 1 task for model/tier transitions
    """

    # Default worker count when not specified by user
    DEFAULT_WORKERS = 8

    def __init__(
        self,
        elements: list[CodeElement],
        context_sizes: dict[str, int] | None = None,
        max_num_workers: int | None = None,
        timeout: float = 360.0,  # Default timeout for throttle calculations
    ) -> None:
        # RLock for reentrant calls (get_parallelism_stats -> get_current_max_workers)
        self._lock = threading.RLock()
        self._elements = {e.element_id: e for e in elements}
        self._completed: set[str] = set()
        self._in_progress: set[str] = set()
        self._context_sizes = context_sizes or {}
        self._current_model: str | None = None  # Current model key ("large" or "small")
        self._current_tier: int | None = None  # Current context tier all workers use
        self._previous_tier: int | None = None  # Previous tier (to detect changes)
        self._tier_changing: bool = False  # True when switching model/tier (warmup period)
        self._warmup_task_id: str | None = None  # ID of warmup task (first task of new tier)
        self._tier_just_changed: bool = False  # True when tier changed (for throughput reset)
        self._post_warmup: bool = False  # True after warmup completes (for gradual ramp)
        self._last_ramp_time: float = 0.0  # Last time we ramped up (for cooldown)
        self._last_recommended_workers: int = 0  # Track to detect ramp-ups
        # Use provided workers or default
        self._max_num_workers = max_num_workers if max_num_workers else self.DEFAULT_WORKERS
        self._timeout = timeout  # Timeout for throttle calculations

        # GSS exploration orchestrator (shared with ThrottleContext)
        self._exploration = ExplorationOrchestrator(self._max_num_workers)
        # Throughput tracker reference — set by caller via set_throughput_tracker()
        self._throughput_tracker: ThroughputTracker | None = None

        # Build parent lookup: element_id -> parent_element_id
        self._parents: dict[str, str | None] = {}
        for e in elements:
            self._parents[e.element_id] = e.parent_id

    def set_throughput_tracker(self, tracker: ThroughputTracker) -> None:
        """Set the throughput tracker reference for GSS exploration."""
        self._throughput_tracker = tracker

    def _get_level(self, element: CodeElement) -> int:
        """Get hierarchy level from element type.

        Level 0: files
        Level 1: classes
        Level 2: functions, methods
        Level 3: variables, constants
        """
        level_map = {
            "file": 0,
            "class": 1,
            "function": 2,
            "method": 2,
            "variable": 3,
            "constant": 3,
        }
        return level_map.get(element.element_type, 2)

    def _get_model_key(self, element: CodeElement) -> str:
        """Get model key for an element (for batching by model).

        Returns 'large' for files/classes/interfaces/type_aliases, 'small' for functions/methods/variables.
        This matches ProcessingConfig.get_model_for_element_type().
        """
        if element.element_type in ("function", "method", "variable", "constant"):
            return "small"
        return "large"

    def _get_tier(self, element_id: str) -> int:
        """Get the context tier for an element (snaps to standard tiers)."""
        ctx = self._context_sizes.get(element_id, 2048)
        # Find the tier this context size belongs to
        for tier in CONTEXT_TIERS:
            if ctx <= tier:
                return tier  # type: ignore[no-any-return]
        return CONTEXT_TIERS[-1]  # type: ignore[no-any-return]  # Max tier

    def _get_all_ready(self) -> list[CodeElement]:
        """Get all ready elements (parent done, not started). Must hold lock."""
        ready = []
        for eid, element in self._elements.items():
            if eid in self._completed or eid in self._in_progress:
                continue

            parent_id = self._parents.get(eid)
            # Element is ready if:
            # - No parent (level 0)
            # - Parent completed
            # - Parent not in elements_to_process (was skipped/unchanged)
            parent_ready = (
                parent_id is None
                or parent_id in self._completed
                or parent_id not in self._elements
            )
            if parent_ready:
                ready.append(element)
        return ready

    def get_ready_elements(
        self,
        max_count: int = 10,
        throttle_limit: int | None = None,
    ) -> list[CodeElement]:
        """Get elements ready for processing (parent done, not started).

        Ordering priority:
        1. By hierarchy level (files → classes → methods) to respect parent deps
        2. Within level, by model+tier to minimize Ollama reloads
        3. Prefer current model+tier, then same model different tier, then switch

        Throttling:
        - Runtime-based throttle_limit reduces workers when approaching timeout
        - Tier warmup limits to 1 task during model/tier transitions

        Args:
            max_count: Maximum elements to return.
            throttle_limit: If set, reduces worker limit for runtime throttling.
        """
        with self._lock:
            ready = self._get_all_ready()
            if not ready:
                return []

            # First, group by hierarchy level to ensure parents processed before children
            by_level: dict[int, list[CodeElement]] = {}
            for elem in ready:
                level = self._get_level(elem)
                by_level.setdefault(level, []).append(elem)

            # Pick the lowest level with ready elements (files before classes before methods)
            target_level = min(by_level.keys())
            level_ready = by_level[target_level]

            # Within the level, group by (model, tier) for optimal batching
            by_model_tier: dict[tuple[str, int], list[CodeElement]] = {}
            for elem in level_ready:
                model_key = self._get_model_key(elem)
                tier = self._get_tier(elem.element_id)
                by_model_tier.setdefault((model_key, tier), []).append(elem)

            # Sort each batch by context size (largest first - finish big ones first)
            for key in by_model_tier:
                by_model_tier[key].sort(
                    key=lambda e: self._context_sizes.get(e.element_id, 0),
                    reverse=True
                )

            # Determine which (model, tier) to use with priority:
            # 1. Current model + current tier (if available at this level)
            # 2. Current model + largest available tier (at this level)
            # 3. Different model + largest available tier (at this level)
            # Rationale: finish big tasks first to avoid long tail at the end
            selected_model = None
            selected_tier = None

            current_key = (self._current_model, self._current_tier)
            if self._current_model is not None and current_key in by_model_tier:
                # Priority 1: same model + same tier
                selected_model = self._current_model
                selected_tier = self._current_tier
            elif self._current_model is not None:
                # Priority 2: same model + different tier (largest first)
                same_model_tiers = [
                    tier for (model, tier) in by_model_tier
                    if model == self._current_model
                ]
                if same_model_tiers:
                    selected_model = self._current_model
                    selected_tier = max(same_model_tiers)

            if selected_model is None:
                # Priority 3: different model, pick largest tier overall
                # Group by model first, then pick model with largest max tier
                models_with_max_tier = {}
                for (model, tier) in by_model_tier:
                    if model not in models_with_max_tier:
                        models_with_max_tier[model] = tier
                    else:
                        models_with_max_tier[model] = max(models_with_max_tier[model], tier)

                # Pick model with largest maximum tier (start big)
                selected_model = max(models_with_max_tier.keys(), key=lambda m: models_with_max_tier[m])
                selected_tier = models_with_max_tier[selected_model]

            # Check if model or tier is changing
            model_changing = self._current_model is not None and selected_model != self._current_model
            tier_changing = self._current_tier is not None and selected_tier != self._current_tier

            # Drain on model OR tier change: wait for current tasks to finish before switching
            # This ensures proper warmup and ramp-up for each new tier
            if (model_changing or tier_changing) and len(self._in_progress) > 0:
                return []  # Drain: wait for current tasks to finish

            # Set tier_changing flag for warmup (model change or tier change)
            if model_changing or tier_changing or self._current_model is None:
                self._tier_changing = True
                self._tier_just_changed = True  # Signal for throughput reset

            self._current_model = selected_model
            self._previous_tier = self._current_tier
            self._current_tier = selected_tier

            # Worker limit: start with configured max, apply throttling
            worker_limit = self._max_num_workers

            # Apply runtime-based throttle limit if set
            if throttle_limit is not None:
                worker_limit = min(worker_limit, throttle_limit)

            # When tier is changing, wait for old tasks to complete first
            # then start 1 task to load new model before ramping up
            if self._tier_changing:
                # If any tasks still running, wait for them to finish
                if len(self._in_progress) > 0:
                    return []  # Don't start new tier until old tier drains
                worker_limit = 1

            # Ensure at least 1 worker always
            worker_limit = max(1, worker_limit)

            # When throttling, count ALL in-progress to limit total concurrency
            # Otherwise, count only current (model, tier) for normal batched operation
            if throttle_limit is not None:
                total_in_progress = len(self._in_progress)
                slots_available = max(0, worker_limit - total_in_progress)
            else:
                batch_in_progress = sum(
                    1 for eid in self._in_progress
                    if self._get_tier(eid) == selected_tier
                    and self._get_model_key(self._elements[eid]) == selected_model
                )
                slots_available = max(0, worker_limit - batch_in_progress)
            effective_limit = min(max_count, slots_available)

            # During warmup, only return 1 element maximum
            if self._tier_changing:
                effective_limit = 1

            # Get elements from current (model, tier) batch
            assert selected_tier is not None
            batch_key = (selected_model, selected_tier)
            batch_ready = by_model_tier[batch_key][:effective_limit]

            # Mark as in-progress
            for e in batch_ready:
                self._in_progress.add(e.element_id)
                # Track warmup task (first task dispatched when tier_changing)
                if self._tier_changing and self._warmup_task_id is None:
                    self._warmup_task_id = e.element_id

            return batch_ready

    def is_tier_changing(self) -> bool:
        """Check if we're in startup warmup (first task loading model)."""
        with self._lock:
            return self._tier_changing

    def did_tier_just_change(self) -> bool:
        """Check if tier just changed (for throughput reset).

        Returns True once per tier change, then clears the flag.
        Also resets ramp cooldown state and exploration orchestrator
        so the new tier starts fresh.
        """
        with self._lock:
            if self._tier_just_changed:
                self._tier_just_changed = False
                self._last_ramp_time = 0.0
                self._last_recommended_workers = 0
                self._exploration.reset()
                return True
            return False

    def clear_post_warmup(self) -> None:
        """Clear the post_warmup flag after main throttle calculation.

        Must be called ONCE per loop iteration after the main throttle decision,
        NOT after display-only throttle calls. This ensures the flag is consumed
        only when the throttle limit is actually applied.
        """
        with self._lock:
            if self._post_warmup:
                _log_warmup("CLEAR: post_warmup flag cleared")
                self._post_warmup = False

    def get_current_tier_timeout(self) -> float:
        """Get timeout for the current tier, scaled by worker count."""
        with self._lock:
            if self._current_tier is not None:
                return float(get_tier_timeout(self._current_tier, self._max_num_workers))
            return self._timeout  # Fallback to default

    def get_current_model(self) -> str | None:
        """Get the current model key being used ('large' or 'small')."""
        with self._lock:
            return self._current_model

    def mark_complete(self, element_id: str) -> None:
        """Mark element as completed."""
        with self._lock:
            self._in_progress.discard(element_id)
            self._completed.add(element_id)
            # Clear warmup flag only when the warmup task itself completes
            # This ensures the model is fully loaded before ramping up
            if self._tier_changing and element_id == self._warmup_task_id:
                self._tier_changing = False
                self._warmup_task_id = None
                self._post_warmup = True  # Signal for gradual ramp-up
                _log_warmup("WARMUP COMPLETE: set post_warmup=True")

    def mark_failed(self, element_id: str) -> None:
        """Mark element as failed (won't block children)."""
        with self._lock:
            self._in_progress.discard(element_id)
            self._completed.add(element_id)  # Treat as done so children can proceed
            # Clear warmup flag if warmup task failed (so processing can continue)
            if self._tier_changing and element_id == self._warmup_task_id:
                self._tier_changing = False
                self._warmup_task_id = None
                self._post_warmup = True  # Signal for gradual ramp-up

    def is_complete(self) -> bool:
        """Check if all elements are processed."""
        with self._lock:
            return len(self._completed) == len(self._elements)

    def pending_count(self) -> int:
        """Count elements not yet completed."""
        with self._lock:
            return len(self._elements) - len(self._completed)

    def get_current_tier(self) -> int | None:
        """Get the current context tier being processed."""
        with self._lock:
            return self._current_tier

    def get_current_max_workers(self) -> int:
        """Get max workers (for status display)."""
        with self._lock:
            return self._max_num_workers

    def get_parallelism_stats(
        self,
        max_possible: int,
        throttle_decision: ThrottleDecision | None = None,
    ) -> ParallelismStats:
        """Get current parallelism statistics for display."""
        from .status import ParallelismStats

        with self._lock:
            tier_limit = self.get_current_max_workers()
            running = len(self._in_progress)
            return ParallelismStats(
                max_possible=max_possible,
                tier_limit=tier_limit,
                running=running,
                current_tier=self._current_tier,
                throttle_decision=throttle_decision,
                tier_changing=self._tier_changing,
            )

    def compute_throttle_decision(
        self,
        current_max_runtime: float,
        active_workers: int,
        throughput: float = 0.0,
        avg_runtime: float = 0.0,
        completion_count: int = 0,
        avg_concurrency: float = 0.0,
        avg_base_time: float = 0.0,
        is_display_call: bool = False,
        all_levels: dict[int, tuple[float, int]] | None = None,
        remaining: int | None = None,
    ) -> ThrottleDecision:
        """Compute throttle decision with GSS exploration.

        Key insight: runtime scales linearly with concurrent workers (GPU contention).
        base_time = runtime / workers is the normalized per-worker cost.

        Uses ExplorationOrchestrator for GSS lifecycle (warmup → GSS → converge).

        Args:
            current_max_runtime: Max runtime of currently active workers.
            active_workers: Number of currently active workers.
            throughput: Actual completions per second.
            avg_runtime: Average completion time from recent completions.
            completion_count: Number of completions in tracking window.
            avg_concurrency: Average workers active at task start.
            avg_base_time: Average of (runtime/workers) from completions.
            is_display_call: If True, don't apply cooldown (display only, not actual throttle).
            all_levels: Per-level throughput data from ThroughputTracker.
            remaining: Elements left to process (for budget-aware exploration).

        Returns:
            ThrottleDecision with recommended action.
        """
        with self._lock:
            # Note: post_warmup is NOT consumed here - it must be explicitly cleared
            # via clear_post_warmup() after the main throttle calculation.
            post_warmup = self._post_warmup
            if post_warmup:
                _log_warmup(f"THROTTLE: post_warmup=True, active_workers={active_workers}")

            # Run GSS exploration orchestration
            peak_concurrency: int | None = None
            exploration_target: int | None = None
            gss_probe: int | None = None
            expl = None

            if all_levels is not None and self._throughput_tracker is not None:
                legacy_peak = self._throughput_tracker.get_peak_concurrency()
                expl = self._exploration.orchestrate(
                    all_levels, legacy_peak, self._throughput_tracker, remaining
                )
                gss_probe = expl.gss_probe
                peak_concurrency = expl.effective_peak
                exploration_target = expl.exploration_target

            decision = compute_throttle_decision(
                current_max_runtime=current_max_runtime,
                tier_timeout=self.get_current_tier_timeout(),
                base_workers=self._max_num_workers,
                active_workers=active_workers,
                throughput=throughput,
                avg_runtime=avg_runtime,
                completion_count=completion_count,
                avg_concurrency=avg_concurrency,
                avg_base_time=avg_base_time,
                post_warmup=post_warmup,
                peak_concurrency=peak_concurrency,
                exploration_target=exploration_target,
                gss_probe=gss_probe,
            )

            # Apply warmup override from orchestrator (pre-GSS phases)
            if expl is not None and expl.warmup_override is not None:
                decision.recommended_workers = expl.warmup_override
                decision.reason = expl.warmup_reason or decision.reason

            # Apply ramp cooldown: scales with context tier
            # GSS probes bypass cooldown — need data at probe level ASAP
            now = time.time()
            cooldown_seconds = get_ramp_cooldown(self._current_tier or 2048)
            cooldown_elapsed = (now - self._last_ramp_time) >= cooldown_seconds

            is_ramping = decision.recommended_workers > self._last_recommended_workers

            if is_ramping and not cooldown_elapsed and gss_probe is None:
                # Hold at current level during cooldown (but not for GSS probes)
                from shared.throttling import _log_throttle
                cooldown_remaining = cooldown_seconds - (now - self._last_ramp_time)
                if not is_display_call:
                    _log_throttle(
                        f"RAMP COOLDOWN: want {decision.recommended_workers} but holding at "
                        f"{self._last_recommended_workers} ({cooldown_remaining:.1f}s remaining, tier={self._current_tier})"
                    )
                decision = ThrottleDecision(
                    should_throttle=decision.should_throttle,
                    current_max=decision.current_max,
                    historical_max=decision.historical_max,
                    completed_avg=decision.completed_avg,
                    recommended_workers=self._last_recommended_workers,
                    reason=f"{decision.reason} (cooldown)",
                    completion_count=decision.completion_count,
                )

            if not is_display_call:
                if is_ramping and cooldown_elapsed:
                    self._last_ramp_time = now
                    self._last_recommended_workers = decision.recommended_workers
                elif decision.recommended_workers != self._last_recommended_workers:
                    self._last_recommended_workers = decision.recommended_workers

            # Attach per-level data and GSS display fields
            decision.all_levels = all_levels if all_levels else None
            decision.peak_concurrency = peak_concurrency
            decision.exploration_target = exploration_target
            decision.gss_probe = gss_probe
            if expl is not None:
                decision.gss_lo = expl.gss_lo
                decision.gss_hi = expl.gss_hi
                decision.gss_signal = expl.gss_signal
                decision.prob_map_data = expl.prob_map_data
                decision.exploration_status = expl.exploration_status
            return decision

    def get_tier_stats(self) -> dict[int, tuple[int, int]]:
        """Get (ready, total) counts per tier for pending elements."""
        with self._lock:
            ready = self._get_all_ready()
            ready_ids = {e.element_id for e in ready}

            stats: dict[int, tuple[int, int]] = {}
            for eid in self._elements:
                if eid in self._completed:
                    continue
                tier = self._get_tier(eid)
                r, t = stats.get(tier, (0, 0))
                is_ready = 1 if eid in ready_ids else 0
                stats[tier] = (r + is_ready, t + 1)
            return stats
