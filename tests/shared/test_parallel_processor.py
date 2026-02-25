"""Tests for shared.parallel_processor — warmup, cooldown, and run_throttled_tier."""

from __future__ import annotations

import time

from shared.parallel_processor import ThrottleContext, ThrottleDisplayInfo, run_throttled_tier
from shared.throttling import ThroughputTracker

# =============================================================================
# ThrottleContext tests
# =============================================================================


class TestThrottleContextWarmup:
    """Test post_warmup flag handling in ThrottleContext."""

    def test_post_warmup_consumed_once(self):
        """post_warmup=True should be passed once then cleared."""
        tracker = ThroughputTracker(window_seconds=60.0)
        # Seed with one completion so compute_throttle_decision has data
        tracker.record_completion(1.0, 1.0)

        ctx = ThrottleContext(
            tier_timeout=120.0,
            base_workers=8,
            throughput_tracker=tracker,
            tier=2048,
            post_warmup=True,
        )

        # First call should use post_warmup=True
        info1 = ctx.get_throttle_decision(active_workers=0, current_max_runtime=0.0)
        assert info1.allowed_workers >= 1

        # post_warmup should now be False
        assert ctx.post_warmup is False

        # Second call should behave differently (no longer post_warmup)
        info2 = ctx.get_throttle_decision(active_workers=0, current_max_runtime=0.0)
        assert info2.allowed_workers >= 1

    def test_post_warmup_starts_at_one_worker(self):
        """With post_warmup=True and active_workers=0, should start at 1."""
        tracker = ThroughputTracker(window_seconds=60.0)
        tracker.record_completion(1.0, 1.0)

        ctx = ThrottleContext(
            tier_timeout=120.0,
            base_workers=12,
            throughput_tracker=tracker,
            tier=2048,
            post_warmup=True,
        )

        info = ctx.get_throttle_decision(active_workers=0, current_max_runtime=0.0)
        # post_warmup forces start at 1 worker
        assert info.allowed_workers == 1

    def test_last_decision_stored(self):
        """last_decision should be set after each get_throttle_decision call."""
        tracker = ThroughputTracker(window_seconds=60.0)
        ctx = ThrottleContext(
            tier_timeout=120.0,
            base_workers=4,
            throughput_tracker=tracker,
            tier=2048,
        )

        assert ctx.last_decision is None
        ctx.get_throttle_decision(active_workers=0, current_max_runtime=0.0)
        assert ctx.last_decision is not None


class TestThrottleContextCooldown:
    """Test cooldown timer gating in ThrottleContext."""

    def test_cooldown_holds_ramp(self):
        """Ramp-up should be held during cooldown period."""
        tracker = ThroughputTracker(window_seconds=60.0)
        # Record fast completions so throttler wants many workers
        for _ in range(5):
            tracker.record_completion(0.5, 1.0)

        ctx = ThrottleContext(
            tier_timeout=120.0,
            base_workers=12,
            throughput_tracker=tracker,
            tier=2048,  # cooldown = 2048 // 1024 = 2 seconds
        )

        # First call: establishes baseline
        info1 = ctx.get_throttle_decision(active_workers=1, current_max_runtime=0.1)
        first_allowed = info1.allowed_workers

        # Immediate second call: should be held by cooldown (can't ramp further)
        info2 = ctx.get_throttle_decision(active_workers=first_allowed, current_max_runtime=0.1)
        assert info2.allowed_workers <= first_allowed + 1  # Can't jump more than 1

    def test_scale_down_ignores_cooldown(self):
        """Emergency scale-down should bypass cooldown."""
        tracker = ThroughputTracker(window_seconds=60.0)
        tracker.record_completion(1.0, 1.0)

        ctx = ThrottleContext(
            tier_timeout=10.0,  # Short timeout
            base_workers=8,
            throughput_tracker=tracker,
            tier=2048,
        )

        # Establish baseline at some workers
        ctx.get_throttle_decision(active_workers=4, current_max_runtime=0.5)

        # Now simulate emergency: max_runtime approaching timeout (>60%)
        info = ctx.get_throttle_decision(active_workers=4, current_max_runtime=7.0)
        # Emergency should force to 1 worker regardless of cooldown
        assert info.allowed_workers == 1


# =============================================================================
# run_throttled_tier tests
# =============================================================================


class TestRunThrottledTierWarmup:
    """Test warmup behavior in run_throttled_tier."""

    def test_warmup_runs_first_item_before_loop(self):
        """With warmup=True, first item should complete before others start."""
        order: list[int] = []

        def process_fn(item: int) -> int:
            order.append(item)
            return item * 10

        completed: list[tuple[int, int, float]] = []

        def on_complete(item: int, result: int, avg_workers: float) -> None:
            completed.append((item, result, avg_workers))

        tracker = ThroughputTracker(window_seconds=60.0)
        ctx = ThrottleContext(
            tier_timeout=120.0,
            base_workers=4,
            throughput_tracker=tracker,
            tier=2048,
        )

        items = [1, 2, 3, 4, 5]
        run_throttled_tier(
            items=items,
            tier=2048,
            effective_workers=4,
            process_fn=process_fn,
            throttle_ctx=ctx,
            get_max_runtime=lambda: 0.0,
            on_complete=on_complete,
            warmup=True,
        )

        # All items should be processed
        assert len(completed) == 5
        # First completion should be item 1 (warmup)
        assert completed[0] == (1, 10, 1.0)
        # post_warmup should have been set then consumed
        assert ctx.post_warmup is False

    def test_warmup_false_skips_warmup(self):
        """With warmup=False, items should go directly to the pool."""
        tracker = ThroughputTracker(window_seconds=60.0)
        ctx = ThrottleContext(
            tier_timeout=120.0,
            base_workers=4,
            throughput_tracker=tracker,
            tier=2048,
        )

        completed: list[int] = []

        def on_complete(item: int, _result: int, _avg_workers: float) -> None:
            completed.append(item)

        items = [1, 2, 3]
        run_throttled_tier(
            items=items,
            tier=2048,
            effective_workers=4,
            process_fn=lambda x: x,
            throttle_ctx=ctx,
            get_max_runtime=lambda: 0.0,
            on_complete=on_complete,
            warmup=False,
        )

        assert len(completed) == 3

    def test_warmup_records_in_tracker(self):
        """Warmup completion should be recorded in throughput tracker."""
        tracker = ThroughputTracker(window_seconds=60.0)
        ctx = ThrottleContext(
            tier_timeout=120.0,
            base_workers=4,
            throughput_tracker=tracker,
            tier=2048,
        )

        def slow_fn(item: int) -> int:
            time.sleep(0.05)
            return item

        run_throttled_tier(
            items=[1, 2],
            tier=2048,
            effective_workers=2,
            process_fn=slow_fn,
            throttle_ctx=ctx,
            get_max_runtime=lambda: 0.0,
            on_complete=lambda _i, _r, _w: None,
            warmup=True,
        )

        # Tracker should have completions recorded
        _, _, count, _, _ = tracker.get_stats_with_concurrency()
        assert count >= 1  # At least the warmup

    def test_empty_items(self):
        """Empty items list should return immediately."""
        tracker = ThroughputTracker(window_seconds=60.0)
        ctx = ThrottleContext(
            tier_timeout=120.0,
            base_workers=4,
            throughput_tracker=tracker,
            tier=2048,
        )

        # Should not raise
        run_throttled_tier(
            items=[],
            tier=2048,
            effective_workers=4,
            process_fn=lambda x: x,
            throttle_ctx=ctx,
            get_max_runtime=lambda: 0.0,
            on_complete=lambda _i, _r, _w: None,
        )

    def test_single_item_warmup_only(self):
        """Single item should be handled by warmup, no loop needed."""
        completed: list[int] = []

        def on_complete(item: int, _result: int, _avg_workers: float) -> None:
            completed.append(item)

        tracker = ThroughputTracker(window_seconds=60.0)
        ctx = ThrottleContext(
            tier_timeout=120.0,
            base_workers=4,
            throughput_tracker=tracker,
            tier=2048,
        )

        run_throttled_tier(
            items=[42],
            tier=2048,
            effective_workers=4,
            process_fn=lambda x: x * 2,
            throttle_ctx=ctx,
            get_max_runtime=lambda: 0.0,
            on_complete=on_complete,
            warmup=True,
        )

        assert completed == [42]

    def test_on_tick_called_during_processing(self):
        """on_tick should be called during the processing loop."""
        tick_calls: list[ThrottleDisplayInfo] = []

        def on_tick(info: ThrottleDisplayInfo) -> None:
            tick_calls.append(info)

        def slow_fn(item: int) -> int:
            time.sleep(0.05)
            return item

        tracker = ThroughputTracker(window_seconds=60.0)
        ctx = ThrottleContext(
            tier_timeout=120.0,
            base_workers=2,
            throughput_tracker=tracker,
            tier=2048,
        )

        run_throttled_tier(
            items=[1, 2, 3, 4, 5],
            tier=2048,
            effective_workers=2,
            process_fn=slow_fn,
            throttle_ctx=ctx,
            get_max_runtime=lambda: 0.0,
            on_complete=lambda _i, _r, _w: None,
            on_tick=on_tick,
            warmup=True,
        )

        # Should have tick calls from the processing loop
        assert len(tick_calls) >= 1
        assert all(isinstance(t, ThrottleDisplayInfo) for t in tick_calls)
