"""Tests for throughput-based throttling module."""

import time

import pytest

from shared.throttling import (
    ThroughputTracker,
    ThrottleDecision,
    TimeoutEvent,
    compute_throttle_decision,
)


class TestThroughputTracker:
    """Tests for ThroughputTracker class."""

    def test_empty_tracker_returns_zeros(self):
        """Empty tracker should return zero stats."""
        tracker = ThroughputTracker()
        throughput, avg, count = tracker.get_stats()
        assert throughput == 0.0
        assert avg == 0.0
        assert count == 0

    def test_single_completion(self):
        """Single completion should be tracked."""
        tracker = ThroughputTracker(window_seconds=10.0)
        tracker.record_completion(5.0)
        throughput, avg, count = tracker.get_stats()
        assert count == 1
        assert avg == 5.0
        assert throughput > 0  # Should have some throughput

    def test_multiple_completions(self):
        """Multiple completions should be averaged."""
        tracker = ThroughputTracker(window_seconds=10.0)
        tracker.record_completion(3.0)
        tracker.record_completion(7.0)
        tracker.record_completion(5.0)
        throughput, avg, count = tracker.get_stats()
        assert count == 3
        assert avg == 5.0  # (3 + 7 + 5) / 3

    def test_window_expiration(self):
        """Completions outside window should be pruned."""
        tracker = ThroughputTracker(window_seconds=0.1)

        # First completion
        tracker.record_completion(10.0)
        throughput, avg, count = tracker.get_stats()
        assert count == 1
        assert avg == 10.0

        # Wait for window to expire
        time.sleep(0.15)

        # Add new completion
        tracker.record_completion(5.0)
        throughput, avg, count = tracker.get_stats()
        assert count == 1  # Old one expired
        assert avg == 5.0

    def test_reset_clears_all(self):
        """Reset should clear all data."""
        tracker = ThroughputTracker()
        tracker.record_completion(100.0)
        tracker.record_completion(200.0)

        throughput, avg, count = tracker.get_stats()
        assert count == 2

        tracker.reset()
        throughput, avg, count = tracker.get_stats()
        assert count == 0
        assert avg == 0.0
        assert throughput == 0.0

    def test_throughput_calculation(self):
        """Throughput should be completions per second."""
        tracker = ThroughputTracker(window_seconds=10.0)

        # Record several completions
        for _ in range(5):
            tracker.record_completion(2.0)

        throughput, avg, count = tracker.get_stats()
        assert count == 5
        assert avg == 2.0
        # Throughput should be approximately count / time_span
        # Since all completions are nearly instant, time_span is ~1s minimum
        assert throughput >= 1.0

    def test_base_time_calculation(self):
        """Base time should be runtime / workers (normalized per-worker cost).

        Example: 10 workers each taking 10s → base_time = 1s
        This tells us that with 1 worker, a task would take ~1s.
        """
        tracker = ThroughputTracker(window_seconds=10.0)

        # 10 workers, 10s each → base_time = 1s
        tracker.record_completion(10.0, concurrent_workers=10)
        # 8 workers, 16s each → base_time = 2s
        tracker.record_completion(16.0, concurrent_workers=8)
        # 4 workers, 12s each → base_time = 3s
        tracker.record_completion(12.0, concurrent_workers=4)

        throughput, avg_runtime, count, avg_conc, avg_base_time = (
            tracker.get_stats_with_concurrency()
        )
        assert count == 3
        assert avg_runtime == pytest.approx((10 + 16 + 12) / 3)  # ~12.67
        assert avg_conc == pytest.approx((10 + 8 + 4) / 3)  # ~7.33
        # avg_base_time = (1 + 2 + 3) / 3 = 2s
        assert avg_base_time == pytest.approx(2.0)

    def test_warmup_tasks_included_in_base_time(self):
        """Warmup tasks (workers=0) are included, treated as 1 worker.

        Warmup tasks give us baseline timing - how long a task takes with no contention.

        Example: If we have:
        - warmup: 50s with 0 workers → base_time = 50s (treated as 1)
        - warmup: 60s with 0 workers → base_time = 60s (treated as 1)
        - normal: 40s with 2 workers → base_time = 20s
        avg_base_time = (50+60+20)/3 = 43.3s
        """
        tracker = ThroughputTracker(window_seconds=10.0)

        # Warmup tasks (workers=0) - treated as 1 worker
        tracker.record_completion(50.0, concurrent_workers=0)
        tracker.record_completion(60.0, concurrent_workers=0)

        # Normal task with contention data
        tracker.record_completion(40.0, concurrent_workers=2)

        throughput, avg_runtime, count, avg_conc, avg_base_time = (
            tracker.get_stats_with_concurrency()
        )

        # Count includes all completions
        assert count == 3

        # avg_base_time includes all: (50/1 + 60/1 + 40/2) / 3 = (50+60+20)/3 = 43.3s
        assert avg_base_time == pytest.approx((50 + 60 + 20) / 3)

    def test_only_warmup_tasks_gives_base_time(self):
        """If only warmup tasks exist, avg_base_time is their average.

        Warmup tasks are treated as 1 worker for base_time calculation.
        """
        tracker = ThroughputTracker(window_seconds=10.0)

        # Only warmup tasks
        tracker.record_completion(50.0, concurrent_workers=0)
        tracker.record_completion(60.0, concurrent_workers=0)

        throughput, avg_runtime, count, avg_conc, avg_base_time = (
            tracker.get_stats_with_concurrency()
        )

        assert count == 2
        # (50/1 + 60/1) / 2 = 55s
        assert avg_base_time == pytest.approx(55.0)


class TestComputeThrottleDecision:
    """Tests for compute_throttle_decision function.

    KEY INSIGHT: Runtime scales linearly with concurrent workers (GPU contention).
    base_time = runtime / workers  (normalized per-worker cost)
    max_workers = (timeout * 0.7) / base_time

    Example: 10 workers each taking 10s → base_time = 1s
    If timeout = 7s → max_workers = 7/1 = 7
    """

    def test_no_data_returns_fresh_start(self):
        """With no running tasks and no completion data, starts with 1 worker."""
        decision = compute_throttle_decision(
            current_max_runtime=0.0,
            tier_timeout=180.0,
            base_workers=8,
            active_workers=0,
            throughput=0.0,
            avg_runtime=0.0,
            completion_count=0,
        )
        assert not decision.should_throttle
        assert decision.recommended_workers == 1  # Fresh start
        assert "No data" in decision.reason

    def test_no_data_with_running_tasks_ramps(self):
        """With running tasks but no completion history, ramps up gradually.

        Even without enough data, we ramp up instead of jumping to max.
        active=4, base=8 → ramp: 4 + min(max(1, 1), 3) = 5
        """
        decision = compute_throttle_decision(
            current_max_runtime=10.0,
            tier_timeout=180.0,
            base_workers=8,
            active_workers=4,
            throughput=1.0,
            avg_runtime=5.0,
            completion_count=2,  # < 3, so no throttling but still ramps
        )
        assert not decision.should_throttle
        assert decision.recommended_workers == 5  # Ramped from 4
        assert "ramped from 4" in decision.reason

    def test_no_data_without_running_tasks_fresh(self):
        """With no running tasks AND few completions, starts fresh with 1."""
        decision = compute_throttle_decision(
            current_max_runtime=0.0,
            tier_timeout=180.0,
            base_workers=8,
            active_workers=0,
            throughput=0.0,
            avg_runtime=5.0,
            completion_count=2,
            avg_base_time=0.0,  # No base_time data either
        )
        assert not decision.should_throttle
        assert decision.recommended_workers == 1  # Fresh start
        assert "No data" in decision.reason

    def test_emergency_near_timeout(self):
        """Runtime >= 60% of timeout triggers emergency throttling."""
        decision = compute_throttle_decision(
            current_max_runtime=110.0,  # ~61% of 180 (threshold is 60%)
            tier_timeout=180.0,
            base_workers=8,
            active_workers=4,
            throughput=0.5,
            avg_runtime=5.0,
            completion_count=10,
        )
        assert decision.should_throttle
        assert decision.recommended_workers == 1
        assert "Emergency" in decision.reason

    def test_throttle_based_on_base_time(self):
        """Throttling based on formula: workers = (timeout * 0.65) / base_time.

        base_time from history = 20s
        max_workers = 117 / 20 = 5.85 → 5 workers
        """
        decision = compute_throttle_decision(
            current_max_runtime=0.0,  # No running tasks
            tier_timeout=180.0,
            base_workers=32,
            active_workers=0,
            throughput=1.0,
            avg_runtime=100.0,
            completion_count=10,
            avg_base_time=20.0,  # 117/20 = 5.85 → 5 workers max
        )
        assert decision.should_throttle
        assert decision.recommended_workers == 5
        assert "Throttle" in decision.reason

    def test_low_base_time_allows_full_workers(self):
        """Low base_time from history allows full workers.

        avg_base_time = 0.625s (from completion history)
        max_workers = 117 / 0.625 = 187.2 → capped at 8 → Normal
        """
        decision = compute_throttle_decision(
            current_max_runtime=5.0,
            tier_timeout=180.0,
            base_workers=8,
            active_workers=8,
            throughput=1.6,
            avg_runtime=5.0,
            completion_count=10,
            avg_base_time=0.625,  # From completion history
        )
        assert not decision.should_throttle
        assert decision.recommended_workers == 8
        assert decision.reason == "Normal"

    def test_high_base_time_limits_workers(self):
        """High base_time limits workers significantly.

        base_time from history = 60s
        max_workers = 117 / 60 = 1.95 → 1 worker
        """
        decision = compute_throttle_decision(
            current_max_runtime=0.0,  # No running tasks
            tier_timeout=180.0,
            base_workers=32,
            active_workers=0,
            throughput=0.1,
            avg_runtime=300.0,
            completion_count=10,
            avg_base_time=60.0,  # 117/60 = 1.95 → 1 worker max
        )
        assert decision.should_throttle
        assert decision.recommended_workers == 1

    def test_very_high_base_time_limits_to_one(self):
        """Very high base_time (> effective timeout) limits to 1 worker.

        base_time = 200s → max_workers = 117/200 = 0.585 → clamped to 1
        """
        decision = compute_throttle_decision(
            current_max_runtime=0.0,
            tier_timeout=180.0,
            base_workers=8,
            active_workers=0,
            throughput=0.01,
            avg_runtime=1000.0,
            completion_count=10,
            avg_base_time=200.0,  # 117/200 = 0.585 → 1 worker
        )
        assert decision.should_throttle
        assert decision.recommended_workers == 1

    def test_minimum_one_worker(self):
        """Should always recommend at least 1 worker."""
        decision = compute_throttle_decision(
            current_max_runtime=170.0,  # Very near timeout but not emergency
            tier_timeout=180.0,
            base_workers=1,
            active_workers=1,
            throughput=0.0,
            avg_runtime=500.0,
            completion_count=10,
            avg_base_time=500.0,  # Would give 0.25 workers
        )
        assert decision.recommended_workers >= 1

    def test_base_time_from_history_only(self):
        """Only uses historical avg_base_time, ignores current running tasks.

        We no longer calculate base_time from current running tasks because
        we don't know how many workers were active when each task started.
        avg_base_time=5.0s → optimal = 117/5 = 23 workers
        Ramp-up applies: 8 + min(max(1, int(15*0.25)), 3) = 8 + 3 = 11 workers
        """
        decision = compute_throttle_decision(
            current_max_runtime=80.0,  # Ignored for base_time calculation
            tier_timeout=180.0,
            base_workers=32,
            active_workers=8,
            throughput=1.5,
            avg_runtime=40.0,
            completion_count=10,
            avg_base_time=5.0,  # 117/5 = 23.4 → 23 optimal
        )
        assert decision.should_throttle
        # Optimal is 23, ramp from 8: 8 + min(max(1, 3), 3) = 11
        assert decision.recommended_workers == 11
        assert "Throttle" in decision.reason
        assert "ramped" in decision.reason

    def test_uses_historical_base_time(self):
        """Uses historical avg_base_time for throttle decisions.

        avg_base_time = 10s → optimal = 117/10 = 11.7 → 11 workers
        Ramp-up applies: 8 + min(max(1, int(3*0.25)), 3) = 8 + 1 = 9 workers
        """
        decision = compute_throttle_decision(
            current_max_runtime=40.0,
            tier_timeout=180.0,
            base_workers=32,
            active_workers=8,
            throughput=1.0,
            avg_runtime=50.0,
            completion_count=10,
            avg_base_time=10.0,  # Historical is worse than current
        )
        assert decision.should_throttle
        assert decision.recommended_workers == 9  # Ramped from 8 toward 11
        assert "Throttle" in decision.reason

    def test_ramp_up_during_warmup(self):
        """During warmup, ramp up gradually instead of jumping to optimal.

        With 2 active workers and base_time suggesting 23 optimal:
        delta = 23 - 2 = 21
        increment = min(max(1, int(21*0.25)), 3) = min(5, 3) = 3
        ramped = 2 + 3 = 5
        """
        decision = compute_throttle_decision(
            current_max_runtime=10.0,  # 10s with 2 workers = 5s base_time
            tier_timeout=180.0,
            base_workers=32,
            active_workers=2,
            throughput=0.5,
            avg_runtime=10.0,
            completion_count=5,
            avg_base_time=5.0,  # 117/5 = 23.4 → 23 optimal
        )
        assert decision.should_throttle
        # Optimal is 23, but we ramp with max increment of 3: 2 + 3 = 5
        assert decision.recommended_workers == 5
        assert "ramped from 2" in decision.reason

    def test_scale_down_immediate(self):
        """When scaling down, apply immediately without ramping.

        If optimal is less than active, we're seeing slowness - react fast.
        """
        decision = compute_throttle_decision(
            current_max_runtime=100.0,  # 100s with 20 workers = 5s base_time
            tier_timeout=180.0,
            base_workers=32,
            active_workers=20,
            throughput=0.1,
            avg_runtime=100.0,
            completion_count=10,
            avg_base_time=30.0,  # Historical shows 30s base → 117/30 = 3.9 → 3 optimal
        )
        assert decision.should_throttle
        # Optimal is 3, active is 20 - scale down immediately, no ramp
        assert decision.recommended_workers == 3
        assert "ramped" not in decision.reason


class TestThrottleDecisionDataclass:
    """Tests for ThrottleDecision dataclass."""

    def test_dataclass_fields(self):
        """ThrottleDecision should have all expected fields."""
        decision = ThrottleDecision(
            should_throttle=True,
            current_max=45.0,
            historical_max=50.0,
            completed_avg=5.0,
            recommended_workers=4,
            reason="Test reason",
        )
        assert decision.should_throttle is True
        assert decision.current_max == 45.0
        assert decision.historical_max == 50.0
        assert decision.completed_avg == 5.0
        assert decision.recommended_workers == 4
        assert decision.reason == "Test reason"


class TestTimeoutEvent:
    """Tests for TimeoutEvent dataclass."""

    def test_timeout_event_fields(self):
        """TimeoutEvent should have all expected fields."""
        event = TimeoutEvent(
            element_id="scope:repo:user:path/file.py:function:my_func:10",
            element_type="function",
            tier=8192,
            workers_active=5,
            avg_runtime=25.0,
            max_runtime=45.0,
            timeout_limit=180.0,
        )
        assert event.element_id == "scope:repo:user:path/file.py:function:my_func:10"
        assert event.element_type == "function"
        assert event.tier == 8192
        assert event.workers_active == 5
        assert event.avg_runtime == 25.0
        assert event.max_runtime == 45.0
        assert event.timeout_limit == 180.0
        assert event.timestamp > 0

    def test_to_log_line_format(self):
        """to_log_line should return properly formatted string."""
        event = TimeoutEvent(
            element_id="scope:repo:user:path/file.py:function:my_func:10",
            element_type="function",
            tier=8192,
            workers_active=5,
            avg_runtime=25.3,
            max_runtime=45.7,
            timeout_limit=180.0,
        )
        log_line = event.to_log_line()
        assert "[TIMEOUT]" in log_line
        assert "element=function:my_func" in log_line
        assert "tier=8192" in log_line
        assert "workers=5" in log_line
        assert "avg=25.3s" in log_line
        assert "max=45.7s" in log_line
        assert "limit=180s" in log_line
