"""Tests for throughput-based throttling module."""

import time

import pytest

from shared.throttling import (
    ThrottleDecision,
    ThroughputByLevel,
    ThroughputTracker,
    TimeoutEvent,
    build_throughput_levels_text,
    compute_throttle_decision,
    format_throughput_levels,
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
        Ramp-up applies: 8 + 1 = 9 workers (MAX_RAMP_INCREMENT=1)
        current_max must be under 30% of timeout to allow ramp (not hold).
        """
        decision = compute_throttle_decision(
            current_max_runtime=50.0,  # 27% of 180s, under hold threshold
            tier_timeout=180.0,
            base_workers=32,
            active_workers=8,
            throughput=1.5,
            avg_runtime=40.0,
            completion_count=10,
            avg_base_time=5.0,  # 117/5 = 23.4 → 23 optimal
        )
        assert decision.should_throttle
        # Optimal is 23, ramp from 8: 8 + 1 = 9 (MAX_RAMP_INCREMENT=1)
        assert decision.recommended_workers == 9
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
        increment = min(max(1, int(21*0.25)), 1) = 1 (MAX_RAMP_INCREMENT=1)
        ramped = 2 + 1 = 3
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
        # Optimal is 23, but we ramp with max increment of 1: 2 + 1 = 3
        assert decision.recommended_workers == 3
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


class TestThroughputByLevel:
    """Tests for ThroughputByLevel class.

    Tracks throughput at each concurrency level to find the peak — the
    concurrency where tasks/second is maximized.
    """

    def test_empty_returns_none(self):
        """Empty tracker has no peak."""
        tracker = ThroughputByLevel()
        assert tracker.get_peak_level() is None

    def test_single_level_returns_none(self):
        """Single level is insufficient — need >= 2 to detect a trend."""
        tracker = ThroughputByLevel()
        for _ in range(10):  # level 4 needs max(10, 8)=10
            tracker.record(4, 10.0)
        assert tracker.get_peak_level() is None

    def test_two_levels_finds_peak(self):
        """With 2 levels, finds the one with lowest base time.

        Level 2: runtime=4.0, base_time=4.0/2=2.0s  ← better
        Level 4: runtime=10.0, base_time=10.0/4=2.5s
        Both need max(10, level*2) samples: level 2→10, level 4→10.
        """
        tracker = ThroughputByLevel()
        for _ in range(10):
            tracker.record(2, 4.0)  # base = 2.0s (10 >= 10)
        for _ in range(10):
            tracker.record(4, 10.0)  # base = 2.5s (10 >= 10)

        result = tracker.get_peak_level()
        assert result is not None
        level, bt = result
        assert level == 2  # Lower base time = better per-worker cost
        assert bt == pytest.approx(2.0)

    def test_higher_concurrency_lower_throughput(self):
        """Detects when high concurrency has worse base time (GPU contention).

        Level 3: runtime=5.0, base_time=5.0/3=1.67s  ← better
        Level 6: runtime=15.0, base_time=15.0/6=2.5s  ← contention
        Level 3 needs max(10, 6)=10, level 6 needs max(10, 12)=12.
        """
        tracker = ThroughputByLevel()
        for _ in range(10):
            tracker.record(3, 5.0)  # base = 1.67s (10 >= 10)
        for _ in range(12):
            tracker.record(6, 15.0)  # base = 2.5s (12 >= 12)

        result = tracker.get_peak_level()
        assert result is not None
        level, _ = result
        assert level == 3

    def test_below_min_samples_not_qualified(self):
        """Levels with fewer than level-proportional min samples are ignored.

        Level 2 needs max(10, 4)=10 samples. Level 4 needs max(10, 8)=10.
        """
        tracker = ThroughputByLevel()
        # Level 2: 10 completions (qualified: 10 >= 10)
        for _ in range(10):
            tracker.record(2, 5.0)
        # Level 4: only 5 completions (not qualified: 5 < 10)
        for _ in range(5):
            tracker.record(4, 5.0)

        # Only 1 qualified level → returns None
        assert tracker.get_peak_level() is None

    def test_data_persists_over_time(self):
        """Data is kept for the entire tier lifetime (no time-based pruning)."""
        tracker = ThroughputByLevel()

        # Record data at two levels (both need 10 samples)
        for _ in range(10):
            tracker.record(2, 1.0)
            tracker.record(4, 2.0)

        # Verify peak exists
        assert tracker.get_peak_level() is not None

        # Wait a bit — data should still be there
        time.sleep(0.05)

        # Data persists (no time-based expiration)
        assert tracker.get_peak_level() is not None
        assert len(tracker.get_all_levels()) == 2

        # Only reset() clears it
        tracker.reset()
        assert tracker.get_peak_level() is None
        assert tracker.get_all_levels() == {}

    def test_reset_clears_all(self):
        """Reset should clear all level data."""
        tracker = ThroughputByLevel()
        for _ in range(10):
            tracker.record(2, 1.0)
            tracker.record(4, 2.0)

        assert tracker.get_peak_level() is not None
        tracker.reset()
        assert tracker.get_peak_level() is None
        assert tracker.get_all_levels() == {}

    def test_get_all_levels(self):
        """get_all_levels returns avg base time (runtime/level) and count.

        Note: get_all_levels shows ALL levels regardless of sample count.
        Only get_peak_level filters by min samples.
        """
        tracker = ThroughputByLevel()
        for _ in range(3):
            tracker.record(2, 5.0)  # base_time = 5.0/2 = 2.5
        for _ in range(5):
            tracker.record(4, 8.0)  # base_time = 8.0/4 = 2.0

        levels = tracker.get_all_levels()
        assert 2 in levels
        assert 4 in levels
        assert levels[2][0] == pytest.approx(2.5)  # avg base time at level 2
        assert levels[2][1] == 3  # 3 samples at level 2
        assert levels[4][0] == pytest.approx(2.0)  # avg base time at level 4
        assert levels[4][1] == 5  # 5 samples at level 4

    def test_concurrency_zero_clamped_to_one(self):
        """Concurrency 0 (warmup) should be clamped to 1."""
        tracker = ThroughputByLevel()
        tracker.record(0, 5.0)

        levels = tracker.get_all_levels()
        assert 1 in levels
        assert 0 not in levels

    def test_three_levels_peak_in_middle(self):
        """Peak can be at a middle concurrency level (lowest base time).

        Level 1: runtime=5.0, base_time=5.0/1=5.0s  (underutilized, needs 10)
        Level 3: runtime=3.0, base_time=3.0/3=1.0s  (sweet spot, needs 10) ← best
        Level 6: runtime=12.0, base_time=12.0/6=2.0s (contention, needs 12)
        """
        tracker = ThroughputByLevel()
        for _ in range(10):
            tracker.record(1, 5.0)  # base = 5.0s (10 >= 10)
        for _ in range(10):
            tracker.record(3, 3.0)  # base = 1.0s (10 >= 10)
        for _ in range(12):
            tracker.record(6, 12.0)  # base = 2.0s (12 >= 12)

        result = tracker.get_peak_level()
        assert result is not None
        level, _ = result
        assert level == 3  # Lowest base time = best per-worker efficiency


class TestThroughputTrackerPeakIntegration:
    """Tests for ThroughputTracker.get_peak_concurrency() integration."""

    def test_peak_none_with_no_data(self):
        """No data → no peak."""
        tracker = ThroughputTracker(window_seconds=10.0)
        assert tracker.get_peak_concurrency() is None

    def test_peak_none_with_single_level(self):
        """Data at only one concurrency level → no peak."""
        tracker = ThroughputTracker(window_seconds=10.0)
        for _ in range(5):
            tracker.record_completion(5.0, concurrent_workers=4)
        assert tracker.get_peak_concurrency() is None

    def test_peak_detected_with_two_levels(self):
        """Data at two levels → peak detected.

        Level 2 needs max(10, 4)=10, level 6 needs max(10, 12)=12.
        """
        tracker = ThroughputTracker(window_seconds=10.0)
        # Level 2: 10 completions (10 >= 10)
        for _ in range(10):
            tracker.record_completion(3.0, concurrent_workers=2)
        # Level 6: 12 completions (12 >= 12)
        for _ in range(12):
            tracker.record_completion(10.0, concurrent_workers=6)

        peak = tracker.get_peak_concurrency()
        assert peak is not None
        assert peak == 2  # base=3.0/2=1.5s < 10.0/6=1.67s → lower base time

    def test_reset_clears_peak(self):
        """Reset clears both regular stats and peak data."""
        tracker = ThroughputTracker(window_seconds=10.0)
        for _ in range(10):
            tracker.record_completion(3.0, concurrent_workers=2)
        for _ in range(12):
            tracker.record_completion(10.0, concurrent_workers=6)

        assert tracker.get_peak_concurrency() is not None
        tracker.reset()
        assert tracker.get_peak_concurrency() is None


class TestPeakConcurrencyThrottleDecision:
    """Tests for peak_concurrency parameter in compute_throttle_decision.

    When peak throughput data is available, it becomes the primary optimization
    target — but the formula-based limit acts as a safety cap.
    """

    def test_peak_none_falls_back_to_formula(self):
        """When peak_concurrency is None, behavior is identical to formula-only.

        avg_base_time=20s → formula_optimal = 117/20 = 5
        """
        decision = compute_throttle_decision(
            current_max_runtime=0.0,
            tier_timeout=180.0,
            base_workers=32,
            active_workers=0,
            avg_base_time=20.0,
            completion_count=10,
            peak_concurrency=None,
        )
        assert decision.recommended_workers == 5
        assert decision.peak_concurrency is None

    def test_peak_below_formula_uses_peak(self):
        """When peak < formula_optimal, use peak (it's the real sweet spot).

        formula_optimal = 117/5 = 23 workers (safety cap)
        peak = 4 workers (actual throughput sweet spot)
        → should use 4
        """
        decision = compute_throttle_decision(
            current_max_runtime=0.0,
            tier_timeout=180.0,
            base_workers=32,
            active_workers=0,
            avg_base_time=5.0,
            completion_count=10,
            peak_concurrency=4,
        )
        assert decision.should_throttle
        assert decision.recommended_workers == 4
        assert decision.peak_concurrency == 4
        assert "peak@4" in decision.reason

    def test_peak_above_formula_capped_by_formula(self):
        """When peak > formula_optimal, formula acts as safety cap.

        formula_optimal = 117/20 = 5 workers (safety cap)
        peak = 8 workers (but formula says that would risk timeouts)
        → should use 5
        """
        decision = compute_throttle_decision(
            current_max_runtime=0.0,
            tier_timeout=180.0,
            base_workers=32,
            active_workers=0,
            avg_base_time=20.0,
            completion_count=10,
            peak_concurrency=8,
        )
        assert decision.should_throttle
        assert decision.recommended_workers == 5
        assert decision.peak_concurrency == 8

    def test_peak_equals_base_workers_no_throttle(self):
        """When peak >= base_workers and formula allows, no throttle needed.

        formula_optimal = 117/0.5 = 234 → capped at 8 → no throttle
        peak = 8 → matches base_workers → no throttle
        """
        decision = compute_throttle_decision(
            current_max_runtime=5.0,
            tier_timeout=180.0,
            base_workers=8,
            active_workers=8,
            avg_base_time=0.5,
            completion_count=10,
            peak_concurrency=8,
        )
        assert not decision.should_throttle
        assert decision.recommended_workers == 8

    def test_peak_with_ramp_up(self):
        """Peak should work with ramp-up logic (gradual increase).

        peak = 6, formula = 23, active = 3
        target = min(6, 23) = 6, ramp from 3: 3 + 1 = 4
        """
        decision = compute_throttle_decision(
            current_max_runtime=10.0,  # Under hold threshold
            tier_timeout=180.0,
            base_workers=32,
            active_workers=3,
            avg_base_time=5.0,
            completion_count=10,
            peak_concurrency=6,
        )
        assert decision.should_throttle
        assert decision.recommended_workers == 4  # Ramped from 3 toward 6
        assert "ramped from 3" in decision.reason

    def test_peak_one_minimizes_workers(self):
        """Peak at 1 means highest throughput is serial — use 1 worker."""
        decision = compute_throttle_decision(
            current_max_runtime=0.0,
            tier_timeout=180.0,
            base_workers=8,
            active_workers=0,
            avg_base_time=5.0,
            completion_count=10,
            peak_concurrency=1,
        )
        assert decision.should_throttle
        assert decision.recommended_workers == 1

    def test_emergency_overrides_peak(self):
        """Emergency throttle (>60% timeout) still overrides peak."""
        decision = compute_throttle_decision(
            current_max_runtime=110.0,  # 61% of 180
            tier_timeout=180.0,
            base_workers=8,
            active_workers=4,
            avg_base_time=5.0,
            completion_count=10,
            peak_concurrency=6,
        )
        assert decision.should_throttle
        assert decision.recommended_workers == 1
        assert "Emergency" in decision.reason

    def test_peak_capped_by_base_workers(self):
        """Peak cannot exceed base_workers even if throughput says so.

        peak = 20, base_workers = 8 → capped at 8
        formula = 117/0.5 = 234 → capped at 8
        → no throttle, 8 workers
        """
        decision = compute_throttle_decision(
            current_max_runtime=5.0,
            tier_timeout=180.0,
            base_workers=8,
            active_workers=8,
            avg_base_time=0.5,
            completion_count=10,
            peak_concurrency=20,
        )
        assert not decision.should_throttle
        assert decision.recommended_workers == 8


def _render_to_text(renderable: object) -> str:
    """Render a Rich renderable to plain text for testing."""
    from io import StringIO

    from rich.console import Console

    buf = StringIO()
    console = Console(file=buf, force_terminal=False, width=200, no_color=True)
    console.print(renderable)
    return buf.getvalue()


class TestFormatThroughputLevels:
    """Tests for color-graded table visualization (format_throughput_levels and build_throughput_levels_text)."""

    def test_empty_returns_empty(self):
        """None or empty dict returns empty string."""
        assert format_throughput_levels(None) == ""
        assert format_throughput_levels({}) == ""

    def test_single_level_renders_table(self):
        """Single level renders a table with level number and base time."""
        result = format_throughput_levels({1: (2.5, 3)})
        assert result != ""  # Not empty string
        text = _render_to_text(result)
        assert "1" in text
        assert "2.5s" in text

    def test_multiple_levels_with_peak(self):
        """Multiple levels show level numbers and base times."""
        levels = {1: (2.1, 3), 2: (3.4, 5), 3: (1.3, 8), 4: (6.2, 4)}
        result = format_throughput_levels(levels, peak_concurrency=3)
        text = _render_to_text(result)
        # All level numbers should appear
        assert " 1 " in text
        assert " 2 " in text
        assert " 3 " in text
        assert " 4 " in text
        # Base times should appear
        assert "1.3s" in text  # Peak level's time
        assert "6.2s" in text  # Worst level's time

    def test_max_workers_shows_empty_slots(self):
        """max_workers > max level shows placeholders for empty slots."""
        levels = {1: (2.0, 3), 2: (3.0, 5)}
        result = format_throughput_levels(levels, max_workers=5)
        text = _render_to_text(result)
        # Levels 3, 4, 5 have no data → show "···" placeholders
        assert "···" in text
        # But levels 1 and 2 should have times
        assert "2.0s" in text
        assert "3.0s" in text

    def test_max_workers_zero_uses_max_level(self):
        """max_workers=0 defaults to highest level in data."""
        levels = {1: (2.0, 3), 3: (4.0, 5)}
        result = format_throughput_levels(levels, max_workers=0)
        text = _render_to_text(result)
        # Positions 1, 2, 3 — level 2 missing (placeholder)
        assert "···" in text
        assert "2.0s" in text
        assert "4.0s" in text

    def test_color_gradient_best_is_green(self):
        """Best level (lowest base time) should get green color."""
        from shared.throttling import _lerp_color
        green = _lerp_color(0.0)  # Best
        red = _lerp_color(1.0)  # Worst
        assert green == "#00c800"  # (0, 200, 0)
        assert red == "#dc0000"  # (220, 0, 0)

    def test_color_gradient_midpoint_is_yellow(self):
        """Midpoint should be yellow-ish."""
        from shared.throttling import _lerp_color
        mid = _lerp_color(0.5)
        assert mid == "#dcc800"

    def test_text_color_contrast(self):
        """Light backgrounds get black text, dark backgrounds get white."""
        from shared.throttling import _lerp_color, _text_color_for_bg
        # Green (#00c800) is light → black text
        assert _text_color_for_bg(_lerp_color(0.0)) == "black"
        # Yellow (#dcc800) is light → black text
        assert _text_color_for_bg(_lerp_color(0.5)) == "black"
        # Red (#dc0000) is dark → white text
        assert _text_color_for_bg(_lerp_color(1.0)) == "white"

    def test_peak_without_levels_returns_empty(self):
        """Peak concurrency without level data returns empty."""
        assert format_throughput_levels(None, peak_concurrency=3) == ""

    def test_build_text_returns_table(self):
        """build_throughput_levels_text returns a Rich Table with level data."""
        levels = {1: (2.5, 3), 2: (3.0, 5)}
        result = build_throughput_levels_text(levels, peak_concurrency=2)
        assert result is not None
        text = _render_to_text(result)
        assert "1" in text
        assert "2" in text
        assert "2.5s" in text
        assert "3.0s" in text

    def test_build_text_with_max_workers(self):
        """build_throughput_levels_text respects max_workers for empty slots."""
        levels = {1: (2.0, 3)}
        result = build_throughput_levels_text(levels, max_workers=4)
        assert result is not None
        text = _render_to_text(result)
        # Level 1 has data, levels 2-4 are placeholders
        assert "2.0s" in text
        assert "···" in text

    def test_build_text_none_returns_none(self):
        """build_throughput_levels_text with None returns None."""
        assert build_throughput_levels_text(None) is None

    def test_build_text_empty_returns_none(self):
        """build_throughput_levels_text with empty dict returns None."""
        assert build_throughput_levels_text({}) is None

    def test_single_chunk_returns_table(self):
        """≤16 levels returns a single Table with 3 rows (num + time + count)."""
        from shared.throttling import _build_levels_row
        levels = {1: (2.0, 3), 2: (1.5, 5)}
        table = _build_levels_row(range(1, 3), levels, 1.5, 0.5, None)
        assert table.row_count == 3
        # 1 indent column + 2 level columns = 3
        assert len(table.columns) == 3

    def test_wraps_after_16_levels(self):
        """More than 16 levels wraps into multiple rows."""
        levels = {i: (float(i), 3) for i in range(1, 21)}  # 20 levels
        result = format_throughput_levels(levels, max_workers=20)
        text = _render_to_text(result)
        # All 20 level numbers should appear
        for i in range(1, 21):
            assert str(i) in text
        # Both chunks' base times should appear
        assert "1.0s" in text
        assert "20.0s" in text

    def test_color_consistent_across_chunks(self):
        """Color scaling is consistent across wrapped rows (uses global min/max)."""
        from shared.throttling import _build_levels_table
        from rich.console import Group
        # 20 levels: level 1 = 1.0s (best), level 20 = 20.0s (worst)
        levels = {i: (float(i), 3) for i in range(1, 21)}
        result = _build_levels_table(levels, max_workers=20)
        # Should be a Group (>16 levels)
        assert isinstance(result, Group)

    def test_confidence_style(self):
        """Text style varies by sample count: dim(1), normal(2), bold(3+)."""
        from shared.throttling import _confidence_style
        assert _confidence_style(1) == "dim"
        assert _confidence_style(2) == ""
        assert _confidence_style(3) == "bold"
        assert _confidence_style(10) == "bold"

    def test_sample_count_row_displayed(self):
        """Third row shows sample count (without n= prefix) for each level."""
        levels = {1: (2.0, 3), 2: (3.0, 7), 3: (4.0, 1)}
        result = format_throughput_levels(levels, max_workers=3)
        text = _render_to_text(result)
        # Counts appear as plain numbers, not "n=X"
        assert "n=" not in text
        assert " 3 " in text
        assert " 7 " in text

    def test_exploration_target_marked_with_question(self):
        """Exploration target level shows '?' prefix."""
        levels = {1: (2.0, 3), 2: (3.0, 5)}
        result = format_throughput_levels(levels, max_workers=4, exploration_target=3)
        text = _render_to_text(result)
        assert "?3" in text

    def test_exploration_target_with_data(self):
        """Exploration target shows '?' even when it has data."""
        levels = {1: (2.0, 3), 2: (3.0, 5), 3: (4.0, 1)}
        result = format_throughput_levels(levels, max_workers=3, exploration_target=3)
        text = _render_to_text(result)
        assert "?3" in text

    def test_all_same_base_time(self):
        """All levels with same base time renders without errors."""
        levels = {1: (3.0, 5), 2: (3.0, 5), 3: (3.0, 5)}
        result = format_throughput_levels(levels, peak_concurrency=2, max_workers=3)
        text = _render_to_text(result)
        # All same time → no division error, all levels shown
        assert "3.0s" in text


class TestExplorationTarget:
    """Tests for ThroughputByLevel.get_exploration_target().

    Exploration: when peak is confident but levels within ±max(3, max_level // 3)
    lack data, return one to explore (collect more samples before trusting peak).

    Significance = max(10, level * 2). Floor of 10 ensures enough data.
    Level 1 → 10, level 5 → 10, level 6 → 12, level 10 → 20.
    Radius = max(3, max_level // 3).
    """

    def test_no_peak_returns_none(self):
        """No peak (insufficient data) → no exploration."""
        tracker = ThroughputByLevel()
        for _ in range(10):
            tracker.record(2, 5.0)
        # Only 1 level → no peak → no exploration
        assert tracker.get_exploration_target(max_level=8) is None

    def test_peak_not_confident_returns_none(self):
        """Peak exists but has < 10 samples → don't explore from weak peak.

        Both levels need at least 10 samples (min floor). Level 2 has only 9.
        Since get_peak_level now also uses level-proportional thresholds,
        the peak won't even be detected with insufficient samples.
        """
        tracker = ThroughputByLevel()
        # Level 2: 9 samples (below min floor 10 — not even qualified as peak)
        for _ in range(9):
            tracker.record(2, 4.0)
        # Level 4: 5 samples (also below 10)
        for _ in range(5):
            tracker.record(4, 10.0)
        # No peak detected (neither level qualifies) → no exploration
        assert tracker.get_exploration_target(max_level=8) is None

    def test_explore_upward_nearest(self):
        """Peak confident, nearest above missing → returns peak + 1.

        Peak at level 2 needs 10. max_level=8 → radius=max(3,2)=3, range 1..5.
        """
        tracker = ThroughputByLevel()
        for _ in range(11):
            tracker.record(2, 4.0)  # confident peak (11 >= 10)
        for _ in range(10):
            tracker.record(4, 10.0)  # another level (10 >= 10)
        # Level 3 has no data → explore upward
        assert tracker.get_exploration_target(max_level=8) == 3

    def test_explore_upward_skips_explored(self):
        """Peak confident, nearest above explored but peak+2 missing → returns peak+2.

        Peak at level 2 (10 needed). max_level=8 → radius=3, range=[1,5].
        Levels 1, 3, 5 explored. Level 4 is the only unexplored candidate.
        """
        tracker = ThroughputByLevel()
        for _ in range(11):
            tracker.record(2, 4.0)  # confident peak
        for _ in range(10):
            tracker.record(1, 5.0)  # level 1 explored (10 >= 10)
        for _ in range(10):
            tracker.record(3, 6.0)  # level 3 explored (10 >= 10)
        for _ in range(10):
            tracker.record(5, 12.0)  # level 5 explored (10 >= 10)
        # Level 4 has no data (needs 10) → only unexplored candidate
        assert tracker.get_exploration_target(max_level=8) == 4

    def test_explore_downward_when_above_explored(self):
        """All above in range explored, below missing → returns peak - 1.

        Peak at level 4 (needs 10). max_level=8 → radius=3, range=1..7.
        Levels 5-7 explored. Level 3 (needs 10) has no data.
        """
        tracker = ThroughputByLevel()
        for _ in range(12):
            tracker.record(4, 4.0)  # peak at 4 (12 >= 10)
        for _ in range(10):
            tracker.record(5, 5.0)  # above explored (10 >= 10)
        for _ in range(12):
            tracker.record(6, 6.0)  # above explored (12 >= 12)
        for _ in range(14):
            tracker.record(7, 7.0)  # above explored (14 >= 14)
        for _ in range(10):
            tracker.record(2, 3.0)  # other level (10 >= 10)
        # Level 3 has no data (needs 10) → explore downward
        assert tracker.get_exploration_target(max_level=8) == 3

    def test_all_in_range_explored_returns_none(self):
        """All levels within ±radius have enough data → no exploration.

        Peak at level 4. max_level=8 → radius=3, range=1..7.
        All levels 1-7 have max(10, level*2) samples.
        """
        tracker = ThroughputByLevel()
        for level in range(1, 8):
            needed = max(10, level * 2) + 2
            for _ in range(needed):
                tracker.record(level, float(level) * 2)
        assert tracker.get_exploration_target(max_level=8) is None

    def test_peak_at_level_1_only_explores_above(self):
        """Peak at 1 → no below to check, only explores above.

        Peak at level 1 needs 10. max_level=8 → radius=3, range 1..4.
        """
        tracker = ThroughputByLevel()
        for _ in range(11):
            tracker.record(1, 3.0)  # peak at 1 (11 >= 10)
        for _ in range(10):
            tracker.record(3, 9.0)  # level 3 explored (10 >= 10)
        # Level 2 has no data (needs 10) → explore upward
        assert tracker.get_exploration_target(max_level=8) == 2

    def test_capped_by_max_level(self):
        """Exploration doesn't return levels above max_level.

        Peak at level 3 needs 10. max_level=3 → radius=max(3,1)=3.
        Upward capped at 3 (max_level). Only downward level 2 and 1 in range.
        """
        tracker = ThroughputByLevel()
        for _ in range(11):
            tracker.record(3, 3.0)  # peak at 3 (11 >= 10)
        for _ in range(10):
            tracker.record(1, 5.0)  # level 1 explored (10 >= 10)
        # max_level=3 → upward capped at 3, downward: level 2 has no data → return 2
        assert tracker.get_exploration_target(max_level=3) == 2

    def test_capped_by_max_level_all_in_range_explored(self):
        """At max_level boundary with all in range explored → returns None."""
        tracker = ThroughputByLevel()
        for _ in range(11):
            tracker.record(3, 3.0)  # peak at 3 (11 >= 10)
        for _ in range(10):
            tracker.record(1, 5.0)  # level 1 (10 >= 10)
        for _ in range(10):
            tracker.record(2, 4.0)  # level 2 (10 >= 10)
        # max_level=3 → radius=3, all within range explored
        assert tracker.get_exploration_target(max_level=3) is None

    def test_radius_is_third_of_max_level(self):
        """Radius = max(3, max_level // 3). With max_level=12, radius=4.

        Peak at level 3 (needs 10). Levels 4-7 are within range (3+4=7).
        Levels 4-6 explored. Level 7 needs 14 (7*2).
        """
        tracker = ThroughputByLevel()
        for _ in range(11):
            tracker.record(3, 3.0)  # peak at 3 (11 >= 10)
        for _ in range(10):
            tracker.record(4, 4.0)  # explored (10 >= 10)
        for _ in range(10):
            tracker.record(5, 5.0)  # explored (10 >= 10)
        for _ in range(12):
            tracker.record(6, 6.0)  # explored (12 >= 12)
        for _ in range(10):
            tracker.record(1, 6.0)  # level 1 explored (10 >= 10)
        for _ in range(10):
            tracker.record(2, 5.0)  # level 2 explored (10 >= 10)
        # Level 7 = peak+4, within radius(4), needs 14 (7*2), has 0 → explore
        assert tracker.get_exploration_target(max_level=12) == 7

    def test_beyond_max_level_not_explored(self):
        """Levels beyond max_level are not explored.

        Peak at level 5. max_level=10 → radius=3, range=2..8.
        Fill all levels 2-8 → returns None (levels 1 and 9-10 beyond radius).
        """
        tracker = ThroughputByLevel()
        # Need peak at 5 (lowest base time). Make level 5 the best.
        for _ in range(11):
            tracker.record(5, 2.5)  # base=2.5/5=0.5 (best)
        for level in [2, 3, 4, 6, 7, 8]:
            needed = max(10, level * 2) + 1
            for _ in range(needed):
                tracker.record(level, float(level) * 1.5)  # worse base times
        assert tracker.get_exploration_target(max_level=10) is None

    def test_exploration_stops_after_collecting_data(self):
        """Once level gets enough samples, exploration moves to next.

        Peak at level 4 (needs 10). max_level=8 → radius=3, range=[1,7].
        Progressively fill levels and verify exploration advances.
        """
        tracker = ThroughputByLevel()
        for _ in range(12):
            tracker.record(4, 4.0)  # peak (12 >= 10)
        for _ in range(10):
            tracker.record(2, 6.0)  # level 2 explored (10 >= 10)

        # Level 5 (nearest above, prox=0.67) → explore
        assert tracker.get_exploration_target(max_level=8) == 5

        # Fill level 5
        for _ in range(10):
            tracker.record(5, 5.0)

        # Level 5 explored → next candidates: 3 (prox=0.67) and 6 (prox=0.33)
        # Level 3 wins on proximity (same distance, but upward-first tiebreak gives 5→explored, 3 wins)
        result = tracker.get_exploration_target(max_level=8)
        assert result in (3, 6)  # Either is valid depending on scoring balance

        # Fill the remaining explored target
        for _ in range(max(10, result * 2)):
            tracker.record(result, float(result))

    def test_level_proportional_significance(self):
        """Significance = max(10, level * 2).

        Low levels hit the floor of 10. High levels use level * 2.
        """
        from shared.throttling import ThroughputByLevel as TBL
        assert TBL._min_samples_for_level(1) == 10  # max(10, 2) = 10
        assert TBL._min_samples_for_level(2) == 10  # max(10, 4) = 10
        assert TBL._min_samples_for_level(5) == 10  # max(10, 10) = 10
        assert TBL._min_samples_for_level(6) == 12  # max(10, 12) = 12
        assert TBL._min_samples_for_level(10) == 20  # max(10, 20) = 20

    def test_small_max_level_has_minimum_radius_of_3(self):
        """max_level=2 → radius=max(3, 0)=3. Radius always at least 3.

        Peak needs both levels to exist. Both explored → None.
        """
        tracker = ThroughputByLevel()
        for _ in range(11):
            tracker.record(1, 3.0)  # peak at 1 (11 >= 10)
        for _ in range(10):
            tracker.record(2, 6.0)  # level 2 (10 >= 10)
        # max_level=2, radius=3 but capped by max_level → range 1..2, both explored
        assert tracker.get_exploration_target(max_level=2) is None

    def test_radius_limits_downward_exploration(self):
        """Downward exploration is capped by radius.

        Peak at level 8 (needs 16). max_level=10 → radius=max(3, 3)=3.
        Lower bound = max(8-3, 1) = 5. Level 4 is below radius → not explored.
        """
        tracker = ThroughputByLevel()
        for _ in range(16):
            tracker.record(8, 4.0)  # peak at 8 (16 >= 16)
        for _ in range(18):
            tracker.record(9, 5.0)  # explored (18 >= 18)
        for _ in range(20):
            tracker.record(10, 6.0)  # explored (20 >= 20)
        # Fill levels 6-7 (within radius 3 downward from 8: 5..7)
        for level in [6, 7]:
            needed = max(10, level * 2) + 1
            for _ in range(needed):
                tracker.record(level, float(level))
        # Level 5 = 8-3 = lower boundary, needs 10, has 0 → explore
        assert tracker.get_exploration_target(max_level=10) == 5
        # Level 4 is below radius (8-3=5, level 4 < 5) → NOT explored
        # If we fill level 5:
        for _ in range(10):
            tracker.record(5, 5.0)
        # All within radius explored → returns None (level 4 is outside radius)
        assert tracker.get_exploration_target(max_level=10) is None

    def test_large_max_level_scales_radius(self):
        """With max_level=30, radius=max(3, 10)=10. Scales for large pools.

        Peak at 15. Levels 14 and 16 both at distance 1, but level 14
        needs fewer samples (28 vs 32) → slightly higher cost score → wins.
        """
        tracker = ThroughputByLevel()
        # Peak at 15 needs max(10, 15*2)=30 samples to be confident
        for _ in range(31):
            tracker.record(15, 1.5)  # peak at 15: base=1.5/15=0.1 (best)
        for _ in range(20):
            tracker.record(10, 5.0)  # level 10: base=5.0/10=0.5 (worse, 20 >= 20)
        # radius = 30 // 3 = 10, range=[5,25]
        # Level 14 (needs 28) and 16 (needs 32) are equidistant from peak.
        # Level 14 wins on cost score (cheaper to explore).
        assert tracker.get_exploration_target(max_level=30) == 14


class TestExplorationScoring:
    """Tests for scoring-based exploration target selection.

    Verifies that the scoring formula correctly weighs completion progress,
    proximity to peak, trend direction, and exploration cost.
    """

    def test_prefers_partially_explored_over_empty(self):
        """Level with partial data scores higher than empty level at same distance.

        Peak at 4. Level 5 has 5/10 samples (50% complete).
        Level 3 has 0/10 (0%). Both at distance 1, but level 5 wins
        on completion score (0.40 weight).
        """
        tracker = ThroughputByLevel()
        for _ in range(12):
            tracker.record(4, 4.0)  # peak (12 >= 10)
        for _ in range(10):
            tracker.record(2, 6.0)  # second level for peak detection
        for _ in range(5):
            tracker.record(5, 5.0)  # partial data (5/10)
        # Level 5 at 50% completion beats level 3 at 0%
        assert tracker.get_exploration_target(max_level=8) == 5

    def test_trend_upward_prefers_higher_level(self):
        """When base time decreases at higher levels, explore upward.

        Level 2: base=3.0, Level 4: base=2.0, Level 6: base=1.5 (peak)
        Trend is negative (improving upward) → prefer level 7 over level 5.
        """
        tracker = ThroughputByLevel()
        for _ in range(10):
            tracker.record(2, 6.0)   # base=3.0
        for _ in range(10):
            tracker.record(4, 8.0)   # base=2.0
        for _ in range(12):
            tracker.record(6, 9.0)   # base=1.5 (peak)
        # 3 explored levels → reliable trend, slope < 0
        # Level 7 (above peak, trend-aligned) beats level 5 (below peak)
        result = tracker.get_exploration_target(max_level=10)
        assert result == 7

    def test_trend_downward_prefers_lower_level(self):
        """When base time increases at higher levels, explore downward.

        Level 2: base=1.0 (peak), Level 4: base=2.0, Level 6: base=3.0
        Trend is positive (worsening upward) → prefer level 1 over level 3.
        """
        tracker = ThroughputByLevel()
        for _ in range(10):
            tracker.record(2, 2.0)   # base=1.0 (peak)
        for _ in range(10):
            tracker.record(4, 8.0)   # base=2.0
        for _ in range(12):
            tracker.record(6, 18.0)  # base=3.0
        # Trend positive (worsening upward) → explore below peak
        result = tracker.get_exploration_target(max_level=10)
        assert result == 1

    def test_insufficient_trend_data_no_bias(self):
        """With only 2 explored levels, trend is neutral (score=0.5 for all).

        Selection falls back to completion + proximity + cost.
        Upward-first tiebreak applies when scores are equal.
        """
        tracker = ThroughputByLevel()
        for _ in range(11):
            tracker.record(4, 4.0)  # peak
        for _ in range(10):
            tracker.record(2, 6.0)  # second level
        # Only 2 explored levels → no trend signal
        # Level 5 (dist=1, above peak) and level 3 (dist=1, below peak)
        # have same proximity and neutral trend → upward-first tiebreak → 5
        result = tracker.get_exploration_target(max_level=8)
        assert result == 5

    def test_completion_dominates_when_nearly_done(self):
        """A level at 85% completion beats a closer level at 0%.

        Peak at 5. Level 7 has 12/14 data (distance=2, 85% complete).
        Level 6 has 0/12 data (distance=1, 0% complete).
        Completion weight (0.40) overcomes proximity advantage.
        """
        tracker = ThroughputByLevel()
        for _ in range(11):
            tracker.record(5, 5.0)  # peak
        for _ in range(10):
            tracker.record(3, 9.0)  # second level for peak
        for _ in range(12):
            tracker.record(7, 10.5)  # partial: 12/14 needed (85%)
        # Level 7 completion=0.857 vs level 6 completion=0.0
        # Completion advantage (0.857 * 0.40 = 0.343) beats
        # proximity advantage (0.67 vs 0.33 → 0.34 * 0.25 = 0.085)
        result = tracker.get_exploration_target(max_level=10)
        assert result == 7

    def test_budget_filters_before_scoring(self):
        """Budget check eliminates candidates before scoring.

        With remaining=15, levels needing 10 samples require 10*3=30 > 15 → filtered.
        """
        tracker = ThroughputByLevel()
        for _ in range(11):
            tracker.record(3, 3.0)  # peak
        for _ in range(10):
            tracker.record(1, 5.0)  # second level
        # All candidates need 10 samples: 10*3=30 > 15 → all filtered
        assert tracker.get_exploration_target(max_level=8, remaining=15) is None

    def test_budget_partial_data_passes_filter(self):
        """Partial data reduces samples_left, passing budget filter.

        Level 4 needs 10 samples, has 6 → only 4 left → 4*3=12 <= 15 → passes.
        """
        tracker = ThroughputByLevel()
        for _ in range(11):
            tracker.record(3, 3.0)  # peak
        for _ in range(10):
            tracker.record(1, 5.0)  # second level
        for _ in range(6):
            tracker.record(4, 6.0)  # partial: 6/10
        # Level 4 needs 4 more: 4*3=12 <= 15 → passes budget
        assert tracker.get_exploration_target(max_level=8, remaining=15) == 4

    def test_scoring_deterministic(self):
        """Same input always produces same output (no randomness)."""
        tracker = ThroughputByLevel()
        for _ in range(11):
            tracker.record(4, 4.0)
        for _ in range(10):
            tracker.record(2, 6.0)
        results = [tracker.get_exploration_target(max_level=8) for _ in range(10)]
        assert len(set(results)) == 1  # All identical

    def test_trend_with_flat_base_times(self):
        """All explored levels have identical base time → flat trend → neutral.

        No directional bias applied. Upward-first tiebreak applies.
        """
        tracker = ThroughputByLevel()
        for _ in range(10):
            tracker.record(2, 4.0)   # base=2.0
        for _ in range(10):
            tracker.record(4, 8.0)   # base=2.0
        for _ in range(12):
            tracker.record(6, 12.0)  # base=2.0
        # All base times equal → slope=0 → trend neutral
        result = tracker.get_exploration_target(max_level=10)
        assert result is not None

    def test_trend_computation_uses_only_explored_levels(self):
        """Trend ignores levels that haven't reached significance threshold.

        Only levels with enough samples contribute to the trend.
        """
        tracker = ThroughputByLevel()
        for _ in range(10):
            tracker.record(2, 6.0)   # base=3.0 (explored)
        for _ in range(10):
            tracker.record(4, 8.0)   # base=2.0 (explored)
        for _ in range(12):
            tracker.record(6, 9.0)   # base=1.5 (peak, explored)
        # Add partial data at level 8 (only 3 samples, needs 16)
        for _ in range(3):
            tracker.record(8, 40.0)  # base=5.0 but NOT explored (3 < 16)
        # Trend should still be negative (improving upward) from levels 2,4,6
        # Level 8's high base time should NOT pollute the trend
        result = tracker.get_exploration_target(max_level=10)
        assert result == 7  # Still prefers upward (trend-aligned)


class TestExplorationInThrottleDecision:
    """Tests for exploration_target parameter in compute_throttle_decision."""

    def test_exploration_overrides_peak(self):
        """Exploration target overrides peak as optimization target.

        peak=4, exploration=6, formula=23 → target=6 (exploring beyond peak)
        """
        decision = compute_throttle_decision(
            current_max_runtime=0.0,
            tier_timeout=180.0,
            base_workers=32,
            active_workers=0,
            avg_base_time=5.0,
            completion_count=10,
            peak_concurrency=4,
            exploration_target=6,
        )
        assert decision.should_throttle
        assert decision.recommended_workers == 6
        assert decision.exploration_target == 6
        assert "explore@6" in decision.reason

    def test_exploration_capped_by_formula(self):
        """Formula safety cap still limits exploration target.

        formula=5, exploration=8 → capped to 5
        """
        decision = compute_throttle_decision(
            current_max_runtime=0.0,
            tier_timeout=180.0,
            base_workers=32,
            active_workers=0,
            avg_base_time=20.0,  # formula = 117/20 = 5
            completion_count=10,
            peak_concurrency=3,
            exploration_target=8,
        )
        assert decision.recommended_workers == 5

    def test_exploration_none_falls_back_to_peak(self):
        """When exploration is None, peak is used (existing behavior)."""
        decision = compute_throttle_decision(
            current_max_runtime=0.0,
            tier_timeout=180.0,
            base_workers=32,
            active_workers=0,
            avg_base_time=5.0,
            completion_count=10,
            peak_concurrency=4,
            exploration_target=None,
        )
        assert decision.recommended_workers == 4
        assert decision.exploration_target is None
        assert "peak@4" in decision.reason

    def test_emergency_overrides_exploration(self):
        """Emergency throttle overrides exploration."""
        decision = compute_throttle_decision(
            current_max_runtime=110.0,  # 61% of 180
            tier_timeout=180.0,
            base_workers=8,
            active_workers=4,
            avg_base_time=5.0,
            completion_count=10,
            peak_concurrency=4,
            exploration_target=6,
        )
        assert decision.recommended_workers == 1
        assert "Emergency" in decision.reason

    def test_exploration_with_ramp_up(self):
        """Ramp-up still applies when exploring.

        exploration=6, active=3 → ramp from 3: 3+1=4
        """
        decision = compute_throttle_decision(
            current_max_runtime=10.0,
            tier_timeout=180.0,
            base_workers=32,
            active_workers=3,
            avg_base_time=5.0,
            completion_count=10,
            peak_concurrency=4,
            exploration_target=6,
        )
        assert decision.recommended_workers == 4  # Ramped from 3 toward 6
        assert "ramped from 3" in decision.reason
