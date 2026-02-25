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
        tracker = ThroughputByLevel(min_samples=2)
        for _ in range(5):
            tracker.record(4, 10.0)
        assert tracker.get_peak_level() is None

    def test_two_levels_finds_peak(self):
        """With 2 levels, finds the one with lowest base time.

        Level 2: runtime=5.0, base_time=5.0/2=2.5s
        Level 4: runtime=10.0, base_time=10.0/4=2.5s
        Same base time → either is valid. Make them different:
        Level 2: runtime=4.0, base_time=4.0/2=2.0s  ← better
        Level 4: runtime=10.0, base_time=10.0/4=2.5s
        """
        tracker = ThroughputByLevel(min_samples=3)
        for _ in range(5):
            tracker.record(2, 4.0)  # base = 2.0s
        for _ in range(3):
            tracker.record(4, 10.0)  # base = 2.5s

        result = tracker.get_peak_level()
        assert result is not None
        level, bt = result
        assert level == 2  # Lower base time = better per-worker cost
        assert bt == pytest.approx(2.0)

    def test_higher_concurrency_lower_throughput(self):
        """Detects when high concurrency has worse base time (GPU contention).

        Level 3: runtime=5.0, base_time=5.0/3=1.67s  ← better
        Level 6: runtime=15.0, base_time=15.0/6=2.5s  ← contention
        """
        tracker = ThroughputByLevel(min_samples=3)
        for _ in range(6):
            tracker.record(3, 5.0)  # base = 1.67s
        for _ in range(3):
            tracker.record(6, 15.0)  # base = 2.5s

        result = tracker.get_peak_level()
        assert result is not None
        level, _ = result
        assert level == 3

    def test_below_min_samples_not_qualified(self):
        """Levels with fewer than min_samples are ignored."""
        tracker = ThroughputByLevel(min_samples=3)
        # Level 2: 5 completions (qualified)
        for _ in range(5):
            tracker.record(2, 5.0)
        # Level 4: only 2 completions (not qualified)
        tracker.record(4, 5.0)
        tracker.record(4, 5.0)

        # Only 1 qualified level → returns None
        assert tracker.get_peak_level() is None

    def test_data_persists_over_time(self):
        """Data is kept for the entire tier lifetime (no time-based pruning)."""
        tracker = ThroughputByLevel(min_samples=2)

        # Record data at two levels
        for _ in range(3):
            tracker.record(2, 1.0)
            tracker.record(4, 1.0)

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
        tracker = ThroughputByLevel(min_samples=2)
        for _ in range(5):
            tracker.record(2, 1.0)
            tracker.record(4, 1.0)

        assert tracker.get_peak_level() is not None
        tracker.reset()
        assert tracker.get_peak_level() is None
        assert tracker.get_all_levels() == {}

    def test_get_all_levels(self):
        """get_all_levels returns avg base time (runtime/level) and count."""
        tracker = ThroughputByLevel(min_samples=1)
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
        tracker = ThroughputByLevel(min_samples=1)
        tracker.record(0, 5.0)

        levels = tracker.get_all_levels()
        assert 1 in levels
        assert 0 not in levels

    def test_three_levels_peak_in_middle(self):
        """Peak can be at a middle concurrency level (lowest base time).

        Level 1: runtime=5.0, base_time=5.0/1=5.0s  (underutilized)
        Level 3: runtime=3.0, base_time=3.0/3=1.0s  (sweet spot) ← best
        Level 6: runtime=12.0, base_time=12.0/6=2.0s (contention)
        """
        tracker = ThroughputByLevel(min_samples=2)
        for _ in range(2):
            tracker.record(1, 5.0)  # base = 5.0s
        for _ in range(6):
            tracker.record(3, 3.0)  # base = 1.0s
        for _ in range(3):
            tracker.record(6, 12.0)  # base = 2.0s

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
        """Data at two levels → peak detected."""
        tracker = ThroughputTracker(window_seconds=10.0)
        # Level 2: 5 completions
        for _ in range(5):
            tracker.record_completion(3.0, concurrent_workers=2)
        # Level 6: 3 completions
        for _ in range(3):
            tracker.record_completion(10.0, concurrent_workers=6)

        peak = tracker.get_peak_concurrency()
        assert peak is not None
        assert peak == 2  # base=3.0/2=1.5s < 10.0/6=1.67s → lower base time

    def test_reset_clears_peak(self):
        """Reset clears both regular stats and peak data."""
        tracker = ThroughputTracker(window_seconds=10.0)
        for _ in range(5):
            tracker.record_completion(3.0, concurrent_workers=2)
        for _ in range(3):
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
        """≤16 levels returns a single Table with 4 rows (border/num/time/border)."""
        from shared.throttling import _build_levels_row
        levels = {1: (2.0, 3), 2: (1.5, 5)}
        table = _build_levels_row(range(1, 3), levels, 1.5, 0.5, None)
        assert table.row_count == 4
        # 1 indent column + 2 level columns = 3
        assert len(table.columns) == 3

    def test_wraps_after_16_levels(self):
        """More than 16 levels wraps into multiple rows."""
        levels = {i: (float(i), 3) for i in range(1, 21)}  # 20 levels
        result = format_throughput_levels(levels, max_workers=20)
        text = _render_to_text(result)
        # All 20 level numbers should appear
        for i in range(1, 21):
            assert f" {i} " in text or str(i) in text
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

    def test_border_varies_by_sample_count(self):
        """Border style changes based on number of data points."""
        # 1 sample → dotted (┄), 2 → dashed (╌), 3 → thin (─), 4+ → bold (━)
        levels = {1: (2.0, 1), 2: (2.0, 2), 3: (2.0, 3), 4: (2.0, 5)}
        result = format_throughput_levels(levels, max_workers=4)
        text = _render_to_text(result)
        assert "┄" in text  # 1 sample: dotted
        assert "╌" in text  # 2 samples: dashed
        assert "─" in text  # 3 samples: thin solid
        assert "━" in text  # 4+ samples: bold

    def test_box_chars_function(self):
        """_box_chars returns correct characters for each confidence tier."""
        from shared.throttling import _box_chars
        # 1 sample: dotted
        tl, th, tr, side, *_ = _box_chars(1)
        assert th == "┄" and side == "┆"
        # 2 samples: dashed
        tl, th, tr, side, *_ = _box_chars(2)
        assert th == "╌" and side == "╎"
        # 3 samples: thin
        tl, th, tr, side, *_ = _box_chars(3)
        assert th == "─" and side == "│"
        # 4+ samples: bold
        tl, th, tr, side, *_ = _box_chars(4)
        assert th == "━" and side == "┃"
        tl, th, tr, side, *_ = _box_chars(100)
        assert th == "━" and side == "┃"

    def test_all_same_base_time(self):
        """All levels with same base time renders without errors."""
        levels = {1: (3.0, 5), 2: (3.0, 5), 3: (3.0, 5)}
        result = format_throughput_levels(levels, peak_concurrency=2, max_workers=3)
        text = _render_to_text(result)
        # All same time → no division error, all levels shown
        assert "3.0s" in text
