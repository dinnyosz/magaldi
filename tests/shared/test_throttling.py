"""Tests for throughput-based throttling module."""

import time

import pytest

from shared.throttling import (
    ThroughputTracker,
    ThrottleDecision,
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


class TestComputeThrottleDecision:
    """Tests for compute_throttle_decision function."""

    def test_no_data_returns_normal(self):
        """With no completion data, should return normal operation."""
        decision = compute_throttle_decision(
            current_max_runtime=0.0,
            tier_timeout=180.0,
            base_workers=8,
            active_workers=4,
            throughput=0.0,
            avg_runtime=0.0,
            completion_count=0,
        )
        assert not decision.should_throttle
        assert decision.recommended_workers == 8
        assert decision.reason == "No data"

    def test_few_completions_returns_normal(self):
        """With fewer than 3 completions, should return normal."""
        decision = compute_throttle_decision(
            current_max_runtime=10.0,
            tier_timeout=180.0,
            base_workers=8,
            active_workers=4,
            throughput=1.0,
            avg_runtime=5.0,
            completion_count=2,
        )
        assert not decision.should_throttle
        assert decision.recommended_workers == 8
        assert decision.reason == "No data"

    def test_emergency_near_timeout(self):
        """Runtime >= 80% of timeout triggers emergency throttling."""
        decision = compute_throttle_decision(
            current_max_runtime=150.0,  # ~83% of 180
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

    def test_throttle_based_on_avg_runtime(self):
        """Throttling based on formula: workers = timeout / avg_runtime."""
        # timeout=180s, avg=20s → max 9 workers
        decision = compute_throttle_decision(
            current_max_runtime=10.0,
            tier_timeout=180.0,
            base_workers=32,
            active_workers=20,
            throughput=1.0,
            avg_runtime=20.0,  # 180/20 = 9 workers max
            completion_count=10,
        )
        assert decision.should_throttle
        assert decision.recommended_workers == 9
        assert "Throttle" in decision.reason

    def test_low_avg_allows_full_workers(self):
        """Low avg runtime allows full workers."""
        # timeout=180s, avg=5s → max 36 workers, but base is 8
        decision = compute_throttle_decision(
            current_max_runtime=5.0,
            tier_timeout=180.0,
            base_workers=8,
            active_workers=8,
            throughput=1.6,
            avg_runtime=5.0,  # 180/5 = 36 workers max, capped at base
            completion_count=10,
        )
        assert not decision.should_throttle
        assert decision.recommended_workers == 8
        assert decision.reason == "Normal"

    def test_high_avg_limits_workers(self):
        """High avg runtime limits workers significantly."""
        # timeout=180s, avg=60s → max 3 workers
        decision = compute_throttle_decision(
            current_max_runtime=30.0,
            tier_timeout=180.0,
            base_workers=32,
            active_workers=10,
            throughput=0.1,
            avg_runtime=60.0,  # 180/60 = 3 workers max
            completion_count=10,
        )
        assert decision.should_throttle
        assert decision.recommended_workers == 3

    def test_very_high_avg_limits_to_one(self):
        """Very high avg runtime (> timeout) limits to 1 worker."""
        # timeout=180s, avg=200s → max 0.9 → clamped to 1
        decision = compute_throttle_decision(
            current_max_runtime=100.0,
            tier_timeout=180.0,
            base_workers=8,
            active_workers=2,
            throughput=0.01,
            avg_runtime=200.0,  # 180/200 = 0.9 → 1 worker
            completion_count=10,
        )
        assert decision.should_throttle
        assert decision.recommended_workers == 1

    def test_minimum_one_worker(self):
        """Should always recommend at least 1 worker."""
        decision = compute_throttle_decision(
            current_max_runtime=170.0,  # Very near timeout
            tier_timeout=180.0,
            base_workers=1,
            active_workers=1,
            throughput=0.0,
            avg_runtime=500.0,  # Would give 0.36 workers
            completion_count=10,
        )
        assert decision.recommended_workers >= 1

    def test_decision_includes_values(self):
        """Decision should include current max and avg values."""
        decision = compute_throttle_decision(
            current_max_runtime=45.2,
            tier_timeout=180.0,
            base_workers=8,
            active_workers=4,
            throughput=1.5,
            avg_runtime=3.5,
            completion_count=10,
        )
        assert decision.current_max == 45.2
        assert decision.completed_avg == 3.5


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
