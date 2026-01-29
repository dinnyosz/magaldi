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
        """Runtime >= 70% of timeout triggers emergency throttling."""
        decision = compute_throttle_decision(
            current_max_runtime=130.0,  # ~72% of 180
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

    def test_critical_above_50_percent(self):
        """Runtime >= 50% of timeout triggers critical throttling."""
        decision = compute_throttle_decision(
            current_max_runtime=100.0,  # ~56% of 180
            tier_timeout=180.0,
            base_workers=8,
            active_workers=4,
            throughput=0.5,
            avg_runtime=5.0,
            completion_count=10,
        )
        assert decision.should_throttle
        assert decision.recommended_workers <= 2
        assert "Critical" in decision.reason

    def test_saturation_detection(self):
        """Low actual vs expected throughput should trigger saturation throttling."""
        # 10 workers with 5s avg runtime = expected throughput of 2/sec
        # But actual throughput is 0.5/sec = 25% efficiency = saturated
        decision = compute_throttle_decision(
            current_max_runtime=10.0,  # Not near timeout
            tier_timeout=180.0,
            base_workers=8,
            active_workers=10,
            throughput=0.5,  # Low actual throughput
            avg_runtime=5.0,  # Expected = 10/5 = 2/sec
            completion_count=10,
        )
        assert decision.should_throttle
        assert decision.recommended_workers < 10
        assert "Saturated" in decision.reason

    def test_normal_throughput(self):
        """Good throughput (>70% efficiency) should not throttle."""
        # 8 workers with 4s avg runtime = expected throughput of 2/sec
        # Actual throughput is 1.6/sec = 80% efficiency = OK
        decision = compute_throttle_decision(
            current_max_runtime=5.0,  # Not near timeout
            tier_timeout=180.0,
            base_workers=8,
            active_workers=8,
            throughput=1.6,  # Good throughput
            avg_runtime=4.0,  # Expected = 8/4 = 2/sec
            completion_count=10,
        )
        assert not decision.should_throttle
        assert decision.recommended_workers == 8
        assert decision.reason == "Normal"

    def test_optimal_workers_calculation(self):
        """Optimal workers should be based on actual throughput."""
        # Actual throughput of 1/sec with 10s avg = optimal ~1.2 workers
        # (throughput * avg_runtime * 1.2 = 1 * 10 * 1.2 = 12)
        # But should be capped at base_workers
        decision = compute_throttle_decision(
            current_max_runtime=5.0,
            tier_timeout=180.0,
            base_workers=8,
            active_workers=20,
            throughput=1.0,
            avg_runtime=10.0,
            completion_count=10,
        )
        assert decision.should_throttle
        # optimal = 1.0 * 10.0 * 1.2 = 12, but capped at base_workers=8
        assert decision.recommended_workers == 8

    def test_minimum_one_worker(self):
        """Should always recommend at least 1 worker."""
        decision = compute_throttle_decision(
            current_max_runtime=170.0,  # Very near timeout
            tier_timeout=180.0,
            base_workers=1,  # Small base
            active_workers=1,
            throughput=0.0,
            avg_runtime=0.0,
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
