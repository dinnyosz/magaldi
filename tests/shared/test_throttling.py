"""Tests for runtime-aware throttling module."""

import time

import pytest

from shared.throttling import (
    RuntimeHistory,
    ThrottleDecision,
    compute_throttle_decision,
)


class TestRuntimeHistory:
    """Tests for RuntimeHistory class."""

    def test_empty_history_returns_zero(self):
        """Empty history should return 0.0."""
        history = RuntimeHistory()
        assert history.get_historical_max() == 0.0

    def test_single_record(self):
        """Single record should be returned as max."""
        history = RuntimeHistory()
        history.record_runtime(5.0)
        assert history.get_historical_max() == 5.0

    def test_multiple_records_same_window(self):
        """Multiple records in same window should track max."""
        history = RuntimeHistory(window_seconds=10.0)
        history.record_runtime(3.0)
        history.record_runtime(7.0)
        history.record_runtime(5.0)
        assert history.get_historical_max() == 7.0

    def test_window_rotation(self):
        """Records after window_seconds should rotate to new window."""
        history = RuntimeHistory(window_seconds=0.1, history_windows=3)

        # First window
        history.record_runtime(5.0)
        assert history.get_historical_max() == 5.0

        # Wait for window to expire
        time.sleep(0.15)

        # Second window with higher value
        history.record_runtime(10.0)
        assert history.get_historical_max() == 10.0

        # Wait again
        time.sleep(0.15)

        # Third window with lower value
        history.record_runtime(3.0)
        # Should still remember the 10.0 from previous window
        assert history.get_historical_max() == 10.0

    def test_reset_clears_history(self):
        """Reset should clear all data."""
        history = RuntimeHistory()
        history.record_runtime(100.0)
        assert history.get_historical_max() == 100.0

        history.reset()
        assert history.get_historical_max() == 0.0


class TestComputeThrottleDecision:
    """Tests for compute_throttle_decision function."""

    def test_no_data_returns_normal(self):
        """With no runtime data, should return normal operation."""
        decision = compute_throttle_decision(
            current_max_runtime=0.0,
            historical_max_runtime=0.0,
            tier_timeout=180.0,
            base_workers=8,
        )
        assert not decision.should_throttle
        assert decision.recommended_workers == 8
        assert decision.reason == "No data"

    def test_low_runtime_returns_normal(self):
        """Runtime < 30% of timeout should be normal."""
        decision = compute_throttle_decision(
            current_max_runtime=50.0,  # ~28% of 180
            historical_max_runtime=45.0,
            tier_timeout=180.0,
            base_workers=8,
        )
        assert not decision.should_throttle
        assert decision.recommended_workers == 8
        assert "Normal" in decision.reason

    def test_caution_threshold_30_percent(self):
        """Runtime 30-50% of timeout should trigger caution."""
        decision = compute_throttle_decision(
            current_max_runtime=60.0,  # ~33% of 180
            historical_max_runtime=55.0,
            tier_timeout=180.0,
            base_workers=8,
        )
        assert decision.should_throttle
        assert decision.recommended_workers == 6  # 75% of 8
        assert "Caution" in decision.reason

    def test_warning_threshold_50_percent(self):
        """Runtime 50-80% of timeout should trigger warning."""
        decision = compute_throttle_decision(
            current_max_runtime=100.0,  # ~56% of 180
            historical_max_runtime=90.0,
            tier_timeout=180.0,
            base_workers=8,
        )
        assert decision.should_throttle
        assert decision.recommended_workers == 4  # 50% of 8
        assert "Warning" in decision.reason

    def test_critical_threshold_80_percent(self):
        """Runtime >= 80% of timeout should trigger critical."""
        decision = compute_throttle_decision(
            current_max_runtime=150.0,  # ~83% of 180
            historical_max_runtime=140.0,
            tier_timeout=180.0,
            base_workers=8,
        )
        assert decision.should_throttle
        assert decision.recommended_workers == 2  # 25% of 8
        assert "Critical" in decision.reason

    def test_uses_higher_of_current_or_historical(self):
        """Should use the higher of current or historical max."""
        # Current is higher
        decision1 = compute_throttle_decision(
            current_max_runtime=150.0,
            historical_max_runtime=50.0,
            tier_timeout=180.0,
            base_workers=8,
        )
        assert decision1.should_throttle
        assert "Critical" in decision1.reason

        # Historical is higher
        decision2 = compute_throttle_decision(
            current_max_runtime=50.0,
            historical_max_runtime=150.0,
            tier_timeout=180.0,
            base_workers=8,
        )
        assert decision2.should_throttle
        assert "Critical" in decision2.reason

    def test_minimum_one_worker(self):
        """Should always recommend at least 1 worker."""
        decision = compute_throttle_decision(
            current_max_runtime=170.0,
            historical_max_runtime=175.0,
            tier_timeout=180.0,
            base_workers=2,  # Small base
        )
        assert decision.should_throttle
        assert decision.recommended_workers >= 1

    def test_decision_includes_runtime_values(self):
        """Decision should include both runtime values for display."""
        decision = compute_throttle_decision(
            current_max_runtime=45.2,
            historical_max_runtime=52.1,
            tier_timeout=180.0,
            base_workers=8,
        )
        assert decision.current_max == 45.2
        assert decision.historical_max == 52.1


class TestThrottleDecisionDataclass:
    """Tests for ThrottleDecision dataclass."""

    def test_dataclass_fields(self):
        """ThrottleDecision should have all expected fields."""
        decision = ThrottleDecision(
            should_throttle=True,
            current_max=45.0,
            historical_max=50.0,
            recommended_workers=4,
            reason="Test reason",
        )
        assert decision.should_throttle is True
        assert decision.current_max == 45.0
        assert decision.historical_max == 50.0
        assert decision.recommended_workers == 4
        assert decision.reason == "Test reason"
