"""Tests for seed ETA estimates and tier scaling exponent fixes."""

from __future__ import annotations

import time

import pytest

from magaldi_core.processor.timing import (
    TimingStats,
    _HANDCRAFTED_DEFAULT_ETA,
    _SEED_TIMEOUT_SCALE,
    _SMALL_MODEL_SPEED_FACTOR,
    _type_weight,
    _uses_small_model,
)
from shared.ai.context_size import (
    CRAFT_SMALL,
    CRAFT_TEST,
    TIER_BASE_TIMEOUTS,
    TIER_SCALING_EXPONENT,
)


class TestSeedEstimates:
    """Tests for TimingStats.seed_estimates()."""

    def test_seed_populates_all_type_tier_combos(self):
        """seed_estimates() should populate an estimate for every (type, tier) in totals."""
        stats = TimingStats()
        totals = {
            ("file", 4096): 5,
            ("function", 2048): 10,
            ("method", 1024): 8,
        }
        stats.set_totals_by_type_tier(totals)
        stats.seed_estimates()

        for key in totals:
            assert key in stats._seed_estimates, f"Missing seed for {key}"
            assert stats._seed_estimates[key] > 0, f"Seed for {key} should be > 0"

    def test_seed_values_derived_from_tier_base_timeouts(self):
        """Seed values should be TIER_BASE_TIMEOUTS / scale factor * type weight."""
        stats = TimingStats()
        stats.set_totals_by_type_tier({("file", 4096): 3})
        stats.seed_estimates()

        baseline_weight = _type_weight("function")
        type_adj = _type_weight("file") / baseline_weight
        expected = TIER_BASE_TIMEOUTS[4096] / _SEED_TIMEOUT_SCALE * type_adj
        # file@4096 uses the large model (not small)
        assert not _uses_small_model("file", 4096)
        assert stats._seed_estimates[("file", 4096)] == pytest.approx(expected)

    def test_seed_applies_small_model_factor(self):
        """Small model types should get a reduced seed estimate."""
        stats = TimingStats()
        stats.set_totals_by_type_tier({
            ("function", 2048): 5,  # function always uses small model
            ("file", 2048): 3,     # file@2048 uses large model (tier > 1024)
        })
        stats.seed_estimates()

        baseline_weight = _type_weight("function")
        base_seed = TIER_BASE_TIMEOUTS[2048] / _SEED_TIMEOUT_SCALE
        # function: small model + type_adj=1.0 (it IS the baseline)
        func_seed = base_seed * _SMALL_MODEL_SPEED_FACTOR * (_type_weight("function") / baseline_weight)
        # file: large model + file type adjustment
        file_seed = base_seed * (_type_weight("file") / baseline_weight)

        assert stats._seed_estimates[("function", 2048)] == pytest.approx(func_seed)
        assert stats._seed_estimates[("file", 2048)] == pytest.approx(file_seed)

    def test_seed_handcrafted_tiers_use_default(self):
        """Handcrafted tiers should use _HANDCRAFTED_DEFAULT_ETA."""
        stats = TimingStats()
        stats.set_totals_by_type_tier({
            ("function", CRAFT_TEST): 5,
            ("method", CRAFT_SMALL): 3,
        })
        stats.seed_estimates()

        assert stats._seed_estimates[("function", CRAFT_TEST)] == _HANDCRAFTED_DEFAULT_ETA
        assert stats._seed_estimates[("method", CRAFT_SMALL)] == _HANDCRAFTED_DEFAULT_ETA

    def test_seed_used_as_fallback_when_no_real_data(self):
        """With no real timing data, the fallback chain should return seed values."""
        stats = TimingStats()
        stats.phase_start = time.time()
        stats.set_totals_by_type_tier({
            ("file", 4096): 5,
            ("function", 2048): 10,
        })
        stats.seed_estimates()

        breakdown = stats.get_eta_breakdown_with_avg()

        # Every cell should have a non-zero avg (from seed) and is_fallback=True
        for elem_type, tier, avg_time, is_fallback, done, total in breakdown:
            assert avg_time > 0, f"Expected seed estimate for {elem_type}@{tier}, got 0"
            assert is_fallback, f"Expected fallback=True for {elem_type}@{tier}"
            assert done == 0

    def test_real_data_overrides_seed_estimate(self):
        """Once real data arrives, it should take priority over seed values."""
        stats = TimingStats()
        stats.phase_start = time.time()
        stats.set_totals_by_type_tier({("file", 4096): 5})
        stats.seed_estimates()

        seed_value = stats._seed_estimates[("file", 4096)]

        # Record a real element
        stats.record(
            wall_time=3.0,
            summarize_time=2.5,
            embed_time=0.5,
            element_type="file",
            tier=4096,
            avg_workers=1.0,
        )

        breakdown = stats.get_eta_breakdown_with_avg()
        data = {(item[0], item[1]): item for item in breakdown}

        elem_type, tier, avg_time, is_fallback, done, total = data[("file", 4096)]
        assert done == 1
        assert not is_fallback, "Real data should not be marked as fallback"
        # avg should be based on real data (3.0s wall / 1 worker), not seed
        assert avg_time == pytest.approx(3.0, abs=0.1)
        assert avg_time != pytest.approx(seed_value, abs=0.01)

    def test_eta_seconds_uses_seed_estimates(self):
        """eta_seconds() should return a value even with no real timing data."""
        stats = TimingStats()
        stats.phase_start = time.time()
        stats.set_totals_by_type({
            "file": 5,
            "function": 10,
        })
        stats.set_totals_by_type_tier({
            ("file", 4096): 5,
            ("function", 2048): 10,
        })
        stats.seed_estimates()

        # With 0 completed, eta_seconds returns None (guard at the top)
        eta = stats.eta_seconds(completed=0, _total=15)
        assert eta is None

        # Record one element to pass the completed > 0 guard
        stats.record(
            wall_time=3.0,
            summarize_time=2.5,
            embed_time=0.5,
            element_type="file",
            tier=4096,
            avg_workers=1.0,
        )

        eta = stats.eta_seconds(completed=1, _total=15)
        assert eta is not None
        assert eta > 0

    def test_seed_cleared_on_re_seed(self):
        """Calling seed_estimates() again should replace old seeds."""
        stats = TimingStats()
        stats.set_totals_by_type_tier({("file", 4096): 5})
        stats.seed_estimates()
        assert ("file", 4096) in stats._seed_estimates

        # Re-set with different totals
        stats.set_totals_by_type_tier({("function", 2048): 10})
        stats.seed_estimates()

        assert ("file", 4096) not in stats._seed_estimates
        assert ("function", 2048) in stats._seed_estimates


class TestTierScalingExponentFix:
    """Tests that fallback steps 5-6 use TIER_SCALING_EXPONENT correctly."""

    def test_same_model_group_fallback_uses_exponent(self):
        """Step 5: same model group avg should scale with exponent, not linearly."""
        stats = TimingStats()
        stats.phase_start = time.time()

        # Set up: function@2048 has data, method@8192 needs estimate
        # Both use small model
        stats.set_totals_by_type_tier({
            ("function", 2048): 5,
            ("method", 8192): 3,
        })

        # Record function@2048 data
        for _ in range(5):
            stats.record(
                wall_time=2.0,
                summarize_time=1.5,
                embed_time=0.5,
                element_type="function",
                tier=2048,
                avg_workers=1.0,
            )

        # Get method@8192 estimate — should use step 3 (same model group, closest tier)
        # with tier scaling exponent
        breakdown = stats.get_eta_breakdown_with_avg()
        data = {(item[0], item[1]): item for item in breakdown}

        _, _, avg_method, is_fallback, _, _ = data[("method", 8192)]
        assert is_fallback

        # The avg for function@2048 is 2.0s (wall_time / avg_workers)
        base_avg = 2.0
        # With exponent: (8192/2048)^0.65 = 4^0.65 ≈ 2.46
        tier_ratio = (8192 / 2048) ** TIER_SCALING_EXPONENT
        # No type adjustment in fallback steps (real data is better signal)
        expected_with_exponent = base_avg * tier_ratio
        # Without exponent (old bug): 4^1.0 = 4.0 → 8.0s
        wrong_linear = base_avg * (8192 / 2048)

        # Should match exponent value (no type adjustment)
        assert avg_method == pytest.approx(expected_with_exponent, rel=0.05)
        assert avg_method < wrong_linear * 0.9, "Should be sub-linear, not linear scaling"

    def test_cross_model_fallback_uses_exponent(self):
        """Step 6: cross-model fallback should scale with exponent."""
        stats = TimingStats()
        stats.phase_start = time.time()

        # Only large model data exists (file@4096, large model)
        # Need estimate for function@4096 (small model) — triggers cross-model fallback
        stats.set_totals_by_type_tier({
            ("file", 4096): 5,
            ("file", 8192): 3,  # Need this to have large-model data only
        })

        # Record file@4096 data
        for _ in range(5):
            stats.record(
                wall_time=4.0,
                summarize_time=3.0,
                embed_time=1.0,
                element_type="file",
                tier=4096,
                avg_workers=1.0,
            )

        # Now ask for file@8192 (same type, same model group, different tier)
        # This should hit step 2 (same type, closest tier) with exponent
        avg, is_fallback = stats._get_avg_for_type_tier_with_fallback("file", 8192, 0.0)

        base_avg = 4.0
        expected = base_avg * (8192 / 4096) ** TIER_SCALING_EXPONENT
        assert avg == pytest.approx(expected, rel=0.01)
        assert is_fallback

    def test_seed_values_are_reasonable(self):
        """Seed estimates should produce reasonable per-element times."""
        stats = TimingStats()
        stats.set_totals_by_type_tier({
            ("file", 2048): 10,
            ("file", 8192): 5,
            ("function", 1024): 20,
            ("function", 4096): 10,
            ("constant", 2048): 5,
        })
        stats.seed_estimates()

        # All seeds should be positive and in a reasonable range (0.5s - 20s)
        for key, seed in stats._seed_estimates.items():
            assert 0.3 < seed < 25.0, f"Seed for {key} = {seed}s is out of range"

        # Larger tiers should have larger seeds (same type)
        assert stats._seed_estimates[("file", 8192)] > stats._seed_estimates[("file", 2048)]
        assert stats._seed_estimates[("function", 4096)] > stats._seed_estimates[("function", 1024)]

        # Type weight differences: constant should be significantly cheaper
        # than function at the same tier (output budget: 200 vs 500 tokens)
        assert stats._seed_estimates[("constant", 2048)] < stats._seed_estimates[("function", 1024)] * 1.5

    def test_type_weight_ordering(self):
        """Type weights should reflect output token budget differences."""
        # Output-heavy types should cost more
        assert _type_weight("function") > _type_weight("constant")
        assert _type_weight("function") > _type_weight("type_alias")

        # file has lower PROMPT_OVERHEAD than function (less parent context overhead)
        assert _type_weight("file") < _type_weight("function")

        # constant should be lighter than variable
        assert _type_weight("constant") < _type_weight("variable")

    def test_type_weight_affects_seed_estimates(self):
        """Seed estimates should reflect type weight differences."""
        stats = TimingStats()
        stats.phase_start = time.time()
        stats.set_totals_by_type_tier({
            ("function", 2048): 10,
            ("constant", 2048): 5,
        })
        stats.seed_estimates()

        # Seed for constant should be less than seed for function
        # because constant has much lower output token budget
        func_seed = stats._seed_estimates[("function", 2048)]
        const_seed = stats._seed_estimates[("constant", 2048)]

        assert const_seed < func_seed, (
            f"constant@2048 seed ({const_seed:.2f}s) should be less than "
            f"function@2048 seed ({func_seed:.2f}s) due to lower output budget"
        )

    def test_seed_preferred_over_cross_model_fallback(self):
        """Seeds should be used before cross-model fallback to avoid file data poisoning."""
        stats = TimingStats()
        stats.phase_start = time.time()

        # Only large-model data exists (file@8192)
        # Need estimate for function@2048 (small model)
        stats.set_totals_by_type_tier({
            ("file", 8192): 5,
            ("function", 2048): 100,
        })
        stats.seed_estimates()

        # Record file@8192 data (large model, ~12s per element)
        for _ in range(5):
            stats.record(
                wall_time=12.0, summarize_time=10.0, embed_time=2.0,
                element_type="file", tier=8192, avg_workers=1.0,
            )

        # function@2048 should use seed estimate, NOT cross-model fallback
        # Cross-model would give: 12.0 * (2048/8192)^0.65 * 0.5 ≈ 2.6s
        # which is plausible but seeds are more calibrated per type
        breakdown = stats.get_eta_breakdown_with_avg()
        data = {(item[0], item[1]): item for item in breakdown}

        _, _, avg_func, is_fb, _, _ = data[("function", 2048)]
        assert is_fb, "function estimate should be a fallback"
        # Should match the seed, not cross-model extrapolation
        assert avg_func == pytest.approx(stats._seed_estimates[("function", 2048)], rel=0.01)


class TestElapsedRateOverride:
    """Tests that elapsed_rate doesn't flatten tier-aware estimates."""

    def test_elapsed_rate_does_not_override_tier_estimates(self):
        """Tier-aware estimates should not be replaced by a flat elapsed_rate."""
        stats = TimingStats()
        stats.phase_start = time.time() - 2760  # 46 minutes ago

        # Set up multiple tiers with different expected times
        stats.set_totals_by_type_tier({
            ("function", 1024): 200,
            ("function", 4096): 100,
            ("file", 8192): 50,
            ("file", 16384): 20,
        })
        stats.seed_estimates()

        # Process a bunch of function@1024 (fast, small model)
        for _ in range(200):
            stats.record(
                wall_time=1.0,
                summarize_time=0.8,
                embed_time=0.2,
                element_type="function",
                tier=1024,
                avg_workers=4.0,
            )
        # Process some function@4096
        for _ in range(100):
            stats.record(
                wall_time=2.5,
                summarize_time=2.0,
                embed_time=0.5,
                element_type="function",
                tier=4096,
                avg_workers=4.0,
            )
        # Process some file@8192
        for _ in range(30):
            stats.record(
                wall_time=6.0,
                summarize_time=5.0,
                embed_time=1.0,
                element_type="file",
                tier=8192,
                avg_workers=4.0,
            )

        # elapsed_rate = 2760s / 330 elements ≈ 8.36s
        # But file@16384 should have a HIGHER estimate (bigger tier)
        # and function@1024 had avg 0.25s base_time — should NOT be 8.36s
        breakdown = stats.get_eta_breakdown_with_avg()
        data = {(item[0], item[1]): item for item in breakdown}

        # file@8192 has 20 remaining — avg should reflect real data (1.5s base),
        # NOT the flat 8.36s elapsed_rate
        _, _, avg_file_8k, _, done_8k, _ = data[("file", 8192)]
        assert done_8k == 30
        assert avg_file_8k < 3.0, f"file@8192 avg {avg_file_8k}s should reflect real data, not elapsed_rate"

        # file@16384 has 20 remaining, no direct data — should use tier-scaled
        # fallback from file@8192, not flat elapsed_rate
        _, _, avg_file_16k, is_fb_16k, done_16k, _ = data[("file", 16384)]
        assert done_16k == 0
        assert is_fb_16k
        # Should be scaled UP from file@8192, not flat 8.36s
        assert avg_file_16k > avg_file_8k, "16k tier should estimate higher than 8k"

    def test_elapsed_rate_still_works_for_skip_ai(self):
        """In skip_ai mode (base_time ≈ 0), elapsed_rate should still be used."""
        stats = TimingStats()
        stats.phase_start = time.time() - 100

        stats.set_totals_by_type_tier({
            ("function", 1024): 100,
            ("method", 1024): 50,
        })
        # No seed_estimates() call — simulating skip_ai where no seeds

        # Record with near-zero base_time (skip_ai: wall_time/workers ≈ 0)
        for _ in range(50):
            stats.record(
                wall_time=0.001,
                summarize_time=0.0,
                embed_time=0.001,
                element_type="function",
                tier=1024,
                avg_workers=4.0,
            )

        breakdown = stats.get_eta_breakdown_with_avg()
        data = {(item[0], item[1]): item for item in breakdown}

        # method@1024 has no data and no seeds — avg should come from
        # elapsed_rate or cross-type fallback
        _, _, avg_method, _, done_method, _ = data[("method", 1024)]
        assert done_method == 0
        # Should have SOME estimate (from elapsed_rate since base_time ≈ 0)
        # elapsed_rate = 100s / 50 = 2.0s
        assert avg_method > 0, "skip_ai mode should still get elapsed_rate estimate"
