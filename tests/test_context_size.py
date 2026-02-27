"""Tests for context size computation utilities."""

import math

import pytest

from shared.ai.context_size import (
    CONTEXT_TIERS,
    DEFAULT_BASE_TIMEOUT,
    HANDCRAFTED_TIER,
    TIER_BASE_TIMEOUTS,
    compute_aggregation_num_ctx,
    compute_context_sizes,
    compute_element_num_ctx,
    compute_num_ctx,
    get_tier_timeout,
)


class TestComputeNumCtx:
    """Tests for compute_num_ctx function."""

    def test_small_variable_returns_1024_tier(self):
        """Variable with 100 chars should use 1024 context.

        Calculation: 100/4 = 25 + 550 overhead = 575 < 1024.
        """
        result = compute_num_ctx("variable", 100)
        assert result == 1024

    def test_medium_function_returns_4096(self):
        """Function with 8000 chars should use 4096 context.

        Calculation: 8000/4 = 2000 + 800 overhead = 2800 total.
        2800 > 2048, so it needs 4096 tier.
        """
        result = compute_num_ctx("function", 8000)
        assert result == 4096

    def test_large_class_returns_8192(self):
        """Class with 20000 chars should use 8192 context."""
        result = compute_num_ctx("class", 20000)
        assert result == 8192

    def test_xlarge_file_returns_16384(self):
        """File with 50000 chars should use 16384 context."""
        result = compute_num_ctx("file", 50000)
        assert result == 16384

    def test_huge_file_returns_largest_tier(self):
        """File with 200000 chars should use largest tier."""
        result = compute_num_ctx("file", 200000)
        assert result == CONTEXT_TIERS[-1]

    def test_unknown_type_uses_default_overhead(self):
        """Unknown element type should use default overhead."""
        result = compute_num_ctx("unknown", 1000)
        assert result in CONTEXT_TIERS


class TestComputeContextSizes:
    """Tests for compute_context_sizes function."""

    def test_computes_sizes_for_all_types(self):
        """Should compute context size for each element type in max_chars."""
        max_chars = {
            "file": 40000,
            "class": 15000,
            "function": 3000,
            "method": 2000,
            "variable": 200,
            "constant": 100,
        }
        result = compute_context_sizes(max_chars)

        assert "file" in result
        assert "class" in result
        assert "function" in result
        assert "method" in result
        assert "variable" in result
        assert "constant" in result
        # All values should be valid tiers
        for tier in result.values():
            assert tier in CONTEXT_TIERS

    def test_empty_max_chars_returns_empty(self):
        """Empty input should return empty dict."""
        result = compute_context_sizes({})
        assert result == {}


class TestComputeElementNumCtx:
    """Tests for per-element context size computation."""

    def test_tiny_function_uses_2048_tier(self):
        """A 200 char function should use 2048 context.

        Calculation: 200/4 = 50 + 1100 overhead = 1150 < 2048.
        """
        result = compute_element_num_ctx("function", 200)
        assert result == 2048

    def test_small_function_uses_4096_tier(self):
        """A 4000 char function should use 4096 context.

        Calculation: 4000/4 = 1000 + 1100 overhead = 2100 < 4096.
        """
        result = compute_element_num_ctx("function", 4000)
        assert result == 4096

    def test_medium_function_uses_4096(self):
        """A 10000 char function should use 4096 context.

        Calculation: 10000/4 = 2500 + 1100 overhead = 3600 total.
        3600 > 2048 but < 4096, so it uses 4096 tier.
        """
        result = compute_element_num_ctx("function", 10000)
        assert result == 4096

    def test_large_file_uses_32768(self):
        """A 72000 char file should use 32768 context.

        Calculation: 72000/4 = 18000 + 950 overhead = 18950 total.
        18950 > 16384 but < 32768, so it uses 32768 tier.
        """
        result = compute_element_num_ctx("file", 72000)
        assert result == 32768

    def test_different_elements_same_size_different_tiers(self):
        """Different element types with same char count may get different tiers.

        5000 chars:
        - function: 5000/4 = 1250 + 1100 = 2350 → 4096
        - import: 5000/4 = 1250 + 400 = 1650 → 2048
        """
        func_result = compute_element_num_ctx("function", 5000)
        import_result = compute_element_num_ctx("import", 5000)
        assert func_result == 4096
        assert import_result == 2048

    def test_empty_code_uses_1024_tier(self):
        """Empty code with just overhead should use 1024 tier."""
        # import: 0 + 400 = 400 → 1024
        result = compute_element_num_ctx("import", 0)
        assert result == 1024

    def test_empty_code_function_uses_2048_tier(self):
        """Empty function code with overhead should use 2048 tier."""
        # function: 0 + 1100 = 1100 → 2048
        result = compute_element_num_ctx("function", 0)
        assert result == 2048

    def test_huge_element_uses_largest_tier(self):
        """Very large elements should use the largest available tier."""
        # 500000 chars = 125000 tokens + overhead > 32768
        result = compute_element_num_ctx("file", 500000)
        assert result == CONTEXT_TIERS[-1]

    def test_1024_tier_used_for_small_elements(self):
        """Small enums/imports should use the 1024 tier."""
        # enum: 200/4 = 50 + 600 = 650 < 1024
        assert compute_element_num_ctx("enum", 200) == 1024
        # import: 200/4 = 50 + 400 = 450 < 1024
        assert compute_element_num_ctx("import", 200) == 1024
        # type_alias: 200/4 = 50 + 550 = 600 < 1024
        assert compute_element_num_ctx("type_alias", 200) == 1024


class TestComputeAggregationNumCtx:
    """Tests for aggregation context size computation (features, glossary)."""

    def test_small_prompt_uses_smallest_tier(self):
        """Small prompts should use 1024 tier.

        1000 chars / 3.5 = 286 tokens + 200 overhead = 486 total.
        With 2x multiplier = 972 < 1024 → 1024 tier.
        """
        result = compute_aggregation_num_ctx(1000, task_type="feature")
        assert result == 1024

    def test_medium_prompt_uses_4096(self):
        """Medium prompts should use 4096 tier.

        4000 chars / 3.5 = 1143 tokens + 200 overhead = 1343 total.
        With 2x multiplier = 2686 > 2048 but < 4096 → 4096 tier.
        """
        result = compute_aggregation_num_ctx(4000, task_type="feature")
        assert result == 4096

    def test_large_prompt_uses_8192(self):
        """Large prompts should use 8192 tier.

        10000 chars / 3.5 = 2857 tokens + 200 overhead = 3057 total.
        With 2x multiplier = 6114 > 4096 but < 8192 → 8192 tier.
        """
        result = compute_aggregation_num_ctx(10000, task_type="feature")
        assert result == 8192

    def test_different_task_types_have_different_overhead(self):
        """Different task types should have different overhead values."""
        # Same prompt chars, different overhead
        labeling = compute_aggregation_num_ctx(500, task_type="labeling")
        glossary_summary = compute_aggregation_num_ctx(500, task_type="glossary_summary")

        # glossary_summary has more overhead (350 vs 150), may result in different tier
        assert labeling in CONTEXT_TIERS
        assert glossary_summary in CONTEXT_TIERS

    def test_custom_safety_multiplier(self):
        """Safety multiplier should affect tier selection.

        3000 chars / 3.5 = 857 tokens + 200 overhead = 1057 total.
        With 1.5x multiplier = 1586 < 2048 → 2048 tier.
        With 3.0x multiplier = 3171 > 2048 but < 4096 → 4096 tier.
        """
        result_1_5x = compute_aggregation_num_ctx(3000, safety_multiplier=1.5)
        result_3x = compute_aggregation_num_ctx(3000, safety_multiplier=3.0)

        assert result_1_5x == 2048
        assert result_3x == 4096

    def test_returns_valid_tier(self):
        """Result should always be a valid context tier."""
        for chars in [100, 500, 1000, 5000, 10000, 50000]:
            result = compute_aggregation_num_ctx(chars)
            assert result in CONTEXT_TIERS

    def test_huge_prompt_uses_largest_tier(self):
        """Very large prompts should use the largest available tier."""
        # 100000 chars / 3.5 = 28571 tokens + overhead > 32768
        result = compute_aggregation_num_ctx(100000)
        assert result == CONTEXT_TIERS[-1]


class TestGetTierTimeout:
    """Tests for worker-scaled tier timeouts."""

    def test_1_worker_returns_base_timeout(self):
        """With 1 worker, timeout equals the base timeout."""
        for tier, base in TIER_BASE_TIMEOUTS.items():
            assert get_tier_timeout(tier, 1) == pytest.approx(float(base))

    def test_scales_with_sqrt_of_workers(self):
        """Timeout scales with sqrt(base_workers)."""
        t1 = get_tier_timeout(2048, 1)
        t4 = get_tier_timeout(2048, 4)
        t16 = get_tier_timeout(2048, 16)
        assert t4 == pytest.approx(t1 * 2.0)   # sqrt(4) = 2
        assert t16 == pytest.approx(t1 * 4.0)  # sqrt(16) = 4

    def test_16_workers_at_1024_matches_old_360(self):
        """16 workers at 1024 tier should reproduce the original 360s value.

        Old TIER_TIMEOUTS[1024] was 360. Base is 90, sqrt(16)=4, 90*4=360.
        """
        assert get_tier_timeout(1024, 16) == pytest.approx(360.0)

    def test_unknown_tier_uses_default_base(self):
        """Unknown tier falls back to DEFAULT_BASE_TIMEOUT."""
        t = get_tier_timeout(9999, 4)
        assert t == pytest.approx(DEFAULT_BASE_TIMEOUT * math.sqrt(4))

    def test_handcrafted_tier(self):
        """Handcrafted tier has a very short base timeout."""
        t = get_tier_timeout(HANDCRAFTED_TIER, 1)
        assert t == pytest.approx(15.0)

    def test_32768_single_worker(self):
        """32768 tier with 1 worker stays at 600s (max workers is 1 for this tier)."""
        assert get_tier_timeout(32768, 1) == pytest.approx(600.0)

    def test_zero_workers_treated_as_1(self):
        """0 or negative workers should be clamped to 1."""
        assert get_tier_timeout(2048, 0) == get_tier_timeout(2048, 1)
        assert get_tier_timeout(2048, -5) == get_tier_timeout(2048, 1)

    def test_monotonically_increases_with_workers(self):
        """More workers should always mean a higher (or equal) timeout."""
        for tier in TIER_BASE_TIMEOUTS:
            prev = 0.0
            for workers in [1, 2, 4, 8, 16, 32]:
                t = get_tier_timeout(tier, workers)
                assert t >= prev, f"tier={tier}, workers={workers}: {t} < {prev}"
                prev = t

    def test_larger_tiers_have_higher_base_timeouts(self):
        """Larger context tiers should have higher or equal base timeouts."""
        tiers = sorted(TIER_BASE_TIMEOUTS.keys())
        for i in range(1, len(tiers)):
            assert TIER_BASE_TIMEOUTS[tiers[i]] >= TIER_BASE_TIMEOUTS[tiers[i - 1]]
