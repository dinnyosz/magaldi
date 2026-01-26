"""Tests for context size computation utilities."""

from shared.ai.context_size import (
    CONTEXT_TIERS,
    compute_aggregation_num_ctx,
    compute_context_sizes,
    compute_element_num_ctx,
    compute_num_ctx,
)


class TestComputeNumCtx:
    """Tests for compute_num_ctx function."""

    def test_small_variable_returns_smallest_tier(self):
        """Variable with 100 chars should use 2048 context."""
        result = compute_num_ctx("variable", 100)
        assert result == 2048

    def test_medium_function_returns_4096(self):
        """Function with 8000 chars should use 4096 context.

        Calculation: 8000 chars / 4 = 2000 tokens + 700 overhead = 2700 total.
        2700 > 2048, so it needs 4096 tier.
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

    def test_tiny_function_uses_smallest_tier(self):
        """A 200 char function should use 2048 context.

        Calculation: 200 chars / 4 = 50 tokens + 700 overhead = 750 total.
        750 < 2048, so it uses smallest tier.
        """
        result = compute_element_num_ctx("function", 200)
        assert result == 2048

    def test_small_function_uses_smallest_tier(self):
        """A 2000 char function should use 2048 context.

        Calculation: 2000 chars / 4 = 500 tokens + 700 overhead = 1200 total.
        1200 < 2048, so it uses smallest tier.
        """
        result = compute_element_num_ctx("function", 2000)
        assert result == 2048

    def test_medium_function_uses_4096(self):
        """A 10000 char function should use 4096 context.

        Calculation: 10000 chars / 4 = 2500 tokens + 700 overhead = 3200 total.
        3200 > 2048 but < 4096, so it uses 4096 tier.
        """
        result = compute_element_num_ctx("function", 10000)
        assert result == 4096

    def test_large_file_uses_32768(self):
        """A 72000 char file should use 32768 context.

        Calculation: 72000 chars / 4 = 18000 tokens + 300 overhead = 18300 total.
        18300 > 16384 but < 32768, so it uses 32768 tier.
        """
        result = compute_element_num_ctx("file", 72000)
        assert result == 32768

    def test_different_elements_same_size_different_tiers(self):
        """Different element types with same char count may get different tiers.

        A 5000 char element:
        - function: 5000/4 = 1250 + 700 = 1950 → 2048
        - file: 5000/4 = 1250 + 300 = 1550 → 2048

        Both fit in 2048, but larger code will show the difference.
        """
        # 6000 chars:
        # - function: 6000/4 = 1500 + 700 = 2200 → 4096
        # - file: 6000/4 = 1500 + 300 = 1800 → 2048
        func_result = compute_element_num_ctx("function", 6000)
        file_result = compute_element_num_ctx("file", 6000)
        assert func_result == 4096
        assert file_result == 2048

    def test_empty_code_uses_smallest_tier(self):
        """Empty code should still account for overhead and use smallest tier."""
        result = compute_element_num_ctx("function", 0)
        assert result == 2048

    def test_huge_element_uses_largest_tier(self):
        """Very large elements should use the largest available tier."""
        # 500000 chars = 125000 tokens + overhead > 32768
        result = compute_element_num_ctx("file", 500000)
        assert result == CONTEXT_TIERS[-1]


class TestComputeAggregationNumCtx:
    """Tests for aggregation context size computation (features, glossary)."""

    def test_small_prompt_uses_smallest_tier(self):
        """Small prompts should use 2048 tier.

        1000 chars / 3.5 = 286 tokens + 200 overhead = 486 total.
        With 2x multiplier = 972 < 2048 → 2048 tier.
        """
        result = compute_aggregation_num_ctx(1000, task_type="feature")
        assert result == 2048

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
