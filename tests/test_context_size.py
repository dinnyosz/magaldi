"""Tests for context size computation utilities."""

from shared.ai.context_size import (
    CONTEXT_TIERS,
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
