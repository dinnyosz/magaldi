"""Tests for context size computation utilities."""

from shared.ai.context_size import (
    CONTEXT_TIERS,
    compute_context_sizes,
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
