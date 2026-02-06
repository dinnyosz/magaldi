"""Tests for Ollama model alias management."""

import pytest

from shared.ai.ollama_models import (
    CONTEXT_TIERS,
    TIER_SUFFIXES,
    get_base_model_name,
    get_tiered_model_name,
    is_tiered_model,
)


class TestGetTieredModelName:
    """Tests for get_tiered_model_name function."""

    def test_returns_2k_for_small_context(self):
        """Context <= 2048 should return -2k suffix."""
        result = get_tiered_model_name("qwen3:4b-instruct", 1000)
        assert result == "qwen3:4b-instruct-2k"

    def test_returns_2k_for_exact_tier(self):
        """Context == 2048 should return -2k suffix."""
        result = get_tiered_model_name("qwen3:4b-instruct", 2048)
        assert result == "qwen3:4b-instruct-2k"

    def test_returns_4k_for_medium_context(self):
        """Context between 2049-4096 should return -4k suffix."""
        result = get_tiered_model_name("qwen3:4b-instruct", 3000)
        assert result == "qwen3:4b-instruct-4k"

    def test_returns_8k_for_large_context(self):
        """Context between 4097-8192 should return -8k suffix."""
        result = get_tiered_model_name("qwen3:4b-instruct", 6000)
        assert result == "qwen3:4b-instruct-8k"

    def test_returns_16k_for_xlarge_context(self):
        """Context between 8193-16384 should return -16k suffix."""
        result = get_tiered_model_name("qwen3:4b-instruct", 12000)
        assert result == "qwen3:4b-instruct-16k"

    def test_returns_32k_for_huge_context(self):
        """Context > 16384 should return -32k suffix."""
        result = get_tiered_model_name("qwen3:4b-instruct", 20000)
        assert result == "qwen3:4b-instruct-32k"

    def test_returns_largest_tier_for_overflow(self):
        """Context > largest tier should return largest tier suffix."""
        result = get_tiered_model_name("qwen3:4b-instruct", 100000)
        assert result == "qwen3:4b-instruct-32k"

    def test_handles_model_with_tag(self):
        """Model names with tags should work correctly."""
        result = get_tiered_model_name("llama3.1:8b-instruct-q4_K_M", 4096)
        assert result == "llama3.1:8b-instruct-q4_K_M-4k"


class TestIsTieredModel:
    """Tests for is_tiered_model function."""

    def test_returns_true_for_tiered_model(self):
        """Tiered model names should be detected."""
        assert is_tiered_model("qwen3:4b-instruct-2k") is True
        assert is_tiered_model("qwen3:4b-instruct-4k") is True
        assert is_tiered_model("qwen3:4b-instruct-8k") is True
        assert is_tiered_model("qwen3:4b-instruct-16k") is True
        assert is_tiered_model("qwen3:4b-instruct-32k") is True

    def test_returns_false_for_base_model(self):
        """Base model names should not be detected as tiered."""
        assert is_tiered_model("qwen3:4b-instruct") is False
        assert is_tiered_model("llama3.1:8b") is False
        assert is_tiered_model("qwen3-embedding:0.6b") is False

    def test_returns_false_for_similar_suffix(self):
        """Similar but non-tier suffixes should not match."""
        assert is_tiered_model("qwen3:4b-instruct-latest") is False
        assert is_tiered_model("qwen3:4b-instruct-v2") is False


class TestGetBaseModelName:
    """Tests for get_base_model_name function."""

    def test_removes_tier_suffix(self):
        """Tier suffix should be removed."""
        assert get_base_model_name("qwen3:4b-instruct-2k") == "qwen3:4b-instruct"
        assert get_base_model_name("qwen3:4b-instruct-4k") == "qwen3:4b-instruct"
        assert get_base_model_name("qwen3:4b-instruct-32k") == "qwen3:4b-instruct"

    def test_preserves_base_model(self):
        """Base model names should be unchanged."""
        assert get_base_model_name("qwen3:4b-instruct") == "qwen3:4b-instruct"
        assert get_base_model_name("llama3.1:8b") == "llama3.1:8b"


class TestContextTiers:
    """Tests for context tier constants."""

    def test_tiers_are_power_of_two(self):
        """All tiers should be powers of 2."""
        for tier in CONTEXT_TIERS:
            # Check if tier is a power of 2
            assert tier > 0 and (tier & (tier - 1)) == 0

    def test_tiers_are_sorted(self):
        """Tiers should be in ascending order."""
        assert CONTEXT_TIERS == sorted(CONTEXT_TIERS)

    def test_suffixes_match_tiers(self):
        """All tiers should have corresponding suffixes."""
        for tier in CONTEXT_TIERS:
            assert tier in TIER_SUFFIXES


class TestResolveOllamaModel:
    """Tests for resolve_ollama_model function."""

    def test_non_ollama_model_unchanged(self):
        """Non-Ollama models should return unchanged."""
        from shared.ai.ollama_models import resolve_ollama_model

        model, ctx = resolve_ollama_model("openai/gpt-4", None, 4096)
        assert model == "openai/gpt-4"
        assert ctx == 4096

    def test_ollama_model_without_ctx_unchanged(self):
        """Ollama models without num_ctx should return unchanged."""
        from shared.ai.ollama_models import resolve_ollama_model

        model, ctx = resolve_ollama_model("ollama/qwen3:4b-instruct", None, None)
        assert model == "ollama/qwen3:4b-instruct"
        assert ctx is None

    def test_already_tiered_model_unchanged(self):
        """Already tiered models should return unchanged."""
        from shared.ai.ollama_models import resolve_ollama_model

        model, ctx = resolve_ollama_model("ollama/qwen3:4b-instruct-4k", None, 4096)
        assert model == "ollama/qwen3:4b-instruct-4k"
        assert ctx == 4096
