"""Tests for the magaldi llamacpp CLI command (llama-server backend)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from shared.cli.llamacpp import (
    DEFAULT_CTX_SIZE,
    DEFAULT_MODELS_MAX,
    DEFAULT_PARALLEL,
    DEFAULT_PORT,
    MODELS_DIR,
    _format_size,
    _generate_presets,
    _get_llamacpp_models,
    _get_llamacpp_port,
    _logfile,
    _pidfile,
)
from shared.config import LLMConfig, ModelConfig

# =============================================================================
# Helper functions
# =============================================================================


class TestHelpers:
    """Test helper functions."""

    def test_pidfile_path(self) -> None:
        pf = _pidfile(8090)
        assert pf.name == "llama-server-8090.pid"

    def test_logfile_path(self) -> None:
        lf = _logfile(8090)
        assert lf.name == "llama-server-8090.log"

    def test_format_size_gb(self) -> None:
        assert _format_size(2_740_000_000) == "2.6 GB"

    def test_format_size_mb(self) -> None:
        assert _format_size(150_000_000) == "143 MB"

    def test_format_size_kb(self) -> None:
        assert _format_size(500_000) == "488 KB"

    def test_defaults(self) -> None:
        assert DEFAULT_PORT == 8090
        assert DEFAULT_PARALLEL == 4
        assert DEFAULT_MODELS_MAX == 2
        assert DEFAULT_CTX_SIZE == 131072


# =============================================================================
# _get_llamacpp_models
# =============================================================================


class TestGetLlamacppModels:
    """Test llamacpp model filtering from config."""

    def test_no_llamacpp_models_returns_empty(self) -> None:
        config = LLMConfig(
            models={
                "qwen3-4b": ModelConfig(
                    name="qwen3:4b-instruct",
                    provider="ollama",
                    url="http://localhost:11434",
                ),
            }
        )
        assert _get_llamacpp_models(config) == []

    def test_filters_llamacpp_only(self) -> None:
        config = LLMConfig(
            models={
                "ollama-model": ModelConfig(
                    name="qwen3:4b-instruct",
                    provider="ollama",
                    url="http://localhost:11434",
                ),
                "llamacpp-model": ModelConfig(
                    name="Qwen3.5-4B-Q4_K_M",
                    provider="llamacpp",
                    url="http://localhost:8090",
                ),
                "cloud-model": ModelConfig(
                    name="gpt-4o-mini",
                    provider="openai",
                    url="",
                ),
            }
        )
        models = _get_llamacpp_models(config)
        assert len(models) == 1
        assert models[0].name == "Qwen3.5-4B-Q4_K_M"

    def test_multiple_llamacpp_models(self) -> None:
        config = LLMConfig(
            models={
                "qwen3.5-4b": ModelConfig(
                    name="Qwen3.5-4B-Q4_K_M",
                    provider="llamacpp",
                    url="http://localhost:8090",
                ),
                "qwen3.5-2b": ModelConfig(
                    name="Qwen3.5-2B-Q4_K_M",
                    provider="llamacpp",
                    url="http://localhost:8090",
                ),
            }
        )
        models = _get_llamacpp_models(config)
        assert len(models) == 2


# =============================================================================
# _get_llamacpp_port
# =============================================================================


class TestGetLlamacppPort:
    """Test port extraction from config."""

    def test_returns_port_from_config(self) -> None:
        config = LLMConfig(
            models={
                "qwen3.5-4b": ModelConfig(
                    name="Qwen3.5-4B-Q4_K_M",
                    provider="llamacpp",
                    url="http://localhost:8090",
                ),
            }
        )
        assert _get_llamacpp_port(config) == 8090

    def test_returns_default_when_no_llamacpp(self) -> None:
        config = LLMConfig(
            models={
                "ollama": ModelConfig(
                    name="qwen3:4b",
                    provider="ollama",
                    url="http://localhost:11434",
                ),
            }
        )
        assert _get_llamacpp_port(config) == DEFAULT_PORT

    def test_custom_port(self) -> None:
        config = LLMConfig(
            models={
                "model": ModelConfig(
                    name="Qwen3.5-4B-Q4_K_M",
                    provider="llamacpp",
                    url="http://localhost:9999",
                ),
            }
        )
        assert _get_llamacpp_port(config) == 9999


# =============================================================================
# _generate_presets
# =============================================================================


class TestGeneratePresets:
    """Test INI presets file generation."""

    def test_generates_presets_with_num_ctx(self, tmp_path: Path) -> None:
        config = LLMConfig(
            models={
                "qwen3.5-4b": ModelConfig(
                    name="Qwen3.5-4B-Q4_K_M",
                    provider="llamacpp",
                    url="http://localhost:8090",
                    num_ctx=16384,
                ),
            }
        )
        with patch("shared.cli.llamacpp._presets_file", return_value=tmp_path / "presets.ini"):
            path = _generate_presets(config)
            content = path.read_text()
            assert "[model:Qwen3.5-4B-Q4_K_M]" in content
            assert "n_ctx = 16384" in content

    def test_empty_presets_when_no_num_ctx(self, tmp_path: Path) -> None:
        config = LLMConfig(
            models={
                "qwen3.5-4b": ModelConfig(
                    name="Qwen3.5-4B-Q4_K_M",
                    provider="llamacpp",
                    url="http://localhost:8090",
                ),
            }
        )
        with patch("shared.cli.llamacpp._presets_file", return_value=tmp_path / "presets.ini"):
            path = _generate_presets(config)
            content = path.read_text()
            assert "[model:Qwen3.5-4B-Q4_K_M]" in content

    def test_skips_non_llamacpp_models(self, tmp_path: Path) -> None:
        config = LLMConfig(
            models={
                "ollama-model": ModelConfig(
                    name="qwen3:4b",
                    provider="ollama",
                    url="http://localhost:11434",
                    num_ctx=8192,
                ),
            }
        )
        with patch("shared.cli.llamacpp._presets_file", return_value=tmp_path / "presets.ini"):
            path = _generate_presets(config)
            content = path.read_text()
            assert "qwen3:4b" not in content


# =============================================================================
# ModelConfig with llamacpp + gguf
# =============================================================================


class TestModelConfigLlamacpp:
    """Test ModelConfig with llamacpp provider and gguf field."""

    def test_get_litellm_model(self) -> None:
        cfg = ModelConfig(
            name="Qwen3.5-4B-Q4_K_M",
            provider="llamacpp",
            url="http://localhost:8090",
        )
        assert cfg.get_litellm_model() == "openai/Qwen3.5-4B-Q4_K_M"

    def test_get_api_base(self) -> None:
        cfg = ModelConfig(
            name="Qwen3.5-4B-Q4_K_M",
            provider="llamacpp",
            url="http://localhost:8090",
        )
        assert cfg.get_api_base() == "http://localhost:8090/v1"

    def test_get_api_base_strips_trailing_slash(self) -> None:
        cfg = ModelConfig(
            name="Qwen3.5-4B-Q4_K_M",
            provider="llamacpp",
            url="http://localhost:8090/",
        )
        assert cfg.get_api_base() == "http://localhost:8090/v1"

    def test_gguf_field(self) -> None:
        cfg = ModelConfig(
            name="Qwen3.5-4B-Q4_K_M",
            provider="llamacpp",
            url="http://localhost:8090",
            gguf="unsloth/Qwen3.5-4B-GGUF:Qwen3.5-4B-Q4_K_M.gguf",
        )
        assert cfg.gguf == "unsloth/Qwen3.5-4B-GGUF:Qwen3.5-4B-Q4_K_M.gguf"

    def test_gguf_default_none(self) -> None:
        cfg = ModelConfig(name="qwen3:4b", provider="ollama")
        assert cfg.gguf is None

    def test_ollama_provider_unchanged(self) -> None:
        cfg = ModelConfig(name="qwen3:4b-instruct", provider="ollama")
        assert cfg.get_litellm_model() == "ollama_chat/qwen3:4b-instruct"
        assert cfg.get_api_base() == "http://localhost:11434"

    def test_is_not_thinking_model_qwen35(self) -> None:
        """Qwen3.5 GGUF model is NOT a thinking model."""
        cfg = ModelConfig(
            name="Qwen3.5-4B-Q4_K_M",
            provider="llamacpp",
            url="http://localhost:8090",
        )
        assert cfg.is_thinking_model() is False

    def test_models_dir_exists(self) -> None:
        """MODELS_DIR points to tools/models/ relative to project root."""
        assert MODELS_DIR.name == "models"
        assert MODELS_DIR.parent.name == "tools"
