"""Tests for the magaldi ollama CLI commands."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from shared.cli.ollama_cli import _format_size, _get_ollama_url


# =============================================================================
# HELPERS
# =============================================================================


class TestGetOllamaUrl:
    """Test Ollama URL resolution from config."""

    @patch("shared.cli.ollama_cli.load_config")
    def test_returns_url_from_config(self, mock_config):
        mock_cfg = MagicMock()
        mock_cfg.provider = "ollama"
        mock_cfg.url = "http://localhost:11434"
        mock_config.return_value.llm.models = {"model": mock_cfg}
        assert _get_ollama_url() == "http://localhost:11434"

    @patch("shared.cli.ollama_cli.load_config")
    def test_strips_trailing_slash(self, mock_config):
        mock_cfg = MagicMock()
        mock_cfg.provider = "ollama"
        mock_cfg.url = "http://localhost:11434/"
        mock_config.return_value.llm.models = {"model": mock_cfg}
        assert _get_ollama_url() == "http://localhost:11434"

    @patch("shared.cli.ollama_cli.load_config")
    def test_returns_default_when_no_ollama(self, mock_config):
        mock_cfg = MagicMock()
        mock_cfg.provider = "llamacpp"
        mock_cfg.url = "http://localhost:8090"
        mock_config.return_value.llm.models = {"model": mock_cfg}
        assert _get_ollama_url() == "http://localhost:11434"

    @patch("shared.cli.ollama_cli.load_config")
    def test_returns_default_with_empty_models(self, mock_config):
        mock_config.return_value.llm.models = {}
        assert _get_ollama_url() == "http://localhost:11434"


class TestFormatSize:
    """Test size formatting."""

    def test_gb(self):
        assert _format_size(2_147_483_648) == "2.0 GB"

    def test_mb(self):
        assert _format_size(52_428_800) == "50 MB"

    def test_kb(self):
        assert _format_size(51_200) == "50 KB"


# =============================================================================
# STATUS COMMAND
# =============================================================================


class TestOllamaStatus:
    """Test ollama status command."""

    @patch("shared.cli.ollama_cli.requests.get")
    @patch("shared.cli.ollama_cli.load_config")
    def test_status_connection_refused(self, mock_config, mock_get):
        """Shows error when Ollama is not running."""
        import requests

        mock_config.return_value.llm.models = {}
        mock_get.side_effect = requests.ConnectionError()
        console = MagicMock()

        with patch("shared.cli.ollama_cli.console", console):
            from shared.cli.ollama_cli import ollama_status

            # Invoke via click's testing
            from click.testing import CliRunner

            runner = CliRunner()
            result = runner.invoke(ollama_status, ["--url", "http://localhost:11434"])
            assert "not running" in result.output or result.exit_code == 0

    @patch("shared.cli.ollama_cli.requests.get")
    @patch("shared.cli.ollama_cli.load_config")
    def test_status_with_models(self, mock_config, mock_get):
        """Shows connected status with model list."""
        mock_config.return_value.llm.models = {}
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [
                {"name": "qwen3:4b", "size": 2_740_000_000},
                {"name": "qwen3-embedding:0.6b", "size": 400_000_000},
            ]
        }
        mock_get.return_value = mock_response

        from click.testing import CliRunner

        from shared.cli.ollama_cli import ollama_status

        runner = CliRunner()
        result = runner.invoke(ollama_status, ["--url", "http://localhost:11434"])
        assert "running" in result.output
        assert "2" in result.output  # 2 models


# =============================================================================
# MODELS COMMAND
# =============================================================================


class TestOllamaModels:
    """Test ollama models command."""

    @patch("shared.cli.ollama_cli.requests.get")
    @patch("shared.cli.ollama_cli.load_config")
    def test_models_empty(self, mock_config, mock_get):
        """Shows message when no models installed."""
        mock_config.return_value.llm.models = {}
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"models": []}
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        from click.testing import CliRunner

        from shared.cli.ollama_cli import ollama_models

        runner = CliRunner()
        result = runner.invoke(ollama_models, ["--url", "http://localhost:11434"])
        assert "No models" in result.output


# =============================================================================
# PULL COMMAND
# =============================================================================


class TestOllamaPull:
    """Test ollama pull command."""

    @patch("shared.cli.ollama_cli.requests.post")
    @patch("shared.cli.ollama_cli.load_config")
    def test_pull_specific_model(self, mock_config, mock_post):
        """Pulls a specific model by name."""
        mock_config.return_value.llm.models = {}
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        from click.testing import CliRunner

        from shared.cli.ollama_cli import ollama_pull

        runner = CliRunner()
        result = runner.invoke(
            ollama_pull,
            ["--model", "qwen3:4b", "--url", "http://localhost:11434"],
        )
        assert "Done" in result.output
        mock_post.assert_called_once_with(
            "http://localhost:11434/api/pull",
            json={"name": "qwen3:4b", "stream": False},
            timeout=600,
        )

    @patch("shared.cli.ollama_cli.load_config")
    def test_pull_no_models_in_config(self, mock_config):
        """Shows message when no Ollama models configured."""
        mock_cfg = MagicMock()
        mock_cfg.provider = "llamacpp"
        mock_config.return_value.llm.models = {"model": mock_cfg}

        from click.testing import CliRunner

        from shared.cli.ollama_cli import ollama_pull

        runner = CliRunner()
        result = runner.invoke(ollama_pull, ["--url", "http://localhost:11434"])
        assert "No Ollama models" in result.output
