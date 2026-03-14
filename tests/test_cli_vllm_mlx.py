"""Tests for the magaldi vllm-mlx CLI commands."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from shared.cli.vllm_mlx import DEFAULT_PORT, PREFIX


# =============================================================================
# CONSTANTS
# =============================================================================


class TestConstants:
    def test_prefix(self):
        assert PREFIX == "vllm-mlx"

    def test_default_port(self):
        assert DEFAULT_PORT == 8000


# =============================================================================
# SERVE COMMAND
# =============================================================================


class TestVllmMlxServe:
    @patch("shared.cli.vllm_mlx.load_config")
    @patch("shared.cli.vllm_mlx.shutil.which")
    def test_serve_no_binary(self, mock_which, mock_config):
        """Shows error when vllm-mlx is not installed."""
        mock_which.return_value = None
        mock_config.return_value.llm.models = {}

        from shared.cli.vllm_mlx import vllm_mlx_serve

        runner = CliRunner()
        result = runner.invoke(vllm_mlx_serve, [])
        assert "not found" in result.output
        assert "pip install vllm-mlx" in result.output

    @patch("shared.cli.vllm_mlx.load_config")
    @patch("shared.cli.vllm_mlx.shutil.which")
    def test_serve_no_model(self, mock_which, mock_config):
        """Shows error when no model specified and none in config."""
        mock_which.return_value = "/usr/local/bin/vllm-mlx"
        mock_config.return_value.llm.models = {}

        from shared.cli.vllm_mlx import vllm_mlx_serve

        runner = CliRunner()
        result = runner.invoke(vllm_mlx_serve, [])
        assert "No model specified" in result.output

    @patch("shared.cli.vllm_mlx.load_config")
    @patch("shared.cli.vllm_mlx.shutil.which")
    @patch("shared.cli.vllm_mlx.get_pid")
    @patch("shared.cli.vllm_mlx.is_healthy")
    def test_serve_already_running(self, mock_healthy, mock_pid, mock_which, mock_config):
        """Shows message when server is already running."""
        mock_which.return_value = "/usr/local/bin/vllm-mlx"
        mock_config.return_value.llm.models = {}
        mock_pid.return_value = 12345
        mock_healthy.return_value = True

        from shared.cli.vllm_mlx import vllm_mlx_serve

        runner = CliRunner()
        result = runner.invoke(vllm_mlx_serve, ["--model", "mlx-community/Qwen3-4B-4bit"])
        assert "already running" in result.output


# =============================================================================
# STOP COMMAND
# =============================================================================


class TestVllmMlxStop:
    @patch("shared.cli.vllm_mlx.load_config")
    @patch("shared.cli.vllm_mlx.stop_server")
    def test_stop_calls_server_utils(self, mock_stop, mock_config):
        """Delegates to shared stop_server utility."""
        mock_config.return_value.llm.models = {}

        from shared.cli.vllm_mlx import vllm_mlx_stop

        runner = CliRunner()
        runner.invoke(vllm_mlx_stop, ["--port", "8000"])
        mock_stop.assert_called_once()
        args = mock_stop.call_args
        assert args[0][0] == "vllm-mlx"
        assert args[0][1] == 8000


# =============================================================================
# STATUS COMMAND
# =============================================================================


class TestVllmMlxStatus:
    @patch("shared.cli.vllm_mlx.load_config")
    @patch("shared.cli.vllm_mlx.get_pid")
    @patch("shared.cli.vllm_mlx.is_healthy")
    def test_status_not_running(self, mock_healthy, mock_pid, mock_config):
        """Shows not running when no PID."""
        mock_config.return_value.llm.models = {}
        mock_pid.return_value = None
        mock_healthy.return_value = False

        from shared.cli.vllm_mlx import vllm_mlx_status

        runner = CliRunner()
        result = runner.invoke(vllm_mlx_status, ["--port", "8000"])
        assert "not running" in result.output

    @patch("shared.cli.vllm_mlx.requests.get")
    @patch("shared.cli.vllm_mlx.load_config")
    @patch("shared.cli.vllm_mlx.get_pid")
    @patch("shared.cli.vllm_mlx.is_healthy")
    def test_status_running_with_models(self, mock_healthy, mock_pid, mock_config, mock_get):
        """Shows running status with model list."""
        mock_config.return_value.llm.models = {}
        mock_pid.return_value = 12345
        mock_healthy.return_value = True
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {"id": "mlx-community/Qwen3-4B-4bit", "object": "model"},
            ]
        }
        mock_get.return_value = mock_response

        from shared.cli.vllm_mlx import vllm_mlx_status

        runner = CliRunner()
        result = runner.invoke(vllm_mlx_status, ["--port", "8000"])
        assert "running" in result.output
        assert "Qwen3-4B-4bit" in result.output


# =============================================================================
# LOGS COMMAND
# =============================================================================


class TestVllmMlxLogs:
    @patch("shared.cli.vllm_mlx.load_config")
    @patch("shared.cli.vllm_mlx.tail_log")
    def test_logs_delegates_to_tail_log(self, mock_tail, mock_config):
        """Delegates to shared tail_log utility."""
        mock_config.return_value.llm.models = {}

        from shared.cli.vllm_mlx import vllm_mlx_logs

        runner = CliRunner()
        runner.invoke(vllm_mlx_logs, ["--port", "8000", "--lines", "20"])
        mock_tail.assert_called_once()
        args = mock_tail.call_args
        assert args[0][0] == "vllm-mlx"
        assert args[0][1] == 8000
