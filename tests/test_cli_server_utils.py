"""Tests for shared server management utilities."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

from shared.cli._server_utils import (
    PIDFILE_DIR,
    format_size,
    get_pid,
    get_port_from_config,
    is_healthy,
    logfile,
    pidfile,
    stop_server,
    tail_log,
)

# =============================================================================
# PATH HELPERS
# =============================================================================


class TestPathHelpers:
    def test_pidfile_path(self):
        result = pidfile("llama-server", 8090)
        assert result == PIDFILE_DIR / "llama-server-8090.pid"

    def test_pidfile_path_vllm_mlx(self):
        result = pidfile("vllm-mlx", 8000)
        assert result == PIDFILE_DIR / "vllm-mlx-8000.pid"

    def test_logfile_path(self):
        result = logfile("llama-server", 8090)
        assert result == PIDFILE_DIR / "llama-server-8090.log"

    def test_logfile_path_vllm_mlx(self):
        result = logfile("vllm-mlx", 8000)
        assert result == PIDFILE_DIR / "vllm-mlx-8000.log"


# =============================================================================
# PID MANAGEMENT
# =============================================================================


class TestGetPid:
    def test_no_pidfile_returns_none(self, tmp_path):
        with patch("shared.cli._server_utils.PIDFILE_DIR", tmp_path):
            assert get_pid("test-server", 9999) is None

    def test_valid_pid_returned(self, tmp_path):
        pf = tmp_path / "test-server-9999.pid"
        pf.write_text(str(os.getpid()))  # Use our own PID (guaranteed alive)
        with patch("shared.cli._server_utils.PIDFILE_DIR", tmp_path):
            assert get_pid("test-server", 9999) == os.getpid()

    def test_stale_pid_cleaned_up(self, tmp_path):
        pf = tmp_path / "test-server-9999.pid"
        pf.write_text("999999999")  # Non-existent PID
        with patch("shared.cli._server_utils.PIDFILE_DIR", tmp_path):
            assert get_pid("test-server", 9999) is None
            assert not pf.exists()  # Cleaned up

    def test_invalid_pid_content(self, tmp_path):
        pf = tmp_path / "test-server-9999.pid"
        pf.write_text("not-a-number")
        with patch("shared.cli._server_utils.PIDFILE_DIR", tmp_path):
            assert get_pid("test-server", 9999) is None
            assert not pf.exists()


# =============================================================================
# HEALTH CHECK
# =============================================================================


class TestIsHealthy:
    @patch("shared.cli._server_utils.requests.get")
    def test_healthy_server(self, mock_get):
        mock_get.return_value.status_code = 200
        assert is_healthy(8090) is True
        mock_get.assert_called_once_with("http://localhost:8090/health", timeout=3)

    @patch("shared.cli._server_utils.requests.get")
    def test_custom_path(self, mock_get):
        mock_get.return_value.status_code = 200
        assert is_healthy(8000, path="/api/tags") is True
        mock_get.assert_called_once_with("http://localhost:8000/api/tags", timeout=3)

    @patch("shared.cli._server_utils.requests.get")
    def test_unhealthy_server(self, mock_get):
        mock_get.return_value.status_code = 503
        assert is_healthy(8090) is False

    @patch("shared.cli._server_utils.requests.get")
    def test_connection_refused(self, mock_get):
        mock_get.side_effect = ConnectionRefusedError()
        assert is_healthy(8090) is False


# =============================================================================
# STOP SERVER
# =============================================================================


class TestStopServer:
    def test_stop_no_server_running(self, tmp_path):
        console = MagicMock()
        with patch("shared.cli._server_utils.PIDFILE_DIR", tmp_path):
            stop_server("test-server", 9999, console)
        console.print.assert_called_with("[dim]No test-server running on port 9999[/]")

    @patch("shared.cli._server_utils.os.kill")
    def test_stop_running_server(self, mock_kill, tmp_path):
        pf = tmp_path / "test-server-9999.pid"
        pf.write_text("12345")
        console = MagicMock()

        # First os.kill(pid, 0) succeeds (alive), then os.kill(pid, SIGTERM),
        # then os.kill(pid, 0) raises ProcessLookupError (died)
        mock_kill.side_effect = [
            None,  # get_pid: os.kill(12345, 0) -> alive
            None,  # _stop_pid: os.kill(12345, SIGTERM)
            ProcessLookupError,  # _stop_pid: os.kill(12345, 0) -> dead
        ]

        with patch("shared.cli._server_utils.PIDFILE_DIR", tmp_path), \
             patch("shared.cli._server_utils.time.sleep"):
            stop_server("test-server", 9999, console)

        # PID file should be cleaned up
        assert not pf.exists()


# =============================================================================
# FORMAT SIZE
# =============================================================================


class TestFormatSize:
    def test_gb(self):
        assert format_size(2_147_483_648) == "2.0 GB"

    def test_mb(self):
        assert format_size(52_428_800) == "50 MB"

    def test_kb(self):
        assert format_size(51_200) == "50 KB"

    def test_fractional_gb(self):
        assert format_size(3_758_096_384) == "3.5 GB"


# =============================================================================
# PORT FROM CONFIG
# =============================================================================


class TestGetPortFromConfig:
    def test_finds_port_from_matching_provider(self):
        mock_cfg = MagicMock()
        mock_cfg.provider = "llamacpp"
        mock_cfg.url = "http://localhost:8090"
        models = {"model1": mock_cfg}
        assert get_port_from_config("llamacpp", models, 9999) == 8090

    def test_returns_default_when_no_match(self):
        mock_cfg = MagicMock()
        mock_cfg.provider = "ollama"
        mock_cfg.url = "http://localhost:11434"
        models = {"model1": mock_cfg}
        assert get_port_from_config("vllm-mlx", models, 8000) == 8000

    def test_returns_default_with_empty_models(self):
        assert get_port_from_config("llamacpp", {}, 8090) == 8090

    def test_finds_vllm_mlx_port(self):
        mock_cfg = MagicMock()
        mock_cfg.provider = "vllm-mlx"
        mock_cfg.url = "http://localhost:8000"
        models = {"model1": mock_cfg}
        assert get_port_from_config("vllm-mlx", models, 9999) == 8000


# =============================================================================
# TAIL LOG
# =============================================================================


class TestTailLog:
    def test_no_logfile(self, tmp_path):
        console = MagicMock()
        with patch("shared.cli._server_utils.PIDFILE_DIR", tmp_path):
            tail_log("test-server", 9999, console)
        console.print.assert_any_call("[red]No log file found for port 9999[/]")

    @patch("shared.cli._server_utils.subprocess.run")
    def test_tail_without_follow(self, mock_run, tmp_path):
        log = tmp_path / "test-server-9999.log"
        log.write_text("some log output\n")
        mock_run.return_value.stdout = "some log output\n"
        console = MagicMock()

        with patch("shared.cli._server_utils.PIDFILE_DIR", tmp_path):
            tail_log("test-server", 9999, console, follow=False, lines=20)

        mock_run.assert_called_once_with(
            ["tail", "-n", "20", str(log)],
            capture_output=True,
            text=True,
        )
