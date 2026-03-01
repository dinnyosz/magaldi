"""Tests for shared.cli.parse_logger module."""

from __future__ import annotations

import json
import time
from pathlib import Path

from shared.cli.parse_logger import ParseRunLogger, _format_duration_precise


class TestParseRunLogger:
    """Tests for ParseRunLogger."""

    def test_init_creates_log_directory(self, tmp_path: Path) -> None:
        """Logger creates logs/ directory on init."""
        repo_path = str(tmp_path)
        logger = ParseRunLogger(repo_path, "test-scope", "test-repo", "main")
        assert (tmp_path / "logs").is_dir()
        assert logger.log_path.parent == tmp_path / "logs"
        assert "parse_test-scope_test-repo_" in logger.log_path.name
        assert logger.log_path.suffix == ".log"

    def test_init_watch_mode(self, tmp_path: Path) -> None:
        """Logger in watch mode uses 'watch' prefix."""
        logger = ParseRunLogger(str(tmp_path), "s", "r", "u", mode="watch")
        assert "watch_s_r_" in logger.log_path.name

    def test_phase_timing(self, tmp_path: Path) -> None:
        """Phase start/end tracking with durations."""
        logger = ParseRunLogger(str(tmp_path), "s", "r", "u")
        logger.start_phase("Phase 1: Discovery")
        time.sleep(0.05)
        duration = logger.end_phase({"files": 42})
        assert duration >= 0.04  # Allow some tolerance

        timings = logger.get_phase_timings()
        assert len(timings) == 1
        assert timings[0][0] == "Phase 1: Discovery"
        assert timings[0][1] >= 0.04

    def test_phase_auto_end(self, tmp_path: Path) -> None:
        """Starting a new phase auto-ends the previous one."""
        logger = ParseRunLogger(str(tmp_path), "s", "r", "u")
        logger.start_phase("Phase 1")
        time.sleep(0.02)
        logger.start_phase("Phase 2")
        time.sleep(0.02)
        logger.end_phase()

        timings = logger.get_phase_timings()
        assert len(timings) == 2
        assert timings[0][0] == "Phase 1"
        assert timings[1][0] == "Phase 2"

    def test_end_phase_without_start(self, tmp_path: Path) -> None:
        """Ending a phase without starting returns 0."""
        logger = ParseRunLogger(str(tmp_path), "s", "r", "u")
        duration = logger.end_phase()
        assert duration == 0.0

    def test_error_logging(self, tmp_path: Path) -> None:
        """Errors are accumulated with context."""
        logger = ParseRunLogger(str(tmp_path), "s", "r", "u")
        logger.log_error("parsing", "Failed to parse file", {"file": "foo.py"})
        logger.log_error("processing", "LLM timeout", {"element_id": "s:r:u:foo.py:fn:bar:10"})

        assert logger.has_errors
        assert len(logger._errors) == 2

    def test_no_errors(self, tmp_path: Path) -> None:
        """has_errors is False when no errors logged."""
        logger = ParseRunLogger(str(tmp_path), "s", "r", "u")
        assert not logger.has_errors

    def test_budget_exceeded(self, tmp_path: Path) -> None:
        """Token budget exceeded entries are logged."""
        logger = ParseRunLogger(str(tmp_path), "s", "r", "u")
        logger.log_budget_exceeded(
            input_rows=[("function", 2048, 10, 2, 15.0, -5.0)],
            output_rows=[("class", 300, 600, 450)],
        )

        assert logger.has_budget_exceeded
        assert len(logger._budget_exceeded) == 1
        entry = logger._budget_exceeded[0]
        assert len(entry["input_overflows"]) == 1
        assert entry["input_overflows"][0]["element_type"] == "function"
        assert entry["input_overflows"][0]["overflows"] == 2
        assert len(entry["output_exceeded"]) == 1
        assert entry["output_exceeded"][0]["exceeded_by"] == 150

    def test_no_budget_exceeded(self, tmp_path: Path) -> None:
        """has_budget_exceeded is False when no entries."""
        logger = ParseRunLogger(str(tmp_path), "s", "r", "u")
        assert not logger.has_budget_exceeded

    def test_processing_stats(self, tmp_path: Path) -> None:
        """Processing stats are recorded."""
        logger = ParseRunLogger(str(tmp_path), "s", "r", "u")
        logger.log_processing_stats(
            processed=100, skipped=20, indexed=80, deleted=5,
            failed_elements=[("id1", "timeout"), ("id2", "bad embedding")],
            avg_wall=1.5, avg_summ=0.8, avg_embed=0.3, elapsed=120.0,
        )

        assert logger._processing_stats["processed"] == 100
        assert logger._processing_stats["failed_count"] == 2
        # Failed elements are also logged as errors
        assert len(logger._errors) == 2

    def test_write_creates_file(self, tmp_path: Path) -> None:
        """write() creates a log file with proper structure."""
        logger = ParseRunLogger(str(tmp_path), "test-scope", "test-repo", "main")
        logger.start_phase("Phase 1: Discovery")
        time.sleep(0.01)
        logger.end_phase({"files": 10})
        logger.log_error("parsing", "Test error", {"file": "test.py"})
        logger.log_budget_exceeded(
            input_rows=[],
            output_rows=[("function", 200, 600, 500)],
        )
        logger.log_processing_stats(
            processed=50, skipped=10, indexed=40, deleted=0,
            failed_elements=[], elapsed=30.0,
        )

        log_path = logger.write()
        assert log_path.exists()

        content = log_path.read_text()

        # Check sections exist
        assert "PHASE TIMING" in content
        assert "Phase 1: Discovery" in content
        assert "TOTAL" in content
        assert "ERRORS (1 total)" in content
        assert "Test error" in content
        assert "TOKEN BUDGET EXCEEDED" in content
        assert "Output Token Budget Exceeded" in content
        assert "PROCESSING" in content
        assert "JSON SUMMARY" in content

        # Parse JSON summary
        json_start = content.index("{", content.index("JSON SUMMARY"))
        json_end = content.rindex("}") + 1
        summary = json.loads(content[json_start:json_end])
        assert summary["scope"] == "test-scope"
        assert summary["repository"] == "test-repo"
        assert summary["error_count"] == 1
        assert summary["budget_exceeded"] is True
        assert "Phase 1: Discovery" in summary["phase_timings"]

    def test_write_minimal(self, tmp_path: Path) -> None:
        """write() works with no data (empty run)."""
        logger = ParseRunLogger(str(tmp_path), "s", "r", "u")
        log_path = logger.write()
        assert log_path.exists()
        content = log_path.read_text()
        assert "PHASE TIMING" in content
        assert "TOTAL" in content
        # No error or budget sections
        assert "ERRORS" not in content
        assert "TOKEN BUDGET EXCEEDED" not in content

    def test_write_ends_open_phase(self, tmp_path: Path) -> None:
        """write() automatically ends any open phase."""
        logger = ParseRunLogger(str(tmp_path), "s", "r", "u")
        logger.start_phase("Open Phase")
        time.sleep(0.01)
        # Don't end it manually
        log_path = logger.write()
        content = log_path.read_text()
        assert "Open Phase" in content

    def test_scope_repo_update(self, tmp_path: Path) -> None:
        """Scope/repo can be updated after init (for parse command pattern)."""
        logger = ParseRunLogger(str(tmp_path), "unknown", "unknown", "main")
        logger.scope = "real-scope"
        logger.repository = "real-repo"

        ts = logger.run_start_dt.strftime("%Y%m%d_%H%M%S")
        log_dir = tmp_path / "logs"
        logger.log_path = log_dir / f"parse_real-scope_real-repo_{ts}.log"

        log_path = logger.write()
        content = log_path.read_text()
        assert "real-scope/real-repo" in content

    def test_get_total_elapsed(self, tmp_path: Path) -> None:
        """Total elapsed time is calculated from run start."""
        logger = ParseRunLogger(str(tmp_path), "s", "r", "u")
        time.sleep(0.05)
        elapsed = logger.get_total_elapsed()
        assert elapsed >= 0.04

    def test_safe_scope_repo_in_filename(self, tmp_path: Path) -> None:
        """Slashes in scope/repo are replaced in filename."""
        logger = ParseRunLogger(str(tmp_path), "org/team", "my/repo", "u")
        assert "/" not in logger.log_path.name
        assert "org_team" in logger.log_path.name
        assert "my_repo" in logger.log_path.name


class TestFormatDurationPrecise:
    """Tests for _format_duration_precise."""

    def test_sub_second(self) -> None:
        assert _format_duration_precise(0.5) == "0.50s"

    def test_seconds(self) -> None:
        assert _format_duration_precise(15.234) == "15.23s"

    def test_minutes(self) -> None:
        result = _format_duration_precise(65.5)
        assert result == "1:05.5"

    def test_hours(self) -> None:
        result = _format_duration_precise(3723.0)
        assert result == "1:02:03"

    def test_zero(self) -> None:
        assert _format_duration_precise(0.0) == "0.00s"
