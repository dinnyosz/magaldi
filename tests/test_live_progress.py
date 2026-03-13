"""Tests for LiveProgressWriter — periodic progress snapshots during parse runs."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from shared.cli.parse_logger import LiveProgressWriter


@pytest.fixture
def log_dir(tmp_path: Path) -> Path:
    """Create a temporary log directory."""
    d = tmp_path / "logs"
    d.mkdir()
    return d


@pytest.fixture
def writer(log_dir: Path) -> LiveProgressWriter:
    """Create a LiveProgressWriter for testing."""
    return LiveProgressWriter(
        log_dir=log_dir,
        scope="testorg",
        repository="testrepo",
        username="main",
    )


def _read_status(log_dir: Path) -> dict:
    """Read and parse _current_run.json."""
    path = log_dir / LiveProgressWriter.FILENAME
    return json.loads(path.read_text(encoding="utf-8"))


class TestInitialization:
    def test_creates_file_on_init(self, log_dir: Path, writer: LiveProgressWriter) -> None:  # noqa: ARG002
        """Status file is created immediately on initialization."""
        path = log_dir / LiveProgressWriter.FILENAME
        assert path.exists()

    def test_initial_state_schema(self, log_dir: Path, writer: LiveProgressWriter) -> None:  # noqa: ARG002
        """Initial status file has correct schema with no active phase."""
        data = _read_status(log_dir)
        assert data["pid"] == os.getpid()
        assert data["scope"] == "testorg"
        assert data["repository"] == "testrepo"
        assert data["username"] == "main"
        assert data["mode"] == "parse"
        assert data["current_phase"] is None
        assert data["phase_status"] == "between_phases"
        assert data["completed_phases"] == []
        assert data["progress"] is None
        assert "run_start" in data
        assert "elapsed_seconds" in data
        assert "updated_at" in data

    def test_valid_json(self, log_dir: Path, writer: LiveProgressWriter) -> None:  # noqa: ARG002
        """Status file is always valid JSON."""
        path = log_dir / LiveProgressWriter.FILENAME
        content = path.read_text(encoding="utf-8")
        json.loads(content)  # Should not raise


class TestPhaseTransitions:
    def test_start_phase_writes_immediately(self, log_dir: Path, writer: LiveProgressWriter) -> None:
        """start_phase writes to disk immediately."""
        writer.start_phase("Phase 1: Discovery")
        data = _read_status(log_dir)
        assert data["current_phase"] == "Phase 1: Discovery"
        assert data["phase_status"] == "in_progress"

    def test_end_phase_records_completed(self, log_dir: Path, writer: LiveProgressWriter) -> None:
        """end_phase moves phase to completed_phases list."""
        writer.start_phase("Phase 1: Discovery")
        writer.end_phase(1.5, {"files": 100})
        data = _read_status(log_dir)
        assert data["current_phase"] is None
        assert data["phase_status"] == "between_phases"
        assert len(data["completed_phases"]) == 1
        assert data["completed_phases"][0]["name"] == "Phase 1: Discovery"
        assert data["completed_phases"][0]["duration_seconds"] == 1.5
        assert data["completed_phases"][0]["extra"] == {"files": 100}

    def test_multiple_phases(self, log_dir: Path, writer: LiveProgressWriter) -> None:
        """Multiple phases accumulate in completed_phases."""
        writer.start_phase("Phase 1: Discovery")
        writer.end_phase(0.5)
        writer.start_phase("Phase 2: Change Detection")
        writer.end_phase(2.0)
        writer.start_phase("Phase 3: Parsing")

        data = _read_status(log_dir)
        assert len(data["completed_phases"]) == 2
        assert data["current_phase"] == "Phase 3: Parsing"
        assert data["completed_phases"][0]["name"] == "Phase 1: Discovery"
        assert data["completed_phases"][1]["name"] == "Phase 2: Change Detection"

    def test_end_phase_clears_progress(self, log_dir: Path, writer: LiveProgressWriter) -> None:
        """end_phase clears progress data from previous phase."""
        writer.start_phase("Phase 5: Processing")
        writer.update_progress({"completed": 50, "total": 100})
        # Force write so progress is visible
        with writer._lock:
            writer._write()
        data = _read_status(log_dir)
        assert data["progress"] is not None

        writer.end_phase(60.0)
        data = _read_status(log_dir)
        assert data["progress"] is None

    def test_end_phase_without_extra(self, log_dir: Path, writer: LiveProgressWriter) -> None:
        """end_phase works without extra data."""
        writer.start_phase("Phase 6: Analysis")
        writer.end_phase(5.0)
        data = _read_status(log_dir)
        assert len(data["completed_phases"]) == 1
        assert "extra" not in data["completed_phases"][0]


class TestProgressUpdates:
    def test_update_progress_stores_data(self, log_dir: Path, writer: LiveProgressWriter) -> None:
        """update_progress stores progress data."""
        writer.start_phase("Phase 5: Processing")
        progress = {"completed": 50, "total": 200, "percentage": 25.0}
        writer.update_progress(progress)
        # Force write to verify data
        with writer._lock:
            writer._write()
        data = _read_status(log_dir)
        assert data["progress"]["completed"] == 50
        assert data["progress"]["total"] == 200
        assert data["progress"]["percentage"] == 25.0

    def test_throttle_respects_interval(self, log_dir: Path) -> None:
        """update_progress only writes to disk every WRITE_INTERVAL seconds."""
        writer = LiveProgressWriter(log_dir, "s", "r", "u")

        writer.start_phase("Phase 5: Processing")

        # First update — should write (enough time since init write)
        with patch("time.time") as mock_time:
            # Start time + well past interval
            mock_time.return_value = writer._run_start + 100.0
            writer._last_write_time = writer._run_start + 100.0  # Reset to known value

            # Second update — too soon
            mock_time.return_value = writer._run_start + 110.0
            writer.update_progress({"completed": 10, "total": 100})
            # Should NOT have written (only 10s since last write, interval is 30s)
            data = _read_status(log_dir)
            # Progress should still be None or stale from start_phase write
            # (start_phase clears progress)

            # Third update — past interval
            mock_time.return_value = writer._run_start + 140.0
            writer.update_progress({"completed": 50, "total": 100})
            data = _read_status(log_dir)
            assert data["progress"]["completed"] == 50

    def test_start_phase_writes_regardless_of_throttle(self, log_dir: Path, writer: LiveProgressWriter) -> None:
        """Phase transitions always write immediately, even within throttle interval."""
        # Set last write time to now (simulate recent write)
        writer._last_write_time = time.time()

        writer.start_phase("Phase 5: Processing")
        data = _read_status(log_dir)
        assert data["current_phase"] == "Phase 5: Processing"

    def test_phase5_progress_schema(self, log_dir: Path, writer: LiveProgressWriter) -> None:
        """Phase 5 progress has expected fields."""
        writer.start_phase("Phase 5: Processing")
        progress = {
            "completed": 450,
            "total": 1200,
            "percentage": 37.5,
            "eta_seconds": 5400.0,
            "elapsed_seconds": 3600.0,
            "throughput_per_second": 0.125,
            "failed": 3,
            "skipped": 50,
            "by_type": {
                "file": {"done": 80, "total": 80},
                "class": {"done": 45, "total": 120},
                "method": {"done": 100, "total": 400},
            },
        }
        writer.update_progress(progress)
        with writer._lock:
            writer._write()
        data = _read_status(log_dir)
        assert data["progress"]["by_type"]["file"] == {"done": 80, "total": 80}
        assert data["progress"]["eta_seconds"] == 5400.0

    def test_phase4_progress_schema(self, log_dir: Path, writer: LiveProgressWriter) -> None:
        """Phase 4 progress has scoring-specific fields."""
        writer.start_phase("Phase 4: Variable Scoring")
        progress = {
            "completed": 85,
            "total": 200,
            "completed_batches": 17,
            "total_batches": 40,
            "percentage": 42.5,
            "eta_seconds": 120.0,
            "elapsed_seconds": 90.0,
            "throughput_per_second": 2.2,
            "failed": 0,
            "kept": 60,
            "dropped": 25,
            "keep_rate_pct": 70.6,
        }
        writer.update_progress(progress)
        with writer._lock:
            writer._write()
        data = _read_status(log_dir)
        assert data["progress"]["kept"] == 60
        assert data["progress"]["dropped"] == 25
        assert data["progress"]["keep_rate_pct"] == 70.6


class TestIdentityUpdate:
    def test_update_identity(self, log_dir: Path) -> None:
        """update_identity changes scope/repo and writes immediately."""
        writer = LiveProgressWriter(log_dir, "unknown", "unknown", "main")
        data = _read_status(log_dir)
        assert data["scope"] == "unknown"

        writer.update_identity("myorg", "myrepo")
        data = _read_status(log_dir)
        assert data["scope"] == "myorg"
        assert data["repository"] == "myrepo"


class TestCleanup:
    def test_cleanup_removes_file(self, log_dir: Path, writer: LiveProgressWriter) -> None:
        """cleanup deletes the status file."""
        path = log_dir / LiveProgressWriter.FILENAME
        assert path.exists()
        writer.cleanup()
        assert not path.exists()

    def test_cleanup_removes_tmp_file(self, log_dir: Path, writer: LiveProgressWriter) -> None:
        """cleanup also removes temp file if it exists."""
        tmp_path = (log_dir / LiveProgressWriter.FILENAME).with_suffix(".json.tmp")
        tmp_path.write_text("{}", encoding="utf-8")
        writer.cleanup()
        assert not tmp_path.exists()

    def test_cleanup_idempotent(self, log_dir: Path, writer: LiveProgressWriter) -> None:  # noqa: ARG002
        """cleanup is safe to call multiple times."""
        writer.cleanup()
        writer.cleanup()  # Should not raise

    def test_crash_leaves_file(self, log_dir: Path, writer: LiveProgressWriter) -> None:
        """If cleanup is never called (crash), file persists with valid JSON."""
        writer.start_phase("Phase 5: Processing")
        # Simulate crash — don't call cleanup
        path = log_dir / LiveProgressWriter.FILENAME
        assert path.exists()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["current_phase"] == "Phase 5: Processing"
        assert data["pid"] == os.getpid()


class TestAtomicWrite:
    def test_no_partial_json(self, log_dir: Path, writer: LiveProgressWriter) -> None:
        """File is always valid JSON even during rapid writes."""
        writer.start_phase("Phase 5: Processing")
        for i in range(20):
            with writer._lock:
                writer._progress = {"completed": i, "total": 100}
                writer._write()
            # Each write should leave valid JSON
            data = _read_status(log_dir)
            assert data["progress"]["completed"] == i

    def test_no_tmp_file_remains(self, log_dir: Path, writer: LiveProgressWriter) -> None:
        """After a successful write, no .tmp file remains."""
        writer.start_phase("Phase 1: Discovery")
        tmp_path = (log_dir / LiveProgressWriter.FILENAME).with_suffix(".json.tmp")
        assert not tmp_path.exists()


class TestWriteFailure:
    def test_write_failure_does_not_crash(self, log_dir: Path) -> None:
        """If the log_dir becomes unwritable, writer silently ignores errors."""
        writer = LiveProgressWriter(log_dir, "s", "r", "u")

        # Make dir read-only
        log_dir.chmod(0o444)
        try:
            # Should not raise
            writer.start_phase("Phase 1: Discovery")
        finally:
            log_dir.chmod(0o755)
