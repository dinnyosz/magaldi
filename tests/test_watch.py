"""Tests for the watch command."""

from __future__ import annotations

import queue
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from shared.cli.watch import MagaldiFileHandler, SUPPORTED_EXTENSIONS


class TestMagaldiFileHandler:
    """Tests for MagaldiFileHandler file filtering."""

    @pytest.fixture
    def discovery_result(self, tmp_path: Path) -> MagicMock:
        """Create a mock discovery result."""
        result = MagicMock()
        result.repo_path = tmp_path
        result.exclude_directories = ["node_modules", "__pycache__", ".git", "*.egg-info"]
        result.exclude_files = ["*.min.js", "*.lock"]
        return result

    @pytest.fixture
    def handler(self, discovery_result: MagicMock) -> MagaldiFileHandler:
        """Create a file handler."""
        change_queue: queue.Queue = queue.Queue()
        return MagaldiFileHandler(change_queue, discovery_result)

    def test_should_process_python_file(self, handler: MagaldiFileHandler, tmp_path: Path):
        """Test that Python files are processed."""
        test_file = tmp_path / "test.py"
        assert handler._should_process(str(test_file)) is True

    def test_should_process_javascript_file(self, handler: MagaldiFileHandler, tmp_path: Path):
        """Test that JavaScript files are processed."""
        test_file = tmp_path / "test.js"
        assert handler._should_process(str(test_file)) is True

    def test_should_process_typescript_file(self, handler: MagaldiFileHandler, tmp_path: Path):
        """Test that TypeScript files are processed."""
        test_file = tmp_path / "test.ts"
        assert handler._should_process(str(test_file)) is True

    def test_should_not_process_text_file(self, handler: MagaldiFileHandler, tmp_path: Path):
        """Test that text files are not processed."""
        test_file = tmp_path / "readme.txt"
        assert handler._should_process(str(test_file)) is False

    def test_should_not_process_markdown_file(self, handler: MagaldiFileHandler, tmp_path: Path):
        """Test that markdown files are not processed."""
        test_file = tmp_path / "README.md"
        assert handler._should_process(str(test_file)) is False

    def test_should_not_process_node_modules(self, handler: MagaldiFileHandler, tmp_path: Path):
        """Test that files in node_modules are not processed."""
        test_file = tmp_path / "node_modules" / "lodash" / "index.js"
        assert handler._should_process(str(test_file)) is False

    def test_should_not_process_pycache(self, handler: MagaldiFileHandler, tmp_path: Path):
        """Test that files in __pycache__ are not processed."""
        test_file = tmp_path / "__pycache__" / "module.cpython-311.pyc"
        assert handler._should_process(str(test_file)) is False

    def test_should_not_process_git_directory(self, handler: MagaldiFileHandler, tmp_path: Path):
        """Test that files in .git are not processed."""
        test_file = tmp_path / ".git" / "config"
        assert handler._should_process(str(test_file)) is False

    def test_should_not_process_minified_js(self, handler: MagaldiFileHandler, tmp_path: Path):
        """Test that minified JS files are not processed."""
        test_file = tmp_path / "app.min.js"
        assert handler._should_process(str(test_file)) is False

    def test_should_not_process_lock_files(self, handler: MagaldiFileHandler, tmp_path: Path):
        """Test that lock files are not processed."""
        test_file = tmp_path / "package-lock.json"
        # Note: .json is not in supported extensions anyway, but the exclude pattern should also match
        assert handler._should_process(str(test_file)) is False

    def test_should_process_nested_python_file(self, handler: MagaldiFileHandler, tmp_path: Path):
        """Test that nested Python files are processed."""
        test_file = tmp_path / "src" / "modules" / "utils.py"
        assert handler._should_process(str(test_file)) is True

    def test_should_not_process_outside_repo(self, handler: MagaldiFileHandler, tmp_path: Path):
        """Test that files outside repo are not processed."""
        test_file = Path("/some/other/path/test.py")
        assert handler._should_process(str(test_file)) is False

    def test_on_modified_queues_valid_file(self, handler: MagaldiFileHandler, tmp_path: Path):
        """Test that modified events queue valid files."""
        test_file = tmp_path / "test.py"
        event = MagicMock()
        event.is_directory = False
        event.src_path = str(test_file)

        handler.on_modified(event)

        assert not handler.change_queue.empty()
        event_type, path = handler.change_queue.get()
        assert event_type == "modified"
        assert path == str(test_file)

    def test_on_created_queues_valid_file(self, handler: MagaldiFileHandler, tmp_path: Path):
        """Test that created events queue valid files."""
        test_file = tmp_path / "new_file.py"
        event = MagicMock()
        event.is_directory = False
        event.src_path = str(test_file)

        handler.on_created(event)

        assert not handler.change_queue.empty()
        event_type, path = handler.change_queue.get()
        assert event_type == "created"
        assert path == str(test_file)

    def test_on_deleted_queues_valid_file(self, handler: MagaldiFileHandler, tmp_path: Path):
        """Test that deleted events queue valid files."""
        test_file = tmp_path / "removed.py"
        event = MagicMock()
        event.is_directory = False
        event.src_path = str(test_file)

        handler.on_deleted(event)

        assert not handler.change_queue.empty()
        event_type, path = handler.change_queue.get()
        assert event_type == "deleted"
        assert path == str(test_file)

    def test_ignores_directory_events(self, handler: MagaldiFileHandler, tmp_path: Path):
        """Test that directory events are ignored."""
        test_dir = tmp_path / "new_dir"
        event = MagicMock()
        event.is_directory = True
        event.src_path = str(test_dir)

        handler.on_created(event)
        handler.on_modified(event)
        handler.on_deleted(event)

        assert handler.change_queue.empty()


class TestSupportedExtensions:
    """Test that supported extensions are correct."""

    def test_python_supported(self):
        assert ".py" in SUPPORTED_EXTENSIONS

    def test_javascript_supported(self):
        assert ".js" in SUPPORTED_EXTENSIONS
        assert ".jsx" in SUPPORTED_EXTENSIONS
        assert ".mjs" in SUPPORTED_EXTENSIONS
        assert ".cjs" in SUPPORTED_EXTENSIONS

    def test_typescript_supported(self):
        assert ".ts" in SUPPORTED_EXTENSIONS
        assert ".tsx" in SUPPORTED_EXTENSIONS

    def test_php_supported(self):
        assert ".php" in SUPPORTED_EXTENSIONS

    def test_rust_supported(self):
        assert ".rs" in SUPPORTED_EXTENSIONS


class TestWatchCommand:
    """Tests for the watch CLI command."""

    @pytest.fixture
    def cli_runner(self) -> CliRunner:
        return CliRunner()

    def test_watch_requires_user(self, cli_runner: CliRunner):
        """Test that --user is required."""
        from shared.cli._shared import main

        result = cli_runner.invoke(main, ["watch", "."])
        assert result.exit_code != 0
        assert "Missing option" in result.output or "required" in result.output.lower()

    def test_watch_requires_valid_path(self, cli_runner: CliRunner):
        """Test that path must exist."""
        from shared.cli._shared import main

        result = cli_runner.invoke(main, ["watch", "/nonexistent/path", "--user", "test"])
        assert result.exit_code != 0

    def test_watch_help(self, cli_runner: CliRunner):
        """Test watch --help output."""
        from shared.cli._shared import main

        result = cli_runner.invoke(main, ["watch", "--help"])
        assert result.exit_code == 0
        assert "Watch a repository for changes" in result.output
        assert "--debounce" in result.output
        assert "--skip-ai" in result.output
        assert "--user" in result.output

    @patch("shared.cli.watch.Observer")
    @patch("shared.cli.watch.run_discovery")
    @patch("shared.cli.watch.run_change_detection")
    @patch("shared.cli.watch.load_config")
    def test_watch_starts_observer(
        self,
        mock_load_config: MagicMock,
        mock_change_detection: MagicMock,
        mock_discovery: MagicMock,
        mock_observer_class: MagicMock,
        cli_runner: CliRunner,
        tmp_path: Path,
    ):
        """Test that watch command initializes and starts the observer."""
        from shared.cli._shared import main

        # Setup mocks
        mock_config = MagicMock()
        mock_config.llm.models = {}
        mock_load_config.return_value = mock_config

        mock_discovery_result = MagicMock()
        mock_discovery_result.scope = "test"
        mock_discovery_result.repository = "repo"
        mock_discovery_result.repo_path = tmp_path
        mock_discovery_result.total_files = 10
        mock_discovery_result.exclude_directories = []
        mock_discovery_result.exclude_files = []
        mock_discovery.return_value = mock_discovery_result

        mock_manifest = MagicMock()
        mock_manifest.files_to_parse = 0
        mock_change_detection.return_value = mock_manifest

        # Mock observer to raise KeyboardInterrupt when start is called
        mock_observer = MagicMock()
        mock_observer.start.side_effect = KeyboardInterrupt()
        mock_observer_class.return_value = mock_observer

        # Create a magaldi.yaml in the temp path
        config_file = tmp_path / "magaldi.yaml"
        config_file.write_text("scope: test\nrepository: repo\n")

        result = cli_runner.invoke(main, ["watch", str(tmp_path), "--user", "test"])

        # Verify observer was created and scheduled
        mock_observer_class.assert_called_once()
        mock_observer.schedule.assert_called_once()

        # The command should show the watching message before the interrupt
        assert "Watching for changes" in result.output
