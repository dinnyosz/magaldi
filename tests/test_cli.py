"""Tests for the CLI module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from shared.cli import (
    format_duration,
    check_ollama_models,
    main,
    print_discovery_result,
    print_change_manifest,
    print_parsing_result,
    print_feature_result,
    print_processing_result,
    print_summary,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def cli_runner():
    """Create a CLI runner for testing."""
    return CliRunner()


@pytest.fixture
def mock_config():
    """Create a mock configuration."""
    config = MagicMock()
    config.llm.url = "http://localhost:11434"
    config.llm.summarize_model = "qwen2.5-coder:1.5b"
    config.llm.summarize_model_small = "qwen2.5-coder:0.5b"
    config.llm.embed_model = "nomic-embed-text"
    config.llm.provider = "ollama"
    config.llm.api_key = None
    config.web.host = "127.0.0.1"
    config.web.port = 8080
    return config


@pytest.fixture
def mock_discovery_result():
    """Create a mock discovery result."""
    result = MagicMock()
    result.scope = "test-scope"
    result.repository = "test-repo"
    result.username = "testuser"
    result.total_files = 10
    result.total_lines = 1000
    result.languages = {
        "python": MagicMock(files=5),
        "javascript": MagicMock(files=5),
    }
    return result


@pytest.fixture
def mock_change_manifest():
    """Create a mock change manifest."""
    manifest = MagicMock()
    manifest.scope = "test-scope"
    manifest.repository = "test-repo"
    manifest.username = "testuser"
    manifest.total_files_scanned = 10
    manifest.new_files = [MagicMock(relative_path="a.py", hash="abc123")]
    manifest.modified_files = [MagicMock(relative_path="b.py", hash="def456")]
    manifest.deleted_files = []
    manifest.unchanged_count = 8
    manifest.skipped_count = 0
    manifest.files_to_parse = 2
    return manifest


@pytest.fixture
def mock_parsing_result():
    """Create a mock parsing result."""
    result = MagicMock()
    result.parsed_files = [MagicMock(), MagicMock()]
    result.total_elements = 20
    result.elements_by_type = {"function": 15, "class": 5}
    result.failed_files = []
    return result


# =============================================================================
# FORMAT DURATION TESTS
# =============================================================================


class TestFormatDuration:
    """Tests for format_duration function."""

    def test_format_seconds_only(self):
        """Test formatting duration with only seconds."""
        assert format_duration(30) == "0:30"
        assert format_duration(0) == "0:00"
        assert format_duration(59) == "0:59"

    def test_format_minutes_and_seconds(self):
        """Test formatting duration with minutes and seconds."""
        assert format_duration(60) == "1:00"
        assert format_duration(90) == "1:30"
        assert format_duration(3599) == "59:59"

    def test_format_hours(self):
        """Test formatting duration with hours."""
        assert format_duration(3600) == "1:00:00"
        assert format_duration(3661) == "1:01:01"
        assert format_duration(7200) == "2:00:00"

    def test_format_float_truncates(self):
        """Test that float values are truncated to int."""
        assert format_duration(30.9) == "0:30"


# =============================================================================
# CHECK OLLAMA MODELS TESTS
# =============================================================================


class TestCheckOllamaModels:
    """Tests for check_ollama_models function."""

    def test_skip_ai_returns_empty(self, mock_config):
        """Test that skip_ai=True returns no errors."""
        result = check_ollama_models(mock_config, skip_ai=True)
        assert result == []

    @patch("requests.get")
    def test_connection_error(self, mock_get, mock_config):
        """Test handling of connection error."""
        import requests
        mock_get.side_effect = requests.exceptions.ConnectionError()

        result = check_ollama_models(mock_config, skip_ai=False)

        assert len(result) == 1
        assert "Cannot connect to Ollama" in result[0]

    @patch("requests.get")
    def test_all_models_available(self, mock_get, mock_config):
        """Test when all models are available."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "models": [
                {"name": "qwen2.5-coder:1.5b"},
                {"name": "qwen2.5-coder:0.5b"},
                {"name": "nomic-embed-text"},
            ]
        }
        mock_get.return_value = mock_response

        result = check_ollama_models(mock_config, skip_ai=False)

        assert result == []

    @patch("requests.get")
    def test_model_missing(self, mock_get, mock_config):
        """Test when a model is missing."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "models": [
                {"name": "qwen2.5-coder:1.5b"},
                {"name": "nomic-embed-text"},
            ]
        }
        mock_get.return_value = mock_response

        result = check_ollama_models(mock_config, skip_ai=False)

        assert len(result) == 1
        assert "qwen2.5-coder:0.5b" in result[0]

    @patch("requests.get")
    def test_model_with_latest_tag(self, mock_get, mock_config):
        """Test that model with :latest suffix is detected."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "models": [
                # When configured model is "qwen2.5-coder:1.5b", and available is
                # "qwen2.5-coder:1.5b:latest", it will match via f"{model}:latest" check
                {"name": "qwen2.5-coder:1.5b:latest"},
                {"name": "qwen2.5-coder:0.5b:latest"},
                {"name": "nomic-embed-text:latest"},
            ]
        }
        mock_get.return_value = mock_response

        result = check_ollama_models(mock_config, skip_ai=False)

        # Models match with :latest suffix appended
        assert len(result) == 0

    @patch("requests.get")
    def test_generic_exception(self, mock_get, mock_config):
        """Test handling of generic exception."""
        mock_get.side_effect = Exception("Unexpected error")

        result = check_ollama_models(mock_config, skip_ai=False)

        assert len(result) == 1
        assert "Error connecting to Ollama" in result[0]


# =============================================================================
# CLI COMMAND TESTS
# =============================================================================


class TestMainCli:
    """Tests for main CLI group."""

    def test_main_help(self, cli_runner):
        """Test main --help output."""
        result = cli_runner.invoke(main, ["--help"])

        assert result.exit_code == 0
        assert "Magaldi" in result.output
        assert "parse" in result.output or "COMMAND" in result.output

    def test_main_version(self, cli_runner):
        """Test main --version output."""
        result = cli_runner.invoke(main, ["--version"])

        assert result.exit_code == 0
        assert "0.1.0" in result.output or "magaldi" in result.output.lower()


class TestParseCommand:
    """Tests for parse command."""

    def test_parse_help(self, cli_runner):
        """Test parse --help output."""
        result = cli_runner.invoke(main, ["parse", "--help"])

        assert result.exit_code == 0
        assert "--user" in result.output
        assert "--dry-run" in result.output

    @patch("shared.cli.load_config")
    @patch("shared.cli.run_discovery")
    @patch("shared.cli.run_change_detection")
    @patch("shared.cli.run_parsing")
    @patch("shared.cli.run_processing")
    def test_parse_dry_run(
        self,
        mock_run_processing,
        mock_run_parsing,
        mock_run_change_detection,
        mock_run_discovery,
        mock_load_config,
        cli_runner,
        mock_config,
        mock_discovery_result,
        mock_change_manifest,
        mock_parsing_result,
        tmp_path,
    ):
        """Test parse command in dry-run mode."""
        # Create a temp directory as repo
        repo_path = tmp_path / "test-repo"
        repo_path.mkdir()
        (repo_path / "magaldi.yaml").write_text("scope: test")

        mock_load_config.return_value = mock_config
        mock_run_discovery.return_value = mock_discovery_result
        mock_run_change_detection.return_value = mock_change_manifest
        mock_run_parsing.return_value = mock_parsing_result
        mock_run_processing.return_value = (10, 0, 10, 0.5, 0.3, 0.2, 5.0, None, [])

        result = cli_runner.invoke(main, [
            "parse",
            str(repo_path),
            "--user", "testuser",
            "--dry-run",
        ])

        assert result.exit_code == 0
        assert "Dry run" in result.output or mock_run_processing.called

    def test_parse_missing_user(self, cli_runner, tmp_path):
        """Test parse command fails without --user."""
        repo_path = tmp_path / "test-repo"
        repo_path.mkdir()

        result = cli_runner.invoke(main, [
            "parse",
            str(repo_path),
        ])

        assert result.exit_code != 0
        assert "Missing option" in result.output or "user" in result.output.lower()

    def test_parse_nonexistent_path(self, cli_runner):
        """Test parse command fails with nonexistent path."""
        result = cli_runner.invoke(main, [
            "parse",
            "/nonexistent/path",
            "--user", "testuser",
        ])

        assert result.exit_code != 0


class TestExtractFeaturesCommand:
    """Tests for extract-features command."""

    def test_extract_features_help(self, cli_runner):
        """Test extract-features --help output."""
        result = cli_runner.invoke(main, ["extract-features", "--help"])

        assert result.exit_code == 0
        assert "--user" in result.output
        assert "--min-cluster-size" in result.output

    def test_extract_features_missing_user(self, cli_runner, tmp_path):
        """Test extract-features fails without --user."""
        repo_path = tmp_path / "test-repo"
        repo_path.mkdir()

        result = cli_runner.invoke(main, [
            "extract-features",
            str(repo_path),
        ])

        assert result.exit_code != 0

    @patch("shared.cli.load_config")
    @patch("shared.cli.run_feature_extraction")
    def test_extract_features_no_config(
        self,
        mock_run_feature_extraction,
        mock_load_config,
        cli_runner,
        mock_config,
        tmp_path,
    ):
        """Test extract-features without magaldi.yaml."""
        repo_path = tmp_path / "test-repo"
        repo_path.mkdir()
        # Don't create magaldi.yaml

        mock_load_config.return_value = mock_config

        result = cli_runner.invoke(main, [
            "extract-features",
            str(repo_path),
            "--user", "testuser",
        ])

        assert result.exit_code != 0
        assert "magaldi.yaml" in result.output


class TestExtractGlossaryCommand:
    """Tests for extract-glossary CLI command."""

    def test_extract_glossary_command_exists(self, cli_runner):
        """Test that extract-glossary command is registered."""
        result = cli_runner.invoke(main, ["extract-glossary", "--help"])

        assert result.exit_code == 0
        assert "Extract glossary" in result.output
        assert "--link-features" in result.output

    def test_extract_glossary_missing_user(self, cli_runner, tmp_path):
        """Test extract-glossary fails without --user."""
        repo_path = tmp_path / "test-repo"
        repo_path.mkdir()

        result = cli_runner.invoke(main, [
            "extract-glossary",
            str(repo_path),
        ])

        assert result.exit_code != 0

    @patch("shared.cli.load_config")
    @patch("shared.cli.run_glossary_extraction")
    def test_extract_glossary_no_config(
        self,
        mock_run_glossary_extraction,
        mock_load_config,
        cli_runner,
        mock_config,
        tmp_path,
    ):
        """Test extract-glossary without magaldi.yaml."""
        repo_path = tmp_path / "test-repo"
        repo_path.mkdir()
        # Don't create magaldi.yaml

        mock_load_config.return_value = mock_config

        result = cli_runner.invoke(main, [
            "extract-glossary",
            str(repo_path),
            "--user", "testuser",
        ])

        assert result.exit_code != 0
        assert "magaldi.yaml" in result.output


class TestWebCommands:
    """Tests for web subcommands."""

    def test_web_help(self, cli_runner):
        """Test web --help output."""
        result = cli_runner.invoke(main, ["web", "--help"])

        assert result.exit_code == 0
        assert "serve" in result.output

    def test_web_serve_help(self, cli_runner):
        """Test web serve --help output."""
        result = cli_runner.invoke(main, ["web", "serve", "--help"])

        assert result.exit_code == 0
        assert "--host" in result.output
        assert "--port" in result.output
        assert "--reload" in result.output


# =============================================================================
# PRINT FUNCTION TESTS
# =============================================================================


class TestPrintFunctions:
    """Tests for print/output formatting functions."""

    def test_print_discovery_result(self, mock_discovery_result, capsys):
        """Test print_discovery_result output."""
        print_discovery_result(mock_discovery_result)

        captured = capsys.readouterr()
        assert "test-scope" in captured.out
        assert "test-repo" in captured.out
        assert "testuser" in captured.out

    def test_print_change_manifest(self, mock_change_manifest, capsys):
        """Test print_change_manifest output."""
        print_change_manifest(mock_change_manifest)

        captured = capsys.readouterr()
        assert "scanned" in captured.out
        # Should include info about new/modified files
        assert "1 new" in captured.out or "new" in captured.out

    def test_print_change_manifest_no_changes(self, capsys):
        """Test print_change_manifest with no changes."""
        manifest = MagicMock()
        manifest.total_files_scanned = 10
        manifest.new_files = []
        manifest.modified_files = []
        manifest.deleted_files = []
        manifest.unchanged_count = 10
        manifest.skipped_count = 0

        print_change_manifest(manifest)

        captured = capsys.readouterr()
        assert "unchanged" in captured.out

    def test_print_parsing_result(self, mock_parsing_result, capsys):
        """Test print_parsing_result output."""
        print_parsing_result(mock_parsing_result)

        captured = capsys.readouterr()
        assert "20" in captured.out  # total_elements
        assert "function" in captured.out
        assert "class" in captured.out

    def test_print_parsing_result_with_failures(self, capsys):
        """Test print_parsing_result with failed files."""
        result = MagicMock()
        result.parsed_files = [MagicMock()]
        result.total_elements = 10
        result.elements_by_type = {"function": 10}
        result.failed_files = ["error.py"]

        print_parsing_result(result)

        captured = capsys.readouterr()
        assert "failed" in captured.out

    def test_print_feature_result_none(self, capsys):
        """Test print_feature_result with None."""
        print_feature_result(None)

        captured = capsys.readouterr()
        assert captured.out == ""

    def test_print_feature_result_with_data(self, capsys):
        """Test print_feature_result with data."""
        result = {
            "cluster_count": 5,
            "outlier_count": 2,
            "total_elements": 100,
            "elements_covered": 80,
            "coverage_pct": 80.0,
            "processed": 5,
            "failed": 0,
            "elapsed": 10.5,
            "subfeatures_created": 3,
            "parent_features_with_subfeatures": 2,
            "clusters": [
                {
                    "cluster_id": 0,
                    "label": "authentication",
                    "size": 20,
                    "sample_names": ["login", "logout", "verify"],
                }
            ],
        }

        print_feature_result(result)

        captured = capsys.readouterr()
        assert "5 features" in captured.out
        assert "80" in captured.out  # coverage
        assert "authentication" in captured.out

    def test_print_processing_result_basic(self, capsys):
        """Test print_processing_result with basic data."""
        print_processing_result(
            processed=10,
            skipped=2,
            indexed=10,
            skip_ai=False,
        )

        captured = capsys.readouterr()
        assert "10" in captured.out
        assert "processed" in captured.out

    def test_print_processing_result_skip_ai(self, capsys):
        """Test print_processing_result with skip_ai."""
        print_processing_result(
            processed=10,
            skipped=0,
            indexed=10,
            skip_ai=True,
        )

        captured = capsys.readouterr()
        assert "skipped" in captured.out.lower()

    def test_print_processing_result_with_timing(self, capsys):
        """Test print_processing_result with timing stats."""
        print_processing_result(
            processed=10,
            skipped=0,
            indexed=10,
            skip_ai=False,
            avg_wall=0.5,
            avg_summ=0.3,
            avg_embed=0.2,
            elapsed=5.0,
        )

        captured = capsys.readouterr()
        assert "elapsed" in captured.out

    def test_print_summary(self, mock_discovery_result, mock_change_manifest, capsys):
        """Test print_summary output."""
        print_summary(
            discovery=mock_discovery_result,
            manifest=mock_change_manifest,
            processed=10,
            indexed=10,
            skip_ai=False,
        )

        captured = capsys.readouterr()
        assert "Complete" in captured.out
        assert "test-scope" in captured.out
        assert "test-repo" in captured.out

    def test_print_summary_skip_ai(self, mock_discovery_result, mock_change_manifest, capsys):
        """Test print_summary with skip_ai."""
        print_summary(
            discovery=mock_discovery_result,
            manifest=mock_change_manifest,
            processed=10,
            indexed=10,
            skip_ai=True,
        )

        captured = capsys.readouterr()
        assert "Skipped" in captured.out


# =============================================================================
# PHASE RUNNER TESTS
# =============================================================================


class TestPhaseRunners:
    """Tests for phase runner functions."""

    @patch("shared.cli.discover")
    def test_run_discovery(self, mock_discover, mock_discovery_result):
        """Test run_discovery function."""
        from shared.cli import run_discovery

        mock_discover.return_value = mock_discovery_result

        result = run_discovery("/path/to/repo", "testuser")

        assert result == mock_discovery_result
        mock_discover.assert_called_once_with("/path/to/repo", "testuser")

    @patch("shared.cli.detect_changes")
    @patch("shared.cli.InMemoryFileStateRepository")
    def test_run_change_detection_dry_run(
        self,
        mock_in_memory_repo,
        mock_detect_changes,
        mock_discovery_result,
        mock_config,
        mock_change_manifest,
    ):
        """Test run_change_detection in dry-run mode."""
        from shared.cli import run_change_detection

        mock_detect_changes.return_value = mock_change_manifest
        mock_in_memory_repo.return_value = MagicMock()

        result = run_change_detection(mock_discovery_result, mock_config, dry_run=True)

        assert result == mock_change_manifest
        mock_in_memory_repo.assert_called_once()

    @patch("shared.cli.parse_files")
    def test_run_parsing(self, mock_parse_files, mock_change_manifest, mock_parsing_result):
        """Test run_parsing function."""
        from shared.cli import run_parsing

        mock_parse_files.return_value = mock_parsing_result

        result = run_parsing(mock_change_manifest)

        assert result == mock_parsing_result

    @patch("shared.cli.process_elements")
    @patch("shared.cli.ElasticsearchRepository", create=True)
    def test_run_processing_dry_run(
        self,
        mock_es_repo_class,
        mock_process_elements,
        mock_parsing_result,
        mock_change_manifest,
        mock_config,
    ):
        """Test run_processing in dry-run mode."""
        from shared.cli import run_processing

        result = run_processing(
            mock_parsing_result,
            mock_change_manifest,
            mock_config,
            dry_run=True,
            skip_ai=True,
            workers=4,
        )

        # Dry run should return zeros
        assert result == (0, 0, 0, 0.0, 0.0, 0.0, 0.0, None, [])


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================


class TestErrorHandling:
    """Tests for error handling in CLI."""

    @patch("shared.cli.load_config")
    @patch("shared.cli.run_discovery")
    def test_parse_discovery_error(
        self,
        mock_run_discovery,
        mock_load_config,
        cli_runner,
        mock_config,
        tmp_path,
    ):
        """Test parse command handles discovery error."""
        from magaldi_core.discovery import DiscoveryError

        repo_path = tmp_path / "test-repo"
        repo_path.mkdir()

        mock_load_config.return_value = mock_config
        mock_run_discovery.side_effect = DiscoveryError("Test error")

        result = cli_runner.invoke(main, [
            "parse",
            str(repo_path),
            "--user", "testuser",
            "--dry-run",
        ])

        assert result.exit_code != 0
        assert "error" in result.output.lower()

    @patch("shared.cli.load_config")
    @patch("shared.cli.run_discovery")
    def test_parse_generic_error(
        self,
        mock_run_discovery,
        mock_load_config,
        cli_runner,
        mock_config,
        tmp_path,
    ):
        """Test parse command handles generic error."""
        repo_path = tmp_path / "test-repo"
        repo_path.mkdir()

        mock_load_config.return_value = mock_config
        mock_run_discovery.side_effect = Exception("Unexpected error")

        result = cli_runner.invoke(main, [
            "parse",
            str(repo_path),
            "--user", "testuser",
            "--dry-run",
        ])

        assert result.exit_code != 0
        assert "Error" in result.output


# =============================================================================
# INTEGRATION-STYLE TESTS
# =============================================================================


class TestCliIntegration:
    """Integration-style tests for CLI."""

    @patch("shared.cli.load_config")
    @patch("shared.cli.run_discovery")
    @patch("shared.cli.run_change_detection")
    def test_parse_no_changes(
        self,
        mock_run_change_detection,
        mock_run_discovery,
        mock_load_config,
        cli_runner,
        mock_config,
        mock_discovery_result,
        tmp_path,
    ):
        """Test parse when no changes detected."""
        repo_path = tmp_path / "test-repo"
        repo_path.mkdir()

        mock_load_config.return_value = mock_config
        mock_run_discovery.return_value = mock_discovery_result

        # Manifest with no files to parse
        manifest = MagicMock()
        manifest.files_to_parse = 0
        mock_run_change_detection.return_value = manifest

        result = cli_runner.invoke(main, [
            "parse",
            str(repo_path),
            "--user", "testuser",
            "--dry-run",
        ])

        assert result.exit_code == 0
        assert "up to date" in result.output

    @patch("shared.cli.load_config")
    @patch("shared.cli.run_discovery")
    @patch("shared.cli.run_change_detection")
    @patch("shared.cli.check_ollama_models")
    @patch("shared.cli.run_parsing")
    @patch("shared.cli.run_processing")
    def test_parse_model_check_fails(
        self,
        mock_run_processing,
        mock_run_parsing,
        mock_check_ollama,
        mock_run_change_detection,
        mock_run_discovery,
        mock_load_config,
        cli_runner,
        mock_config,
        mock_discovery_result,
        mock_change_manifest,
        mock_parsing_result,
        tmp_path,
    ):
        """Test parse when Ollama model check fails."""
        repo_path = tmp_path / "test-repo"
        repo_path.mkdir()

        mock_load_config.return_value = mock_config
        mock_run_discovery.return_value = mock_discovery_result
        mock_run_change_detection.return_value = mock_change_manifest
        mock_run_parsing.return_value = mock_parsing_result
        mock_check_ollama.return_value = ["Model 'test' not found"]

        result = cli_runner.invoke(main, [
            "parse",
            str(repo_path),
            "--user", "testuser",
        ])

        assert result.exit_code != 0
        assert "Pre-flight check failed" in result.output or "not found" in result.output
