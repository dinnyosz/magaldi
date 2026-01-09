"""End-to-end tests for the Magaldi CLI.

These tests verify the full pipeline from CLI invocation to output.
Tests marked with @pytest.mark.integration require external services (ES, Ollama).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from shared.cli import main


# =============================================================================
# FIXTURES
# =============================================================================

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
VALID_REPO = FIXTURES_DIR / "repos" / "valid_repo"


@pytest.fixture
def cli_runner() -> CliRunner:
    """Create a CLI runner for testing."""
    return CliRunner()


# =============================================================================
# DRY-RUN TESTS (No external services required)
# =============================================================================


class TestDryRunParsing:
    """Tests for --dry-run mode which uses in-memory storage.

    Note: We use --user main because non-main users require the main branch
    to exist first (for diffing). In dry-run mode with in-memory storage,
    there's no persisted main branch, so we test as main user.
    """

    def test_parse_valid_repo_dry_run(self, cli_runner: CliRunner):
        """Test parsing a valid repository in dry-run mode."""
        result = cli_runner.invoke(
            main, ["parse", str(VALID_REPO), "--user", "main", "--dry-run", "--skip-ai"]
        )

        assert result.exit_code == 0, f"CLI failed with output: {result.output}"
        assert "Phase 1:" in result.output
        assert "Discovery" in result.output
        assert "Phase 2:" in result.output
        assert "Change Detection" in result.output
        assert "Phase 3:" in result.output
        assert "Parsing" in result.output
        assert "Phase 4:" in result.output
        assert "Processing" in result.output

    def test_parse_shows_scope_and_repo(self, cli_runner: CliRunner):
        """Test that scope and repository are displayed."""
        result = cli_runner.invoke(
            main, ["parse", str(VALID_REPO), "--user", "main", "--dry-run", "--skip-ai"]
        )

        assert result.exit_code == 0, f"CLI failed with output: {result.output}"
        assert "test-scope" in result.output
        assert "valid-repo" in result.output

    def test_parse_shows_file_count(self, cli_runner: CliRunner):
        """Test that file count is displayed correctly."""
        result = cli_runner.invoke(
            main, ["parse", str(VALID_REPO), "--user", "main", "--dry-run", "--skip-ai"]
        )

        assert result.exit_code == 0, f"CLI failed with output: {result.output}"
        # valid_repo has 3 files: main.py, utils.py, helper.js
        assert "3 files" in result.output or "3 new" in result.output

    def test_parse_shows_element_types(self, cli_runner: CliRunner):
        """Test that element types are shown in the output."""
        result = cli_runner.invoke(
            main, ["parse", str(VALID_REPO), "--user", "main", "--dry-run", "--skip-ai"]
        )

        assert result.exit_code == 0, f"CLI failed with output: {result.output}"
        # Should show file, class, function counts
        assert "file" in result.output.lower()
        assert "function" in result.output.lower()

    def test_parse_dry_run_shows_warning(self, cli_runner: CliRunner):
        """Test that dry-run mode shows a warning."""
        result = cli_runner.invoke(
            main, ["parse", str(VALID_REPO), "--user", "main", "--dry-run", "--skip-ai"]
        )

        assert result.exit_code == 0, f"CLI failed with output: {result.output}"
        assert "Dry run mode" in result.output
        assert "in-memory storage" in result.output.lower()

    def test_parse_with_custom_workers(self, cli_runner: CliRunner):
        """Test parsing with custom worker count."""
        result = cli_runner.invoke(
            main,
            ["parse", str(VALID_REPO), "--user", "main", "--dry-run", "--skip-ai", "--workers", "2"],
        )

        assert result.exit_code == 0, f"CLI failed with output: {result.output}"
        # Should complete successfully with 2 workers
        assert "Phase 4:" in result.output

    def test_parse_nonexistent_repo_fails(self, cli_runner: CliRunner):
        """Test that parsing a nonexistent repo fails."""
        result = cli_runner.invoke(
            main, ["parse", "/nonexistent/path", "--user", "main", "--dry-run"]
        )

        assert result.exit_code != 0

    def test_parse_no_config_repo_fails(self, cli_runner: CliRunner):
        """Test that parsing a repo without magaldi.yaml fails."""
        no_config_repo = FIXTURES_DIR / "repos" / "no_config_repo"
        result = cli_runner.invoke(
            main, ["parse", str(no_config_repo), "--user", "main", "--dry-run"]
        )

        assert result.exit_code != 0
        assert "magaldi.yaml" in result.output.lower() or "error" in result.output.lower()

    def test_parse_requires_user_option(self, cli_runner: CliRunner):
        """Test that --user option is required."""
        result = cli_runner.invoke(main, ["parse", str(VALID_REPO), "--dry-run"])

        assert result.exit_code != 0
        assert "user" in result.output.lower()


class TestSecondRunDetection:
    """Tests for change detection on subsequent runs."""

    def test_second_run_no_changes(self, cli_runner: CliRunner):
        """Test that second run completes successfully.

        In dry-run mode, in-memory state is not persisted between runs,
        so each run processes all files. This tests that the pipeline completes.
        """
        # First run
        result1 = cli_runner.invoke(
            main, ["parse", str(VALID_REPO), "--user", "main", "--dry-run", "--skip-ai"]
        )
        assert result1.exit_code == 0, f"First run failed: {result1.output}"

        # Second run - in dry-run mode, in-memory state is not persisted
        # so it will parse again. This tests that the pipeline completes.
        result2 = cli_runner.invoke(
            main, ["parse", str(VALID_REPO), "--user", "main", "--dry-run", "--skip-ai"]
        )
        assert result2.exit_code == 0, f"Second run failed: {result2.output}"


# =============================================================================
# INTEGRATION TESTS (Require external services)
# =============================================================================


@pytest.mark.integration
class TestForceClean:
    """Tests for --force-clean option. Requires Elasticsearch.

    Note: We use --user main because user branches require main to exist first.
    """

    def test_force_clean_deletes_existing_data(self, cli_runner: CliRunner):
        """Test that --force-clean deletes existing indexed data."""
        from shared.config import load_config, reset_config
        from shared.db.elasticsearch import ElasticsearchRepository

        # First run: index some data as main
        result1 = cli_runner.invoke(
            main, ["parse", str(VALID_REPO), "--user", "main", "--skip-ai"]
        )
        assert result1.exit_code == 0, f"First run failed: {result1.output}"

        # Second run with force-clean: should delete and re-index
        result2 = cli_runner.invoke(
            main, ["parse", str(VALID_REPO), "--user", "main", "--skip-ai", "--force-clean"]
        )

        assert result2.exit_code == 0, f"Second run failed: {result2.output}"
        assert "Force clean" in result2.output
        assert "Deleted" in result2.output

        # Cleanup
        reset_config()
        config = load_config()
        es_repo = ElasticsearchRepository(config)
        es_repo.delete_by_repository("test-scope", "valid-repo", "main")
        es_repo.close()

    def test_force_clean_without_existing_data(self, cli_runner: CliRunner):
        """Test --force-clean when no data exists (fresh start)."""
        from shared.config import load_config, reset_config
        from shared.db.elasticsearch import ElasticsearchRepository

        # Clean up any existing data first
        reset_config()
        config = load_config()
        es_repo = ElasticsearchRepository(config)
        es_repo.delete_by_repository("test-scope", "valid-repo", "main")
        es_repo.close()
        reset_config()

        result = cli_runner.invoke(
            main,
            ["parse", str(VALID_REPO), "--user", "main", "--skip-ai", "--force-clean"],
        )

        assert result.exit_code == 0, f"Run failed: {result.output}"
        assert "Force clean" in result.output
        # Should show 0 deleted (no existing data)
        assert "Deleted 0" in result.output

        # Cleanup
        reset_config()
        config = load_config()
        es_repo = ElasticsearchRepository(config)
        es_repo.delete_by_repository("test-scope", "valid-repo", "main")
        es_repo.close()

    def test_force_clean_ignored_in_dry_run(self, cli_runner: CliRunner):
        """Test that --force-clean is ignored in dry-run mode."""
        result = cli_runner.invoke(
            main,
            ["parse", str(VALID_REPO), "--user", "main", "--dry-run", "--skip-ai", "--force-clean"],
        )

        assert result.exit_code == 0, f"Run failed: {result.output}"
        # Force clean message should not appear in dry-run
        assert "Force clean" not in result.output


@pytest.mark.integration
class TestElasticsearchIndexing:
    """Tests for Elasticsearch indexing. Requires Elasticsearch."""

    def test_elements_indexed_to_elasticsearch(self, cli_runner: CliRunner):
        """Test that elements are indexed to Elasticsearch."""
        from shared.config import load_config, reset_config
        from shared.db.elasticsearch import ElasticsearchRepository

        # Run parsing as main user with --force-clean to ensure fresh indexing
        result = cli_runner.invoke(
            main, ["parse", str(VALID_REPO), "--user", "main", "--skip-ai", "--force-clean"]
        )
        assert result.exit_code == 0, f"Parsing failed: {result.output}"

        # Verify elements in ES
        reset_config()
        config = load_config()
        es_repo = ElasticsearchRepository(config)

        try:
            # Refresh the index to make sure all documents are searchable
            from shared.db.elasticsearch import INDEX_NAME
            es_repo._get_client().indices.refresh(index=INDEX_NAME)

            # Search for indexed elements using a broad query
            results = es_repo.search_by_text(
                query="hello OR add OR format",  # Matches function names in fixtures
                scope="test-scope",
                repository="valid-repo",
                username="main",
                size=100,
            )

            # Should have elements indexed
            assert len(results) > 0, f"Expected elements to be indexed. Search returned: {results}"

            # Check for expected element types
            element_types = {r["element_type"] for r in results}
            assert "function" in element_types, f"Expected function elements, got types: {element_types}"

        finally:
            # Cleanup
            es_repo.delete_by_repository("test-scope", "valid-repo", "main")
            es_repo.close()


@pytest.mark.integration
class TestParallelProcessing:
    """Tests for parallel worker processing. Requires Elasticsearch."""

    def test_workers_process_in_parallel(self, cli_runner: CliRunner):
        """Test that multiple workers process elements."""
        from shared.config import load_config, reset_config
        from shared.db.elasticsearch import ElasticsearchRepository

        # Run with 4 workers as main user
        result = cli_runner.invoke(
            main,
            ["parse", str(VALID_REPO), "--user", "main", "--skip-ai", "--workers", "4"],
        )

        assert result.exit_code == 0, f"Run failed: {result.output}"
        # Should complete successfully
        assert "Processing" in result.output

        # Cleanup
        reset_config()
        config = load_config()
        es_repo = ElasticsearchRepository(config)
        es_repo.delete_by_repository("test-scope", "valid-repo", "main")
        es_repo.close()

    def test_single_worker_mode(self, cli_runner: CliRunner):
        """Test processing with a single worker."""
        from shared.config import load_config, reset_config
        from shared.db.elasticsearch import ElasticsearchRepository

        result = cli_runner.invoke(
            main,
            ["parse", str(VALID_REPO), "--user", "main", "--skip-ai", "--workers", "1"],
        )

        assert result.exit_code == 0, f"Run failed: {result.output}"

        # Cleanup
        reset_config()
        config = load_config()
        es_repo = ElasticsearchRepository(config)
        es_repo.delete_by_repository("test-scope", "valid-repo", "main")
        es_repo.close()


# =============================================================================
# OUTPUT FORMAT TESTS
# =============================================================================


class TestOutputFormat:
    """Tests for CLI output formatting."""

    def test_summary_shows_totals(self, cli_runner: CliRunner):
        """Test that the summary shows total counts."""
        result = cli_runner.invoke(
            main, ["parse", str(VALID_REPO), "--user", "main", "--dry-run", "--skip-ai"]
        )

        assert result.exit_code == 0, f"CLI failed with output: {result.output}"
        # Summary should be in output
        assert "indexed" in result.output.lower() or "processed" in result.output.lower()

    def test_output_shows_phase_headers(self, cli_runner: CliRunner):
        """Test that output shows all phase headers."""
        result = cli_runner.invoke(
            main, ["parse", str(VALID_REPO), "--user", "main", "--dry-run", "--skip-ai"]
        )

        assert result.exit_code == 0, f"CLI failed with output: {result.output}"
        # All phase headers should be present
        assert "Phase 1" in result.output
        assert "Phase 2" in result.output
        assert "Phase 3" in result.output
        assert "Phase 4" in result.output
