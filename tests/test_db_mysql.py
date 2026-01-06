"""Integration tests for MySQL repositories.

These tests require a running MySQL instance (via Docker).
Run with: pytest -m integration tests/test_db_mysql.py
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta

import pytest

from magaldi.config import load_config
from magaldi.parser.code_parser import CodeElement


# Skip all tests if MySQL is not available
pytestmark = pytest.mark.integration


def mysql_available() -> bool:
    """Check if MySQL is available."""
    try:
        from magaldi.db.mysql import MySQLRepository
        config = load_config(skip_validation=True)
        # Override password from env if not set
        if not config.mysql.password:
            config.mysql.password = os.environ.get("MAGALDI_MYSQL_PASSWORD", "")
        repo = MySQLRepository(config)
        conn = repo._get_connection()
        conn.ping()
        repo.close()
        return True
    except Exception:
        return False


@pytest.fixture(scope="function")
def config():
    """Load config with MySQL password from environment."""
    from magaldi.config import reset_config, MagaldiConfig, MySQLConfig

    # Reset config singleton
    reset_config()

    # Create config directly with known good values for integration tests
    # Don't rely on env vars or config files which may be polluted by other tests
    return MagaldiConfig(
        mysql=MySQLConfig(
            host=os.environ.get("MAGALDI_MYSQL_HOST", "localhost"),
            port=int(os.environ.get("MAGALDI_MYSQL_PORT", "3306")),
            database=os.environ.get("MAGALDI_MYSQL_DATABASE", "magaldi"),
            user=os.environ.get("MAGALDI_MYSQL_USER", "magaldi"),
            password=os.environ.get("MAGALDI_MYSQL_PASSWORD", "magaldipassword"),
        ),
    )


@pytest.fixture
def cleanup_test_data(config):
    """Clean up test data before and after each test."""
    from magaldi.db.mysql import MySQLRepository

    repo = MySQLRepository(config)

    def cleanup():
        with repo._cursor() as cursor:
            # Delete test data (scope starts with 'test-')
            cursor.execute(
                "DELETE FROM code_elements WHERE scope LIKE 'test-%'"
            )
            cursor.execute(
                "DELETE FROM file_states WHERE scope LIKE 'test-%'"
            )
            cursor.execute(
                "DELETE FROM summarization_jobs WHERE element_id LIKE 'test-%'"
            )
            cursor.execute(
                "DELETE FROM embedding_jobs WHERE element_id LIKE 'test-%'"
            )

    cleanup()
    yield
    cleanup()
    repo.close()


@pytest.fixture
def sample_element() -> CodeElement:
    """Create a sample code element for testing."""
    return CodeElement(
        element_id="test-scope:test-repo:main:src/main.py:function:hello:10",
        scope="test-scope",
        repository="test-repo",
        username="main",
        relative_path="src/main.py",
        element_type="function",
        name="hello",
        language="python",
        line_start=10,
        line_end=15,
        raw_code="def hello():\n    return 'Hello, World!'",
        signature="def hello()",
        docstring="Say hello.",
        decorators=[],
        is_async=False,
        visibility="public",
        level=2,
        parent_id=None,
    )


# =============================================================================
# FILE STATE REPOSITORY TESTS
# =============================================================================


class TestMySQLFileStateRepository:
    """Tests for MySQLFileStateRepository."""

    def test_get_empty_file_states(self, config, cleanup_test_data):
        """Test getting file states when none exist."""
        from magaldi.db.mysql import MySQLFileStateRepository

        repo = MySQLFileStateRepository(config)
        states = repo.get_file_states("test-scope", "test-repo", "main")

        assert states == {}
        repo.close()

    def test_main_branch_exists_false(self, config, cleanup_test_data):
        """Test main_branch_exists when no data."""
        from magaldi.db.mysql import MySQLFileStateRepository

        repo = MySQLFileStateRepository(config)
        exists = repo.main_branch_exists("test-scope", "test-repo")

        assert exists is False
        repo.close()


# =============================================================================
# DATABASE REPOSITORY TESTS
# =============================================================================


class TestMySQLDatabaseRepository:
    """Tests for MySQLDatabaseRepository."""

    def test_store_and_get_file_state(self, config, cleanup_test_data):
        """Test storing and retrieving file state."""
        from magaldi.db.mysql import MySQLDatabaseRepository

        repo = MySQLDatabaseRepository(config)

        repo.store_file_state(
            scope="test-scope",
            repository="test-repo",
            username="main",
            relative_path="src/main.py",
            file_hash="abc123",
            language="python",
            file_size=100,
            line_count=10,
        )

        state = repo.get_file_state("test-scope", "test-repo", "main", "src/main.py")

        assert state is not None
        assert state["file_hash"] == "abc123"
        assert state["language"] == "python"
        assert state["file_size"] == 100
        repo.close()

    def test_store_file_state_with_expiry(self, config, cleanup_test_data):
        """Test storing file state with expiry for user branches."""
        from magaldi.db.mysql import MySQLDatabaseRepository

        repo = MySQLDatabaseRepository(config)
        expires = datetime.now() + timedelta(days=30)

        repo.store_file_state(
            scope="test-scope",
            repository="test-repo",
            username="alice",
            relative_path="src/feature.py",
            file_hash="def456",
            language="python",
            file_size=200,
            line_count=20,
            expires_at=expires,
        )

        state = repo.get_file_state("test-scope", "test-repo", "alice", "src/feature.py")

        assert state is not None
        assert state["expires_at"] is not None
        repo.close()

    def test_store_and_get_element(self, config, cleanup_test_data, sample_element):
        """Test storing and retrieving a code element."""
        from magaldi.db.mysql import MySQLDatabaseRepository

        repo = MySQLDatabaseRepository(config)

        repo.store_element(sample_element)
        retrieved = repo.get_element(sample_element.element_id)

        assert retrieved is not None
        assert retrieved.element_id == sample_element.element_id
        assert retrieved.name == "hello"
        assert retrieved.element_type == "function"
        assert retrieved.raw_code == sample_element.raw_code
        repo.close()

    def test_get_nonexistent_element(self, config, cleanup_test_data):
        """Test getting an element that doesn't exist."""
        from magaldi.db.mysql import MySQLDatabaseRepository

        repo = MySQLDatabaseRepository(config)
        retrieved = repo.get_element("nonexistent-element-id")

        assert retrieved is None
        repo.close()

    def test_delete_elements_by_file(self, config, cleanup_test_data, sample_element):
        """Test deleting elements by file path."""
        from magaldi.db.mysql import MySQLDatabaseRepository

        repo = MySQLDatabaseRepository(config)

        repo.store_element(sample_element)
        count = repo.delete_elements_by_file(
            "test-scope", "test-repo", "main", "src/main.py"
        )

        assert count == 1
        assert repo.get_element(sample_element.element_id) is None
        repo.close()

    def test_store_and_get_summarization_job(self, config, cleanup_test_data, sample_element):
        """Test storing and retrieving summarization jobs."""
        from magaldi.db.mysql import MySQLDatabaseRepository

        repo = MySQLDatabaseRepository(config)

        # Must store element first due to foreign key
        repo.store_element(sample_element)

        repo.store_summarization_job(
            element_id=sample_element.element_id,
            level=2,
            parent_id=None,
            dependencies_met=True,
            priority=98,
        )

        job = repo.get_summarization_job(sample_element.element_id)

        assert job is not None
        assert job["level"] == 2
        assert job["dependencies_met"] == 1  # MySQL returns 1 for True
        assert job["status"] == "pending"
        repo.close()

    def test_store_and_get_embedding_job(self, config, cleanup_test_data, sample_element):
        """Test storing and retrieving embedding jobs."""
        from magaldi.db.mysql import MySQLDatabaseRepository

        repo = MySQLDatabaseRepository(config)

        # Must store element first due to foreign key
        repo.store_element(sample_element)

        repo.store_embedding_job(sample_element.element_id)

        job = repo.get_embedding_job(sample_element.element_id)

        assert job is not None
        assert job["status"] == "pending"
        repo.close()


# =============================================================================
# SUMMARIZATION JOB REPOSITORY TESTS
# =============================================================================


class TestMySQLSummarizationJobRepository:
    """Tests for MySQLSummarizationJobRepository."""

    def test_claim_pending_jobs(self, config, cleanup_test_data, sample_element):
        """Test claiming pending jobs."""
        from magaldi.db.mysql import MySQLDatabaseRepository, MySQLSummarizationJobRepository

        db_repo = MySQLDatabaseRepository(config)
        job_repo = MySQLSummarizationJobRepository(config)

        # Store element and job
        db_repo.store_element(sample_element)
        job_repo.add_job(
            element_id=sample_element.element_id,
            level=2,
            parent_id=None,
            dependencies_met=True,
        )

        # Claim jobs
        jobs = job_repo.claim_pending_jobs("test-worker", batch_size=10)

        assert len(jobs) == 1
        assert jobs[0]["element_id"] == sample_element.element_id
        assert jobs[0]["status"] == "running"

        db_repo.close()
        job_repo.close()

    def test_mark_completed(self, config, cleanup_test_data, sample_element):
        """Test marking a job as completed."""
        from magaldi.db.mysql import MySQLDatabaseRepository, MySQLSummarizationJobRepository

        db_repo = MySQLDatabaseRepository(config)
        job_repo = MySQLSummarizationJobRepository(config)

        db_repo.store_element(sample_element)
        job_repo.add_job(
            element_id=sample_element.element_id,
            level=2,
            parent_id=None,
            dependencies_met=True,
        )

        job_repo.mark_completed(sample_element.element_id)

        job = job_repo.get_job(sample_element.element_id)
        assert job["status"] == "completed"

        db_repo.close()
        job_repo.close()

    def test_mark_failed(self, config, cleanup_test_data, sample_element):
        """Test marking a job as failed."""
        from magaldi.db.mysql import MySQLDatabaseRepository, MySQLSummarizationJobRepository

        db_repo = MySQLDatabaseRepository(config)
        job_repo = MySQLSummarizationJobRepository(config)

        db_repo.store_element(sample_element)
        job_repo.add_job(
            element_id=sample_element.element_id,
            level=2,
            parent_id=None,
            dependencies_met=True,
        )

        job_repo.mark_failed(sample_element.element_id, "Test error message")

        job = job_repo.get_job(sample_element.element_id)
        assert job["status"] == "failed"
        assert job["error_message"] == "Test error message"

        db_repo.close()
        job_repo.close()

    def test_unlock_dependencies(self, config, cleanup_test_data):
        """Test unlocking dependent jobs."""
        from magaldi.db.mysql import MySQLDatabaseRepository, MySQLSummarizationJobRepository

        db_repo = MySQLDatabaseRepository(config)
        job_repo = MySQLSummarizationJobRepository(config)

        # Create parent element
        parent = CodeElement(
            element_id="test-scope:test-repo:main:src/main.py:class:MyClass:1",
            scope="test-scope",
            repository="test-repo",
            username="main",
            relative_path="src/main.py",
            element_type="class",
            name="MyClass",
            language="python",
            line_start=1,
            line_end=20,
            level=1,
        )
        db_repo.store_element(parent)
        job_repo.add_job(parent.element_id, level=1, parent_id=None, dependencies_met=True)

        # Create child element
        child = CodeElement(
            element_id="test-scope:test-repo:main:src/main.py:method:my_method:5",
            scope="test-scope",
            repository="test-repo",
            username="main",
            relative_path="src/main.py",
            element_type="method",
            name="my_method",
            language="python",
            line_start=5,
            line_end=10,
            level=2,
            parent_id=parent.element_id,
        )
        db_repo.store_element(child)
        job_repo.add_job(child.element_id, level=2, parent_id=parent.element_id, dependencies_met=False)

        # Initially child should have dependencies_met=False
        child_job = job_repo.get_job(child.element_id)
        assert child_job["dependencies_met"] == 0

        # Unlock dependencies
        count = job_repo.unlock_dependencies(parent.element_id)
        assert count == 1

        # Now child should have dependencies_met=True
        child_job = job_repo.get_job(child.element_id)
        assert child_job["dependencies_met"] == 1

        db_repo.close()
        job_repo.close()


# =============================================================================
# SUMMARY STORE TESTS
# =============================================================================


class TestMySQLSummaryStore:
    """Tests for MySQLSummaryStore."""

    def test_store_and_get_summary(self, config, cleanup_test_data, sample_element):
        """Test storing and retrieving summaries."""
        from magaldi.db.mysql import MySQLDatabaseRepository, MySQLSummaryStore

        db_repo = MySQLDatabaseRepository(config)
        summary_store = MySQLSummaryStore(config)

        db_repo.store_element(sample_element)

        summary_store.store_summary(sample_element.element_id, "This function says hello.")

        summary = summary_store.get_summary(sample_element.element_id)
        assert summary == "This function says hello."

        db_repo.close()
        summary_store.close()

    def test_get_parent_summaries(self, config, cleanup_test_data):
        """Test getting parent summaries for context."""
        from magaldi.db.mysql import MySQLDatabaseRepository, MySQLSummaryStore

        db_repo = MySQLDatabaseRepository(config)
        summary_store = MySQLSummaryStore(config)

        # Create file element
        file_elem = CodeElement(
            element_id="test-scope:test-repo:main:src/main.py:file:main.py:1",
            scope="test-scope",
            repository="test-repo",
            username="main",
            relative_path="src/main.py",
            element_type="file",
            name="main.py",
            language="python",
            line_start=1,
            line_end=50,
            level=0,
        )
        db_repo.store_element(file_elem)
        summary_store.store_summary(file_elem.element_id, "Main application file.")

        # Create class element
        class_elem = CodeElement(
            element_id="test-scope:test-repo:main:src/main.py:class:MyClass:10",
            scope="test-scope",
            repository="test-repo",
            username="main",
            relative_path="src/main.py",
            element_type="class",
            name="MyClass",
            language="python",
            line_start=10,
            line_end=40,
            level=1,
            parent_id=file_elem.element_id,
        )
        db_repo.store_element(class_elem)
        summary_store.store_summary(class_elem.element_id, "A sample class.")

        # Create method element
        method_elem = CodeElement(
            element_id="test-scope:test-repo:main:src/main.py:method:do_stuff:20",
            scope="test-scope",
            repository="test-repo",
            username="main",
            relative_path="src/main.py",
            element_type="method",
            name="do_stuff",
            language="python",
            line_start=20,
            line_end=30,
            level=2,
            parent_id=class_elem.element_id,
        )
        db_repo.store_element(method_elem)

        # Get parent summaries for method
        summaries = summary_store.get_parent_summaries(method_elem)

        assert "file" in summaries
        assert summaries["file"] == "Main application file."
        assert "class" in summaries
        assert summaries["class"] == "A sample class."

        db_repo.close()
        summary_store.close()


# =============================================================================
# EMBEDDING JOB REPOSITORY TESTS
# =============================================================================


class TestMySQLEmbeddingJobRepository:
    """Tests for MySQLEmbeddingJobRepository."""

    def test_claim_and_complete_jobs(self, config, cleanup_test_data, sample_element):
        """Test the full embedding job lifecycle."""
        from magaldi.db.mysql import MySQLDatabaseRepository, MySQLEmbeddingJobRepository

        db_repo = MySQLDatabaseRepository(config)
        job_repo = MySQLEmbeddingJobRepository(config)

        db_repo.store_element(sample_element)
        job_repo.add_job(sample_element.element_id)

        # Claim
        jobs = job_repo.claim_pending_jobs("test-worker", batch_size=10)
        assert len(jobs) == 1

        # Complete
        job_repo.mark_completed(sample_element.element_id)

        job = job_repo.get_job(sample_element.element_id)
        assert job["status"] == "completed"

        db_repo.close()
        job_repo.close()
