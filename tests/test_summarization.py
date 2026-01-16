"""Tests for the summarization module (Phase 5)."""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from magaldi_core.code_parser import CodeElement
from shared.ai.summarization import (
    InMemoryJobRepository,
    InMemorySummaryStore,
    SummarizationLLMClient,
    SummarizationConfig,
    SummarizationResult,
    build_prompt,
    clean_summary,
    generate_summary,
    process_summarization_job,
    truncate_code,
    update_dependencies_after_completion,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def config() -> SummarizationConfig:
    return SummarizationConfig(
        ollama_url="http://localhost:11434",
        model="qwen2.5-coder:7b",
        temperature=0.3,
        max_tokens=256,
    )


@pytest.fixture
def file_element() -> CodeElement:
    return CodeElement(
        element_id="scope:repo:main:src/app.py:file:app.py:1",
        scope="scope",
        repository="repo",
        username="main",
        relative_path="src/app.py",
        element_type="file",
        name="app.py",
        language="python",
        line_start=1,
        line_end=100,
        raw_code='"""Main application module."""\n\nfrom flask import Flask\n\napp = Flask(__name__)',
        level=0,
    )


@pytest.fixture
def class_element() -> CodeElement:
    return CodeElement(
        element_id="scope:repo:main:src/app.py:class:UserService:10",
        scope="scope",
        repository="repo",
        username="main",
        relative_path="src/app.py",
        element_type="class",
        name="UserService",
        language="python",
        line_start=10,
        line_end=50,
        raw_code="class UserService:\n    def __init__(self): pass",
        docstring="Service for user management.",
        decorators=["dataclass"],
        level=1,
        parent_id="scope:repo:main:src/app.py:file:app.py:1",
    )


@pytest.fixture
def method_element() -> CodeElement:
    return CodeElement(
        element_id="scope:repo:main:src/app.py:method:get_user:20",
        scope="scope",
        repository="repo",
        username="main",
        relative_path="src/app.py",
        element_type="method",
        name="get_user",
        language="python",
        line_start=20,
        line_end=30,
        raw_code="def get_user(self, user_id: int) -> User:\n    return self.db.get(user_id)",
        signature="def get_user(self, user_id: int) -> User",
        docstring="Get user by ID.",
        level=2,
        parent_id="scope:repo:main:src/app.py:class:UserService:10",
    )


@pytest.fixture
def job_repo() -> InMemoryJobRepository:
    return InMemoryJobRepository()


@pytest.fixture
def summary_store() -> InMemorySummaryStore:
    return InMemorySummaryStore()


# =============================================================================
# OLLAMA CLIENT
# =============================================================================


class TestSummarizationLLMClient:
    """Tests for LLM summarization client."""

    def test_generate_returns_response(self):
        """Test text generation returns LLM response."""
        client = SummarizationLLMClient("http://localhost:11434", "qwen2.5-coder:7b")
        # Mock the internal client
        client._client = MagicMock()
        client._client.generate.return_value = "This is a summary."

        result = client.generate("Summarize this code")

        assert result == "This is a summary."
        client._client.generate.assert_called_once()

    def test_verify_model_returns_true_when_available(self):
        """Test model verification returns True when model is available."""
        client = SummarizationLLMClient("http://localhost:11434", "qwen2.5-coder:7b")
        # Mock the internal client
        client._client = MagicMock()
        client._client.verify_model.return_value = True

        assert client.verify_model() is True

    def test_verify_model_returns_false_when_unavailable(self):
        """Test model verification returns False when model is unavailable."""
        client = SummarizationLLMClient("http://localhost:11434", "qwen2.5-coder:7b")
        # Mock the internal client
        client._client = MagicMock()
        client._client.verify_model.return_value = False

        assert client.verify_model() is False


# =============================================================================
# CODE TRUNCATION
# =============================================================================


class TestTruncateCode:
    """Tests for code truncation."""

    def test_returns_short_code_unchanged(self):
        code = "def foo(): pass"
        result = truncate_code(code, max_tokens=1000)
        assert result == code

    def test_truncates_long_code(self):
        code = "x = 1\n" * 1000  # Very long code
        result = truncate_code(code, max_tokens=100)
        assert len(result) < len(code)
        assert "# ... (truncated)" in result

    def test_truncates_at_line_boundary(self):
        code = "line1\nline2\nline3\nline4\nline5"
        result = truncate_code(code, max_tokens=5)  # ~20 chars
        # Should end at a newline, not mid-word
        assert result.endswith("# ... (truncated)")


# =============================================================================
# PROMPT BUILDING
# =============================================================================


class TestBuildPrompt:
    """Tests for prompt building."""

    def test_file_prompt(self, file_element: CodeElement):
        prompt = build_prompt(file_element, {})

        assert "python" in prompt.lower()
        assert "src/app.py" in prompt
        assert "flask" in prompt.lower() or "Flask" in prompt

    def test_class_prompt_includes_file_context(self, class_element: CodeElement):
        parent_summaries = {"file": "Flask application for user management."}
        prompt = build_prompt(class_element, parent_summaries)

        assert "Flask application" in prompt
        assert "UserService" in prompt
        assert "dataclass" in prompt  # decorator

    def test_method_prompt_includes_class_context(self, method_element: CodeElement):
        parent_summaries = {
            "file": "Flask application.",
            "class": "Service for user management.",
        }
        prompt = build_prompt(method_element, parent_summaries)

        assert "Service for user management" in prompt
        assert "get_user" in prompt
        assert "user_id: int" in prompt

    def test_includes_docstring_when_present(self, method_element: CodeElement):
        prompt = build_prompt(method_element, {})
        assert "Get user by ID" in prompt


# =============================================================================
# SUMMARY CLEANING
# =============================================================================


class TestCleanSummary:
    """Tests for summary cleaning."""

    def test_strips_whitespace(self):
        result = clean_summary("  A summary  ")
        assert result == "A summary."

    def test_removes_common_prefixes(self):
        result = clean_summary("Summary: Does something important")
        assert result.startswith("Does") or result.startswith("does")

        result = clean_summary("This function calculates the sum")
        assert "This function" not in result

    def test_adds_period_if_missing(self):
        result = clean_summary("Calculates the sum")
        assert result.endswith(".")

    def test_capitalizes_first_letter(self):
        result = clean_summary("calculates the sum")
        assert result[0].isupper()


# =============================================================================
# IN-MEMORY REPOSITORIES
# =============================================================================


class TestInMemoryJobRepository:
    """Tests for in-memory job repository."""

    def test_add_and_get_job(self, job_repo: InMemoryJobRepository):
        job_repo.add_job(
            element_id="test:id",
            scope="scope",
            repository="repo",
            username="main",
            level=1,
            parent_id="parent:id",
            dependencies_met=False,
        )

        job = job_repo.get_job("test:id", "scope", "repo", "main")
        assert job is not None
        assert job["level"] == 1
        assert job["status"] == "pending"

    def test_claim_jobs_by_level(self, job_repo: InMemoryJobRepository):
        # Add jobs at different levels
        job_repo.add_job("level0:job", scope="scope", repository="repo", username="main", level=0, parent_id=None, dependencies_met=True)
        job_repo.add_job("level1:job", scope="scope", repository="repo", username="main", level=1, parent_id="level0:job", dependencies_met=False)

        # Should only get level 0 job (dependencies met)
        claimed = job_repo.claim_pending_jobs(worker_id="w1", scope="scope", repository="repo", username="main", batch_size=10)
        assert len(claimed) == 1
        assert claimed[0]["element_id"] == "level0:job"

    def test_mark_job_completed(self, job_repo: InMemoryJobRepository):
        job_repo.add_job("test:id", scope="scope", repository="repo", username="main", level=0, parent_id=None, dependencies_met=True)
        job_repo.mark_completed("test:id", "scope", "repo", "main")

        job = job_repo.get_job("test:id", "scope", "repo", "main")
        assert job["status"] == "completed"

    def test_mark_job_failed(self, job_repo: InMemoryJobRepository):
        job_repo.add_job("test:id", scope="scope", repository="repo", username="main", level=0, parent_id=None, dependencies_met=True)
        job_repo.mark_failed("test:id", "scope", "repo", "main", "Some error")

        job = job_repo.get_job("test:id", "scope", "repo", "main")
        assert job["status"] == "failed"
        assert job["error_message"] == "Some error"


class TestInMemorySummaryStore:
    """Tests for in-memory summary store."""

    def test_store_and_get_element(
        self, summary_store: InMemorySummaryStore, file_element: CodeElement
    ):
        summary_store.store_element(file_element)

        elem = summary_store.get_element(file_element.element_id)
        assert elem is not None
        assert elem.name == "app.py"

    def test_store_summary(
        self, summary_store: InMemorySummaryStore, file_element: CodeElement
    ):
        summary_store.store_element(file_element)
        summary_store.store_summary(file_element.element_id, "A Flask application.")

        elem = summary_store.get_element(file_element.element_id)
        assert elem is not None
        # Check internal storage
        assert summary_store.get_summary(file_element.element_id) == "A Flask application."

    def test_get_parent_summaries(
        self,
        summary_store: InMemorySummaryStore,
        file_element: CodeElement,
        class_element: CodeElement,
    ):
        # Store file with summary
        summary_store.store_element(file_element)
        summary_store.store_summary(file_element.element_id, "Flask app.")

        # Get parent summaries for class
        class_element.parent_id = file_element.element_id
        summaries = summary_store.get_parent_summaries(class_element)

        assert "file" in summaries
        assert summaries["file"] == "Flask app."


# =============================================================================
# DEPENDENCY UPDATES
# =============================================================================


class TestUpdateDependencies:
    """Tests for dependency resolution."""

    def test_unlocks_child_jobs(self, job_repo: InMemoryJobRepository):
        # Parent job
        job_repo.add_job("parent:id", scope="scope", repository="repo", username="main", level=0, parent_id=None, dependencies_met=True)
        # Child job
        job_repo.add_job("child:id", scope="scope", repository="repo", username="main", level=1, parent_id="parent:id", dependencies_met=False)

        # Complete parent
        update_dependencies_after_completion("parent:id", "scope", "repo", "main", job_repo)

        # Child should now be unlocked
        child = job_repo.get_job("child:id", "scope", "repo", "main")
        assert child["dependencies_met"] is True


# =============================================================================
# SUMMARIZATION FLOW
# =============================================================================


class TestProcessSummarizationJob:
    """Tests for processing summarization jobs."""

    def test_generates_and_stores_summary(
        self,
        job_repo: InMemoryJobRepository,
        summary_store: InMemorySummaryStore,
        file_element: CodeElement,
        config: SummarizationConfig,
    ):
        # Setup
        summary_store.store_element(file_element)
        job_repo.add_job(
            file_element.element_id, scope="scope", repository="repo", username="main", level=0, parent_id=None, dependencies_met=True
        )

        # Mock Ollama
        mock_ollama = MagicMock()
        mock_ollama.generate.return_value = "Flask application for web services."

        # Process
        success = process_summarization_job(
            element_id=file_element.element_id,
            scope="scope",
            repository="repo",
            username="main",
            job_repo=job_repo,
            summary_store=summary_store,
            llm_client=mock_ollama,
            config=config,
        )

        assert success is True
        assert job_repo.get_job(file_element.element_id, "scope", "repo", "main")["status"] == "completed"
        assert summary_store.get_summary(file_element.element_id) is not None

    def test_handles_ollama_error(
        self,
        job_repo: InMemoryJobRepository,
        summary_store: InMemorySummaryStore,
        file_element: CodeElement,
        config: SummarizationConfig,
    ):
        # Setup
        summary_store.store_element(file_element)
        job_repo.add_job(
            file_element.element_id, scope="scope", repository="repo", username="main", level=0, parent_id=None, dependencies_met=True
        )

        # Mock Ollama with error
        mock_ollama = MagicMock()
        mock_ollama.generate.side_effect = Exception("Connection refused")

        # Process - should handle error gracefully
        success = process_summarization_job(
            element_id=file_element.element_id,
            scope="scope",
            repository="repo",
            username="main",
            job_repo=job_repo,
            summary_store=summary_store,
            llm_client=mock_ollama,
            config=config,
        )

        assert success is False
        job = job_repo.get_job(file_element.element_id, "scope", "repo", "main")
        assert job["status"] == "failed"
        assert "Connection refused" in job["error_message"]


class TestGenerateSummary:
    """Tests for summary generation."""

    def test_uses_parent_context(
        self,
        summary_store: InMemorySummaryStore,
        file_element: CodeElement,
        class_element: CodeElement,
        config: SummarizationConfig,
    ):
        # Setup file with summary
        summary_store.store_element(file_element)
        summary_store.store_summary(file_element.element_id, "Main application module.")

        # Store class
        class_element.parent_id = file_element.element_id
        summary_store.store_element(class_element)

        # Mock Ollama
        mock_ollama = MagicMock()
        mock_ollama.generate.return_value = "User service class."

        summary = generate_summary(
            element=class_element,
            summary_store=summary_store,
            llm_client=mock_ollama,
            config=config,
        )

        # Verify parent context was included in prompt
        call_args = mock_ollama.generate.call_args
        prompt = call_args.kwargs.get("prompt", "")
        assert "Main application module" in prompt


# =============================================================================
# DATA CLASSES
# =============================================================================


class TestSummarizationConfig:
    """Tests for configuration dataclass."""

    def test_default_values(self):
        config = SummarizationConfig()

        assert config.model == "qwen2.5-coder:3b"
        assert config.temperature == 0.3
        assert config.max_tokens == 256
        assert config.max_retries == 3

    def test_custom_values(self):
        config = SummarizationConfig(
            model="custom:model",
            temperature=0.5,
            max_tokens=512,
        )

        assert config.model == "custom:model"
        assert config.temperature == 0.5


class TestSummarizationResult:
    """Tests for result dataclass."""

    def test_default_values(self):
        result = SummarizationResult(scope="s", repository="r", username="u")

        assert result.files_summarized == 0
        assert result.total_jobs == 0
        assert result.errors == []
