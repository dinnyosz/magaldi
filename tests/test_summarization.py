"""Tests for the summarization module (Phase 5)."""

from unittest.mock import MagicMock

import pytest

from magaldi_core.code_parser import CodeElement
from shared.ai.summarization import (
    InMemoryJobRepository,
    InMemorySummaryStore,
    SummarizationConfig,
    SummarizationLLMClient,
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
        temperature=0.2,
        max_tokens=512,
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
        result = clean_summary("  A summary text  ")
        assert result == "A summary text."

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

    # --- Orphaned think tag removal ---

    def test_orphaned_closing_think_tag_with_content(self):
        result = clean_summary("</think>\n\nConverts a string to uppercase.")
        assert result == "Converts a string to uppercase."

    def test_multiple_orphaned_closing_tags(self):
        result = clean_summary(
            "</think>\n\n</think>\n\nUsed to parse input data."
        )
        assert result == "Used to parse input data."

    def test_orphaned_closing_tag_only(self):
        """'</think>.' is too short after cleanup -> empty string."""
        result = clean_summary("</think>.")
        assert result == ""

    def test_orphaned_closing_thinking_tag(self):
        result = clean_summary("</thinking>\nHandles HTTP requests for the API.")
        assert result == "Handles HTTP requests for the API."

    def test_unclosed_opening_tag_consumes_to_end(self):
        """Opening <think> at start with content is treated as unclosed thinking (eaten)."""
        result = clean_summary("<think>\nReturns the cached result.")
        assert result == ""

    def test_orphaned_opening_tag_mid_text_consumed_by_unclosed(self):
        """Opening <think> mid-text is treated as unclosed thinking — content after is eaten."""
        result = clean_summary("Returns <think> the cached result.")
        # UNCLOSED pattern eats from <think> to end; "Returns" alone is too short
        assert result == ""

    # --- Paired tag removal still works (regression) ---

    def test_paired_think_tags_removed(self):
        result = clean_summary(
            "<think>I need to summarize this</think>Validates user input."
        )
        assert result == "Validates user input."

    def test_paired_thinking_tags_removed(self):
        result = clean_summary(
            "<thinking>Let me think about this code</thinking> Parses JSON from the request body."
        )
        assert result == "Parses JSON from the request body."

    # --- Leading markdown before anti-pattern prefix ---

    def test_leading_bullet_before_prefix(self):
        result = clean_summary("- This function parses config files")
        assert "This function" not in result
        assert result.startswith("Parses")

    def test_leading_bold_before_prefix(self):
        result = clean_summary("**This method validates input data")
        assert "This method" not in result
        assert result.startswith("Validates")

    def test_leading_header_before_prefix(self):
        result = clean_summary("# This class manages database connections")
        assert "This class" not in result
        assert result.startswith("Manages")

    # --- Expanded prefix removal ---

    def test_removes_this_module_prefix(self):
        result = clean_summary("This module provides utility functions")
        assert result.startswith("Provides")

    def test_removes_the_function_prefix(self):
        result = clean_summary("The function calculates the hash value")
        assert result.startswith("Calculates")

    def test_removes_the_class_prefix(self):
        result = clean_summary("The class represents a user session")
        assert result.startswith("Represents")

    def test_removes_bold_summary_prefix(self):
        result = clean_summary("**Summary:** Extracts tokens from source code")
        assert result.startswith("Extracts")

    # --- Quality validation ---

    def test_too_short_summary_returns_empty(self):
        result = clean_summary("Hi.")
        assert result == ""

    def test_no_ascii_letters_returns_empty(self):
        """Arabic/non-ASCII only text -> empty string."""
        result = clean_summary("مرحبا بالعالم")
        assert result == ""

    def test_pure_punctuation_returns_empty(self):
        result = clean_summary("... --- !!!")
        assert result == ""

    # --- Real-world examples from OpenSearch ---

    def test_real_example_think_dot(self):
        result = clean_summary("</think>.")
        assert result == ""

    def test_real_example_multiple_orphans_with_antipattern(self):
        result = clean_summary(
            "</think>\n\n</think>\n\nThe function is used to validate input"
        )
        assert "The function" not in result
        assert result.startswith("Is used") or result.startswith("Used") or result.startswith("Validate")

    def test_real_example_hallucination_after_orphan(self):
        """'Convert\\n\\n</think>\\n\\nIslamic emoticon.' -> too short or no sense."""
        result = clean_summary("Convert\n\n</think>\n\nIslamic emoticon.")
        # After orphan removal: "Convert  Islamic emoticon." — may or may not pass length
        # but if it does pass, at least it's cleaned of tags
        assert "</think>" not in result


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
        claimed = job_repo.claim_pending_jobs(worker_id="w1", _scope="scope", _repository="repo", _username="main", batch_size=10)
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

        # Mock Ollama (uses generate_from_messages for prefix caching)
        mock_ollama = MagicMock()
        mock_ollama.generate_from_messages.return_value = "Flask application for web services."

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

        # Mock Ollama with error (uses generate_from_messages for prefix caching)
        mock_ollama = MagicMock()
        mock_ollama.generate_from_messages.side_effect = Exception("Connection refused")

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

        # Mock Ollama (uses generate_from_messages for prefix caching)
        mock_ollama = MagicMock()
        mock_ollama.generate_from_messages.return_value = "User service class."

        generate_summary(
            element=class_element,
            summary_store=summary_store,
            llm_client=mock_ollama,
            config=config,
        )

        # Verify parent context was included in user message
        call_args = mock_ollama.generate_from_messages.call_args
        messages = call_args.kwargs.get("messages", [])
        # User message is second (after system message) and contains file context
        user_content = messages[1]["content"] if len(messages) > 1 else ""
        assert "Main application module" in user_content


# =============================================================================
# DATA CLASSES
# =============================================================================


class TestSummarizationConfig:
    """Tests for configuration dataclass."""

    def test_default_values(self):
        config = SummarizationConfig()

        assert config.model == "qwen3.5:4b"
        assert config.temperature == 0.6
        assert config.max_tokens == 512
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


# =============================================================================
# NUM_CTX PARAMETER
# =============================================================================


class TestSummarizationLLMClientNumCtx:
    """Tests for num_ctx parameter in SummarizationLLMClient."""

    def test_generate_passes_num_ctx(self):
        """Should pass num_ctx to underlying client."""
        mock_client = MagicMock()
        mock_client.generate.return_value = "test summary"

        client = SummarizationLLMClient(
            url="http://localhost:11434",
            model="qwen3:4b",
            provider="ollama",
        )
        client._client = mock_client
        client.generate("test prompt", num_ctx=4096)

        mock_client.generate.assert_called_once()
        call_kwargs = mock_client.generate.call_args[1]
        assert call_kwargs.get("num_ctx") == 4096

    def test_generate_from_messages_passes_num_ctx(self):
        """Should pass num_ctx in generate_from_messages."""
        mock_client = MagicMock()
        mock_client.generate_from_messages.return_value = "test summary"

        client = SummarizationLLMClient(
            url="http://localhost:11434",
            model="qwen3:4b",
            provider="ollama",
        )
        client._client = mock_client
        client.generate_from_messages(
            [{"role": "user", "content": "test"}],
            num_ctx=8192
        )

        mock_client.generate_from_messages.assert_called_once()
        call_kwargs = mock_client.generate_from_messages.call_args[1]
        assert call_kwargs.get("num_ctx") == 8192


class TestSummarizationLLMClientSamplingParams:
    """Tests for top_k, min_p, presence_penalty, repetition_penalty passthrough."""

    def test_generate_passes_sampling_params(self):
        """Should forward top_k, min_p, presence_penalty, repetition_penalty to underlying client."""
        mock_client = MagicMock()
        mock_client.generate.return_value = "test summary"

        client = SummarizationLLMClient(
            url="http://localhost:11434",
            model="qwen3:4b",
            provider="ollama",
        )
        client._client = mock_client
        client.generate(
            "test prompt",
            top_k=20,
            min_p=0.05,
            presence_penalty=1.5,
            repetition_penalty=1.1,
        )

        mock_client.generate.assert_called_once()
        call_kwargs = mock_client.generate.call_args[1]
        assert call_kwargs.get("top_k") == 20
        assert call_kwargs.get("min_p") == 0.05
        assert call_kwargs.get("presence_penalty") == 1.5
        assert call_kwargs.get("repetition_penalty") == 1.1

    def test_generate_from_messages_passes_sampling_params(self):
        """Should forward top_k, min_p, presence_penalty, repetition_penalty in generate_from_messages."""
        mock_client = MagicMock()
        mock_client.generate_from_messages.return_value = "test summary"

        client = SummarizationLLMClient(
            url="http://localhost:11434",
            model="qwen3:4b",
            provider="ollama",
        )
        client._client = mock_client
        client.generate_from_messages(
            [{"role": "user", "content": "test"}],
            top_k=20,
            min_p=0.05,
            presence_penalty=1.5,
            repetition_penalty=1.1,
        )

        mock_client.generate_from_messages.assert_called_once()
        call_kwargs = mock_client.generate_from_messages.call_args[1]
        assert call_kwargs.get("top_k") == 20
        assert call_kwargs.get("min_p") == 0.05
        assert call_kwargs.get("presence_penalty") == 1.5
        assert call_kwargs.get("repetition_penalty") == 1.1

    def test_generate_defaults_sampling_params_to_none(self):
        """Sampling params should default to None when not provided."""
        mock_client = MagicMock()
        mock_client.generate.return_value = "test summary"

        client = SummarizationLLMClient(
            url="http://localhost:11434",
            model="qwen3:4b",
            provider="ollama",
        )
        client._client = mock_client
        client.generate("test prompt")

        call_kwargs = mock_client.generate.call_args[1]
        assert call_kwargs.get("top_k") is None
        assert call_kwargs.get("min_p") is None
        assert call_kwargs.get("presence_penalty") is None
        assert call_kwargs.get("repetition_penalty") is None

    def test_generate_from_messages_defaults_sampling_params_to_none(self):
        """Sampling params should default to None in generate_from_messages."""
        mock_client = MagicMock()
        mock_client.generate_from_messages.return_value = "test summary"

        client = SummarizationLLMClient(
            url="http://localhost:11434",
            model="qwen3:4b",
            provider="ollama",
        )
        client._client = mock_client
        client.generate_from_messages([{"role": "user", "content": "test"}])

        call_kwargs = mock_client.generate_from_messages.call_args[1]
        assert call_kwargs.get("top_k") is None
        assert call_kwargs.get("min_p") is None
        assert call_kwargs.get("presence_penalty") is None
        assert call_kwargs.get("repetition_penalty") is None


# =============================================================================
# PER-TYPE OUTPUT TOKEN BUDGETS
# =============================================================================


class TestOutputTokenBudgets:
    """Tests for per-element-type max_tokens enforcement."""

    def test_get_max_tokens_known_types(self):
        """All known element types should return their specific budget."""
        from shared.ai.prompts import OUTPUT_TOKEN_BUDGETS, get_max_tokens_for_element_type

        for elem_type, expected in OUTPUT_TOKEN_BUDGETS.items():
            assert get_max_tokens_for_element_type(elem_type) == expected

    def test_get_max_tokens_unknown_type_returns_default(self):
        """Unknown element types should return the default."""
        from shared.ai.prompts import get_max_tokens_for_element_type

        assert get_max_tokens_for_element_type("unknown_type") == 512
        assert get_max_tokens_for_element_type("unknown_type", default=256) == 256

    def test_generate_summary_uses_per_type_max_tokens(
        self, config: SummarizationConfig, method_element: CodeElement,
    ):
        """generate_summary should pass per-type max_tokens to LLM, not config default."""
        from shared.ai.prompts import OUTPUT_TOKEN_BUDGETS

        client = SummarizationLLMClient("http://localhost:11434", "qwen2.5-coder:7b")
        client._client = MagicMock()
        client._client.generate_from_messages.return_value = "Retrieves a user by ID."

        store = InMemorySummaryStore()
        store.store_element(method_element)
        # Store parent summaries so build_messages has context
        store.store_summary(
            "scope:repo:main:src/app.py:file:app.py:1",
            "Flask application.",
        )

        generate_summary(method_element, store, client, config)

        call_kwargs = client._client.generate_from_messages.call_args[1]
        expected_max_tokens = OUTPUT_TOKEN_BUDGETS["method"]
        assert call_kwargs["max_tokens"] == expected_max_tokens

    def test_generate_summary_constant_gets_small_budget(
        self, config: SummarizationConfig,
    ):
        """Constants should get a much smaller max_tokens than the default 512."""
        from shared.ai.prompts import OUTPUT_TOKEN_BUDGETS

        constant_element = CodeElement(
            element_id="scope:repo:main:src/config.py:constant:MAX_RETRIES:5",
            scope="scope",
            repository="repo",
            username="main",
            relative_path="src/config.py",
            element_type="constant",
            name="MAX_RETRIES",
            language="python",
            line_start=5,
            line_end=5,
            raw_code="MAX_RETRIES = 3",
            level=0,
        )

        client = SummarizationLLMClient("http://localhost:11434", "qwen2.5-coder:7b")
        client._client = MagicMock()
        client._client.generate_from_messages.return_value = "Maximum retry attempts."

        store = InMemorySummaryStore()
        store.store_element(constant_element)

        generate_summary(constant_element, store, client, config)

        call_kwargs = client._client.generate_from_messages.call_args[1]
        expected_max_tokens = OUTPUT_TOKEN_BUDGETS["constant"]
        assert call_kwargs["max_tokens"] == expected_max_tokens
        assert expected_max_tokens < config.max_tokens  # 200 < 512
