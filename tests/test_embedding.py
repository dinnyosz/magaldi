"""Tests for the embedding module (Phase 6)."""

import math
from unittest.mock import MagicMock

import pytest

from magaldi_core.code_parser import CodeElement
from magaldi_core.parsers.base import Call
from shared.ai.embedding import (
    CodeEmbeddingClient,
    EmbeddingConfig,
    EmbeddingResult,
    InMemoryEmbeddingJobRepository,
    InMemoryEmbeddingStore,
    build_caller_embedding_text,
    build_code_embedding_text,
    build_embedding_text,
    build_summary_embedding_text,
    estimate_tokens,
    normalize_vector,
    process_embedding_job,
    validate_context_length,
    validate_vector,
)

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def config() -> EmbeddingConfig:
    return EmbeddingConfig(
        ollama_url="http://localhost:11434",
        model="qwen3-embedding:0.6b",
        dimensions=1024,
        max_context=32768,
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
        docstring="Service for user management.",
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
        signature="def get_user(self, user_id: int) -> User",
        docstring="Get user by ID.",
        level=2,
        parent_id="scope:repo:main:src/app.py:class:UserService:10",
    )


@pytest.fixture
def job_repo() -> InMemoryEmbeddingJobRepository:
    return InMemoryEmbeddingJobRepository()


@pytest.fixture
def embedding_store() -> InMemoryEmbeddingStore:
    return InMemoryEmbeddingStore()


# =============================================================================
# OLLAMA EMBED CLIENT
# =============================================================================


class TestCodeEmbeddingClient:
    """Tests for LLM embedding client."""

    def test_embed_single_returns_vector(self):
        """Test single text embedding generation."""
        client = CodeEmbeddingClient("http://localhost:11434", "test-model")
        # Mock the internal client
        client._client = MagicMock()
        client._client.embed.return_value = [0.1, 0.2, 0.3]

        result = client.embed_single("test text")

        assert result == [0.1, 0.2, 0.3]
        client._client.embed.assert_called_once_with("test text", timeout=30)

    def test_embed_batch_returns_vectors(self):
        """Test batch embedding generation."""
        client = CodeEmbeddingClient("http://localhost:11434", "test-model")
        # Mock the internal client
        client._client = MagicMock()
        client._client.embed_batch.return_value = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]

        result = client.embed_batch(["text1", "text2", "text3"])

        assert len(result) == 3
        assert result[0] == [0.1, 0.2]

    def test_verify_model_returns_true_when_available(self):
        """Test model verification returns True when model is available."""
        client = CodeEmbeddingClient("http://localhost:11434", "qwen3-embedding:0.6b")
        # Mock the internal client
        client._client = MagicMock()
        client._client.verify_model.return_value = True

        assert client.verify_model() is True


# =============================================================================
# VECTOR OPERATIONS
# =============================================================================


class TestNormalizeVector:
    """Tests for vector normalization."""

    def test_normalizes_to_unit_length(self):
        vector = [3.0, 4.0]  # 3-4-5 triangle
        normalized = normalize_vector(vector)

        # Check unit length
        magnitude = math.sqrt(sum(x * x for x in normalized))
        assert abs(magnitude - 1.0) < 0.0001

    def test_preserves_direction(self):
        vector = [1.0, 2.0, 3.0]
        normalized = normalize_vector(vector)

        # Ratios should be preserved
        assert abs(normalized[1] / normalized[0] - 2.0) < 0.0001
        assert abs(normalized[2] / normalized[0] - 3.0) < 0.0001

    def test_handles_zero_vector(self):
        vector = [0.0, 0.0, 0.0]
        normalized = normalize_vector(vector)

        assert normalized == [0.0, 0.0, 0.0]


class TestValidateVector:
    """Tests for vector validation."""

    def test_valid_vector(self):
        vector = [0.1] * 1024
        assert validate_vector(vector, expected_dims=1024) is True

    def test_wrong_dimensions(self):
        vector = [0.1] * 512
        assert validate_vector(vector, expected_dims=1024) is False

    def test_nan_values(self):
        vector = [0.1, float("nan"), 0.3]
        assert validate_vector(vector, expected_dims=3) is False

    def test_inf_values(self):
        vector = [0.1, float("inf"), 0.3]
        assert validate_vector(vector, expected_dims=3) is False


# =============================================================================
# TOKEN ESTIMATION
# =============================================================================


class TestEstimateTokens:
    """Tests for token estimation."""

    def test_estimates_tokens(self):
        text = "This is a test sentence."
        tokens = estimate_tokens(text)

        # Rough estimate: len/3 for code
        assert tokens > 0
        assert tokens < len(text)

    def test_empty_string(self):
        assert estimate_tokens("") == 0


class TestValidateContextLength:
    """Tests for context length validation."""

    def test_short_text_unchanged(self):
        text = "Short text."
        result = validate_context_length(text, max_tokens=1000)
        assert result == text

    def test_long_text_truncated(self):
        text = "Line\n" * 10000
        result = validate_context_length(text, max_tokens=100)
        assert len(result) < len(text)
        assert "truncated" in result


# =============================================================================
# EMBEDDING TEXT BUILDING
# =============================================================================


class TestBuildSummaryEmbeddingText:
    """Tests for summary embedding text construction."""

    def test_file_embedding_text(
        self, file_element: CodeElement, embedding_store: InMemoryEmbeddingStore
    ):
        # Add summary
        embedding_store.store_element(file_element)
        embedding_store.store_summary(file_element.element_id, "Main application module.")

        text = build_summary_embedding_text(file_element, embedding_store)

        assert "src/app.py" in text
        assert "python" in text.lower()
        assert "Main application module" in text

    def test_backwards_compat_alias(
        self, file_element: CodeElement, embedding_store: InMemoryEmbeddingStore
    ):
        """Test that build_embedding_text is an alias for build_summary_embedding_text."""
        embedding_store.store_element(file_element)
        embedding_store.store_summary(file_element.element_id, "Main application module.")

        text1 = build_embedding_text(file_element, embedding_store)
        text2 = build_summary_embedding_text(file_element, embedding_store)

        assert text1 == text2

    def test_class_embedding_text_includes_file_context(
        self,
        file_element: CodeElement,
        class_element: CodeElement,
        embedding_store: InMemoryEmbeddingStore,
    ):
        # Setup file with summary
        embedding_store.store_element(file_element)
        embedding_store.store_summary(file_element.element_id, "Flask application.")

        # Setup class
        embedding_store.store_element(class_element)
        embedding_store.store_summary(class_element.element_id, "User service class.")

        text = build_summary_embedding_text(class_element, embedding_store)

        assert "Flask application" in text  # File context
        assert "Class: user service" in text  # Humanized name
        assert "User service class" in text

    def test_method_embedding_text_includes_class_context(
        self,
        file_element: CodeElement,
        class_element: CodeElement,
        method_element: CodeElement,
        embedding_store: InMemoryEmbeddingStore,
    ):
        # Setup file
        embedding_store.store_element(file_element)
        embedding_store.store_summary(file_element.element_id, "Main module.")

        # Setup class
        embedding_store.store_element(class_element)
        embedding_store.store_summary(class_element.element_id, "User management service.")

        # Setup method
        embedding_store.store_element(method_element)
        embedding_store.store_summary(method_element.element_id, "Retrieves user by ID.")

        text = build_summary_embedding_text(method_element, embedding_store)

        assert "Main module" in text  # File context
        assert "User management service" in text  # Class context
        assert "Function: get user" in text  # Humanized name
        assert "Retrieves user by ID" in text

    def test_includes_signature(
        self, method_element: CodeElement, embedding_store: InMemoryEmbeddingStore
    ):
        embedding_store.store_element(method_element)

        text = build_summary_embedding_text(method_element, embedding_store)

        assert "def get_user" in text
        assert "user_id: int" in text

    def test_includes_docstring(
        self, method_element: CodeElement, embedding_store: InMemoryEmbeddingStore
    ):
        embedding_store.store_element(method_element)

        text = build_summary_embedding_text(method_element, embedding_store)

        assert "Get user by ID" in text

    def test_summary_excludes_outbound_calls(
        self, embedding_store: InMemoryEmbeddingStore
    ):
        """Summary embedding (passport) should NOT include calls."""
        element = CodeElement(
            element_id="scope:repo:main:src/app.py:function:process:50",
            scope="scope",
            repository="repo",
            username="main",
            relative_path="src/app.py",
            element_type="function",
            name="process",
            language="python",
            line_start=50,
            line_end=70,
            level=2,
            calls=[
                Call(name="validate", receiver="self", line=55),
                Call(name="save", receiver="db", line=60),
            ],
        )
        embedding_store.store_element(element)

        text = build_summary_embedding_text(element, embedding_store)

        assert "Calls:" not in text

    def test_no_calls_line_when_empty(
        self, method_element: CodeElement, embedding_store: InMemoryEmbeddingStore
    ):
        embedding_store.store_element(method_element)

        text = build_summary_embedding_text(method_element, embedding_store)

        assert "Calls:" not in text

    def test_function_name_humanized_in_passport(
        self, embedding_store: InMemoryEmbeddingStore
    ):
        """Function names should be humanized (snake_case -> space-separated)."""
        element = CodeElement(
            element_id="scope:repo:main:src/app.py:function:get_user_by_email:10",
            scope="scope",
            repository="repo",
            username="main",
            relative_path="src/app.py",
            element_type="function",
            name="get_user_by_email",
            language="python",
            line_start=10,
            line_end=20,
            level=2,
        )
        embedding_store.store_element(element)

        text = build_summary_embedding_text(element, embedding_store)

        assert "Function: get user by email" in text
        # Raw name should NOT appear in the Function: line
        assert "Function: get_user_by_email" not in text

    def test_class_name_humanized_in_passport(
        self, embedding_store: InMemoryEmbeddingStore
    ):
        """Class names should be humanized (PascalCase -> space-separated)."""
        element = CodeElement(
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
            level=1,
        )
        embedding_store.store_element(element)

        text = build_summary_embedding_text(element, embedding_store)

        assert "Class: user service" in text

    def test_variable_name_humanized_in_passport(
        self, embedding_store: InMemoryEmbeddingStore
    ):
        """Variable names should be humanized."""
        element = CodeElement(
            element_id="scope:repo:main:src/app.py:variable:max_retry_count:5",
            scope="scope",
            repository="repo",
            username="main",
            relative_path="src/app.py",
            element_type="variable",
            name="MAX_RETRY_COUNT",
            language="python",
            line_start=5,
            line_end=5,
            level=2,
        )
        embedding_store.store_element(element)

        text = build_summary_embedding_text(element, embedding_store)

        assert "Name: max retry count" in text


class TestBuildCallerEmbeddingText:
    """Tests for caller embedding text construction (passport + calls)."""

    def test_includes_outbound_calls(
        self, embedding_store: InMemoryEmbeddingStore
    ):
        element = CodeElement(
            element_id="scope:repo:main:src/app.py:function:process:50",
            scope="scope",
            repository="repo",
            username="main",
            relative_path="src/app.py",
            element_type="function",
            name="process",
            language="python",
            line_start=50,
            line_end=70,
            level=2,
            calls=[
                Call(name="validate", receiver="self", line=55),
                Call(name="save", receiver="db", line=60),
                Call(name="log", receiver=None, line=65),
                # Duplicate should be deduplicated
                Call(name="validate", receiver="self", line=68),
            ],
        )
        embedding_store.store_element(element)

        text = build_caller_embedding_text(element, embedding_store)

        assert "Calls: self.validate, db.save, log" in text

    def test_no_calls_produces_same_as_summary(
        self, method_element: CodeElement, embedding_store: InMemoryEmbeddingStore
    ):
        """Without calls, caller text is identical to summary text."""
        embedding_store.store_element(method_element)

        summary_text = build_summary_embedding_text(method_element, embedding_store)
        caller_text = build_caller_embedding_text(method_element, embedding_store)

        assert summary_text == caller_text
        assert "Calls:" not in caller_text

    def test_includes_passport_and_calls(
        self,
        file_element: CodeElement,
        embedding_store: InMemoryEmbeddingStore,
    ):
        """Caller text includes both passport info and calls."""
        element = CodeElement(
            element_id="scope:repo:main:src/app.py:function:handle:10",
            scope="scope",
            repository="repo",
            username="main",
            relative_path="src/app.py",
            element_type="function",
            name="handle",
            language="python",
            line_start=10,
            line_end=30,
            signature="def handle(request)",
            level=2,
            calls=[
                Call(name="process", receiver=None, line=15),
            ],
        )
        embedding_store.store_element(file_element)
        embedding_store.store_summary(file_element.element_id, "Main app module.")
        embedding_store.store_element(element)
        embedding_store.store_summary(element.element_id, "Handles incoming requests.")

        text = build_caller_embedding_text(element, embedding_store)

        # Passport parts
        assert "handle" in text
        assert "Handles incoming requests" in text
        # Calls part
        assert "Calls: process" in text


class TestBuildCodeEmbeddingText:
    """Tests for code embedding text construction."""

    def test_includes_element_type_and_name(self, method_element: CodeElement):
        text = build_code_embedding_text(method_element)

        assert "# method: get_user" in text

    def test_includes_file_path(self, method_element: CodeElement):
        text = build_code_embedding_text(method_element)

        assert "# File: src/app.py" in text

    def test_includes_signature(self, method_element: CodeElement):
        text = build_code_embedding_text(method_element)

        assert "# Signature: def get_user(self, user_id: int) -> User" in text

    def test_includes_raw_code(self):
        element = CodeElement(
            element_id="scope:repo:main:src/app.py:function:process:10",
            scope="scope",
            repository="repo",
            username="main",
            relative_path="src/app.py",
            element_type="function",
            name="process",
            language="python",
            line_start=10,
            line_end=15,
            raw_code="def process(data):\n    return data.strip()",
        )

        text = build_code_embedding_text(element)

        assert "def process(data):" in text
        assert "return data.strip()" in text

    def test_no_store_required(self, method_element: CodeElement):
        """Code embedding doesn't need embedding store (unlike summary embedding)."""
        # This should work without any store
        text = build_code_embedding_text(method_element)
        assert text  # Just verify it produces output

    def test_respects_max_tokens(self):
        # Create element with very long code
        long_code = "x = 1\n" * 10000
        element = CodeElement(
            element_id="scope:repo:main:src/long.py:function:long_fn:1",
            scope="scope",
            repository="repo",
            username="main",
            relative_path="src/long.py",
            element_type="function",
            name="long_fn",
            language="python",
            line_start=1,
            line_end=10000,
            raw_code=long_code,
        )

        text = build_code_embedding_text(element, max_tokens=100)

        assert len(text) < len(long_code)
        assert "truncated" in text


# =============================================================================
# IN-MEMORY REPOSITORIES
# =============================================================================


class TestInMemoryEmbeddingJobRepository:
    """Tests for in-memory job repository."""

    def test_add_and_get_job(self, job_repo: InMemoryEmbeddingJobRepository):
        job_repo.add_job("test:id", "scope", "repo", "main")

        job = job_repo.get_job("test:id", "scope", "repo", "main")
        assert job is not None
        assert job["status"] == "pending"

    def test_claim_jobs(self, job_repo: InMemoryEmbeddingJobRepository):
        job_repo.add_job("job1", "scope", "repo", "main")
        job_repo.add_job("job2", "scope", "repo", "main")
        job_repo.add_job("job3", "scope", "repo", "main")

        claimed = job_repo.claim_pending_jobs(worker_id="w1", _scope="scope", _repository="repo", _username="main", batch_size=2)
        assert len(claimed) == 2

        # Remaining jobs still pending
        remaining = job_repo.claim_pending_jobs(worker_id="w1", _scope="scope", _repository="repo", _username="main", batch_size=2)
        assert len(remaining) == 1

    def test_mark_completed(self, job_repo: InMemoryEmbeddingJobRepository):
        job_repo.add_job("test:id", "scope", "repo", "main")
        job_repo.mark_completed("test:id", "scope", "repo", "main")

        job = job_repo.get_job("test:id", "scope", "repo", "main")
        assert job["status"] == "completed"

    def test_mark_failed(self, job_repo: InMemoryEmbeddingJobRepository):
        job_repo.add_job("test:id", "scope", "repo", "main")
        job_repo.mark_failed("test:id", "scope", "repo", "main", "Some error")

        job = job_repo.get_job("test:id", "scope", "repo", "main")
        assert job["status"] == "failed"
        assert job["error_message"] == "Some error"


class TestInMemoryEmbeddingStore:
    """Tests for in-memory embedding store."""

    def test_store_and_get_element(
        self, embedding_store: InMemoryEmbeddingStore, file_element: CodeElement
    ):
        embedding_store.store_element(file_element)

        elem = embedding_store.get_element(file_element.element_id)
        assert elem is not None
        assert elem.name == "app.py"

    def test_store_and_get_summary(
        self, embedding_store: InMemoryEmbeddingStore, file_element: CodeElement
    ):
        embedding_store.store_element(file_element)
        embedding_store.store_summary(file_element.element_id, "A module.")

        summary = embedding_store.get_summary(file_element.element_id)
        assert summary == "A module."

    def test_store_and_get_embedding(
        self, embedding_store: InMemoryEmbeddingStore, file_element: CodeElement
    ):
        embedding_store.store_element(file_element)
        vector = [0.1] * 1024
        embedding_store.store_embedding(file_element.element_id, vector)

        retrieved = embedding_store.get_embedding(file_element.element_id)
        assert retrieved == vector

    def test_get_file_summary(
        self,
        embedding_store: InMemoryEmbeddingStore,
        file_element: CodeElement,
        class_element: CodeElement,
    ):
        embedding_store.store_element(file_element)
        embedding_store.store_summary(file_element.element_id, "File summary.")
        embedding_store.store_element(class_element)

        summary = embedding_store.get_file_summary(class_element)
        assert summary == "File summary."

    def test_get_class_summary(
        self,
        embedding_store: InMemoryEmbeddingStore,
        class_element: CodeElement,
        method_element: CodeElement,
    ):
        embedding_store.store_element(class_element)
        embedding_store.store_summary(class_element.element_id, "Class summary.")
        embedding_store.store_element(method_element)

        summary = embedding_store.get_class_summary(method_element)
        assert summary == "Class summary."


# =============================================================================
# EMBEDDING FLOW
# =============================================================================


class TestProcessEmbeddingJob:
    """Tests for processing embedding jobs."""

    def test_generates_and_stores_embedding(
        self,
        job_repo: InMemoryEmbeddingJobRepository,
        embedding_store: InMemoryEmbeddingStore,
        file_element: CodeElement,
        config: EmbeddingConfig,
    ):
        # Setup
        embedding_store.store_element(file_element)
        embedding_store.store_summary(file_element.element_id, "Main module.")
        job_repo.add_job(file_element.element_id, "scope", "repo", "main")

        # Mock Ollama
        mock_ollama = MagicMock()
        mock_ollama.embed_single.return_value = [0.1] * 1024

        # Process
        success = process_embedding_job(
            element_id=file_element.element_id,
            scope="scope",
            repository="repo",
            username="main",
            job_repo=job_repo,
            embedding_store=embedding_store,
            embed_client=mock_ollama,
            config=config,
        )

        assert success is True
        assert job_repo.get_job(file_element.element_id, "scope", "repo", "main")["status"] == "completed"
        assert embedding_store.get_embedding(file_element.element_id) is not None

    def test_handles_ollama_error(
        self,
        job_repo: InMemoryEmbeddingJobRepository,
        embedding_store: InMemoryEmbeddingStore,
        file_element: CodeElement,
        config: EmbeddingConfig,
    ):
        # Setup
        embedding_store.store_element(file_element)
        job_repo.add_job(file_element.element_id, "scope", "repo", "main")

        # Mock Ollama with error
        mock_ollama = MagicMock()
        mock_ollama.embed_single.side_effect = Exception("Connection refused")

        # Process
        success = process_embedding_job(
            element_id=file_element.element_id,
            scope="scope",
            repository="repo",
            username="main",
            job_repo=job_repo,
            embedding_store=embedding_store,
            embed_client=mock_ollama,
            config=config,
        )

        assert success is False
        job = job_repo.get_job(file_element.element_id, "scope", "repo", "main")
        assert job["status"] == "failed"
        assert "Connection refused" in job["error_message"]

    def test_validates_dimensions(
        self,
        job_repo: InMemoryEmbeddingJobRepository,
        embedding_store: InMemoryEmbeddingStore,
        file_element: CodeElement,
        config: EmbeddingConfig,
    ):
        # Setup
        embedding_store.store_element(file_element)
        job_repo.add_job(file_element.element_id, "scope", "repo", "main")

        # Mock Ollama with wrong dimensions
        mock_ollama = MagicMock()
        mock_ollama.embed_single.return_value = [0.1] * 512  # Wrong!

        # Process
        success = process_embedding_job(
            element_id=file_element.element_id,
            scope="scope",
            repository="repo",
            username="main",
            job_repo=job_repo,
            embedding_store=embedding_store,
            embed_client=mock_ollama,
            config=config,
        )

        assert success is False
        job = job_repo.get_job(file_element.element_id, "scope", "repo", "main")
        assert job["status"] == "failed"


# =============================================================================
# DATA CLASSES
# =============================================================================


class TestEmbeddingConfig:
    """Tests for configuration dataclass."""

    def test_default_values(self):
        config = EmbeddingConfig()

        assert config.model == "qwen3-embedding:0.6b"
        assert config.dimensions == 1024
        assert config.max_context == 32768

    def test_custom_values(self):
        config = EmbeddingConfig(
            model="custom:model",
            dimensions=768,
        )

        assert config.model == "custom:model"
        assert config.dimensions == 768


class TestEmbeddingResult:
    """Tests for result dataclass."""

    def test_default_values(self):
        result = EmbeddingResult(scope="s", repository="r", username="u")

        assert result.total_elements == 0
        assert result.embedded_elements == 0
        assert result.errors == []
