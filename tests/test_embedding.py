"""Tests for the embedding module (Phase 6)."""

import math
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from magaldi.parser.code_parser import CodeElement
from magaldi.embedding import (
    EmbeddingConfig,
    EmbeddingResult,
    InMemoryEmbeddingJobRepository,
    InMemoryEmbeddingStore,
    OllamaEmbedClient,
    build_embedding_text,
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
        model="snowflake-arctic-embed2",
        dimensions=1024,
        max_context=8192,
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


class TestOllamaEmbedClient:
    """Tests for Ollama embedding client."""

    def test_embed_single_returns_vector(self):
        with patch("requests.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session_class.return_value = mock_session

            mock_response = MagicMock()
            mock_response.json.return_value = {"embeddings": [[0.1, 0.2, 0.3]]}
            mock_response.raise_for_status = MagicMock()
            mock_session.post.return_value = mock_response

            client = OllamaEmbedClient("http://localhost:11434", "test-model")
            result = client.embed_single("test text")

            assert result == [0.1, 0.2, 0.3]

    def test_embed_batch_returns_vectors(self):
        with patch("requests.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session_class.return_value = mock_session

            mock_response = MagicMock()
            mock_response.json.return_value = {
                "embeddings": [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
            }
            mock_response.raise_for_status = MagicMock()
            mock_session.post.return_value = mock_response

            client = OllamaEmbedClient("http://localhost:11434", "test-model")
            result = client.embed_batch(["text1", "text2", "text3"])

            assert len(result) == 3
            assert result[0] == [0.1, 0.2]

    def test_verify_model_returns_true_when_available(self):
        with patch("requests.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session_class.return_value = mock_session

            mock_response = MagicMock()
            mock_response.json.return_value = {
                "models": [{"name": "snowflake-arctic-embed2"}]
            }
            mock_session.get.return_value = mock_response

            client = OllamaEmbedClient("http://localhost:11434", "snowflake-arctic-embed2")
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


class TestBuildEmbeddingText:
    """Tests for embedding text construction."""

    def test_file_embedding_text(
        self, file_element: CodeElement, embedding_store: InMemoryEmbeddingStore
    ):
        # Add summary
        embedding_store.store_element(file_element)
        embedding_store.store_summary(file_element.element_id, "Main application module.")

        text = build_embedding_text(file_element, embedding_store)

        assert "src/app.py" in text
        assert "python" in text.lower()
        assert "Main application module" in text

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

        text = build_embedding_text(class_element, embedding_store)

        assert "Flask application" in text  # File context
        assert "UserService" in text
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

        text = build_embedding_text(method_element, embedding_store)

        assert "Main module" in text  # File context
        assert "User management service" in text  # Class context
        assert "get_user" in text
        assert "Retrieves user by ID" in text

    def test_includes_signature(
        self, method_element: CodeElement, embedding_store: InMemoryEmbeddingStore
    ):
        embedding_store.store_element(method_element)

        text = build_embedding_text(method_element, embedding_store)

        assert "def get_user" in text
        assert "user_id: int" in text

    def test_includes_docstring(
        self, method_element: CodeElement, embedding_store: InMemoryEmbeddingStore
    ):
        embedding_store.store_element(method_element)

        text = build_embedding_text(method_element, embedding_store)

        assert "Get user by ID" in text


# =============================================================================
# IN-MEMORY REPOSITORIES
# =============================================================================


class TestInMemoryEmbeddingJobRepository:
    """Tests for in-memory job repository."""

    def test_add_and_get_job(self, job_repo: InMemoryEmbeddingJobRepository):
        job_repo.add_job("test:id", "main")

        job = job_repo.get_job("test:id", "main")
        assert job is not None
        assert job["status"] == "pending"

    def test_claim_jobs(self, job_repo: InMemoryEmbeddingJobRepository):
        job_repo.add_job("job1", "main")
        job_repo.add_job("job2", "main")
        job_repo.add_job("job3", "main")

        claimed = job_repo.claim_pending_jobs(worker_id="w1", username="main", batch_size=2)
        assert len(claimed) == 2

        # Remaining jobs still pending
        remaining = job_repo.claim_pending_jobs(worker_id="w1", username="main", batch_size=2)
        assert len(remaining) == 1

    def test_mark_completed(self, job_repo: InMemoryEmbeddingJobRepository):
        job_repo.add_job("test:id", "main")
        job_repo.mark_completed("test:id", "main")

        job = job_repo.get_job("test:id", "main")
        assert job["status"] == "completed"

    def test_mark_failed(self, job_repo: InMemoryEmbeddingJobRepository):
        job_repo.add_job("test:id", "main")
        job_repo.mark_failed("test:id", "main", "Some error")

        job = job_repo.get_job("test:id", "main")
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
        job_repo.add_job(file_element.element_id, "main")

        # Mock Ollama
        mock_ollama = MagicMock()
        mock_ollama.embed_single.return_value = [0.1] * 1024

        # Process
        success = process_embedding_job(
            element_id=file_element.element_id,
            username="main",
            job_repo=job_repo,
            embedding_store=embedding_store,
            ollama=mock_ollama,
            config=config,
        )

        assert success is True
        assert job_repo.get_job(file_element.element_id, "main")["status"] == "completed"
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
        job_repo.add_job(file_element.element_id, "main")

        # Mock Ollama with error
        mock_ollama = MagicMock()
        mock_ollama.embed_single.side_effect = Exception("Connection refused")

        # Process
        success = process_embedding_job(
            element_id=file_element.element_id,
            username="main",
            job_repo=job_repo,
            embedding_store=embedding_store,
            ollama=mock_ollama,
            config=config,
        )

        assert success is False
        job = job_repo.get_job(file_element.element_id, "main")
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
        job_repo.add_job(file_element.element_id, "main")

        # Mock Ollama with wrong dimensions
        mock_ollama = MagicMock()
        mock_ollama.embed_single.return_value = [0.1] * 512  # Wrong!

        # Process
        success = process_embedding_job(
            element_id=file_element.element_id,
            username="main",
            job_repo=job_repo,
            embedding_store=embedding_store,
            ollama=mock_ollama,
            config=config,
        )

        assert success is False
        job = job_repo.get_job(file_element.element_id, "main")
        assert job["status"] == "failed"


# =============================================================================
# DATA CLASSES
# =============================================================================


class TestEmbeddingConfig:
    """Tests for configuration dataclass."""

    def test_default_values(self):
        config = EmbeddingConfig()

        assert config.model == "snowflake-arctic-embed2"
        assert config.dimensions == 1024
        assert config.max_context == 8192

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
