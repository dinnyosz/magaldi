"""Integration tests for Elasticsearch repository.

These tests require a running Elasticsearch instance (via Docker).
Run with: pytest -m integration tests/test_db_elasticsearch.py
"""

from __future__ import annotations

import time

import pytest

from magaldi.config import load_config
from magaldi.db.elasticsearch import INDEX_NAME
from magaldi.parser.code_parser import CodeElement


# Skip all tests if Elasticsearch is not available
pytestmark = pytest.mark.integration


@pytest.fixture(scope="function")
def config():
    """Load config for ES connection."""
    import os
    from magaldi.config import reset_config, MagaldiConfig, ElasticsearchConfig

    # Reset config singleton
    reset_config()

    # Create config directly with known good values for integration tests
    return MagaldiConfig(
        elasticsearch=ElasticsearchConfig(
            host=os.environ.get("MAGALDI_ELASTICSEARCH_HOST", "localhost"),
            port=int(os.environ.get("MAGALDI_ELASTICSEARCH_PORT", "9200")),
            scheme=os.environ.get("MAGALDI_ELASTICSEARCH_SCHEME", "http"),
        ),
    )


@pytest.fixture
def es_repo(config):
    """Create ES repository and clean up test data."""
    from magaldi.db.elasticsearch import ElasticsearchRepository

    repo = ElasticsearchRepository(config)

    # Delete test documents before test
    try:
        client = repo._get_client()
        client.delete_by_query(
            index=INDEX_NAME,
            body={"query": {"prefix": {"scope": "test-"}}},
            refresh=True,
            ignore=[404],
        )
    except Exception:
        pass

    yield repo

    # Delete test documents after test
    try:
        client = repo._get_client()
        client.delete_by_query(
            index=INDEX_NAME,
            body={"query": {"prefix": {"scope": "test-"}}},
            refresh=True,
            ignore=[404],
        )
    except Exception:
        pass

    repo.close()


@pytest.fixture
def sample_element() -> CodeElement:
    """Create a sample code element for testing."""
    return CodeElement(
        element_id="test-es:test-repo:main:src/app.py:function:process:10",
        scope="test-es",
        repository="test-repo",
        username="main",
        relative_path="src/app.py",
        element_type="function",
        name="process",
        language="python",
        line_start=10,
        line_end=25,
        raw_code="def process(data):\n    return data.upper()",
        signature="def process(data)",
        docstring="Process input data.",
        decorators=["staticmethod"],
        is_async=False,
        visibility="public",
        level=2,
        parent_id=None,
    )


@pytest.fixture
def multiple_elements() -> list[CodeElement]:
    """Create multiple code elements for search testing."""
    return [
        CodeElement(
            element_id="test-es:test-repo:main:src/utils.py:file:utils.py:1",
            scope="test-es",
            repository="test-repo",
            username="main",
            relative_path="src/utils.py",
            element_type="file",
            name="utils.py",
            language="python",
            line_start=1,
            line_end=100,
            level=0,
        ),
        CodeElement(
            element_id="test-es:test-repo:main:src/utils.py:function:calculate:10",
            scope="test-es",
            repository="test-repo",
            username="main",
            relative_path="src/utils.py",
            element_type="function",
            name="calculate",
            language="python",
            line_start=10,
            line_end=20,
            raw_code="def calculate(x, y):\n    return x + y",
            signature="def calculate(x, y)",
            docstring="Calculate the sum of two numbers.",
            level=2,
        ),
        CodeElement(
            element_id="test-es:test-repo:main:src/utils.py:function:validate:30",
            scope="test-es",
            repository="test-repo",
            username="main",
            relative_path="src/utils.py",
            element_type="function",
            name="validate",
            language="python",
            line_start=30,
            line_end=45,
            raw_code="def validate(input):\n    return bool(input)",
            signature="def validate(input)",
            docstring="Validate the input data is not empty.",
            level=2,
        ),
        CodeElement(
            element_id="test-es:test-repo:main:src/models.py:class:User:1",
            scope="test-es",
            repository="test-repo",
            username="main",
            relative_path="src/models.py",
            element_type="class",
            name="User",
            language="python",
            line_start=1,
            line_end=50,
            raw_code="class User:\n    def __init__(self, name):\n        self.name = name",
            docstring="User model for authentication.",
            level=1,
        ),
    ]


# =============================================================================
# BASIC OPERATIONS TESTS
# =============================================================================


class TestElasticsearchRepository:
    """Tests for basic ES repository operations."""

    def test_index_creates_index_if_missing(self, es_repo):
        """Test that indexing creates the index if it doesn't exist."""
        client = es_repo._get_client()
        assert client.indices.exists(index=INDEX_NAME)

    def test_index_and_get_element(self, es_repo, sample_element):
        """Test indexing and retrieving an element."""
        result = es_repo.index_element(sample_element)
        assert result is True

        # ES needs a refresh to make the document searchable
        es_repo._get_client().indices.refresh(index=INDEX_NAME)

        doc = es_repo.get_document(sample_element.element_id)

        assert doc is not None
        assert doc["element_id"] == sample_element.element_id
        assert doc["name"] == "process"
        assert doc["element_type"] == "function"
        assert doc["language"] == "python"

    def test_get_nonexistent_document(self, es_repo):
        """Test getting a document that doesn't exist."""
        doc = es_repo.get_document("nonexistent-element-id")
        assert doc is None

    def test_delete_by_file(self, es_repo, multiple_elements):
        """Test deleting all documents for a file."""
        # Index elements
        for elem in multiple_elements:
            es_repo.index_element(elem)

        es_repo._get_client().indices.refresh(index=INDEX_NAME)

        # Delete elements from utils.py
        count = es_repo.delete_by_file(
            "test-es", "test-repo", "main", "src/utils.py"
        )

        # Should delete file + 2 functions = 3 elements
        assert count == 3

        # Verify deletion
        es_repo._get_client().indices.refresh(index=INDEX_NAME)
        doc = es_repo.get_document(multiple_elements[1].element_id)
        assert doc is None

        # User class should still exist
        doc = es_repo.get_document(multiple_elements[3].element_id)
        assert doc is not None

    def test_delete_by_repository(self, es_repo, multiple_elements):
        """Test deleting all documents for a repository/user combination."""
        # Index elements
        for elem in multiple_elements:
            es_repo.index_element(elem)

        es_repo._get_client().indices.refresh(index=INDEX_NAME)

        # Verify elements exist before deletion
        doc = es_repo.get_document(multiple_elements[0].element_id)
        assert doc is not None
        doc = es_repo.get_document(multiple_elements[3].element_id)
        assert doc is not None

        # Delete all elements for test-es:test-repo:main
        count = es_repo.delete_by_repository("test-es", "test-repo", "main")

        # Should delete all 4 elements
        assert count == 4

        # Verify all documents are deleted
        es_repo._get_client().indices.refresh(index=INDEX_NAME)
        for elem in multiple_elements:
            doc = es_repo.get_document(elem.element_id)
            assert doc is None

    def test_delete_by_repository_only_deletes_matching(self, es_repo):
        """Test that delete_by_repository only deletes matching scope/repo/user."""
        # Create elements for different repos/users
        elem1 = CodeElement(
            element_id="test-del:repo-a:user1:src/a.py:function:func1:1",
            scope="test-del",
            repository="repo-a",
            username="user1",
            relative_path="src/a.py",
            element_type="function",
            name="func1",
            language="python",
            line_start=1,
            line_end=10,
            level=2,
        )
        elem2 = CodeElement(
            element_id="test-del:repo-a:user2:src/b.py:function:func2:1",
            scope="test-del",
            repository="repo-a",
            username="user2",
            relative_path="src/b.py",
            element_type="function",
            name="func2",
            language="python",
            line_start=1,
            line_end=10,
            level=2,
        )
        elem3 = CodeElement(
            element_id="test-del:repo-b:user1:src/c.py:function:func3:1",
            scope="test-del",
            repository="repo-b",
            username="user1",
            relative_path="src/c.py",
            element_type="function",
            name="func3",
            language="python",
            line_start=1,
            line_end=10,
            level=2,
        )

        es_repo.index_element(elem1)
        es_repo.index_element(elem2)
        es_repo.index_element(elem3)
        es_repo._get_client().indices.refresh(index=INDEX_NAME)

        # Delete only repo-a:user1
        count = es_repo.delete_by_repository("test-del", "repo-a", "user1")
        assert count == 1

        es_repo._get_client().indices.refresh(index=INDEX_NAME)

        # elem1 should be deleted
        assert es_repo.get_document(elem1.element_id) is None

        # elem2 (different user) and elem3 (different repo) should still exist
        assert es_repo.get_document(elem2.element_id) is not None
        assert es_repo.get_document(elem3.element_id) is not None

        # Cleanup
        es_repo.delete_by_repository("test-del", "repo-a", "user2")
        es_repo.delete_by_repository("test-del", "repo-b", "user1")


# =============================================================================
# SUMMARY AND EMBEDDING STORAGE TESTS
# =============================================================================


class TestElasticsearchSummaryAndEmbedding:
    """Tests for summary and embedding storage in ES."""

    def test_store_and_get_summary(self, es_repo, sample_element):
        """Test storing and retrieving summaries."""
        es_repo.index_element(sample_element)
        es_repo._get_client().indices.refresh(index=INDEX_NAME)

        result = es_repo.store_summary(
            sample_element.element_id,
            "Processes input data by converting to uppercase."
        )
        assert result is True

        es_repo._get_client().indices.refresh(index=INDEX_NAME)

        doc = es_repo.get_document(sample_element.element_id)
        assert doc["summary"] == "Processes input data by converting to uppercase."

    def test_store_and_get_embedding(self, es_repo, sample_element):
        """Test storing and retrieving embedding vectors."""
        es_repo.index_element(sample_element)
        es_repo._get_client().indices.refresh(index=INDEX_NAME)

        # Create a 1024-dimension vector (matching index mapping)
        embedding = [0.1] * 1024

        result = es_repo.store_embedding(sample_element.element_id, embedding)
        assert result is True

        es_repo._get_client().indices.refresh(index=INDEX_NAME)

        retrieved = es_repo.get_embedding(sample_element.element_id)
        assert retrieved is not None
        assert len(retrieved) == 1024
        assert retrieved[0] == pytest.approx(0.1, rel=1e-5)

    def test_store_embedding_nonexistent_element(self, es_repo):
        """Test storing embedding for nonexistent element."""
        embedding = [0.1] * 1024
        result = es_repo.store_embedding("nonexistent-id", embedding)
        assert result is False


# =============================================================================
# TEXT SEARCH TESTS
# =============================================================================


class TestElasticsearchTextSearch:
    """Tests for text-based search."""

    def test_search_by_name(self, es_repo, multiple_elements):
        """Test searching by element name."""
        for elem in multiple_elements:
            es_repo.index_element(elem)
        es_repo._get_client().indices.refresh(index=INDEX_NAME)

        results = es_repo.search_by_text("calculate")

        assert len(results) >= 1
        assert any(r["name"] == "calculate" for r in results)

    def test_search_by_docstring(self, es_repo, multiple_elements):
        """Test searching by docstring content."""
        for elem in multiple_elements:
            es_repo.index_element(elem)
        es_repo._get_client().indices.refresh(index=INDEX_NAME)

        results = es_repo.search_by_text("authentication")

        assert len(results) >= 1
        assert any(r["name"] == "User" for r in results)

    def test_search_with_scope_filter(self, es_repo, multiple_elements):
        """Test searching with scope filter."""
        for elem in multiple_elements:
            es_repo.index_element(elem)
        es_repo._get_client().indices.refresh(index=INDEX_NAME)

        # Search with matching scope
        results = es_repo.search_by_text("validate", scope="test-es")
        assert len(results) >= 1

        # Search with non-matching scope
        results = es_repo.search_by_text("validate", scope="other-scope")
        assert len(results) == 0

    def test_search_with_type_filter(self, es_repo, multiple_elements):
        """Test searching with element type filter."""
        for elem in multiple_elements:
            es_repo.index_element(elem)
        es_repo._get_client().indices.refresh(index=INDEX_NAME)

        # Search only for classes
        results = es_repo.search_by_text("User", element_types=["class"])
        assert len(results) >= 1
        assert all(r["element_type"] == "class" for r in results)

        # Search only for functions - should not find User class
        results = es_repo.search_by_text("User", element_types=["function"])
        assert not any(r["name"] == "User" for r in results)


# =============================================================================
# VECTOR SEARCH TESTS
# =============================================================================


class TestElasticsearchVectorSearch:
    """Tests for vector-based semantic search."""

    def test_vector_search(self, es_repo, multiple_elements):
        """Test searching by vector similarity."""
        # Index elements with embeddings
        for i, elem in enumerate(multiple_elements):
            es_repo.index_element(elem)
            # Give each element a slightly different embedding
            embedding = [0.1 + (i * 0.1)] * 1024
            es_repo.store_embedding(elem.element_id, embedding)

        es_repo._get_client().indices.refresh(index=INDEX_NAME)

        # Search with a query vector
        query_vector = [0.1] * 1024
        results = es_repo.search_by_vector(query_vector, min_score=0.5)

        # Should return results
        assert len(results) >= 1
        # Results should have scores
        assert "_score" in results[0]
        # All indexed elements should be findable
        result_ids = {r["element_id"] for r in results}
        assert any(elem.element_id in result_ids for elem in multiple_elements)

    def test_vector_search_with_filters(self, es_repo, multiple_elements):
        """Test vector search with additional filters."""
        for i, elem in enumerate(multiple_elements):
            es_repo.index_element(elem)
            embedding = [0.5] * 1024
            es_repo.store_embedding(elem.element_id, embedding)

        es_repo._get_client().indices.refresh(index=INDEX_NAME)

        query_vector = [0.5] * 1024

        # Filter by element type
        results = es_repo.search_by_vector(
            query_vector,
            element_types=["function"],
            min_score=0.5
        )

        assert len(results) >= 1
        assert all(r["element_type"] == "function" for r in results)


# =============================================================================
# EMBEDDING STORE TESTS
# =============================================================================


class TestElasticsearchEmbeddingStore:
    """Tests for ElasticsearchEmbeddingStore with MySQL integration."""

    def test_store_element_indexes_to_es(self, config):
        """Test that store_element indexes to Elasticsearch."""
        from magaldi.db.elasticsearch import ElasticsearchEmbeddingStore

        store = ElasticsearchEmbeddingStore(config)

        elem = CodeElement(
            element_id="test-es-store:test-repo:main:src/test.py:function:test_func:1",
            scope="test-es-store",
            repository="test-repo",
            username="main",
            relative_path="src/test.py",
            element_type="function",
            name="test_func",
            language="python",
            line_start=1,
            line_end=10,
            level=2,
        )

        store.store_element(elem)
        store._get_client().indices.refresh(index=INDEX_NAME)

        # Verify it's in ES
        doc = store.get_document(elem.element_id)
        assert doc is not None
        assert doc["name"] == "test_func"

        # Cleanup
        store.delete_by_file("test-es-store", "test-repo", "main", "src/test.py")
        store.close()
