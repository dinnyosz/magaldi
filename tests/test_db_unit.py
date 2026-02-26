"""Unit tests for repository (mocked, no ES required)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from magaldi_core.code_parser import CodeElement
from shared.db.backends.base import NotFoundError
from shared.db.store import (
    Repository,
    generate_hash_id,
)

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_config():
    """Create a mock configuration."""
    config = MagicMock()
    config.search_backend.host = "localhost"
    config.search_backend.port = 9200
    config.search_backend.scheme = "http"
    return config


@pytest.fixture
def mock_client():
    """Create a mock search client."""
    client = MagicMock()
    client.indices.exists.return_value = True
    return client


@pytest.fixture
def mock_repo(mock_config, mock_client):
    """Create repository with mocked ES client."""
    with patch("shared.db.backends.factory.create_client", return_value=mock_client):
        repo = Repository(mock_config)
        repo._base._client = mock_client
        return repo


@pytest.fixture
def sample_element():
    """Create a sample CodeElement for testing."""
    return CodeElement(
        element_id="scope:repo:user:file.py:function:test:1",
        scope="scope",
        repository="repo",
        username="user",
        relative_path="file.py",
        element_type="function",
        name="test",
        language="python",
        line_start=1,
        line_end=10,
        raw_code="def test(): pass",
        level=2,
    )


# =============================================================================
# HASH ID TESTS
# =============================================================================


class TestGenerateHashId:
    """Tests for generate_hash_id function."""

    def test_generates_consistent_hash(self):
        """Test that same element_id produces same hash."""
        element_id = "scope:repo:user:file.py:function:test:1"

        hash1 = generate_hash_id(element_id)
        hash2 = generate_hash_id(element_id)

        assert hash1 == hash2
        assert len(hash1) == 64  # Full SHA256 hex digest

    def test_different_ids_different_hashes(self):
        """Test that different element_ids produce different hashes."""
        hash1 = generate_hash_id("scope:repo:user:file.py:function:test:1")
        hash2 = generate_hash_id("scope:repo:user:file.py:function:other:1")

        assert hash1 != hash2

    def test_hash_length(self):
        """Test hash is exactly 64 characters (SHA256)."""
        hash_id = generate_hash_id("any:element:id")
        assert len(hash_id) == 64


# =============================================================================
# REPOSITORY INIT TESTS
# =============================================================================


class TestRepositoryInit:
    """Tests for Repository initialization."""

    @patch("shared.db.backends.factory.create_client")
    def test_init_with_config(self, mock_client_factory, mock_config):
        """Test repository initialization with config."""
        mock_client_factory.return_value = MagicMock()

        repo = Repository(mock_config)

        assert repo._config == mock_config


# =============================================================================
# INDEX ELEMENT TESTS
# =============================================================================


class TestIndexElement:
    """Tests for index_element method."""

    def test_index_element_success(self, mock_repo, mock_client, sample_element):
        """Test successful element indexing."""
        mock_client.index_document.return_value = {"result": "created"}

        result = mock_repo.index_element(sample_element)

        assert result is True
        mock_client.index_document.assert_called_once()

    def test_index_element_with_file_hash(self, mock_repo, mock_client, sample_element):
        """Test indexing element with file hash."""
        mock_client.index_document.return_value = {"result": "created"}

        result = mock_repo.index_element(
            sample_element,
            file_hash="abc123",
        )

        assert result is True
        call_args = mock_client.index_document.call_args
        doc = call_args[0][2]  # 3rd positional arg: (index, doc_id, document)
        assert doc["file_hash"] == "abc123"


# =============================================================================
# GET DOCUMENT TESTS
# =============================================================================


class TestGetDocument:
    """Tests for get_document method."""

    def test_get_document_found(self, mock_repo, mock_client):
        """Test getting existing document."""
        mock_client.get_document.return_value = {
            "_source": {"name": "test", "element_id": "test_id"}
        }

        result = mock_repo.get_document("test_id")

        assert result is not None
        assert result["name"] == "test"

    def test_get_document_not_found(self, mock_repo, mock_client):
        """Test getting non-existent document."""
        mock_client.get_document.side_effect = NotFoundError("not found")

        result = mock_repo.get_document("nonexistent")

        assert result is None


class TestGetDocumentByHashId:
    """Tests for get_document_by_hash_id method."""

    def test_get_by_hash_found(self, mock_repo, mock_client):
        """Test getting document by hash ID."""
        mock_client.search.return_value = {
            "hits": {
                "total": {"value": 1},
                "hits": [{"_source": {"name": "test", "hash_id": "abc123"}}],
            }
        }

        result = mock_repo.get_document_by_hash_id("abc123")

        assert result is not None

    def test_get_by_hash_not_found(self, mock_repo, mock_client):
        """Test getting non-existent document by hash."""
        mock_client.search.return_value = {
            "hits": {"total": {"value": 0}, "hits": []}
        }

        result = mock_repo.get_document_by_hash_id("nonexistent")

        assert result is None


# =============================================================================
# ELEMENT EXISTS TESTS
# =============================================================================


class TestElementExists:
    """Tests for element_exists method."""

    def test_element_exists_true(self, mock_repo, mock_client):
        """Test element exists returns true."""
        mock_client.exists_document.return_value = True

        result = mock_repo.element_exists("test_id")

        assert result is True

    def test_element_exists_false(self, mock_repo, mock_client):
        """Test element exists returns false."""
        mock_client.exists_document.return_value = False

        result = mock_repo.element_exists("nonexistent")

        assert result is False


# =============================================================================
# GET EXISTING ELEMENT IDS TESTS
# =============================================================================


class TestGetExistingElementIds:
    """Tests for get_existing_element_ids method."""

    def test_get_existing_empty_list(self, mock_repo):
        """Test with empty list returns empty set."""
        result = mock_repo.get_existing_element_ids([])

        assert result == set()



# =============================================================================
# STORE EMBEDDING TESTS
# =============================================================================


class TestStoreEmbedding:
    """Tests for store_embedding method."""

    def test_store_embedding_success(self, mock_repo, mock_client):
        """Test successful embedding storage."""
        mock_client.update_document.return_value = {"result": "updated"}

        result = mock_repo.store_embedding("test_id", [0.1] * 1024)

        assert result is True
        mock_client.update_document.assert_called_once()


# =============================================================================
# STORE SUMMARY TESTS
# =============================================================================


class TestStoreSummary:
    """Tests for store_summary method."""

    def test_store_summary_success(self, mock_repo, mock_client):
        """Test successful summary storage."""
        mock_client.update_document.return_value = {"result": "updated"}

        result = mock_repo.store_summary("test_id", "Test summary")

        assert result is True


# =============================================================================
# SEARCH BY TEXT TESTS
# =============================================================================


class TestSearchByText:
    """Tests for search_by_text method."""

    def test_search_by_text_with_results(self, mock_repo, mock_client):
        """Test text search with results."""
        mock_client.search.return_value = {
            "hits": {
                "total": {"value": 2},
                "hits": [
                    {"_source": {"name": "func1"}, "_score": 0.9},
                    {"_source": {"name": "func2"}, "_score": 0.8},
                ],
            }
        }

        result = mock_repo.search_by_text("test query")

        assert len(result) == 2

    def test_search_by_text_no_results(self, mock_repo, mock_client):
        """Test text search with no results."""
        mock_client.search.return_value = {
            "hits": {"total": {"value": 0}, "hits": []}
        }

        result = mock_repo.search_by_text("nonexistent")

        assert len(result) == 0


# =============================================================================
# DELETE ELEMENTS TESTS
# =============================================================================


class TestDeleteElements:
    """Tests for delete_elements method."""

    def test_delete_elements_empty_list(self, mock_repo):
        """Test deletion with empty list."""
        result = mock_repo.delete_elements([])

        assert result == 0


# =============================================================================
# DELETE BY REPOSITORY TESTS
# =============================================================================


class TestDeleteByRepository:
    """Tests for delete_by_repository method."""

    def test_delete_by_repository_success(self, mock_repo, mock_client):
        """Test successful repository deletion."""
        mock_client.delete_by_query.return_value = {"deleted": 100}

        result = mock_repo.delete_by_repository("scope", "repo", "user")

        assert result == 100


# =============================================================================
# SEARCH BY VECTOR TESTS
# =============================================================================


class TestSearchByVector:
    """Tests for search_by_vector method."""

    def test_search_by_vector_with_results(self, mock_repo, mock_client):
        """Test vector search with results."""
        mock_client.search.return_value = {
            "hits": {
                "hits": [
                    {"_source": {"name": "func1"}, "_score": 0.95},
                    {"_source": {"name": "func2"}, "_score": 0.85},
                ]
            }
        }

        result = mock_repo.search_by_vector([0.1] * 1024)

        assert len(result) == 2

    def test_search_by_vector_with_filters(self, mock_repo, mock_client):
        """Test vector search with filters."""
        mock_client.search.return_value = {
            "hits": {"hits": [{"_source": {"name": "func1"}, "_score": 0.9}]}
        }

        result = mock_repo.search_by_vector(
            [0.1] * 1024,
            scope="scope",
            repository="repo",
            element_types=["function"],
        )

        assert len(result) == 1


# =============================================================================
# GET ALL EMBEDDINGS TESTS
# =============================================================================


class TestGetAllEmbeddings:
    """Tests for get_all_embeddings method."""

    def test_get_all_embeddings_success(self, mock_repo, mock_client):
        """Test getting all embeddings."""
        mock_client.search.return_value = {
            "hits": {
                "hits": [
                    {"_source": {"element_id": "id1", "embedding": [0.1] * 1024}},
                    {"_source": {"element_id": "id2", "embedding": [0.2] * 1024}},
                ]
            },
            "_scroll_id": "scroll_1",
        }
        # Simulate scroll returning empty on second call
        mock_client.scroll.return_value = {"hits": {"hits": []}}
        mock_client.clear_scroll.return_value = {}

        result = mock_repo.get_all_embeddings("scope", "repo", "user")

        assert len(result) == 2




# =============================================================================
# CLOSE CONNECTION TESTS
# =============================================================================


class TestClose:
    """Tests for close method."""

    def test_close_connection(self, mock_repo, mock_client):
        """Test closing ES connection."""
        mock_repo.close()

        mock_client.close.assert_called_once()
        assert mock_repo._base._client is None

    def test_close_already_closed(self, mock_repo):
        """Test closing already closed connection."""
        mock_repo._base._client = None

        # Should not raise
        mock_repo.close()


# =============================================================================
# MAIN BRANCH EXISTS TESTS
# =============================================================================


class TestMainBranchExists:
    """Tests for main_branch_exists method."""

    def test_main_branch_exists_true(self, mock_repo, mock_client):
        """Test main branch exists returns true."""
        mock_client.count.return_value = {"count": 5}

        result = mock_repo.main_branch_exists("scope", "repo")

        assert result is True

    def test_main_branch_exists_false(self, mock_repo, mock_client):
        """Test main branch exists returns false."""
        mock_client.count.return_value = {"count": 0}

        result = mock_repo.main_branch_exists("scope", "repo")

        assert result is False


# =============================================================================
# GET ELEMENT CONTENT HASHES TESTS
# =============================================================================


class TestGetElementContentHashes:
    """Tests for get_element_content_hashes method."""

    def test_get_content_hashes_empty(self, mock_repo):
        """Test with empty list."""
        result = mock_repo.get_element_content_hashes([])

        assert result == {}

# =============================================================================
# GET EMBEDDING TESTS
# =============================================================================


class TestGetEmbedding:
    """Tests for get_embedding method."""

    def test_get_summary_embedding_found(self, mock_repo, mock_client):
        """Test getting summary embedding for existing element."""
        mock_client.get_document.return_value = {
            "_source": {"summary_embedding": [0.1] * 1024}
        }

        result = mock_repo.get_embedding("test_id", embedding_type="summary")

        assert result is not None
        assert len(result) == 1024

    def test_get_code_embedding_found(self, mock_repo, mock_client):
        """Test getting code embedding for existing element."""
        mock_client.get_document.return_value = {
            "_source": {"code_embedding": [0.2] * 1024}
        }

        result = mock_repo.get_embedding("test_id", embedding_type="code")

        assert result is not None
        assert len(result) == 1024
        assert result[0] == 0.2

    def test_get_embedding_default_is_summary(self, mock_repo, mock_client):
        """Test that default embedding_type is 'summary'."""
        mock_client.get_document.return_value = {
            "_source": {"summary_embedding": [0.3] * 1024}
        }

        result = mock_repo.get_embedding("test_id")

        assert result is not None
        assert result[0] == 0.3

    def test_get_embedding_not_found(self, mock_repo, mock_client):
        """Test getting embedding for non-existent element."""
        mock_client.get_document.side_effect = NotFoundError("not found")

        result = mock_repo.get_embedding("nonexistent")

        assert result is None


# =============================================================================
# DELETE BY FILE TESTS
# =============================================================================


class TestDeleteByFile:
    """Tests for delete_by_file method."""

    def test_delete_by_file_success(self, mock_repo, mock_client):
        """Test successful file deletion."""
        mock_client.delete_by_query.return_value = {"deleted": 10}

        result = mock_repo.delete_by_file("scope", "repo", "user", "file.py")

        assert result == 10


# =============================================================================
# CLEAR CLUSTER ASSIGNMENTS TESTS
# =============================================================================


class TestClearClusterAssignments:
    """Tests for clear_cluster_assignments method."""

    def test_clear_cluster_assignments_success(self, mock_repo, mock_client):
        """Test clearing cluster assignments."""
        mock_client.update_by_query.return_value = {"updated": 50}

        result = mock_repo.clear_cluster_assignments("scope", "repo", "user")

        assert result == 50


# =============================================================================
# UPDATE FILE HASHES TESTS
# =============================================================================


class TestUpdateFileHashes:
    """Tests for update_file_hashes method."""

    def test_update_file_hashes_empty(self, mock_repo):
        """Test with empty file_updates dict returns 0."""
        result = mock_repo.update_file_hashes("scope", "repo", "user", {})
        assert result == 0

    def test_update_file_hashes_single_file(self, mock_repo, mock_client):
        """Test updating file hash for a single file."""
        mock_client.update_by_query.return_value = {"updated": 5}

        result = mock_repo.update_file_hashes(
            "scope", "repo", "user",
            {"src/main.py": "newhash123"}
        )

        assert result == 5
        mock_client.update_by_query.assert_called_once()
        call_args = mock_client.update_by_query.call_args
        assert call_args[1]["body"]["script"]["params"]["file_hash"] == "newhash123"

    def test_update_file_hashes_multiple_files(self, mock_repo, mock_client):
        """Test updating file hashes for multiple files."""
        mock_client.update_by_query.return_value = {"updated": 3}

        result = mock_repo.update_file_hashes(
            "scope", "repo", "user",
            {"src/main.py": "hash1", "src/utils.py": "hash2"}
        )

        # 3 updated per file * 2 files = 6
        assert result == 6
        assert mock_client.update_by_query.call_count == 2
