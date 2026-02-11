"""Integration tests for Elasticsearch embedding repository.

Tests for summary and embedding storage operations.
"""

from __future__ import annotations

import pytest

from magaldi_core.code_parser import CodeElement
from shared.db.store import INDEX_NAME

pytestmark = pytest.mark.integration


class TestElasticsearchSummaryAndEmbedding:
    """Tests for summary and embedding storage in ES."""

    def test_store_and_get_summary(self, repo, sample_element):
        """Test storing and retrieving summaries."""
        repo.index_element(sample_element)
        repo._get_client().indices_refresh(index=INDEX_NAME)

        result = repo.store_summary(
            sample_element.element_id,
            "Processes input data by converting to uppercase."
        )
        assert result is True

        repo._get_client().indices_refresh(index=INDEX_NAME)

        doc = repo.get_document(sample_element.element_id)
        assert doc["summary"] == "Processes input data by converting to uppercase."

    def test_store_and_get_embedding(self, repo, sample_element):
        """Test storing and retrieving embedding vectors."""
        repo.index_element(sample_element)
        repo._get_client().indices_refresh(index=INDEX_NAME)

        # Create a 1024-dimension vector (matching index mapping)
        embedding = [0.1] * 1024

        result = repo.store_embedding(sample_element.element_id, embedding)
        assert result is True

        repo._get_client().indices_refresh(index=INDEX_NAME)

        retrieved = repo.get_embedding(sample_element.element_id)
        assert retrieved is not None
        assert len(retrieved) == 1024
        assert retrieved[0] == pytest.approx(0.1, rel=1e-5)

    def test_store_embedding_nonexistent_element(self, repo):
        """Test storing embedding for nonexistent element."""
        embedding = [0.1] * 1024
        result = repo.store_embedding("nonexistent-id", embedding)
        assert result is False

    # =========================================================================
    # DUAL EMBEDDING TESTS (summary_embedding + code_embedding)
    # =========================================================================

    def test_store_summary_embedding(self, repo, sample_element):
        """Test storing summary embedding with explicit embedding_type."""
        repo.index_element(sample_element)
        repo._get_client().indices_refresh(index=INDEX_NAME)

        embedding = [0.2] * 1024
        result = repo.store_embedding(
            sample_element.element_id, embedding, embedding_type="summary"
        )
        assert result is True

        repo._get_client().indices_refresh(index=INDEX_NAME)

        retrieved = repo.get_embedding(sample_element.element_id, embedding_type="summary")
        assert retrieved is not None
        assert len(retrieved) == 1024
        assert retrieved[0] == pytest.approx(0.2, rel=1e-5)

    def test_store_code_embedding(self, repo, sample_element):
        """Test storing code embedding with embedding_type='code'."""
        repo.index_element(sample_element)
        repo._get_client().indices_refresh(index=INDEX_NAME)

        embedding = [0.3] * 1024
        result = repo.store_embedding(
            sample_element.element_id, embedding, embedding_type="code"
        )
        assert result is True

        repo._get_client().indices_refresh(index=INDEX_NAME)

        retrieved = repo.get_embedding(sample_element.element_id, embedding_type="code")
        assert retrieved is not None
        assert len(retrieved) == 1024
        assert retrieved[0] == pytest.approx(0.3, rel=1e-5)

    def test_store_both_embeddings_independently(self, repo, sample_element):
        """Test storing both summary and code embeddings independently."""
        repo.index_element(sample_element)
        repo._get_client().indices_refresh(index=INDEX_NAME)

        summary_embedding = [0.4] * 1024
        code_embedding = [0.5] * 1024

        # Store both embeddings
        repo.store_embedding(sample_element.element_id, summary_embedding, embedding_type="summary")
        repo.store_embedding(sample_element.element_id, code_embedding, embedding_type="code")
        repo._get_client().indices_refresh(index=INDEX_NAME)

        # Retrieve and verify both
        retrieved_summary = repo.get_embedding(sample_element.element_id, embedding_type="summary")
        retrieved_code = repo.get_embedding(sample_element.element_id, embedding_type="code")

        assert retrieved_summary is not None
        assert retrieved_code is not None
        assert retrieved_summary[0] == pytest.approx(0.4, rel=1e-5)
        assert retrieved_code[0] == pytest.approx(0.5, rel=1e-5)

    def test_store_summary_embedding_convenience_wrapper(self, repo, sample_element):
        """Test store_summary_embedding convenience method."""
        repo.index_element(sample_element)
        repo._get_client().indices_refresh(index=INDEX_NAME)

        embedding = [0.6] * 1024
        result = repo.store_summary_embedding(sample_element.element_id, embedding)
        assert result is True

        repo._get_client().indices_refresh(index=INDEX_NAME)

        retrieved = repo.get_embedding(sample_element.element_id, embedding_type="summary")
        assert retrieved is not None
        assert retrieved[0] == pytest.approx(0.6, rel=1e-5)

    def test_store_code_embedding_convenience_wrapper(self, repo, sample_element):
        """Test store_code_embedding convenience method."""
        repo.index_element(sample_element)
        repo._get_client().indices_refresh(index=INDEX_NAME)

        embedding = [0.7] * 1024
        result = repo.store_code_embedding(sample_element.element_id, embedding)
        assert result is True

        repo._get_client().indices_refresh(index=INDEX_NAME)

        retrieved = repo.get_embedding(sample_element.element_id, embedding_type="code")
        assert retrieved is not None
        assert retrieved[0] == pytest.approx(0.7, rel=1e-5)

    def test_default_embedding_type_is_summary(self, repo, sample_element):
        """Test that default embedding_type is 'summary' for backwards compatibility."""
        repo.index_element(sample_element)
        repo._get_client().indices_refresh(index=INDEX_NAME)

        embedding = [0.8] * 1024

        # Store without specifying type (should default to summary)
        repo.store_embedding(sample_element.element_id, embedding)
        repo._get_client().indices_refresh(index=INDEX_NAME)

        # Should be retrievable as summary embedding
        retrieved = repo.get_embedding(sample_element.element_id, embedding_type="summary")
        assert retrieved is not None
        assert retrieved[0] == pytest.approx(0.8, rel=1e-5)

        # Should also work with default get_embedding (no type specified)
        retrieved_default = repo.get_embedding(sample_element.element_id)
        assert retrieved_default is not None
        assert retrieved_default[0] == pytest.approx(0.8, rel=1e-5)


class TestElasticsearchEmbeddingStore:
    """Tests for ElasticsearchEmbeddingStore with MySQL integration."""

    def test_store_element_indexes_to_es(self, config):
        """Test that store_element indexes to Elasticsearch."""
        from shared.db.store import ElasticsearchEmbeddingStore

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
        store._get_client().indices_refresh(index=INDEX_NAME)

        # Verify it's in ES
        doc = store.get_document(elem.element_id)
        assert doc is not None
        assert doc["name"] == "test_func"

        # Cleanup
        store.delete_by_file("test-es-store", "test-repo", "main", "src/test.py")
        store.close()
