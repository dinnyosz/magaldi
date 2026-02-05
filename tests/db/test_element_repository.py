"""Integration tests for Elasticsearch element repository.

Tests for basic CRUD operations: index, get, delete operations.
"""

from __future__ import annotations

import pytest

from magaldi_core.code_parser import CodeElement
from shared.db.elasticsearch import INDEX_NAME

pytestmark = pytest.mark.integration


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


class TestIsTestIndexing:
    """Tests for is_test field indexing."""

    def test_indexes_is_test_field(self, es_repo, sample_element):
        """Test that is_test field is indexed."""
        sample_element.is_test = True
        es_repo.index_element(sample_element)

        es_repo._get_client().indices.refresh(index=INDEX_NAME)

        doc = es_repo.get_document(sample_element.element_id)
        assert doc is not None
        assert doc.get("is_test") is True

    def test_is_test_defaults_to_false(self, es_repo, sample_element):
        """Test that is_test defaults to False."""
        sample_element.is_test = False
        es_repo.index_element(sample_element)

        es_repo._get_client().indices.refresh(index=INDEX_NAME)

        doc = es_repo.get_document(sample_element.element_id)
        assert doc is not None
        assert doc.get("is_test") is False


class TestInterruptedRunDetection:
    """Tests for detecting and handling interrupted processing runs."""

    def test_incomplete_file_detected_when_element_count_mismatch(self, config):
        """Test that files with mismatched element_count are detected as incomplete."""
        from shared.db.elasticsearch import (
            ElasticsearchFileStateRepository,
            ElasticsearchRepository,
        )

        es_repo = ElasticsearchRepository(config)
        file_state_repo = ElasticsearchFileStateRepository(config)

        scope = "test-interrupted"
        repo = "test-repo"
        username = "main"
        file_path = "src/test_module.py"
        file_hash = "abc123hash"

        # Create a FILE element with element_count=4 (expecting FILE + 3 children = 4 total)
        # But we'll only index FILE + 2 children = 3 elements (simulating interrupted run)
        file_elem = CodeElement(
            element_id=f"{scope}:{repo}:{username}:{file_path}:file:{file_path}:1",
            scope=scope,
            repository=repo,
            username=username,
            relative_path=file_path,
            element_type="file",
            name=file_path,
            language="python",
            line_start=1,
            line_end=100,
            level=0,
        )
        es_repo.index_element(file_elem, file_hash=file_hash, element_count=4)

        # Create only 2 child elements (3 total with FILE), but expected 4
        func1 = CodeElement(
            element_id=f"{scope}:{repo}:{username}:{file_path}:function:func1:10",
            scope=scope,
            repository=repo,
            username=username,
            relative_path=file_path,
            element_type="function",
            name="func1",
            language="python",
            line_start=10,
            line_end=20,
            level=2,
        )
        es_repo.index_element(func1, file_hash=file_hash)

        func2 = CodeElement(
            element_id=f"{scope}:{repo}:{username}:{file_path}:function:func2:25",
            scope=scope,
            repository=repo,
            username=username,
            relative_path=file_path,
            element_type="function",
            name="func2",
            language="python",
            line_start=25,
            line_end=35,
            level=2,
        )
        es_repo.index_element(func2, file_hash=file_hash)

        es_repo._get_client().indices.refresh(index=INDEX_NAME)

        # Get file states - incomplete file should have file_hash=None
        file_states = file_state_repo.get_file_states(scope, repo, username)

        # The file should be detected as incomplete (element_count=4 but only 3 elements)
        assert file_path in file_states
        state = file_states[file_path]
        # file_hash should be None since actual count (3) != expected count (4)
        assert state.file_hash is None

        # Cleanup
        es_repo.delete_by_file(scope, repo, username, file_path)
        es_repo.close()
        file_state_repo.close()

    def test_complete_file_has_valid_hash(self, config):
        """Test that files with matching element_count are detected as complete."""
        from shared.db.elasticsearch import (
            ElasticsearchFileStateRepository,
            ElasticsearchRepository,
        )

        es_repo = ElasticsearchRepository(config)
        file_state_repo = ElasticsearchFileStateRepository(config)

        scope = "test-complete"
        repo = "test-repo"
        username = "main"
        file_path = "src/complete_module.py"
        file_hash = "def456hash"

        # Create a FILE element with element_count=3 (FILE + 2 children = 3 total)
        file_elem = CodeElement(
            element_id=f"{scope}:{repo}:{username}:{file_path}:file:{file_path}:1",
            scope=scope,
            repository=repo,
            username=username,
            relative_path=file_path,
            element_type="file",
            name=file_path,
            language="python",
            line_start=1,
            line_end=50,
            level=0,
        )
        es_repo.index_element(file_elem, file_hash=file_hash, element_count=3)

        # Create exactly 2 child elements (3 total with FILE, matching element_count)
        func1 = CodeElement(
            element_id=f"{scope}:{repo}:{username}:{file_path}:function:func1:10",
            scope=scope,
            repository=repo,
            username=username,
            relative_path=file_path,
            element_type="function",
            name="func1",
            language="python",
            line_start=10,
            line_end=20,
            level=2,
        )
        es_repo.index_element(func1, file_hash=file_hash)

        func2 = CodeElement(
            element_id=f"{scope}:{repo}:{username}:{file_path}:function:func2:25",
            scope=scope,
            repository=repo,
            username=username,
            relative_path=file_path,
            element_type="function",
            name="func2",
            language="python",
            line_start=25,
            line_end=35,
            level=2,
        )
        es_repo.index_element(func2, file_hash=file_hash)

        es_repo._get_client().indices.refresh(index=INDEX_NAME)

        # Get file states - complete file should have valid file_hash
        file_states = file_state_repo.get_file_states(scope, repo, username)

        # The file should be detected as complete
        assert file_path in file_states
        state = file_states[file_path]
        # file_hash should be preserved since actual count (3) == expected count (3)
        assert state.file_hash == file_hash

        # Cleanup
        es_repo.delete_by_file(scope, repo, username, file_path)
        es_repo.close()
        file_state_repo.close()


class TestOldDataHandling:
    """Tests for handling old data without element_count."""

    def test_old_data_without_element_count_treated_as_incomplete(self, config):
        """Test that files without element_count (old data) are treated as incomplete."""
        from shared.db.elasticsearch import (
            ElasticsearchFileStateRepository,
            ElasticsearchRepository,
        )

        es_repo = ElasticsearchRepository(config)
        file_state_repo = ElasticsearchFileStateRepository(config)

        scope = "test-old-data"
        repo = "test-repo"
        username = "main"
        file_path = "src/old_module.py"
        file_hash = "ghi789hash"

        # Create a FILE element WITHOUT element_count (simulating old data)
        file_elem = CodeElement(
            element_id=f"{scope}:{repo}:{username}:{file_path}:file:{file_path}:1",
            scope=scope,
            repository=repo,
            username=username,
            relative_path=file_path,
            element_type="file",
            name=file_path,
            language="python",
            line_start=1,
            line_end=50,
            level=0,
        )
        # Index without element_count
        es_repo.index_element(file_elem, file_hash=file_hash, element_count=None)

        es_repo._get_client().indices.refresh(index=INDEX_NAME)

        # Get file states - old data should be treated as incomplete
        file_states = file_state_repo.get_file_states(scope, repo, username)

        # The file should be detected as incomplete (no element_count)
        assert file_path in file_states
        state = file_states[file_path]
        # file_hash should be None since element_count is None
        assert state.file_hash is None

        # Cleanup
        es_repo.delete_by_file(scope, repo, username, file_path)
        es_repo.close()
        file_state_repo.close()
