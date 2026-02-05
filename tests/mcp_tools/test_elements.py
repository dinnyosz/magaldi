"""Tests for MCP tools - elements category."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from magaldi_mcp.tools import (
    batch_get_elements,
    find_similar,
    get_children,
    get_context,
    get_element,
)


# =============================================================================
# GET ELEMENT TESTS
# =============================================================================


class TestGetElement:
    """Tests for get_element function."""

    def test_get_element_returns_element(self, mock_es_repo):
        """Test get_element returns element details."""
        mock_es_repo.get_document.return_value = {
            "element_id": "scope:repo:user:file.py:function:test:1",
            "name": "test",
            "element_type": "function",
            "relative_path": "file.py",
            "line_start": 1,
            "line_end": 10,
            "raw_code": "def test(): pass",
            "summary": "A test function.",
        }

        result = get_element(
            es=mock_es_repo,
            element_id="scope:repo:user:file.py:function:test:1",
        )

        assert result is not None
        assert result["name"] == "test"

    def test_get_element_not_found_raises(self, mock_es_repo):
        """Test get_element raises when element not found."""
        mock_es_repo.get_document.return_value = None

        with pytest.raises(ValueError, match="Element not found"):
            get_element(
                es=mock_es_repo,
                element_id="nonexistent",
            )

    def test_get_element_with_code(self, mock_es_repo):
        """Test get_element includes code when requested."""
        mock_es_repo.get_document.return_value = {
            "element_id": "scope:repo:user:file.py:function:test:1",
            "name": "test",
            "element_type": "function",
            "raw_code": "def test(): pass",
        }

        result = get_element(
            es=mock_es_repo,
            element_id="scope:repo:user:file.py:function:test:1",
            include_code=True,
        )

        # Result uses 'code' key for raw_code
        assert "code" in result


# =============================================================================
# BATCH GET ELEMENTS TESTS
# =============================================================================

# =============================================================================
# BATCH GET ELEMENTS TESTS
# =============================================================================


class TestBatchGetElements:
    """Tests for batch_get_elements function."""

    def test_batch_get_returns_multiple_elements(self, mock_es_repo):
        """Test batch_get_elements returns multiple elements."""
        mock_es_repo.get_document.side_effect = [
            {"element_id": "id1", "name": "elem1"},
            {"element_id": "id2", "name": "elem2"},
        ]

        result = batch_get_elements(
            es=mock_es_repo,
            element_ids=["id1", "id2"],
        )

        assert isinstance(result, list)
        assert len(result) == 2


# =============================================================================
# LIST REPOS TESTS
# =============================================================================

# =============================================================================
# GET ELEMENT EXTENDED TESTS
# =============================================================================


class TestGetElementExtended:
    """Extended tests for get_element function."""

    def test_get_element_includes_optional_fields(self, mock_es_repo):
        """Test get_element includes all optional fields when brief=False."""
        mock_es_repo.get_document.return_value = {
            "element_id": "id1",
            "name": "test",
            "element_type": "function",
            "relative_path": "file.py",
            "line_start": 1,
            "line_end": 10,
            "signature": "def test():",
            "docstring": "A test function.",
            "decorators": ["@fixture"],
            "is_async": True,
            "parent_id": "parent_id",
        }

        result = get_element(es=mock_es_repo, element_id="id1", brief=False)

        assert result["signature"] == "def test():"
        assert result["docstring"] == "A test function."
        assert result["decorators"] == ["@fixture"]
        assert result["is_async"] is True
        assert result["parent_id"] == "parent_id"


# =============================================================================
# GET CONTEXT EXTENDED TESTS
# =============================================================================

# =============================================================================
# FIND SIMILAR TESTS
# =============================================================================


class TestFindSimilar:
    """Tests for find_similar function."""

    def test_find_similar_returns_similar_elements(self, mock_es_repo):
        """Test find_similar returns similar elements grouped by is_test."""
        mock_es_repo.get_document.return_value = {
            "element_id": "id1",
            "name": "test",
            "summary_embedding": [0.1] * 1024,
        }
        mock_es_repo.search_by_vector.return_value = [
            {"element_id": "id2", "name": "similar", "_score": 0.9}
        ]

        result = find_similar(
            es=mock_es_repo,
            element_id="id1",
        )

        assert isinstance(result, dict)
        assert "code_results" in result
        assert "test_results" in result
        assert len(result["code_results"]) == 1


# =============================================================================
# GET CONTEXT TESTS
# =============================================================================

# =============================================================================
# FIND SIMILAR EXTENDED TESTS
# =============================================================================


class TestFindSimilarExtended:
    """Extended tests for find_similar function."""

    def test_find_similar_element_not_found(self, mock_es_repo):
        """Test find_similar raises when element not found."""
        mock_es_repo.get_document.return_value = None

        with pytest.raises(ValueError, match="Element not found"):
            find_similar(es=mock_es_repo, element_id="nonexistent")

    def test_find_similar_no_embedding(self, mock_es_repo):
        """Test find_similar raises when element has no embedding."""
        mock_es_repo.get_document.return_value = {
            "element_id": "id1",
            "name": "test",
        }

        with pytest.raises(ValueError, match="no embedding"):
            find_similar(es=mock_es_repo, element_id="id1")

    def test_find_similar_excludes_self(self, mock_es_repo):
        """Test find_similar excludes the source element."""
        mock_es_repo.get_document.return_value = {
            "element_id": "id1",
            "name": "test",
            "summary_embedding": [0.1] * 1024,
        }
        mock_es_repo.search_by_vector.return_value = [
            {"element_id": "id1", "name": "test", "_score": 1.0},  # Self
            {"element_id": "id2", "name": "similar", "_score": 0.9},
        ]

        result = find_similar(es=mock_es_repo, element_id="id1", limit=1)

        assert len(result["code_results"]) == 1
        assert result["code_results"][0]["element_id"] == "id2"

    def test_find_similar_same_repo_only(self, mock_es_repo):
        """Test find_similar with same_repo_only filter."""
        mock_es_repo.get_document.return_value = {
            "element_id": "id1",
            "name": "test",
            "summary_embedding": [0.1] * 1024,
            "scope": "github",
            "repository": "myrepo",
        }
        mock_es_repo.search_by_vector.return_value = []

        result = find_similar(es=mock_es_repo, element_id="id1", same_repo_only=True)

        assert isinstance(result, dict)
        assert "code_results" in result
        assert "test_results" in result
        # Verify search_by_vector was called with scope/repo filters
        mock_es_repo.search_by_vector.assert_called_once()


# =============================================================================
# GET ELEMENT EXTENDED TESTS
# =============================================================================

# =============================================================================
# FIND SIMILAR TEST GROUPING TESTS
# =============================================================================


class TestFindSimilarTestGrouping:
    """Tests for find_similar test result grouping."""

    def test_groups_similar_by_is_test(self, mock_es_repo):
        """Test that similar results are grouped by is_test."""
        mock_es_repo.get_document.return_value = {
            "element_id": "id1",
            "summary_embedding": [0.1] * 1024,
        }
        mock_es_repo.search_by_vector.return_value = [
            {"element_id": "id2", "name": "similar_func", "is_test": False},
            {"element_id": "id3", "name": "test_similar", "is_test": True},
        ]

        result = find_similar(es=mock_es_repo, element_id="id1")

        assert "code_results" in result
        assert "test_results" in result

    def test_include_tests_false(self, mock_es_repo):
        """Test include_tests parameter."""
        mock_es_repo.get_document.return_value = {
            "element_id": "id1",
            "summary_embedding": [0.1] * 1024,
        }
        mock_es_repo.search_by_vector.return_value = [
            {"element_id": "id2", "name": "test_func", "is_test": True},
        ]

        result = find_similar(es=mock_es_repo, element_id="id1", include_tests=False)

        assert len(result["test_results"]) == 0

    def test_results_include_is_test_field(self, mock_es_repo):
        """Test that individual results include is_test field."""
        mock_es_repo.get_document.return_value = {
            "element_id": "id1",
            "summary_embedding": [0.1] * 1024,
        }
        mock_es_repo.search_by_vector.return_value = [
            {"element_id": "id2", "name": "similar_func", "is_test": False},
        ]

        result = find_similar(es=mock_es_repo, element_id="id1")

        assert len(result["code_results"]) == 1
        assert result["code_results"][0]["is_test"] is False

    def test_results_include_totals(self, mock_es_repo):
        """Test that results include total counts."""
        mock_es_repo.get_document.return_value = {
            "element_id": "id1",
            "summary_embedding": [0.1] * 1024,
        }
        mock_es_repo.search_by_vector.return_value = [
            {"element_id": "id2", "name": "similar_func", "is_test": False},
            {"element_id": "id3", "name": "test_similar", "is_test": True},
        ]

        result = find_similar(es=mock_es_repo, element_id="id1")

        assert "total_code" in result
        assert "total_tests" in result
        assert result["total_code"] == 1
        assert result["total_tests"] == 1


# =============================================================================
# SEARCH FEATURES GLOSSARY FILTER TESTS
# =============================================================================

# =============================================================================
# GET CONTEXT TESTS
# =============================================================================


class TestGetContext:
    """Tests for get_context function."""

    def test_get_context_returns_parent_and_children(self, mock_es_repo):
        """Test get_context returns context info."""
        mock_es_repo.get_document.return_value = {
            "element_id": "scope:repo:user:file.py:function:test:1",
            "name": "test",
            "parent_id": "scope:repo:user:file.py:class:MyClass:1",
        }
        mock_es_repo.get_children.return_value = []

        result = get_context(
            es=mock_es_repo,
            element_id="scope:repo:user:file.py:function:test:1",
        )

        assert result is not None


# =============================================================================
# GET CHILDREN TESTS
# =============================================================================

# =============================================================================
# GET CONTEXT EXTENDED TESTS
# =============================================================================


class TestGetContextExtended:
    """Extended tests for get_context function."""

    def test_get_context_with_file_and_parent(self, mock_es_repo):
        """Test get_context returns file and parent info."""
        mock_es_repo.get_document.side_effect = [
            # First call: the element
            {
                "element_id": "method_id",
                "name": "my_method",
                "element_type": "method",
                "relative_path": "file.py",
                "line_start": 10,
                "scope": "github",
                "repository": "repo",
                "username": "main",
                "parent_id": "class_id",
            },
            # Second call: the parent class
            {
                "element_id": "class_id",
                "name": "MyClass",
                "element_type": "class",
                "summary": "A class",
                "signature": "class MyClass:",
            },
        ]

        mock_client = MagicMock()
        mock_es_repo._get_client.return_value = mock_client

        # File search
        mock_client.search.return_value = {
            "hits": {
                "hits": [
                    {
                        "_source": {
                            "element_id": "file_id",
                            "name": "file.py",
                            "summary": "File summary",
                        }
                    }
                ]
            }
        }

        result = get_context(es=mock_es_repo, element_id="method_id")

        assert result["file"]["name"] == "file.py"
        assert result["parent"]["name"] == "MyClass"

    def test_get_context_with_siblings(self, mock_es_repo):
        """Test get_context returns siblings when requested."""
        mock_es_repo.get_document.side_effect = [
            # First call: the element
            {
                "element_id": "method1",
                "name": "method1",
                "element_type": "method",
                "relative_path": "file.py",
                "line_start": 10,
                "scope": "github",
                "repository": "repo",
                "username": "main",
                "parent_id": "class_id",
            },
            # Second call: the parent
            {
                "element_id": "class_id",
                "name": "MyClass",
                "element_type": "class",
            },
        ]

        mock_client = MagicMock()
        mock_es_repo._get_client.return_value = mock_client

        # Siblings search (children of parent)
        mock_client.search.return_value = {
            "hits": {
                "hits": [
                    {
                        "_source": {
                            "element_id": "method1",
                            "name": "method1",
                            "element_type": "method",
                            "line_start": 10,
                        }
                    },
                    {
                        "_source": {
                            "element_id": "method2",
                            "name": "method2",
                            "element_type": "method",
                            "line_start": 20,
                        }
                    },
                ]
            }
        }

        result = get_context(es=mock_es_repo, element_id="method1", include_siblings=True)

        # method1 should be excluded from siblings
        assert len(result["siblings"]) == 1
        assert result["siblings"][0]["name"] == "method2"


# =============================================================================
# GET FILE STRUCTURE TESTS
# =============================================================================

# =============================================================================
# GET CHILDREN TESTS
# =============================================================================


class TestGetChildren:
    """Tests for get_children function."""

    def test_get_children_returns_children(self, mock_es_repo):
        """Test get_children returns child elements."""
        mock_client = MagicMock()
        mock_es_repo._get_client.return_value = mock_client
        mock_client.search.return_value = {
            "hits": {
                "hits": [
                    {
                        "_source": {
                            "element_id": "child1",
                            "name": "method1",
                            "element_type": "method",
                            "line_start": 10,
                        }
                    },
                    {
                        "_source": {
                            "element_id": "child2",
                            "name": "method2",
                            "element_type": "method",
                            "line_start": 20,
                        }
                    },
                ]
            }
        }

        result = get_children(
            es=mock_es_repo,
            element_id="parent_id",
        )

        assert isinstance(result, list)
        assert len(result) == 2


# =============================================================================
# SEARCH FEATURES TESTS
# =============================================================================

