"""Tests for MCP tools - search category."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from magaldi_mcp.tools import (
    pattern_search,
    search_code,
)


# =============================================================================
# SEARCH CODE TESTS
# =============================================================================


class TestSearchCode:
    """Tests for search_code function."""

    def test_search_code_returns_formatted_results(self, mock_repo, mock_embed_client):
        """Test search_code returns properly formatted results."""
        mock_repo.search_by_vector.return_value = [
            {
                "element_id": "scope:repo:user:file.py:function:test:1",
                "name": "test",
                "element_type": "function",
                "relative_path": "file.py",
                "line_start": 1,
                "summary": "A test function.",
                "_score": 0.9,
            }
        ]

        result = search_code(
            repo=mock_repo,
            embed_client=mock_embed_client,
            query="test function",
        )

        assert isinstance(result, dict)
        assert "code_results" in result
        assert "test_results" in result
        assert len(result["code_results"]) == 1
        assert result["code_results"][0]["name"] == "test"

    def test_search_code_with_filters(self, mock_repo, mock_embed_client):
        """Test search_code with element type filter."""
        mock_repo.search_by_vector.return_value = []

        result = search_code(
            repo=mock_repo,
            embed_client=mock_embed_client,
            query="test",
            element_types=["function"],
            limit=5,
        )

        assert isinstance(result, dict)
        assert "code_results" in result
        assert "test_results" in result

    def test_search_code_with_repository_filter(self, mock_repo, mock_embed_client):
        """Test search_code with repository filter."""
        mock_repo.search_by_vector.return_value = []

        result = search_code(
            repo=mock_repo,
            embed_client=mock_embed_client,
            query="test",
            repository="test-repo",
            scope="test-scope",
        )

        assert isinstance(result, dict)
        assert "code_results" in result
        assert "test_results" in result


# =============================================================================
# GET ELEMENT TESTS
# =============================================================================

# =============================================================================
# SEARCH CODE FALLBACK TESTS
# =============================================================================


class TestSearchCodeFallback:
    """Tests for search_code fallback behavior."""

    def test_search_code_falls_back_to_keyword(self, mock_repo):
        """Test search_code falls back to keyword search when vector fails."""
        mock_repo.search_by_keyword.return_value = [
            {
                "element_id": "id1",
                "name": "test",
                "element_type": "function",
                "relative_path": "file.py",
                "line_start": 1,
            }
        ]

        result = search_code(
            repo=mock_repo,
            embed_client=None,  # No embed client
            query="test",
        )

        assert len(result["code_results"]) == 1
        mock_repo.search_by_keyword.assert_called_once()

    def test_search_code_filters_by_language(self, mock_repo, mock_embed_client):
        """Test search_code filters results by language."""
        mock_repo.search_by_vector.return_value = [
            {
                "element_id": "id1",
                "name": "test",
                "element_type": "function",
                "relative_path": "file.py",
                "line_start": 1,
                "language": "python",
            },
            {
                "element_id": "id2",
                "name": "test",
                "element_type": "function",
                "relative_path": "file.ts",
                "line_start": 1,
                "language": "typescript",
            },
        ]

        result = search_code(
            repo=mock_repo,
            embed_client=mock_embed_client,
            query="test",
            language="python",
        )

        assert len(result["code_results"]) == 1
        assert result["code_results"][0]["element_id"] == "id1"

    def test_search_code_brief_mode(self, mock_repo, mock_embed_client):
        """Test search_code brief mode excludes summary."""
        mock_repo.search_by_vector.return_value = [
            {
                "element_id": "id1",
                "name": "test",
                "element_type": "function",
                "relative_path": "file.py",
                "line_start": 1,
                "summary": "A test function",
                "signature": "def test():",
            }
        ]

        result = search_code(
            repo=mock_repo,
            embed_client=mock_embed_client,
            query="test",
            brief=True,
        )

        assert len(result["code_results"]) == 1
        assert "summary" not in result["code_results"][0]
        assert "signature" not in result["code_results"][0]

    def test_search_code_includes_code_when_requested(self, mock_repo, mock_embed_client):
        """Test search_code includes code when include_code=True."""
        mock_repo.search_by_vector.return_value = [
            {
                "element_id": "id1",
                "name": "test",
                "element_type": "function",
                "relative_path": "file.py",
                "line_start": 1,
                "raw_code": "def test(): pass",
            }
        ]

        result = search_code(
            repo=mock_repo,
            embed_client=mock_embed_client,
            query="test",
            include_code=True,
        )

        assert len(result["code_results"]) == 1
        assert result["code_results"][0]["code"] == "def test(): pass"

    def test_search_code_method_qualified_name(self, mock_repo, mock_embed_client):
        """Test search_code builds qualified name for methods."""
        mock_repo.search_by_vector.return_value = [
            {
                "element_id": "method_id",
                "name": "my_method",
                "element_type": "method",
                "relative_path": "file.py",
                "line_start": 10,
                "parent_id": "class_id",
            }
        ]
        mock_repo.get_document.return_value = {
            "element_id": "class_id",
            "name": "MyClass",
            "element_type": "class",
        }

        result = search_code(
            repo=mock_repo,
            embed_client=mock_embed_client,
            query="method",
        )

        assert len(result["code_results"]) == 1
        assert result["code_results"][0]["name"] == "MyClass.my_method"


# =============================================================================
# SEARCH CODE TEST GROUPING TESTS
# =============================================================================

# =============================================================================
# SEARCH CODE TEST GROUPING TESTS
# =============================================================================


class TestSearchCodeTestGrouping:
    """Tests for search_code test result grouping."""

    def test_groups_test_and_code_results(self, mock_repo, mock_embed_client):
        """Test that results are grouped by is_test."""
        mock_repo.search_by_vector.return_value = [
            {
                "element_id": "id1",
                "name": "UserService",
                "element_type": "class",
                "is_test": False,
                "relative_path": "service.py",
                "line_start": 1,
            },
            {
                "element_id": "id2",
                "name": "test_user_service",
                "element_type": "function",
                "is_test": True,
                "relative_path": "test_service.py",
                "line_start": 1,
            },
        ]

        result = search_code(
            repo=mock_repo,
            embed_client=mock_embed_client,
            query="user service",
        )

        assert "code_results" in result
        assert "test_results" in result
        assert len(result["code_results"]) == 1
        assert len(result["test_results"]) == 1
        assert result["code_results"][0]["name"] == "UserService"
        assert result["test_results"][0]["name"] == "test_user_service"

    def test_include_tests_false_excludes_tests(self, mock_repo, mock_embed_client):
        """Test that include_tests=False excludes test results."""
        mock_repo.search_by_vector.return_value = [
            {
                "element_id": "id1",
                "name": "UserService",
                "element_type": "class",
                "is_test": False,
                "relative_path": "service.py",
                "line_start": 1,
            },
            {
                "element_id": "id2",
                "name": "test_user_service",
                "element_type": "function",
                "is_test": True,
                "relative_path": "test_service.py",
                "line_start": 1,
            },
        ]

        result = search_code(
            repo=mock_repo,
            embed_client=mock_embed_client,
            query="user service",
            include_tests=False,
        )

        assert len(result["code_results"]) == 1
        assert len(result["test_results"]) == 0

    def test_results_include_is_test_field(self, mock_repo, mock_embed_client):
        """Test that individual results include is_test field."""
        mock_repo.search_by_vector.return_value = [
            {
                "element_id": "id1",
                "name": "foo",
                "element_type": "function",
                "is_test": True,
                "relative_path": "test_foo.py",
                "line_start": 1,
            },
        ]

        result = search_code(
            repo=mock_repo,
            embed_client=mock_embed_client,
            query="foo",
        )

        assert result["test_results"][0]["is_test"] is True

    def test_results_include_totals(self, mock_repo, mock_embed_client):
        """Test that results include total counts."""
        mock_repo.search_by_vector.return_value = [
            {
                "element_id": "id1",
                "name": "UserService",
                "element_type": "class",
                "is_test": False,
                "relative_path": "service.py",
                "line_start": 1,
            },
            {
                "element_id": "id2",
                "name": "test_user_service",
                "element_type": "function",
                "is_test": True,
                "relative_path": "test_service.py",
                "line_start": 1,
            },
        ]

        result = search_code(
            repo=mock_repo,
            embed_client=mock_embed_client,
            query="user service",
        )

        assert "total_code" in result
        assert "total_tests" in result
        assert result["total_code"] == 1
        assert result["total_tests"] == 1


# =============================================================================
# FIND SIMILAR TEST GROUPING TESTS
# =============================================================================

# =============================================================================
# PATTERN SEARCH TESTS
# =============================================================================


class TestPatternSearch:
    """Tests for pattern_search function."""

    def test_pattern_search_regexp_mode(self, mock_repo):
        """Test pattern_search with regexp mode."""
        mock_repo.search_by_regexp.return_value = [
            {
                "element_id": "test:repo:main:file.py:function:add_column:10",
                "name": "add_column",
                "element_type": "function",
                "relative_path": "file.py",
                "line_start": 10,
                "raw_code": "def add_column(table, Model):\n    pass",
                "is_test": False,
            }
        ]

        result = pattern_search(
            repo=mock_repo,
            pattern="add_column.*Model",
            mode="regexp",
            scope="test",
            repository="repo",
        )

        assert "code_results" in result
        assert len(result["code_results"]) == 1
        assert result["code_results"][0]["name"] == "add_column"
        assert result["mode"] == "regexp"
        assert result["pattern"] == "add_column.*Model"
        mock_repo.search_by_regexp.assert_called_once_with(
            pattern="add_column.*Model",
            scope="test",
            repository="repo",
            username=None,
            glob=None,
            size=50,
            include_tests=True,
        )

    def test_pattern_search_wildcard_mode(self, mock_repo):
        """Test pattern_search with wildcard mode."""
        mock_repo.search_by_wildcard.return_value = []

        result = pattern_search(
            repo=mock_repo,
            pattern="*column*",
            mode="wildcard",
            scope="test",
            repository="repo",
        )

        assert "code_results" in result
        assert result["mode"] == "wildcard"
        mock_repo.search_by_wildcard.assert_called_once_with(
            pattern="*column*",
            scope="test",
            repository="repo",
            username=None,
            glob=None,
            size=50,
            include_tests=True,
        )

    def test_pattern_search_proximity_mode(self, mock_repo):
        """Test pattern_search with proximity mode."""
        mock_repo.search_by_proximity.return_value = []

        result = pattern_search(
            repo=mock_repo,
            pattern="add column Model",
            mode="proximity",
            slop=5,
            scope="test",
            repository="repo",
            username="main",
        )

        assert "code_results" in result
        assert result["mode"] == "proximity"
        mock_repo.search_by_proximity.assert_called_once_with(
            terms="add column Model",
            slop=5,
            scope="test",
            repository="repo",
            username="main",
            glob=None,
            size=50,
            include_tests=True,
        )

    def test_pattern_search_invalid_mode(self, mock_repo):
        """Test pattern_search with invalid mode raises error."""
        with pytest.raises(ValueError, match="Invalid mode"):
            pattern_search(
                repo=mock_repo,
                pattern="test",
                mode="invalid",
                scope="test",
                repository="repo",
            )

    def test_pattern_search_groups_test_results(self, mock_repo):
        """Test pattern_search groups code and test results separately."""
        mock_repo.search_by_regexp.return_value = [
            {
                "element_id": "test:repo:main:src/app.py:function:process:10",
                "name": "process",
                "element_type": "function",
                "relative_path": "src/app.py",
                "line_start": 10,
                "raw_code": "def process(): pass",
                "is_test": False,
            },
            {
                "element_id": "test:repo:main:tests/test_app.py:function:test_process:5",
                "name": "test_process",
                "element_type": "function",
                "relative_path": "tests/test_app.py",
                "line_start": 5,
                "raw_code": "def test_process(): pass",
                "is_test": True,
            },
        ]

        result = pattern_search(
            repo=mock_repo,
            pattern="process",
            mode="regexp",
            scope="test",
            repository="repo",
        )

        assert len(result["code_results"]) == 1
        assert len(result["test_results"]) == 1
        assert result["totals"]["code"] == 1
        assert result["totals"]["tests"] == 1
        assert result["code_results"][0]["name"] == "process"
        assert result["test_results"][0]["name"] == "test_process"

    def test_pattern_search_with_glob_filter(self, mock_repo):
        """Test pattern_search passes glob filter to ES method."""
        mock_repo.search_by_regexp.return_value = []

        pattern_search(
            repo=mock_repo,
            pattern="test",
            mode="regexp",
            scope="test",
            repository="repo",
            glob="*.py",
        )

        mock_repo.search_by_regexp.assert_called_once()
        call_kwargs = mock_repo.search_by_regexp.call_args[1]
        assert call_kwargs["glob"] == "*.py"

    def test_pattern_search_with_custom_limit(self, mock_repo):
        """Test pattern_search passes limit to ES method."""
        mock_repo.search_by_wildcard.return_value = []

        pattern_search(
            repo=mock_repo,
            pattern="*test*",
            mode="wildcard",
            scope="test",
            repository="repo",
            limit=100,
        )

        mock_repo.search_by_wildcard.assert_called_once()
        call_kwargs = mock_repo.search_by_wildcard.call_args[1]
        assert call_kwargs["size"] == 100

    def test_pattern_search_with_include_tests_false(self, mock_repo):
        """Test pattern_search passes include_tests to ES method."""
        mock_repo.search_by_proximity.return_value = []

        pattern_search(
            repo=mock_repo,
            pattern="test pattern",
            mode="proximity",
            scope="test",
            repository="repo",
            include_tests=False,
        )

        mock_repo.search_by_proximity.assert_called_once()
        call_kwargs = mock_repo.search_by_proximity.call_args[1]
        assert call_kwargs["include_tests"] is False


# =============================================================================
# FIND USAGES WITH PATTERN SEARCH TESTS
# =============================================================================

