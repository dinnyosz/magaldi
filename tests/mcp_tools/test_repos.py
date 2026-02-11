"""Tests for MCP tools - repos category."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from magaldi_mcp.tools import (
    find_files,
    get_file_structure,
    get_repo_stats,
    list_repos,
)


# =============================================================================
# LIST REPOS TESTS
# =============================================================================


class TestListRepos:
    """Tests for list_repos function."""

    def test_list_repos_returns_repos(self, mock_repo):
        """Test list_repos returns repository list."""
        mock_repo.get_indexed_repositories.return_value = [
            {"scope": "magaldi", "repository": "magaldi", "element_count": 100}
        ]

        result = list_repos(repo=mock_repo)

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["scope"] == "magaldi"

    def test_list_repos_with_scope_filter(self, mock_repo):
        """Test list_repos with scope filter."""
        mock_repo.get_indexed_repositories.return_value = []

        result = list_repos(repo=mock_repo, scope="test-scope")

        assert isinstance(result, list)


# =============================================================================
# FIND FILES TESTS
# =============================================================================

# =============================================================================
# FIND FILES TESTS
# =============================================================================


class TestFindFiles:
    """Tests for find_files function."""

    def test_find_files_returns_matching_files(self, mock_repo):
        """Test find_files returns matching files."""
        mock_repo.find_files.return_value = [
            {"relative_path": "src/main.py", "element_id": "id1"},
            {"relative_path": "src/utils.py", "element_id": "id2"},
        ]

        result = find_files(
            repo=mock_repo,
            pattern="**/*.py",
        )

        assert isinstance(result, list)


# =============================================================================
# GET REPO STATS TESTS
# =============================================================================

# =============================================================================
# FIND FILES EXTENDED TESTS
# =============================================================================


class TestFindFilesExtended:
    """Extended tests for find_files function."""

    def test_find_files_with_glob_pattern(self, mock_repo):
        """Test find_files sends wildcard query and returns results."""
        mock_client = MagicMock()
        mock_repo._get_client.return_value = mock_client
        # Mock returns only matching files (filtering is done server-side via wildcard query)
        mock_client.search.return_value = {
            "hits": {
                "hits": [
                    {
                        "_source": {
                            "element_id": "id1",
                            "relative_path": "src/main.py",
                            "language": "python",
                            "line_end": 100,
                        }
                    },
                    {
                        "_source": {
                            "element_id": "id2",
                            "relative_path": "test/test_main.py",
                            "language": "python",
                            "line_end": 50,
                        }
                    },
                ]
            }
        }

        result = find_files(
            repo=mock_repo,
            pattern="**/*.py",
        )

        assert len(result) == 2
        assert all(r["path"].endswith(".py") for r in result)
        # Verify wildcard query was sent to search backend
        mock_client.search.assert_called_once()
        call_body = mock_client.search.call_args[1]["body"]
        filters = call_body["query"]["bool"]["filter"]
        wildcard_filter = next(f for f in filters if "wildcard" in f)
        assert "*.py" in wildcard_filter["wildcard"]["relative_path"]


# =============================================================================
# FIND USAGES TESTS
# =============================================================================

# =============================================================================
# GET REPO STATS TESTS
# =============================================================================


class TestGetRepoStats:
    """Tests for get_repo_stats function."""

    def test_get_repo_stats_returns_stats(self, mock_repo):
        """Test get_repo_stats returns repository statistics."""
        mock_repo.get_repository_stats.return_value = {
            "total_elements": 100,
            "files": 10,
            "classes": 20,
            "functions": 70,
        }

        result = get_repo_stats(
            repo=mock_repo,
            scope="test-scope",
            repository="test-repo",
        )

        assert result is not None
        assert "total_elements" in result or isinstance(result, dict)


# =============================================================================
# FIND SIMILAR TESTS
# =============================================================================

# =============================================================================
# GET FILE STRUCTURE TESTS
# =============================================================================


class TestGetFileStructure:
    """Tests for get_file_structure function."""

    def test_get_file_structure_returns_tree(self, mock_repo):
        """Test get_file_structure returns proper tree structure."""
        mock_client = MagicMock()
        mock_repo._get_client.return_value = mock_client

        # First search: find file element
        # Second search: find all elements in file
        call_count = [0]

        def search_side_effect(**kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                # File element
                return {
                    "hits": {
                        "hits": [
                            {
                                "_source": {
                                    "element_id": "file_id",
                                    "name": "test.py",
                                    "language": "python",
                                    "summary": "Test file",
                                    "line_end": 100,
                                }
                            }
                        ]
                    }
                }
            else:
                # Elements in file
                return {
                    "hits": {
                        "hits": [
                            {
                                "_source": {
                                    "element_id": "class_id",
                                    "name": "TestClass",
                                    "element_type": "class",
                                    "parent_id": "file_id",
                                    "line_start": 1,
                                    "line_end": 50,
                                }
                            },
                            {
                                "_source": {
                                    "element_id": "func_id",
                                    "name": "test_func",
                                    "element_type": "function",
                                    "parent_id": "file_id",
                                    "line_start": 55,
                                    "line_end": 60,
                                }
                            },
                        ]
                    }
                }

        mock_client.search.side_effect = search_side_effect

        result = get_file_structure(
            repo=mock_repo,
            scope="github",
            repository="repo",
            file_path="test.py",
        )

        assert result["file"] == "test.py"
        assert result["language"] == "python"
        assert result["counts"]["classes"] == 1
        assert result["counts"]["functions"] == 1

    def test_get_file_structure_file_not_found(self, mock_repo):
        """Test get_file_structure raises when file not found."""
        mock_client = MagicMock()
        mock_repo._get_client.return_value = mock_client
        mock_client.search.return_value = {"hits": {"hits": []}}

        with pytest.raises(ValueError, match="File not found"):
            get_file_structure(
                repo=mock_repo,
                scope="github",
                repository="repo",
                file_path="nonexistent.py",
            )


# =============================================================================
# LIST FEATURES TESTS
# =============================================================================

