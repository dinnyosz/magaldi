"""Tests for MCP tools - dependencies category."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from magaldi_mcp.tools import (
    dependency_graph,
    find_dependencies,
    find_dependents,
)


# =============================================================================
# FIND DEPENDENCIES TESTS
# =============================================================================


class TestFindDependencies:
    """Tests for find_dependencies function."""

    def test_find_dependencies_by_element_id(self, mock_repo):
        """Test find_dependencies returns imports for a file element."""
        mock_repo.get_document.return_value = {
            "element_id": "scope:repo:main:src/app.py:file:app.py:1",
            "name": "app.py",
            "element_type": "file",
            "relative_path": "src/app.py",
            "scope": "scope",
            "repository": "repo",
            "username": "main",
        }
        mock_repo.get_imports.return_value = [
            {"name": "helper", "module": "utils", "alias": None, "line": 1},
            {"name": "requests", "module": "requests", "alias": None, "line": 2},
            {"name": "config", "module": ".config", "alias": None, "line": 3},
        ]

        mock_client = MagicMock()
        mock_repo._get_client.return_value = mock_client
        # Return file paths for internal import detection
        mock_client.search.return_value = {
            "hits": {
                "hits": [
                    {"_source": {"relative_path": "src/app.py"}},
                    {"_source": {"relative_path": "src/utils.py"}},
                    {"_source": {"relative_path": "src/config.py"}},
                ]
            }
        }

        result = find_dependencies(
            repo=mock_repo,
            element_id="scope:repo:main:src/app.py:file:app.py:1",
        )

        assert "internal_imports" in result
        assert "external_imports" in result
        assert "all_imports" in result
        assert "stats" in result
        assert result["stats"]["total"] == 3

    def test_find_dependencies_by_file_path(self, mock_repo):
        """Test find_dependencies with file_path parameter."""
        mock_client = MagicMock()
        mock_repo._get_client.return_value = mock_client

        # First call: find file element
        # Second call: get file paths for internal import detection
        call_count = [0]

        def search_side_effect(**kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                # File element search
                return {
                    "hits": {
                        "hits": [
                            {
                                "_source": {
                                    "element_id": "scope:repo:main:utils.py:file:utils.py:1",
                                    "name": "utils.py",
                                    "element_type": "file",
                                    "relative_path": "utils.py",
                                    "scope": "scope",
                                    "repository": "repo",
                                }
                            }
                        ]
                    }
                }
            else:
                # File paths for internal import detection
                return {
                    "hits": {
                        "hits": [
                            {"_source": {"relative_path": "utils.py"}},
                            {"_source": {"relative_path": "app.py"}},
                        ]
                    }
                }

        mock_client.search.side_effect = search_side_effect
        mock_repo.get_imports.return_value = [
            {"name": "os", "module": "os", "alias": None, "line": 1},
        ]

        result = find_dependencies(
            repo=mock_repo,
            file_path="utils.py",
            scope="scope",
            repository="repo",
        )

        assert "external_imports" in result
        assert len(result["external_imports"]) == 1

    def test_find_dependencies_classifies_relative_imports_as_internal(self, mock_repo):
        """Test that relative imports (starting with .) are classified as internal."""
        mock_repo.get_document.return_value = {
            "element_id": "scope:repo:main:src/app.py:file:app.py:1",
            "name": "app.py",
            "element_type": "file",
            "relative_path": "src/app.py",
            "scope": "scope",
            "repository": "repo",
        }
        mock_repo.get_imports.return_value = [
            {"name": "config", "module": ".config", "alias": None, "line": 1},
            {"name": "utils", "module": "..shared.utils", "alias": None, "line": 2},
        ]

        mock_client = MagicMock()
        mock_repo._get_client.return_value = mock_client
        mock_client.search.return_value = {"hits": {"hits": []}}

        result = find_dependencies(
            repo=mock_repo,
            element_id="scope:repo:main:src/app.py:file:app.py:1",
        )

        # Both should be internal (relative imports)
        assert len(result["internal_imports"]) == 2
        assert len(result["external_imports"]) == 0

    def test_find_dependencies_requires_file_path_or_element_id(self, mock_repo):
        """Test find_dependencies raises when neither file_path nor element_id provided."""
        with pytest.raises(ValueError, match="Either file_path or element_id required"):
            find_dependencies(repo=mock_repo)

    def test_find_dependencies_element_not_found(self, mock_repo):
        """Test find_dependencies raises when element not found."""
        mock_repo.get_document.return_value = None

        with pytest.raises(ValueError, match="Element not found"):
            find_dependencies(repo=mock_repo, element_id="nonexistent")

    def test_find_dependencies_requires_scope_repo_for_file_path(self, mock_repo):
        """Test find_dependencies requires scope/repository when using file_path."""
        with pytest.raises(ValueError, match="scope and repository required"):
            find_dependencies(repo=mock_repo, file_path="utils.py")

    def test_find_dependencies_rejects_non_file_elements(self, mock_repo):
        """Test find_dependencies raises when element is not a file."""
        mock_repo.get_document.return_value = {
            "element_id": "scope:repo:main:utils.py:function:helper:10",
            "name": "helper",
            "element_type": "function",
        }

        with pytest.raises(ValueError, match="Element is not a file"):
            find_dependencies(
                repo=mock_repo, element_id="scope:repo:main:utils.py:function:helper:10"
            )


# =============================================================================
# FIND DEPENDENTS TESTS
# =============================================================================

# =============================================================================
# FIND DEPENDENTS TESTS
# =============================================================================


class TestFindDependents:
    """Tests for find_dependents function."""

    def test_find_dependents_returns_files_importing_module(self, mock_repo):
        """Test find_dependents returns files that import a module."""
        mock_repo.find_elements_importing.return_value = [
            {
                "element_id": "scope:repo:main:app.py:file:app.py:1",
                "relative_path": "app.py",
                "language": "python",
            },
            {
                "element_id": "scope:repo:main:cli.py:file:cli.py:1",
                "relative_path": "cli.py",
                "language": "python",
            },
        ]

        result = find_dependents(
            repo=mock_repo,
            module="utils",
            scope="scope",
            repository="repo",
        )

        assert "module" in result
        assert "dependents" in result
        assert "total" in result
        assert result["module"] == "utils"
        assert result["total"] == 2
        assert result["dependents"][0]["file"] == "app.py"
        assert result["dependents"][1]["file"] == "cli.py"

    def test_find_dependents_empty_results(self, mock_repo):
        """Test find_dependents with no dependents."""
        mock_repo.find_elements_importing.return_value = []

        result = find_dependents(
            repo=mock_repo,
            module="nonexistent_module",
            scope="scope",
            repository="repo",
        )

        assert result["total"] == 0
        assert result["dependents"] == []

    def test_find_dependents_respects_limit(self, mock_repo):
        """Test find_dependents passes limit to ES."""
        mock_repo.find_elements_importing.return_value = []

        result = find_dependents(
            repo=mock_repo,
            module="utils",
            scope="scope",
            repository="repo",
            limit=10,
        )

        mock_repo.find_elements_importing.assert_called_once_with(
            module="utils",
            scope="scope",
            repository="repo",
            username="main",
            limit=10,
        )


# =============================================================================
# DEPENDENCY GRAPH TESTS
# =============================================================================

# =============================================================================
# DEPENDENCY GRAPH TESTS
# =============================================================================


class TestDependencyGraph:
    """Tests for dependency_graph function."""

    def test_dependency_graph_builds_nodes_and_edges(self, mock_repo):
        """Test dependency_graph builds nodes and edges."""
        mock_client = MagicMock()
        mock_repo._get_client.return_value = mock_client
        mock_client.search.return_value = {
            "hits": {
                "hits": [
                    {
                        "_source": {
                            "element_id": "scope:repo:main:app.py:file:app.py:1",
                            "relative_path": "app.py",
                            "imports": [
                                {"name": "helper", "module": "utils", "line": 1},
                            ],
                        }
                    },
                    {
                        "_source": {
                            "element_id": "scope:repo:main:utils.py:file:utils.py:1",
                            "relative_path": "utils.py",
                            "imports": [],
                        }
                    },
                ]
            }
        }

        result = dependency_graph(
            repo=mock_repo,
            scope="scope",
            repository="repo",
        )

        assert "nodes" in result
        assert "edges" in result
        assert "cycles" in result
        assert "stats" in result
        assert len(result["nodes"]) == 2
        assert "app.py" in result["nodes"]
        assert "utils.py" in result["nodes"]

    def test_dependency_graph_detects_cycles(self, mock_repo):
        """Test dependency_graph detects circular dependencies."""
        mock_client = MagicMock()
        mock_repo._get_client.return_value = mock_client
        mock_client.search.return_value = {
            "hits": {
                "hits": [
                    {
                        "_source": {
                            "element_id": "scope:repo:main:a.py:file:a.py:1",
                            "relative_path": "a.py",
                            "imports": [
                                {"name": "b", "module": "b", "line": 1},
                            ],
                        }
                    },
                    {
                        "_source": {
                            "element_id": "scope:repo:main:b.py:file:b.py:1",
                            "relative_path": "b.py",
                            "imports": [
                                {"name": "a", "module": "a", "line": 1},
                            ],
                        }
                    },
                ]
            }
        }

        result = dependency_graph(
            repo=mock_repo,
            scope="scope",
            repository="repo",
        )

        assert result["stats"]["has_cycles"] is True
        assert len(result["cycles"]) >= 1

    def test_dependency_graph_no_cycles(self, mock_repo):
        """Test dependency_graph with no circular dependencies."""
        mock_client = MagicMock()
        mock_repo._get_client.return_value = mock_client
        mock_client.search.return_value = {
            "hits": {
                "hits": [
                    {
                        "_source": {
                            "element_id": "scope:repo:main:app.py:file:app.py:1",
                            "relative_path": "app.py",
                            "imports": [
                                {"name": "helper", "module": "utils", "line": 1},
                            ],
                        }
                    },
                    {
                        "_source": {
                            "element_id": "scope:repo:main:utils.py:file:utils.py:1",
                            "relative_path": "utils.py",
                            "imports": [],
                        }
                    },
                ]
            }
        }

        result = dependency_graph(
            repo=mock_repo,
            scope="scope",
            repository="repo",
        )

        assert result["stats"]["has_cycles"] is False
        assert len(result["cycles"]) == 0

    def test_dependency_graph_handles_relative_imports(self, mock_repo):
        """Test dependency_graph resolves relative imports."""
        mock_client = MagicMock()
        mock_repo._get_client.return_value = mock_client
        mock_client.search.return_value = {
            "hits": {
                "hits": [
                    {
                        "_source": {
                            "element_id": "scope:repo:main:src/app.py:file:app.py:1",
                            "relative_path": "src/app.py",
                            "imports": [
                                {"name": "config", "module": ".config", "line": 1},
                            ],
                        }
                    },
                    {
                        "_source": {
                            "element_id": "scope:repo:main:src/config.py:file:config.py:1",
                            "relative_path": "src/config.py",
                            "imports": [],
                        }
                    },
                ]
            }
        }

        result = dependency_graph(
            repo=mock_repo,
            scope="scope",
            repository="repo",
        )

        # Should have edge from src/app.py to src/config.py
        edges = result["edges"]
        found_edge = False
        for edge in edges:
            if edge["from"] == "src/app.py" and edge["to"] == "src/config.py":
                found_edge = True
                break
        assert found_edge

    def test_dependency_graph_empty_repo(self, mock_repo):
        """Test dependency_graph with empty repository."""
        mock_client = MagicMock()
        mock_repo._get_client.return_value = mock_client
        mock_client.search.return_value = {"hits": {"hits": []}}

        result = dependency_graph(
            repo=mock_repo,
            scope="scope",
            repository="repo",
        )

        assert result["nodes"] == []
        assert result["edges"] == []
        assert result["cycles"] == []
        assert result["stats"]["node_count"] == 0

    def test_dependency_graph_internal_only(self, mock_repo):
        """Test dependency_graph filters external imports when internal_only=True."""
        mock_client = MagicMock()
        mock_repo._get_client.return_value = mock_client
        mock_client.search.return_value = {
            "hits": {
                "hits": [
                    {
                        "_source": {
                            "element_id": "scope:repo:main:app.py:file:app.py:1",
                            "relative_path": "app.py",
                            "imports": [
                                {"name": "utils", "module": "utils", "line": 1},  # Internal
                                {"name": "requests", "module": "requests", "line": 2},  # External
                            ],
                        }
                    },
                    {
                        "_source": {
                            "element_id": "scope:repo:main:utils.py:file:utils.py:1",
                            "relative_path": "utils.py",
                            "imports": [],
                        }
                    },
                ]
            }
        }

        result = dependency_graph(
            repo=mock_repo,
            scope="scope",
            repository="repo",
            internal_only=True,
        )

        # Should only have edge to utils.py, not to external requests
        assert len(result["edges"]) == 1
        assert result["edges"][0]["to"] == "utils.py"


# =============================================================================
# LIST PATTERNS TESTS
# =============================================================================

