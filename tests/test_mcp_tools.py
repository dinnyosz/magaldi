"""Tests for the MCP tools module."""

from __future__ import annotations

import warnings
from unittest.mock import MagicMock, patch

import pytest

from magaldi_mcp.tools import (
    search_code,
    search_features,
    find_similar,
    get_element,
    batch_get_elements,
    get_context,
    get_children,
    find_files,
    get_file_structure,
    list_features,
    get_feature_members,
    list_repos,
    get_repo_stats,
    grep_code,
    find_usages,
    find_implementations,
    get_call_graph,
    pattern_search,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_es_repo():
    """Create a mock Elasticsearch repository."""
    repo = MagicMock()
    repo.search_by_vector.return_value = []
    repo.get_document.return_value = None
    repo.search_by_text.return_value = []
    repo.get_indexed_repositories.return_value = []
    return repo


@pytest.fixture
def mock_embed_client():
    """Create a mock embedding client."""
    client = MagicMock()
    client.embed_single.return_value = [0.1] * 1024
    return client


# =============================================================================
# SEARCH CODE TESTS
# =============================================================================


class TestSearchCode:
    """Tests for search_code function."""

    def test_search_code_returns_formatted_results(self, mock_es_repo, mock_embed_client):
        """Test search_code returns properly formatted results."""
        mock_es_repo.search_by_vector.return_value = [
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
            es=mock_es_repo,
            embed_client=mock_embed_client,
            query="test function",
        )

        assert isinstance(result, dict)
        assert "code_results" in result
        assert "test_results" in result
        assert len(result["code_results"]) == 1
        assert result["code_results"][0]["name"] == "test"

    def test_search_code_with_filters(self, mock_es_repo, mock_embed_client):
        """Test search_code with element type filter."""
        mock_es_repo.search_by_vector.return_value = []

        result = search_code(
            es=mock_es_repo,
            embed_client=mock_embed_client,
            query="test",
            element_types=["function"],
            limit=5,
        )

        assert isinstance(result, dict)
        assert "code_results" in result
        assert "test_results" in result

    def test_search_code_with_repository_filter(self, mock_es_repo, mock_embed_client):
        """Test search_code with repository filter."""
        mock_es_repo.search_by_vector.return_value = []

        result = search_code(
            es=mock_es_repo,
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


class TestListRepos:
    """Tests for list_repos function."""

    def test_list_repos_returns_repos(self, mock_es_repo):
        """Test list_repos returns repository list."""
        mock_es_repo.get_indexed_repositories.return_value = [
            {"scope": "magaldi", "repository": "magaldi", "element_count": 100}
        ]

        result = list_repos(es=mock_es_repo)

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["scope"] == "magaldi"

    def test_list_repos_with_scope_filter(self, mock_es_repo):
        """Test list_repos with scope filter."""
        mock_es_repo.get_indexed_repositories.return_value = []

        result = list_repos(es=mock_es_repo, scope="test-scope")

        assert isinstance(result, list)


# =============================================================================
# GREP CODE TESTS
# =============================================================================


class TestGrepCode:
    """Tests for grep_code function."""

    def test_grep_code_returns_matches(self, mock_es_repo):
        """Test grep_code returns pattern matches."""
        mock_es_repo.grep_code.return_value = {
            "matches": [
                {
                    "file": "file.py",
                    "line": 10,
                    "content": "def test_function():",
                }
            ],
            "total_matches": 1,
        }

        result = grep_code(
            es=mock_es_repo,
            pattern="test_function",
        )

        assert result is not None

    def test_grep_code_with_glob_filter(self, mock_es_repo):
        """Test grep_code with glob filter."""
        mock_es_repo.grep_code.return_value = {"matches": [], "total_matches": 0}

        result = grep_code(
            es=mock_es_repo,
            pattern="test",
            glob="*.py",
        )

        assert result is not None


# =============================================================================
# FIND FILES TESTS
# =============================================================================


class TestFindFiles:
    """Tests for find_files function."""

    def test_find_files_returns_matching_files(self, mock_es_repo):
        """Test find_files returns matching files."""
        mock_es_repo.find_files.return_value = [
            {"relative_path": "src/main.py", "element_id": "id1"},
            {"relative_path": "src/utils.py", "element_id": "id2"},
        ]

        result = find_files(
            es=mock_es_repo,
            pattern="**/*.py",
        )

        assert isinstance(result, list)


# =============================================================================
# GET REPO STATS TESTS
# =============================================================================


class TestGetRepoStats:
    """Tests for get_repo_stats function."""

    def test_get_repo_stats_returns_stats(self, mock_es_repo):
        """Test get_repo_stats returns repository statistics."""
        mock_es_repo.get_repository_stats.return_value = {
            "total_elements": 100,
            "files": 10,
            "classes": 20,
            "functions": 70,
        }

        result = get_repo_stats(
            es=mock_es_repo,
            scope="test-scope",
            repository="test-repo",
        )

        assert result is not None
        assert "total_elements" in result or isinstance(result, dict)


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
            "embedding": [0.1] * 1024,
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


class TestGetChildren:
    """Tests for get_children function."""

    def test_get_children_returns_children(self, mock_es_repo):
        """Test get_children returns child elements."""
        mock_client = MagicMock()
        mock_es_repo._get_client.return_value = mock_client
        mock_client.search.return_value = {
            "hits": {
                "hits": [
                    {"_source": {"element_id": "child1", "name": "method1", "element_type": "method", "line_start": 10}},
                    {"_source": {"element_id": "child2", "name": "method2", "element_type": "method", "line_start": 20}},
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


class TestSearchFeatures:
    """Tests for search_features function."""

    def test_search_features_returns_results(self, mock_es_repo, mock_embed_client):
        """Test search_features returns formatted results."""
        mock_es_repo.search_by_vector.return_value = [
            {
                "element_id": "feature1",
                "name": "authentication",
                "element_type": "feature",
                "cluster_label": "authentication",
                "summary": "Auth feature",
                "member_count": 5,
            }
        ]

        result = search_features(
            es=mock_es_repo,
            embed_client=mock_embed_client,
            query="authentication",
        )

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["label"] == "authentication"
        assert result[0]["type"] == "feature"

    def test_search_features_falls_back_to_keyword(self, mock_es_repo):
        """Test search_features falls back to keyword search."""
        mock_es_repo.search_by_keyword.return_value = [
            {
                "element_id": "feature1",
                "name": "auth",
                "element_type": "feature",
                "member_count": 3,
            }
        ]

        result = search_features(
            es=mock_es_repo,
            embed_client=None,
            query="auth",
        )

        assert isinstance(result, list)

    def test_search_features_includes_subfeature_parent_info(self, mock_es_repo, mock_embed_client):
        """Test search_features includes parent info for subfeatures."""
        mock_es_repo.search_by_vector.return_value = [
            {
                "element_id": "subfeature1",
                "name": "token_validation",
                "element_type": "subfeature",
                "summary": "Validates JWT tokens",
                "member_count": 2,
                "parent_feature_label": "authentication",
                "parent_feature_summary": "Auth system",
            }
        ]

        result = search_features(
            es=mock_es_repo,
            embed_client=mock_embed_client,
            query="token",
        )

        assert len(result) == 1
        assert result[0]["type"] == "subfeature"
        assert result[0]["parent_feature_label"] == "authentication"


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
            "embedding": [0.1] * 1024,
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
            "embedding": [0.1] * 1024,
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


class TestGetElementExtended:
    """Extended tests for get_element function."""

    def test_get_element_includes_optional_fields(self, mock_es_repo):
        """Test get_element includes all optional fields when present."""
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

        result = get_element(es=mock_es_repo, element_id="id1")

        assert result["signature"] == "def test():"
        assert result["docstring"] == "A test function."
        assert result["decorators"] == ["@fixture"]
        assert result["is_async"] is True
        assert result["parent_id"] == "parent_id"


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
                    {"_source": {"element_id": "method1", "name": "method1", "element_type": "method", "line_start": 10}},
                    {"_source": {"element_id": "method2", "name": "method2", "element_type": "method", "line_start": 20}},
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


class TestGetFileStructure:
    """Tests for get_file_structure function."""

    def test_get_file_structure_returns_tree(self, mock_es_repo):
        """Test get_file_structure returns proper tree structure."""
        mock_client = MagicMock()
        mock_es_repo._get_client.return_value = mock_client

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
            es=mock_es_repo,
            scope="github",
            repository="repo",
            file_path="test.py",
        )

        assert result["file"]["path"] == "test.py"
        assert result["file"]["language"] == "python"
        assert result["stats"]["classes"] == 1
        assert result["stats"]["functions"] == 1

    def test_get_file_structure_file_not_found(self, mock_es_repo):
        """Test get_file_structure raises when file not found."""
        mock_client = MagicMock()
        mock_es_repo._get_client.return_value = mock_client
        mock_client.search.return_value = {"hits": {"hits": []}}

        with pytest.raises(ValueError, match="File not found"):
            get_file_structure(
                es=mock_es_repo,
                scope="github",
                repository="repo",
                file_path="nonexistent.py",
            )


# =============================================================================
# LIST FEATURES TESTS
# =============================================================================


class TestListFeatures:
    """Tests for list_features function."""

    def test_list_features_returns_combined_list(self, mock_es_repo):
        """Test list_features combines features and subfeatures."""
        mock_es_repo.get_features.return_value = [
            {"label": "auth", "member_count": 10},
        ]
        mock_es_repo.get_subfeatures.return_value = [
            {"label": "token", "member_count": 3},
        ]

        result = list_features(
            es=mock_es_repo,
            scope="github",
            repository="repo",
        )

        assert len(result) == 2
        # Should be sorted by member_count descending
        assert result[0]["member_count"] >= result[1]["member_count"]
        assert result[0]["type"] == "feature"
        assert result[1]["type"] == "subfeature"


# =============================================================================
# GET FEATURE MEMBERS TESTS
# =============================================================================


class TestGetFeatureMembers:
    """Tests for get_feature_members function."""

    def test_get_feature_members_returns_members(self, mock_es_repo):
        """Test get_feature_members returns member elements."""
        mock_es_repo.get_document.side_effect = [
            # Feature document
            {
                "element_id": "feature1",
                "name": "authentication",
                "member_ids": ["id1", "id2"],
            },
            # First member
            {
                "element_id": "id1",
                "name": "login",
                "element_type": "function",
                "relative_path": "auth.py",
                "line_start": 10,
                "summary": "Login function",
            },
            # Second member
            {
                "element_id": "id2",
                "name": "logout",
                "element_type": "function",
                "relative_path": "auth.py",
                "line_start": 20,
            },
        ]
        mock_es_repo.get_glossary_terms.return_value = []

        result = get_feature_members(es=mock_es_repo, feature_id="feature1")

        assert "members" in result
        assert "glossary_terms" in result
        assert len(result["members"]) == 2
        assert result["members"][0]["name"] == "login"
        assert result["members"][1]["name"] == "logout"

    def test_get_feature_members_not_found(self, mock_es_repo):
        """Test get_feature_members raises when feature not found."""
        mock_es_repo.get_document.return_value = None

        with pytest.raises(ValueError, match="not found"):
            get_feature_members(es=mock_es_repo, feature_id="nonexistent")

    def test_get_feature_members_empty(self, mock_es_repo):
        """Test get_feature_members with no members."""
        mock_es_repo.get_document.return_value = {
            "element_id": "feature1",
            "name": "empty_feature",
            "member_ids": [],
        }

        result = get_feature_members(es=mock_es_repo, feature_id="feature1")

        assert result == {"members": [], "glossary_terms": []}


# =============================================================================
# FIND FILES EXTENDED TESTS
# =============================================================================


class TestFindFilesExtended:
    """Extended tests for find_files function."""

    def test_find_files_with_glob_pattern(self, mock_es_repo):
        """Test find_files filters by glob pattern."""
        mock_client = MagicMock()
        mock_es_repo._get_client.return_value = mock_client
        mock_client.search.return_value = {
            "hits": {
                "hits": [
                    {"_source": {"element_id": "id1", "relative_path": "src/main.py", "language": "python", "line_end": 100}},
                    {"_source": {"element_id": "id2", "relative_path": "test/test_main.py", "language": "python", "line_end": 50}},
                    {"_source": {"element_id": "id3", "relative_path": "README.md", "language": "markdown", "line_end": 10}},
                ]
            }
        }

        result = find_files(
            es=mock_es_repo,
            pattern="**/*.py",
        )

        assert len(result) == 2
        assert all(r["path"].endswith(".py") for r in result)


# =============================================================================
# GREP CODE EXTENDED TESTS
# =============================================================================


class TestGrepCodeExtended:
    """Extended tests for grep_code function."""

    def test_grep_code_with_regex_match(self, mock_es_repo):
        """Test grep_code finds regex matches in code."""
        mock_client = MagicMock()
        mock_es_repo._get_client.return_value = mock_client
        mock_client.search.return_value = {
            "hits": {
                "hits": [
                    {
                        "_source": {
                            "element_id": "id1",
                            "name": "test_func",
                            "element_type": "function",
                            "relative_path": "test.py",
                            "line_start": 10,
                            "raw_code": "def test_func():\n    return True\n",
                            "is_test": False,
                        }
                    }
                ]
            }
        }

        result = grep_code(
            es=mock_es_repo,
            pattern=r"def\s+test_",
        )

        assert "code_results" in result
        assert len(result["code_results"]) >= 1
        assert result["code_results"][0]["file"] == "test.py"
        assert "def test_" in result["code_results"][0]["content"]

    def test_grep_code_with_context_lines(self, mock_es_repo):
        """Test grep_code includes context lines."""
        mock_client = MagicMock()
        mock_es_repo._get_client.return_value = mock_client
        mock_client.search.return_value = {
            "hits": {
                "hits": [
                    {
                        "_source": {
                            "element_id": "id1",
                            "name": "func",
                            "element_type": "function",
                            "relative_path": "test.py",
                            "line_start": 1,
                            "raw_code": "line1\nline2\nmatch here\nline4\nline5\n",
                            "is_test": False,
                        }
                    }
                ]
            }
        }

        result = grep_code(
            es=mock_es_repo,
            pattern="match",
            context_lines=2,
        )

        assert "code_results" in result
        assert len(result["code_results"]) >= 1
        assert "context_before" in result["code_results"][0]
        assert "context_after" in result["code_results"][0]


# =============================================================================
# FIND USAGES TESTS
# =============================================================================


class TestFindUsages:
    """Tests for find_usages function."""

    def test_find_usages_returns_usages(self, mock_es_repo):
        """Test find_usages returns usage locations."""
        mock_client = MagicMock()
        mock_es_repo._get_client.return_value = mock_client

        mock_es_repo.get_document.return_value = {
            "element_id": "func_id",
            "name": "my_function",
            "element_type": "function",
            "relative_path": "funcs.py",
            "line_start": 10,
            "scope": "github",
            "repository": "repo",
            "username": "main",
        }

        mock_client.search.return_value = {
            "hits": {
                "hits": [
                    {
                        "_source": {
                            "element_id": "caller1",
                            "name": "other_func",
                            "element_type": "function",
                            "relative_path": "other.py",
                            "line_start": 1,
                            "raw_code": "def other_func():\n    my_function()\n",
                        }
                    }
                ]
            }
        }

        result = find_usages(es=mock_es_repo, element_id="func_id")

        assert isinstance(result, list)

    def test_find_usages_not_found(self, mock_es_repo):
        """Test find_usages raises when element not found."""
        mock_es_repo.get_document.return_value = None

        with pytest.raises(ValueError, match="Element not found"):
            find_usages(es=mock_es_repo, element_id="nonexistent")


# =============================================================================
# FIND IMPLEMENTATIONS TESTS
# =============================================================================


class TestFindImplementations:
    """Tests for find_implementations function."""

    def test_find_implementations_returns_implementations(self, mock_es_repo):
        """Test find_implementations returns implementing classes."""
        mock_client = MagicMock()
        mock_es_repo._get_client.return_value = mock_client

        mock_es_repo.get_document.return_value = {
            "element_id": "base_id",
            "name": "BaseClass",
            "element_type": "class",
            "scope": "github",
            "repository": "repo",
            "username": "main",
        }

        mock_client.search.return_value = {
            "hits": {
                "hits": [
                    {
                        "_source": {
                            "element_id": "impl_id",
                            "name": "impl_class",
                            "element_type": "class",
                            "relative_path": "impl.py",
                            "line_start": 1,
                            "raw_code": "class DerivedClass(BaseClass):\n    pass\n",
                        }
                    }
                ]
            }
        }

        result = find_implementations(es=mock_es_repo, element_id="base_id")

        assert isinstance(result, list)

    def test_find_implementations_by_class_name(self, mock_es_repo):
        """Test find_implementations by class name."""
        mock_client = MagicMock()
        mock_es_repo._get_client.return_value = mock_client
        mock_client.search.return_value = {"hits": {"hits": []}}

        result = find_implementations(
            es=mock_es_repo,
            class_name="Protocol",
        )

        assert isinstance(result, list)

    def test_find_implementations_requires_id_or_name(self, mock_es_repo):
        """Test find_implementations requires element_id or class_name."""
        with pytest.raises(ValueError, match="Either element_id or class_name required"):
            find_implementations(es=mock_es_repo)


# =============================================================================
# GET CALL GRAPH TESTS
# =============================================================================


class TestGetCallGraph:
    """Tests for get_call_graph function."""

    def test_get_call_graph_returns_callers_and_callees(self, mock_es_repo):
        """Test get_call_graph returns both callers and callees."""
        mock_client = MagicMock()
        mock_es_repo._get_client.return_value = mock_client

        mock_es_repo.get_document.return_value = {
            "element_id": "func_id",
            "name": "my_func",
            "element_type": "function",
            "relative_path": "funcs.py",
            "raw_code": "def my_func():\n    helper()\n    other_func()\n",
        }

        mock_client.search.return_value = {"hits": {"hits": []}}

        result = get_call_graph(es=mock_es_repo, element_id="func_id")

        assert "element" in result
        assert "callers" in result
        assert "callees" in result
        assert result["element"]["name"] == "my_func"

    def test_get_call_graph_not_found(self, mock_es_repo):
        """Test get_call_graph raises when element not found."""
        mock_es_repo.get_document.return_value = None

        with pytest.raises(ValueError, match="Element not found"):
            get_call_graph(es=mock_es_repo, element_id="nonexistent")

    def test_get_call_graph_direction_callers(self, mock_es_repo):
        """Test get_call_graph with callers only direction."""
        mock_client = MagicMock()
        mock_es_repo._get_client.return_value = mock_client

        mock_es_repo.get_document.return_value = {
            "element_id": "func_id",
            "name": "my_func",
            "element_type": "function",
            "relative_path": "funcs.py",
            "raw_code": "def my_func(): pass",
        }

        mock_client.search.return_value = {"hits": {"hits": []}}

        result = get_call_graph(es=mock_es_repo, element_id="func_id", direction="callers")

        assert "callers" in result
        assert "callees" in result  # Should still be in result but empty


# =============================================================================
# SEARCH CODE FALLBACK TESTS
# =============================================================================


class TestSearchCodeFallback:
    """Tests for search_code fallback behavior."""

    def test_search_code_falls_back_to_keyword(self, mock_es_repo):
        """Test search_code falls back to keyword search when vector fails."""
        mock_es_repo.search_by_keyword.return_value = [
            {
                "element_id": "id1",
                "name": "test",
                "element_type": "function",
                "relative_path": "file.py",
                "line_start": 1,
            }
        ]

        result = search_code(
            es=mock_es_repo,
            embed_client=None,  # No embed client
            query="test",
        )

        assert len(result["code_results"]) == 1
        mock_es_repo.search_by_keyword.assert_called_once()

    def test_search_code_filters_by_language(self, mock_es_repo, mock_embed_client):
        """Test search_code filters results by language."""
        mock_es_repo.search_by_vector.return_value = [
            {"element_id": "id1", "name": "test", "element_type": "function", "relative_path": "file.py", "line_start": 1, "language": "python"},
            {"element_id": "id2", "name": "test", "element_type": "function", "relative_path": "file.ts", "line_start": 1, "language": "typescript"},
        ]

        result = search_code(
            es=mock_es_repo,
            embed_client=mock_embed_client,
            query="test",
            language="python",
        )

        assert len(result["code_results"]) == 1
        assert result["code_results"][0]["element_id"] == "id1"

    def test_search_code_brief_mode(self, mock_es_repo, mock_embed_client):
        """Test search_code brief mode excludes summary."""
        mock_es_repo.search_by_vector.return_value = [
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
            es=mock_es_repo,
            embed_client=mock_embed_client,
            query="test",
            brief=True,
        )

        assert len(result["code_results"]) == 1
        assert "summary" not in result["code_results"][0]
        assert "signature" not in result["code_results"][0]

    def test_search_code_includes_code_when_requested(self, mock_es_repo, mock_embed_client):
        """Test search_code includes code when include_code=True."""
        mock_es_repo.search_by_vector.return_value = [
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
            es=mock_es_repo,
            embed_client=mock_embed_client,
            query="test",
            include_code=True,
        )

        assert len(result["code_results"]) == 1
        assert result["code_results"][0]["code"] == "def test(): pass"

    def test_search_code_method_qualified_name(self, mock_es_repo, mock_embed_client):
        """Test search_code builds qualified name for methods."""
        mock_es_repo.search_by_vector.return_value = [
            {
                "element_id": "method_id",
                "name": "my_method",
                "element_type": "method",
                "relative_path": "file.py",
                "line_start": 10,
                "parent_id": "class_id",
            }
        ]
        mock_es_repo.get_document.return_value = {
            "element_id": "class_id",
            "name": "MyClass",
            "element_type": "class",
        }

        result = search_code(
            es=mock_es_repo,
            embed_client=mock_embed_client,
            query="method",
        )

        assert len(result["code_results"]) == 1
        assert result["code_results"][0]["name"] == "MyClass.my_method"


# =============================================================================
# SEARCH CODE TEST GROUPING TESTS
# =============================================================================


class TestSearchCodeTestGrouping:
    """Tests for search_code test result grouping."""

    def test_groups_test_and_code_results(self, mock_es_repo, mock_embed_client):
        """Test that results are grouped by is_test."""
        mock_es_repo.search_by_vector.return_value = [
            {"element_id": "id1", "name": "UserService", "element_type": "class", "is_test": False, "relative_path": "service.py", "line_start": 1},
            {"element_id": "id2", "name": "test_user_service", "element_type": "function", "is_test": True, "relative_path": "test_service.py", "line_start": 1},
        ]

        result = search_code(
            es=mock_es_repo,
            embed_client=mock_embed_client,
            query="user service",
        )

        assert "code_results" in result
        assert "test_results" in result
        assert len(result["code_results"]) == 1
        assert len(result["test_results"]) == 1
        assert result["code_results"][0]["name"] == "UserService"
        assert result["test_results"][0]["name"] == "test_user_service"

    def test_include_tests_false_excludes_tests(self, mock_es_repo, mock_embed_client):
        """Test that include_tests=False excludes test results."""
        mock_es_repo.search_by_vector.return_value = [
            {"element_id": "id1", "name": "UserService", "element_type": "class", "is_test": False, "relative_path": "service.py", "line_start": 1},
            {"element_id": "id2", "name": "test_user_service", "element_type": "function", "is_test": True, "relative_path": "test_service.py", "line_start": 1},
        ]

        result = search_code(
            es=mock_es_repo,
            embed_client=mock_embed_client,
            query="user service",
            include_tests=False,
        )

        assert len(result["code_results"]) == 1
        assert len(result["test_results"]) == 0

    def test_results_include_is_test_field(self, mock_es_repo, mock_embed_client):
        """Test that individual results include is_test field."""
        mock_es_repo.search_by_vector.return_value = [
            {"element_id": "id1", "name": "foo", "element_type": "function", "is_test": True, "relative_path": "test_foo.py", "line_start": 1},
        ]

        result = search_code(
            es=mock_es_repo,
            embed_client=mock_embed_client,
            query="foo",
        )

        assert result["test_results"][0]["is_test"] is True

    def test_results_include_totals(self, mock_es_repo, mock_embed_client):
        """Test that results include total counts."""
        mock_es_repo.search_by_vector.return_value = [
            {"element_id": "id1", "name": "UserService", "element_type": "class", "is_test": False, "relative_path": "service.py", "line_start": 1},
            {"element_id": "id2", "name": "test_user_service", "element_type": "function", "is_test": True, "relative_path": "test_service.py", "line_start": 1},
        ]

        result = search_code(
            es=mock_es_repo,
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


class TestFindSimilarTestGrouping:
    """Tests for find_similar test result grouping."""

    def test_groups_similar_by_is_test(self, mock_es_repo):
        """Test that similar results are grouped by is_test."""
        mock_es_repo.get_document.return_value = {
            "element_id": "id1",
            "embedding": [0.1] * 1024,
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
            "embedding": [0.1] * 1024,
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
            "embedding": [0.1] * 1024,
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
            "embedding": [0.1] * 1024,
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
# GREP CODE TEST GROUPING TESTS
# =============================================================================


class TestGrepCodeTestGrouping:
    """Tests for grep_code test result grouping."""

    def test_groups_grep_results_by_is_test(self, mock_es_repo):
        """Test that grep results are grouped by is_test."""
        # Mock ES search to return elements with raw_code
        mock_es_repo._get_client().search.return_value = {
            "hits": {
                "hits": [
                    {
                        "_source": {
                            "element_id": "id1",
                            "name": "func",
                            "element_type": "function",
                            "relative_path": "src/app.py",
                            "line_start": 1,
                            "raw_code": "def func():\n    pattern_match\n",
                            "is_test": False,
                        }
                    },
                    {
                        "_source": {
                            "element_id": "id2",
                            "name": "test_func",
                            "element_type": "function",
                            "relative_path": "tests/test_app.py",
                            "line_start": 1,
                            "raw_code": "def test_func():\n    pattern_match\n",
                            "is_test": True,
                        }
                    },
                ]
            }
        }

        result = grep_code(es=mock_es_repo, pattern="pattern_match")

        assert "code_results" in result
        assert "test_results" in result

    def test_include_tests_false_excludes_tests(self, mock_es_repo):
        """Test that include_tests=False excludes test results."""
        mock_es_repo._get_client().search.return_value = {
            "hits": {
                "hits": [
                    {
                        "_source": {
                            "element_id": "id1",
                            "name": "func",
                            "element_type": "function",
                            "relative_path": "src/app.py",
                            "line_start": 1,
                            "raw_code": "def func():\n    pattern_match\n",
                            "is_test": False,
                        }
                    },
                    {
                        "_source": {
                            "element_id": "id2",
                            "name": "test_func",
                            "element_type": "function",
                            "relative_path": "tests/test_app.py",
                            "line_start": 1,
                            "raw_code": "def test_func():\n    pattern_match\n",
                            "is_test": True,
                        }
                    },
                ]
            }
        }

        result = grep_code(es=mock_es_repo, pattern="pattern_match", include_tests=False)

        assert len(result["code_results"]) == 1
        assert len(result["test_results"]) == 0

    def test_results_include_is_test_field(self, mock_es_repo):
        """Test that individual results include is_test field."""
        mock_es_repo._get_client().search.return_value = {
            "hits": {
                "hits": [
                    {
                        "_source": {
                            "element_id": "id1",
                            "name": "test_func",
                            "element_type": "function",
                            "relative_path": "tests/test_app.py",
                            "line_start": 1,
                            "raw_code": "def test_func():\n    pattern_match\n",
                            "is_test": True,
                        }
                    },
                ]
            }
        }

        result = grep_code(es=mock_es_repo, pattern="pattern_match")

        assert len(result["test_results"]) == 1
        assert result["test_results"][0]["is_test"] is True

    def test_results_include_totals(self, mock_es_repo):
        """Test that results include total counts."""
        mock_es_repo._get_client().search.return_value = {
            "hits": {
                "hits": [
                    {
                        "_source": {
                            "element_id": "id1",
                            "name": "func",
                            "element_type": "function",
                            "relative_path": "src/app.py",
                            "line_start": 1,
                            "raw_code": "def func():\n    pattern_match\n",
                            "is_test": False,
                        }
                    },
                    {
                        "_source": {
                            "element_id": "id2",
                            "name": "test_func",
                            "element_type": "function",
                            "relative_path": "tests/test_app.py",
                            "line_start": 1,
                            "raw_code": "def test_func():\n    pattern_match\n",
                            "is_test": True,
                        }
                    },
                ]
            }
        }

        result = grep_code(es=mock_es_repo, pattern="pattern_match")

        assert "total_code" in result
        assert "total_tests" in result
        assert result["total_code"] == 1
        assert result["total_tests"] == 1


# =============================================================================
# SEARCH FEATURES GLOSSARY FILTER TESTS
# =============================================================================


class TestSearchFeaturesGlossaryFilter:
    """Tests for search_features glossary filtering."""

    @pytest.fixture
    def mock_es(self):
        return MagicMock()

    @pytest.fixture
    def mock_embed(self):
        embed = MagicMock()
        embed.embed_single.return_value = [0.1] * 768
        return embed

    def test_filters_by_glossary_term(self, mock_es, mock_embed):
        """Features are filtered to those associated with glossary term."""
        # Setup: glossary term has feature associations
        mock_es.get_glossary_term.return_value = {
            "term": "user",
            "feature_associations": [
                {"feature_id": "f1", "percentage": 50.0},
                {"feature_id": "f2", "percentage": 20.0},
            ],
        }
        # Setup: search returns 3 features
        mock_es.search_by_vector.return_value = [
            {"element_id": "f1", "cluster_label": "Auth", "member_count": 10},
            {"element_id": "f2", "cluster_label": "Users", "member_count": 5},
            {"element_id": "f3", "cluster_label": "Email", "member_count": 8},
        ]

        result = search_features(
            mock_es, mock_embed, query="test",
            scope="s", repository="r",
            glossary_term="user",
        )

        # Only f1 and f2 should be returned (they're in glossary associations)
        assert len(result) == 2
        labels = [r["label"] for r in result]
        assert "Auth" in labels
        assert "Users" in labels
        assert "Email" not in labels

    def test_filters_by_min_percentage(self, mock_es, mock_embed):
        """Features below min_percentage are filtered out."""
        mock_es.get_glossary_term.return_value = {
            "term": "user",
            "feature_associations": [
                {"feature_id": "f1", "percentage": 50.0},
                {"feature_id": "f2", "percentage": 20.0},
            ],
        }
        mock_es.search_by_vector.return_value = [
            {"element_id": "f1", "cluster_label": "Auth", "member_count": 10},
            {"element_id": "f2", "cluster_label": "Users", "member_count": 5},
        ]

        result = search_features(
            mock_es, mock_embed, query="test",
            scope="s", repository="r",
            glossary_term="user",
            min_percentage=30.0,
        )

        # Only f1 (50%) should pass the 30% threshold
        assert len(result) == 1
        assert result[0]["label"] == "Auth"

    def test_no_filter_when_glossary_term_not_provided(self, mock_es, mock_embed):
        """Without glossary_term, all results are returned."""
        mock_es.search_by_vector.return_value = [
            {"element_id": "f1", "cluster_label": "Auth", "member_count": 10},
            {"element_id": "f2", "cluster_label": "Users", "member_count": 5},
        ]

        result = search_features(
            mock_es, mock_embed, query="test",
            scope="s", repository="r",
        )

        assert len(result) == 2
        mock_es.get_glossary_term.assert_not_called()

    def test_returns_empty_when_glossary_term_not_found(self, mock_es, mock_embed):
        """When glossary term doesn't exist, return empty results."""
        mock_es.get_glossary_term.return_value = None
        mock_es.search_by_vector.return_value = [
            {"element_id": "f1", "cluster_label": "Auth", "member_count": 10},
        ]

        result = search_features(
            mock_es, mock_embed, query="test",
            scope="s", repository="r",
            glossary_term="nonexistent",
        )

        assert len(result) == 0


# =============================================================================
# GET FEATURE MEMBERS GLOSSARY TESTS
# =============================================================================


class TestGetFeatureMembersGlossary:
    """Tests for get_feature_members glossary_terms field."""

    @pytest.fixture
    def mock_es(self):
        return MagicMock()

    def test_returns_glossary_terms_for_feature(self, mock_es):
        """Feature members response includes associated glossary terms."""
        # Setup feature
        mock_es.get_document.side_effect = [
            # Feature document
            {"member_ids": ["elem1", "elem2"]},
            # Member 1
            {"element_id": "elem1", "name": "UserService", "element_type": "class"},
            # Member 2
            {"element_id": "elem2", "name": "createUser", "element_type": "function"},
        ]
        # Setup glossary terms
        mock_es.get_glossary_terms.return_value = [
            {
                "term": "user",
                "feature_associations": [
                    {"feature_id": "scope:repo:main:feature:1", "frequency": 2, "percentage": 100.0},
                ],
            },
            {
                "term": "email",
                "feature_associations": [
                    {"feature_id": "scope:repo:main:feature:2", "frequency": 1, "percentage": 50.0},
                ],
            },
        ]

        result = get_feature_members(mock_es, "scope:repo:main:feature:1")

        assert "members" in result
        assert "glossary_terms" in result
        assert len(result["glossary_terms"]) == 1
        assert result["glossary_terms"][0]["term"] == "user"
        assert result["glossary_terms"][0]["percentage"] == 100.0

    def test_glossary_terms_sorted_by_percentage(self, mock_es):
        """Glossary terms are sorted by percentage descending."""
        mock_es.get_document.side_effect = [
            {"member_ids": ["elem1"]},
            {"element_id": "elem1", "name": "Test", "element_type": "class"},
        ]
        mock_es.get_glossary_terms.return_value = [
            {
                "term": "low",
                "feature_associations": [
                    {"feature_id": "s:r:main:feature:1", "frequency": 1, "percentage": 20.0},
                ],
            },
            {
                "term": "high",
                "feature_associations": [
                    {"feature_id": "s:r:main:feature:1", "frequency": 5, "percentage": 80.0},
                ],
            },
        ]

        result = get_feature_members(mock_es, "s:r:main:feature:1")

        assert result["glossary_terms"][0]["term"] == "high"
        assert result["glossary_terms"][1]["term"] == "low"

    def test_empty_glossary_terms_when_no_associations(self, mock_es):
        """Returns empty glossary_terms when feature has no associations."""
        mock_es.get_document.side_effect = [
            {"member_ids": ["elem1"]},
            {"element_id": "elem1", "name": "Test", "element_type": "class"},
        ]
        mock_es.get_glossary_terms.return_value = []

        result = get_feature_members(mock_es, "s:r:main:feature:1")

        assert result["glossary_terms"] == []

    def test_members_still_returned(self, mock_es):
        """Members are still returned in the new format."""
        mock_es.get_document.side_effect = [
            {"member_ids": ["elem1"]},
            {"element_id": "elem1", "name": "MyClass", "element_type": "class",
             "relative_path": "src/file.py", "line_start": 10},
        ]
        mock_es.get_glossary_terms.return_value = []

        result = get_feature_members(mock_es, "s:r:main:feature:1")

        assert len(result["members"]) == 1
        assert result["members"][0]["name"] == "MyClass"

    def test_empty_members_returns_empty_dict(self, mock_es):
        """Empty feature returns proper structure with empty lists."""
        mock_es.get_document.return_value = {
            "element_id": "s:r:main:feature:1",
            "member_ids": [],
        }
        mock_es.get_glossary_terms.return_value = []

        result = get_feature_members(mock_es, "s:r:main:feature:1")

        assert result == {"members": [], "glossary_terms": []}

    def test_handles_malformed_feature_id(self, mock_es):
        """Handles feature_id with fewer than 3 parts gracefully."""
        mock_es.get_document.side_effect = [
            {"member_ids": ["elem1"]},
            {"element_id": "elem1", "name": "Test", "element_type": "class"},
        ]
        # get_glossary_terms should not be called with invalid parts

        result = get_feature_members(mock_es, "invalid")

        # Should still return members but with empty glossary terms
        assert "members" in result
        assert "glossary_terms" in result
        assert result["glossary_terms"] == []


# =============================================================================
# PATTERN SEARCH TESTS
# =============================================================================


class TestPatternSearch:
    """Tests for pattern_search function."""

    def test_pattern_search_regexp_mode(self, mock_es_repo):
        """Test pattern_search with regexp mode."""
        mock_es_repo.search_by_regexp.return_value = [
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
            es=mock_es_repo,
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
        mock_es_repo.search_by_regexp.assert_called_once_with(
            pattern="add_column.*Model",
            scope="test",
            repository="repo",
            username=None,
            glob=None,
            size=50,
            include_tests=True,
        )

    def test_pattern_search_wildcard_mode(self, mock_es_repo):
        """Test pattern_search with wildcard mode."""
        mock_es_repo.search_by_wildcard.return_value = []

        result = pattern_search(
            es=mock_es_repo,
            pattern="*column*",
            mode="wildcard",
            scope="test",
            repository="repo",
        )

        assert "code_results" in result
        assert result["mode"] == "wildcard"
        mock_es_repo.search_by_wildcard.assert_called_once_with(
            pattern="*column*",
            scope="test",
            repository="repo",
            username=None,
            glob=None,
            size=50,
            include_tests=True,
        )

    def test_pattern_search_proximity_mode(self, mock_es_repo):
        """Test pattern_search with proximity mode."""
        mock_es_repo.search_by_proximity.return_value = []

        result = pattern_search(
            es=mock_es_repo,
            pattern="add column Model",
            mode="proximity",
            slop=5,
            scope="test",
            repository="repo",
            username="main",
        )

        assert "code_results" in result
        assert result["mode"] == "proximity"
        mock_es_repo.search_by_proximity.assert_called_once_with(
            terms="add column Model",
            slop=5,
            scope="test",
            repository="repo",
            username="main",
            glob=None,
            size=50,
            include_tests=True,
        )

    def test_pattern_search_invalid_mode(self, mock_es_repo):
        """Test pattern_search with invalid mode raises error."""
        with pytest.raises(ValueError, match="Invalid mode"):
            pattern_search(
                es=mock_es_repo,
                pattern="test",
                mode="invalid",
                scope="test",
                repository="repo",
            )

    def test_pattern_search_groups_test_results(self, mock_es_repo):
        """Test pattern_search groups code and test results separately."""
        mock_es_repo.search_by_regexp.return_value = [
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
            es=mock_es_repo,
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

    def test_pattern_search_with_glob_filter(self, mock_es_repo):
        """Test pattern_search passes glob filter to ES method."""
        mock_es_repo.search_by_regexp.return_value = []

        pattern_search(
            es=mock_es_repo,
            pattern="test",
            mode="regexp",
            scope="test",
            repository="repo",
            glob="*.py",
        )

        mock_es_repo.search_by_regexp.assert_called_once()
        call_kwargs = mock_es_repo.search_by_regexp.call_args[1]
        assert call_kwargs["glob"] == "*.py"

    def test_pattern_search_with_custom_limit(self, mock_es_repo):
        """Test pattern_search passes limit to ES method."""
        mock_es_repo.search_by_wildcard.return_value = []

        pattern_search(
            es=mock_es_repo,
            pattern="*test*",
            mode="wildcard",
            scope="test",
            repository="repo",
            limit=100,
        )

        mock_es_repo.search_by_wildcard.assert_called_once()
        call_kwargs = mock_es_repo.search_by_wildcard.call_args[1]
        assert call_kwargs["size"] == 100

    def test_pattern_search_with_include_tests_false(self, mock_es_repo):
        """Test pattern_search passes include_tests to ES method."""
        mock_es_repo.search_by_proximity.return_value = []

        pattern_search(
            es=mock_es_repo,
            pattern="test pattern",
            mode="proximity",
            scope="test",
            repository="repo",
            include_tests=False,
        )

        mock_es_repo.search_by_proximity.assert_called_once()
        call_kwargs = mock_es_repo.search_by_proximity.call_args[1]
        assert call_kwargs["include_tests"] is False


# =============================================================================
# FIND USAGES WITH PATTERN SEARCH TESTS
# =============================================================================


class TestFindUsagesWithPatternSearch:
    """Tests for find_usages using search_by_regexp."""

    def test_find_usages_uses_regexp_search(self, mock_es_repo):
        """Test that find_usages uses search_by_regexp internally."""
        mock_es_repo.get_document.return_value = {
            "element_id": "test:repo:main:file.py:function:my_func:10",
            "name": "my_func",
            "element_type": "function",
            "relative_path": "file.py",
            "line_start": 10,
            "scope": "test",
            "repository": "repo",
            "username": "main",
        }
        mock_es_repo.search_by_regexp.return_value = [
            {
                "element_id": "test:repo:main:other.py:function:caller:20",
                "name": "caller",
                "element_type": "function",
                "relative_path": "other.py",
                "line_start": 20,
                "raw_code": "def caller():\n    my_func()",
                "is_test": False,
            }
        ]

        result = find_usages(
            es=mock_es_repo,
            element_id="test:repo:main:file.py:function:my_func:10",
        )

        # Verify search_by_regexp was called (not the old client.search or grep_code)
        mock_es_repo.search_by_regexp.assert_called()
        assert len(result) >= 0  # May filter out definition

    def test_find_usages_builds_function_call_pattern(self, mock_es_repo):
        """Test that find_usages builds correct Lucene regexp for function calls."""
        mock_es_repo.get_document.return_value = {
            "element_id": "test:repo:main:file.py:function:my_func:10",
            "name": "my_func",
            "element_type": "function",
            "relative_path": "file.py",
            "line_start": 10,
            "scope": "test",
            "repository": "repo",
            "username": "main",
        }
        mock_es_repo.search_by_regexp.return_value = []

        find_usages(
            es=mock_es_repo,
            element_id="test:repo:main:file.py:function:my_func:10",
        )

        # Verify the pattern is Lucene-compatible (name followed by paren)
        call_args = mock_es_repo.search_by_regexp.call_args
        pattern = call_args[1]["pattern"]
        # Should match function name followed by optional spaces then paren
        assert "my_func" in pattern
        assert "\\(" in pattern  # Escaped paren for Lucene

    def test_find_usages_builds_method_call_pattern(self, mock_es_repo):
        """Test that find_usages builds correct Lucene regexp for method calls."""
        mock_es_repo.get_document.return_value = {
            "element_id": "test:repo:main:file.py:method:my_method:10",
            "name": "my_method",
            "element_type": "method",
            "relative_path": "file.py",
            "line_start": 10,
            "scope": "test",
            "repository": "repo",
            "username": "main",
        }
        mock_es_repo.search_by_regexp.return_value = []

        find_usages(
            es=mock_es_repo,
            element_id="test:repo:main:file.py:method:my_method:10",
        )

        # Verify the pattern includes dot before method name
        call_args = mock_es_repo.search_by_regexp.call_args
        pattern = call_args[1]["pattern"]
        assert "\\.my_method" in pattern  # Dot then method name

    def test_find_usages_builds_class_reference_pattern(self, mock_es_repo):
        """Test that find_usages builds correct Lucene regexp for class references."""
        mock_es_repo.get_document.return_value = {
            "element_id": "test:repo:main:file.py:class:MyClass:10",
            "name": "MyClass",
            "element_type": "class",
            "relative_path": "file.py",
            "line_start": 10,
            "scope": "test",
            "repository": "repo",
            "username": "main",
        }
        mock_es_repo.search_by_regexp.return_value = []

        find_usages(
            es=mock_es_repo,
            element_id="test:repo:main:file.py:class:MyClass:10",
        )

        # Verify the pattern contains the class name
        call_args = mock_es_repo.search_by_regexp.call_args
        pattern = call_args[1]["pattern"]
        assert "MyClass" in pattern

    def test_find_usages_filters_definitions(self, mock_es_repo):
        """Test that find_usages filters out definition lines."""
        mock_es_repo.get_document.return_value = {
            "element_id": "test:repo:main:file.py:function:my_func:10",
            "name": "my_func",
            "element_type": "function",
            "relative_path": "file.py",
            "line_start": 10,
            "scope": "test",
            "repository": "repo",
            "username": "main",
        }
        mock_es_repo.search_by_regexp.return_value = [
            {
                "element_id": "test:repo:main:file.py:function:my_func:10",
                "name": "my_func",
                "element_type": "function",
                "relative_path": "file.py",
                "line_start": 10,
                "raw_code": "def my_func():\n    pass",
                "is_test": False,
            },
            {
                "element_id": "test:repo:main:other.py:function:caller:20",
                "name": "caller",
                "element_type": "function",
                "relative_path": "other.py",
                "line_start": 20,
                "raw_code": "def caller():\n    my_func()",
                "is_test": False,
            },
        ]

        result = find_usages(
            es=mock_es_repo,
            element_id="test:repo:main:file.py:function:my_func:10",
        )

        # The definition in file.py should be filtered out
        assert len(result) == 1
        assert result[0]["file"] == "other.py"

    def test_find_usages_escapes_special_chars_for_lucene(self, mock_es_repo):
        """Test that find_usages escapes special chars for Lucene regexp."""
        mock_es_repo.get_document.return_value = {
            "element_id": "test:repo:main:file.py:function:func_with_dots:10",
            "name": "func.with.dots",  # Name with dots (unusual but possible)
            "element_type": "function",
            "relative_path": "file.py",
            "line_start": 10,
            "scope": "test",
            "repository": "repo",
            "username": "main",
        }
        mock_es_repo.search_by_regexp.return_value = []

        find_usages(
            es=mock_es_repo,
            element_id="test:repo:main:file.py:function:func_with_dots:10",
        )

        # Verify dots in name are escaped for Lucene
        call_args = mock_es_repo.search_by_regexp.call_args
        pattern = call_args[1]["pattern"]
        # Dots should be escaped as \.
        assert "\\." in pattern


# =============================================================================
# GREP CODE DEPRECATION TESTS
# =============================================================================


class TestGrepCodeDeprecation:
    """Tests for grep_code deprecation."""

    def test_grep_code_emits_deprecation_warning(self, mock_es_repo):
        """Test that grep_code emits a deprecation warning."""
        mock_es_repo._get_client.return_value.search.return_value = {
            "hits": {"hits": []}
        }

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            grep_code(
                es=mock_es_repo,
                pattern="test",
                scope="test",
                repository="repo",
            )

            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "pattern_search" in str(w[0].message)


# =============================================================================
# FIND IMPLEMENTATIONS WITH PATTERN SEARCH TESTS
# =============================================================================


class TestFindImplementationsWithPatternSearch:
    """Tests for find_implementations using pattern_search."""

    def test_find_implementations_uses_regexp_search(self, mock_es_repo):
        """Test that find_implementations uses search_by_regexp."""
        mock_es_repo.get_document.return_value = {
            "element_id": "test:repo:main:base.py:class:BaseClass:1",
            "name": "BaseClass",
            "element_type": "class",
            "scope": "test",
            "repository": "repo",
        }
        mock_es_repo.search_by_regexp.return_value = [
            {
                "element_id": "test:repo:main:impl.py:class:MyImpl:10",
                "name": "MyImpl",
                "element_type": "class",
                "relative_path": "impl.py",
                "line_start": 10,
                "raw_code": "class MyImpl(BaseClass):\n    pass",
                "is_test": False,
            }
        ]

        result = find_implementations(
            es=mock_es_repo,
            element_id="test:repo:main:base.py:class:BaseClass:1",
        )

        mock_es_repo.search_by_regexp.assert_called()
        assert len(result) >= 1

    def test_find_implementations_builds_inheritance_pattern(self, mock_es_repo):
        """Test that find_implementations builds correct Lucene pattern for inheritance."""
        mock_es_repo.get_document.return_value = {
            "element_id": "test:repo:main:base.py:class:BaseClass:1",
            "name": "BaseClass",
            "element_type": "class",
            "scope": "test",
            "repository": "repo",
        }
        mock_es_repo.search_by_regexp.return_value = []

        find_implementations(
            es=mock_es_repo,
            element_id="test:repo:main:base.py:class:BaseClass:1",
        )

        # Verify pattern looks for class inheritance
        call_args = mock_es_repo.search_by_regexp.call_args
        pattern = call_args[1]["pattern"]
        # Pattern should match: class SomeClass(BaseClass) or class SomeClass(Other, BaseClass)
        assert "class" in pattern
        assert "BaseClass" in pattern
        assert "\\(" in pattern  # Escaped paren for Lucene

    def test_find_implementations_by_class_name_uses_regexp(self, mock_es_repo):
        """Test find_implementations by class_name also uses search_by_regexp."""
        mock_es_repo.search_by_regexp.return_value = []

        find_implementations(
            es=mock_es_repo,
            class_name="Protocol",
            scope="test",
            repository="repo",
        )

        mock_es_repo.search_by_regexp.assert_called()
        call_args = mock_es_repo.search_by_regexp.call_args
        pattern = call_args[1]["pattern"]
        assert "Protocol" in pattern

    def test_find_implementations_extracts_class_name(self, mock_es_repo):
        """Test that find_implementations correctly extracts implementing class names."""
        mock_es_repo.get_document.return_value = {
            "element_id": "test:repo:main:base.py:class:BaseClass:1",
            "name": "BaseClass",
            "element_type": "class",
            "scope": "test",
            "repository": "repo",
        }
        mock_es_repo.search_by_regexp.return_value = [
            {
                "element_id": "test:repo:main:impl.py:class:DerivedClass:10",
                "name": "DerivedClass",
                "element_type": "class",
                "relative_path": "impl.py",
                "line_start": 10,
                "raw_code": "class DerivedClass(BaseClass):\n    pass",
                "is_test": False,
            }
        ]

        result = find_implementations(
            es=mock_es_repo,
            element_id="test:repo:main:base.py:class:BaseClass:1",
        )

        assert len(result) == 1
        assert result[0]["class_name"] == "DerivedClass"
        assert result[0]["file"] == "impl.py"
        assert result[0]["line"] == 10

    def test_find_implementations_escapes_special_chars(self, mock_es_repo):
        """Test that find_implementations escapes special chars for Lucene."""
        mock_es_repo.get_document.return_value = {
            "element_id": "test:repo:main:base.py:class:Base.Class:1",
            "name": "Base.Class",  # Name with dot (unusual)
            "element_type": "class",
            "scope": "test",
            "repository": "repo",
        }
        mock_es_repo.search_by_regexp.return_value = []

        find_implementations(
            es=mock_es_repo,
            element_id="test:repo:main:base.py:class:Base.Class:1",
        )

        call_args = mock_es_repo.search_by_regexp.call_args
        pattern = call_args[1]["pattern"]
        # Dot should be escaped
        assert "\\." in pattern
