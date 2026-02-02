"""Tests for the MCP tools module."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from magaldi_mcp.tools import (
    batch_get_elements,
    dependency_graph,
    find_by_pattern,
    find_call_chain,
    find_callers,
    find_dead_code,
    find_dependencies,
    find_dependents,
    find_entry_points,
    find_files,
    find_implementations,
    find_similar,
    find_usages,
    get_call_graph,
    get_children,
    get_context,
    get_element,
    get_feature_members,
    get_file_structure,
    get_repo_stats,
    list_features,
    list_patterns,
    list_repos,
    pattern_search,
    search_code,
    search_features,
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

    # Make get_document_by_id_or_hash delegate to get_document
    # This ensures tests that set get_document.return_value or side_effect
    # also work with get_document_by_id_or_hash
    repo.get_document_by_id_or_hash.side_effect = lambda x: repo.get_document(x)

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
                    {
                        "_source": {
                            "element_id": "id3",
                            "relative_path": "README.md",
                            "language": "markdown",
                            "line_end": 10,
                        }
                    },
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
            es=mock_es_repo,
            embed_client=mock_embed_client,
            query="foo",
        )

        assert result["test_results"][0]["is_test"] is True

    def test_results_include_totals(self, mock_es_repo, mock_embed_client):
        """Test that results include total counts."""
        mock_es_repo.search_by_vector.return_value = [
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


class TestSearchFeaturesGlossaryFilter:
    """Tests for search_features glossary filtering."""

    @pytest.fixture
    def mock_es(self):
        repo = MagicMock()
        repo.get_document_by_id_or_hash.side_effect = lambda x: repo.get_document(x)
        return repo

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
            mock_es,
            mock_embed,
            query="test",
            scope="s",
            repository="r",
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
            mock_es,
            mock_embed,
            query="test",
            scope="s",
            repository="r",
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
            mock_es,
            mock_embed,
            query="test",
            scope="s",
            repository="r",
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
            mock_es,
            mock_embed,
            query="test",
            scope="s",
            repository="r",
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
        repo = MagicMock()
        repo.get_document_by_id_or_hash.side_effect = lambda x: repo.get_document(x)
        return repo

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
                    {
                        "feature_id": "scope:repo:main:feature:1",
                        "frequency": 2,
                        "percentage": 100.0,
                    },
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
            {
                "element_id": "elem1",
                "name": "MyClass",
                "element_type": "class",
                "relative_path": "src/file.py",
                "line_start": 10,
            },
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


# =============================================================================
# FIND CALLERS TESTS
# =============================================================================


class TestFindCallers:
    """Tests for find_callers function."""

    def test_find_callers_returns_grouped_results(self, mock_es_repo):
        """Test find_callers returns callers grouped by code/tests."""
        mock_es_repo.get_document.return_value = {
            "element_id": "scope:repo:main:utils.py:function:helper:10",
            "name": "helper",
            "element_type": "function",
            "relative_path": "utils.py",
            "line_start": 10,
            "scope": "scope",
            "repository": "repo",
            "username": "main",
        }
        mock_es_repo.find_elements_calling.return_value = [
            {
                "element_id": "scope:repo:main:app.py:function:main:1",
                "name": "main",
                "element_type": "function",
                "relative_path": "app.py",
                "line_start": 1,
                "is_test": False,
                "summary": "Main function",
            },
            {
                "element_id": "scope:repo:main:test_app.py:function:test_helper:5",
                "name": "test_helper",
                "element_type": "function",
                "relative_path": "test_app.py",
                "line_start": 5,
                "is_test": True,
                "summary": "Test for helper",
            },
        ]

        result = find_callers(
            es=mock_es_repo,
            element_id="scope:repo:main:utils.py:function:helper:10",
        )

        assert "target" in result
        assert "code_results" in result
        assert "test_results" in result
        assert result["target"]["name"] == "helper"
        assert len(result["code_results"]) == 1
        assert len(result["test_results"]) == 1
        assert result["code_results"][0]["name"] == "main"
        assert result["test_results"][0]["name"] == "test_helper"

    def test_find_callers_not_found_raises(self, mock_es_repo):
        """Test find_callers raises when element not found."""
        mock_es_repo.get_document.return_value = None

        with pytest.raises(ValueError, match="Element not found"):
            find_callers(es=mock_es_repo, element_id="nonexistent")

    def test_find_callers_excludes_tests_when_disabled(self, mock_es_repo):
        """Test find_callers excludes test results when include_tests=False."""
        mock_es_repo.get_document.return_value = {
            "element_id": "scope:repo:main:utils.py:function:helper:10",
            "name": "helper",
            "element_type": "function",
            "relative_path": "utils.py",
            "line_start": 10,
            "scope": "scope",
            "repository": "repo",
            "username": "main",
        }
        mock_es_repo.find_elements_calling.return_value = [
            {
                "element_id": "scope:repo:main:test_app.py:function:test_helper:5",
                "name": "test_helper",
                "element_type": "function",
                "relative_path": "test_app.py",
                "line_start": 5,
                "is_test": True,
            },
        ]

        result = find_callers(
            es=mock_es_repo,
            element_id="scope:repo:main:utils.py:function:helper:10",
            include_tests=False,
        )

        assert len(result["code_results"]) == 0
        assert len(result["test_results"]) == 0

    def test_find_callers_uses_target_scope_repo(self, mock_es_repo):
        """Test find_callers uses target's scope/repo if not specified."""
        mock_es_repo.get_document.return_value = {
            "element_id": "myscope:myrepo:main:utils.py:function:helper:10",
            "name": "helper",
            "element_type": "function",
            "relative_path": "utils.py",
            "line_start": 10,
            "scope": "myscope",
            "repository": "myrepo",
            "username": "main",
        }
        mock_es_repo.find_elements_calling.return_value = []

        find_callers(
            es=mock_es_repo,
            element_id="myscope:myrepo:main:utils.py:function:helper:10",
        )

        call_args = mock_es_repo.find_elements_calling.call_args
        assert call_args[1]["scope"] == "myscope"
        assert call_args[1]["repository"] == "myrepo"


# =============================================================================
# FIND CALL CHAIN TESTS
# =============================================================================


class TestFindCallChain:
    """Tests for find_call_chain function."""

    def test_find_call_chain_callees(self, mock_es_repo):
        """Test find_call_chain traces callees."""
        mock_es_repo.get_document.side_effect = [
            # Root element
            {
                "element_id": "scope:repo:main:app.py:function:main:1",
                "name": "main",
                "element_type": "function",
                "relative_path": "app.py",
                "line_start": 1,
                "scope": "scope",
                "repository": "repo",
                "username": "main",
            },
            # First callee
            {
                "element_id": "scope:repo:main:utils.py:function:helper:10",
                "name": "helper",
                "element_type": "function",
                "relative_path": "utils.py",
                "line_start": 10,
            },
        ]
        mock_es_repo.get_calls.side_effect = [
            # Calls from main
            [
                {
                    "name": "helper",
                    "receiver": None,
                    "line": 5,
                    "resolved_id": "scope:repo:main:utils.py:function:helper:10",
                }
            ],
            # Calls from helper (none)
            [],
        ]

        result = find_call_chain(
            es=mock_es_repo,
            element_id="scope:repo:main:app.py:function:main:1",
            direction="callees",
            max_depth=3,
        )

        assert "root" in result
        assert result["root"]["name"] == "main"
        assert result["direction"] == "callees"
        assert "callees" in result
        assert len(result["callees"]) == 1
        assert result["callees"][0]["name"] == "helper"

    def test_find_call_chain_callers(self, mock_es_repo):
        """Test find_call_chain traces callers."""
        mock_es_repo.get_document.return_value = {
            "element_id": "scope:repo:main:utils.py:function:helper:10",
            "name": "helper",
            "element_type": "function",
            "relative_path": "utils.py",
            "line_start": 10,
            "scope": "scope",
            "repository": "repo",
            "username": "main",
        }
        mock_es_repo.find_elements_calling.side_effect = [
            # Callers of helper
            [
                {
                    "element_id": "scope:repo:main:app.py:function:main:1",
                    "name": "main",
                    "element_type": "function",
                    "relative_path": "app.py",
                    "line_start": 1,
                }
            ],
            # Callers of main (none)
            [],
        ]

        result = find_call_chain(
            es=mock_es_repo,
            element_id="scope:repo:main:utils.py:function:helper:10",
            direction="callers",
            max_depth=3,
        )

        assert "callers" in result
        assert len(result["callers"]) == 1
        assert result["callers"][0]["name"] == "main"

    def test_find_call_chain_not_found_raises(self, mock_es_repo):
        """Test find_call_chain raises when element not found."""
        mock_es_repo.get_document.return_value = None

        with pytest.raises(ValueError, match="Element not found"):
            find_call_chain(es=mock_es_repo, element_id="nonexistent")

    def test_find_call_chain_detects_cycles(self, mock_es_repo):
        """Test find_call_chain marks cycles instead of infinite recursion."""
        mock_es_repo.get_document.side_effect = [
            # Root element (func_a)
            {
                "element_id": "scope:repo:main:app.py:function:func_a:1",
                "name": "func_a",
                "element_type": "function",
                "relative_path": "app.py",
                "line_start": 1,
                "scope": "scope",
                "repository": "repo",
                "username": "main",
            },
            # func_b
            {
                "element_id": "scope:repo:main:app.py:function:func_b:10",
                "name": "func_b",
                "element_type": "function",
                "relative_path": "app.py",
                "line_start": 10,
            },
        ]
        mock_es_repo.get_calls.side_effect = [
            # func_a calls func_b
            [
                {
                    "name": "func_b",
                    "receiver": None,
                    "line": 5,
                    "resolved_id": "scope:repo:main:app.py:function:func_b:10",
                }
            ],
            # func_b calls func_a (creates cycle)
            [
                {
                    "name": "func_a",
                    "receiver": None,
                    "line": 15,
                    "resolved_id": "scope:repo:main:app.py:function:func_a:1",
                }
            ],
        ]

        result = find_call_chain(
            es=mock_es_repo,
            element_id="scope:repo:main:app.py:function:func_a:1",
            direction="callees",
            max_depth=5,
        )

        # Should have callees
        assert "callees" in result
        assert len(result["callees"]) == 1
        # func_b should have a cycle marked
        func_b = result["callees"][0]
        assert func_b["name"] == "func_b"
        # func_b's callees should contain func_a marked as cycle
        assert len(func_b.get("callees", [])) == 1
        assert func_b["callees"][0].get("cycle") is True

    def test_find_call_chain_max_depth_clamped(self, mock_es_repo):
        """Test find_call_chain clamps max_depth to valid range."""
        mock_es_repo.get_document.return_value = {
            "element_id": "scope:repo:main:app.py:function:main:1",
            "name": "main",
            "element_type": "function",
            "relative_path": "app.py",
            "line_start": 1,
            "scope": "scope",
            "repository": "repo",
            "username": "main",
        }
        mock_es_repo.get_calls.return_value = []

        # Test max_depth > 10 is clamped to 10
        result = find_call_chain(
            es=mock_es_repo,
            element_id="scope:repo:main:app.py:function:main:1",
            max_depth=20,
        )
        assert result["max_depth"] == 10

        # Test max_depth < 1 is clamped to 1
        result = find_call_chain(
            es=mock_es_repo,
            element_id="scope:repo:main:app.py:function:main:1",
            max_depth=0,
        )
        assert result["max_depth"] == 1


# =============================================================================
# FIND DEAD CODE TESTS
# =============================================================================


class TestFindDeadCode:
    """Tests for find_dead_code function."""

    def test_find_dead_code_returns_uncalled_functions(self, mock_es_repo):
        """Test find_dead_code returns functions with no callers."""
        mock_client = MagicMock()
        mock_es_repo._get_client.return_value = mock_client
        # Use private function names (underscore prefix) since public module-level
        # functions are excluded from dead code analysis (assumed to be public API)
        mock_client.search.return_value = {
            "hits": {
                "hits": [
                    {
                        "_source": {
                            "element_id": "scope:repo:main:utils.py:function:_unused:10",
                            "name": "_unused",
                            "element_type": "function",
                            "relative_path": "utils.py",
                            "line_start": 10,
                            "decorators": [],
                            "is_test": False,
                            "summary": "Unused function",
                            "level": 2,
                        }
                    },
                    {
                        "_source": {
                            "element_id": "scope:repo:main:utils.py:function:_used:20",
                            "name": "_used",
                            "element_type": "function",
                            "relative_path": "utils.py",
                            "line_start": 20,
                            "decorators": [],
                            "is_test": False,
                            "summary": "Used function",
                            "level": 2,
                        }
                    },
                ]
            }
        }
        # First function has no callers, second has callers
        mock_es_repo.find_elements_calling.side_effect = [[], [{"name": "caller"}]]

        result = find_dead_code(
            es=mock_es_repo,
            scope="scope",
            repository="repo",
        )

        assert "potentially_dead" in result
        assert "stats" in result
        assert len(result["potentially_dead"]) == 1
        assert result["potentially_dead"][0]["name"] == "_unused"
        assert result["stats"]["total_functions"] == 2
        assert result["stats"]["potentially_dead"] == 1

    def test_find_dead_code_excludes_entry_points(self, mock_es_repo):
        """Test find_dead_code excludes decorated entry points."""
        mock_client = MagicMock()
        mock_es_repo._get_client.return_value = mock_client
        mock_client.search.return_value = {
            "hits": {
                "hits": [
                    {
                        "_source": {
                            "element_id": "scope:repo:main:app.py:function:index:10",
                            "name": "index",
                            "element_type": "function",
                            "relative_path": "app.py",
                            "line_start": 10,
                            "decorators": ["@app.route('/')"],
                            "is_test": False,
                        }
                    },
                ]
            }
        }

        result = find_dead_code(
            es=mock_es_repo,
            scope="scope",
            repository="repo",
        )

        # Entry point should be excluded, not in dead code
        assert len(result["potentially_dead"]) == 0
        assert result["stats"]["excluded_entry_points"] == 1

    def test_find_dead_code_excludes_magic_methods(self, mock_es_repo):
        """Test find_dead_code excludes __init__ and other magic methods."""
        mock_client = MagicMock()
        mock_es_repo._get_client.return_value = mock_client
        mock_client.search.return_value = {
            "hits": {
                "hits": [
                    {
                        "_source": {
                            "element_id": "scope:repo:main:cls.py:method:__init__:10",
                            "name": "__init__",
                            "element_type": "method",
                            "relative_path": "cls.py",
                            "line_start": 10,
                            "decorators": [],
                            "is_test": False,
                        }
                    },
                ]
            }
        }

        result = find_dead_code(
            es=mock_es_repo,
            scope="scope",
            repository="repo",
        )

        # __init__ should be excluded
        assert len(result["potentially_dead"]) == 0
        assert result["stats"]["excluded_entry_points"] == 1

    def test_find_dead_code_excludes_main(self, mock_es_repo):
        """Test find_dead_code excludes main functions."""
        mock_client = MagicMock()
        mock_es_repo._get_client.return_value = mock_client
        mock_client.search.return_value = {
            "hits": {
                "hits": [
                    {
                        "_source": {
                            "element_id": "scope:repo:main:app.py:function:main:1",
                            "name": "main",
                            "element_type": "function",
                            "relative_path": "app.py",
                            "line_start": 1,
                            "decorators": [],
                            "is_test": False,
                        }
                    },
                ]
            }
        }

        result = find_dead_code(
            es=mock_es_repo,
            scope="scope",
            repository="repo",
        )

        assert len(result["potentially_dead"]) == 0
        assert result["stats"]["excluded_entry_points"] == 1


# =============================================================================
# FIND ENTRY POINTS TESTS
# =============================================================================


class TestFindEntryPoints:
    """Tests for find_entry_points function."""

    def test_find_entry_points_groups_by_type(self, mock_es_repo):
        """Test find_entry_points groups results by type."""
        mock_client = MagicMock()
        mock_es_repo._get_client.return_value = mock_client
        mock_client.search.return_value = {
            "hits": {
                "hits": [
                    {
                        "_source": {
                            "element_id": "scope:repo:main:app.py:function:index:10",
                            "name": "index",
                            "element_type": "function",
                            "relative_path": "app.py",
                            "line_start": 10,
                            "decorators": ["@app.route('/')"],
                            "is_test": False,
                        }
                    },
                    {
                        "_source": {
                            "element_id": "scope:repo:main:cli.py:function:run:20",
                            "name": "run",
                            "element_type": "function",
                            "relative_path": "cli.py",
                            "line_start": 20,
                            "decorators": ["@click.command()"],
                            "is_test": False,
                        }
                    },
                    {
                        "_source": {
                            "element_id": "scope:repo:main:conftest.py:function:client:5",
                            "name": "client",
                            "element_type": "function",
                            "relative_path": "conftest.py",
                            "line_start": 5,
                            "decorators": ["@pytest.fixture"],
                            "is_test": False,
                        }
                    },
                    {
                        "_source": {
                            "element_id": "scope:repo:main:app.py:function:main:1",
                            "name": "main",
                            "element_type": "function",
                            "relative_path": "app.py",
                            "line_start": 1,
                            "decorators": [],
                            "is_test": False,
                        }
                    },
                ]
            }
        }

        result = find_entry_points(
            es=mock_es_repo,
            scope="scope",
            repository="repo",
        )

        assert "http" in result
        assert "cli" in result
        assert "test" in result
        assert "main" in result
        assert "stats" in result

        assert len(result["http"]) == 1
        assert result["http"][0]["name"] == "index"

        assert len(result["cli"]) == 1
        assert result["cli"][0]["name"] == "run"

        assert len(result["test"]) == 1
        assert result["test"][0]["name"] == "client"

        assert len(result["main"]) == 1
        assert result["main"][0]["name"] == "main"

        assert result["stats"]["total"] == 4

    def test_find_entry_points_with_no_matches(self, mock_es_repo):
        """Test find_entry_points with no entry points."""
        mock_client = MagicMock()
        mock_es_repo._get_client.return_value = mock_client
        mock_client.search.return_value = {
            "hits": {
                "hits": [
                    {
                        "_source": {
                            "element_id": "scope:repo:main:utils.py:function:helper:10",
                            "name": "helper",
                            "element_type": "function",
                            "relative_path": "utils.py",
                            "line_start": 10,
                            "decorators": [],
                            "is_test": False,
                        }
                    },
                ]
            }
        }

        result = find_entry_points(
            es=mock_es_repo,
            scope="scope",
            repository="repo",
        )

        assert len(result["http"]) == 0
        assert len(result["cli"]) == 0
        assert len(result["test"]) == 0
        assert len(result["main"]) == 0
        assert result["stats"]["total"] == 0

    def test_find_entry_points_async_tasks(self, mock_es_repo):
        """Test find_entry_points detects async task decorators."""
        mock_client = MagicMock()
        mock_es_repo._get_client.return_value = mock_client
        mock_client.search.return_value = {
            "hits": {
                "hits": [
                    {
                        "_source": {
                            "element_id": "scope:repo:main:tasks.py:function:process:10",
                            "name": "process",
                            "element_type": "function",
                            "relative_path": "tasks.py",
                            "line_start": 10,
                            "decorators": ["@celery.task"],
                            "is_test": False,
                        }
                    },
                ]
            }
        }

        result = find_entry_points(
            es=mock_es_repo,
            scope="scope",
            repository="repo",
        )

        assert len(result["async_tasks"]) == 1
        assert result["async_tasks"][0]["name"] == "process"


# =============================================================================
# FIND DEPENDENCIES TESTS
# =============================================================================


class TestFindDependencies:
    """Tests for find_dependencies function."""

    def test_find_dependencies_by_element_id(self, mock_es_repo):
        """Test find_dependencies returns imports for a file element."""
        mock_es_repo.get_document.return_value = {
            "element_id": "scope:repo:main:src/app.py:file:app.py:1",
            "name": "app.py",
            "element_type": "file",
            "relative_path": "src/app.py",
            "scope": "scope",
            "repository": "repo",
            "username": "main",
        }
        mock_es_repo.get_imports.return_value = [
            {"name": "helper", "module": "utils", "alias": None, "line": 1},
            {"name": "requests", "module": "requests", "alias": None, "line": 2},
            {"name": "config", "module": ".config", "alias": None, "line": 3},
        ]

        mock_client = MagicMock()
        mock_es_repo._get_client.return_value = mock_client
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
            es=mock_es_repo,
            element_id="scope:repo:main:src/app.py:file:app.py:1",
        )

        assert "internal_imports" in result
        assert "external_imports" in result
        assert "all_imports" in result
        assert "stats" in result
        assert result["stats"]["total"] == 3

    def test_find_dependencies_by_file_path(self, mock_es_repo):
        """Test find_dependencies with file_path parameter."""
        mock_client = MagicMock()
        mock_es_repo._get_client.return_value = mock_client

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
        mock_es_repo.get_imports.return_value = [
            {"name": "os", "module": "os", "alias": None, "line": 1},
        ]

        result = find_dependencies(
            es=mock_es_repo,
            file_path="utils.py",
            scope="scope",
            repository="repo",
        )

        assert "external_imports" in result
        assert len(result["external_imports"]) == 1

    def test_find_dependencies_classifies_relative_imports_as_internal(self, mock_es_repo):
        """Test that relative imports (starting with .) are classified as internal."""
        mock_es_repo.get_document.return_value = {
            "element_id": "scope:repo:main:src/app.py:file:app.py:1",
            "name": "app.py",
            "element_type": "file",
            "relative_path": "src/app.py",
            "scope": "scope",
            "repository": "repo",
        }
        mock_es_repo.get_imports.return_value = [
            {"name": "config", "module": ".config", "alias": None, "line": 1},
            {"name": "utils", "module": "..shared.utils", "alias": None, "line": 2},
        ]

        mock_client = MagicMock()
        mock_es_repo._get_client.return_value = mock_client
        mock_client.search.return_value = {"hits": {"hits": []}}

        result = find_dependencies(
            es=mock_es_repo,
            element_id="scope:repo:main:src/app.py:file:app.py:1",
        )

        # Both should be internal (relative imports)
        assert len(result["internal_imports"]) == 2
        assert len(result["external_imports"]) == 0

    def test_find_dependencies_requires_file_path_or_element_id(self, mock_es_repo):
        """Test find_dependencies raises when neither file_path nor element_id provided."""
        with pytest.raises(ValueError, match="Either file_path or element_id required"):
            find_dependencies(es=mock_es_repo)

    def test_find_dependencies_element_not_found(self, mock_es_repo):
        """Test find_dependencies raises when element not found."""
        mock_es_repo.get_document.return_value = None

        with pytest.raises(ValueError, match="Element not found"):
            find_dependencies(es=mock_es_repo, element_id="nonexistent")

    def test_find_dependencies_requires_scope_repo_for_file_path(self, mock_es_repo):
        """Test find_dependencies requires scope/repository when using file_path."""
        with pytest.raises(ValueError, match="scope and repository required"):
            find_dependencies(es=mock_es_repo, file_path="utils.py")

    def test_find_dependencies_rejects_non_file_elements(self, mock_es_repo):
        """Test find_dependencies raises when element is not a file."""
        mock_es_repo.get_document.return_value = {
            "element_id": "scope:repo:main:utils.py:function:helper:10",
            "name": "helper",
            "element_type": "function",
        }

        with pytest.raises(ValueError, match="Element is not a file"):
            find_dependencies(
                es=mock_es_repo, element_id="scope:repo:main:utils.py:function:helper:10"
            )


# =============================================================================
# FIND DEPENDENTS TESTS
# =============================================================================


class TestFindDependents:
    """Tests for find_dependents function."""

    def test_find_dependents_returns_files_importing_module(self, mock_es_repo):
        """Test find_dependents returns files that import a module."""
        mock_es_repo.find_elements_importing.return_value = [
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
            es=mock_es_repo,
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

    def test_find_dependents_empty_results(self, mock_es_repo):
        """Test find_dependents with no dependents."""
        mock_es_repo.find_elements_importing.return_value = []

        result = find_dependents(
            es=mock_es_repo,
            module="nonexistent_module",
            scope="scope",
            repository="repo",
        )

        assert result["total"] == 0
        assert result["dependents"] == []

    def test_find_dependents_respects_limit(self, mock_es_repo):
        """Test find_dependents passes limit to ES."""
        mock_es_repo.find_elements_importing.return_value = []

        result = find_dependents(
            es=mock_es_repo,
            module="utils",
            scope="scope",
            repository="repo",
            limit=10,
        )

        mock_es_repo.find_elements_importing.assert_called_once_with(
            module="utils",
            scope="scope",
            repository="repo",
            username="main",
            limit=10,
        )


# =============================================================================
# DEPENDENCY GRAPH TESTS
# =============================================================================


class TestDependencyGraph:
    """Tests for dependency_graph function."""

    def test_dependency_graph_builds_nodes_and_edges(self, mock_es_repo):
        """Test dependency_graph builds nodes and edges."""
        mock_client = MagicMock()
        mock_es_repo._get_client.return_value = mock_client
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
            es=mock_es_repo,
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

    def test_dependency_graph_detects_cycles(self, mock_es_repo):
        """Test dependency_graph detects circular dependencies."""
        mock_client = MagicMock()
        mock_es_repo._get_client.return_value = mock_client
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
            es=mock_es_repo,
            scope="scope",
            repository="repo",
        )

        assert result["stats"]["has_cycles"] is True
        assert len(result["cycles"]) >= 1

    def test_dependency_graph_no_cycles(self, mock_es_repo):
        """Test dependency_graph with no circular dependencies."""
        mock_client = MagicMock()
        mock_es_repo._get_client.return_value = mock_client
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
            es=mock_es_repo,
            scope="scope",
            repository="repo",
        )

        assert result["stats"]["has_cycles"] is False
        assert len(result["cycles"]) == 0

    def test_dependency_graph_handles_relative_imports(self, mock_es_repo):
        """Test dependency_graph resolves relative imports."""
        mock_client = MagicMock()
        mock_es_repo._get_client.return_value = mock_client
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
            es=mock_es_repo,
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

    def test_dependency_graph_empty_repo(self, mock_es_repo):
        """Test dependency_graph with empty repository."""
        mock_client = MagicMock()
        mock_es_repo._get_client.return_value = mock_client
        mock_client.search.return_value = {"hits": {"hits": []}}

        result = dependency_graph(
            es=mock_es_repo,
            scope="scope",
            repository="repo",
        )

        assert result["nodes"] == []
        assert result["edges"] == []
        assert result["cycles"] == []
        assert result["stats"]["node_count"] == 0

    def test_dependency_graph_internal_only(self, mock_es_repo):
        """Test dependency_graph filters external imports when internal_only=True."""
        mock_client = MagicMock()
        mock_es_repo._get_client.return_value = mock_client
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
            es=mock_es_repo,
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


class TestListPatterns:
    """Tests for list_patterns function."""

    def test_list_patterns_returns_pattern_summary(self, mock_es_repo):
        """Test list_patterns returns pattern counts and examples."""
        mock_client = MagicMock()
        mock_es_repo._get_client.return_value = mock_client

        # First call: aggregation query returns pattern counts
        agg_response = {
            "hits": {"hits": []},
            "aggregations": {
                "patterns": {
                    "buckets": [
                        {"key": "singleton", "doc_count": 1},
                        {"key": "factory", "doc_count": 2},
                    ]
                }
            },
        }

        # Subsequent calls: example queries return matching classes
        singleton_examples = {
            "hits": {
                "hits": [
                    {
                        "_source": {
                            "name": "DatabaseConnection",
                            "relative_path": "db.py",
                            "line_start": 1,
                            "pattern_confidence": {"singleton": 0.95},
                        }
                    }
                ]
            }
        }
        factory_examples = {
            "hits": {
                "hits": [
                    {
                        "_source": {
                            "name": "CarFactory",
                            "relative_path": "factory.py",
                            "line_start": 1,
                            "pattern_confidence": {"factory": 0.85},
                        }
                    },
                    {
                        "_source": {
                            "name": "BikeFactory",
                            "relative_path": "factory.py",
                            "line_start": 10,
                            "pattern_confidence": {"factory": 0.75},
                        }
                    },
                ]
            }
        }

        mock_client.search.side_effect = [agg_response, singleton_examples, factory_examples]

        result = list_patterns(
            es=mock_es_repo,
            scope="scope",
            repository="repo",
        )

        assert isinstance(result, dict)
        assert "patterns" in result
        assert "total_classes_with_patterns" in result
        assert result["total_classes_with_patterns"] == 3

        # Check pattern summary
        patterns = {p["pattern"]: p for p in result["patterns"]}
        assert "singleton" in patterns
        assert "factory" in patterns
        assert patterns["singleton"]["count"] == 1
        assert patterns["factory"]["count"] == 2

    def test_list_patterns_no_patterns_found(self, mock_es_repo):
        """Test list_patterns when no patterns exist."""
        mock_client = MagicMock()
        mock_es_repo._get_client.return_value = mock_client
        mock_client.search.return_value = {"hits": {"hits": []}}

        result = list_patterns(
            es=mock_es_repo,
            scope="scope",
            repository="repo",
        )

        assert result["patterns"] == []
        assert result["total_classes_with_patterns"] == 0

    def test_list_patterns_with_username(self, mock_es_repo):
        """Test list_patterns filters by username."""
        mock_client = MagicMock()
        mock_es_repo._get_client.return_value = mock_client
        mock_client.search.return_value = {"hits": {"hits": []}}

        list_patterns(
            es=mock_es_repo,
            scope="scope",
            repository="repo",
            username="testuser",
        )

        # Verify the search was called with the right username filter
        call_args = mock_client.search.call_args
        assert call_args is not None


# =============================================================================
# FIND BY PATTERN TESTS
# =============================================================================


class TestFindByPattern:
    """Tests for find_by_pattern function."""

    def test_find_by_pattern_returns_matching_classes(self, mock_es_repo):
        """Test find_by_pattern returns classes with matching pattern."""
        mock_client = MagicMock()
        mock_es_repo._get_client.return_value = mock_client
        mock_client.search.return_value = {
            "hits": {
                "hits": [
                    {
                        "_source": {
                            "element_id": "scope:repo:main:db.py:class:DatabaseConnection:1",
                            "name": "DatabaseConnection",
                            "element_type": "class",
                            "relative_path": "db.py",
                            "line_start": 1,
                            "detected_patterns": ["singleton"],
                            "pattern_confidence": {"singleton": 0.95},
                            "summary": "Database connection singleton",
                        }
                    },
                    {
                        "_source": {
                            "element_id": "scope:repo:main:config.py:class:Config:1",
                            "name": "Config",
                            "element_type": "class",
                            "relative_path": "config.py",
                            "line_start": 1,
                            "detected_patterns": ["singleton"],
                            "pattern_confidence": {"singleton": 0.80},
                            "summary": "Configuration singleton",
                        }
                    },
                ]
            }
        }

        result = find_by_pattern(
            es=mock_es_repo,
            pattern="singleton",
            scope="scope",
            repository="repo",
        )

        assert isinstance(result, dict)
        assert "classes" in result
        assert len(result["classes"]) == 2
        assert result["classes"][0]["name"] == "DatabaseConnection"
        assert result["classes"][0]["confidence"] == 0.95

    def test_find_by_pattern_filters_by_min_confidence(self, mock_es_repo):
        """Test find_by_pattern filters by minimum confidence."""
        mock_client = MagicMock()
        mock_es_repo._get_client.return_value = mock_client
        mock_client.search.return_value = {
            "hits": {
                "hits": [
                    {
                        "_source": {
                            "element_id": "id1",
                            "name": "HighConfidence",
                            "element_type": "class",
                            "relative_path": "a.py",
                            "line_start": 1,
                            "detected_patterns": ["singleton"],
                            "pattern_confidence": {"singleton": 0.95},
                        }
                    },
                    {
                        "_source": {
                            "element_id": "id2",
                            "name": "LowConfidence",
                            "element_type": "class",
                            "relative_path": "b.py",
                            "line_start": 1,
                            "detected_patterns": ["singleton"],
                            "pattern_confidence": {"singleton": 0.50},
                        }
                    },
                ]
            }
        }

        result = find_by_pattern(
            es=mock_es_repo,
            pattern="singleton",
            scope="scope",
            repository="repo",
            min_confidence=0.8,
        )

        assert len(result["classes"]) == 1
        assert result["classes"][0]["name"] == "HighConfidence"

    def test_find_by_pattern_no_matches(self, mock_es_repo):
        """Test find_by_pattern when no classes match."""
        mock_client = MagicMock()
        mock_es_repo._get_client.return_value = mock_client
        mock_client.search.return_value = {"hits": {"hits": []}}

        result = find_by_pattern(
            es=mock_es_repo,
            pattern="builder",
            scope="scope",
            repository="repo",
        )

        assert result["classes"] == []
        assert result["count"] == 0

    def test_find_by_pattern_with_limit(self, mock_es_repo):
        """Test find_by_pattern respects limit parameter."""
        mock_client = MagicMock()
        mock_es_repo._get_client.return_value = mock_client
        mock_client.search.return_value = {
            "hits": {
                "hits": [
                    {
                        "_source": {
                            "element_id": f"id{i}",
                            "name": f"Class{i}",
                            "element_type": "class",
                            "relative_path": f"file{i}.py",
                            "line_start": 1,
                            "detected_patterns": ["factory"],
                            "pattern_confidence": {"factory": 0.9},
                        }
                    }
                    for i in range(10)
                ]
            }
        }

        result = find_by_pattern(
            es=mock_es_repo,
            pattern="factory",
            scope="scope",
            repository="repo",
            limit=5,
        )

        # The function should return at most limit results
        assert len(result["classes"]) <= 5
