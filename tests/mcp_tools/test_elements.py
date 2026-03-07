"""Tests for MCP tools - elements category."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from magaldi_mcp.tools import (
    batch_get_elements,
    find_similar,
    get_children,
    get_context,
    get_element,
    get_file_structure,
)

# =============================================================================
# GET ELEMENT TESTS
# =============================================================================


class TestGetElement:
    """Tests for get_element function."""

    def test_get_element_returns_element(self, mock_repo):
        """Test get_element returns element details."""
        mock_repo.get_document.return_value = {
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
            repo=mock_repo,
            element_id="scope:repo:user:file.py:function:test:1",
        )

        assert result is not None
        assert result["name"] == "test"

    def test_get_element_not_found_raises(self, mock_repo):
        """Test get_element raises when element not found."""
        mock_repo.get_document.return_value = None

        with pytest.raises(ValueError, match="Element not found"):
            get_element(
                repo=mock_repo,
                element_id="nonexistent",
            )

    def test_get_element_with_code(self, mock_repo):
        """Test get_element includes code when requested."""
        mock_repo.get_document.return_value = {
            "element_id": "scope:repo:user:file.py:function:test:1",
            "name": "test",
            "element_type": "function",
            "raw_code": "def test(): pass",
        }

        result = get_element(
            repo=mock_repo,
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

    def test_batch_get_returns_multiple_elements(self, mock_repo):
        """Test batch_get_elements returns multiple elements."""
        mock_repo.get_document.side_effect = [
            {"element_id": "id1", "name": "elem1"},
            {"element_id": "id2", "name": "elem2"},
        ]

        result = batch_get_elements(
            repo=mock_repo,
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

    def test_get_element_includes_optional_fields(self, mock_repo):
        """Test get_element includes all optional fields when brief=False."""
        mock_repo.get_document.return_value = {
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

        result = get_element(repo=mock_repo, element_id="id1", brief=False)

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

    def test_find_similar_returns_similar_elements(self, mock_repo):
        """Test find_similar returns similar elements grouped by is_test."""
        mock_repo.get_document.return_value = {
            "element_id": "id1",
            "name": "test",
            "summary_embedding": [0.1] * 1024,
        }
        mock_repo.search_by_vector.return_value = [
            {"element_id": "id2", "name": "similar", "_score": 0.9}
        ]

        result = find_similar(
            repo=mock_repo,
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

    def test_find_similar_element_not_found(self, mock_repo):
        """Test find_similar raises when element not found."""
        mock_repo.get_document.return_value = None

        with pytest.raises(ValueError, match="Element not found"):
            find_similar(repo=mock_repo, element_id="nonexistent")

    def test_find_similar_no_embedding(self, mock_repo):
        """Test find_similar raises when element has no embedding."""
        mock_repo.get_document.return_value = {
            "element_id": "id1",
            "name": "test",
        }

        with pytest.raises(ValueError, match="no embedding"):
            find_similar(repo=mock_repo, element_id="id1")

    def test_find_similar_excludes_self(self, mock_repo):
        """Test find_similar excludes the source element."""
        mock_repo.get_document.return_value = {
            "element_id": "id1",
            "name": "test",
            "summary_embedding": [0.1] * 1024,
        }
        mock_repo.search_by_vector.return_value = [
            {"element_id": "id1", "name": "test", "_score": 1.0},  # Self
            {"element_id": "id2", "name": "similar", "_score": 0.9},
        ]

        result = find_similar(repo=mock_repo, element_id="id1", limit=1)

        assert len(result["code_results"]) == 1
        assert result["code_results"][0]["element_id"] == "id2"

    def test_find_similar_same_repo_only(self, mock_repo):
        """Test find_similar with same_repo_only filter."""
        mock_repo.get_document.return_value = {
            "element_id": "id1",
            "name": "test",
            "summary_embedding": [0.1] * 1024,
            "scope": "github",
            "repository": "myrepo",
        }
        mock_repo.search_by_vector.return_value = []

        result = find_similar(repo=mock_repo, element_id="id1", same_repo_only=True)

        assert isinstance(result, dict)
        assert "code_results" in result
        assert "test_results" in result
        # Verify search_by_vector was called with scope/repo filters
        mock_repo.search_by_vector.assert_called_once()


# =============================================================================
# GET ELEMENT EXTENDED TESTS
# =============================================================================

# =============================================================================
# FIND SIMILAR TEST GROUPING TESTS
# =============================================================================


class TestFindSimilarTestGrouping:
    """Tests for find_similar test result grouping."""

    def test_groups_similar_by_is_test(self, mock_repo):
        """Test that similar results are grouped by is_test."""
        mock_repo.get_document.return_value = {
            "element_id": "id1",
            "summary_embedding": [0.1] * 1024,
        }
        mock_repo.search_by_vector.return_value = [
            {"element_id": "id2", "name": "similar_func", "is_test": False},
            {"element_id": "id3", "name": "test_similar", "is_test": True},
        ]

        result = find_similar(repo=mock_repo, element_id="id1")

        assert "code_results" in result
        assert "test_results" in result

    def test_include_tests_false(self, mock_repo):
        """Test include_tests parameter."""
        mock_repo.get_document.return_value = {
            "element_id": "id1",
            "summary_embedding": [0.1] * 1024,
        }
        mock_repo.search_by_vector.return_value = [
            {"element_id": "id2", "name": "test_func", "is_test": True},
        ]

        result = find_similar(repo=mock_repo, element_id="id1", include_tests=False)

        assert len(result["test_results"]) == 0

    def test_results_include_is_test_field(self, mock_repo):
        """Test that individual results include is_test field."""
        mock_repo.get_document.return_value = {
            "element_id": "id1",
            "summary_embedding": [0.1] * 1024,
        }
        mock_repo.search_by_vector.return_value = [
            {"element_id": "id2", "name": "similar_func", "is_test": False},
        ]

        result = find_similar(repo=mock_repo, element_id="id1")

        assert len(result["code_results"]) == 1
        assert result["code_results"][0]["is_test"] is False

    def test_results_include_totals(self, mock_repo):
        """Test that results include total counts."""
        mock_repo.get_document.return_value = {
            "element_id": "id1",
            "summary_embedding": [0.1] * 1024,
        }
        mock_repo.search_by_vector.return_value = [
            {"element_id": "id2", "name": "similar_func", "is_test": False},
            {"element_id": "id3", "name": "test_similar", "is_test": True},
        ]

        result = find_similar(repo=mock_repo, element_id="id1")

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

    def test_get_context_returns_parent_and_children(self, mock_repo):
        """Test get_context returns context info."""
        mock_repo.get_document.return_value = {
            "element_id": "scope:repo:user:file.py:function:test:1",
            "name": "test",
            "parent_id": "scope:repo:user:file.py:class:MyClass:1",
        }
        mock_repo.get_children.return_value = []

        result = get_context(
            repo=mock_repo,
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

    def test_get_context_with_file_and_parent(self, mock_repo):
        """Test get_context returns file and parent info."""
        mock_repo.get_document.side_effect = [
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
        mock_repo._get_client.return_value = mock_client

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

        result = get_context(repo=mock_repo, element_id="method_id")

        assert result["file"]["name"] == "file.py"
        assert result["parent"]["name"] == "MyClass"

    def test_get_context_with_siblings(self, mock_repo):
        """Test get_context returns siblings when requested."""
        mock_repo.get_document.side_effect = [
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
        mock_repo._get_client.return_value = mock_client

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

        result = get_context(repo=mock_repo, element_id="method1", include_siblings=True)

        # method1 should be excluded from siblings
        assert len(result["siblings"]) == 1
        assert result["siblings"][0]["name"] == "method2"


# =============================================================================
# GET FILE STRUCTURE TESTS
# =============================================================================


class TestGetFileStructure:
    """Tests for get_file_structure expanded element types."""

    def _setup_file_structure(self, mock_repo, elements):
        """Helper to set up mock for get_file_structure."""
        mock_client = MagicMock()
        mock_repo._get_client.return_value = mock_client

        # First search: _find_file_element
        mock_client.search.side_effect = [
            {
                "hits": {
                    "hits": [
                        {
                            "_source": {
                                "element_id": "s:r:main:file.ts:file:file.ts:0",
                                "name": "file.ts",
                                "element_type": "file",
                                "language": "typescript",
                            }
                        }
                    ]
                }
            },
            # Second search: _find_elements_in_file
            {
                "hits": {
                    "hits": [
                        {"_source": e} for e in elements
                    ]
                }
            },
        ]

    def test_includes_interface_elements(self, mock_repo):
        """Test get_file_structure includes interface elements."""
        self._setup_file_structure(mock_repo, [
            {
                "element_id": "s:r:main:file.ts:interface:IUser:5",
                "hash_id": "abc1",
                "name": "IUser",
                "element_type": "interface",
                "line_start": 5,
                "parent_id": "s:r:main:file.ts:file:file.ts:0",
            },
        ])

        result = get_file_structure(
            repo=mock_repo,
            scope="s",
            repository="r",
            file_path="file.ts",
        )

        assert len(result["structure"]) == 1
        assert result["structure"][0]["name"] == "IUser"
        assert result["structure"][0]["type"] == "interface"
        assert result["counts"]["interface"] == 1

    def test_includes_enum_elements(self, mock_repo):
        """Test get_file_structure includes enum elements."""
        self._setup_file_structure(mock_repo, [
            {
                "element_id": "s:r:main:file.ts:enum:Color:1",
                "hash_id": "abc2",
                "name": "Color",
                "element_type": "enum",
                "line_start": 1,
                "parent_id": "s:r:main:file.ts:file:file.ts:0",
            },
        ])

        result = get_file_structure(
            repo=mock_repo,
            scope="s",
            repository="r",
            file_path="file.ts",
        )

        assert len(result["structure"]) == 1
        assert result["structure"][0]["name"] == "Color"
        assert result["structure"][0]["type"] == "enum"
        assert result["counts"]["enum"] == 1

    def test_includes_type_alias_elements(self, mock_repo):
        """Test get_file_structure includes type_alias elements."""
        self._setup_file_structure(mock_repo, [
            {
                "element_id": "s:r:main:file.ts:type_alias:UserId:1",
                "hash_id": "abc3",
                "name": "UserId",
                "element_type": "type_alias",
                "line_start": 1,
                "parent_id": "s:r:main:file.ts:file:file.ts:0",
            },
        ])

        result = get_file_structure(
            repo=mock_repo,
            scope="s",
            repository="r",
            file_path="file.ts",
        )

        assert len(result["structure"]) == 1
        assert result["structure"][0]["name"] == "UserId"
        assert result["structure"][0]["type"] == "type_alias"
        assert result["counts"]["type_alias"] == 1

    def test_includes_constant_elements(self, mock_repo):
        """Test get_file_structure includes constant elements."""
        self._setup_file_structure(mock_repo, [
            {
                "element_id": "s:r:main:file.ts:constant:MAX_SIZE:1",
                "hash_id": "abc4",
                "name": "MAX_SIZE",
                "element_type": "constant",
                "line_start": 1,
                "parent_id": "s:r:main:file.ts:file:file.ts:0",
            },
        ])

        result = get_file_structure(
            repo=mock_repo,
            scope="s",
            repository="r",
            file_path="file.ts",
        )

        assert len(result["structure"]) == 1
        assert result["structure"][0]["name"] == "MAX_SIZE"
        assert result["structure"][0]["type"] == "constant"
        assert result["counts"]["constant"] == 1

    def test_includes_trait_elements(self, mock_repo):
        """Test get_file_structure includes trait elements."""
        self._setup_file_structure(mock_repo, [
            {
                "element_id": "s:r:main:file.ts:trait:Display:1",
                "hash_id": "abc5",
                "name": "Display",
                "element_type": "trait",
                "line_start": 1,
                "parent_id": "s:r:main:file.ts:file:file.ts:0",
            },
        ])

        result = get_file_structure(
            repo=mock_repo,
            scope="s",
            repository="r",
            file_path="file.ts",
        )

        assert len(result["structure"]) == 1
        assert result["structure"][0]["name"] == "Display"
        assert result["structure"][0]["type"] == "trait"
        assert result["counts"]["trait"] == 1

    def test_counts_only_includes_present_types(self, mock_repo):
        """Test counts dict only includes types that have elements."""
        self._setup_file_structure(mock_repo, [
            {
                "element_id": "s:r:main:file.ts:function:foo:1",
                "hash_id": "abc6",
                "name": "foo",
                "element_type": "function",
                "line_start": 1,
                "parent_id": "s:r:main:file.ts:file:file.ts:0",
            },
            {
                "element_id": "s:r:main:file.ts:interface:IFoo:10",
                "hash_id": "abc7",
                "name": "IFoo",
                "element_type": "interface",
                "line_start": 10,
                "parent_id": "s:r:main:file.ts:file:file.ts:0",
            },
        ])

        result = get_file_structure(
            repo=mock_repo,
            scope="s",
            repository="r",
            file_path="file.ts",
        )

        # Only function and interface should appear in counts
        assert result["counts"] == {"function": 1, "interface": 1}
        assert "class" not in result["counts"]
        assert "method" not in result["counts"]

    def test_excludes_variable_elements(self, mock_repo):
        """Test get_file_structure still excludes variable elements."""
        self._setup_file_structure(mock_repo, [
            {
                "element_id": "s:r:main:file.ts:function:foo:1",
                "hash_id": "abc8",
                "name": "foo",
                "element_type": "function",
                "line_start": 1,
                "parent_id": "s:r:main:file.ts:file:file.ts:0",
            },
            {
                "element_id": "s:r:main:file.ts:variable:x:5",
                "hash_id": "abc9",
                "name": "x",
                "element_type": "variable",
                "line_start": 5,
                "parent_id": "s:r:main:file.ts:file:file.ts:0",
            },
        ])

        result = get_file_structure(
            repo=mock_repo,
            scope="s",
            repository="r",
            file_path="file.ts",
        )

        # Should only have function, not variable
        assert len(result["structure"]) == 1
        assert result["structure"][0]["name"] == "foo"


# =============================================================================
# GET CHILDREN TESTS
# =============================================================================


class TestGetChildren:
    """Tests for get_children function."""

    def test_get_children_returns_children(self, mock_repo):
        """Test get_children returns child elements."""
        mock_repo.get_document_by_id_or_hash.side_effect = None
        mock_repo.get_document_by_id_or_hash.return_value = {
            "element_id": "parent_id",
        }
        mock_client = MagicMock()
        mock_repo._get_client.return_value = mock_client
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
            repo=mock_repo,
            element_id="parent_id",
        )

        assert isinstance(result, list)
        assert len(result) == 2

    def test_get_children_resolves_hash_id(self, mock_repo):
        """Test get_children resolves hash_id to element_id before querying."""
        mock_repo.get_document_by_id_or_hash.side_effect = None
        mock_repo.get_document_by_id_or_hash.return_value = {
            "element_id": "scope:repo:main:file.py:class:Foo:10",
        }
        mock_client = MagicMock()
        mock_repo._get_client.return_value = mock_client
        mock_client.search.return_value = {"hits": {"hits": []}}

        get_children(
            repo=mock_repo,
            element_id="abc12345",  # hash_id
        )

        # Should resolve hash first
        mock_repo.get_document_by_id_or_hash.assert_called_once_with("abc12345")
        # Should query _find_children with canonical element_id, not hash
        search_body = mock_client.search.call_args[1]["body"]
        assert search_body["query"]["term"]["parent_id"] == "scope:repo:main:file.py:class:Foo:10"

    def test_get_children_not_found_raises(self, mock_repo):
        """Test get_children raises ValueError when element not found."""
        import pytest

        mock_repo.get_document_by_id_or_hash.return_value = None

        with pytest.raises(ValueError, match="Element not found"):
            get_children(repo=mock_repo, element_id="nonexistent")


# =============================================================================
# SEARCH FEATURES TESTS
# =============================================================================

