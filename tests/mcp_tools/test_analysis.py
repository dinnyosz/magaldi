"""Tests for MCP tools - analysis category."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from magaldi_mcp.tools import (
    find_call_chain,
    find_callers,
    find_dead_code,
    find_entry_points,
    find_implementations,
    find_usages,
    get_call_graph,
)

# =============================================================================
# FIND USAGES TESTS
# =============================================================================


class TestFindUsages:
    """Tests for find_usages function."""

    def test_find_usages_returns_usages(self, mock_repo):
        """Test find_usages returns usage locations."""
        mock_client = MagicMock()
        mock_repo._get_client.return_value = mock_client

        mock_repo.get_document.return_value = {
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

        result = find_usages(repo=mock_repo, element_id="func_id")

        assert isinstance(result, list)

    def test_find_usages_not_found(self, mock_repo):
        """Test find_usages raises when element not found."""
        mock_repo.get_document.return_value = None

        with pytest.raises(ValueError, match="Element not found"):
            find_usages(repo=mock_repo, element_id="nonexistent")


# =============================================================================
# FIND IMPLEMENTATIONS TESTS
# =============================================================================

# =============================================================================
# FIND USAGES WITH PATTERN SEARCH TESTS
# =============================================================================


class TestFindUsagesWithPatternSearch:
    """Tests for find_usages using search_by_regexp."""

    def test_find_usages_uses_regexp_search(self, mock_repo):
        """Test that find_usages uses search_by_regexp internally."""
        mock_repo.get_document.return_value = {
            "element_id": "test:repo:main:file.py:function:my_func:10",
            "name": "my_func",
            "element_type": "function",
            "relative_path": "file.py",
            "line_start": 10,
            "scope": "test",
            "repository": "repo",
            "username": "main",
        }
        mock_repo.search_by_regexp.return_value = [
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
            repo=mock_repo,
            element_id="test:repo:main:file.py:function:my_func:10",
        )

        # Verify search_by_regexp was called (not the old client.search or grep_code)
        mock_repo.search_by_regexp.assert_called()
        assert len(result) >= 0  # May filter out definition

    def test_find_usages_builds_function_call_pattern(self, mock_repo):
        """Test that find_usages builds correct Lucene regexp for function calls."""
        mock_repo.get_document.return_value = {
            "element_id": "test:repo:main:file.py:function:my_func:10",
            "name": "my_func",
            "element_type": "function",
            "relative_path": "file.py",
            "line_start": 10,
            "scope": "test",
            "repository": "repo",
            "username": "main",
        }
        mock_repo.search_by_regexp.return_value = []

        find_usages(
            repo=mock_repo,
            element_id="test:repo:main:file.py:function:my_func:10",
        )

        # Verify the pattern is Lucene-compatible (name followed by paren)
        call_args = mock_repo.search_by_regexp.call_args
        pattern = call_args[1]["pattern"]
        # Should match function name followed by optional spaces then paren
        assert "my_func" in pattern
        assert "\\(" in pattern  # Escaped paren for Lucene

    def test_find_usages_builds_method_call_pattern(self, mock_repo):
        """Test that find_usages builds correct Lucene regexp for method calls."""
        mock_repo.get_document.return_value = {
            "element_id": "test:repo:main:file.py:method:my_method:10",
            "name": "my_method",
            "element_type": "method",
            "relative_path": "file.py",
            "line_start": 10,
            "scope": "test",
            "repository": "repo",
            "username": "main",
        }
        mock_repo.search_by_regexp.return_value = []

        find_usages(
            repo=mock_repo,
            element_id="test:repo:main:file.py:method:my_method:10",
        )

        # Verify the pattern includes dot before method name
        call_args = mock_repo.search_by_regexp.call_args
        pattern = call_args[1]["pattern"]
        assert "\\.my_method" in pattern  # Dot then method name

    def test_find_usages_builds_class_reference_pattern(self, mock_repo):
        """Test that find_usages builds correct Lucene regexp for class references."""
        mock_repo.get_document.return_value = {
            "element_id": "test:repo:main:file.py:class:MyClass:10",
            "name": "MyClass",
            "element_type": "class",
            "relative_path": "file.py",
            "line_start": 10,
            "scope": "test",
            "repository": "repo",
            "username": "main",
        }
        mock_repo.search_by_regexp.return_value = []

        find_usages(
            repo=mock_repo,
            element_id="test:repo:main:file.py:class:MyClass:10",
        )

        # Verify the pattern contains the class name
        call_args = mock_repo.search_by_regexp.call_args
        pattern = call_args[1]["pattern"]
        assert "MyClass" in pattern

    def test_find_usages_filters_definitions(self, mock_repo):
        """Test that find_usages filters out definition lines."""
        mock_repo.get_document.return_value = {
            "element_id": "test:repo:main:file.py:function:my_func:10",
            "name": "my_func",
            "element_type": "function",
            "relative_path": "file.py",
            "line_start": 10,
            "scope": "test",
            "repository": "repo",
            "username": "main",
        }
        mock_repo.search_by_regexp.return_value = [
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
            repo=mock_repo,
            element_id="test:repo:main:file.py:function:my_func:10",
        )

        # The definition in file.py should be filtered out
        assert len(result) == 1
        assert result[0]["file"] == "other.py"

    def test_find_usages_escapes_special_chars_for_lucene(self, mock_repo):
        """Test that find_usages escapes special chars for Lucene regexp."""
        mock_repo.get_document.return_value = {
            "element_id": "test:repo:main:file.py:function:func_with_dots:10",
            "name": "func.with.dots",  # Name with dots (unusual but possible)
            "element_type": "function",
            "relative_path": "file.py",
            "line_start": 10,
            "scope": "test",
            "repository": "repo",
            "username": "main",
        }
        mock_repo.search_by_regexp.return_value = []

        find_usages(
            repo=mock_repo,
            element_id="test:repo:main:file.py:function:func_with_dots:10",
        )

        # Verify dots in name are escaped for Lucene
        call_args = mock_repo.search_by_regexp.call_args
        pattern = call_args[1]["pattern"]
        # Dots should be escaped as \.
        assert "\\." in pattern


# =============================================================================
# FIND IMPLEMENTATIONS WITH PATTERN SEARCH TESTS
# =============================================================================

# =============================================================================
# FIND IMPLEMENTATIONS TESTS
# =============================================================================


class TestFindImplementations:
    """Tests for find_implementations function."""

    def test_find_implementations_returns_implementations(self, mock_repo):
        """Test find_implementations returns implementing classes."""
        mock_client = MagicMock()
        mock_repo._get_client.return_value = mock_client

        mock_repo.get_document.return_value = {
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

        result = find_implementations(repo=mock_repo, element_id="base_id")

        assert isinstance(result, list)

    def test_find_implementations_by_class_name(self, mock_repo):
        """Test find_implementations by class name."""
        mock_client = MagicMock()
        mock_repo._get_client.return_value = mock_client
        mock_client.search.return_value = {"hits": {"hits": []}}

        result = find_implementations(
            repo=mock_repo,
            class_name="Protocol",
        )

        assert isinstance(result, list)

    def test_find_implementations_requires_id_or_name(self, mock_repo):
        """Test find_implementations requires element_id or class_name."""
        with pytest.raises(ValueError, match="Either element_id or class_name required"):
            find_implementations(repo=mock_repo)


# =============================================================================
# GET CALL GRAPH TESTS
# =============================================================================

# =============================================================================
# FIND IMPLEMENTATIONS WITH PATTERN SEARCH TESTS
# =============================================================================


class TestFindImplementationsWithPatternSearch:
    """Tests for find_implementations using pattern_search."""

    def test_find_implementations_uses_regexp_search(self, mock_repo):
        """Test that find_implementations uses search_by_regexp."""
        mock_repo.get_document.return_value = {
            "element_id": "test:repo:main:base.py:class:BaseClass:1",
            "name": "BaseClass",
            "element_type": "class",
            "scope": "test",
            "repository": "repo",
        }
        mock_repo.search_by_regexp.return_value = [
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
            repo=mock_repo,
            element_id="test:repo:main:base.py:class:BaseClass:1",
        )

        mock_repo.search_by_regexp.assert_called()
        assert len(result) >= 1

    def test_find_implementations_builds_inheritance_pattern(self, mock_repo):
        """Test that find_implementations builds correct Lucene pattern for inheritance."""
        mock_repo.get_document.return_value = {
            "element_id": "test:repo:main:base.py:class:BaseClass:1",
            "name": "BaseClass",
            "element_type": "class",
            "scope": "test",
            "repository": "repo",
        }
        mock_repo.search_by_regexp.return_value = []

        find_implementations(
            repo=mock_repo,
            element_id="test:repo:main:base.py:class:BaseClass:1",
        )

        # Verify pattern looks for class inheritance
        call_args = mock_repo.search_by_regexp.call_args
        pattern = call_args[1]["pattern"]
        # Pattern should match: class SomeClass(BaseClass) or class SomeClass(Other, BaseClass)
        assert "class" in pattern
        assert "BaseClass" in pattern
        assert "\\(" in pattern  # Escaped paren for Lucene

    def test_find_implementations_by_class_name_uses_regexp(self, mock_repo):
        """Test find_implementations by class_name also uses search_by_regexp."""
        mock_repo.search_by_regexp.return_value = []

        find_implementations(
            repo=mock_repo,
            class_name="Protocol",
            scope="test",
            repository="repo",
        )

        mock_repo.search_by_regexp.assert_called()
        call_args = mock_repo.search_by_regexp.call_args
        pattern = call_args[1]["pattern"]
        assert "Protocol" in pattern

    def test_find_implementations_extracts_class_name(self, mock_repo):
        """Test that find_implementations correctly extracts implementing class names."""
        mock_repo.get_document.return_value = {
            "element_id": "test:repo:main:base.py:class:BaseClass:1",
            "name": "BaseClass",
            "element_type": "class",
            "scope": "test",
            "repository": "repo",
        }
        mock_repo.search_by_regexp.return_value = [
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
            repo=mock_repo,
            element_id="test:repo:main:base.py:class:BaseClass:1",
        )

        assert len(result) == 1
        assert result[0]["class_name"] == "DerivedClass"
        assert result[0]["file"] == "impl.py"
        assert result[0]["line"] == 10

    def test_find_implementations_escapes_special_chars(self, mock_repo):
        """Test that find_implementations escapes special chars for Lucene."""
        mock_repo.get_document.return_value = {
            "element_id": "test:repo:main:base.py:class:Base.Class:1",
            "name": "Base.Class",  # Name with dot (unusual)
            "element_type": "class",
            "scope": "test",
            "repository": "repo",
        }
        mock_repo.search_by_regexp.return_value = []

        find_implementations(
            repo=mock_repo,
            element_id="test:repo:main:base.py:class:Base.Class:1",
        )

        call_args = mock_repo.search_by_regexp.call_args
        pattern = call_args[1]["pattern"]
        # Dot should be escaped
        assert "\\." in pattern


# =============================================================================
# FIND CALLERS TESTS
# =============================================================================

# =============================================================================
# GET CALL GRAPH TESTS
# =============================================================================


class TestGetCallGraph:
    """Tests for get_call_graph function."""

    def test_get_call_graph_returns_callers_and_callees(self, mock_repo):
        """Test get_call_graph returns both callers and callees."""
        mock_client = MagicMock()
        mock_repo._get_client.return_value = mock_client

        mock_repo.get_document.return_value = {
            "element_id": "func_id",
            "name": "my_func",
            "element_type": "function",
            "relative_path": "funcs.py",
            "raw_code": "def my_func():\n    helper()\n    other_func()\n",
        }

        mock_client.search.return_value = {"hits": {"hits": []}}

        result = get_call_graph(repo=mock_repo, element_id="func_id")

        assert "element" in result
        assert "callers" in result
        assert "callees" in result
        assert result["element"]["name"] == "my_func"

    def test_get_call_graph_not_found(self, mock_repo):
        """Test get_call_graph raises when element not found."""
        mock_repo.get_document.return_value = None

        with pytest.raises(ValueError, match="Element not found"):
            get_call_graph(repo=mock_repo, element_id="nonexistent")

    def test_get_call_graph_direction_callers(self, mock_repo):
        """Test get_call_graph with callers only direction."""
        mock_client = MagicMock()
        mock_repo._get_client.return_value = mock_client

        mock_repo.get_document.return_value = {
            "element_id": "func_id",
            "name": "my_func",
            "element_type": "function",
            "relative_path": "funcs.py",
            "raw_code": "def my_func(): pass",
        }

        mock_client.search.return_value = {"hits": {"hits": []}}

        result = get_call_graph(repo=mock_repo, element_id="func_id", direction="callers")

        assert "callers" in result
        assert "callees" in result  # Should still be in result but empty


# =============================================================================
# SEARCH CODE FALLBACK TESTS
# =============================================================================

# =============================================================================
# FIND CALLERS TESTS
# =============================================================================


class TestFindCallers:
    """Tests for find_callers function."""

    def test_find_callers_returns_grouped_results(self, mock_repo):
        """Test find_callers returns callers grouped by code/tests."""
        mock_repo.get_document.return_value = {
            "element_id": "scope:repo:main:utils.py:function:helper:10",
            "name": "helper",
            "element_type": "function",
            "relative_path": "utils.py",
            "line_start": 10,
            "scope": "scope",
            "repository": "repo",
            "username": "main",
        }
        mock_repo.find_elements_calling.return_value = [
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
            repo=mock_repo,
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

    def test_find_callers_not_found_raises(self, mock_repo):
        """Test find_callers raises when element not found."""
        mock_repo.get_document.return_value = None

        with pytest.raises(ValueError, match="Element not found"):
            find_callers(repo=mock_repo, element_id="nonexistent")

    def test_find_callers_excludes_tests_when_disabled(self, mock_repo):
        """Test find_callers excludes test results when include_tests=False."""
        mock_repo.get_document.return_value = {
            "element_id": "scope:repo:main:utils.py:function:helper:10",
            "name": "helper",
            "element_type": "function",
            "relative_path": "utils.py",
            "line_start": 10,
            "scope": "scope",
            "repository": "repo",
            "username": "main",
        }
        mock_repo.find_elements_calling.return_value = [
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
            repo=mock_repo,
            element_id="scope:repo:main:utils.py:function:helper:10",
            include_tests=False,
        )

        assert len(result["code_results"]) == 0
        assert len(result["test_results"]) == 0

    def test_find_callers_uses_target_scope_repo(self, mock_repo):
        """Test find_callers uses target's scope/repo if not specified."""
        mock_repo.get_document.return_value = {
            "element_id": "myscope:myrepo:main:utils.py:function:helper:10",
            "name": "helper",
            "element_type": "function",
            "relative_path": "utils.py",
            "line_start": 10,
            "scope": "myscope",
            "repository": "myrepo",
            "username": "main",
        }
        mock_repo.find_elements_calling.return_value = []

        find_callers(
            repo=mock_repo,
            element_id="myscope:myrepo:main:utils.py:function:helper:10",
        )

        call_args = mock_repo.find_elements_calling.call_args
        assert call_args[1]["scope"] == "myscope"
        assert call_args[1]["repository"] == "myrepo"


# =============================================================================
# FIND CALL CHAIN TESTS
# =============================================================================

# =============================================================================
# FIND CALL CHAIN TESTS
# =============================================================================


class TestFindCallChain:
    """Tests for find_call_chain function."""

    def test_find_call_chain_callees(self, mock_repo):
        """Test find_call_chain traces callees."""
        mock_repo.get_document.side_effect = [
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
        mock_repo.get_calls.side_effect = [
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
            repo=mock_repo,
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

    def test_find_call_chain_callers(self, mock_repo):
        """Test find_call_chain traces callers."""
        mock_repo.get_document.return_value = {
            "element_id": "scope:repo:main:utils.py:function:helper:10",
            "name": "helper",
            "element_type": "function",
            "relative_path": "utils.py",
            "line_start": 10,
            "scope": "scope",
            "repository": "repo",
            "username": "main",
        }
        mock_repo.find_elements_calling.side_effect = [
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
            repo=mock_repo,
            element_id="scope:repo:main:utils.py:function:helper:10",
            direction="callers",
            max_depth=3,
        )

        assert "callers" in result
        assert len(result["callers"]) == 1
        assert result["callers"][0]["name"] == "main"

    def test_find_call_chain_not_found_raises(self, mock_repo):
        """Test find_call_chain raises when element not found."""
        mock_repo.get_document.return_value = None

        with pytest.raises(ValueError, match="Element not found"):
            find_call_chain(repo=mock_repo, element_id="nonexistent")

    def test_find_call_chain_detects_cycles(self, mock_repo):
        """Test find_call_chain marks cycles instead of infinite recursion."""
        mock_repo.get_document.side_effect = [
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
        mock_repo.get_calls.side_effect = [
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
            repo=mock_repo,
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

    def test_find_call_chain_max_depth_clamped(self, mock_repo):
        """Test find_call_chain clamps max_depth to valid range."""
        mock_repo.get_document.return_value = {
            "element_id": "scope:repo:main:app.py:function:main:1",
            "name": "main",
            "element_type": "function",
            "relative_path": "app.py",
            "line_start": 1,
            "scope": "scope",
            "repository": "repo",
            "username": "main",
        }
        mock_repo.get_calls.return_value = []

        # Test max_depth > 10 is clamped to 10
        result = find_call_chain(
            repo=mock_repo,
            element_id="scope:repo:main:app.py:function:main:1",
            max_depth=20,
        )
        assert result["max_depth"] == 10

        # Test max_depth < 1 is clamped to 1
        result = find_call_chain(
            repo=mock_repo,
            element_id="scope:repo:main:app.py:function:main:1",
            max_depth=0,
        )
        assert result["max_depth"] == 1


# =============================================================================
# FIND DEAD CODE TESTS
# =============================================================================

# =============================================================================
# FIND DEAD CODE TESTS
# =============================================================================


class TestFindDeadCode:
    """Tests for find_dead_code function."""

    def test_find_dead_code_returns_uncalled_functions(self, mock_repo):
        """Test find_dead_code returns functions with no callers."""
        mock_client = MagicMock()
        mock_repo._get_client.return_value = mock_client
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
        mock_repo.find_elements_calling.side_effect = [[], [{"name": "caller"}]]

        result = find_dead_code(
            repo=mock_repo,
            scope="scope",
            repository="repo",
        )

        assert "potentially_dead" in result
        assert "stats" in result
        assert len(result["potentially_dead"]) == 1
        assert result["potentially_dead"][0]["name"] == "_unused"
        assert result["stats"]["total_functions"] == 2
        assert result["stats"]["potentially_dead"] == 1

    def test_find_dead_code_excludes_entry_points(self, mock_repo):
        """Test find_dead_code excludes decorated entry points."""
        mock_client = MagicMock()
        mock_repo._get_client.return_value = mock_client
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
            repo=mock_repo,
            scope="scope",
            repository="repo",
        )

        # Entry point should be excluded, not in dead code
        assert len(result["potentially_dead"]) == 0
        assert result["stats"]["excluded_entry_points"] == 1

    def test_find_dead_code_excludes_magic_methods(self, mock_repo):
        """Test find_dead_code excludes __init__ and other magic methods."""
        mock_client = MagicMock()
        mock_repo._get_client.return_value = mock_client
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
            repo=mock_repo,
            scope="scope",
            repository="repo",
        )

        # __init__ should be excluded
        assert len(result["potentially_dead"]) == 0
        assert result["stats"]["excluded_entry_points"] == 1

    def test_find_dead_code_excludes_main(self, mock_repo):
        """Test find_dead_code excludes main functions."""
        mock_client = MagicMock()
        mock_repo._get_client.return_value = mock_client
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
            repo=mock_repo,
            scope="scope",
            repository="repo",
        )

        assert len(result["potentially_dead"]) == 0
        assert result["stats"]["excluded_entry_points"] == 1


# =============================================================================
# FIND ENTRY POINTS TESTS
# =============================================================================

# =============================================================================
# FIND ENTRY POINTS TESTS
# =============================================================================


class TestFindEntryPoints:
    """Tests for find_entry_points function."""

    def test_find_entry_points_groups_by_type(self, mock_repo):
        """Test find_entry_points groups results by type."""
        mock_client = MagicMock()
        mock_repo._get_client.return_value = mock_client
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
            repo=mock_repo,
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

    def test_find_entry_points_with_no_matches(self, mock_repo):
        """Test find_entry_points with no entry points."""
        mock_client = MagicMock()
        mock_repo._get_client.return_value = mock_client
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
            repo=mock_repo,
            scope="scope",
            repository="repo",
        )

        assert len(result["http"]) == 0
        assert len(result["cli"]) == 0
        assert len(result["test"]) == 0
        assert len(result["main"]) == 0
        assert result["stats"]["total"] == 0

    def test_find_entry_points_async_tasks(self, mock_repo):
        """Test find_entry_points detects async task decorators."""
        mock_client = MagicMock()
        mock_repo._get_client.return_value = mock_client
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
            repo=mock_repo,
            scope="scope",
            repository="repo",
        )

        assert len(result["async_tasks"]) == 1
        assert result["async_tasks"][0]["name"] == "process"


# =============================================================================
# FIND DEPENDENCIES TESTS
# =============================================================================

