"""Tests for AST-based variable binding extraction.

Tests the scope_bindings module that extracts variable->type bindings
from tree-sitter AST for scope-aware call resolution (Strategy 5.7).
"""

from __future__ import annotations

import pytest

from magaldi_core.scope_bindings import (
    SOURCE_ASSIGNMENT_CALL,
    SOURCE_ASSIGNMENT_METHOD_CALL,
    SOURCE_CONSTRUCTOR,
    SOURCE_EXCEPT_AS,
    SOURCE_FOR_IN,
    SOURCE_WITH_AS,
    BindingInfo,
    extract_variable_bindings,
)


# =============================================================================
# ASSIGNMENT BINDINGS
# =============================================================================


class TestAssignmentBindings:
    """Tests for assignment-based variable bindings."""

    def test_bare_call_assignment(self):
        """var = func() creates an assignment_call binding."""
        code = "def f():\n    result = get_user()\n"
        bindings = extract_variable_bindings(code, "python")
        assert len(bindings) == 1
        b = bindings[0]
        assert b.variable == "result"
        assert b.source == SOURCE_ASSIGNMENT_CALL
        assert b.call_name == "get_user"
        assert b.call_receiver is None
        assert b.type_name is None

    def test_method_call_assignment(self):
        """var = obj.method() creates an assignment_method_call binding."""
        code = "def f():\n    conn = db.connect()\n"
        bindings = extract_variable_bindings(code, "python")
        assert len(bindings) == 1
        b = bindings[0]
        assert b.variable == "conn"
        assert b.source == SOURCE_ASSIGNMENT_METHOD_CALL
        assert b.call_name == "connect"
        assert b.call_receiver == "db"

    def test_constructor_assignment(self):
        """var = ClassName() creates a constructor binding."""
        code = "def f():\n    repo = Repository()\n"
        bindings = extract_variable_bindings(code, "python")
        assert len(bindings) == 1
        b = bindings[0]
        assert b.variable == "repo"
        assert b.source == SOURCE_CONSTRUCTOR
        assert b.type_name == "Repository"
        assert b.call_name == "Repository"

    def test_await_bare_call(self):
        """var = await func() extracts through await."""
        code = "async def f():\n    result = await get_user()\n"
        bindings = extract_variable_bindings(code, "python")
        assert len(bindings) == 1
        b = bindings[0]
        assert b.variable == "result"
        assert b.source == SOURCE_ASSIGNMENT_CALL
        assert b.call_name == "get_user"

    def test_await_method_call(self):
        """var = await obj.method() extracts through await."""
        code = "async def f():\n    resp = await client.fetch()\n"
        bindings = extract_variable_bindings(code, "python")
        assert len(bindings) == 1
        b = bindings[0]
        assert b.variable == "resp"
        assert b.source == SOURCE_ASSIGNMENT_METHOD_CALL
        assert b.call_name == "fetch"
        assert b.call_receiver == "client"

    def test_module_constructor(self):
        """var = module.ClassName() is an assignment_method_call since
        tree-sitter sees it as attribute access, not a constructor pattern."""
        code = "def f():\n    repo = db.Repository()\n"
        bindings = extract_variable_bindings(code, "python")
        assert len(bindings) == 1
        # Tree-sitter sees this as attribute call: db.Repository()
        # Not a constructor pattern since it's attribute access
        b = bindings[0]
        assert b.variable == "repo"
        assert b.source == SOURCE_ASSIGNMENT_METHOD_CALL
        assert b.call_name == "Repository"
        assert b.call_receiver == "db"

    def test_non_call_assignment_skipped(self):
        """var = other_var does NOT create a binding (not a call)."""
        code = "def f():\n    x = some_var\n    y = 42\n    z = [1, 2, 3]\n"
        bindings = extract_variable_bindings(code, "python")
        assert len(bindings) == 0

    def test_tuple_unpack_skipped(self):
        """a, b = func() does NOT create a binding (can't determine types)."""
        code = "def f():\n    a, b = func()\n"
        bindings = extract_variable_bindings(code, "python")
        assert len(bindings) == 0

    def test_nested_attribute_receiver(self):
        """var = a.b.method() captures the nested receiver."""
        code = "def f():\n    result = self.db.query()\n"
        bindings = extract_variable_bindings(code, "python")
        assert len(bindings) == 1
        b = bindings[0]
        assert b.variable == "result"
        assert b.source == SOURCE_ASSIGNMENT_METHOD_CALL
        assert b.call_name == "query"
        assert b.call_receiver == "self.db"


# =============================================================================
# WITH-AS BINDINGS
# =============================================================================


class TestWithAsBindings:
    """Tests for context manager variable bindings."""

    def test_with_bare_call(self):
        """with open(f) as handle: creates a with_as binding."""
        code = 'def f():\n    with open("file.txt") as handle:\n        pass\n'
        bindings = extract_variable_bindings(code, "python")
        assert len(bindings) == 1
        b = bindings[0]
        assert b.variable == "handle"
        assert b.source == SOURCE_WITH_AS
        assert b.call_name == "open"
        assert b.call_receiver is None

    def test_with_method_call(self):
        """with db.connect() as conn: creates a with_as binding."""
        code = "def f():\n    with db.connect() as conn:\n        pass\n"
        bindings = extract_variable_bindings(code, "python")
        assert len(bindings) == 1
        b = bindings[0]
        assert b.variable == "conn"
        assert b.source == SOURCE_WITH_AS
        assert b.call_name == "connect"
        assert b.call_receiver == "db"

    def test_with_constructor(self):
        """with ClassName() as obj: creates a with_as binding with type."""
        code = "def f():\n    with Session() as session:\n        pass\n"
        bindings = extract_variable_bindings(code, "python")
        assert len(bindings) == 1
        b = bindings[0]
        assert b.variable == "session"
        assert b.source == SOURCE_WITH_AS
        assert b.call_name == "Session"
        assert b.type_name == "Session"

    def test_multi_with(self):
        """with a() as x, b() as y: creates two bindings."""
        code = "def f():\n    with open('a') as a, open('b') as b:\n        pass\n"
        bindings = extract_variable_bindings(code, "python")
        assert len(bindings) == 2
        assert bindings[0].variable == "a"
        assert bindings[1].variable == "b"


# =============================================================================
# FOR-IN BINDINGS
# =============================================================================


class TestForInBindings:
    """Tests for loop variable bindings."""

    def test_for_in_bare_call(self):
        """for item in get_items(): creates a for_in binding."""
        code = "def f():\n    for item in get_items():\n        pass\n"
        bindings = extract_variable_bindings(code, "python")
        assert len(bindings) == 1
        b = bindings[0]
        assert b.variable == "item"
        assert b.source == SOURCE_FOR_IN
        assert b.call_name == "get_items"

    def test_for_in_method_call(self):
        """for row in repo.fetch_all(): creates a for_in binding."""
        code = "def f():\n    for row in repo.fetch_all():\n        pass\n"
        bindings = extract_variable_bindings(code, "python")
        assert len(bindings) == 1
        b = bindings[0]
        assert b.variable == "row"
        assert b.source == SOURCE_FOR_IN
        assert b.call_name == "fetch_all"
        assert b.call_receiver == "repo"

    def test_for_in_identifier(self):
        """for item in items: creates a for_in binding with collection reference."""
        code = "def f():\n    for item in items:\n        pass\n"
        bindings = extract_variable_bindings(code, "python")
        assert len(bindings) == 1
        b = bindings[0]
        assert b.variable == "item"
        assert b.source == SOURCE_FOR_IN
        assert b.call_receiver == "items"
        assert b.call_name is None

    def test_for_tuple_unpack_skipped(self):
        """for a, b in items: is skipped (can't determine individual types)."""
        code = "def f():\n    for a, b in items:\n        pass\n"
        bindings = extract_variable_bindings(code, "python")
        assert len(bindings) == 0


# =============================================================================
# EXCEPT-AS BINDINGS
# =============================================================================


class TestExceptAsBindings:
    """Tests for exception handler variable bindings."""

    def test_except_as(self):
        """except ValueError as e: creates an except_as binding."""
        code = "def f():\n    try:\n        pass\n    except ValueError as e:\n        pass\n"
        bindings = extract_variable_bindings(code, "python")
        assert len(bindings) == 1
        b = bindings[0]
        assert b.variable == "e"
        assert b.source == SOURCE_EXCEPT_AS
        assert b.type_name == "ValueError"

    def test_except_module_exception(self):
        """except module.ExcType as e: captures the full exception type."""
        code = "def f():\n    try:\n        pass\n    except http.HTTPError as e:\n        pass\n"
        bindings = extract_variable_bindings(code, "python")
        assert len(bindings) == 1
        b = bindings[0]
        assert b.type_name == "http.HTTPError"

    def test_except_without_as_no_binding(self):
        """except ValueError: (without as) does not create a binding."""
        code = "def f():\n    try:\n        pass\n    except ValueError:\n        pass\n"
        bindings = extract_variable_bindings(code, "python")
        assert len(bindings) == 0

    def test_bare_except_no_binding(self):
        """bare except: does not create a binding."""
        code = "def f():\n    try:\n        pass\n    except:\n        pass\n"
        bindings = extract_variable_bindings(code, "python")
        assert len(bindings) == 0


# =============================================================================
# EDGE CASES & COMBINATIONS
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and combined patterns."""

    def test_nested_function_bindings_excluded(self):
        """Bindings inside nested functions are NOT extracted."""
        code = "def outer():\n    x = Foo()\n    def inner():\n        y = Bar()\n    z = Baz()\n"
        bindings = extract_variable_bindings(code, "python")
        assert len(bindings) == 2
        assert {b.variable for b in bindings} == {"x", "z"}

    def test_nested_class_bindings_excluded(self):
        """Bindings inside nested classes are NOT extracted."""
        code = "def outer():\n    x = Foo()\n    class Inner:\n        y = Bar()\n    z = Baz()\n"
        bindings = extract_variable_bindings(code, "python")
        assert len(bindings) == 2
        assert {b.variable for b in bindings} == {"x", "z"}

    def test_multiple_bindings_all_patterns(self):
        """All binding patterns in one function are extracted."""
        code = (
            "def process():\n"
            "    conn = db.connect()\n"
            "    result = get_user()\n"
            "    repo = Repository()\n"
            "    with open('f') as handle:\n"
            "        pass\n"
            "    for item in get_items():\n"
            "        pass\n"
            "    try:\n"
            "        pass\n"
            "    except ValueError as e:\n"
            "        pass\n"
        )
        bindings = extract_variable_bindings(code, "python")
        variables = {b.variable for b in bindings}
        assert variables == {"conn", "result", "repo", "handle", "item", "e"}

    def test_decorated_function(self):
        """Bindings from decorated function body are extracted."""
        code = "@decorator\ndef f():\n    conn = db.connect()\n"
        bindings = extract_variable_bindings(code, "python")
        assert len(bindings) == 1
        assert bindings[0].variable == "conn"

    def test_class_raw_code_returns_empty(self):
        """Class raw_code (not a function) returns no bindings."""
        code = "class Foo:\n    def __init__(self):\n        self.x = 1\n"
        bindings = extract_variable_bindings(code, "python")
        assert len(bindings) == 0

    def test_empty_code_returns_empty(self):
        """Empty raw_code returns no bindings."""
        assert extract_variable_bindings("", "python") == []

    def test_non_python_returns_empty(self):
        """Non-Python language returns no bindings."""
        code = "def f():\n    x = Foo()\n"
        assert extract_variable_bindings(code, "javascript") == []

    def test_reassignment_both_extracted(self):
        """Multiple assignments to same variable are all extracted (last wins at resolution)."""
        code = "def f():\n    x = Foo()\n    x = Bar()\n"
        bindings = extract_variable_bindings(code, "python")
        assert len(bindings) == 2
        assert bindings[0].type_name == "Foo"
        assert bindings[1].type_name == "Bar"

    def test_line_numbers_correct(self):
        """Binding line numbers are 1-indexed and correct."""
        code = "def f():\n    x = Foo()\n    y = Bar()\n"
        bindings = extract_variable_bindings(code, "python")
        assert bindings[0].line == 2
        assert bindings[1].line == 3


# =============================================================================
# UNWRAP ITERABLE TYPE
# =============================================================================


class TestUnwrapIterableType:
    """Tests for _unwrap_iterable_type helper."""

    def test_list_type(self):
        from magaldi_core.call_resolution import _unwrap_iterable_type

        assert _unwrap_iterable_type("list[Item]") == "Item"
        assert _unwrap_iterable_type("List[Item]") == "Item"

    def test_set_type(self):
        from magaldi_core.call_resolution import _unwrap_iterable_type

        assert _unwrap_iterable_type("set[Item]") == "Item"
        assert _unwrap_iterable_type("Set[Item]") == "Item"

    def test_iterable_type(self):
        from magaldi_core.call_resolution import _unwrap_iterable_type

        assert _unwrap_iterable_type("Iterable[User]") == "User"
        assert _unwrap_iterable_type("Iterator[User]") == "User"

    def test_sequence_type(self):
        from magaldi_core.call_resolution import _unwrap_iterable_type

        assert _unwrap_iterable_type("Sequence[Element]") == "Element"

    def test_non_generic_returns_none(self):
        from magaldi_core.call_resolution import _unwrap_iterable_type

        assert _unwrap_iterable_type("User") is None
        assert _unwrap_iterable_type("str") is None

    def test_empty_returns_none(self):
        from magaldi_core.call_resolution import _unwrap_iterable_type

        assert _unwrap_iterable_type("") is None
        assert _unwrap_iterable_type(None) is None  # type: ignore[arg-type]

    def test_optional_in_generic(self):
        from magaldi_core.call_resolution import _unwrap_iterable_type

        assert _unwrap_iterable_type("list[Item | None]") == "Item"

    def test_nested_generic_returns_none(self):
        from magaldi_core.call_resolution import _unwrap_iterable_type

        assert _unwrap_iterable_type("list[dict[str, int]]") is None
