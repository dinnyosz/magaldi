"""Tests for language-specific call categorization.

Ensures built-in methods (dict.get, list.append, Array.push, Vec.iter, etc.)
are correctly categorized as 'builtin' so they never reach embedding resolution,
while typed-parameter receivers still get 'type_resolvable' to allow Strategy 5.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from magaldi_core.extractors.call_categorizer import (
    BUILTIN_METHODS,
    _get_builtin_methods,
    categorize_call,
    categorize_calls,
)
from magaldi_core.extractors.types import CallCategory


@dataclass
class FakeCall:
    """Minimal call object satisfying CallLike protocol."""

    name: str
    receiver: str | None
    resolved_id: str | None = None
    category: str = "unknown"


# =============================================================================
# BUILTIN_METHODS structure
# =============================================================================


class TestBuiltinMethodsStructure:
    """Verify the BUILTIN_METHODS dict is well-formed."""

    def test_python_methods_present(self) -> None:
        methods = BUILTIN_METHODS["python"]
        assert methods is not None
        assert "get" in methods
        assert "append" in methods
        assert "split" in methods
        assert "add" in methods

    def test_javascript_methods_present(self) -> None:
        methods = BUILTIN_METHODS["javascript"]
        assert methods is not None
        assert "push" in methods
        assert "map" in methods
        assert "filter" in methods
        assert "forEach" in methods

    def test_php_methods_present(self) -> None:
        methods = BUILTIN_METHODS["php"]
        assert methods is not None
        assert "get" in methods
        assert "count" in methods
        assert "toArray" in methods

    def test_rust_methods_present(self) -> None:
        methods = BUILTIN_METHODS["rust"]
        assert methods is not None
        assert "unwrap" in methods
        assert "iter" in methods
        assert "push" in methods
        assert "clone" in methods

    def test_typescript_inherits_from_javascript(self) -> None:
        assert BUILTIN_METHODS["typescript"] is None
        ts_methods = _get_builtin_methods("typescript")
        js_methods = _get_builtin_methods("javascript")
        assert ts_methods is js_methods

    def test_tsx_inherits_from_javascript(self) -> None:
        assert BUILTIN_METHODS["tsx"] is None
        tsx_methods = _get_builtin_methods("tsx")
        js_methods = _get_builtin_methods("javascript")
        assert tsx_methods is js_methods

    def test_bash_empty(self) -> None:
        assert BUILTIN_METHODS["bash"] == set()

    def test_unknown_language_returns_empty(self) -> None:
        methods = _get_builtin_methods("cobol")
        assert methods == set()


# =============================================================================
# Python categorization
# =============================================================================


class TestPythonBuiltinMethods:
    """Python dict/list/str/set methods → builtin."""

    @pytest.mark.parametrize(
        "receiver,name",
        [
            ("state", "get"),
            ("my_dict", "items"),
            ("data", "keys"),
            ("mapping", "values"),
            ("cache", "setdefault"),
            ("results", "append"),
            ("items_list", "extend"),
            ("seen_paths", "add"),
            ("content", "split"),
            ("text", "strip"),
            ("line", "encode"),
            ("logger", "debug"),
            ("logger", "info"),
            ("executor", "submit"),
            ("future", "result"),
        ],
    )
    def test_untyped_builtin_method(self, receiver: str, name: str) -> None:
        call = FakeCall(name=name, receiver=receiver)
        result = categorize_call(call, "python")
        assert result == CallCategory.BUILTIN

    def test_typed_param_overrides_builtin(self) -> None:
        """repo.get() where repo: Repository → type_resolvable, not builtin."""
        call = FakeCall(name="get", receiver="repo")
        result = categorize_call(call, "python", param_types={"repo": "Repository"})
        assert result == CallCategory.TYPE_RESOLVABLE

    def test_non_builtin_method_stays_untyped(self) -> None:
        """obj.process() is not a builtin method → untyped."""
        call = FakeCall(name="process", receiver="obj")
        result = categorize_call(call, "python")
        assert result == CallCategory.UNTYPED

    def test_bare_builtin_function(self) -> None:
        """len() with no receiver → builtin (existing behavior)."""
        call = FakeCall(name="len", receiver=None)
        result = categorize_call(call, "python")
        assert result == CallCategory.BUILTIN

    def test_self_method_not_affected(self) -> None:
        """self.get() is a self-method call → untyped, not builtin."""
        call = FakeCall(name="get", receiver="self")
        result = categorize_call(call, "python")
        assert result == CallCategory.UNTYPED

    def test_stdlib_receiver_takes_priority(self) -> None:
        """os.read() → stdlib, not builtin even though 'read' is in builtin methods."""
        call = FakeCall(name="read", receiver="os")
        result = categorize_call(call, "python")
        assert result == CallCategory.STDLIB


# =============================================================================
# JavaScript / TypeScript categorization
# =============================================================================


class TestJavaScriptBuiltinMethods:
    """JS/TS Array/String/Map/Promise methods → builtin."""

    @pytest.mark.parametrize(
        "receiver,name",
        [
            ("arr", "push"),
            ("items", "map"),
            ("results", "filter"),
            ("data", "forEach"),
            ("list", "includes"),
            ("text", "split"),
            ("str", "trim"),
            ("str", "toLowerCase"),
            ("promise", "then"),
            ("promise", "catch"),
            ("cache", "get"),
            ("cache", "set"),
            ("cache", "has"),
            ("console", "log"),
        ],
    )
    def test_untyped_builtin_method(self, receiver: str, name: str) -> None:
        call = FakeCall(name=name, receiver=receiver)
        result = categorize_call(call, "javascript")
        assert result == CallCategory.BUILTIN

    def test_typescript_inherits_js_builtins(self) -> None:
        call = FakeCall(name="push", receiver="arr")
        result = categorize_call(call, "typescript")
        assert result == CallCategory.BUILTIN

    def test_tsx_inherits_js_builtins(self) -> None:
        call = FakeCall(name="filter", receiver="items")
        result = categorize_call(call, "tsx")
        assert result == CallCategory.BUILTIN

    def test_this_method_not_affected(self) -> None:
        """this.get() is a this-method call → untyped, not builtin."""
        call = FakeCall(name="get", receiver="this")
        result = categorize_call(call, "javascript")
        assert result == CallCategory.UNTYPED

    def test_non_builtin_method_stays_untyped(self) -> None:
        call = FakeCall(name="fetchData", receiver="api")
        result = categorize_call(call, "javascript")
        assert result == CallCategory.UNTYPED


# =============================================================================
# PHP categorization
# =============================================================================


class TestPhpBuiltinMethods:
    """PHP collection/string/object methods → builtin."""

    @pytest.mark.parametrize(
        "receiver,name",
        [
            ("collection", "get"),
            ("items", "count"),
            ("data", "toArray"),
            ("results", "filter"),
            ("arr", "push"),
            ("str", "trim"),
            ("logger", "error"),
        ],
    )
    def test_untyped_builtin_method(self, receiver: str, name: str) -> None:
        call = FakeCall(name=name, receiver=receiver)
        result = categorize_call(call, "php")
        assert result == CallCategory.BUILTIN

    def test_this_method_not_affected(self) -> None:
        """$this->get() is a this-method call → untyped."""
        call = FakeCall(name="get", receiver="this")
        result = categorize_call(call, "php")
        assert result == CallCategory.UNTYPED

    def test_non_builtin_method_stays_untyped(self) -> None:
        call = FakeCall(name="handleRequest", receiver="controller")
        result = categorize_call(call, "php")
        assert result == CallCategory.UNTYPED


# =============================================================================
# Rust categorization
# =============================================================================


class TestRustBuiltinMethods:
    """Rust Option/Result/Vec/Iterator/HashMap methods → builtin."""

    @pytest.mark.parametrize(
        "receiver,name",
        [
            ("result", "unwrap"),
            ("option", "is_some"),
            ("vec", "push"),
            ("items", "iter"),
            ("map", "get"),
            ("data", "clone"),
            ("s", "to_string"),
            ("value", "into"),
            ("collection", "collect"),
            ("entry", "or_insert"),
        ],
    )
    def test_untyped_builtin_method(self, receiver: str, name: str) -> None:
        call = FakeCall(name=name, receiver=receiver)
        result = categorize_call(call, "rust")
        assert result == CallCategory.BUILTIN

    def test_self_method_not_affected(self) -> None:
        """self.push() is a self-method call → untyped."""
        call = FakeCall(name="push", receiver="self")
        result = categorize_call(call, "rust")
        assert result == CallCategory.UNTYPED

    def test_non_builtin_method_stays_untyped(self) -> None:
        call = FakeCall(name="process_request", receiver="handler")
        result = categorize_call(call, "rust")
        assert result == CallCategory.UNTYPED


# =============================================================================
# categorize_calls batch function
# =============================================================================


class TestCategorizeCalls:
    """Test the batch categorize_calls function with builtin methods."""

    def test_batch_categorization_mixed(self) -> None:
        """Mix of builtin methods, typed params, and regular calls."""
        calls = [
            FakeCall(name="get", receiver="state"),           # builtin
            FakeCall(name="get", receiver="repo"),            # type_resolvable
            FakeCall(name="process", receiver="handler"),     # untyped
            FakeCall(name="len", receiver=None),              # builtin (bare)
            FakeCall(name="split", receiver="line"),          # builtin
        ]
        params = [{"name": "repo", "type": "Repository"}]
        categorize_calls(calls, "python", params)

        assert calls[0].category == CallCategory.BUILTIN
        assert calls[1].category == CallCategory.TYPE_RESOLVABLE
        assert calls[2].category == CallCategory.UNTYPED
        assert calls[3].category == CallCategory.BUILTIN
        assert calls[4].category == CallCategory.BUILTIN

    def test_already_resolved_calls_not_recategorized(self) -> None:
        """Calls with resolved_id keep their resolved status."""
        call = FakeCall(
            name="get",
            receiver="state",
            resolved_id="some:element:id",
            category="resolved",
        )
        categorize_calls([call], "python")
        # category should not change because it's not "unknown"
        assert call.category == "resolved"

    def test_skips_non_unknown_categories(self) -> None:
        """Only recategorizes calls with category='unknown'."""
        call = FakeCall(name="get", receiver="state", category="untyped")
        categorize_calls([call], "python")
        # Already "untyped", not "unknown", so not recategorized
        assert call.category == "untyped"


# =============================================================================
# Priority / precedence tests
# =============================================================================


class TestCategorizationPriority:
    """Verify that priority order is correct across all checks."""

    def test_resolved_beats_everything(self) -> None:
        call = FakeCall(name="get", receiver="state", resolved_id="some:id")
        result = categorize_call(call, "python")
        assert result == CallCategory.RESOLVED

    def test_stdlib_beats_builtin_method(self) -> None:
        """time.time() → stdlib, even though 'time' might look like a method."""
        call = FakeCall(name="time", receiver="time")
        result = categorize_call(call, "python")
        assert result == CallCategory.STDLIB

    def test_external_beats_builtin_method(self) -> None:
        """requests.get() → external, not builtin."""
        call = FakeCall(name="get", receiver="requests")
        result = categorize_call(call, "python")
        assert result == CallCategory.EXTERNAL

    def test_type_resolvable_beats_builtin_method(self) -> None:
        """repo.get() where repo: Repository → type_resolvable."""
        call = FakeCall(name="get", receiver="repo")
        result = categorize_call(
            call, "python", param_types={"repo": "Repository"}
        )
        assert result == CallCategory.TYPE_RESOLVABLE

    def test_builtin_method_beats_untyped(self) -> None:
        """state.get() without type info → builtin (not untyped)."""
        call = FakeCall(name="get", receiver="state")
        result = categorize_call(call, "python")
        assert result == CallCategory.BUILTIN
