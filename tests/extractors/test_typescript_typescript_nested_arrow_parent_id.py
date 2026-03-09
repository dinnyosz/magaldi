"""Tests for nested arrow function parent_id assignment (Issue 2).

Arrow functions defined inside other functions (via `const x = () => {}`) should
get parent_id pointing to the enclosing function, not the file element.
"""

import pytest
from pathlib import Path

from magaldi_core.code_parser import JavaScriptParser
from magaldi_core.change_detection import FileInfo


class TestNestedArrowParentId:
    """Tests for nested arrow function parent_id in TypeScript."""

    def test_nested_arrow_functions_parent_id(self):
        """Nested arrow functions should have parent_id of enclosing function."""
        code = (
            "export default function timedOut(request: Request, delays: Delays): () => void {\n"
            "  const addTimeout = (delay: number, callback: () => void): () => void => {\n"
            "    const cancel = (): void => {\n"
            "      clearTimeout(timer);\n"
            "    };\n"
            "\n"
            "    const timer = setTimeout(callback, delay);\n"
            "    return cancel;\n"
            "  };\n"
            "\n"
            "  return () => {};\n"
            "}"
        )

        parser = JavaScriptParser("typescript")
        file_info = FileInfo(
            relative_path="timed-out.ts",
            absolute_path=Path("/fake/timed-out.ts"),
            language="typescript",
        )

        elements = parser.parse(code, file_info, "scope", "repo", "main")

        # Get elements by name
        file_elem = next(e for e in elements if e.element_type == "file")
        timed_out = next(
            (e for e in elements if e.name == "timedOut" and e.element_type == "function"),
            None,
        )
        add_timeout = next(
            (e for e in elements if e.name == "addTimeout" and e.element_type == "function"),
            None,
        )
        cancel = next(
            (e for e in elements if e.name == "cancel" and e.element_type == "function"),
            None,
        )
        timer = next(
            (e for e in elements if e.name == "timer" and e.element_type == "variable"),
            None,
        )

        assert timed_out is not None, "Expected function 'timedOut' not found"
        assert add_timeout is not None, "Expected function 'addTimeout' not found"
        assert cancel is not None, "Expected function 'cancel' not found"
        assert timer is not None, "Expected variable 'timer' not found"

        # timedOut should be child of file
        assert timed_out.parent_id == file_elem.element_id, (
            f"timedOut should be child of file, got parent_id={timed_out.parent_id}"
        )

        # addTimeout should be child of timedOut (NOT file)
        assert add_timeout.parent_id == timed_out.element_id, (
            f"addTimeout should be child of timedOut, got parent_id={add_timeout.parent_id}"
        )

        # cancel should be child of addTimeout (NOT file or timedOut)
        assert cancel.parent_id == add_timeout.element_id, (
            f"cancel should be child of addTimeout, got parent_id={cancel.parent_id}"
        )

        # timer should be child of addTimeout
        assert timer.parent_id == add_timeout.element_id, (
            f"timer should be child of addTimeout, got parent_id={timer.parent_id}"
        )

    def test_nested_function_in_regular_function(self):
        """Regular function declarations nested inside functions should have correct parent."""
        code = (
            "function outer() {\n"
            "  const inner = () => {\n"
            "    return 42;\n"
            "  };\n"
            "  return inner();\n"
            "}"
        )

        parser = JavaScriptParser("javascript")
        file_info = FileInfo(
            relative_path="test.js",
            absolute_path=Path("/fake/test.js"),
            language="javascript",
        )

        elements = parser.parse(code, file_info, "scope", "repo", "main")

        file_elem = next(e for e in elements if e.element_type == "file")
        outer = next(
            (e for e in elements if e.name == "outer" and e.element_type == "function"),
            None,
        )
        inner = next(
            (e for e in elements if e.name == "inner" and e.element_type == "function"),
            None,
        )

        assert outer is not None, "Expected function 'outer' not found"
        assert inner is not None, "Expected function 'inner' not found"

        # outer should be child of file
        assert outer.parent_id == file_elem.element_id

        # inner should be child of outer (NOT file)
        assert inner.parent_id == outer.element_id, (
            f"inner should be child of outer, got parent_id={inner.parent_id}"
        )

    def test_arrow_function_nested_in_class_method(self):
        """Arrow functions inside class methods should have method as parent."""
        code = (
            "class EventEmitter {\n"
            "  on(event: string, callback: () => void): void {\n"
            "    const wrappedCallback = (): void => {\n"
            "      callback();\n"
            "    };\n"
            "    this.listeners.push(wrappedCallback);\n"
            "  }\n"
            "}"
        )

        parser = JavaScriptParser("typescript")
        file_info = FileInfo(
            relative_path="emitter.ts",
            absolute_path=Path("/fake/emitter.ts"),
            language="typescript",
        )

        elements = parser.parse(code, file_info, "scope", "repo", "main")

        on_method = next(
            (e for e in elements if e.name == "on" and e.element_type == "method"),
            None,
        )
        wrapped = next(
            (e for e in elements if e.name == "wrappedCallback" and e.element_type == "function"),
            None,
        )

        assert on_method is not None, "Expected method 'on' not found"
        assert wrapped is not None, (
            f"Expected function 'wrappedCallback' not found. "
            f"Elements: {[(e.name, e.element_type) for e in elements]}"
        )

        # wrappedCallback should be child of on method (NOT class or file)
        assert wrapped.parent_id == on_method.element_id, (
            f"wrappedCallback should be child of on method, got parent_id={wrapped.parent_id}"
        )
