"""Tests for static class property arrow function extraction (Issue 1).

Tests that `public_field_definition` (TypeScript) and `field_definition` (JavaScript)
nodes inside class bodies where the value is an arrow_function or function expression
are extracted as methods, not variables.
"""

import pytest
from pathlib import Path

from magaldi_core.code_parser import JavaScriptParser
from magaldi_core.change_detection import FileInfo


class TestTypescriptStaticArrowMethod:
    """Tests for TypeScript static arrow method extraction."""

    def test_static_arrow_function_extracted_as_method(self):
        """Static arrow function properties should be extracted as methods."""
        code = (
            "class ZodString {\n"
            "  static create = (params?: RawCreateParams): ZodString => {\n"
            "    return new ZodString({});\n"
            "  };\n"
            "\n"
            "  static defaults = {\n"
            "    timeout: 1000,\n"
            "  };\n"
            "\n"
            "  normalMethod() {\n"
            "    return true;\n"
            "  }\n"
            "}"
        )

        parser = JavaScriptParser("typescript")
        file_info = FileInfo(
            relative_path="test.ts",
            absolute_path=Path("/fake/test.ts"),
            language="typescript",
        )

        elements = parser.parse(code, file_info, "scope", "repo", "main")

        # Check class exists
        class_elem = next(
            (e for e in elements if e.name == "ZodString" and e.element_type == "class"),
            None,
        )
        assert class_elem is not None, "Expected class 'ZodString' not found"

        # Check static arrow function is extracted as method
        create_elem = next(
            (e for e in elements if e.name == "create" and e.element_type == "method"),
            None,
        )
        assert create_elem is not None, (
            f"Expected method 'create' not found. "
            f"Elements: {[(e.name, e.element_type) for e in elements]}"
        )
        assert create_elem.parent_id == class_elem.element_id, (
            f"create should be child of ZodString, got parent_id={create_elem.parent_id}"
        )
        assert "static" in create_elem.decorators, "create should have 'static' decorator"

        # Check normalMethod is extracted as method
        normal_elem = next(
            (e for e in elements if e.name == "normalMethod" and e.element_type == "method"),
            None,
        )
        assert normal_elem is not None, "Expected method 'normalMethod' not found"
        assert normal_elem.parent_id == class_elem.element_id

        # Check defaults is extracted as variable (not a function)
        defaults_elem = next(
            (e for e in elements if e.name == "defaults" and e.element_type == "variable"),
            None,
        )
        assert defaults_elem is not None, (
            f"Expected variable 'defaults' not found. "
            f"Elements: {[(e.name, e.element_type) for e in elements]}"
        )

    def test_instance_arrow_function_extracted_as_method(self):
        """Non-static arrow function properties should also be extracted as methods."""
        code = (
            "class MyClass {\n"
            "  handleClick = (event: Event): void => {\n"
            "    console.log(event);\n"
            "  };\n"
            "}"
        )

        parser = JavaScriptParser("typescript")
        file_info = FileInfo(
            relative_path="test.ts",
            absolute_path=Path("/fake/test.ts"),
            language="typescript",
        )

        elements = parser.parse(code, file_info, "scope", "repo", "main")

        handler_elem = next(
            (e for e in elements if e.name == "handleClick" and e.element_type == "method"),
            None,
        )
        assert handler_elem is not None, (
            f"Expected method 'handleClick' not found. "
            f"Elements: {[(e.name, e.element_type) for e in elements]}"
        )
        # Should NOT have static decorator
        assert "static" not in (handler_elem.decorators or [])

    def test_async_static_arrow_function(self):
        """Async static arrow functions should be extracted with is_async=True."""
        code = (
            "class ApiClient {\n"
            "  static fetch = async (url: string): Promise<Response> => {\n"
            "    return await globalFetch(url);\n"
            "  };\n"
            "}"
        )

        parser = JavaScriptParser("typescript")
        file_info = FileInfo(
            relative_path="test.ts",
            absolute_path=Path("/fake/test.ts"),
            language="typescript",
        )

        elements = parser.parse(code, file_info, "scope", "repo", "main")

        fetch_elem = next(
            (e for e in elements if e.name == "fetch" and e.element_type == "method"),
            None,
        )
        assert fetch_elem is not None, (
            f"Expected method 'fetch' not found. "
            f"Elements: {[(e.name, e.element_type) for e in elements]}"
        )
        assert fetch_elem.is_async, "fetch should be async"
        assert "static" in fetch_elem.decorators


class TestJavascriptStaticArrowMethod:
    """Tests for JavaScript static arrow method extraction (field_definition)."""

    def test_js_static_arrow_function_extracted_as_method(self):
        """JavaScript field_definition with arrow function should be extracted as method."""
        code = (
            "class ZodString {\n"
            "  static create = (params) => {\n"
            "    return new ZodString({});\n"
            "  };\n"
            "\n"
            "  static defaults = {\n"
            "    timeout: 1000,\n"
            "  };\n"
            "\n"
            "  normalMethod() {\n"
            "    return true;\n"
            "  }\n"
            "}"
        )

        parser = JavaScriptParser("javascript")
        file_info = FileInfo(
            relative_path="test.js",
            absolute_path=Path("/fake/test.js"),
            language="javascript",
        )

        elements = parser.parse(code, file_info, "scope", "repo", "main")

        # Check static arrow function is extracted as method
        create_elem = next(
            (e for e in elements if e.name == "create" and e.element_type == "method"),
            None,
        )
        assert create_elem is not None, (
            f"Expected method 'create' not found. "
            f"Elements: {[(e.name, e.element_type) for e in elements]}"
        )
        assert "static" in create_elem.decorators

        # Check defaults is still a variable
        defaults_elem = next(
            (e for e in elements if e.name == "defaults" and e.element_type == "variable"),
            None,
        )
        assert defaults_elem is not None, "Expected variable 'defaults' not found"
