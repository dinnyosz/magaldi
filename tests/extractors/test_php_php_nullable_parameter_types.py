"""Auto-generated parser test for php_nullable_parameter_types."""

from pathlib import Path

from magaldi_core.change_detection import FileInfo
from magaldi_core.code_parser import PhpParser


class TestPhpNullableParameterTypes:
    """Tests for php_nullable_parameter_types extraction."""

    def test_php_nullable_parameter_types(self):
        code = (
            "<?php\n"
            "class Example {\n"
            "    public function test(?int $count = null, ?string $name = null, "
            "?ResponseInterface $response = null): ?string {\n"
            "        return $name;\n"
            "    }\n"
            "}"
        )

        parser = PhpParser()
        file_info = FileInfo(
            relative_path="test.php",
            absolute_path=Path("/fake/test.php"),
            language="php",
        )

        elements = parser.parse(code, file_info, "scope", "repo", "main")

        # Check for Example class
        class_elem = next(
            (e for e in elements if e.name == "Example" and e.element_type == "class"),
            None,
        )
        assert class_elem is not None, "Expected class 'Example' not found"

        # Check for test method
        method_elem = next(
            (e for e in elements if e.name == "test" and e.element_type == "method"),
            None,
        )
        assert method_elem is not None, "Expected method 'test' not found"

        # Verify parameter types are correctly extracted (the main fix)
        assert method_elem.parameters, "Method should have parameters"
        assert len(method_elem.parameters) == 3, (
            f"Expected 3 params, got {len(method_elem.parameters)}"
        )

        param_count = method_elem.parameters[0]
        assert param_count["name"] == "count", f"Expected 'count', got '{param_count['name']}'"
        assert param_count["type"] == "?int", f"Expected '?int', got '{param_count['type']}'"
        assert param_count["default"] == "null", (
            f"Expected default 'null', got '{param_count['default']}'"
        )

        param_name = method_elem.parameters[1]
        assert param_name["name"] == "name", f"Expected 'name', got '{param_name['name']}'"
        assert param_name["type"] == "?string", f"Expected '?string', got '{param_name['type']}'"
        assert param_name["default"] == "null", (
            f"Expected default 'null', got '{param_name['default']}'"
        )

        param_response = method_elem.parameters[2]
        assert param_response["name"] == "response", (
            f"Expected 'response', got '{param_response['name']}'"
        )
        assert param_response["type"] == "?ResponseInterface", (
            f"Expected '?ResponseInterface', got '{param_response['type']}'"
        )
        assert param_response["default"] == "null", (
            f"Expected default 'null', got '{param_response['default']}'"
        )

        # Verify return type is also correctly extracted
        assert method_elem.return_type == "?string", (
            f"Expected return type '?string', got '{method_elem.return_type}'"
        )

    def test_nullable_type_without_default(self):
        """Nullable types should work even without default values."""
        code = (
            "<?php\n"
            "function greet(?string $name): void {\n"
            "    echo $name ?? 'World';\n"
            "}"
        )

        parser = PhpParser()
        file_info = FileInfo(
            relative_path="test.php",
            absolute_path=Path("/fake/test.php"),
            language="php",
        )

        elements = parser.parse(code, file_info, "scope", "repo", "main")

        func_elem = next(
            (e for e in elements if e.name == "greet" and e.element_type == "function"),
            None,
        )
        assert func_elem is not None, "Expected function 'greet' not found"
        assert func_elem.parameters, "Function should have parameters"
        assert len(func_elem.parameters) == 1

        param = func_elem.parameters[0]
        assert param["name"] == "name", f"Expected 'name', got '{param['name']}'"
        assert param["type"] == "?string", f"Expected '?string', got '{param['type']}'"
        assert param.get("default") is None, f"Expected no default, got '{param.get('default')}'"

    def test_mixed_nullable_and_regular_types(self):
        """Mixed nullable and regular parameter types."""
        code = (
            "<?php\n"
            "function process(int $id, ?string $label = null, array $items = []): ?bool {\n"
            "    return true;\n"
            "}"
        )

        parser = PhpParser()
        file_info = FileInfo(
            relative_path="test.php",
            absolute_path=Path("/fake/test.php"),
            language="php",
        )

        elements = parser.parse(code, file_info, "scope", "repo", "main")

        func_elem = next(
            (e for e in elements if e.name == "process" and e.element_type == "function"),
            None,
        )
        assert func_elem is not None, "Expected function 'process' not found"
        assert func_elem.parameters
        assert len(func_elem.parameters) == 3

        assert func_elem.parameters[0]["type"] == "int"
        assert func_elem.parameters[1]["type"] == "?string"
        assert func_elem.parameters[2]["type"] == "array"

        assert func_elem.return_type == "?bool"
