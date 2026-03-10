"""Test that parser_lab_analyze extracts class members (methods, fields, etc.).

This tests the fix for ZOD-007: Parser Lab's analyze function should extract
class members (methods, getters, setters, static arrow methods), not just the
class itself.
"""

from unittest.mock import patch

from magaldi_mcp.tools.parser_lab import parser_lab_analyze


class TestParserLabClassMembers:
    """Verify parser_lab_analyze returns class members alongside the class element."""

    def test_typescript_class_methods_extracted(self):
        """Parser Lab should extract methods from TypeScript class bodies."""
        code = """\
class ZodString extends ZodType {
  static create = (params?: string): ZodString => {
    return new ZodString(params);
  };

  _parse(input: string): string {
    return input;
  }
}"""
        with patch("magaldi_mcp.tools.parser_lab._check_magaldi_scope"):
            result = parser_lab_analyze(code=code, language="typescript")

        assert "error" not in result, f"Unexpected error: {result.get('error')}"

        element_names = [(e["type"], e["name"]) for e in result["elements"]]
        assert ("class", "ZodString") in element_names, (
            f"Expected class 'ZodString', got: {element_names}"
        )
        assert ("method", "create") in element_names, (
            f"Expected method 'create', got: {element_names}"
        )
        assert ("method", "_parse") in element_names, (
            f"Expected method '_parse', got: {element_names}"
        )

    def test_javascript_class_methods_extracted(self):
        """Parser Lab should extract methods from JavaScript class bodies."""
        code = """\
class BoxOffice {
  constructor(name) {
    this.name = name;
  }

  sellTicket(showId) {
    return { showId, sold: true };
  }
}"""
        with patch("magaldi_mcp.tools.parser_lab._check_magaldi_scope"):
            result = parser_lab_analyze(code=code, language="javascript")

        assert "error" not in result
        element_names = [(e["type"], e["name"]) for e in result["elements"]]
        assert ("class", "BoxOffice") in element_names
        assert ("method", "constructor") in element_names, (
            f"Expected method 'constructor', got: {element_names}"
        )
        assert ("method", "sellTicket") in element_names, (
            f"Expected method 'sellTicket', got: {element_names}"
        )

    def test_python_class_methods_extracted(self):
        """Parser Lab should extract methods from Python class bodies."""
        code = """\
class MyClass:
    def __init__(self):
        self.value = 0

    def process(self, data):
        return data
"""
        with patch("magaldi_mcp.tools.parser_lab._check_magaldi_scope"):
            result = parser_lab_analyze(code=code, language="python")

        assert "error" not in result
        element_names = [(e["type"], e["name"]) for e in result["elements"]]
        assert ("class", "MyClass") in element_names
        assert ("method", "__init__") in element_names, (
            f"Expected method '__init__', got: {element_names}"
        )
        assert ("method", "process") in element_names, (
            f"Expected method 'process', got: {element_names}"
        )

    def test_class_fields_extracted(self):
        """Parser Lab should extract class fields (non-method members)."""
        code = """\
class Config {
  maxRetries = 3;
  timeout = 5000;

  reset() {
    this.maxRetries = 3;
  }
}"""
        with patch("magaldi_mcp.tools.parser_lab._check_magaldi_scope"):
            result = parser_lab_analyze(code=code, language="javascript")

        assert "error" not in result
        element_types_names = [(e["type"], e["name"]) for e in result["elements"]]
        assert ("class", "Config") in element_types_names
        assert ("method", "reset") in element_types_names, (
            f"Expected method 'reset', got: {element_types_names}"
        )
        # Fields should be extracted as variables
        field_names = [e["name"] for e in result["elements"] if e["type"] == "variable"]
        assert "maxRetries" in field_names or "timeout" in field_names, (
            f"Expected at least one field variable, got: {element_types_names}"
        )

    def test_php_class_methods_extracted(self):
        """Parser Lab should extract methods from PHP class bodies."""
        code = """\
<?php
class UserService {
    public function findUser($id) {
        return $this->db->find($id);
    }

    private function validate($data) {
        return true;
    }
}"""
        with patch("magaldi_mcp.tools.parser_lab._check_magaldi_scope"):
            result = parser_lab_analyze(code=code, language="php")

        assert "error" not in result
        element_names = [(e["type"], e["name"]) for e in result["elements"]]
        assert ("class", "UserService") in element_names
        assert ("method", "findUser") in element_names, (
            f"Expected method 'findUser', got: {element_names}"
        )
        assert ("method", "validate") in element_names, (
            f"Expected method 'validate', got: {element_names}"
        )

    def test_element_count_includes_members(self):
        """Element count should include class members."""
        code = """\
class Foo {
  bar() { return 1; }
  baz() { return 2; }
}"""
        with patch("magaldi_mcp.tools.parser_lab._check_magaldi_scope"):
            result = parser_lab_analyze(code=code, language="javascript")

        # Should have at least 3: class Foo + method bar + method baz
        assert result["element_count"] >= 3, (
            f"Expected at least 3 elements (class + 2 methods), got {result['element_count']}: "
            f"{[(e['type'], e['name']) for e in result['elements']]}"
        )
