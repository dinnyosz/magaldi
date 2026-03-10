"""Auto-generated parser test for typescript_static_arrow_property."""

from pathlib import Path

from magaldi_core.change_detection import FileInfo
from magaldi_core.parsers.javascript import JavaScriptParser


class TestTypescriptStaticArrowProperty:
    """Tests for typescript_static_arrow_property extraction."""

    def _parse_typescript(self, code: str):
        parser = JavaScriptParser("typescript")
        file_info = FileInfo(
            relative_path="test.ts",
            absolute_path=Path("/fake/test.ts"),
            language="typescript",
        )
        return parser.parse(code, file_info, "scope", "repo", "main")

    def test_static_arrow_method_extracted(self):
        """Static arrow function class property should be extracted as a method."""
        code = """\
class ZodString extends ZodType {
  static create = (params?: string | core.$ZodStringParams): ZodString => {
    return new ZodString({
      type: "string",
      ...util.normalizeParams(params),
    });
  };

  _parse(input: string): string {
    return input;
  }
}"""
        elements = self._parse_typescript(code)

        # Class should be extracted
        cls = next(
            (e for e in elements if e.name == "ZodString" and e.element_type == "class"),
            None,
        )
        assert cls is not None, "Expected class 'ZodString' not found"

        # Static arrow function should be extracted as a method
        create = next(
            (e for e in elements if e.name == "create" and e.element_type == "method"),
            None,
        )
        assert create is not None, (
            f"Expected method 'create' not found. Got: "
            f"{[(e.name, e.element_type) for e in elements]}"
        )
        assert create.parent_id == cls.element_id, "create should be a child of ZodString"
        assert "static" in (create.decorators or []), "create should have 'static' decorator"

        # Regular method should also be extracted
        parse_method = next(
            (e for e in elements if e.name == "_parse" and e.element_type == "method"),
            None,
        )
        assert parse_method is not None, "Expected method '_parse' not found"
        assert parse_method.parent_id == cls.element_id

    def test_static_arrow_method_has_signature(self):
        """Static arrow function method should have a meaningful signature."""
        code = """\
class Foo {
  static bar = (x: number): string => {
    return String(x);
  };
}"""
        elements = self._parse_typescript(code)

        bar = next(
            (e for e in elements if e.name == "bar" and e.element_type == "method"),
            None,
        )
        assert bar is not None, "Expected method 'bar' not found"
        assert bar.signature is not None, "Method should have a signature"
        assert "static" in bar.signature, "Signature should include 'static'"
        assert "bar" in bar.signature

    def test_non_static_arrow_field_method(self):
        """Non-static arrow function class property should also be extracted as method."""
        code = """\
class Handler {
  handle = (req: Request) => {
    return new Response();
  };
}"""
        elements = self._parse_typescript(code)

        handle = next(
            (e for e in elements if e.name == "handle" and e.element_type == "method"),
            None,
        )
        assert handle is not None, (
            f"Expected method 'handle' not found. Got: "
            f"{[(e.name, e.element_type) for e in elements]}"
        )

    def test_multiple_static_arrow_methods(self):
        """Multiple static arrow function methods in same class should all be extracted."""
        code = """\
class ZodType {
  static create = (params?: any): ZodType => {
    return new ZodType(params);
  };

  static createWithCoerce = (params?: any): ZodType => {
    return new ZodType(params);
  };

  validate(data: any): boolean {
    return true;
  }
}"""
        elements = self._parse_typescript(code)

        methods = [e for e in elements if e.element_type == "method"]
        method_names = [m.name for m in methods]
        assert "create" in method_names, f"Expected 'create' method, got: {method_names}"
        assert "createWithCoerce" in method_names, f"Expected 'createWithCoerce' method, got: {method_names}"
        assert "validate" in method_names, f"Expected 'validate' method, got: {method_names}"
