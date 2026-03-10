"""Parser test for TypeScript static arrow function class properties.

Verifies that `static propName = () => {}` inside a class body is extracted
as a method (not a field), with the correct signature, decorators, visibility,
and parameter metadata.
"""

from pathlib import Path

from magaldi_core.change_detection import FileInfo
from magaldi_core.code_parser import JavaScriptParser


class TestTypescriptStaticArrowClassProperty:
    """Tests for static arrow function class property extraction."""

    CODE = """\
class ZodString extends ZodType {
  _parse(input: ParseInput): ParseReturnType<string> {
    return OK(input.data);
  }

  static create = (params?: RawCreateParams): ZodString => {
    return new ZodString({
      checks: [],
      typeName: ZodFirstPartyTypeKind.ZodString,
      coerce: false,
      ...processCreateParams(params),
    });
  };

  static override = (config: Config): ZodString => {
    return new ZodString(config);
  };

  url(message?: errorUtil.ErrMessage) {
    return this._addCheck({ kind: "url", ...errorUtil.errToObj(message) });
  }

  static fromConfig(config: Config): ZodString {
    return new ZodString(config);
  }
}"""

    def _parse(self):
        parser = JavaScriptParser("typescript")
        file_info = FileInfo(
            relative_path="test.ts",
            absolute_path=Path("/fake/test.ts"),
            language="typescript",
        )
        return parser.parse(self.CODE, file_info, "scope", "repo", "main")

    def _find(self, elements, name, element_type):
        return next(
            (e for e in elements if e.name == name and e.element_type == element_type),
            None,
        )

    def test_class_extracted(self):
        elements = self._parse()
        cls = self._find(elements, "ZodString", "class")
        assert cls is not None, "Expected class 'ZodString' not found"

    def test_regular_method_extracted(self):
        elements = self._parse()
        method = self._find(elements, "_parse", "method")
        assert method is not None, "Expected method '_parse' not found"
        assert method.element_type == "method"

    def test_static_arrow_create_extracted_as_method(self):
        elements = self._parse()
        create = self._find(elements, "create", "method")
        assert create is not None, "static arrow property 'create' should be extracted as method"
        assert create.element_type == "method"
        assert "static" in create.decorators, "should have 'static' decorator"

    def test_static_arrow_override_extracted_as_method(self):
        elements = self._parse()
        override = self._find(elements, "override", "method")
        assert override is not None, "static arrow property 'override' should be extracted as method"
        assert override.element_type == "method"
        assert "static" in override.decorators, "should have 'static' decorator"

    def test_instance_method_extracted(self):
        elements = self._parse()
        url = self._find(elements, "url", "method")
        assert url is not None, "Expected method 'url' not found"

    def test_regular_static_method_extracted(self):
        elements = self._parse()
        from_config = self._find(elements, "fromConfig", "method")
        assert from_config is not None, "Expected static method 'fromConfig' not found"

    def test_static_arrow_signature_includes_static_prefix(self):
        elements = self._parse()
        create = self._find(elements, "create", "method")
        assert create is not None
        assert create.signature is not None
        assert create.signature.startswith("static "), (
            f"Signature should start with 'static ', got: {create.signature}"
        )

    def test_static_arrow_return_type_extracted(self):
        elements = self._parse()
        create = self._find(elements, "create", "method")
        assert create is not None
        assert create.return_type == "ZodString", (
            f"Expected return type 'ZodString', got: {create.return_type}"
        )

    def test_static_arrow_parameters_extracted(self):
        elements = self._parse()
        create = self._find(elements, "create", "method")
        assert create is not None
        assert create.parameters is not None and len(create.parameters) > 0, (
            "Expected parameters to be extracted for static arrow method"
        )
        param = create.parameters[0]
        assert param["name"] == "params", f"Expected param name 'params', got: {param['name']}"

    def test_static_arrow_visibility_is_public(self):
        elements = self._parse()
        create = self._find(elements, "create", "method")
        assert create is not None
        assert create.visibility == "public", (
            f"Expected visibility 'public', got: {create.visibility}"
        )

    def test_static_arrow_not_extracted_as_variable(self):
        """Static arrow functions should NOT appear as variables/fields."""
        elements = self._parse()
        create_var = self._find(elements, "create", "variable")
        assert create_var is None, (
            "static arrow property 'create' should NOT be extracted as a variable"
        )
        override_var = self._find(elements, "override", "variable")
        assert override_var is None, (
            "static arrow property 'override' should NOT be extracted as a variable"
        )


class TestTypescriptStaticArrowWithVisibility:
    """Tests for static arrow functions with TypeScript visibility modifiers."""

    CODE = """\
class Example {
  private static create = (params: string): Example => {
    return new Example();
  };

  public static build = async (config: Config): Promise<Example> => {
    return new Example();
  };

  protected static init = (): void => {
    console.log("init");
  };
}"""

    def _parse(self):
        parser = JavaScriptParser("typescript")
        file_info = FileInfo(
            relative_path="test.ts",
            absolute_path=Path("/fake/test.ts"),
            language="typescript",
        )
        return parser.parse(self.CODE, file_info, "scope", "repo", "main")

    def _find(self, elements, name, element_type):
        return next(
            (e for e in elements if e.name == name and e.element_type == element_type),
            None,
        )

    def test_private_static_arrow_extracted(self):
        elements = self._parse()
        create = self._find(elements, "create", "method")
        assert create is not None, "private static arrow 'create' should be extracted"
        assert create.visibility == "private"
        assert "static" in create.decorators

    def test_public_static_async_arrow_extracted(self):
        elements = self._parse()
        build = self._find(elements, "build", "method")
        assert build is not None, "public static async arrow 'build' should be extracted"
        assert build.visibility == "public"
        assert build.is_async is True
        assert "static" in build.decorators

    def test_protected_static_arrow_extracted(self):
        elements = self._parse()
        init = self._find(elements, "init", "method")
        assert init is not None, "protected static arrow 'init' should be extracted"
        assert init.visibility == "protected"
        assert "static" in init.decorators


class TestJavascriptStaticArrowClassProperty:
    """Tests for static arrow function class properties in JavaScript (not TypeScript)."""

    CODE = """\
class MyClass {
  static create = (params) => {
    return new MyClass(params);
  };

  regularMethod() {
    return true;
  }
}"""

    def _parse(self):
        parser = JavaScriptParser("javascript")
        file_info = FileInfo(
            relative_path="test.js",
            absolute_path=Path("/fake/test.js"),
            language="javascript",
        )
        return parser.parse(self.CODE, file_info, "scope", "repo", "main")

    def _find(self, elements, name, element_type):
        return next(
            (e for e in elements if e.name == name and e.element_type == element_type),
            None,
        )

    def test_js_static_arrow_extracted_as_method(self):
        """JavaScript uses field_definition (not public_field_definition) for class fields."""
        elements = self._parse()
        create = self._find(elements, "create", "method")
        assert create is not None, "JS static arrow 'create' should be extracted as method"
        assert "static" in create.decorators

    def test_js_regular_method_still_works(self):
        elements = self._parse()
        method = self._find(elements, "regularMethod", "method")
        assert method is not None, "Regular method should still be extracted"
