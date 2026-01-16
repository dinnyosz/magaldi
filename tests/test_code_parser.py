"""Tests for the code parser module (Phase 3)."""

from pathlib import Path

import pytest

from magaldi_core.change_detection import ChangeManifest, FileInfo
from magaldi_core.code_parser import (
    CodeElement,
    JavaScriptParser,
    ParsedFile,
    ParsingResult,
    PythonParser,
    generate_element_id,
    parse_file,
    parse_files,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def python_code() -> str:
    return '''"""Module docstring."""

CONSTANT = 42
MAX_SIZE = 100

def standalone_function(x: int, y: int = 0) -> int:
    """Add two numbers."""
    return x + y

async def async_function():
    """Async example."""
    pass

class MyClass:
    """A sample class."""

    def __init__(self, name: str):
        """Initialize."""
        self.name = name

    def greet(self) -> str:
        """Return greeting."""
        return f"Hello, {self.name}!"

    @staticmethod
    def static_method():
        """Static method."""
        pass

    def _private_method(self):
        """Private method."""
        pass

class _PrivateClass:
    """A private class."""
    pass
'''


@pytest.fixture
def javascript_code() -> str:
    return '''
function regularFunction(a, b) {
    return a + b;
}

async function asyncFunction() {
    await something();
}

const arrowFunc = (x) => x * 2;

class MyClass {
    constructor(name) {
        this.name = name;
    }

    greet() {
        return `Hello, ${this.name}!`;
    }
}

export class ExportedClass {
    doSomething() {}
}
'''


@pytest.fixture
def temp_python_file(tmp_path: Path, python_code: str) -> FileInfo:
    file = tmp_path / "test_module.py"
    file.write_text(python_code)
    return FileInfo(
        relative_path="test_module.py",
        absolute_path=file,
        language="python",
        hash="abc123",
    )


@pytest.fixture
def temp_js_file(tmp_path: Path, javascript_code: str) -> FileInfo:
    file = tmp_path / "test_module.js"
    file.write_text(javascript_code)
    return FileInfo(
        relative_path="test_module.js",
        absolute_path=file,
        language="javascript",
        hash="def456",
    )


# =============================================================================
# ELEMENT ID GENERATION
# =============================================================================


class TestGenerateElementId:
    """Tests for element ID generation."""

    def test_format(self):
        element_id = generate_element_id(
            scope="backend",
            repository="auth-service",
            username="main",
            relative_path="src/auth.py",
            element_type="function",
            name="login",
            line_start=42,
        )

        assert element_id == "backend:auth-service:main:src/auth.py:function:login:42"

    def test_different_users_different_ids(self):
        id1 = generate_element_id("scope", "repo", "main", "file.py", "function", "foo", 1)
        id2 = generate_element_id("scope", "repo", "alice", "file.py", "function", "foo", 1)

        assert id1 != id2

    def test_different_lines_different_ids(self):
        id1 = generate_element_id("scope", "repo", "main", "file.py", "function", "foo", 1)
        id2 = generate_element_id("scope", "repo", "main", "file.py", "function", "foo", 2)

        assert id1 != id2


# =============================================================================
# PYTHON PARSER
# =============================================================================


class TestPythonParser:
    """Tests for Python parsing."""

    def test_extracts_file_element(self, python_code: str):
        parser = PythonParser()
        file_info = FileInfo(
            relative_path="module.py",
            absolute_path=Path("/fake/module.py"),
            language="python",
        )

        elements = parser.parse(python_code, file_info, "scope", "repo", "main")

        file_elements = [e for e in elements if e.element_type == "file"]
        assert len(file_elements) == 1
        assert file_elements[0].name == "module.py"
        assert file_elements[0].level == 0

    def test_extracts_classes(self, python_code: str):
        parser = PythonParser()
        file_info = FileInfo(
            relative_path="module.py",
            absolute_path=Path("/fake/module.py"),
            language="python",
        )

        elements = parser.parse(python_code, file_info, "scope", "repo", "main")

        classes = [e for e in elements if e.element_type == "class"]
        class_names = [c.name for c in classes]

        assert "MyClass" in class_names
        assert "_PrivateClass" in class_names

    def test_extracts_class_docstring(self, python_code: str):
        parser = PythonParser()
        file_info = FileInfo(
            relative_path="module.py",
            absolute_path=Path("/fake/module.py"),
            language="python",
        )

        elements = parser.parse(python_code, file_info, "scope", "repo", "main")

        my_class = next(e for e in elements if e.name == "MyClass" and e.element_type == "class")
        assert my_class.docstring is not None
        assert "sample class" in my_class.docstring.lower()

    def test_extracts_standalone_functions(self, python_code: str):
        parser = PythonParser()
        file_info = FileInfo(
            relative_path="module.py",
            absolute_path=Path("/fake/module.py"),
            language="python",
        )

        elements = parser.parse(python_code, file_info, "scope", "repo", "main")

        functions = [e for e in elements if e.element_type == "function"]
        func_names = [f.name for f in functions]

        assert "standalone_function" in func_names
        assert "async_function" in func_names

    def test_extracts_methods(self, python_code: str):
        parser = PythonParser()
        file_info = FileInfo(
            relative_path="module.py",
            absolute_path=Path("/fake/module.py"),
            language="python",
        )

        elements = parser.parse(python_code, file_info, "scope", "repo", "main")

        methods = [e for e in elements if e.element_type == "method"]
        method_names = [m.name for m in methods]

        assert "__init__" in method_names
        assert "greet" in method_names
        assert "static_method" in method_names
        assert "_private_method" in method_names

    def test_extracts_constants(self, python_code: str):
        parser = PythonParser()
        file_info = FileInfo(
            relative_path="module.py",
            absolute_path=Path("/fake/module.py"),
            language="python",
        )

        elements = parser.parse(python_code, file_info, "scope", "repo", "main")

        # CONSTANT and MAX_SIZE are uppercase, so they're extracted as 'constant' type
        constants = [e for e in elements if e.element_type == "constant"]
        const_names = [c.name for c in constants]

        assert "CONSTANT" in const_names
        assert "MAX_SIZE" in const_names

    def test_async_function_detection(self, python_code: str):
        parser = PythonParser()
        file_info = FileInfo(
            relative_path="module.py",
            absolute_path=Path("/fake/module.py"),
            language="python",
        )

        elements = parser.parse(python_code, file_info, "scope", "repo", "main")

        async_func = next(e for e in elements if e.name == "async_function")
        assert async_func.is_async is True

        sync_func = next(e for e in elements if e.name == "standalone_function")
        assert sync_func.is_async is False

    def test_visibility_detection(self, python_code: str):
        parser = PythonParser()
        file_info = FileInfo(
            relative_path="module.py",
            absolute_path=Path("/fake/module.py"),
            language="python",
        )

        elements = parser.parse(python_code, file_info, "scope", "repo", "main")

        # Single underscore = protected (by Python convention)
        protected_class = next(e for e in elements if e.name == "_PrivateClass")
        assert protected_class.visibility == "protected"

        public_class = next(e for e in elements if e.name == "MyClass")
        assert public_class.visibility == "public"

        protected_method = next(e for e in elements if e.name == "_private_method")
        assert protected_method.visibility == "protected"

    def test_decorator_extraction(self, python_code: str):
        parser = PythonParser()
        file_info = FileInfo(
            relative_path="module.py",
            absolute_path=Path("/fake/module.py"),
            language="python",
        )

        elements = parser.parse(python_code, file_info, "scope", "repo", "main")

        static_method = next(e for e in elements if e.name == "static_method")
        assert "staticmethod" in static_method.decorators

    def test_signature_extraction(self, python_code: str):
        parser = PythonParser()
        file_info = FileInfo(
            relative_path="module.py",
            absolute_path=Path("/fake/module.py"),
            language="python",
        )

        elements = parser.parse(python_code, file_info, "scope", "repo", "main")

        func = next(e for e in elements if e.name == "standalone_function")
        assert "def standalone_function" in func.signature
        assert "int" in func.signature

    def test_method_parent_id_set(self, python_code: str):
        parser = PythonParser()
        file_info = FileInfo(
            relative_path="module.py",
            absolute_path=Path("/fake/module.py"),
            language="python",
        )

        elements = parser.parse(python_code, file_info, "scope", "repo", "main")

        greet_method = next(e for e in elements if e.name == "greet")
        my_class = next(e for e in elements if e.name == "MyClass" and e.element_type == "class")

        assert greet_method.parent_id == my_class.element_id

    def test_hierarchy_levels(self, python_code: str):
        parser = PythonParser()
        file_info = FileInfo(
            relative_path="module.py",
            absolute_path=Path("/fake/module.py"),
            language="python",
        )

        elements = parser.parse(python_code, file_info, "scope", "repo", "main")

        file_elem = next(e for e in elements if e.element_type == "file")
        assert file_elem.level == 0

        class_elem = next(e for e in elements if e.element_type == "class")
        assert class_elem.level == 1

        method_elem = next(e for e in elements if e.element_type == "method")
        assert method_elem.level == 2

        func_elem = next(e for e in elements if e.element_type == "function")
        assert func_elem.level == 2

        # CONSTANT and MAX_SIZE are uppercase, so they're 'constant' type
        const_elem = next(e for e in elements if e.element_type == "constant")
        assert const_elem.level == 3


# =============================================================================
# JAVASCRIPT PARSER
# =============================================================================


class TestJavaScriptParser:
    """Tests for JavaScript parsing."""

    def test_extracts_file_element(self, javascript_code: str):
        parser = JavaScriptParser()
        file_info = FileInfo(
            relative_path="module.js",
            absolute_path=Path("/fake/module.js"),
            language="javascript",
        )

        elements = parser.parse(javascript_code, file_info, "scope", "repo", "main")

        file_elements = [e for e in elements if e.element_type == "file"]
        assert len(file_elements) == 1

    def test_extracts_functions(self, javascript_code: str):
        parser = JavaScriptParser()
        file_info = FileInfo(
            relative_path="module.js",
            absolute_path=Path("/fake/module.js"),
            language="javascript",
        )

        elements = parser.parse(javascript_code, file_info, "scope", "repo", "main")

        functions = [e for e in elements if e.element_type == "function"]
        func_names = [f.name for f in functions]

        assert "regularFunction" in func_names
        assert "asyncFunction" in func_names
        assert "arrowFunc" in func_names

    def test_extracts_classes(self, javascript_code: str):
        parser = JavaScriptParser()
        file_info = FileInfo(
            relative_path="module.js",
            absolute_path=Path("/fake/module.js"),
            language="javascript",
        )

        elements = parser.parse(javascript_code, file_info, "scope", "repo", "main")

        classes = [e for e in elements if e.element_type == "class"]
        class_names = [c.name for c in classes]

        assert "MyClass" in class_names
        assert "ExportedClass" in class_names

    def test_async_function_detection(self, javascript_code: str):
        parser = JavaScriptParser()
        file_info = FileInfo(
            relative_path="module.js",
            absolute_path=Path("/fake/module.js"),
            language="javascript",
        )

        elements = parser.parse(javascript_code, file_info, "scope", "repo", "main")

        async_func = next(e for e in elements if e.name == "asyncFunction")
        assert async_func.is_async is True


# =============================================================================
# PARSE FILE FUNCTION
# =============================================================================


class TestParseFile:
    """Tests for the parse_file function."""

    def test_parses_python_file(self, temp_python_file: FileInfo):
        result = parse_file(temp_python_file, "scope", "repo", "main")

        assert isinstance(result, ParsedFile)
        assert len(result.elements) > 0
        assert result.line_count > 0

    def test_parses_javascript_file(self, temp_js_file: FileInfo):
        result = parse_file(temp_js_file, "scope", "repo", "main")

        assert isinstance(result, ParsedFile)
        assert len(result.elements) > 0

    def test_handles_missing_file(self, tmp_path: Path):
        file_info = FileInfo(
            relative_path="missing.py",
            absolute_path=tmp_path / "missing.py",
            language="python",
        )

        result = parse_file(file_info, "scope", "repo", "main")

        assert len(result.parse_errors) > 0

    def test_handles_unsupported_language(self, tmp_path: Path):
        file = tmp_path / "test.xyz"
        file.write_text("content")

        file_info = FileInfo(
            relative_path="test.xyz",
            absolute_path=file,
            language="unknown",
        )

        result = parse_file(file_info, "scope", "repo", "main")

        assert len(result.parse_errors) > 0
        assert "parser" in result.parse_errors[0].lower()


# =============================================================================
# PARSE FILES FUNCTION
# =============================================================================


class TestParseFiles:
    """Tests for the parse_files function."""

    def test_parses_manifest_files(self, temp_python_file: FileInfo, temp_js_file: FileInfo):
        from datetime import datetime

        manifest = ChangeManifest(
            scope="test-scope",
            repository="test-repo",
            username="main",
            timestamp=datetime.now(),
            total_files_scanned=2,
            new_files=[temp_python_file, temp_js_file],
        )

        result = parse_files(manifest)

        assert isinstance(result, ParsingResult)
        assert len(result.parsed_files) == 2
        assert result.total_elements > 0

    def test_handles_empty_manifest(self):
        from datetime import datetime

        manifest = ChangeManifest(
            scope="test-scope",
            repository="test-repo",
            username="main",
            timestamp=datetime.now(),
            total_files_scanned=0,
        )

        result = parse_files(manifest)

        assert len(result.parsed_files) == 0
        assert result.total_elements == 0


# =============================================================================
# DATA CLASSES
# =============================================================================


class TestCodeElement:
    """Tests for CodeElement dataclass."""

    def test_default_values(self):
        element = CodeElement()

        assert element.element_id == ""
        assert element.element_type == ""
        assert element.level == 0
        assert element.is_async is False
        assert element.visibility == "public"
        assert element.decorators == []
        assert element.parameters == []

    def test_is_test_default_false(self):
        element = CodeElement()
        assert element.is_test is False

    def test_is_test_can_be_set(self):
        element = CodeElement(is_test=True)
        assert element.is_test is True


class TestParsingResult:
    """Tests for ParsingResult dataclass."""

    def test_total_elements_calculation(self):
        result = ParsingResult(scope="s", repository="r", username="u")
        result.parsed_files = [
            ParsedFile(
                file_info=FileInfo("a.py", Path(), "python"),
                elements=[CodeElement(), CodeElement()],
            ),
            ParsedFile(
                file_info=FileInfo("b.py", Path(), "python"),
                elements=[CodeElement()],
            ),
        ]

        assert result.total_elements == 3

    def test_elements_by_type(self):
        result = ParsingResult(scope="s", repository="r", username="u")
        result.parsed_files = [
            ParsedFile(
                file_info=FileInfo("a.py", Path(), "python"),
                elements=[
                    CodeElement(element_type="class"),
                    CodeElement(element_type="function"),
                    CodeElement(element_type="function"),
                ],
            ),
        ]

        by_type = result.elements_by_type
        assert by_type["class"] == 1
        assert by_type["function"] == 2
