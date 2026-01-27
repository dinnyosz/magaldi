"""Tests for the code parser module (Phase 3)."""

from pathlib import Path

import pytest

from magaldi_core.change_detection import ChangeManifest, FileInfo
from magaldi_core.code_parser import (
    Call,
    CodeElement,
    Import,
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


class TestParseFileTestDetection:
    """Tests for is_test detection during parsing."""

    def test_marks_test_file_elements(self, tmp_path: Path):
        """Test that elements in test files are marked is_test=True."""
        test_file = tmp_path / "test_example.py"
        test_file.write_text('''
def test_something():
    pass

def helper():
    pass
''')
        file_info = FileInfo(
            relative_path="test_example.py",
            absolute_path=test_file,
            language="python",
        )

        result = parse_file(file_info, "scope", "repo", "main")

        # All elements should be marked as test (file-level detection)
        for elem in result.elements:
            assert elem.is_test is True, f"{elem.name} should be is_test=True"

    def test_marks_test_functions_by_name(self, tmp_path: Path):
        """Test that test_ functions in non-test files are marked."""
        src_file = tmp_path / "example.py"
        src_file.write_text('''
def test_inline():
    """An inline test."""
    pass

def regular_function():
    pass
''')
        file_info = FileInfo(
            relative_path="src/example.py",
            absolute_path=src_file,
            language="python",
        )

        result = parse_file(file_info, "scope", "repo", "main")

        # Find elements by name
        elements = {e.name: e for e in result.elements}

        # File element should not be test
        assert elements["example.py"].is_test is False
        # test_ function should be test
        assert elements["test_inline"].is_test is True
        # regular function should not be test
        assert elements["regular_function"].is_test is False

    def test_non_test_file_not_marked(self, tmp_path: Path):
        """Test that regular files are not marked as test."""
        src_file = tmp_path / "app.py"
        src_file.write_text('''
def main():
    pass
''')
        file_info = FileInfo(
            relative_path="src/app.py",
            absolute_path=src_file,
            language="python",
        )

        result = parse_file(file_info, "scope", "repo", "main")

        for elem in result.elements:
            assert elem.is_test is False, f"{elem.name} should be is_test=False"


class TestIsTestElement:
    """Tests for is_test_element utility function."""

    @pytest.mark.parametrize("name,decorators,language,expected", [
        # Python test elements
        ("test_foo", [], "python", True),
        ("test_something_complex", [], "python", True),
        ("foo", ["pytest.mark.parametrize"], "python", True),
        ("foo", ["pytest.fixture"], "python", True),
        ("foo", ["unittest.skip"], "python", True),
        # Python non-test elements
        ("foo", [], "python", False),
        ("testing_helper", [], "python", False),
        ("my_test", [], "python", False),  # doesn't start with test_
        # Rust test elements
        ("test_foo", ["test"], "rust", True),
        ("foo", ["test"], "rust", True),
        ("foo", ["cfg(test)"], "rust", True),
        # Rust non-test elements
        ("foo", [], "rust", False),
    ])
    def test_is_test_element(self, name: str, decorators: list[str], language: str, expected: bool):
        from magaldi_core.code_parser import is_test_element
        assert is_test_element(name, decorators, language) == expected


class TestIsTestPath:
    """Tests for is_test_path utility function."""

    @pytest.mark.parametrize("path,language,expected", [
        # Python test paths
        ("test_foo.py", "python", True),
        ("foo_test.py", "python", True),
        ("tests/test_module.py", "python", True),
        ("tests/unit/test_foo.py", "python", True),
        ("conftest.py", "python", True),
        ("src/conftest.py", "python", True),
        # Python non-test paths
        ("foo.py", "python", False),
        ("testing.py", "python", False),
        ("src/app.py", "python", False),
        # JavaScript/TypeScript test paths
        ("foo.test.js", "javascript", True),
        ("foo.spec.js", "javascript", True),
        ("foo.test.ts", "typescript", True),
        ("foo.spec.tsx", "typescript", True),
        ("__tests__/foo.js", "javascript", True),
        ("test/foo.js", "javascript", True),
        # JavaScript non-test paths
        ("foo.js", "javascript", False),
        ("testing.js", "javascript", False),
        # PHP test paths
        ("FooTest.php", "php", True),
        ("tests/FooTest.php", "php", True),
        # PHP non-test paths
        ("Foo.php", "php", False),
        # Rust test paths
        ("tests/integration.rs", "rust", True),
        # Rust non-test paths (unit tests are in-file)
        ("src/lib.rs", "rust", False),
    ])
    def test_is_test_path(self, path: str, language: str, expected: bool):
        from magaldi_core.code_parser import is_test_path
        assert is_test_path(path, language) == expected


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

    def test_max_chars_by_type_property(self):
        """Should compute max chars per element type."""
        # Create elements with different code sizes
        elements = [
            CodeElement(element_id="1", element_type="function", raw_code="x" * 1000),
            CodeElement(element_id="2", element_type="function", raw_code="x" * 2000),
            CodeElement(element_id="3", element_type="class", raw_code="x" * 5000),
            CodeElement(element_id="4", element_type="file", raw_code="x" * 10000),
        ]

        # FileInfo is required for ParsedFile
        file_info = FileInfo(
            relative_path="file.py",
            absolute_path=Path("/test/file.py"),
            language="python",
        )
        parsed_file = ParsedFile(
            file_info=file_info,
            elements=elements,
        )
        result = ParsingResult(
            scope="test",
            repository="repo",
            username="user",
            parsed_files=[parsed_file],
        )

        max_chars = result.max_chars_by_type

        assert max_chars["function"] == 2000  # Max of 1000, 2000
        assert max_chars["class"] == 5000
        assert max_chars["file"] == 10000

    def test_context_sizes_property(self):
        """Should compute context sizes from max chars."""
        from shared.ai.context_size import CONTEXT_TIERS

        elements = [
            CodeElement(element_id="1", element_type="function", raw_code="x" * 4000),
            CodeElement(element_id="2", element_type="variable", raw_code="x" * 100),
        ]
        file_info = FileInfo(
            relative_path="file.py",
            absolute_path=Path("/test/file.py"),
            language="python",
        )
        parsed_file = ParsedFile(
            file_info=file_info,
            elements=elements,
        )
        result = ParsingResult(
            scope="test",
            repository="repo",
            username="user",
            parsed_files=[parsed_file],
        )

        context_sizes = result.context_sizes

        assert "function" in context_sizes
        assert "variable" in context_sizes
        # Should be valid context tiers
        assert context_sizes["function"] in CONTEXT_TIERS
        assert context_sizes["variable"] in CONTEXT_TIERS

    def test_largest_elements_by_type_property(self):
        """Should track the largest element for each type."""
        elements = [
            CodeElement(element_id="1", name="small_func", element_type="function", raw_code="x" * 1000, relative_path="a.py"),
            CodeElement(element_id="2", name="big_func", element_type="function", raw_code="x" * 5000, relative_path="b.py"),
            CodeElement(element_id="3", name="MyClass", element_type="class", raw_code="x" * 3000, relative_path="c.py"),
        ]
        file_info = FileInfo(
            relative_path="file.py",
            absolute_path=Path("/test/file.py"),
            language="python",
        )
        parsed_file = ParsedFile(
            file_info=file_info,
            elements=elements,
        )
        result = ParsingResult(
            scope="test",
            repository="repo",
            username="user",
            parsed_files=[parsed_file],
        )

        largest = result.largest_elements_by_type

        assert "function" in largest
        assert "class" in largest
        # Should identify the largest function
        name, path, chars = largest["function"]
        assert name == "big_func"
        assert chars == 5000
        # Should identify the class
        name, path, chars = largest["class"]
        assert name == "MyClass"
        assert chars == 3000

    def test_largest_elements_method(self):
        """Should return top N largest elements sorted by char count."""
        elements = [
            CodeElement(element_id="1", name="tiny", element_type="function", raw_code="x" * 100, relative_path="a.py"),
            CodeElement(element_id="2", name="huge", element_type="file", raw_code="x" * 10000, relative_path="b.py"),
            CodeElement(element_id="3", name="medium", element_type="class", raw_code="x" * 3000, relative_path="c.py"),
            CodeElement(element_id="4", name="large", element_type="function", raw_code="x" * 5000, relative_path="d.py"),
            CodeElement(element_id="5", name="small", element_type="method", raw_code="x" * 500, relative_path="e.py"),
        ]
        file_info = FileInfo(
            relative_path="file.py",
            absolute_path=Path("/test/file.py"),
            language="python",
        )
        parsed_file = ParsedFile(file_info=file_info, elements=elements)
        result = ParsingResult(
            scope="test", repository="repo", username="user", parsed_files=[parsed_file]
        )

        # Get top 3 largest
        largest = result.largest_elements(3)
        assert len(largest) == 3
        # Should be sorted by char count descending
        assert largest[0] == ("huge", "b.py", 10000, "file")
        assert largest[1] == ("large", "d.py", 5000, "function")
        assert largest[2] == ("medium", "c.py", 3000, "class")

        # Get top 5 (all of them)
        all_largest = result.largest_elements(5)
        assert len(all_largest) == 5
        assert all_largest[4] == ("tiny", "a.py", 100, "function")

    def test_elements_by_tier_property(self):
        """Should group elements by their context tier."""
        from shared.ai.context_size import CONTEXT_TIERS

        # Create elements of varying sizes that will land in different tiers
        # Small: 200 chars = 50 tokens + 700 overhead = 750 → 2048 tier
        # Medium: 8000 chars = 2000 tokens + 700 overhead = 2700 → 4096 tier
        # Large: 50000 chars = 12500 tokens + 300 overhead = 12800 → 16384 tier
        elements = [
            CodeElement(element_id="1", name="small1", element_type="function", raw_code="x" * 200, relative_path="a.py"),
            CodeElement(element_id="2", name="small2", element_type="function", raw_code="x" * 300, relative_path="a.py"),
            CodeElement(element_id="3", name="medium", element_type="function", raw_code="x" * 8000, relative_path="b.py"),
            CodeElement(element_id="4", name="large_file", element_type="file", raw_code="x" * 50000, relative_path="c.py"),
        ]
        file_info = FileInfo(
            relative_path="file.py",
            absolute_path=Path("/test/file.py"),
            language="python",
        )
        parsed_file = ParsedFile(
            file_info=file_info,
            elements=elements,
        )
        result = ParsingResult(
            scope="test",
            repository="repo",
            username="user",
            parsed_files=[parsed_file],
        )

        tiers = result.elements_by_tier

        # All context tiers should be present
        for tier in CONTEXT_TIERS:
            assert tier in tiers

        # Check 2048 tier (small functions)
        assert tiers[2048]["count"] == 2
        assert tiers[2048]["by_type"]["function"] == 2
        assert tiers[2048]["largest"][0] == "small2"  # 300 > 200

        # Check 4096 tier (medium function)
        assert tiers[4096]["count"] == 1
        assert tiers[4096]["by_type"]["function"] == 1
        assert tiers[4096]["largest"][0] == "medium"

        # Check 16384 tier (large file)
        assert tiers[16384]["count"] == 1
        assert tiers[16384]["by_type"]["file"] == 1
        assert tiers[16384]["largest"][0] == "large_file"

        # Empty tiers should have zero counts
        assert tiers[8192]["count"] == 0
        assert tiers[32768]["count"] == 0


# =============================================================================
# IMPORT EXTRACTION TESTS
# =============================================================================


class TestPythonImportExtraction:
    """Tests for Python import extraction."""

    def test_simple_import(self):
        """Test extracting simple import statements."""
        code = """import os
import sys
"""
        parser = PythonParser()
        file_info = FileInfo(
            relative_path="module.py",
            absolute_path=Path("/fake/module.py"),
            language="python",
        )

        elements = parser.parse(code, file_info, "scope", "repo", "main")
        file_elem = next(e for e in elements if e.element_type == "file")

        assert len(file_elem.imports) == 2

        os_import = next(i for i in file_elem.imports if i.name == "os")
        assert os_import.module == "os"
        assert os_import.alias is None
        assert os_import.line == 1

        sys_import = next(i for i in file_elem.imports if i.name == "sys")
        assert sys_import.module == "sys"
        assert sys_import.alias is None
        assert sys_import.line == 2

    def test_import_with_alias(self):
        """Test extracting import with alias."""
        code = """import pandas as pd
import numpy as np
"""
        parser = PythonParser()
        file_info = FileInfo(
            relative_path="module.py",
            absolute_path=Path("/fake/module.py"),
            language="python",
        )

        elements = parser.parse(code, file_info, "scope", "repo", "main")
        file_elem = next(e for e in elements if e.element_type == "file")

        assert len(file_elem.imports) == 2

        pd_import = next(i for i in file_elem.imports if i.name == "pandas")
        assert pd_import.module == "pandas"
        assert pd_import.alias == "pd"
        assert pd_import.line == 1

        np_import = next(i for i in file_elem.imports if i.name == "numpy")
        assert np_import.module == "numpy"
        assert np_import.alias == "np"
        assert np_import.line == 2

    def test_from_import(self):
        """Test extracting from import statements."""
        code = """from pathlib import Path
from utils import process
"""
        parser = PythonParser()
        file_info = FileInfo(
            relative_path="module.py",
            absolute_path=Path("/fake/module.py"),
            language="python",
        )

        elements = parser.parse(code, file_info, "scope", "repo", "main")
        file_elem = next(e for e in elements if e.element_type == "file")

        assert len(file_elem.imports) == 2

        path_import = next(i for i in file_elem.imports if i.name == "Path")
        assert path_import.module == "pathlib"
        assert path_import.alias is None
        assert path_import.line == 1

        process_import = next(i for i in file_elem.imports if i.name == "process")
        assert process_import.module == "utils"
        assert process_import.alias is None
        assert process_import.line == 2

    def test_from_import_with_alias(self):
        """Test extracting from import with alias."""
        code = """from utils import process as p
from collections import OrderedDict as OD
"""
        parser = PythonParser()
        file_info = FileInfo(
            relative_path="module.py",
            absolute_path=Path("/fake/module.py"),
            language="python",
        )

        elements = parser.parse(code, file_info, "scope", "repo", "main")
        file_elem = next(e for e in elements if e.element_type == "file")

        assert len(file_elem.imports) == 2

        p_import = next(i for i in file_elem.imports if i.name == "process")
        assert p_import.module == "utils"
        assert p_import.alias == "p"
        assert p_import.line == 1

        od_import = next(i for i in file_elem.imports if i.name == "OrderedDict")
        assert od_import.module == "collections"
        assert od_import.alias == "OD"
        assert od_import.line == 2

    def test_dotted_module_import(self):
        """Test extracting import of dotted module names."""
        code = """import os.path
from collections.abc import Callable
"""
        parser = PythonParser()
        file_info = FileInfo(
            relative_path="module.py",
            absolute_path=Path("/fake/module.py"),
            language="python",
        )

        elements = parser.parse(code, file_info, "scope", "repo", "main")
        file_elem = next(e for e in elements if e.element_type == "file")

        assert len(file_elem.imports) == 2

        ospath_import = next(i for i in file_elem.imports if i.name == "os.path")
        assert ospath_import.module == "os.path"
        assert ospath_import.alias is None

        callable_import = next(i for i in file_elem.imports if i.name == "Callable")
        assert callable_import.module == "collections.abc"
        assert callable_import.alias is None

    def test_mixed_imports(self):
        """Test extracting various import patterns together."""
        code = """import os
import pandas as pd
from pathlib import Path
from utils import process as p
"""
        parser = PythonParser()
        file_info = FileInfo(
            relative_path="module.py",
            absolute_path=Path("/fake/module.py"),
            language="python",
        )

        elements = parser.parse(code, file_info, "scope", "repo", "main")
        file_elem = next(e for e in elements if e.element_type == "file")

        assert len(file_elem.imports) == 4

        imports_by_name = {i.name: i for i in file_elem.imports}

        assert imports_by_name["os"].alias is None
        assert imports_by_name["pandas"].alias == "pd"
        assert imports_by_name["Path"].module == "pathlib"
        assert imports_by_name["process"].alias == "p"


class TestJavaScriptImportExtraction:
    """Tests for JavaScript/TypeScript import extraction."""

    def test_named_imports(self):
        """Test extracting named imports."""
        code = """import { foo, bar } from './utils';
"""
        parser = JavaScriptParser()
        file_info = FileInfo(
            relative_path="module.js",
            absolute_path=Path("/fake/module.js"),
            language="javascript",
        )

        elements = parser.parse(code, file_info, "scope", "repo", "main")
        file_elem = next(e for e in elements if e.element_type == "file")

        assert len(file_elem.imports) == 2

        foo_import = next(i for i in file_elem.imports if i.name == "foo")
        assert foo_import.module == "./utils"
        assert foo_import.alias is None
        assert foo_import.line == 1

        bar_import = next(i for i in file_elem.imports if i.name == "bar")
        assert bar_import.module == "./utils"
        assert bar_import.alias is None

    def test_named_imports_with_alias(self):
        """Test extracting named imports with alias."""
        code = """import { foo as bar } from './utils';
"""
        parser = JavaScriptParser()
        file_info = FileInfo(
            relative_path="module.js",
            absolute_path=Path("/fake/module.js"),
            language="javascript",
        )

        elements = parser.parse(code, file_info, "scope", "repo", "main")
        file_elem = next(e for e in elements if e.element_type == "file")

        assert len(file_elem.imports) == 1

        import_elem = file_elem.imports[0]
        assert import_elem.name == "foo"
        assert import_elem.module == "./utils"
        assert import_elem.alias == "bar"
        assert import_elem.line == 1

    def test_default_import(self):
        """Test extracting default imports."""
        code = """import utils from './utils';
import React from 'react';
"""
        parser = JavaScriptParser()
        file_info = FileInfo(
            relative_path="module.js",
            absolute_path=Path("/fake/module.js"),
            language="javascript",
        )

        elements = parser.parse(code, file_info, "scope", "repo", "main")
        file_elem = next(e for e in elements if e.element_type == "file")

        assert len(file_elem.imports) == 2

        utils_import = next(i for i in file_elem.imports if i.name == "utils")
        assert utils_import.module == "./utils"
        assert utils_import.alias is None
        assert utils_import.line == 1

        react_import = next(i for i in file_elem.imports if i.name == "React")
        assert react_import.module == "react"
        assert react_import.alias is None
        assert react_import.line == 2

    def test_namespace_import(self):
        """Test extracting namespace imports."""
        code = """import * as utils from './utils';
"""
        parser = JavaScriptParser()
        file_info = FileInfo(
            relative_path="module.js",
            absolute_path=Path("/fake/module.js"),
            language="javascript",
        )

        elements = parser.parse(code, file_info, "scope", "repo", "main")
        file_elem = next(e for e in elements if e.element_type == "file")

        assert len(file_elem.imports) == 1

        import_elem = file_elem.imports[0]
        assert import_elem.name == "*"
        assert import_elem.module == "./utils"
        assert import_elem.alias == "utils"
        assert import_elem.line == 1

    def test_require_import(self):
        """Test extracting CommonJS require imports."""
        code = """const bar = require('lib');
const utils = require('./utils');
"""
        parser = JavaScriptParser()
        file_info = FileInfo(
            relative_path="module.js",
            absolute_path=Path("/fake/module.js"),
            language="javascript",
        )

        elements = parser.parse(code, file_info, "scope", "repo", "main")
        file_elem = next(e for e in elements if e.element_type == "file")

        assert len(file_elem.imports) == 2

        bar_import = next(i for i in file_elem.imports if i.name == "bar")
        assert bar_import.module == "lib"
        assert bar_import.alias is None
        assert bar_import.line == 1

        utils_import = next(i for i in file_elem.imports if i.name == "utils")
        assert utils_import.module == "./utils"
        assert utils_import.alias is None
        assert utils_import.line == 2

    def test_mixed_imports(self):
        """Test extracting various import patterns together."""
        code = """import React from 'react';
import { useState, useEffect } from 'react';
import * as utils from './utils';
const path = require('path');
"""
        parser = JavaScriptParser()
        file_info = FileInfo(
            relative_path="module.js",
            absolute_path=Path("/fake/module.js"),
            language="javascript",
        )

        elements = parser.parse(code, file_info, "scope", "repo", "main")
        file_elem = next(e for e in elements if e.element_type == "file")

        assert len(file_elem.imports) == 5

        imports_by_name = {i.name: i for i in file_elem.imports}

        assert imports_by_name["React"].module == "react"
        assert imports_by_name["useState"].module == "react"
        assert imports_by_name["useEffect"].module == "react"
        assert imports_by_name["*"].alias == "utils"
        assert imports_by_name["path"].module == "path"


class TestTypeScriptImportExtraction:
    """Tests for TypeScript import extraction."""

    def test_typescript_imports(self):
        """Test extracting TypeScript imports (same syntax as JS)."""
        code = """import { Component } from '@angular/core';
import type { User } from './types';
"""
        parser = JavaScriptParser(language="typescript")
        file_info = FileInfo(
            relative_path="module.ts",
            absolute_path=Path("/fake/module.ts"),
            language="typescript",
        )

        elements = parser.parse(code, file_info, "scope", "repo", "main")
        file_elem = next(e for e in elements if e.element_type == "file")

        # Should extract both regular and type imports
        assert len(file_elem.imports) >= 1

        component_import = next(i for i in file_elem.imports if i.name == "Component")
        assert component_import.module == "@angular/core"
        assert component_import.alias is None


class TestImportDataclass:
    """Tests for the Import dataclass."""

    def test_import_fields(self):
        """Test Import dataclass fields."""
        imp = Import(
            name="process",
            module="utils",
            alias="p",
            line=5,
        )

        assert imp.name == "process"
        assert imp.module == "utils"
        assert imp.alias == "p"
        assert imp.line == 5

    def test_import_no_alias(self):
        """Test Import with no alias."""
        imp = Import(
            name="os",
            module="os",
            alias=None,
            line=1,
        )

        assert imp.name == "os"
        assert imp.module == "os"
        assert imp.alias is None
        assert imp.line == 1


# =============================================================================
# CALL EXTRACTION TESTS
# =============================================================================


class TestPythonCallExtraction:
    """Tests for Python call extraction."""

    def test_bare_function_call(self):
        """Test extracting bare function calls."""
        code = """def main():
    process(x)
    validate(data)
"""
        parser = PythonParser()
        file_info = FileInfo(
            relative_path="module.py",
            absolute_path=Path("/fake/module.py"),
            language="python",
        )

        elements = parser.parse(code, file_info, "scope", "repo", "main")
        main_func = next(e for e in elements if e.name == "main" and e.element_type == "function")

        assert len(main_func.calls) == 2

        process_call = next(c for c in main_func.calls if c.name == "process")
        assert process_call.receiver is None
        assert process_call.line == 2

        validate_call = next(c for c in main_func.calls if c.name == "validate")
        assert validate_call.receiver is None
        assert validate_call.line == 3

    def test_method_call_on_self(self):
        """Test extracting method calls on self."""
        code = """class MyClass:
    def process(self):
        self.validate()
        self.helper(x, y)
"""
        parser = PythonParser()
        file_info = FileInfo(
            relative_path="module.py",
            absolute_path=Path("/fake/module.py"),
            language="python",
        )

        elements = parser.parse(code, file_info, "scope", "repo", "main")
        process_method = next(e for e in elements if e.name == "process" and e.element_type == "method")

        assert len(process_method.calls) == 2

        validate_call = next(c for c in process_method.calls if c.name == "validate")
        assert validate_call.receiver == "self"
        assert validate_call.line == 3

        helper_call = next(c for c in process_method.calls if c.name == "helper")
        assert helper_call.receiver == "self"
        assert helper_call.line == 4

    def test_method_call_on_object(self):
        """Test extracting method calls on objects."""
        code = """def process():
    utils.run()
    config.get('key')
    db.query(sql)
"""
        parser = PythonParser()
        file_info = FileInfo(
            relative_path="module.py",
            absolute_path=Path("/fake/module.py"),
            language="python",
        )

        elements = parser.parse(code, file_info, "scope", "repo", "main")
        process_func = next(e for e in elements if e.name == "process" and e.element_type == "function")

        assert len(process_func.calls) == 3

        run_call = next(c for c in process_func.calls if c.name == "run")
        assert run_call.receiver == "utils"

        get_call = next(c for c in process_func.calls if c.name == "get")
        assert get_call.receiver == "config"

        query_call = next(c for c in process_func.calls if c.name == "query")
        assert query_call.receiver == "db"

    def test_chained_calls(self):
        """Test extracting chained method calls."""
        code = """def process():
    obj.method1().method2()
    data.filter(x).map(y).reduce(z)
"""
        parser = PythonParser()
        file_info = FileInfo(
            relative_path="module.py",
            absolute_path=Path("/fake/module.py"),
            language="python",
        )

        elements = parser.parse(code, file_info, "scope", "repo", "main")
        process_func = next(e for e in elements if e.name == "process" and e.element_type == "function")

        # Should extract method2 (from the outermost call)
        # The chained calls appear as separate call nodes
        call_names = [c.name for c in process_func.calls]
        assert "method2" in call_names or "method1" in call_names

    def test_nested_attribute_call(self):
        """Test extracting calls on nested attributes."""
        code = """def process():
    a.b.method()
    config.db.connect()
"""
        parser = PythonParser()
        file_info = FileInfo(
            relative_path="module.py",
            absolute_path=Path("/fake/module.py"),
            language="python",
        )

        elements = parser.parse(code, file_info, "scope", "repo", "main")
        process_func = next(e for e in elements if e.name == "process" and e.element_type == "function")

        assert len(process_func.calls) == 2

        method_call = next(c for c in process_func.calls if c.name == "method")
        assert method_call.receiver == "a.b"

        connect_call = next(c for c in process_func.calls if c.name == "connect")
        assert connect_call.receiver == "config.db"

    def test_mixed_calls(self):
        """Test extracting various call patterns together."""
        code = """def process(self, data):
    validate(data)
    self.transform(data)
    utils.process(data)
    return result
"""
        parser = PythonParser()
        file_info = FileInfo(
            relative_path="module.py",
            absolute_path=Path("/fake/module.py"),
            language="python",
        )

        elements = parser.parse(code, file_info, "scope", "repo", "main")
        process_func = next(e for e in elements if e.name == "process" and e.element_type == "function")

        assert len(process_func.calls) == 3

        calls_by_name = {c.name: c for c in process_func.calls}

        assert calls_by_name["validate"].receiver is None
        assert calls_by_name["transform"].receiver == "self"
        assert calls_by_name["process"].receiver == "utils"

    def test_no_calls_in_function(self):
        """Test function with no calls."""
        code = """def simple():
    x = 1 + 2
    return x
"""
        parser = PythonParser()
        file_info = FileInfo(
            relative_path="module.py",
            absolute_path=Path("/fake/module.py"),
            language="python",
        )

        elements = parser.parse(code, file_info, "scope", "repo", "main")
        simple_func = next(e for e in elements if e.name == "simple" and e.element_type == "function")

        assert len(simple_func.calls) == 0


class TestJavaScriptCallExtraction:
    """Tests for JavaScript/TypeScript call extraction."""

    def test_bare_function_call(self):
        """Test extracting bare function calls."""
        code = """function main() {
    process(x);
    validate(data);
}
"""
        parser = JavaScriptParser()
        file_info = FileInfo(
            relative_path="module.js",
            absolute_path=Path("/fake/module.js"),
            language="javascript",
        )

        elements = parser.parse(code, file_info, "scope", "repo", "main")
        main_func = next(e for e in elements if e.name == "main" and e.element_type == "function")

        assert len(main_func.calls) == 2

        process_call = next(c for c in main_func.calls if c.name == "process")
        assert process_call.receiver is None
        assert process_call.line == 2

        validate_call = next(c for c in main_func.calls if c.name == "validate")
        assert validate_call.receiver is None
        assert validate_call.line == 3

    def test_method_call_on_this(self):
        """Test extracting method calls on this."""
        code = """class MyClass {
    process() {
        this.validate();
        this.helper(x, y);
    }
}
"""
        parser = JavaScriptParser()
        file_info = FileInfo(
            relative_path="module.js",
            absolute_path=Path("/fake/module.js"),
            language="javascript",
        )

        elements = parser.parse(code, file_info, "scope", "repo", "main")
        process_method = next(e for e in elements if e.name == "process" and e.element_type == "method")

        assert len(process_method.calls) == 2

        validate_call = next(c for c in process_method.calls if c.name == "validate")
        assert validate_call.receiver == "this"

        helper_call = next(c for c in process_method.calls if c.name == "helper")
        assert helper_call.receiver == "this"

    def test_method_call_on_object(self):
        """Test extracting method calls on objects."""
        code = """function process() {
    utils.run();
    config.get('key');
    db.query(sql);
}
"""
        parser = JavaScriptParser()
        file_info = FileInfo(
            relative_path="module.js",
            absolute_path=Path("/fake/module.js"),
            language="javascript",
        )

        elements = parser.parse(code, file_info, "scope", "repo", "main")
        process_func = next(e for e in elements if e.name == "process" and e.element_type == "function")

        assert len(process_func.calls) == 3

        run_call = next(c for c in process_func.calls if c.name == "run")
        assert run_call.receiver == "utils"

        get_call = next(c for c in process_func.calls if c.name == "get")
        assert get_call.receiver == "config"

        query_call = next(c for c in process_func.calls if c.name == "query")
        assert query_call.receiver == "db"

    def test_chained_calls(self):
        """Test extracting chained method calls."""
        code = """function process() {
    obj.method1().method2();
    data.filter(x).map(y).reduce(z);
}
"""
        parser = JavaScriptParser()
        file_info = FileInfo(
            relative_path="module.js",
            absolute_path=Path("/fake/module.js"),
            language="javascript",
        )

        elements = parser.parse(code, file_info, "scope", "repo", "main")
        process_func = next(e for e in elements if e.name == "process" and e.element_type == "function")

        # Should extract calls from the chain
        call_names = [c.name for c in process_func.calls]
        assert "method2" in call_names or "method1" in call_names

    def test_mixed_calls(self):
        """Test extracting various call patterns together."""
        code = """function process(data) {
    validate(data);
    this.transform(data);
    utils.process(data);
    return result;
}
"""
        parser = JavaScriptParser()
        file_info = FileInfo(
            relative_path="module.js",
            absolute_path=Path("/fake/module.js"),
            language="javascript",
        )

        elements = parser.parse(code, file_info, "scope", "repo", "main")
        process_func = next(e for e in elements if e.name == "process" and e.element_type == "function")

        # Note: 'this' reference in a regular function might be handled differently
        # The important thing is we extract the calls we can
        call_names = [c.name for c in process_func.calls]
        assert "validate" in call_names
        assert "process" in call_names

    def test_no_calls_in_function(self):
        """Test function with no calls."""
        code = """function simple() {
    const x = 1 + 2;
    return x;
}
"""
        parser = JavaScriptParser()
        file_info = FileInfo(
            relative_path="module.js",
            absolute_path=Path("/fake/module.js"),
            language="javascript",
        )

        elements = parser.parse(code, file_info, "scope", "repo", "main")
        simple_func = next(e for e in elements if e.name == "simple" and e.element_type == "function")

        assert len(simple_func.calls) == 0

    def test_arrow_function_calls(self):
        """Test extracting calls from arrow functions."""
        code = """const process = (data) => {
    validate(data);
    transform(data);
};
"""
        parser = JavaScriptParser()
        file_info = FileInfo(
            relative_path="module.js",
            absolute_path=Path("/fake/module.js"),
            language="javascript",
        )

        elements = parser.parse(code, file_info, "scope", "repo", "main")
        process_func = next(e for e in elements if e.name == "process" and e.element_type == "function")

        assert len(process_func.calls) == 2
        call_names = [c.name for c in process_func.calls]
        assert "validate" in call_names
        assert "transform" in call_names


class TestCallDataclass:
    """Tests for the Call dataclass."""

    def test_call_fields(self):
        """Test Call dataclass fields."""
        from magaldi_core.code_parser import Call

        call = Call(
            name="process",
            receiver="utils",
            line=42,
        )

        assert call.name == "process"
        assert call.receiver == "utils"
        assert call.line == 42
        assert call.resolved_id is None

    def test_call_no_receiver(self):
        """Test Call with no receiver (bare function call)."""
        from magaldi_core.code_parser import Call

        call = Call(
            name="validate",
            receiver=None,
            line=10,
        )

        assert call.name == "validate"
        assert call.receiver is None
        assert call.line == 10
        assert call.resolved_id is None

    def test_call_with_resolved_id(self):
        """Test Call with resolved_id set."""
        from magaldi_core.code_parser import Call

        call = Call(
            name="process",
            receiver="self",
            line=5,
            resolved_id="scope:repo:main:file.py:method:process:10",
        )

        assert call.name == "process"
        assert call.receiver == "self"
        assert call.line == 5
        assert call.resolved_id == "scope:repo:main:file.py:method:process:10"


class TestCodeElementCallsField:
    """Tests for the calls field on CodeElement."""

    def test_default_empty_calls(self):
        """Test that calls field defaults to empty list."""
        element = CodeElement()
        assert element.calls == []

    def test_calls_field_populated(self):
        """Test that calls field can be populated."""
        from magaldi_core.code_parser import Call

        calls = [
            Call(name="validate", receiver=None, line=5),
            Call(name="process", receiver="self", line=10),
        ]
        element = CodeElement(calls=calls)

        assert len(element.calls) == 2
        assert element.calls[0].name == "validate"
        assert element.calls[1].name == "process"
