"""Tests for the code parser module (Phase 3).

This module tests the core code_parser.py functionality:
- Element ID generation
- parse_file and parse_files functions
- Data classes (CodeElement, ParsedFile, ParsingResult)
- Test detection utilities (is_test_path, is_test_element)
- Import and Call dataclasses

Parser-specific tests are in tests/parsers/:
- test_python_parser.py - PythonParser tests
- test_javascript_parser.py - JavaScriptParser tests
"""

from pathlib import Path

import pytest

from magaldi_core.change_detection import ChangeManifest, FileInfo
from magaldi_core.code_parser import (
    Call,
    CodeElement,
    Import,
    ParsedFile,
    ParsingResult,
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
            byte_offset=420,
        )

        assert element_id == "backend:auth-service:main:src/auth.py:function:login:420"

    def test_different_users_different_ids(self):
        id1 = generate_element_id("scope", "repo", "main", "file.py", "function", "foo", 1)
        id2 = generate_element_id("scope", "repo", "alice", "file.py", "function", "foo", 1)

        assert id1 != id2

    def test_different_lines_different_ids(self):
        id1 = generate_element_id("scope", "repo", "main", "file.py", "function", "foo", 1)
        id2 = generate_element_id("scope", "repo", "main", "file.py", "function", "foo", 2)

        assert id1 != id2


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
# TEST DETECTION
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
        # Small: 200 chars = 50 tokens + 1100 overhead = 1150 -> 2048 tier
        # Medium: 8000 chars = 2000 tokens + 1100 overhead = 3100 -> 4096 tier
        # Large: 50000 chars = 12500 tokens + 950 overhead = 13450 -> 16384 tier
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
        assert tiers[1024]["count"] == 0
        assert tiers[8192]["count"] == 0
        assert tiers[32768]["count"] == 0


# =============================================================================
# IMPORT AND CALL DATACLASSES
# =============================================================================


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


class TestCallDataclass:
    """Tests for the Call dataclass."""

    def test_call_fields(self):
        """Test Call dataclass fields."""
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
        calls = [
            Call(name="validate", receiver=None, line=5),
            Call(name="process", receiver="self", line=10),
        ]
        element = CodeElement(calls=calls)

        assert len(element.calls) == 2
        assert element.calls[0].name == "validate"
        assert element.calls[1].name == "process"
