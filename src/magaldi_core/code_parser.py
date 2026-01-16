"""Phase 3: Parsing - Extract code elements using Tree-sitter.

This module parses source files and extracts structured code elements:
- Classes
- Functions/Methods
- Module-level variables and constants

Each element includes position, content, and hierarchy information.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from magaldi_core.change_detection import ChangeManifest, FileInfo
from magaldi_core.tree_sitter_manager import (
    ExtractedElement,
    extract_javascript_class_members,
    extract_javascript_elements,
    extract_python_class_members,
    extract_python_elements,
    get_manager,
)


class ParsingError(Exception):
    """Raised when parsing fails."""

    pass


# =============================================================================
# TEST PATH DETECTION
# =============================================================================

# Test path patterns by language
TEST_PATH_PATTERNS: dict[str, list[str]] = {
    "python": [
        r"(^|/)test_[^/]+\.py$",      # test_*.py
        r"(^|/)[^/]+_test\.py$",       # *_test.py
        r"(^|/)tests/",                # tests/ directory
        r"(^|/)conftest\.py$",         # conftest.py
    ],
    "javascript": [
        r"\.(test|spec)\.[jt]sx?$",    # *.test.js, *.spec.ts, etc.
        r"(^|/)__tests__/",            # __tests__/ directory
        r"(^|/)test/",                 # test/ directory
    ],
    "typescript": [
        r"\.(test|spec)\.[jt]sx?$",    # *.test.ts, *.spec.tsx, etc.
        r"(^|/)__tests__/",            # __tests__/ directory
        r"(^|/)test/",                 # test/ directory
    ],
    "php": [
        r"Test\.php$",                 # *Test.php
        r"(^|/)tests/",                # tests/ directory
    ],
    "rust": [
        r"(^|/)tests/",                # tests/ directory (integration tests)
    ],
}


def is_test_path(relative_path: str, language: str) -> bool:
    """Check if a file path indicates test code.

    Args:
        relative_path: File path relative to repository root.
        language: Programming language of the file.

    Returns:
        True if the path matches test file patterns.
    """
    patterns = TEST_PATH_PATTERNS.get(language, [])
    for pattern in patterns:
        if re.search(pattern, relative_path):
            return True
    return False


def is_test_element(name: str, decorators: list[str], language: str) -> bool:
    """Check if an element is test code based on name/decorators.

    Args:
        name: Element name (function/method/class name).
        decorators: List of decorator/attribute names.
        language: Programming language.

    Returns:
        True if the element appears to be test code.
    """
    # Python: test_ prefix or pytest/unittest decorators
    if language == "python":
        if name.startswith("test_"):
            return True
        test_decorators = {"pytest", "unittest", "pytest.mark", "pytest.fixture"}
        for dec in decorators:
            for test_dec in test_decorators:
                if dec.startswith(test_dec):
                    return True
        return False

    # Rust: #[test] or #[cfg(test)] attributes
    if language == "rust":
        test_attrs = {"test", "cfg(test)"}
        return any(dec in test_attrs for dec in decorators)

    # JavaScript/TypeScript: detected via call patterns (describe/it/test)
    # These are handled separately during parsing
    # PHP: @test annotation or Test suffix handled via path/class name
    return False


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class CodeElement:
    """A parsed code element (class, function, method, variable)."""

    # Identity
    element_id: str = ""

    # Location
    scope: str = ""
    repository: str = ""
    username: str = ""
    relative_path: str = ""

    # Element info
    element_type: str = ""  # 'file', 'class', 'function', 'method', 'constant', 'variable'
    name: str = ""
    language: str = ""

    # Position (1-indexed lines)
    line_start: int = 0
    line_end: int = 0

    # Content
    raw_code: str = ""
    signature: str | None = None
    docstring: str | None = None

    # Metadata
    decorators: list[str] = field(default_factory=list)
    is_async: bool = False
    is_test: bool = False  # Whether this element is test code
    visibility: str = "public"  # 'public', 'private', 'protected'

    # Hierarchy
    level: int = 0  # 0=file, 1=class, 2=function/method, 3=variable
    parent_id: str | None = None

    # Type info
    return_type: str | None = None
    parameters: list[dict[str, Any]] = field(default_factory=list)

    # Context (for variables - how they're used)
    context_usages: list[str] = field(default_factory=list)

    # Content hash for change detection (computed from raw_code)
    content_hash: str | None = None

    def compute_content_hash(self) -> str:
        """Compute SHA256 hash of the element's content for change detection.

        The hash is based on raw_code which contains the actual source code.
        This allows detecting when an element's implementation changes,
        even if the file around it changed but this element didn't.
        """
        content = self.raw_code or ""
        self.content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        return self.content_hash


@dataclass
class ParsedFile:
    """Result of parsing a single file."""

    file_info: FileInfo
    elements: list[CodeElement] = field(default_factory=list)
    parse_errors: list[str] = field(default_factory=list)
    line_count: int = 0


@dataclass
class ParsingResult:
    """Result of parsing all files in a manifest."""

    scope: str
    repository: str
    username: str

    parsed_files: list[ParsedFile] = field(default_factory=list)
    failed_files: list[tuple[FileInfo, str]] = field(default_factory=list)

    @property
    def total_elements(self) -> int:
        return sum(len(pf.elements) for pf in self.parsed_files)

    @property
    def elements_by_type(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for pf in self.parsed_files:
            for elem in pf.elements:
                counts[elem.element_type] = counts.get(elem.element_type, 0) + 1
        return counts


# =============================================================================
# ELEMENT ID GENERATION
# =============================================================================


def generate_element_id(
    scope: str,
    repository: str,
    username: str,
    relative_path: str,
    element_type: str,
    name: str,
    line_start: int,
) -> str:
    """Generate unique element ID.

    Format: {scope}:{repository}:{username}:{relative_path}:{type}:{name}:{line}
    """
    return ":".join([
        scope,
        repository,
        username,
        relative_path,
        element_type,
        name,
        str(line_start),
    ])


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _find_variable_usages(
    name: str, lines: list[str], declaration_line: int, max_usages: int = 5
) -> list[str]:
    """Find lines where a variable is used (excluding declaration).

    Args:
        name: Variable name to search for.
        lines: All lines in the file.
        declaration_line: Line number of the declaration (1-indexed).
        max_usages: Maximum number of usages to return.

    Returns:
        List of "line N: <code>" strings showing usages.
    """
    usages = []
    # Pattern to match the variable name as a whole word
    pattern = re.compile(rf"\b{re.escape(name)}\b")

    for i, line in enumerate(lines):
        line_num = i + 1
        if line_num == declaration_line:
            continue  # Skip the declaration itself

        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue  # Skip empty lines and comments

        if pattern.search(line):
            # Truncate long lines
            display_line = stripped[:80] + "..." if len(stripped) > 80 else stripped
            usages.append(f"line {line_num}: {display_line}")
            if len(usages) >= max_usages:
                break

    return usages


def _extract_docstring(lines: list[str], block_start: int) -> str | None:
    """Extract docstring from the start of a block."""
    if block_start >= len(lines):
        return None

    for i in range(block_start, min(block_start + 3, len(lines))):
        line = lines[i].strip()
        if line.startswith('"""') or line.startswith("'''"):
            quote = line[:3]
            if line.count(quote) >= 2:
                # Single line docstring
                return line[3:-3].strip()
            else:
                # Multi-line docstring
                docstring_lines = [line[3:]]
                for j in range(i + 1, len(lines)):
                    end_line = lines[j]
                    if quote in end_line:
                        docstring_lines.append(end_line[: end_line.index(quote)])
                        break
                    docstring_lines.append(end_line)
                return "\n".join(docstring_lines).strip()

    return None


def _determine_visibility(name: str) -> str:
    """Determine visibility from name convention."""
    if name.startswith("__") and not name.endswith("__"):
        return "private"
    elif name.startswith("_"):
        return "protected"
    return "public"


# =============================================================================
# TREE-SITTER BASED PARSERS
# =============================================================================


class TreeSitterParser:
    """Base class for tree-sitter based parsing."""

    def __init__(self, language: str):
        self.language = language
        self.manager = get_manager()

    def parse(
        self,
        content: str,
        file_info: FileInfo,
        scope: str,
        repository: str,
        username: str,
    ) -> list[CodeElement]:
        """Parse content and extract code elements."""
        raise NotImplementedError


class PythonParser(TreeSitterParser):
    """Parse Python files using tree-sitter."""

    def __init__(self) -> None:
        super().__init__("python")

    def parse(
        self,
        content: str,
        file_info: FileInfo,
        scope: str,
        repository: str,
        username: str,
    ) -> list[CodeElement]:
        """Parse Python content and extract elements."""
        elements: list[CodeElement] = []
        lines = content.split("\n")
        line_count = len(lines)

        # Create file-level element
        file_element = CodeElement(
            scope=scope,
            repository=repository,
            username=username,
            relative_path=file_info.relative_path,
            element_type="file",
            name=Path(file_info.relative_path).name,
            language="python",
            line_start=1,
            line_end=line_count,
            level=0,
            raw_code="",  # Don't store full file content
        )
        file_element.element_id = generate_element_id(
            scope, repository, username, file_info.relative_path, "file", file_element.name, 1
        )
        elements.append(file_element)

        # Parse with tree-sitter
        tree = self.manager.parse(content.encode("utf-8"), "python")
        extracted = extract_python_elements(tree, lines)

        # Convert ExtractedElements to CodeElements
        for ext in extracted:
            if ext.element_type == "class":
                class_elem = self._convert_class(ext, file_info, scope, repository, username, lines)
                elements.append(class_elem)

                # Extract class members
                if ext.node:
                    methods, class_vars = extract_python_class_members(ext.node, lines)

                    for method_ext in methods:
                        method_elem = self._convert_method(
                            method_ext, class_elem, file_info, scope, repository, username, lines
                        )
                        elements.append(method_elem)

                    for var_ext in class_vars:
                        var_elem = self._convert_variable(
                            var_ext, file_info, scope, repository, username, lines, parent=class_elem
                        )
                        elements.append(var_elem)

            elif ext.element_type == "function":
                func_elem = self._convert_function(ext, file_info, scope, repository, username, lines)
                elements.append(func_elem)

            elif ext.element_type in ("constant", "variable"):
                var_elem = self._convert_variable(ext, file_info, scope, repository, username, lines)
                elements.append(var_elem)

        # Set parent IDs for elements without explicit parents
        self._set_hierarchy(elements, file_element)

        return elements

    def _convert_class(
        self,
        ext: ExtractedElement,
        file_info: FileInfo,
        scope: str,
        repository: str,
        username: str,
        lines: list[str],
    ) -> CodeElement:
        """Convert extracted class to CodeElement."""
        docstring = _extract_docstring(lines, ext.line_start)

        elem = CodeElement(
            scope=scope,
            repository=repository,
            username=username,
            relative_path=file_info.relative_path,
            element_type="class",
            name=ext.name,
            language="python",
            line_start=ext.line_start,
            line_end=ext.line_end,
            raw_code=ext.raw_code,
            docstring=docstring,
            decorators=ext.decorators or [],
            level=1,
            visibility=_determine_visibility(ext.name),
        )
        elem.element_id = generate_element_id(
            scope, repository, username, file_info.relative_path, "class", ext.name, ext.line_start
        )
        return elem

    def _convert_function(
        self,
        ext: ExtractedElement,
        file_info: FileInfo,
        scope: str,
        repository: str,
        username: str,
        lines: list[str],
    ) -> CodeElement:
        """Convert extracted function to CodeElement."""
        docstring = _extract_docstring(lines, ext.line_start)

        elem = CodeElement(
            scope=scope,
            repository=repository,
            username=username,
            relative_path=file_info.relative_path,
            element_type="function",
            name=ext.name,
            language="python",
            line_start=ext.line_start,
            line_end=ext.line_end,
            raw_code=ext.raw_code,
            signature=ext.signature,
            docstring=docstring,
            decorators=ext.decorators or [],
            is_async=ext.is_async,
            visibility=_determine_visibility(ext.name),
            level=2,
        )
        elem.element_id = generate_element_id(
            scope, repository, username, file_info.relative_path, "function", ext.name, ext.line_start
        )
        return elem

    def _convert_method(
        self,
        ext: ExtractedElement,
        parent_class: CodeElement,
        file_info: FileInfo,
        scope: str,
        repository: str,
        username: str,
        lines: list[str],
    ) -> CodeElement:
        """Convert extracted method to CodeElement."""
        docstring = _extract_docstring(lines, ext.line_start)

        elem = CodeElement(
            scope=scope,
            repository=repository,
            username=username,
            relative_path=file_info.relative_path,
            element_type="method",
            name=ext.name,
            language="python",
            line_start=ext.line_start,
            line_end=ext.line_end,
            raw_code=ext.raw_code,
            signature=ext.signature,
            docstring=docstring,
            decorators=ext.decorators or [],
            is_async=ext.is_async,
            visibility=_determine_visibility(ext.name),
            level=2,
            parent_id=parent_class.element_id,
        )
        elem.element_id = generate_element_id(
            scope, repository, username, file_info.relative_path, "method", ext.name, ext.line_start
        )
        return elem

    def _convert_variable(
        self,
        ext: ExtractedElement,
        file_info: FileInfo,
        scope: str,
        repository: str,
        username: str,
        lines: list[str],
        parent: CodeElement | None = None,
    ) -> CodeElement:
        """Convert extracted variable/constant to CodeElement."""
        # Find usages
        usages = _find_variable_usages(ext.name, lines, ext.line_start)

        elem = CodeElement(
            scope=scope,
            repository=repository,
            username=username,
            relative_path=file_info.relative_path,
            element_type=ext.element_type,  # 'constant' or 'variable'
            name=ext.name,
            language="python",
            line_start=ext.line_start,
            line_end=ext.line_end,
            raw_code=ext.raw_code,
            level=3,
            visibility=_determine_visibility(ext.name),
            context_usages=usages,
            parent_id=parent.element_id if parent else None,
        )
        elem.element_id = generate_element_id(
            scope, repository, username, file_info.relative_path, ext.element_type, ext.name, ext.line_start
        )
        return elem

    def _set_hierarchy(self, elements: list[CodeElement], file_element: CodeElement) -> None:
        """Set parent IDs for elements without explicit parents."""
        for elem in elements:
            if elem.element_type == "file":
                continue
            if not elem.parent_id:
                elem.parent_id = file_element.element_id


class JavaScriptParser(TreeSitterParser):
    """Parse JavaScript/TypeScript files using tree-sitter."""

    def __init__(self, language: str = "javascript") -> None:
        super().__init__(language)

    def parse(
        self,
        content: str,
        file_info: FileInfo,
        scope: str,
        repository: str,
        username: str,
    ) -> list[CodeElement]:
        """Parse JavaScript content and extract elements."""
        elements: list[CodeElement] = []
        lines = content.split("\n")
        line_count = len(lines)

        # Create file-level element
        file_element = CodeElement(
            scope=scope,
            repository=repository,
            username=username,
            relative_path=file_info.relative_path,
            element_type="file",
            name=Path(file_info.relative_path).name,
            language=file_info.language,
            line_start=1,
            line_end=line_count,
            level=0,
        )
        file_element.element_id = generate_element_id(
            scope, repository, username, file_info.relative_path, "file", file_element.name, 1
        )
        elements.append(file_element)

        # Parse with tree-sitter
        tree = self.manager.parse(content.encode("utf-8"), self.language)
        extracted = extract_javascript_elements(tree, lines)

        # Convert ExtractedElements to CodeElements
        for ext in extracted:
            if ext.element_type == "class":
                class_elem = self._convert_class(ext, file_info, scope, repository, username, lines)
                elements.append(class_elem)

                # Extract class members
                if ext.node:
                    methods, fields = extract_javascript_class_members(ext.node, lines)

                    for method_ext in methods:
                        method_elem = self._convert_method(
                            method_ext, class_elem, file_info, scope, repository, username, lines
                        )
                        elements.append(method_elem)

                    for field_ext in fields:
                        field_elem = self._convert_variable(
                            field_ext, file_info, scope, repository, username, lines, parent=class_elem
                        )
                        elements.append(field_elem)

            elif ext.element_type == "function":
                func_elem = self._convert_function(ext, file_info, scope, repository, username, lines)
                elements.append(func_elem)

        # Set parent IDs
        self._set_hierarchy(elements, file_element)

        return elements

    def _convert_class(
        self,
        ext: ExtractedElement,
        file_info: FileInfo,
        scope: str,
        repository: str,
        username: str,
        lines: list[str],
    ) -> CodeElement:
        """Convert extracted class to CodeElement."""
        elem = CodeElement(
            scope=scope,
            repository=repository,
            username=username,
            relative_path=file_info.relative_path,
            element_type="class",
            name=ext.name,
            language=file_info.language,
            line_start=ext.line_start,
            line_end=ext.line_end,
            raw_code=ext.raw_code,
            level=1,
        )
        elem.element_id = generate_element_id(
            scope, repository, username, file_info.relative_path, "class", ext.name, ext.line_start
        )
        return elem

    def _convert_function(
        self,
        ext: ExtractedElement,
        file_info: FileInfo,
        scope: str,
        repository: str,
        username: str,
        lines: list[str],
    ) -> CodeElement:
        """Convert extracted function to CodeElement."""
        elem = CodeElement(
            scope=scope,
            repository=repository,
            username=username,
            relative_path=file_info.relative_path,
            element_type="function",
            name=ext.name,
            language=file_info.language,
            line_start=ext.line_start,
            line_end=ext.line_end,
            raw_code=ext.raw_code,
            signature=ext.signature,
            is_async=ext.is_async,
            level=2,
        )
        elem.element_id = generate_element_id(
            scope, repository, username, file_info.relative_path, "function", ext.name, ext.line_start
        )
        return elem

    def _convert_method(
        self,
        ext: ExtractedElement,
        parent_class: CodeElement,
        file_info: FileInfo,
        scope: str,
        repository: str,
        username: str,
        lines: list[str],
    ) -> CodeElement:
        """Convert extracted method to CodeElement."""
        elem = CodeElement(
            scope=scope,
            repository=repository,
            username=username,
            relative_path=file_info.relative_path,
            element_type="method",
            name=ext.name,
            language=file_info.language,
            line_start=ext.line_start,
            line_end=ext.line_end,
            raw_code=ext.raw_code,
            signature=ext.signature,
            is_async=ext.is_async,
            level=2,
            parent_id=parent_class.element_id,
        )
        elem.element_id = generate_element_id(
            scope, repository, username, file_info.relative_path, "method", ext.name, ext.line_start
        )
        return elem

    def _convert_variable(
        self,
        ext: ExtractedElement,
        file_info: FileInfo,
        scope: str,
        repository: str,
        username: str,
        lines: list[str],
        parent: CodeElement | None = None,
    ) -> CodeElement:
        """Convert extracted variable/field to CodeElement."""
        elem = CodeElement(
            scope=scope,
            repository=repository,
            username=username,
            relative_path=file_info.relative_path,
            element_type="variable",
            name=ext.name,
            language=file_info.language,
            line_start=ext.line_start,
            line_end=ext.line_end,
            raw_code=ext.raw_code,
            level=3,
            parent_id=parent.element_id if parent else None,
        )
        elem.element_id = generate_element_id(
            scope, repository, username, file_info.relative_path, "variable", ext.name, ext.line_start
        )
        return elem

    def _set_hierarchy(self, elements: list[CodeElement], file_element: CodeElement) -> None:
        """Set parent IDs for elements without explicit parents."""
        for elem in elements:
            if elem.element_type == "file":
                continue
            if not elem.parent_id:
                elem.parent_id = file_element.element_id


# =============================================================================
# PARSER REGISTRY
# =============================================================================


PARSERS: dict[str, TreeSitterParser] = {
    "python": PythonParser(),
    "javascript": JavaScriptParser("javascript"),
    "typescript": JavaScriptParser("typescript"),
}


def get_parser(language: str) -> TreeSitterParser | None:
    """Get parser for language."""
    return PARSERS.get(language)


# =============================================================================
# MAIN PARSE FUNCTION
# =============================================================================


def parse_file(
    file_info: FileInfo,
    scope: str,
    repository: str,
    username: str,
) -> ParsedFile:
    """Parse a single file and extract code elements.

    Args:
        file_info: File information from change detection.
        scope: Repository scope.
        repository: Repository name.
        username: Username (branch).

    Returns:
        ParsedFile with extracted elements.
    """
    result = ParsedFile(file_info=file_info)

    try:
        # Read file content
        content = file_info.absolute_path.read_text(encoding="utf-8", errors="replace")
        result.line_count = content.count("\n") + 1

        # Get parser for language
        parser = get_parser(file_info.language)
        if parser is None:
            result.parse_errors.append(f"No parser for language: {file_info.language}")
            return result

        # Parse and extract elements
        result.elements = parser.parse(content, file_info, scope, repository, username)

        # Detect if this is a test file
        file_is_test = is_test_path(file_info.relative_path, file_info.language)

        # Apply is_test to all elements
        for elem in result.elements:
            if file_is_test:
                # All elements in test files are test code
                elem.is_test = True
            else:
                # Check individual elements for test markers
                elem.is_test = is_test_element(elem.name, elem.decorators, file_info.language)

        # Compute content hash for each element (for change detection)
        for elem in result.elements:
            elem.compute_content_hash()

    except Exception as e:
        result.parse_errors.append(str(e))

    return result


def parse_files(
    manifest: ChangeManifest,
    on_progress: Callable[[int, int], None] | None = None,
) -> ParsingResult:
    """Parse all files in a change manifest.

    Args:
        manifest: Change manifest from Phase 2.
        on_progress: Optional callback(completed, total) for progress updates.

    Returns:
        ParsingResult with all parsed files and elements.
    """
    result = ParsingResult(
        scope=manifest.scope,
        repository=manifest.repository,
        username=manifest.username,
    )

    # Parse new and modified files
    files_to_parse = manifest.new_files + manifest.modified_files
    total = len(files_to_parse)

    for i, file_info in enumerate(files_to_parse):
        try:
            parsed = parse_file(
                file_info,
                manifest.scope,
                manifest.repository,
                manifest.username,
            )
            result.parsed_files.append(parsed)
        except Exception as e:
            result.failed_files.append((file_info, str(e)))

        if on_progress:
            on_progress(i + 1, total)

    return result
