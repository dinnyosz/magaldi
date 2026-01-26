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
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from magaldi_core.change_detection import ChangeManifest, FileInfo
from magaldi_core.tree_sitter_manager import (
    DecoratorInfo,
    ExtractedCall,
    ExtractedElement,
    ExtractedReference,
    analyze_purity,
    associate_comments,
    detect_cli_commands,
    detect_http_routes,
    detect_patterns,
    detect_public_api,
    extract_comments,
    extract_javascript_base_class,
    extract_javascript_calls,
    extract_javascript_class_fields,
    extract_javascript_class_members,
    extract_javascript_elements,
    extract_javascript_imports,
    extract_javascript_modified_properties,
    extract_javascript_references,
    extract_javascript_thrown_exceptions,
    extract_php_base_class,
    extract_php_calls,
    extract_php_class_members,
    extract_php_class_properties,
    extract_php_elements,
    extract_php_imports,
    extract_php_modified_properties,
    extract_php_thrown_exceptions,
    extract_python_base_classes,
    extract_python_calls,
    extract_python_class_attributes,
    extract_python_class_members,
    extract_python_elements,
    extract_python_imports,
    extract_python_modified_attributes,
    extract_python_raised_exceptions,
    extract_python_references,
    extract_rust_calls,
    extract_rust_elements,
    extract_rust_impl_members,
    extract_rust_impl_traits,
    extract_rust_imports,
    extract_rust_modified_fields,
    extract_rust_panics,
    extract_rust_struct_fields,
    extract_section_markers,
    extract_side_effects,
    # Extended code intelligence extractors
    extract_todos,
    extract_type_annotations,
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
class Import:
    """Represents an import statement."""

    name: str  # Imported name (e.g., "os", "process")
    module: str  # Source module (e.g., "os", "utils", "./utils")
    alias: str | None  # Alias if any (e.g., "pd" for "import pandas as pd")
    line: int  # Line number


@dataclass
class Call:
    """Represents a function call within an element."""

    name: str  # Function name (e.g., "process", "validate")
    receiver: str | None  # Receiver object (e.g., "self", "utils", None for bare calls)
    line: int  # Line number
    resolved_id: str | None = None  # Filled in resolution phase (not at parse time)


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
    decorator_details: list[dict[str, Any]] | None = None  # Rich decorator info: {name, args, full}
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

    # Imports (only populated on file elements)
    imports: list[Import] = field(default_factory=list)

    # Calls (only populated on function/method elements)
    calls: list[Call] = field(default_factory=list)

    # Enhanced context for summarization (populated during parsing)
    # For classes: instance attributes from __init__
    class_attributes: list[dict[str, Any]] | None = None
    # For classes: base class names
    base_classes: list[str] | None = None
    # For functions/methods: exception types raised
    exceptions_raised: list[str] | None = None
    # For methods: attributes modified (self.X = ...)
    attributes_modified: list[str] | None = None

    # Content hash for change detection (computed from raw_code)
    content_hash: str | None = None

    # === EXTENDED CODE INTELLIGENCE FIELDS ===

    # Type Flow
    type_annotations: list[dict[str, Any]] = field(default_factory=list)

    # Pattern Detection
    detected_patterns: list[str] = field(default_factory=list)  # ["singleton", "factory"]
    pattern_confidence: dict[str, float] = field(default_factory=dict)  # {"singleton": 0.95}

    # Documentation
    todos: list[dict[str, Any]] = field(default_factory=list)
    section_markers: list[dict[str, Any]] = field(default_factory=list)
    associated_comments: list[dict[str, Any]] = field(default_factory=list)

    # API Surface
    is_public_api: bool = False
    http_routes: list[dict[str, Any]] = field(default_factory=list)
    cli_commands: list[dict[str, Any]] = field(default_factory=list)

    # Purity/Mutation
    purity: dict[str, Any] | None = None
    side_effects: list[dict[str, Any]] = field(default_factory=list)
    mutated_state: list[str] = field(default_factory=list)

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
    references: list[ExtractedReference] = field(default_factory=list)  # Cross-file refs
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

    @property
    def max_chars_by_type(self) -> dict[str, int]:
        """Get max character count per element type.

        Used for computing optimal context sizes for KV cache optimization.
        """
        max_chars: dict[str, int] = {}
        for pf in self.parsed_files:
            for elem in pf.elements:
                code_len = len(elem.raw_code or "")
                current_max = max_chars.get(elem.element_type, 0)
                max_chars[elem.element_type] = max(current_max, code_len)
        return max_chars

    @property
    def context_sizes(self) -> dict[str, int]:
        """Get computed context sizes per element type.

        Returns optimal num_ctx values for each element type based on
        observed maximum code sizes. Used for KV cache optimization
        during summarization.
        """
        from shared.ai.context_size import compute_context_sizes

        return compute_context_sizes(self.max_chars_by_type)  # type: ignore[no-any-return]


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


def _get_line_indent(line: str) -> int:
    """Get the indentation level of a line (number of leading spaces)."""
    return len(line) - len(line.lstrip())


def _extract_control_block(lines: list[str], start_idx: int, max_lines: int = 6) -> list[str]:
    """Extract a control flow block starting at the given index.

    Captures the control statement and its indented body up to max_lines.

    Args:
        lines: All lines in the file.
        start_idx: Index of the control flow statement (0-indexed).
        max_lines: Maximum lines to capture including the control statement.

    Returns:
        List of lines forming the block.
    """
    if start_idx >= len(lines):
        return []

    block = [lines[start_idx].rstrip()]
    base_indent = _get_line_indent(lines[start_idx])

    for i in range(start_idx + 1, min(start_idx + max_lines, len(lines))):
        line = lines[i]
        stripped = line.strip()

        # Stop at empty lines or lines with same/less indentation (block ended)
        if stripped and _get_line_indent(line) <= base_indent:
            break

        # Include the line (even if empty, for readability)
        block.append(line.rstrip())

    return block


# Control flow keywords that introduce blocks we want to capture
_CONTROL_FLOW_PATTERN = re.compile(
    r"^\s*(if|elif|else|for|while|with|try|except|finally|match|case)\b"
)


def _find_variable_usages(
    name: str, lines: list[str], declaration_line: int, max_usages: int = 3
) -> list[str]:
    """Find lines where a variable is used with contextual block information.

    For control flow statements (if, for, while, etc.), captures the block
    to show what the variable controls. For other usages, shows the line
    with minimal context.

    Args:
        name: Variable name to search for.
        lines: All lines in the file.
        declaration_line: Line number of the declaration (1-indexed).
        max_usages: Maximum number of usages to return (default 3 for richer context).

    Returns:
        List of usage descriptions with contextual code blocks.
    """
    usages = []
    # Pattern to match the variable name as a whole word
    pattern = re.compile(rf"\b{re.escape(name)}\b")
    # Track which lines we've already included to avoid duplicates
    seen_lines: set[int] = set()

    for i, line in enumerate(lines):
        line_num = i + 1
        if line_num == declaration_line:
            continue  # Skip the declaration itself
        if line_num in seen_lines:
            continue

        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue  # Skip empty lines and comments

        if pattern.search(line):
            # Check if this is a control flow statement
            if _CONTROL_FLOW_PATTERN.match(line):
                # Extract the block this variable controls
                block_lines = _extract_control_block(lines, i, max_lines=6)
                # Mark all block lines as seen
                for j in range(len(block_lines)):
                    seen_lines.add(line_num + j)

                # Format as a code block
                block_code = "\n".join(block_lines)
                # Truncate if too long
                if len(block_code) > 300:
                    block_code = block_code[:300] + "\n..."
                usages.append(f"line {line_num} (controls block):\n{block_code}")
            else:
                # Regular usage - show the line with 1 line of context after if available
                seen_lines.add(line_num)
                context_lines = [stripped]

                # Add one line of context after if it's meaningful
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    if next_line and not next_line.startswith("#"):
                        context_lines.append(next_line)
                        seen_lines.add(line_num + 1)

                display = "\n    ".join(context_lines)
                if len(display) > 150:
                    display = display[:150] + "..."
                usages.append(f"line {line_num}: {display}")

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

        # Extract imports and populate on file element
        extracted_imports = extract_python_imports(tree, lines)
        file_element.imports = [
            Import(
                name=imp.name,
                module=imp.module,
                alias=imp.alias,
                line=imp.line,
            )
            for imp in extracted_imports
        ]

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

        # === EXTENDED CODE INTELLIGENCE EXTRACTION ===

        # Extract file-level documentation
        todos = extract_todos(content)
        section_markers = extract_section_markers(content)
        all_comments = extract_comments(content)

        # Populate file element with documentation
        file_element.todos = [
            {"kind": t.kind, "text": t.text, "line": t.line,
             "assignee": t.assignee, "priority": t.priority, "issue_ref": t.issue_ref}
            for t in todos
        ]
        file_element.section_markers = [
            {"label": m.label, "line": m.line, "style": m.style}
            for m in section_markers
        ]

        # Build list of ExtractedCall objects from element calls for purity analysis
        def build_extracted_calls(elem_calls: list[Call] | None) -> list[ExtractedCall]:
            if not elem_calls:
                return []
            return [
                ExtractedCall(
                    name=c.name,
                    receiver=c.receiver,
                    line=c.line
                )
                for c in elem_calls
            ]

        # Process each element for extended intelligence
        for elem in elements:
            if elem.element_type == "file":
                continue

            # Associate comments
            assoc_comments = associate_comments(elem.line_start, all_comments)
            elem.associated_comments = [
                {"text": c.text, "line": c.line, "kind": c.kind, "position": c.position}
                for c in assoc_comments
            ]

            # Type annotations (for functions/methods)
            if elem.element_type in ("function", "method"):
                # Parse for type annotations if we have raw_code
                if elem.raw_code:
                    try:
                        elem_tree = self.manager.parse(elem.raw_code.encode("utf-8"), "python")
                        type_annots = extract_type_annotations(elem_tree.root_node, "python")
                        elem.type_annotations = [
                            {"name": a.name, "kind": a.kind, "location": a.location,
                             "line": a.line, "generic_args": a.generic_args}
                            for a in type_annots
                        ]
                    except Exception:
                        pass  # Skip type extraction if parsing fails

                # Purity analysis
                calls = build_extracted_calls(elem.calls)
                mutations = elem.attributes_modified or []
                purity = analyze_purity(calls, mutations, "python")
                elem.purity = {
                    "level": purity.level,
                    "confidence": purity.confidence,
                    "reasons": purity.reasons,
                }
                effects = extract_side_effects(calls, mutations, "python")
                elem.side_effects = [
                    {"kind": e.kind, "target": e.target, "line": e.line}
                    for e in effects
                ]
                elem.mutated_state = mutations

            # API surface detection
            if elem.decorator_details:
                dec_infos = [
                    DecoratorInfo(name=d.get("name", ""), args=d.get("args"), full=d.get("full"))
                    for d in elem.decorator_details
                ]
                routes = detect_http_routes(dec_infos, "python")
                elem.http_routes = [
                    {"method": r.method, "path": r.path,
                     "path_params": r.path_params, "framework": r.framework}
                    for r in routes
                ]
                commands = detect_cli_commands(dec_infos, elem.name, "python")
                elem.cli_commands = [
                    {"name": c.name, "options": c.options, "framework": c.framework}
                    for c in commands
                ]
                elem.is_public_api = detect_public_api(
                    elem.name, dec_infos, elem.visibility, "python"
                )
            else:
                elem.is_public_api = detect_public_api(
                    elem.name, [], elem.visibility, "python"
                )

            # Pattern detection (for classes)
            if elem.element_type == "class":
                # Collect method names from child elements
                class_methods = [
                    e.name for e in elements
                    if e.parent_id == elem.element_id and e.element_type == "method"
                ]
                class_info = {
                    "name": elem.name,
                    "attributes": [a.get("name", "") for a in (elem.class_attributes or [])],
                    "methods": class_methods,
                }
                patterns, confidence = detect_patterns(class_info, [], "python")
                elem.detected_patterns = patterns
                elem.pattern_confidence = confidence

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

        # Extract class attributes and base classes from AST
        class_attributes = None
        base_classes = None
        if ext.node:
            class_attributes = extract_python_class_attributes(ext.node) or None
            base_classes = extract_python_base_classes(ext.node) or None

        # Convert decorator_details to dicts for storage
        decorator_details = None
        if ext.decorator_details:
            decorator_details = [
                {"name": d.name, "args": d.args, "full": d.full}
                for d in ext.decorator_details
            ]

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
            decorator_details=decorator_details,
            level=1,
            visibility=_determine_visibility(ext.name),
            class_attributes=class_attributes,
            base_classes=base_classes,
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

        # Extract calls and exceptions from function body
        calls: list[Call] = []
        exceptions_raised = None
        if ext.node:
            extracted_calls = extract_python_calls(ext.node)
            calls = [
                Call(name=c.name, receiver=c.receiver, line=c.line)
                for c in extracted_calls
            ]
            exceptions_raised = extract_python_raised_exceptions(ext.node) or None

        # Convert decorator_details to dicts for storage
        decorator_details = None
        if ext.decorator_details:
            decorator_details = [
                {"name": d.name, "args": d.args, "full": d.full}
                for d in ext.decorator_details
            ]

        # Convert parameters to dicts for storage
        parameters = None
        if ext.parameters:
            parameters = [
                {"name": p.name, "type": p.type, "default": p.default}
                for p in ext.parameters
            ]

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
            decorator_details=decorator_details,
            is_async=ext.is_async,
            visibility=_determine_visibility(ext.name),
            level=2,
            calls=calls,
            exceptions_raised=exceptions_raised,
            return_type=ext.return_type,
            parameters=parameters or [],
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

        # Extract calls, exceptions, and modified attributes from method body
        calls: list[Call] = []
        exceptions_raised = None
        attributes_modified = None
        if ext.node:
            extracted_calls = extract_python_calls(ext.node)
            calls = [
                Call(name=c.name, receiver=c.receiver, line=c.line)
                for c in extracted_calls
            ]
            exceptions_raised = extract_python_raised_exceptions(ext.node) or None
            attributes_modified = extract_python_modified_attributes(ext.node) or None

        # Convert decorator_details to dicts for storage
        decorator_details = None
        if ext.decorator_details:
            decorator_details = [
                {"name": d.name, "args": d.args, "full": d.full}
                for d in ext.decorator_details
            ]

        # Convert parameters to dicts for storage
        parameters = None
        if ext.parameters:
            parameters = [
                {"name": p.name, "type": p.type, "default": p.default}
                for p in ext.parameters
            ]

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
            decorator_details=decorator_details,
            is_async=ext.is_async,
            visibility=_determine_visibility(ext.name),
            level=2,
            parent_id=parent_class.element_id,
            calls=calls,
            exceptions_raised=exceptions_raised,
            attributes_modified=attributes_modified,
            return_type=ext.return_type,
            parameters=parameters or [],
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

        # Extract imports and populate on file element
        extracted_imports = extract_javascript_imports(tree, lines)
        file_element.imports = [
            Import(
                name=imp.name,
                module=imp.module,
                alias=imp.alias,
                line=imp.line,
            )
            for imp in extracted_imports
        ]

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
        # Extract class fields and base class from AST
        class_attributes = None
        base_classes = None
        if ext.node:
            fields = extract_javascript_class_fields(ext.node)
            class_attributes = fields if fields else None
            base_classes = extract_javascript_base_class(ext.node) or None

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
            class_attributes=class_attributes,
            base_classes=base_classes,
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
        # Extract calls and exceptions from function body
        calls: list[Call] = []
        exceptions_raised = None
        if ext.node:
            extracted_calls = extract_javascript_calls(ext.node)
            calls = [
                Call(name=c.name, receiver=c.receiver, line=c.line)
                for c in extracted_calls
            ]
            exceptions_raised = extract_javascript_thrown_exceptions(ext.node) or None

        # Convert parameters to dicts for storage
        parameters = None
        if ext.parameters:
            parameters = [
                {"name": p.name, "type": p.type, "default": p.default}
                for p in ext.parameters
            ]

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
            calls=calls,
            exceptions_raised=exceptions_raised,
            return_type=ext.return_type,
            parameters=parameters or [],
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
        # Extract calls, exceptions, and modified properties from method body
        calls: list[Call] = []
        exceptions_raised = None
        attributes_modified = None
        if ext.node:
            extracted_calls = extract_javascript_calls(ext.node)
            calls = [
                Call(name=c.name, receiver=c.receiver, line=c.line)
                for c in extracted_calls
            ]
            exceptions_raised = extract_javascript_thrown_exceptions(ext.node) or None
            attributes_modified = extract_javascript_modified_properties(ext.node) or None

        # Convert parameters to dicts for storage
        parameters = None
        if ext.parameters:
            parameters = [
                {"name": p.name, "type": p.type, "default": p.default}
                for p in ext.parameters
            ]

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
            calls=calls,
            exceptions_raised=exceptions_raised,
            attributes_modified=attributes_modified,
            return_type=ext.return_type,
            parameters=parameters or [],
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


class PhpParser(TreeSitterParser):
    """Parse PHP files using tree-sitter."""

    def __init__(self) -> None:
        super().__init__("php")

    def parse(
        self,
        content: str,
        file_info: FileInfo,
        scope: str,
        repository: str,
        username: str,
    ) -> list[CodeElement]:
        """Parse PHP content and extract elements."""
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
            language="php",
            line_start=1,
            line_end=line_count,
            level=0,
        )
        file_element.element_id = generate_element_id(
            scope, repository, username, file_info.relative_path, "file", file_element.name, 1
        )
        elements.append(file_element)

        # Parse with tree-sitter
        tree = self.manager.parse(content.encode("utf-8"), "php")
        extracted = extract_php_elements(tree, lines)

        # Extract imports
        extracted_imports = extract_php_imports(tree, lines)
        file_element.imports = [
            Import(name=imp.name, module=imp.module, alias=imp.alias, line=imp.line)
            for imp in extracted_imports
        ]

        # Convert ExtractedElements to CodeElements
        for ext in extracted:
            if ext.element_type == "class":
                class_elem = self._convert_class(ext, file_info, scope, repository, username, lines)
                elements.append(class_elem)

                # Extract class members
                if ext.node:
                    methods, properties = extract_php_class_members(ext.node, lines)

                    for method_ext in methods:
                        method_elem = self._convert_method(
                            method_ext, class_elem, file_info, scope, repository, username, lines
                        )
                        elements.append(method_elem)

                    for prop_ext in properties:
                        prop_elem = self._convert_variable(
                            prop_ext, file_info, scope, repository, username, lines, parent=class_elem
                        )
                        elements.append(prop_elem)

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
        class_attributes = None
        base_classes = None
        if ext.node:
            props = extract_php_class_properties(ext.node)
            class_attributes = props if props else None
            base_classes = extract_php_base_class(ext.node) or None

        elem = CodeElement(
            scope=scope,
            repository=repository,
            username=username,
            relative_path=file_info.relative_path,
            element_type="class",
            name=ext.name,
            language="php",
            line_start=ext.line_start,
            line_end=ext.line_end,
            raw_code=ext.raw_code,
            level=1,
            class_attributes=class_attributes,
            base_classes=base_classes,
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
        calls: list[Call] = []
        exceptions_raised = None
        if ext.node:
            extracted_calls = extract_php_calls(ext.node)
            calls = [Call(name=c.name, receiver=c.receiver, line=c.line) for c in extracted_calls]
            exceptions_raised = extract_php_thrown_exceptions(ext.node) or None

        # Convert parameters to dicts for storage
        parameters = None
        if ext.parameters:
            parameters = [
                {"name": p.name, "type": p.type, "default": p.default}
                for p in ext.parameters
            ]

        elem = CodeElement(
            scope=scope,
            repository=repository,
            username=username,
            relative_path=file_info.relative_path,
            element_type="function",
            name=ext.name,
            language="php",
            line_start=ext.line_start,
            line_end=ext.line_end,
            raw_code=ext.raw_code,
            signature=ext.signature,
            level=2,
            calls=calls,
            exceptions_raised=exceptions_raised,
            return_type=ext.return_type,
            parameters=parameters or [],
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
        calls: list[Call] = []
        exceptions_raised = None
        attributes_modified = None
        if ext.node:
            extracted_calls = extract_php_calls(ext.node)
            calls = [Call(name=c.name, receiver=c.receiver, line=c.line) for c in extracted_calls]
            exceptions_raised = extract_php_thrown_exceptions(ext.node) or None
            attributes_modified = extract_php_modified_properties(ext.node) or None

        # Convert parameters to dicts for storage
        parameters = None
        if ext.parameters:
            parameters = [
                {"name": p.name, "type": p.type, "default": p.default}
                for p in ext.parameters
            ]

        elem = CodeElement(
            scope=scope,
            repository=repository,
            username=username,
            relative_path=file_info.relative_path,
            element_type="method",
            name=ext.name,
            language="php",
            line_start=ext.line_start,
            line_end=ext.line_end,
            raw_code=ext.raw_code,
            signature=ext.signature,
            decorators=ext.decorators or [],
            level=2,
            parent_id=parent_class.element_id,
            calls=calls,
            exceptions_raised=exceptions_raised,
            attributes_modified=attributes_modified,
            return_type=ext.return_type,
            parameters=parameters or [],
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
        """Convert extracted variable/property to CodeElement."""
        elem = CodeElement(
            scope=scope,
            repository=repository,
            username=username,
            relative_path=file_info.relative_path,
            element_type="variable",
            name=ext.name,
            language="php",
            line_start=ext.line_start,
            line_end=ext.line_end,
            raw_code=ext.raw_code,
            decorators=ext.decorators or [],
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


class RustParser(TreeSitterParser):
    """Parse Rust files using tree-sitter."""

    def __init__(self) -> None:
        super().__init__("rust")

    def parse(
        self,
        content: str,
        file_info: FileInfo,
        scope: str,
        repository: str,
        username: str,
    ) -> list[CodeElement]:
        """Parse Rust content and extract elements."""
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
            language="rust",
            line_start=1,
            line_end=line_count,
            level=0,
        )
        file_element.element_id = generate_element_id(
            scope, repository, username, file_info.relative_path, "file", file_element.name, 1
        )
        elements.append(file_element)

        # Parse with tree-sitter
        tree = self.manager.parse(content.encode("utf-8"), "rust")
        extracted = extract_rust_elements(tree, lines)

        # Extract imports
        extracted_imports = extract_rust_imports(tree, lines)
        file_element.imports = [
            Import(name=imp.name, module=imp.module, alias=imp.alias, line=imp.line)
            for imp in extracted_imports
        ]

        # Convert ExtractedElements to CodeElements
        for ext in extracted:
            if ext.element_type == "class":
                # Could be struct, enum, or impl
                class_elem = self._convert_class(ext, file_info, scope, repository, username, lines)
                elements.append(class_elem)

                # For impl blocks, extract methods
                if ext.node and ext.node.type == "impl_item":
                    methods, constants = extract_rust_impl_members(ext.node, lines)

                    for method_ext in methods:
                        method_elem = self._convert_method(
                            method_ext, class_elem, file_info, scope, repository, username, lines
                        )
                        elements.append(method_elem)

                    for const_ext in constants:
                        const_elem = self._convert_constant(
                            const_ext, file_info, scope, repository, username, lines, parent=class_elem
                        )
                        elements.append(const_elem)

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
        """Convert extracted struct/enum/impl to CodeElement."""
        class_attributes = None
        base_classes = None

        if ext.node:
            if ext.node.type == "struct_item":
                fields = extract_rust_struct_fields(ext.node)
                class_attributes = fields if fields else None
            elif ext.node.type == "impl_item":
                traits = extract_rust_impl_traits(ext.node)
                base_classes = traits if traits else None

        elem = CodeElement(
            scope=scope,
            repository=repository,
            username=username,
            relative_path=file_info.relative_path,
            element_type="class",
            name=ext.name,
            language="rust",
            line_start=ext.line_start,
            line_end=ext.line_end,
            raw_code=ext.raw_code,
            decorators=ext.decorators or [],
            level=1,
            class_attributes=class_attributes,
            base_classes=base_classes,
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
        calls: list[Call] = []
        exceptions_raised = None
        if ext.node:
            extracted_calls = extract_rust_calls(ext.node)
            calls = [Call(name=c.name, receiver=c.receiver, line=c.line) for c in extracted_calls]
            exceptions_raised = extract_rust_panics(ext.node) or None

        # Convert parameters to dicts for storage
        parameters = None
        if ext.parameters:
            parameters = [
                {"name": p.name, "type": p.type, "default": p.default}
                for p in ext.parameters
            ]

        elem = CodeElement(
            scope=scope,
            repository=repository,
            username=username,
            relative_path=file_info.relative_path,
            element_type="function",
            name=ext.name,
            language="rust",
            line_start=ext.line_start,
            line_end=ext.line_end,
            raw_code=ext.raw_code,
            signature=ext.signature,
            is_async=ext.is_async,
            decorators=ext.decorators or [],
            level=2,
            calls=calls,
            exceptions_raised=exceptions_raised,
            return_type=ext.return_type,
            parameters=parameters or [],
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
        calls: list[Call] = []
        exceptions_raised = None
        attributes_modified = None
        if ext.node:
            extracted_calls = extract_rust_calls(ext.node)
            calls = [Call(name=c.name, receiver=c.receiver, line=c.line) for c in extracted_calls]
            exceptions_raised = extract_rust_panics(ext.node) or None
            attributes_modified = extract_rust_modified_fields(ext.node) or None

        # Convert parameters to dicts for storage
        parameters = None
        if ext.parameters:
            parameters = [
                {"name": p.name, "type": p.type, "default": p.default}
                for p in ext.parameters
            ]

        elem = CodeElement(
            scope=scope,
            repository=repository,
            username=username,
            relative_path=file_info.relative_path,
            element_type=ext.element_type,  # method or function (associated)
            name=ext.name,
            language="rust",
            line_start=ext.line_start,
            line_end=ext.line_end,
            raw_code=ext.raw_code,
            signature=ext.signature,
            is_async=ext.is_async,
            decorators=ext.decorators or [],
            level=2,
            parent_id=parent_class.element_id,
            calls=calls,
            exceptions_raised=exceptions_raised,
            attributes_modified=attributes_modified,
            return_type=ext.return_type,
            parameters=parameters or [],
        )
        elem.element_id = generate_element_id(
            scope, repository, username, file_info.relative_path, ext.element_type, ext.name, ext.line_start
        )
        return elem

    def _convert_constant(
        self,
        ext: ExtractedElement,
        file_info: FileInfo,
        scope: str,
        repository: str,
        username: str,
        lines: list[str],
        parent: CodeElement | None = None,
    ) -> CodeElement:
        """Convert extracted constant to CodeElement."""
        elem = CodeElement(
            scope=scope,
            repository=repository,
            username=username,
            relative_path=file_info.relative_path,
            element_type="constant",
            name=ext.name,
            language="rust",
            line_start=ext.line_start,
            line_end=ext.line_end,
            raw_code=ext.raw_code,
            level=3,
            parent_id=parent.element_id if parent else None,
        )
        elem.element_id = generate_element_id(
            scope, repository, username, file_info.relative_path, "constant", ext.name, ext.line_start
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
    "php": PhpParser(),
    "rust": RustParser(),
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

        # Extract cross-file references
        lines = content.split("\n")
        if file_info.language == "python":
            tree = parser.manager.parse(content.encode("utf-8"), "python")
            result.references = extract_python_references(tree, lines)
        elif file_info.language in ("javascript", "typescript"):
            tree = parser.manager.parse(content.encode("utf-8"), file_info.language)
            result.references = extract_javascript_references(tree, lines)

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


# =============================================================================
# CROSS-FILE REFERENCE LINKING
# =============================================================================


def _build_rich_context(
    ref: ExtractedReference,
    source_file: str,
) -> str:
    """Build a rich context string describing a reference.

    Args:
        ref: The extracted reference.
        source_file: Relative path of the file containing the reference.

    Returns:
        Human-readable description like "instantiated in setup() at db/init.py:45"
    """
    # Build location part
    location = f"at {source_file}:{ref.line}"

    # Build action part based on ref type
    if ref.ref_type == "instantiation":
        action = "instantiated"
    elif ref.ref_type == "function_call":
        action = "called"
    elif ref.ref_type == "method_call":
        action = "method called"
    elif ref.ref_type == "type_hint":
        action = "used as type"
    else:
        action = "referenced"

    # Build context part
    context = f"in {ref.containing_element}()" if ref.containing_element else "at module level"

    # Add snippet for method calls
    extra = ""
    if ref.ref_type == "method_call" and ref.context_snippet.startswith("called on "):
        extra = f" ({ref.context_snippet})"

    return f"{action} {context} {location}{extra}"


def link_references(result: ParsingResult) -> None:
    """Link extracted references to their target definitions.

    Populates context_usages on CodeElement with rich descriptions.
    This enables better summarization by showing how elements are used
    across the codebase.

    Args:
        result: ParsingResult to process (modified in place).
    """
    from collections import defaultdict

    # Build name -> elements lookup (handle duplicate names across files)
    definitions: dict[str, list[CodeElement]] = defaultdict(list)
    for pf in result.parsed_files:
        for elem in pf.elements:
            if elem.element_type in ("class", "function", "method", "constant"):
                definitions[elem.name].append(elem)

    # Track which usages we've added to avoid duplicates
    added: dict[str, set[str]] = defaultdict(set)  # element_id -> set of context strings

    # Match references to definitions
    for pf in result.parsed_files:
        source_file = pf.file_info.relative_path

        for ref in pf.references:
            if ref.target_name not in definitions:
                continue

            # Build rich context string
            context = _build_rich_context(ref, source_file)

            # Add to all matching definitions (usually just one)
            for elem in definitions[ref.target_name]:
                # Skip if this is a self-reference (same file, same element)
                if elem.relative_path == source_file and ref.containing_element == elem.name:
                    continue

                # Skip duplicates
                if context in added[elem.element_id]:
                    continue

                # Limit usages per element to avoid bloat
                if len(elem.context_usages) >= 10:
                    continue

                elem.context_usages.append(context)
                added[elem.element_id].add(context)


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

    # Link cross-file references to their definitions
    link_references(result)

    return result
