"""Base parser class and shared utilities for language-specific parsers.

This module provides:
- TreeSitterParser base class for all language parsers
- Common helper functions used across parsers
- Call data class for representing function calls
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from magaldi_core.tree_sitter_manager import get_manager

if TYPE_CHECKING:
    from magaldi_core.change_detection import FileInfo
    from magaldi_core.tree_sitter_manager import ExtractedCall


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
    category: str = "unknown"  # Call category (builtin, stdlib, external, type_resolvable, etc.)


@dataclass
class CodeElement:
    """A parsed code element (class, interface, type_alias, function, method, variable)."""

    # Identity
    element_id: str = ""

    # Location
    scope: str = ""
    repository: str = ""
    username: str = ""
    relative_path: str = ""

    # Element info
    element_type: str = ""  # 'file', 'class', 'interface', 'trait', 'enum', 'type_alias', 'function', 'method', 'constant', 'variable', 'import'
    name: str = ""
    language: str = ""

    # Position (1-indexed lines)
    line_start: int = 0
    line_end: int = 0
    byte_offset: int = 0  # Byte offset from start of file (unique ID for minified files)

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

    # Code Metrics (Tier 1)
    complexity: dict[str, Any] | None = None  # {cyclomatic, nesting_depth, branch_count}
    code_metrics: dict[str, Any] | None = None  # {line_count, param_count, char_count}
    docstring_quality: dict[str, Any] | None = None  # {has_docstring, coverage, ...}

    # Security
    security_issues: list[dict[str, Any]] = field(default_factory=list)

    # Environment & Concurrency
    env_vars: list[dict[str, Any]] = field(default_factory=list)
    concurrency: dict[str, Any] | None = None  # {is_async, uses_locks, patterns}

    # Document structure (for markdown/doc file elements)
    document_sections: list[dict[str, Any]] = field(default_factory=list)
    # Each: {title: str, level: int, line_start: int, line_end: int}

    # Roll-up statistics (for file and class elements)
    metrics_summary: dict[str, Any] | None = None  # Aggregated from child elements

    def compute_content_hash(self) -> str:
        """Compute SHA256 hash of the element's content for change detection.

        The hash is based on raw_code which contains the actual source code.
        This allows detecting when an element's implementation changes,
        even if the file around it changed but this element didn't.
        """
        import hashlib

        content = self.raw_code or ""
        self.content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        return self.content_hash


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
    byte_offset: int,
) -> str:
    """Generate unique element ID.

    Format: {scope}:{repository}:{username}:{relative_path}:{type}:{name}:{byte_offset}

    Uses byte_offset instead of line number to handle minified files where
    multiple elements may be on the same line.
    """
    return ":".join([
        scope,
        repository,
        username,
        relative_path,
        element_type,
        name,
        str(byte_offset),
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


def find_variable_usages(
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


def extract_docstring(lines: list[str], block_start: int) -> str | None:
    """Extract docstring from the start of a block.

    Handles multi-line function signatures by searching past the signature
    for the colon that ends the function header, then looking for the
    docstring from there.
    """
    if block_start >= len(lines):
        return None

    # Find where the function body starts (after the colon ending the signature).
    # For multi-line signatures like:
    #   def request(
    #       self,
    #       method,
    #   ):
    #       '''docstring'''
    # We need to search past the closing ): to find the docstring.
    body_start = block_start
    for i in range(block_start, min(block_start + 30, len(lines))):
        stripped = lines[i].rstrip()
        # Strip inline comments: "def foo():  # noqa" -> "def foo():"
        # Uses "  #" (PEP 8 convention) to avoid false positives on strings with #.
        code_part = stripped.split("  #")[0].rstrip() if "  #" in stripped else stripped
        if code_part.endswith(":") or code_part.endswith(":{"):
            body_start = i + 1
            break

    # Search for docstring within 3 lines after the body start
    for i in range(body_start, min(body_start + 3, len(lines))):
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


# =============================================================================
# Preceding doc comment extraction (JSDoc, rustdoc, PHPDoc, bash/python #)
# =============================================================================

# Languages that support /** ... */ block doc comments
_BLOCK_DOC_LANGUAGES = frozenset({"javascript", "typescript", "php", "rust"})

# Languages where #[...] attribute lines should be skipped when scanning backward
_ATTRIBUTE_LANGUAGES = frozenset({"rust", "php"})

# Section marker patterns that stop backward scanning for # comments
_SECTION_MARKERS = frozenset({"===", "---", "***", "###"})


def _extract_block_doc_comment(lines: list[str], end_idx: int) -> str | None:
    """Extract a /* ... */ or /** ... */ block doc comment ending at end_idx.

    Scans backward from end_idx to find the opening /* or /**, then extracts
    and cleans the content between them.

    Args:
        lines: Source code lines (0-indexed).
        end_idx: 0-indexed line where */ appears.

    Returns:
        Cleaned doc comment text, or None if not a valid block.
    """
    end_line = lines[end_idx].strip()
    if not end_line.endswith("*/"):
        return None

    # Single-line: /** Short description */ or /* Short description */
    if end_line.startswith("/**"):
        content = end_line[3:-2].strip()
        return content if content else None
    if end_line.startswith("/*"):
        content = end_line[2:-2].strip()
        return content if content else None

    # Multi-line: scan backward for /** or /*
    doc_lines: list[str] = []
    for i in range(end_idx, -1, -1):
        current = lines[i]
        stripped = current.strip()

        if stripped.startswith("/**") or stripped.startswith("/*"):
            # Opening line — extract content after /** or /*
            offset = 3 if stripped.startswith("/**") else 2
            after_open = stripped[offset:].strip()
            if after_open and after_open != "*":
                doc_lines.insert(0, after_open)
            break

        # Reject: another */ means we crossed a block boundary
        if i != end_idx and "*/" in stripped:
            return None

        if i == end_idx:
            # Closing line — extract content before */
            before_close = stripped[:-2].strip()
            # Strip leading * from content lines
            if before_close.startswith("*"):
                before_close = before_close[1:].strip()
            if before_close:
                doc_lines.insert(0, before_close)
        else:
            # Middle line — must start with * (standard block comment formatting)
            # or be blank. Non-comment code means we've crossed a boundary.
            if stripped and not stripped.startswith("*"):
                return None
            content = stripped
            if content.startswith("*"):
                content = content[1:].strip()
            doc_lines.insert(0, content)
    else:
        # Never found opening, invalid block
        return None

    text = "\n".join(doc_lines).strip()
    return text[:2000] if text else None


def _extract_line_doc_comments(
    lines: list[str], start_idx: int, prefix: str
) -> str | None:
    """Extract consecutive line doc comments (/// or #) scanning backward.

    Args:
        lines: Source code lines (0-indexed).
        start_idx: 0-indexed line to start scanning from (should already
            be known to have the prefix).
        prefix: Comment prefix to match ("///", "#").

    Returns:
        Cleaned doc comment text, or None.
    """
    doc_lines: list[str] = []
    for i in range(start_idx, -1, -1):
        stripped = lines[i].strip()
        if not stripped.startswith(prefix):
            break

        # For #: skip shebang and section markers
        if prefix == "#":
            if stripped.startswith("#!"):
                break
            # Check for section markers like # ===, # ---, # ***
            after_hash = stripped[1:].strip()
            if after_hash and len(after_hash) >= 3:
                first_3_chars = after_hash[:3]
                if first_3_chars in _SECTION_MARKERS:
                    break

        # Strip prefix and clean
        content = stripped[len(prefix) :]
        # Strip single leading space if present (conventional formatting)
        if content.startswith(" "):
            content = content[1:]
        doc_lines.insert(0, content)

    if not doc_lines:
        return None

    text = "\n".join(doc_lines).strip()
    return text[:2000] if text else None


def extract_preceding_doc_comment(
    lines: list[str],
    element_line: int,
    language: str,
) -> str | None:
    """Extract a doc comment immediately preceding a code element.

    Scans backward from the element's start line looking for doc comment
    patterns specific to the language:
    - JS/TS/PHP/Rust: /** ... */ block comments (JSDoc, PHPDoc, rustdoc)
    - Rust: /// line doc comments
    - Python/Bash: # comment blocks

    Allows at most 1 blank line between the doc comment and the element
    (or attribute). Does NOT extract inner doc comments (Rust //!).

    Args:
        lines: Source code lines (0-indexed list).
        element_line: 1-indexed line number of the element start.
        language: Language identifier ("python", "javascript", "typescript",
            "rust", "php", "bash").

    Returns:
        Cleaned doc comment text, or None if no doc comment found.
    """
    if element_line <= 1 or not lines:
        return None

    # Start scanning from the line above the element (0-indexed)
    idx = element_line - 2

    # Skip blank lines (allow at most 1)
    blank_count = 0
    while idx >= 0 and not lines[idx].strip():
        blank_count += 1
        if blank_count > 1:
            return None
        idx -= 1

    if idx < 0:
        return None

    # Skip #[...] attribute lines (Rust, PHP)
    if language in _ATTRIBUTE_LANGUAGES:
        while idx >= 0:
            stripped = lines[idx].strip()
            if stripped.startswith("#[") and stripped.endswith("]"):
                idx -= 1
                # Allow one blank line between attribute and doc comment
                if idx >= 0 and not lines[idx].strip():
                    idx -= 1
            else:
                break
        if idx < 0:
            return None

    stripped = lines[idx].strip()

    # Try block doc comment (/** ... */) for applicable languages
    if language in _BLOCK_DOC_LANGUAGES and stripped.endswith("*/"):
        return _extract_block_doc_comment(lines, idx)

    # Try Rust line doc comments (///)
    if language == "rust" and stripped.startswith("///"):
        return _extract_line_doc_comments(lines, idx, "///")

    # Try # line comments for Python and Bash
    if language in ("python", "bash") and stripped.startswith("#"):
        # Skip shebang
        if stripped.startswith("#!"):
            return None
        return _extract_line_doc_comments(lines, idx, "#")

    return None


def determine_visibility(name: str) -> str:
    """Determine visibility from name convention."""
    if name.startswith("__") and not name.endswith("__"):
        return "private"
    elif name.startswith("_"):
        return "protected"
    return "public"


def build_extracted_calls(elem_calls: list[Call] | None) -> list[ExtractedCall]:
    """Build list of ExtractedCall objects from element calls for purity analysis."""
    from magaldi_core.tree_sitter_manager import ExtractedCall

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


# =============================================================================
# BASE PARSER CLASS
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

    def _create_file_element(
        self,
        content: str,
        file_info: FileInfo,
        scope: str,
        repository: str,
        username: str,
        language: str,
    ) -> CodeElement:
        """Create a file-level CodeElement."""
        lines = content.split("\n")
        line_count = len(lines)

        file_element = CodeElement(
            scope=scope,
            repository=repository,
            username=username,
            relative_path=file_info.relative_path,
            element_type="file",
            name=Path(file_info.relative_path).name,
            language=language,
            line_start=1,
            line_end=line_count,
            level=0,
            raw_code=content,
        )
        file_element.element_id = generate_element_id(
            scope, repository, username, file_info.relative_path, "file", file_element.name, 1
        )
        return file_element

    def _set_hierarchy(self, elements: list[CodeElement], file_element: CodeElement) -> None:
        """Set parent IDs for elements without explicit parents."""
        for elem in elements:
            if elem.element_type == "file":
                continue
            if not elem.parent_id:
                elem.parent_id = file_element.element_id

    def _resolve_calls_in_file(
        self,
        elements: list[CodeElement],
        self_keyword: str = "self",
    ) -> None:
        """Resolve calls that can be determined at parse time (Phase 1).

        Resolves:
        - Same-file bare function calls: func() -> file-level function
        - Self-method calls: self.method() -> sibling method in same class

        Also categorizes unresolved calls.

        Args:
            elements: List of code elements to process.
            self_keyword: The keyword used for self-reference (e.g., 'self', 'this', '$this').
        """
        from magaldi_core.extractors.call_categorizer import categorize_calls

        # Build lookup for file-level functions
        file_functions: dict[str, str] = {
            e.name: e.element_id
            for e in elements
            if e.element_type == "function"
        }

        # Build lookup for methods grouped by parent class
        class_methods: dict[str, dict[str, str]] = {}
        for e in elements:
            if e.element_type == "method" and e.parent_id:
                if e.parent_id not in class_methods:
                    class_methods[e.parent_id] = {}
                class_methods[e.parent_id][e.name] = e.element_id

        # Resolve calls in each element
        for elem in elements:
            if not elem.calls:
                continue

            for call in elem.calls:
                if call.resolved_id:
                    continue

                # Strategy 1: Same-file bare function call
                if call.receiver is None and call.name in file_functions:
                    call.resolved_id = file_functions[call.name]
                    continue

                # Strategy 2: Self-method call within class
                # Handle multiple possible self keywords (e.g., PHP uses both '$this' and 'this')
                self_keywords = [self_keyword] if isinstance(self_keyword, str) else self_keyword
                if isinstance(self_keyword, str):
                    self_keywords = [self_keyword]

                if call.receiver in self_keywords and elem.parent_id:
                    sibling_methods = class_methods.get(elem.parent_id, {})
                    if call.name in sibling_methods:
                        call.resolved_id = sibling_methods[call.name]

            # Categorize all calls (resolved and unresolved)
            categorize_calls(elem.calls, self.language, elem.parameters)
