"""Phase 3: Parsing - Extract code elements using Tree-sitter.

This module parses source files and extracts structured code elements:
- Classes
- Functions/Methods
- Module-level variables and constants

Each element includes position, content, and hierarchy information.

The actual parsing logic is in the parsers/ subpackage:
- parsers/python.py - PythonParser
- parsers/javascript.py - JavaScriptParser
- parsers/php.py - PhpParser
- parsers/rust.py - RustParser
- parsers/java.py - JavaParser
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

# Re-export classes from parsers package for backward compatibility
from magaldi_core.parsers import (
    BashParser,
    Call,
    CodeElement,
    DockerfileParser,
    Import,
    JavaParser,
    JavaScriptParser,
    MarkdownParser,
    PhpParser,
    PlainTextParser,
    PythonParser,
    RustParser,
    TomlParser,
    TreeSitterParser,
    YamlParser,
    generate_element_id,
)
from magaldi_core.tree_sitter_manager import (
    ExtractedReference,
    extract_javascript_references,
    extract_python_references,
)

if TYPE_CHECKING:
    from magaldi_core.change_detection import ChangeManifest, FileInfo


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
    return any(re.search(pattern, relative_path) for pattern in patterns)


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
# DATA CLASSES FOR PARSING RESULTS
# =============================================================================


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
    def largest_elements_by_type(self) -> dict[str, tuple[str, str, int]]:
        """Get the largest element info for each type.

        Returns dict mapping element_type to (name, relative_path, char_count).
        """
        largest: dict[str, tuple[str, str, int]] = {}
        for pf in self.parsed_files:
            for elem in pf.elements:
                code_len = len(elem.raw_code or "")
                current = largest.get(elem.element_type)
                if current is None or code_len > current[2]:
                    largest[elem.element_type] = (elem.name, elem.relative_path, code_len)
        return largest

    @property
    def context_sizes(self) -> dict[str, int]:
        """Get computed context sizes per element type.

        Returns optimal num_ctx values for each element type based on
        observed maximum code sizes. Used for KV cache optimization
        during summarization.
        """
        from shared.ai.context_size import compute_context_sizes

        return compute_context_sizes(self.max_chars_by_type)  # type: ignore[no-any-return]

    @property
    def elements_by_tier(self) -> dict[int, dict]:
        """Get element statistics grouped by context tier.

        Returns dict mapping context tier to stats dict containing:
        - count: number of elements in this tier
        - max_chars: maximum char count in this tier
        - max_tokens: estimated max tokens (chars/4 + overhead)
        - largest: (name, path, chars, element_type) of largest element
        - by_type: dict of element_type -> count in this tier

        This shows the distribution of elements across tiers for
        understanding KV cache utilization with per-element context sizing.
        """
        from shared.ai.context_size import (
            CONTEXT_TIERS,
            DEFAULT_OVERHEAD,
            PROMPT_OVERHEAD,
            compute_element_num_ctx,
        )

        # Initialize all tiers
        tiers: dict[int, dict] = {
            tier: {
                "count": 0,
                "max_chars": 0,
                "max_tokens": 0,
                "largest": None,  # (name, path, chars, element_type)
                "by_type": {},
            }
            for tier in CONTEXT_TIERS
        }

        for pf in self.parsed_files:
            for elem in pf.elements:
                char_count = len(elem.raw_code or "")
                tier = compute_element_num_ctx(elem.element_type, char_count)
                overhead = PROMPT_OVERHEAD.get(elem.element_type, DEFAULT_OVERHEAD)
                tokens = char_count // 4 + overhead

                stats = tiers[tier]
                stats["count"] += 1
                stats["by_type"][elem.element_type] = stats["by_type"].get(elem.element_type, 0) + 1

                if char_count > stats["max_chars"]:
                    stats["max_chars"] = char_count
                    stats["max_tokens"] = tokens
                    stats["largest"] = (elem.name, elem.relative_path, char_count, elem.element_type)

        return tiers

    def largest_elements(self, n: int = 5) -> list[tuple[str, str, int, str]]:
        """Get the N largest elements by character count.

        Returns list of (name, path, chars, element_type) tuples sorted by chars descending.
        """
        all_elements: list[tuple[str, str, int, str]] = []
        for pf in self.parsed_files:
            for elem in pf.elements:
                char_count = len(elem.raw_code or "")
                all_elements.append((elem.name, elem.relative_path, char_count, elem.element_type))

        # Sort by char count descending and return top N
        all_elements.sort(key=lambda x: x[2], reverse=True)
        return all_elements[:n]


# =============================================================================
# PARSER REGISTRY
# =============================================================================


PARSERS: dict[str, TreeSitterParser | PlainTextParser] = {
    "python": PythonParser(),
    "javascript": JavaScriptParser("javascript"),
    "typescript": JavaScriptParser("typescript"),
    "tsx": JavaScriptParser("tsx"),
    "php": PhpParser(),
    "rust": RustParser(),
    "java": JavaParser(),
    "markdown": MarkdownParser(),
    "yaml": YamlParser(),
    "toml": TomlParser(),
    "dockerfile": DockerfileParser(),
    "bash": BashParser(),
    "text": PlainTextParser(),
}


def get_parser(language: str) -> TreeSitterParser | PlainTextParser | None:
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
