"""Language-specific parsers for extracting code elements.

This package contains tree-sitter based parsers for different programming languages:
- PythonParser: Python files with extended code intelligence
- JavaScriptParser: JavaScript and TypeScript files
- PhpParser: PHP files
- RustParser: Rust files

All parsers inherit from TreeSitterParser base class.
"""

from magaldi_core.parsers.base import (
    Call,
    CodeElement,
    Import,
    TreeSitterParser,
    build_extracted_calls,
    determine_visibility,
    extract_docstring,
    find_variable_usages,
    generate_element_id,
)
from magaldi_core.parsers.javascript import JavaScriptParser
from magaldi_core.parsers.php import PhpParser
from magaldi_core.parsers.python import PythonParser
from magaldi_core.parsers.rust import RustParser

__all__ = [
    # Base classes and utilities
    "Call",
    "CodeElement",
    "Import",
    "TreeSitterParser",
    "build_extracted_calls",
    "determine_visibility",
    "extract_docstring",
    "find_variable_usages",
    "generate_element_id",
    # Language-specific parsers
    "JavaScriptParser",
    "PhpParser",
    "PythonParser",
    "RustParser",
]
