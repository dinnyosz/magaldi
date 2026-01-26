"""Data classes for extracted code elements.

This module contains all the dataclass definitions used by extractors
to represent code elements, imports, references, and related data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tree_sitter import Node


class TreeSitterError(Exception):
    """Raised when tree-sitter operations fail."""

    pass


@dataclass
class ExtractedImport:
    """An import statement extracted from the AST."""

    name: str  # Imported name (e.g., "os", "process")
    module: str  # Source module (e.g., "os", "utils", "./utils")
    alias: str | None  # Alias if any (e.g., "pd" for "import pandas as pd")
    line: int  # Line number (1-indexed)


@dataclass
class DecoratorInfo:
    """Detailed information about a decorator."""

    name: str  # e.g., "router.get", "click.option"
    args: str | None = None  # e.g., '"/users/{id}"', '"--verbose"'
    full: str | None = None  # e.g., 'router.get("/users/{id}")'


@dataclass
class ParameterInfo:
    """Information about a function parameter."""

    name: str
    type: str | None = None
    default: str | None = None


@dataclass
class ExtractedElement:
    """A code element extracted from the AST."""

    element_type: str  # 'class', 'function', 'method', 'constant', 'variable'
    name: str
    line_start: int  # 1-indexed
    line_end: int  # 1-indexed
    raw_code: str
    signature: str | None = None
    decorators: list[str] | None = None
    decorator_details: list[DecoratorInfo] | None = None  # Rich decorator info
    is_async: bool = False
    parent_node: Node | None = None  # For tracking hierarchy
    node: Node | None = None  # The AST node itself
    return_type: str | None = None  # For functions/methods
    parameters: list[ParameterInfo] | None = None  # For functions/methods

    @property
    def char_count(self) -> int:
        """Character count of the raw code for identifying large elements."""
        return len(self.raw_code) if self.raw_code else 0


@dataclass
class ExtractedReference:
    """A reference to a code element from another location.

    Used to track cross-file usages for better summarization context.
    """

    ref_type: str  # 'instantiation', 'function_call', 'method_call', 'type_hint'
    target_name: str  # name being referenced (e.g., 'MyClass', 'my_function')
    line: int  # 1-indexed line number
    containing_element: str | None = None  # function/method/class name containing this ref
    context_snippet: str = ""  # brief code context for rich descriptions


@dataclass
class ExtractedCall:
    """A function/method call extracted from within a function body.

    Used for building call graphs between functions.
    """

    name: str  # Function/method name (e.g., "process", "validate")
    receiver: str | None  # Receiver object (e.g., "self", "utils", None for bare calls)
    line: int  # 1-indexed line number


# =============================================================================
# EXTENDED CODE INTELLIGENCE DATA STRUCTURES
# =============================================================================


@dataclass
class TypeAnnotation:
    """Type annotation extracted from source code."""

    name: str  # "User", "List[str]", "Optional[int]"
    kind: str  # "parameter", "return", "variable", "attribute"
    location: str  # "param:user_id", "return", "var:result"
    line: int
    generic_args: list[str] | None = None  # ["str"] for List[str]


@dataclass
class TodoItem:
    """A TODO/FIXME comment extracted from source code."""

    kind: str  # "TODO", "FIXME", "HACK", "XXX", "BUG", "NOTE"
    text: str
    line: int
    assignee: str | None = None  # "alice" from TODO(alice)
    priority: str | None = None  # "high", "low" (from ! markers)
    issue_ref: str | None = None  # "GH-123", "#456"


@dataclass
class SectionMarker:
    """A section marker comment (e.g., # === HELPERS ===)."""

    label: str  # "HELPERS", "PRIVATE METHODS"
    line: int
    style: str  # "equals", "dashes", "hash"


@dataclass
class Comment:
    """A comment associated with a code element."""

    text: str
    line: int
    kind: str  # "inline", "block", "docstring"
    position: str  # "above", "inline", "below"


@dataclass
class HttpRoute:
    """An HTTP route extracted from a web framework."""

    method: str  # "GET", "POST", "PUT", "DELETE"
    path: str  # "/users/{id}"
    path_params: list[str]  # ["id"]
    framework: str  # "fastapi", "flask", "express"


@dataclass
class CliCommand:
    """A CLI command extracted from a CLI framework."""

    name: str  # "parse", "index"
    options: list[dict[str, Any]]  # CliOption as dicts
    framework: str  # "click", "typer", "argparse"


@dataclass
class PurityInfo:
    """Purity analysis result for a function."""

    level: str  # "pure", "read_only", "mutates_self", "mutates_external"
    confidence: str  # "high", "medium", "low"
    reasons: list[str]  # ["calls print()", "modifies self.cache"]


@dataclass
class SideEffect:
    """A side effect detected in a function."""

    kind: str  # "state_mutation", "io_file", "io_network", "console", "subprocess"
    target: str | None  # "self.cache", "/tmp/file.txt"
    line: int
