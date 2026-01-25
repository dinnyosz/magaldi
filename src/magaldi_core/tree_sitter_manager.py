"""Tree-sitter parser manager for multiple languages.

This module provides:
- Lazy loading of tree-sitter language parsers
- Unified interface for parsing source code into ASTs
- Tree traversal utilities for extracting code elements
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import tree_sitter_javascript as ts_javascript
import tree_sitter_php as ts_php
import tree_sitter_python as ts_python
import tree_sitter_rust as ts_rust
import tree_sitter_typescript as ts_typescript
from tree_sitter import Language, Node, Parser, Tree


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


class TreeSitterManager:
    """Manages tree-sitter parsers for multiple languages.

    This class provides:
    - Language registration with their tree-sitter bindings
    - Parser creation and caching
    - AST parsing for source code
    """

    # Language configuration: name -> (module, language_func)
    LANGUAGE_CONFIG: dict[str, tuple[Any, str]] = {
        "python": (ts_python, "language"),
        "javascript": (ts_javascript, "language"),
        "typescript": (ts_typescript, "language_typescript"),
        "tsx": (ts_typescript, "language_tsx"),
        "php": (ts_php, "language_php"),
        "rust": (ts_rust, "language"),
    }

    def __init__(self) -> None:
        self._languages: dict[str, Language] = {}
        self._parsers: dict[str, Parser] = {}

    def _get_language(self, language: str) -> Language:
        """Get or create Language object for a language."""
        if language in self._languages:
            return self._languages[language]

        if language not in self.LANGUAGE_CONFIG:
            raise TreeSitterError(f"Unsupported language: {language}")

        module, func_name = self.LANGUAGE_CONFIG[language]
        lang_func = getattr(module, func_name)
        lang = Language(lang_func())

        self._languages[language] = lang
        return lang

    def get_parser(self, language: str) -> Parser:
        """Get or create parser for a language."""
        if language in self._parsers:
            return self._parsers[language]

        parser = Parser(self._get_language(language))
        self._parsers[language] = parser
        return parser

    def parse(self, content: bytes, language: str) -> Tree:
        """Parse source code into AST."""
        parser = self.get_parser(language)
        return parser.parse(content)

    def is_language_supported(self, language: str) -> bool:
        """Check if a language is supported."""
        return language in self.LANGUAGE_CONFIG


def walk_tree(node: Node) -> Iterator[Node]:
    """Walk all nodes in the tree depth-first."""
    yield node
    for child in node.children:
        yield from walk_tree(child)


def find_nodes(root: Node, node_type: str) -> Iterator[Node]:
    """Find all nodes of a specific type."""
    for node in walk_tree(root):
        if node.type == node_type:
            yield node


def get_node_text(node: Node) -> str:
    """Get the text content of a node."""
    return node.text.decode("utf-8") if node.text else ""


def get_child_by_field(node: Node, field_name: str) -> Node | None:
    """Get a child node by its field name."""
    return node.child_by_field_name(field_name)


def get_children_by_type(node: Node, node_type: str) -> list[Node]:
    """Get all direct children of a specific type."""
    return [child for child in node.children if child.type == node_type]


# =============================================================================
# PYTHON EXTRACTOR
# =============================================================================


def extract_python_elements(tree: Tree, lines: list[str]) -> list[ExtractedElement]:
    """Extract code elements from a Python AST.

    Args:
        tree: Parsed tree-sitter Tree.
        lines: Source code lines for raw code extraction.

    Returns:
        List of extracted elements.
    """
    elements: list[ExtractedElement] = []
    root = tree.root_node

    # Extract classes (both decorated and undecorated)
    for node in root.children:
        if node.type == "class_definition":
            elements.append(_extract_python_class(node, lines))
        elif node.type == "decorated_definition":
            inner = get_child_by_field(node, "definition")
            if inner and inner.type == "class_definition":
                deco_names, deco_details = _get_decorators(node)
                elements.append(
                    _extract_python_class(inner, lines, decorators=deco_names, decorator_details=deco_details, decorated_node=node)
                )
            elif inner and inner.type == "function_definition":
                deco_names, deco_details = _get_decorators(node)
                elements.append(
                    _extract_python_function(inner, lines, decorators=deco_names, decorator_details=deco_details, decorated_node=node)
                )
        elif node.type == "function_definition":
            elements.append(_extract_python_function(node, lines))
        elif node.type == "expression_statement":
            # Module-level assignments
            assign = get_children_by_type(node, "assignment")
            if assign:
                elem = _extract_python_assignment(assign[0], lines, is_module_level=True)
                if elem:
                    elements.append(elem)

    return elements


def _extract_python_class(
    node: Node,
    lines: list[str],
    decorators: list[str] | None = None,
    decorator_details: list[DecoratorInfo] | None = None,
    decorated_node: Node | None = None,
) -> ExtractedElement:
    """Extract a class definition.

    Args:
        node: The class_definition node.
        lines: Source code lines.
        decorators: List of decorator names (if any).
        decorated_node: The outer decorated_definition node (if class is decorated).
                       Used to include decorator lines in raw_code.
    """
    name_node = get_child_by_field(node, "name")
    name = get_node_text(name_node) if name_node else "unknown"

    # Use decorated_node's start if available (to include decorators in raw_code)
    start_node = decorated_node if decorated_node else node
    line_start = start_node.start_point[0] + 1  # 0-indexed to 1-indexed
    line_end = node.end_point[0] + 1
    raw_code = "\n".join(lines[line_start - 1 : line_end])

    elem = ExtractedElement(
        element_type="class",
        name=name,
        line_start=line_start,
        line_end=line_end,
        raw_code=raw_code,
        decorators=decorators,
        decorator_details=decorator_details,
        node=node,
    )

    return elem


def _extract_python_parameters(params_node: Node) -> list[ParameterInfo]:
    """Extract structured parameter info from Python parameters node.

    Args:
        params_node: The 'parameters' AST node.

    Returns:
        List of ParameterInfo with name, type, and default value.
    """
    parameters: list[ParameterInfo] = []
    if not params_node:
        return parameters

    for child in params_node.children:
        # Skip commas, parens, etc.
        if child.type not in (
            "identifier",
            "typed_parameter",
            "default_parameter",
            "typed_default_parameter",
            "list_splat_pattern",
            "dictionary_splat_pattern",
        ):
            continue

        param_name: str | None = None
        param_type: str | None = None
        param_default: str | None = None

        if child.type == "identifier":
            # Simple parameter: x
            param_name = get_node_text(child)
        elif child.type == "typed_parameter":
            # Typed parameter: x: int
            name_node = child.children[0] if child.children else None
            param_name = get_node_text(name_node) if name_node else None
            type_node = get_child_by_field(child, "type")
            param_type = get_node_text(type_node) if type_node else None
        elif child.type == "default_parameter":
            # Default parameter: x=5
            name_node = get_child_by_field(child, "name")
            param_name = get_node_text(name_node) if name_node else None
            value_node = get_child_by_field(child, "value")
            param_default = get_node_text(value_node) if value_node else None
        elif child.type == "typed_default_parameter":
            # Typed default: x: int = 5
            name_node = get_child_by_field(child, "name")
            param_name = get_node_text(name_node) if name_node else None
            type_node = get_child_by_field(child, "type")
            param_type = get_node_text(type_node) if type_node else None
            value_node = get_child_by_field(child, "value")
            param_default = get_node_text(value_node) if value_node else None
        elif child.type == "list_splat_pattern":
            # *args
            for c in child.children:
                if c.type == "identifier":
                    param_name = f"*{get_node_text(c)}"
                    break
        elif child.type == "dictionary_splat_pattern":
            # **kwargs
            for c in child.children:
                if c.type == "identifier":
                    param_name = f"**{get_node_text(c)}"
                    break

        if param_name:
            parameters.append(ParameterInfo(
                name=param_name,
                type=param_type,
                default=param_default,
            ))

    return parameters


def _extract_python_function(
    node: Node,
    lines: list[str],
    decorators: list[str] | None = None,
    decorator_details: list[DecoratorInfo] | None = None,
    is_method: bool = False,
    decorated_node: Node | None = None,
) -> ExtractedElement:
    """Extract a function/method definition.

    Args:
        node: The function_definition node.
        lines: Source code lines.
        decorators: List of decorator names (if any).
        is_method: Whether this is a method (vs standalone function).
        decorated_node: The outer decorated_definition node (if function is decorated).
                       Used to include decorator lines in raw_code.
    """
    name_node = get_child_by_field(node, "name")
    name = get_node_text(name_node) if name_node else "unknown"

    params_node = get_child_by_field(node, "parameters")
    params = get_node_text(params_node) if params_node else "()"

    return_node = get_child_by_field(node, "return_type")
    return_type = get_node_text(return_node) if return_node else None

    # Extract structured parameters
    parameters = _extract_python_parameters(params_node) if params_node else []

    # Check for async
    is_async = any(child.type == "async" for child in node.children)

    # Use decorated_node's start if available (to include decorators in raw_code)
    start_node = decorated_node if decorated_node else node
    line_start = start_node.start_point[0] + 1
    line_end = node.end_point[0] + 1
    raw_code = "\n".join(lines[line_start - 1 : line_end])

    # Build signature
    signature = f"{'async ' if is_async else ''}def {name}{params}"
    if return_type:
        signature += f" -> {return_type}"

    elem_type = "method" if is_method else "function"

    return ExtractedElement(
        element_type=elem_type,
        name=name,
        line_start=line_start,
        line_end=line_end,
        raw_code=raw_code,
        signature=signature,
        decorators=decorators,
        decorator_details=decorator_details,
        is_async=is_async,
        node=node,
        return_type=return_type,
        parameters=parameters or None,
    )


def _extract_python_assignment(
    node: Node, lines: list[str], is_module_level: bool = False, parent_class: Node | None = None
) -> ExtractedElement | None:
    """Extract a variable/constant assignment."""
    left_node = get_child_by_field(node, "left")
    if not left_node or left_node.type != "identifier":
        return None

    name = get_node_text(left_node)

    # Skip common non-interesting patterns
    if name in ("i", "j", "k", "x", "y", "z", "_"):
        return None

    line_start = node.start_point[0] + 1
    line_end = node.end_point[0] + 1
    raw_code = lines[line_start - 1].strip() if line_start <= len(lines) else ""

    # Determine element type: constant (UPPER_CASE) or variable
    if is_module_level and name.isupper() and "_" in name or (name.isupper() and len(name) > 1):
        elem_type = "constant"
    else:
        elem_type = "variable"

    return ExtractedElement(
        element_type=elem_type,
        name=name,
        line_start=line_start,
        line_end=line_end,
        raw_code=raw_code,
        parent_node=parent_class,
        node=node,
    )


def _get_decorators(decorated_node: Node) -> tuple[list[str], list[DecoratorInfo]]:
    """Extract decorator names and details from a decorated_definition node.

    Handles both simple decorators (@foo) and call decorators (@foo.bar(...)).

    Returns:
        Tuple of (decorator_names, decorator_details).
        decorator_names: Simple list of names for backwards compatibility.
        decorator_details: Rich info with args for entry point display.
    """
    decorators: list[str] = []
    details: list[DecoratorInfo] = []

    for child in decorated_node.children:
        if child.type == "decorator":
            # Get the full decorator text (excluding @)
            full_text = get_node_text(child)
            if full_text.startswith("@"):
                full_text = full_text[1:].strip()

            # Get the name part (after @, before ())
            for deco_child in child.children:
                if deco_child.type == "identifier":
                    # Simple decorator: @foo
                    name = get_node_text(deco_child)
                    decorators.append(name)
                    details.append(DecoratorInfo(name=name, args=None, full=full_text))
                    break
                elif deco_child.type == "attribute":
                    # Attribute decorator: @foo.bar
                    name = get_node_text(deco_child)
                    decorators.append(name)
                    details.append(DecoratorInfo(name=name, args=None, full=full_text))
                    break
                elif deco_child.type == "call":
                    # Call decorator: @foo(...) or @foo.bar(...)
                    func_node = get_child_by_field(deco_child, "function")
                    args_node = get_child_by_field(deco_child, "arguments")
                    if func_node:
                        name = get_node_text(func_node)
                        args = get_node_text(args_node) if args_node else None
                        decorators.append(name)
                        details.append(DecoratorInfo(name=name, args=args, full=full_text))
                    break

    return decorators, details


def extract_python_class_members(
    class_node: Node, lines: list[str]
) -> tuple[list[ExtractedElement], list[ExtractedElement]]:
    """Extract methods and class variables from a class.

    Args:
        class_node: The class_definition node.
        lines: Source code lines.

    Returns:
        Tuple of (methods, class_variables).
    """
    methods: list[ExtractedElement] = []
    class_vars: list[ExtractedElement] = []

    body_node = get_child_by_field(class_node, "body")
    if not body_node:
        return methods, class_vars

    for child in body_node.children:
        if child.type == "function_definition":
            methods.append(_extract_python_function(child, lines, is_method=True))
        elif child.type == "decorated_definition":
            inner = get_child_by_field(child, "definition")
            if inner and inner.type == "function_definition":
                deco_names, deco_details = _get_decorators(child)
                methods.append(
                    _extract_python_function(
                        inner, lines, decorators=deco_names, decorator_details=deco_details, is_method=True, decorated_node=child
                    )
                )
        elif child.type == "expression_statement":
            assign = get_children_by_type(child, "assignment")
            if assign:
                elem = _extract_python_assignment(assign[0], lines, parent_class=class_node)
                if elem:
                    elem.element_type = "variable"  # Class variables are always 'variable' type
                    class_vars.append(elem)

    return methods, class_vars


# =============================================================================
# PYTHON IMPORT EXTRACTION
# =============================================================================


def extract_python_imports(tree: Tree, lines: list[str]) -> list[ExtractedImport]:
    """Extract import statements from a Python AST.

    Handles:
    - import os -> Import(name="os", module="os", alias=None)
    - import pandas as pd -> Import(name="pandas", module="pandas", alias="pd")
    - from utils import process -> Import(name="process", module="utils", alias=None)
    - from utils import process as p -> Import(name="process", module="utils", alias="p")

    Args:
        tree: Parsed tree-sitter Tree.
        lines: Source code lines.

    Returns:
        List of extracted imports.
    """
    imports: list[ExtractedImport] = []
    root = tree.root_node

    for node in root.children:
        if node.type == "import_statement":
            # Handle: import os, import pandas as pd
            imports.extend(_extract_python_import_statement(node))
        elif node.type == "import_from_statement":
            # Handle: from utils import process, from utils import process as p
            imports.extend(_extract_python_import_from_statement(node))

    return imports


def _extract_python_import_statement(node: Node) -> list[ExtractedImport]:
    """Extract imports from 'import x' or 'import x as y' statements."""
    imports: list[ExtractedImport] = []
    line = node.start_point[0] + 1

    for child in node.children:
        if child.type == "dotted_name":
            # Simple import: import os
            name = get_node_text(child)
            imports.append(ExtractedImport(
                name=name,
                module=name,
                alias=None,
                line=line,
            ))
        elif child.type == "aliased_import":
            # Import with alias: import pandas as pd
            name_node = get_child_by_field(child, "name")
            alias_node = get_child_by_field(child, "alias")
            if name_node:
                name = get_node_text(name_node)
                alias = get_node_text(alias_node) if alias_node else None
                imports.append(ExtractedImport(
                    name=name,
                    module=name,
                    alias=alias,
                    line=line,
                ))

    return imports


def _extract_python_import_from_statement(node: Node) -> list[ExtractedImport]:
    """Extract imports from 'from x import y' or 'from x import y as z' statements."""
    imports: list[ExtractedImport] = []
    line = node.start_point[0] + 1

    # Get the module name
    module_node = get_child_by_field(node, "module_name")
    module = get_node_text(module_node) if module_node else ""

    # Handle relative imports (dots before module name)
    # e.g., 'from . import foo' or 'from ..utils import bar'
    relative_prefix = ""
    for child in node.children:
        if child.type == "relative_import":
            # Get all dots and the module name from relative import
            for rel_child in child.children:
                if rel_child.type == "import_prefix":
                    relative_prefix = get_node_text(rel_child)
                elif rel_child.type == "dotted_name":
                    module = relative_prefix + get_node_text(rel_child)
            break

    # If we only have dots (from . import foo), module is empty
    if not module and relative_prefix:
        module = relative_prefix

    # Get imported names
    for child in node.children:
        if child.type == "dotted_name" and child != module_node:
            # Simple import: from utils import process
            name = get_node_text(child)
            imports.append(ExtractedImport(
                name=name,
                module=module,
                alias=None,
                line=line,
            ))
        elif child.type == "aliased_import":
            # Import with alias: from utils import process as p
            name_node = get_child_by_field(child, "name")
            alias_node = get_child_by_field(child, "alias")
            if name_node:
                name = get_node_text(name_node)
                alias = get_node_text(alias_node) if alias_node else None
                imports.append(ExtractedImport(
                    name=name,
                    module=module,
                    alias=alias,
                    line=line,
                ))
        elif child.type == "wildcard_import":
            # from utils import *
            imports.append(ExtractedImport(
                name="*",
                module=module,
                alias=None,
                line=line,
            ))

    return imports


# =============================================================================
# PYTHON REFERENCE EXTRACTION
# =============================================================================


def _find_containing_element(node: Node) -> str | None:
    """Walk up AST to find the containing function/method/class name."""
    current = node.parent
    while current:
        if current.type in ("function_definition", "class_definition"):
            name_node = get_child_by_field(current, "name")
            if name_node:
                return get_node_text(name_node)
        current = current.parent
    return None


def _is_likely_class_name(name: str) -> bool:
    """Check if name looks like a class (PascalCase)."""
    if not name:
        return False
    # Starts with uppercase, not all uppercase (to exclude constants like HTTP)
    return name[0].isupper() and not name.isupper()


def _get_call_context(node: Node, lines: list[str]) -> str:
    """Get a brief context snippet for a call expression."""
    line_idx = node.start_point[0]
    if line_idx < len(lines):
        line = lines[line_idx].strip()
        # Truncate long lines
        if len(line) > 80:
            return line[:77] + "..."
        return line
    return ""


def extract_python_references(tree: Tree, lines: list[str]) -> list[ExtractedReference]:
    """Extract all references (calls, type hints) from Python AST.

    This captures cross-file usage patterns:
    - Class instantiation: MyClass()
    - Function calls: my_function()
    - Method calls: obj.method()
    - Type hints: x: MyClass, -> MyClass

    Args:
        tree: Parsed tree-sitter Tree.
        lines: Source code lines.

    Returns:
        List of extracted references.
    """
    refs: list[ExtractedReference] = []
    seen: set[tuple[str, int, str]] = set()  # (name, line, ref_type) to dedupe

    for node in walk_tree(tree.root_node):
        # Skip nodes inside import statements
        parent = node.parent
        in_import = False
        while parent:
            if parent.type in ("import_statement", "import_from_statement"):
                in_import = True
                break
            parent = parent.parent
        if in_import:
            continue

        if node.type == "call":
            func_node = get_child_by_field(node, "function")
            if not func_node:
                continue

            line = node.start_point[0] + 1
            containing = _find_containing_element(node)
            context = _get_call_context(node, lines)

            if func_node.type == "identifier":
                # Direct call: func() or MyClass()
                name = get_node_text(func_node)
                ref_type = "instantiation" if _is_likely_class_name(name) else "function_call"

                key = (name, line, ref_type)
                if key not in seen:
                    seen.add(key)
                    refs.append(ExtractedReference(
                        ref_type=ref_type,
                        target_name=name,
                        line=line,
                        containing_element=containing,
                        context_snippet=context,
                    ))

            elif func_node.type == "attribute":
                # Method call: obj.method()
                attr_node = get_child_by_field(func_node, "attribute")
                if attr_node:
                    method_name = get_node_text(attr_node)
                    # Get the object being called on
                    obj_node = get_child_by_field(func_node, "object")
                    obj_name = get_node_text(obj_node) if obj_node else ""

                    key = (method_name, line, "method_call")
                    if key not in seen:
                        seen.add(key)
                        refs.append(ExtractedReference(
                            ref_type="method_call",
                            target_name=method_name,
                            line=line,
                            containing_element=containing,
                            context_snippet=f"called on {obj_name}" if obj_name else context,
                        ))

        elif node.type == "type":
            # Type hints in function parameters or return types
            # The type node contains the actual type name
            type_text = get_node_text(node)
            line = node.start_point[0] + 1
            containing = _find_containing_element(node)

            # Extract simple type names (handle List[X], Optional[X], etc.)
            # For now, just get identifiers that look like class names
            for type_node in walk_tree(node):
                if type_node.type == "identifier":
                    name = get_node_text(type_node)
                    if _is_likely_class_name(name):
                        key = (name, line, "type_hint")
                        if key not in seen:
                            seen.add(key)
                            refs.append(ExtractedReference(
                                ref_type="type_hint",
                                target_name=name,
                                line=line,
                                containing_element=containing,
                                context_snippet=f"type annotation: {type_text[:50]}",
                            ))

    return refs


# =============================================================================
# PYTHON CALL EXTRACTION
# =============================================================================


def extract_python_calls(function_node: Node) -> list[ExtractedCall]:
    """Extract all function/method calls from within a Python function/method body.

    Handles:
    - process(x) -> ExtractedCall(name="process", receiver=None, line=45)
    - self.validate() -> ExtractedCall(name="validate", receiver="self", line=48)
    - utils.run() -> ExtractedCall(name="run", receiver="utils", line=52)
    - obj.method().chain() -> Extracts each call in the chain

    Args:
        function_node: A function_definition node from tree-sitter.

    Returns:
        List of extracted calls.
    """
    calls: list[ExtractedCall] = []
    seen: set[tuple[str, str | None, int]] = set()  # (name, receiver, line) to dedupe

    # Get the function body
    body_node = get_child_by_field(function_node, "body")
    if not body_node:
        return calls

    # Walk all nodes in the function body
    for node in walk_tree(body_node):
        if node.type == "call":
            extracted = _extract_python_call(node)
            if extracted:
                for call in extracted:
                    key = (call.name, call.receiver, call.line)
                    if key not in seen:
                        seen.add(key)
                        calls.append(call)

    return calls


def _extract_python_call(node: Node) -> list[ExtractedCall]:
    """Extract call information from a Python call node.

    Handles method chaining by extracting each call in the chain.

    Args:
        node: A 'call' node from tree-sitter.

    Returns:
        List of ExtractedCall objects (multiple for chained calls).
    """
    calls: list[ExtractedCall] = []
    func_node = get_child_by_field(node, "function")
    if not func_node:
        return calls

    line = node.start_point[0] + 1

    if func_node.type == "identifier":
        # Direct function call: func()
        name = get_node_text(func_node)
        calls.append(ExtractedCall(name=name, receiver=None, line=line))

    elif func_node.type == "attribute":
        # Method call: obj.method() or self.method() or utils.run()
        attr_node = get_child_by_field(func_node, "attribute")
        obj_node = get_child_by_field(func_node, "object")

        if attr_node:
            method_name = get_node_text(attr_node)
            receiver = None

            if obj_node:
                if obj_node.type == "identifier":
                    # Simple receiver: self.method(), utils.run()
                    receiver = get_node_text(obj_node)
                elif obj_node.type == "call":
                    # Chained call: obj.method1().method2()
                    # Get the receiver text for context (first identifier in chain)
                    receiver = _get_chain_root(obj_node)
                elif obj_node.type == "attribute":
                    # Nested attribute: a.b.method()
                    receiver = get_node_text(obj_node)

            calls.append(ExtractedCall(name=method_name, receiver=receiver, line=line))

    return calls


def _get_chain_root(node: Node) -> str | None:
    """Get the root identifier from a call chain.

    For obj.method1().method2(), returns "obj".
    """
    current = node
    while current:
        if current.type == "identifier":
            return get_node_text(current)
        elif current.type == "call":
            func = get_child_by_field(current, "function")
            current = func
        elif current.type == "attribute":
            obj = get_child_by_field(current, "object")
            current = obj
        else:
            break
    return None


# =============================================================================
# PYTHON ENHANCED CONTEXT EXTRACTION
# =============================================================================


def extract_python_class_attributes(class_node: Node) -> list[dict[str, str | int]]:
    """Extract instance attributes defined in __init__.

    Finds assignments to self.X in __init__ method to identify class state.

    Args:
        class_node: A class_definition node from tree-sitter.

    Returns:
        List of dicts with "name" and "line" keys.
        Example: [{"name": "x", "line": 10}, {"name": "_cache", "line": 11}]
    """
    attrs: list[dict[str, str | int]] = []
    seen: set[str] = set()

    body_node = get_child_by_field(class_node, "body")
    if not body_node:
        return attrs

    # Find __init__ method
    for child in body_node.children:
        method_node = None
        if child.type == "function_definition":
            method_node = child
        elif child.type == "decorated_definition":
            inner = get_child_by_field(child, "definition")
            if inner and inner.type == "function_definition":
                method_node = inner

        if not method_node:
            continue

        name_node = get_child_by_field(method_node, "name")
        if not name_node or get_node_text(name_node) != "__init__":
            continue

        # Found __init__, now find self.X = assignments
        method_body = get_child_by_field(method_node, "body")
        if not method_body:
            continue

        for node in walk_tree(method_body):
            if node.type == "assignment":
                left = node.children[0] if node.children else None
                if left and left.type == "attribute":
                    # Check if it's self.X
                    obj = left.children[0] if left.children else None
                    if obj and get_node_text(obj) == "self" and len(left.children) >= 2:
                        attr_name = get_node_text(left.children[-1])
                        if attr_name and attr_name not in seen:
                            seen.add(attr_name)
                            attrs.append({
                                "name": attr_name,
                                "line": left.start_point[0] + 1,
                            })

        break  # Only process first __init__

    return attrs


def extract_python_base_classes(class_node: Node) -> list[str]:
    """Extract base class names from class definition.

    For: class Foo(Bar, Baz): ...
    Returns: ["Bar", "Baz"]

    Args:
        class_node: A class_definition node from tree-sitter.

    Returns:
        List of base class names.
    """
    bases: list[str] = []

    if class_node.type != "class_definition":
        return bases

    # Find argument_list (superclasses)
    for child in class_node.children:
        if child.type == "argument_list":
            for arg in child.children:
                if arg.type == "identifier":
                    bases.append(get_node_text(arg))
                elif arg.type == "attribute":
                    # Handle qualified names like typing.Protocol
                    bases.append(get_node_text(arg))
                elif arg.type == "subscript":
                    # Handle Generic[T], Protocol[T], etc.
                    # Just get the base type, not the subscript
                    subscripted = arg.children[0] if arg.children else None
                    if subscripted:
                        bases.append(get_node_text(subscripted))
            break

    return bases


def extract_python_raised_exceptions(function_node: Node) -> list[str]:
    """Extract exception types from raise statements in a function/method.

    For: raise ValueError("bad")
    Returns: ["ValueError"]

    Args:
        function_node: A function_definition node from tree-sitter.

    Returns:
        List of exception type names (deduplicated).
    """
    exceptions: list[str] = []
    seen: set[str] = set()

    body_node = get_child_by_field(function_node, "body")
    if not body_node:
        return exceptions

    for node in walk_tree(body_node):
        if node.type == "raise_statement":
            # raise ValueError("msg") -> find the exception type
            for child in node.children:
                if child.type == "call":
                    # raise Exception("msg")
                    func = get_child_by_field(child, "function")
                    if func:
                        exc_name = get_node_text(func)
                        if exc_name and exc_name not in seen:
                            seen.add(exc_name)
                            exceptions.append(exc_name)
                    break
                elif child.type == "identifier":
                    # raise exc (re-raising or bare exception name)
                    exc_name = get_node_text(child)
                    if exc_name and exc_name not in seen:
                        seen.add(exc_name)
                        exceptions.append(exc_name)
                    break
                elif child.type == "attribute":
                    # raise module.Exception
                    exc_name = get_node_text(child)
                    if exc_name and exc_name not in seen:
                        seen.add(exc_name)
                        exceptions.append(exc_name)
                    break

    return exceptions


def extract_python_modified_attributes(method_node: Node) -> list[str]:
    """Extract self.X attributes that are assigned to in a method.

    Finds both regular assignments (self.x = ...) and augmented
    assignments (self.x += ...).

    Args:
        method_node: A function_definition node representing a method.

    Returns:
        List of attribute names that are modified (deduplicated).
    """
    modified: list[str] = []
    seen: set[str] = set()

    body_node = get_child_by_field(method_node, "body")
    if not body_node:
        return modified

    for node in walk_tree(body_node):
        # Check both assignment and augmented_assignment
        if node.type in ("assignment", "augmented_assignment"):
            left = node.children[0] if node.children else None
            if left and left.type == "attribute":
                # Check if it's self.X
                obj = left.children[0] if left.children else None
                if obj and get_node_text(obj) == "self" and len(left.children) >= 2:
                    attr_name = get_node_text(left.children[-1])
                    if attr_name and attr_name not in seen:
                        seen.add(attr_name)
                        modified.append(attr_name)

    return modified


# =============================================================================
# JAVASCRIPT EXTRACTOR
# =============================================================================


def extract_javascript_elements(tree: Tree, lines: list[str]) -> list[ExtractedElement]:
    """Extract code elements from a JavaScript/TypeScript AST."""
    elements: list[ExtractedElement] = []
    root = tree.root_node

    for node in walk_tree(root):
        if node.type == "class_declaration":
            elements.append(_extract_js_class(node, lines))
        elif node.type == "function_declaration":
            elements.append(_extract_js_function(node, lines))
        elif node.type == "lexical_declaration":
            # const/let declarations - check for arrow functions
            for decl in get_children_by_type(node, "variable_declarator"):
                name_node = get_child_by_field(decl, "name")
                value_node = get_child_by_field(decl, "value")
                if value_node and value_node.type == "arrow_function":
                    name = get_node_text(name_node) if name_node else "unknown"
                    elements.append(_extract_js_arrow_function(decl, name, lines))

    return elements


def _extract_js_class(node: Node, lines: list[str]) -> ExtractedElement:
    """Extract a JavaScript class."""
    name_node = get_child_by_field(node, "name")
    name = get_node_text(name_node) if name_node else "unknown"

    line_start = node.start_point[0] + 1
    line_end = node.end_point[0] + 1
    raw_code = "\n".join(lines[line_start - 1 : line_end])

    return ExtractedElement(
        element_type="class",
        name=name,
        line_start=line_start,
        line_end=line_end,
        raw_code=raw_code,
        node=node,
    )


def _extract_js_parameters(params_node: Node) -> list[ParameterInfo]:
    """Extract structured parameter info from JavaScript/TypeScript parameters node.

    Args:
        params_node: The 'formal_parameters' AST node.

    Returns:
        List of ParameterInfo with name, type, and default value.
    """
    parameters: list[ParameterInfo] = []
    if not params_node:
        return parameters

    for child in params_node.children:
        # Skip commas, parens
        if child.type in ("(", ")", ","):
            continue

        param_name: str | None = None
        param_type: str | None = None
        param_default: str | None = None

        if child.type == "identifier":
            # Simple parameter: x
            param_name = get_node_text(child)
        elif child.type == "required_parameter":
            # TypeScript required parameter with type: x: number
            pattern_node = get_child_by_field(child, "pattern")
            param_name = get_node_text(pattern_node) if pattern_node else None
            type_node = get_child_by_field(child, "type")
            param_type = get_node_text(type_node) if type_node else None
            # Strip leading ': ' from type annotation
            if param_type and param_type.startswith(": "):
                param_type = param_type[2:]
        elif child.type == "optional_parameter":
            # TypeScript optional parameter: x?: number or x: number = 5
            pattern_node = get_child_by_field(child, "pattern")
            param_name = get_node_text(pattern_node) if pattern_node else None
            type_node = get_child_by_field(child, "type")
            param_type = get_node_text(type_node) if type_node else None
            # Strip leading ': ' from type annotation
            if param_type and param_type.startswith(": "):
                param_type = param_type[2:]
            value_node = get_child_by_field(child, "value")
            param_default = get_node_text(value_node) if value_node else None
        elif child.type == "assignment_pattern":
            # JavaScript default parameter: x = 5
            left_node = get_child_by_field(child, "left")
            param_name = get_node_text(left_node) if left_node else None
            right_node = get_child_by_field(child, "right")
            param_default = get_node_text(right_node) if right_node else None
        elif child.type == "rest_pattern":
            # ...args
            for c in child.children:
                if c.type == "identifier":
                    param_name = f"...{get_node_text(c)}"
                    break

        if param_name:
            parameters.append(ParameterInfo(
                name=param_name,
                type=param_type,
                default=param_default,
            ))

    return parameters


def _extract_js_return_type(node: Node) -> str | None:
    """Extract return type from JavaScript/TypeScript function node.

    Args:
        node: A function_declaration or method_definition node.

    Returns:
        Return type string or None.
    """
    # TypeScript return type annotation comes after parameters
    return_type_node = get_child_by_field(node, "return_type")
    if return_type_node:
        ret_type = get_node_text(return_type_node)
        # Strip leading ': ' from type annotation
        if ret_type and ret_type.startswith(": "):
            ret_type = ret_type[2:]
        return ret_type
    return None


def _extract_js_function(node: Node, lines: list[str]) -> ExtractedElement:
    """Extract a JavaScript function."""
    name_node = get_child_by_field(node, "name")
    name = get_node_text(name_node) if name_node else "unknown"

    params_node = get_child_by_field(node, "parameters")
    params = get_node_text(params_node) if params_node else "()"

    # Extract structured parameters and return type
    parameters = _extract_js_parameters(params_node) if params_node else []
    return_type = _extract_js_return_type(node)

    # Check for async
    is_async = any(child.type == "async" for child in node.children)

    line_start = node.start_point[0] + 1
    line_end = node.end_point[0] + 1
    raw_code = "\n".join(lines[line_start - 1 : line_end])

    signature = f"{'async ' if is_async else ''}function {name}{params}"
    if return_type:
        signature += f": {return_type}"

    return ExtractedElement(
        element_type="function",
        name=name,
        line_start=line_start,
        line_end=line_end,
        raw_code=raw_code,
        signature=signature,
        is_async=is_async,
        node=node,
        return_type=return_type,
        parameters=parameters or None,
    )


def _extract_js_arrow_function(node: Node, name: str, lines: list[str]) -> ExtractedElement:
    """Extract a JavaScript arrow function.

    Args:
        node: The variable_declarator node containing the arrow function.
        name: The name of the arrow function (from the variable name).
        lines: Source code lines.
    """
    line_start = node.start_point[0] + 1
    line_end = node.end_point[0] + 1
    raw_code = "\n".join(lines[line_start - 1 : line_end])

    # Get the actual arrow function node for call extraction
    value_node = get_child_by_field(node, "value")
    arrow_func_node = value_node if value_node and value_node.type == "arrow_function" else node

    return ExtractedElement(
        element_type="function",
        name=name,
        line_start=line_start,
        line_end=line_end,
        raw_code=raw_code,
        node=arrow_func_node,
    )


def extract_javascript_class_members(
    class_node: Node, lines: list[str]
) -> tuple[list[ExtractedElement], list[ExtractedElement]]:
    """Extract methods and class fields from a JavaScript class."""
    methods: list[ExtractedElement] = []
    fields: list[ExtractedElement] = []

    body_node = get_child_by_field(class_node, "body")
    if not body_node:
        return methods, fields

    for child in body_node.children:
        if child.type == "method_definition":
            name_node = get_child_by_field(child, "name")
            name = get_node_text(name_node) if name_node else "unknown"

            params_node = get_child_by_field(child, "parameters")
            params = get_node_text(params_node) if params_node else "()"

            # Extract structured parameters and return type
            parameters = _extract_js_parameters(params_node) if params_node else []
            return_type = _extract_js_return_type(child)

            is_async = any(c.type == "async" for c in child.children)

            line_start = child.start_point[0] + 1
            line_end = child.end_point[0] + 1
            raw_code = "\n".join(lines[line_start - 1 : line_end])

            signature = f"{name}{params}"
            if return_type:
                signature += f": {return_type}"

            methods.append(
                ExtractedElement(
                    element_type="method",
                    name=name,
                    line_start=line_start,
                    line_end=line_end,
                    raw_code=raw_code,
                    signature=signature,
                    is_async=is_async,
                    node=child,
                    return_type=return_type,
                    parameters=parameters or None,
                )
            )
        elif child.type == "field_definition":
            name_node = get_child_by_field(child, "property")
            name = get_node_text(name_node) if name_node else "unknown"

            line_start = child.start_point[0] + 1
            line_end = child.end_point[0] + 1
            raw_code = lines[line_start - 1].strip() if line_start <= len(lines) else ""

            fields.append(
                ExtractedElement(
                    element_type="variable",
                    name=name,
                    line_start=line_start,
                    line_end=line_end,
                    raw_code=raw_code,
                    node=child,
                )
            )

    return methods, fields


# =============================================================================
# JAVASCRIPT IMPORT EXTRACTION
# =============================================================================


def extract_javascript_imports(tree: Tree, lines: list[str]) -> list[ExtractedImport]:
    """Extract import statements from a JavaScript/TypeScript AST.

    Handles:
    - import { foo } from './utils' -> Import(name="foo", module="./utils", alias=None)
    - import { foo as bar } from './utils' -> Import(name="foo", module="./utils", alias="bar")
    - import utils from './utils' -> Import(name="utils", module="./utils", alias=None)
    - import * as utils from './utils' -> Import(name="*", module="./utils", alias="utils")
    - const bar = require('lib') -> Import(name="bar", module="lib", alias=None)

    Args:
        tree: Parsed tree-sitter Tree.
        lines: Source code lines.

    Returns:
        List of extracted imports.
    """
    imports: list[ExtractedImport] = []
    root = tree.root_node

    for node in walk_tree(root):
        if node.type == "import_statement":
            imports.extend(_extract_js_import_statement(node))
        elif node.type == "lexical_declaration" or node.type == "variable_declaration":
            # Check for require() calls: const foo = require('bar')
            imports.extend(_extract_js_require_statement(node))

    return imports


def _extract_js_import_statement(node: Node) -> list[ExtractedImport]:
    """Extract imports from ES6 import statements."""
    imports: list[ExtractedImport] = []
    line = node.start_point[0] + 1

    # Get the module source (the string after 'from')
    source_node = get_child_by_field(node, "source")
    if not source_node:
        return imports

    # Remove quotes from module path
    module = get_node_text(source_node).strip("'\"")

    # Find import clause (the part before 'from')
    for child in node.children:
        if child.type == "import_clause":
            imports.extend(_extract_js_import_clause(child, module, line))

    return imports


def _extract_js_import_clause(node: Node, module: str, line: int) -> list[ExtractedImport]:
    """Extract imports from an import clause."""
    imports: list[ExtractedImport] = []

    for child in node.children:
        if child.type == "identifier":
            # Default import: import utils from './utils'
            name = get_node_text(child)
            imports.append(ExtractedImport(
                name=name,
                module=module,
                alias=None,
                line=line,
            ))
        elif child.type == "named_imports":
            # Named imports: import { foo, bar as baz } from './utils'
            for spec in child.children:
                if spec.type == "import_specifier":
                    name_node = get_child_by_field(spec, "name")
                    alias_node = get_child_by_field(spec, "alias")
                    if name_node:
                        name = get_node_text(name_node)
                        alias = get_node_text(alias_node) if alias_node else None
                        imports.append(ExtractedImport(
                            name=name,
                            module=module,
                            alias=alias,
                            line=line,
                        ))
        elif child.type == "namespace_import":
            # Namespace import: import * as utils from './utils'
            # Find the identifier after 'as'
            for ns_child in child.children:
                if ns_child.type == "identifier":
                    alias = get_node_text(ns_child)
                    imports.append(ExtractedImport(
                        name="*",
                        module=module,
                        alias=alias,
                        line=line,
                    ))
                    break

    return imports


def _extract_js_require_statement(node: Node) -> list[ExtractedImport]:
    """Extract imports from CommonJS require() calls."""
    imports: list[ExtractedImport] = []
    line = node.start_point[0] + 1

    for child in node.children:
        if child.type == "variable_declarator":
            name_node = get_child_by_field(child, "name")
            value_node = get_child_by_field(child, "value")

            if not name_node or not value_node:
                continue

            # Check if value is a require() call
            if value_node.type == "call_expression":
                func_node = get_child_by_field(value_node, "function")
                if func_node and get_node_text(func_node) == "require":
                    # Get the module argument
                    args_node = get_child_by_field(value_node, "arguments")
                    if args_node and len(args_node.children) > 0:
                        for arg in args_node.children:
                            if arg.type == "string":
                                module = get_node_text(arg).strip("'\"")
                                name = get_node_text(name_node)
                                imports.append(ExtractedImport(
                                    name=name,
                                    module=module,
                                    alias=None,
                                    line=line,
                                ))
                                break

    return imports


# =============================================================================
# JAVASCRIPT REFERENCE EXTRACTION
# =============================================================================


def _find_js_containing_element(node: Node) -> str | None:
    """Walk up AST to find the containing function/method/class name in JS."""
    current = node.parent
    while current:
        if current.type in ("function_declaration", "method_definition", "class_declaration"):
            name_node = get_child_by_field(current, "name")
            if name_node:
                return get_node_text(name_node)
        elif current.type == "variable_declarator":
            # Arrow function assigned to variable
            name_node = get_child_by_field(current, "name")
            if name_node:
                return get_node_text(name_node)
        current = current.parent
    return None


def extract_javascript_references(tree: Tree, lines: list[str]) -> list[ExtractedReference]:
    """Extract all references (calls, type annotations) from JavaScript/TypeScript AST.

    This captures cross-file usage patterns:
    - Class instantiation: new MyClass()
    - Function calls: myFunction()
    - Method calls: obj.method()
    - Type annotations (TypeScript): x: MyClass, <MyClass>

    Args:
        tree: Parsed tree-sitter Tree.
        lines: Source code lines.

    Returns:
        List of extracted references.
    """
    refs: list[ExtractedReference] = []
    seen: set[tuple[str, int, str]] = set()  # (name, line, ref_type) to dedupe

    for node in walk_tree(tree.root_node):
        # Skip nodes inside import statements
        parent = node.parent
        in_import = False
        while parent:
            if parent.type in ("import_statement", "import_specifier"):
                in_import = True
                break
            parent = parent.parent
        if in_import:
            continue

        if node.type == "new_expression":
            # Class instantiation: new MyClass()
            constructor = get_child_by_field(node, "constructor")
            if constructor and constructor.type == "identifier":
                name = get_node_text(constructor)
                line = node.start_point[0] + 1
                containing = _find_js_containing_element(node)

                key = (name, line, "instantiation")
                if key not in seen:
                    seen.add(key)
                    line_text = lines[line - 1].strip() if line <= len(lines) else ""
                    refs.append(ExtractedReference(
                        ref_type="instantiation",
                        target_name=name,
                        line=line,
                        containing_element=containing,
                        context_snippet=line_text[:80] if len(line_text) > 80 else line_text,
                    ))

        elif node.type == "call_expression":
            func_node = get_child_by_field(node, "function")
            if not func_node:
                continue

            line = node.start_point[0] + 1
            containing = _find_js_containing_element(node)

            if func_node.type == "identifier":
                # Direct function call: func()
                name = get_node_text(func_node)
                key = (name, line, "function_call")
                if key not in seen:
                    seen.add(key)
                    line_text = lines[line - 1].strip() if line <= len(lines) else ""
                    refs.append(ExtractedReference(
                        ref_type="function_call",
                        target_name=name,
                        line=line,
                        containing_element=containing,
                        context_snippet=line_text[:80] if len(line_text) > 80 else line_text,
                    ))

            elif func_node.type == "member_expression":
                # Method call: obj.method()
                prop_node = get_child_by_field(func_node, "property")
                if prop_node:
                    method_name = get_node_text(prop_node)
                    obj_node = get_child_by_field(func_node, "object")
                    obj_name = get_node_text(obj_node) if obj_node else ""

                    key = (method_name, line, "method_call")
                    if key not in seen:
                        seen.add(key)
                        refs.append(ExtractedReference(
                            ref_type="method_call",
                            target_name=method_name,
                            line=line,
                            containing_element=containing,
                            context_snippet=f"called on {obj_name}" if obj_name else "",
                        ))

        elif node.type == "type_annotation":
            # TypeScript type annotation: x: MyClass
            line = node.start_point[0] + 1
            containing = _find_js_containing_element(node)
            type_text = get_node_text(node)

            # Extract type identifiers
            for type_node in walk_tree(node):
                if type_node.type == "type_identifier":
                    name = get_node_text(type_node)
                    if _is_likely_class_name(name):
                        key = (name, line, "type_hint")
                        if key not in seen:
                            seen.add(key)
                            refs.append(ExtractedReference(
                                ref_type="type_hint",
                                target_name=name,
                                line=line,
                                containing_element=containing,
                                context_snippet=f"type annotation: {type_text[:50]}",
                            ))

    return refs


# =============================================================================
# JAVASCRIPT CALL EXTRACTION
# =============================================================================


def extract_javascript_calls(function_node: Node) -> list[ExtractedCall]:
    """Extract all function/method calls from within a JavaScript function body.

    Handles:
    - process(x) -> ExtractedCall(name="process", receiver=None, line=45)
    - this.validate() -> ExtractedCall(name="validate", receiver="this", line=48)
    - utils.run() -> ExtractedCall(name="run", receiver="utils", line=52)
    - obj.method().chain() -> Extracts each call in the chain

    Args:
        function_node: A function_declaration, method_definition, or arrow_function node.

    Returns:
        List of extracted calls.
    """
    calls: list[ExtractedCall] = []
    seen: set[tuple[str, str | None, int]] = set()  # (name, receiver, line) to dedupe

    # Get the function body
    body_node = get_child_by_field(function_node, "body")
    if not body_node:
        return calls

    # Walk all nodes in the function body
    for node in walk_tree(body_node):
        if node.type == "call_expression":
            extracted = _extract_js_call(node)
            if extracted:
                for call in extracted:
                    key = (call.name, call.receiver, call.line)
                    if key not in seen:
                        seen.add(key)
                        calls.append(call)

    return calls


def _extract_js_call(node: Node) -> list[ExtractedCall]:
    """Extract call information from a JavaScript call_expression node.

    Handles method chaining by extracting each call in the chain.

    Args:
        node: A 'call_expression' node from tree-sitter.

    Returns:
        List of ExtractedCall objects (multiple for chained calls).
    """
    calls: list[ExtractedCall] = []
    func_node = get_child_by_field(node, "function")
    if not func_node:
        return calls

    line = node.start_point[0] + 1

    if func_node.type == "identifier":
        # Direct function call: func()
        name = get_node_text(func_node)
        calls.append(ExtractedCall(name=name, receiver=None, line=line))

    elif func_node.type == "member_expression":
        # Method call: obj.method() or this.method() or utils.run()
        prop_node = get_child_by_field(func_node, "property")
        obj_node = get_child_by_field(func_node, "object")

        if prop_node:
            method_name = get_node_text(prop_node)
            receiver = None

            if obj_node:
                if obj_node.type == "identifier":
                    # Simple receiver: this.method(), utils.run()
                    receiver = get_node_text(obj_node)
                elif obj_node.type == "this":
                    # Explicit this keyword
                    receiver = "this"
                elif obj_node.type == "call_expression":
                    # Chained call: obj.method1().method2()
                    receiver = _get_js_chain_root(obj_node)
                elif obj_node.type == "member_expression":
                    # Nested property: a.b.method()
                    receiver = get_node_text(obj_node)

            calls.append(ExtractedCall(name=method_name, receiver=receiver, line=line))

    return calls


def _get_js_chain_root(node: Node) -> str | None:
    """Get the root identifier from a JavaScript call chain.

    For obj.method1().method2(), returns "obj".
    """
    current = node
    while current:
        if current.type == "identifier":
            return get_node_text(current)
        elif current.type == "this":
            return "this"
        elif current.type == "call_expression":
            func = get_child_by_field(current, "function")
            current = func
        elif current.type == "member_expression":
            obj = get_child_by_field(current, "object")
            current = obj
        else:
            break
    return None


# =============================================================================
# JAVASCRIPT ENHANCED CONTEXT EXTRACTION
# =============================================================================


def extract_javascript_class_fields(class_node: Node) -> list[dict[str, str | int]]:
    """Extract class fields from a JavaScript/TypeScript class.

    Finds:
    - Class field definitions: `field = value;`
    - Constructor assignments: `this.x = x;`

    Args:
        class_node: A class_declaration node from tree-sitter.

    Returns:
        List of dicts with "name" and "line" keys.
    """
    fields: list[dict[str, str | int]] = []
    seen: set[str] = set()

    body_node = get_child_by_field(class_node, "body")
    if not body_node:
        return fields

    for child in body_node.children:
        # Class field definitions: `field = value;` or `#field = value;`
        if child.type in ("field_definition", "public_field_definition"):
            prop_node = get_child_by_field(child, "property")
            if prop_node:
                name = get_node_text(prop_node)
                if name and name not in seen:
                    seen.add(name)
                    fields.append({
                        "name": name,
                        "line": child.start_point[0] + 1,
                    })

        # Constructor method - find this.x = assignments
        elif child.type == "method_definition":
            name_node = get_child_by_field(child, "name")
            if name_node and get_node_text(name_node) == "constructor":
                method_body = get_child_by_field(child, "body")
                if method_body:
                    for node in walk_tree(method_body):
                        if node.type == "assignment_expression":
                            left = node.children[0] if node.children else None
                            if left and left.type == "member_expression":
                                obj = get_child_by_field(left, "object")
                                prop = get_child_by_field(left, "property")
                                if obj and obj.type == "this" and prop:
                                    field_name = get_node_text(prop)
                                    if field_name and field_name not in seen:
                                        seen.add(field_name)
                                        fields.append({
                                            "name": field_name,
                                            "line": left.start_point[0] + 1,
                                        })

    return fields


def extract_javascript_base_class(class_node: Node) -> list[str]:
    """Extract the base class name from a JavaScript/TypeScript class.

    For: class Foo extends Bar { ... }
    Returns: ["Bar"]

    Args:
        class_node: A class_declaration node from tree-sitter.

    Returns:
        List with single base class name, or empty list.
    """
    bases: list[str] = []

    if class_node.type != "class_declaration":
        return bases

    # Find class_heritage node which contains extends clause
    for child in class_node.children:
        if child.type == "class_heritage":
            # The base class identifier is directly under class_heritage
            for heritage_child in child.children:
                if heritage_child.type == "identifier":
                    bases.append(get_node_text(heritage_child))
                    break
                elif heritage_child.type == "member_expression":
                    # Handle qualified names like module.ClassName
                    bases.append(get_node_text(heritage_child))
                    break
            break

    return bases


def extract_javascript_thrown_exceptions(function_node: Node) -> list[str]:
    """Extract exception types from throw statements in a JS/TS function.

    For: throw new Error("msg")
    Returns: ["Error"]

    Args:
        function_node: A function_declaration, method_definition, or arrow_function node.

    Returns:
        List of exception type names (deduplicated).
    """
    exceptions: list[str] = []
    seen: set[str] = set()

    body_node = get_child_by_field(function_node, "body")
    if not body_node:
        return exceptions

    for node in walk_tree(body_node):
        if node.type == "throw_statement":
            # throw new Error("msg") -> find the exception type
            for child in node.children:
                if child.type == "new_expression":
                    # throw new Error("msg")
                    constructor = get_child_by_field(child, "constructor")
                    if constructor:
                        exc_name = get_node_text(constructor)
                        if exc_name and exc_name not in seen:
                            seen.add(exc_name)
                            exceptions.append(exc_name)
                    break
                elif child.type == "identifier":
                    # throw err (re-throwing)
                    exc_name = get_node_text(child)
                    if exc_name and exc_name not in seen:
                        seen.add(exc_name)
                        exceptions.append(exc_name)
                    break
                elif child.type == "call_expression":
                    # throw createError("msg") - less common but possible
                    func = get_child_by_field(child, "function")
                    if func:
                        exc_name = get_node_text(func)
                        if exc_name and exc_name not in seen:
                            seen.add(exc_name)
                            exceptions.append(exc_name)
                    break

    return exceptions


def extract_javascript_modified_properties(method_node: Node) -> list[str]:
    """Extract this.X properties that are assigned to in a method.

    Finds both regular assignments (this.x = ...) and compound
    assignments (this.x += ...).

    Args:
        method_node: A method_definition or function node.

    Returns:
        List of property names that are modified (deduplicated).
    """
    modified: list[str] = []
    seen: set[str] = set()

    body_node = get_child_by_field(method_node, "body")
    if not body_node:
        return modified

    for node in walk_tree(body_node):
        # Check both assignment_expression and augmented_assignment_expression
        if node.type in ("assignment_expression", "augmented_assignment_expression"):
            left = node.children[0] if node.children else None
            if left and left.type == "member_expression":
                obj = get_child_by_field(left, "object")
                prop = get_child_by_field(left, "property")
                if obj and obj.type == "this" and prop:
                    prop_name = get_node_text(prop)
                    if prop_name and prop_name not in seen:
                        seen.add(prop_name)
                        modified.append(prop_name)

    return modified


# =============================================================================
# PHP CORE EXTRACTION
# =============================================================================


def extract_php_elements(tree: Tree, lines: list[str]) -> list[ExtractedElement]:
    """Extract classes and functions from PHP code.

    Args:
        tree: Parsed tree-sitter tree.
        lines: Source code lines.

    Returns:
        List of extracted elements (classes, functions).
    """
    elements: list[ExtractedElement] = []
    root = tree.root_node

    for node in root.children:
        if node.type == "class_declaration":
            elem = _extract_php_class(node, lines)
            if elem:
                elements.append(elem)
        elif node.type == "function_definition":
            elem = _extract_php_function(node, lines)
            if elem:
                elements.append(elem)

    return elements


def _extract_php_class(node: Node, lines: list[str]) -> ExtractedElement | None:
    """Extract a PHP class definition."""
    name = None
    for child in node.children:
        if child.type == "name":
            name = get_node_text(child)
            break

    if not name:
        return None

    return ExtractedElement(
        element_type="class",
        name=name,
        line_start=node.start_point[0] + 1,
        line_end=node.end_point[0] + 1,
        raw_code="\n".join(lines[node.start_point[0]:node.end_point[0] + 1]),
        node=node,
    )


def _extract_php_parameters(params_node: Node) -> list[ParameterInfo]:
    """Extract structured parameter info from PHP formal_parameters node.

    Args:
        params_node: The 'formal_parameters' AST node.

    Returns:
        List of ParameterInfo with name, type, and default value.
    """
    parameters: list[ParameterInfo] = []
    if not params_node:
        return parameters

    for child in params_node.children:
        if child.type not in ("simple_parameter", "variadic_parameter", "property_promotion_parameter"):
            continue

        param_name: str | None = None
        param_type: str | None = None
        param_default: str | None = None

        # Get variable name
        for c in child.children:
            if c.type == "variable_name":
                for nc in c.children:
                    if nc.type == "name":
                        param_name = get_node_text(nc)
                        break
            elif c.type in ("named_type", "primitive_type", "nullable_type", "union_type"):
                param_type = get_node_text(c)
            elif c.type == "variadic_placeholder":
                param_name = "..." + (param_name or "")

        # Get default value if present
        default_node = get_child_by_field(child, "default_value")
        if default_node:
            param_default = get_node_text(default_node)

        if param_name:
            parameters.append(ParameterInfo(
                name=param_name,
                type=param_type,
                default=param_default,
            ))

    return parameters


def _extract_php_return_type(node: Node) -> str | None:
    """Extract return type from PHP function/method node."""
    return_type_node = get_child_by_field(node, "return_type")
    if return_type_node:
        return get_node_text(return_type_node)
    return None


def _extract_php_function(node: Node, lines: list[str]) -> ExtractedElement | None:
    """Extract a PHP function definition."""
    name = None
    params_node = None

    for child in node.children:
        if child.type == "name":
            name = get_node_text(child)
        elif child.type == "formal_parameters":
            params_node = child

    if not name:
        return None

    # Extract structured parameters and return type
    parameters = _extract_php_parameters(params_node) if params_node else []
    return_type = _extract_php_return_type(node)

    # Build signature from formal_parameters
    signature = f"function {name}"
    if params_node:
        signature += get_node_text(params_node)
    if return_type:
        signature += f": {return_type}"

    return ExtractedElement(
        element_type="function",
        name=name,
        line_start=node.start_point[0] + 1,
        line_end=node.end_point[0] + 1,
        raw_code="\n".join(lines[node.start_point[0]:node.end_point[0] + 1]),
        signature=signature,
        node=node,
        return_type=return_type,
        parameters=parameters or None,
    )


def extract_php_class_members(
    class_node: Node, lines: list[str]
) -> tuple[list[ExtractedElement], list[ExtractedElement]]:
    """Extract methods and properties from a PHP class.

    Args:
        class_node: A class_declaration node.
        lines: Source code lines.

    Returns:
        Tuple of (methods, properties).
    """
    methods: list[ExtractedElement] = []
    properties: list[ExtractedElement] = []

    # Find declaration_list (class body)
    decl_list = None
    for child in class_node.children:
        if child.type == "declaration_list":
            decl_list = child
            break

    if not decl_list:
        return methods, properties

    for child in decl_list.children:
        if child.type == "method_declaration":
            elem = _extract_php_method(child, lines)
            if elem:
                methods.append(elem)
        elif child.type == "property_declaration":
            elems = _extract_php_property(child, lines)
            properties.extend(elems)

    return methods, properties


def _extract_php_method(node: Node, lines: list[str]) -> ExtractedElement | None:
    """Extract a PHP method definition."""
    name = None
    is_async = False
    visibility = "public"
    decorators: list[str] = []
    params_node = None

    for child in node.children:
        if child.type == "name":
            name = get_node_text(child)
        elif child.type == "visibility_modifier":
            visibility = get_node_text(child)
        elif child.type == "static_modifier":
            decorators.append("static")
        elif child.type == "formal_parameters":
            params_node = child

    if not name:
        return None

    # Extract structured parameters and return type
    parameters = _extract_php_parameters(params_node) if params_node else []
    return_type = _extract_php_return_type(node)

    # Build signature
    signature = f"{visibility} function {name}"
    if params_node:
        signature += get_node_text(params_node)
    if return_type:
        signature += f": {return_type}"

    return ExtractedElement(
        element_type="method",
        name=name,
        line_start=node.start_point[0] + 1,
        line_end=node.end_point[0] + 1,
        raw_code="\n".join(lines[node.start_point[0]:node.end_point[0] + 1]),
        signature=signature,
        decorators=decorators,
        node=node,
        return_type=return_type,
        parameters=parameters or None,
    )


def _extract_php_property(node: Node, lines: list[str]) -> list[ExtractedElement]:
    """Extract PHP property declarations."""
    properties: list[ExtractedElement] = []
    visibility = "public"

    for child in node.children:
        if child.type == "visibility_modifier":
            visibility = get_node_text(child)
        elif child.type == "property_element":
            for prop_child in child.children:
                if prop_child.type == "variable_name":
                    for name_child in prop_child.children:
                        if name_child.type == "name":
                            name = get_node_text(name_child)
                            properties.append(ExtractedElement(
                                element_type="variable",
                                name=name,
                                line_start=node.start_point[0] + 1,
                                line_end=node.end_point[0] + 1,
                                raw_code="\n".join(lines[node.start_point[0]:node.end_point[0] + 1]),
                                decorators=[visibility],
                            ))

    return properties


def extract_php_imports(tree: Tree, lines: list[str]) -> list[ExtractedImport]:
    """Extract use statements from PHP code.

    Args:
        tree: Parsed tree-sitter tree.
        lines: Source code lines.

    Returns:
        List of extracted imports.
    """
    imports: list[ExtractedImport] = []
    root = tree.root_node

    for node in walk_tree(root):
        if node.type == "namespace_use_declaration":
            # use Foo\Bar\Baz;
            for child in node.children:
                if child.type == "namespace_use_clause":
                    name = None
                    alias = None
                    for clause_child in child.children:
                        if clause_child.type == "qualified_name":
                            name = get_node_text(clause_child)
                        elif clause_child.type == "namespace_aliasing_clause":
                            for alias_child in clause_child.children:
                                if alias_child.type == "name":
                                    alias = get_node_text(alias_child)

                    if name:
                        # Extract the class name from full path
                        parts = name.split("\\")
                        class_name = parts[-1]
                        imports.append(ExtractedImport(
                            name=alias or class_name,
                            module=name,
                            alias=alias,
                            line=node.start_point[0] + 1,
                        ))

    return imports


def extract_php_calls(function_node: Node) -> list[ExtractedCall]:
    """Extract function/method calls from PHP code.

    Args:
        function_node: A function_definition or method_declaration node.

    Returns:
        List of extracted calls.
    """
    calls: list[ExtractedCall] = []
    seen: set[tuple[str, str | None, int]] = set()

    for node in walk_tree(function_node):
        if node.type == "function_call_expression":
            # Regular function call: func()
            func_node = node.children[0] if node.children else None
            if func_node and func_node.type == "name":
                name = get_node_text(func_node)
                line = node.start_point[0] + 1
                key = (name, None, line)
                if key not in seen:
                    seen.add(key)
                    calls.append(ExtractedCall(name=name, receiver=None, line=line))

        elif node.type == "member_call_expression":
            # Method call: $obj->method()
            line = node.start_point[0] + 1
            receiver = None
            method_name = None

            for child in node.children:
                if child.type == "variable_name":
                    for vn_child in child.children:
                        if vn_child.type == "name":
                            receiver = get_node_text(vn_child)
                elif child.type == "name":
                    method_name = get_node_text(child)

            if method_name:
                key = (method_name, receiver, line)
                if key not in seen:
                    seen.add(key)
                    calls.append(ExtractedCall(name=method_name, receiver=receiver, line=line))

        elif node.type == "scoped_call_expression":
            # Static call: Class::method()
            line = node.start_point[0] + 1
            receiver = None
            method_name = None

            for child in node.children:
                if child.type in ("name", "qualified_name", "relative_scope"):
                    receiver = get_node_text(child)
                elif child.type == "name" and receiver:
                    method_name = get_node_text(child)

            # Try getting method name differently
            if not method_name:
                for i, child in enumerate(node.children):
                    if child.type == "::":
                        if i + 1 < len(node.children):
                            next_child = node.children[i + 1]
                            if next_child.type == "name":
                                method_name = get_node_text(next_child)

            if method_name:
                key = (method_name, receiver, line)
                if key not in seen:
                    seen.add(key)
                    calls.append(ExtractedCall(name=method_name, receiver=receiver, line=line))

    return calls


# =============================================================================
# PHP ENHANCED CONTEXT EXTRACTION
# =============================================================================


def extract_php_class_properties(class_node: Node) -> list[dict[str, str | int]]:
    """Extract class properties from a PHP class.

    Finds:
    - Property declarations: `private $config;`
    - Constructor assignments: `$this->x = $x;`

    Args:
        class_node: A class_declaration node from tree-sitter.

    Returns:
        List of dicts with "name" and "line" keys.
    """
    properties: list[dict[str, str | int]] = []
    seen: set[str] = set()

    if class_node.type != "class_declaration":
        return properties

    # Find declaration_list (class body)
    decl_list = None
    for child in class_node.children:
        if child.type == "declaration_list":
            decl_list = child
            break

    if not decl_list:
        return properties

    for child in decl_list.children:
        # Property declarations: `private $config;`
        if child.type == "property_declaration":
            for prop_child in child.children:
                if prop_child.type == "property_element":
                    for var_child in prop_child.children:
                        if var_child.type == "variable_name":
                            # Get the property name (without $)
                            for name_child in var_child.children:
                                if name_child.type == "name":
                                    name = get_node_text(name_child)
                                    if name and name not in seen:
                                        seen.add(name)
                                        properties.append({
                                            "name": name,
                                            "line": var_child.start_point[0] + 1,
                                        })

        # Also check constructor for $this->x = ... assignments
        elif child.type == "method_declaration":
            method_name = None
            for method_child in child.children:
                if method_child.type == "name":
                    method_name = get_node_text(method_child)
                    break

            if method_name == "__construct":
                # Find assignments in constructor body
                for node in walk_tree(child):
                    if node.type == "assignment_expression":
                        left = node.children[0] if node.children else None
                        if left and left.type == "member_access_expression":
                            # Check if it's $this->property
                            obj = None
                            prop = None
                            for mac_child in left.children:
                                if mac_child.type == "variable_name":
                                    for vn_child in mac_child.children:
                                        if vn_child.type == "name" and get_node_text(vn_child) == "this":
                                            obj = mac_child
                                elif mac_child.type == "name":
                                    prop = mac_child

                            if obj and prop:
                                prop_name = get_node_text(prop)
                                if prop_name and prop_name not in seen:
                                    seen.add(prop_name)
                                    properties.append({
                                        "name": prop_name,
                                        "line": left.start_point[0] + 1,
                                    })

    return properties


def extract_php_base_class(class_node: Node) -> list[str]:
    """Extract the base class and interfaces from a PHP class.

    For: class Foo extends Bar implements Baz { ... }
    Returns: ["Bar", "Baz"]

    Args:
        class_node: A class_declaration node from tree-sitter.

    Returns:
        List of base class and interface names.
    """
    bases: list[str] = []

    if class_node.type != "class_declaration":
        return bases

    for child in class_node.children:
        # Base class: extends BaseClass
        if child.type == "base_clause":
            for bc_child in child.children:
                if bc_child.type == "name":
                    bases.append(get_node_text(bc_child))

        # Interfaces: implements Interface1, Interface2
        elif child.type == "class_interface_clause":
            for ic_child in child.children:
                if ic_child.type == "name":
                    bases.append(get_node_text(ic_child))

    return bases


def extract_php_thrown_exceptions(method_node: Node) -> list[str]:
    """Extract exception types from throw statements in a PHP method.

    For: throw new ValidationException("msg")
    Returns: ["ValidationException"]

    Args:
        method_node: A method_declaration or function_definition node.

    Returns:
        List of exception type names (deduplicated).
    """
    exceptions: list[str] = []
    seen: set[str] = set()

    for node in walk_tree(method_node):
        if node.type == "throw_expression":
            # throw new Exception("msg") -> find the exception type
            for child in node.children:
                if child.type == "object_creation_expression":
                    # Find the class name being instantiated
                    for oc_child in child.children:
                        if oc_child.type == "name":
                            exc_name = get_node_text(oc_child)
                            if exc_name and exc_name not in seen:
                                seen.add(exc_name)
                                exceptions.append(exc_name)
                            break
                        elif oc_child.type == "qualified_name":
                            exc_name = get_node_text(oc_child)
                            if exc_name and exc_name not in seen:
                                seen.add(exc_name)
                                exceptions.append(exc_name)
                            break
                    break
                elif child.type == "variable_name":
                    # throw $exception (re-throwing)
                    for vn_child in child.children:
                        if vn_child.type == "name":
                            exc_name = get_node_text(vn_child)
                            if exc_name and exc_name not in seen:
                                seen.add(exc_name)
                                exceptions.append(exc_name)
                    break

    return exceptions


def extract_php_modified_properties(method_node: Node) -> list[str]:
    """Extract $this->X properties that are assigned to in a PHP method.

    Args:
        method_node: A method_declaration node.

    Returns:
        List of property names that are modified (deduplicated).
    """
    modified: list[str] = []
    seen: set[str] = set()

    for node in walk_tree(method_node):
        if node.type in ("assignment_expression", "augmented_assignment_expression"):
            left = node.children[0] if node.children else None
            if left and left.type == "member_access_expression":
                # Check if it's $this->property
                is_this = False
                prop_name = None

                for mac_child in left.children:
                    if mac_child.type == "variable_name":
                        for vn_child in mac_child.children:
                            if vn_child.type == "name" and get_node_text(vn_child) == "this":
                                is_this = True
                    elif mac_child.type == "name":
                        prop_name = get_node_text(mac_child)

                if is_this and prop_name and prop_name not in seen:
                    seen.add(prop_name)
                    modified.append(prop_name)

    return modified


# =============================================================================
# RUST CORE EXTRACTION
# =============================================================================


def extract_rust_elements(tree: Tree, lines: list[str]) -> list[ExtractedElement]:
    """Extract structs, enums, and functions from Rust code.

    Args:
        tree: Parsed tree-sitter tree.
        lines: Source code lines.

    Returns:
        List of extracted elements.
    """
    elements: list[ExtractedElement] = []
    root = tree.root_node

    for node in root.children:
        if node.type == "struct_item":
            elem = _extract_rust_struct(node, lines)
            if elem:
                elements.append(elem)
        elif node.type == "enum_item":
            elem = _extract_rust_enum(node, lines)
            if elem:
                elements.append(elem)
        elif node.type == "function_item":
            elem = _extract_rust_function(node, lines)
            if elem:
                elements.append(elem)
        elif node.type == "impl_item":
            # impl blocks become "class" elements for consistency
            elem = _extract_rust_impl(node, lines)
            if elem:
                elements.append(elem)

    return elements


def _extract_rust_struct(node: Node, lines: list[str]) -> ExtractedElement | None:
    """Extract a Rust struct definition."""
    name = None
    for child in node.children:
        if child.type == "type_identifier":
            name = get_node_text(child)
            break

    if not name:
        return None

    return ExtractedElement(
        element_type="class",  # Treat struct as class for consistency
        name=name,
        line_start=node.start_point[0] + 1,
        line_end=node.end_point[0] + 1,
        raw_code="\n".join(lines[node.start_point[0]:node.end_point[0] + 1]),
        node=node,
    )


def _extract_rust_enum(node: Node, lines: list[str]) -> ExtractedElement | None:
    """Extract a Rust enum definition."""
    name = None
    for child in node.children:
        if child.type == "type_identifier":
            name = get_node_text(child)
            break

    if not name:
        return None

    return ExtractedElement(
        element_type="class",  # Treat enum as class for consistency
        name=name,
        line_start=node.start_point[0] + 1,
        line_end=node.end_point[0] + 1,
        raw_code="\n".join(lines[node.start_point[0]:node.end_point[0] + 1]),
        decorators=["enum"],
        node=node,
    )


def _extract_rust_parameters(params_node: Node) -> list[ParameterInfo]:
    """Extract structured parameter info from Rust parameters node.

    Args:
        params_node: The 'parameters' AST node.

    Returns:
        List of ParameterInfo with name, type, and default value.
    """
    parameters: list[ParameterInfo] = []
    if not params_node:
        return parameters

    for child in params_node.children:
        if child.type == "self_parameter":
            # &self, &mut self, self
            self_text = get_node_text(child)
            parameters.append(ParameterInfo(name=self_text, type=None, default=None))
        elif child.type == "parameter":
            param_name: str | None = None
            param_type: str | None = None

            # Find pattern (name) and type
            pattern_node = get_child_by_field(child, "pattern")
            type_node = get_child_by_field(child, "type")

            if pattern_node:
                param_name = get_node_text(pattern_node)
            if type_node:
                param_type = get_node_text(type_node)

            if param_name:
                parameters.append(ParameterInfo(
                    name=param_name,
                    type=param_type,
                    default=None,  # Rust doesn't have default parameters
                ))

    return parameters


def _extract_rust_return_type(node: Node) -> str | None:
    """Extract return type from Rust function/method node."""
    for i, child in enumerate(node.children):
        if child.type == "->":
            if i + 1 < len(node.children):
                ret_type = node.children[i + 1]
                return get_node_text(ret_type)
    return None


def _extract_rust_function(node: Node, lines: list[str]) -> ExtractedElement | None:
    """Extract a Rust function definition."""
    name = None
    is_async = False
    visibility = "private"
    decorators: list[str] = []
    params_node = None

    for child in node.children:
        if child.type == "identifier":
            name = get_node_text(child)
        elif child.type == "visibility_modifier":
            visibility = "public"
        elif child.type == "async":
            is_async = True
            decorators.append("async")
        elif child.type == "parameters":
            params_node = child

    if not name:
        return None

    # Extract structured parameters and return type
    parameters = _extract_rust_parameters(params_node) if params_node else []
    return_type = _extract_rust_return_type(node)

    # Build signature
    signature = f"fn {name}"
    if params_node:
        signature += get_node_text(params_node)
    if return_type:
        signature += f" -> {return_type}"

    return ExtractedElement(
        element_type="function",
        name=name,
        line_start=node.start_point[0] + 1,
        line_end=node.end_point[0] + 1,
        raw_code="\n".join(lines[node.start_point[0]:node.end_point[0] + 1]),
        signature=signature,
        is_async=is_async,
        decorators=decorators,
        node=node,
        return_type=return_type,
        parameters=parameters or None,
    )


def _extract_rust_impl(node: Node, lines: list[str]) -> ExtractedElement | None:
    """Extract a Rust impl block."""
    # Find the type being implemented for
    impl_type = None
    trait_name = None
    has_for = False

    for child in node.children:
        if child.type == "for":
            has_for = True
        elif child.type == "type_identifier":
            if has_for:
                impl_type = get_node_text(child)
            else:
                # This could be the trait or the type
                if impl_type is None:
                    impl_type = get_node_text(child)
                else:
                    trait_name = impl_type
                    impl_type = get_node_text(child)
        elif child.type == "generic_type":
            # For impl Trait for Type or impl<T> Type
            for gt_child in child.children:
                if gt_child.type == "type_identifier":
                    if has_for:
                        impl_type = get_node_text(gt_child)
                    elif trait_name is None:
                        trait_name = get_node_text(gt_child)
                    break

    if not impl_type:
        return None

    name = impl_type
    if trait_name:
        name = f"{impl_type}::{trait_name}"

    return ExtractedElement(
        element_type="class",  # Treat impl as class for consistency
        name=name,
        line_start=node.start_point[0] + 1,
        line_end=node.end_point[0] + 1,
        raw_code="\n".join(lines[node.start_point[0]:node.end_point[0] + 1]),
        decorators=["impl"] + ([trait_name] if trait_name else []),
        node=node,
    )


def extract_rust_impl_members(
    impl_node: Node, lines: list[str]
) -> tuple[list[ExtractedElement], list[ExtractedElement]]:
    """Extract methods and associated items from a Rust impl block.

    Args:
        impl_node: An impl_item node.
        lines: Source code lines.

    Returns:
        Tuple of (methods, constants).
    """
    methods: list[ExtractedElement] = []
    constants: list[ExtractedElement] = []

    # Find declaration_list
    decl_list = None
    for child in impl_node.children:
        if child.type == "declaration_list":
            decl_list = child
            break

    if not decl_list:
        return methods, constants

    for child in decl_list.children:
        if child.type == "function_item":
            elem = _extract_rust_method(child, lines)
            if elem:
                methods.append(elem)
        elif child.type == "const_item":
            elem = _extract_rust_const(child, lines)
            if elem:
                constants.append(elem)

    return methods, constants


def _extract_rust_method(node: Node, lines: list[str]) -> ExtractedElement | None:
    """Extract a Rust method from an impl block."""
    name = None
    is_async = False
    visibility = "private"
    decorators: list[str] = []
    has_self = False
    params_node = None

    for child in node.children:
        if child.type == "identifier":
            name = get_node_text(child)
        elif child.type == "visibility_modifier":
            visibility = "public"
        elif child.type == "async":
            is_async = True
            decorators.append("async")
        elif child.type == "parameters":
            params_node = child
            # Check for self parameter
            for param in child.children:
                if param.type in ("self_parameter", "self"):
                    has_self = True
                    break
                if param.type == "parameter":
                    for pc in param.children:
                        if pc.type == "self":
                            has_self = True
                            break

    if not name:
        return None

    # Extract structured parameters and return type
    parameters = _extract_rust_parameters(params_node) if params_node else []
    return_type = _extract_rust_return_type(node)

    # Build signature
    signature = f"fn {name}"
    if params_node:
        signature += get_node_text(params_node)
    if return_type:
        signature += f" -> {return_type}"

    # Method vs associated function
    elem_type = "method" if has_self else "function"

    return ExtractedElement(
        element_type=elem_type,
        name=name,
        line_start=node.start_point[0] + 1,
        line_end=node.end_point[0] + 1,
        raw_code="\n".join(lines[node.start_point[0]:node.end_point[0] + 1]),
        signature=signature,
        is_async=is_async,
        decorators=decorators,
        node=node,
        return_type=return_type,
        parameters=parameters or None,
    )


def _extract_rust_const(node: Node, lines: list[str]) -> ExtractedElement | None:
    """Extract a Rust const from an impl block."""
    name = None
    for child in node.children:
        if child.type == "identifier":
            name = get_node_text(child)
            break

    if not name:
        return None

    return ExtractedElement(
        element_type="constant",
        name=name,
        line_start=node.start_point[0] + 1,
        line_end=node.end_point[0] + 1,
        raw_code="\n".join(lines[node.start_point[0]:node.end_point[0] + 1]),
    )


def extract_rust_imports(tree: Tree, lines: list[str]) -> list[ExtractedImport]:
    """Extract use statements from Rust code.

    Args:
        tree: Parsed tree-sitter tree.
        lines: Source code lines.

    Returns:
        List of extracted imports.
    """
    imports: list[ExtractedImport] = []
    root = tree.root_node

    for node in walk_tree(root):
        if node.type == "use_declaration":
            # Extract imports from use statement
            for child in node.children:
                if child.type == "use_clause" or child.type == "scoped_identifier":
                    path = get_node_text(child)
                    if path:
                        # Get the last component as the name
                        parts = path.replace("::", ".").split(".")
                        name = parts[-1] if parts else path
                        imports.append(ExtractedImport(
                            name=name,
                            module=path,
                            alias=None,
                            line=node.start_point[0] + 1,
                        ))
                elif child.type == "use_as_clause":
                    # use foo as bar
                    original = None
                    alias = None
                    for ac_child in child.children:
                        if ac_child.type == "scoped_identifier":
                            original = get_node_text(ac_child)
                        elif ac_child.type == "identifier" and original:
                            alias = get_node_text(ac_child)
                    if original:
                        parts = original.replace("::", ".").split(".")
                        name = alias or (parts[-1] if parts else original)
                        imports.append(ExtractedImport(
                            name=name,
                            module=original,
                            alias=alias,
                            line=node.start_point[0] + 1,
                        ))

    return imports


def extract_rust_calls(function_node: Node) -> list[ExtractedCall]:
    """Extract function/method calls from Rust code.

    Args:
        function_node: A function_item node.

    Returns:
        List of extracted calls.
    """
    calls: list[ExtractedCall] = []
    seen: set[tuple[str, str | None, int]] = set()

    # Find function body (block)
    body = None
    for child in function_node.children:
        if child.type == "block":
            body = child
            break

    if not body:
        return calls

    for node in walk_tree(body):
        if node.type == "call_expression":
            func_node = node.children[0] if node.children else None
            line = node.start_point[0] + 1

            if func_node:
                if func_node.type == "identifier":
                    # Direct call: func()
                    name = get_node_text(func_node)
                    key = (name, None, line)
                    if key not in seen:
                        seen.add(key)
                        calls.append(ExtractedCall(name=name, receiver=None, line=line))

                elif func_node.type == "field_expression":
                    # Method call: self.method() or obj.method()
                    receiver = None
                    method = None
                    for fe_child in func_node.children:
                        if fe_child.type == "identifier":
                            receiver = get_node_text(fe_child)
                        elif fe_child.type == "field_identifier":
                            method = get_node_text(fe_child)

                    if method:
                        key = (method, receiver, line)
                        if key not in seen:
                            seen.add(key)
                            calls.append(ExtractedCall(name=method, receiver=receiver, line=line))

                elif func_node.type == "scoped_identifier":
                    # Static/associated call: Type::method()
                    text = get_node_text(func_node)
                    if "::" in text:
                        parts = text.split("::")
                        receiver = parts[0] if len(parts) > 1 else None
                        method = parts[-1]
                        key = (method, receiver, line)
                        if key not in seen:
                            seen.add(key)
                            calls.append(ExtractedCall(name=method, receiver=receiver, line=line))

    return calls


# =============================================================================
# RUST ENHANCED CONTEXT EXTRACTION
# =============================================================================


def extract_rust_struct_fields(struct_node: Node) -> list[dict[str, str | int]]:
    """Extract fields from a Rust struct.

    For: struct Foo { x: i32, y: String }
    Returns: [{"name": "x", "line": 2}, {"name": "y", "line": 3}]

    Args:
        struct_node: A struct_item node from tree-sitter.

    Returns:
        List of dicts with "name" and "line" keys.
    """
    fields: list[dict[str, str | int]] = []
    seen: set[str] = set()

    if struct_node.type != "struct_item":
        return fields

    # Find field_declaration_list
    for child in struct_node.children:
        if child.type == "field_declaration_list":
            for field_child in child.children:
                if field_child.type == "field_declaration":
                    # Find field_identifier
                    for fd_child in field_child.children:
                        if fd_child.type == "field_identifier":
                            name = get_node_text(fd_child)
                            if name and name not in seen:
                                seen.add(name)
                                fields.append({
                                    "name": name,
                                    "line": fd_child.start_point[0] + 1,
                                })
                            break

    return fields


def extract_rust_impl_traits(impl_node: Node) -> list[str]:
    """Extract trait implementations from a Rust impl block.

    For: impl From<Config> for MyService { ... }
    Returns: ["From"]

    For: impl MyService { ... }
    Returns: []

    Args:
        impl_node: An impl_item node from tree-sitter.

    Returns:
        List of trait names being implemented.
    """
    traits: list[str] = []

    if impl_node.type != "impl_item":
        return traits

    # Check if this is a trait impl (has 'for' keyword)
    has_for = False
    for child in impl_node.children:
        if child.type == "for":
            has_for = True
            break

    if not has_for:
        return traits

    # Find the trait type (before 'for')
    for child in impl_node.children:
        if child.type == "for":
            break
        if child.type == "type_identifier":
            traits.append(get_node_text(child))
        elif child.type == "generic_type":
            # From<Config> -> get "From"
            for gt_child in child.children:
                if gt_child.type == "type_identifier":
                    traits.append(get_node_text(gt_child))
                    break

    return traits


def extract_rust_panics(function_node: Node) -> list[str]:
    """Extract panic/error types from a Rust function.

    Finds:
    - panic!("msg") -> ["panic"]
    - Err(Error::ValidationError(...)) -> inside Err call
    - return Err(...) -> ["Err"]

    Args:
        function_node: A function_item node from tree-sitter.

    Returns:
        List of panic/error type names (deduplicated).
    """
    errors: list[str] = []
    seen: set[str] = set()

    # Find the function body (block)
    body_node = None
    for child in function_node.children:
        if child.type == "block":
            body_node = child
            break

    if not body_node:
        return errors

    for node in walk_tree(body_node):
        # panic!(...) macro
        if node.type == "macro_invocation":
            for mac_child in node.children:
                if mac_child.type == "identifier":
                    macro_name = get_node_text(mac_child)
                    if macro_name in ("panic", "unreachable", "unimplemented", "todo"):
                        if macro_name not in seen:
                            seen.add(macro_name)
                            errors.append(macro_name)
                    break

        # Err(...) call
        elif node.type == "call_expression":
            func = node.children[0] if node.children else None
            if func:
                if func.type == "identifier":
                    name = get_node_text(func)
                    if name == "Err":
                        if name not in seen:
                            seen.add(name)
                            errors.append(name)
                        # Also check for error type inside Err(...)
                        args = None
                        for child in node.children:
                            if child.type == "arguments":
                                args = child
                                break
                        if args:
                            for arg in walk_tree(args):
                                if arg.type == "scoped_identifier":
                                    text = get_node_text(arg)
                                    if "::" in text and "Error" in text:
                                        parts = text.split("::")
                                        variant = parts[-1]
                                        if variant not in seen:
                                            seen.add(variant)
                                            errors.append(variant)
                elif func.type == "scoped_identifier":
                    # Only capture if it looks like an error (contains "Error")
                    text = get_node_text(func)
                    if "::" in text and "Error" in text:
                        parts = text.split("::")
                        variant = parts[-1]
                        if variant not in seen:
                            seen.add(variant)
                            errors.append(variant)

    return errors


def extract_rust_modified_fields(function_node: Node) -> list[str]:
    """Extract self.X fields that are assigned to in a Rust method.

    Args:
        function_node: A function_item node.

    Returns:
        List of field names that are modified (deduplicated).
    """
    modified: list[str] = []
    seen: set[str] = set()

    # Find the function body (block)
    body_node = None
    for child in function_node.children:
        if child.type == "block":
            body_node = child
            break

    if not body_node:
        return modified

    for node in walk_tree(body_node):
        if node.type == "assignment_expression":
            left = node.children[0] if node.children else None
            if left and left.type == "field_expression":
                # Check if it's self.field
                obj = None
                field = None
                for fe_child in left.children:
                    if fe_child.type == "identifier" and get_node_text(fe_child) == "self":
                        obj = fe_child
                    elif fe_child.type == "field_identifier":
                        field = fe_child

                if obj and field:
                    field_name = get_node_text(field)
                    if field_name and field_name not in seen:
                        seen.add(field_name)
                        modified.append(field_name)

    return modified


# =============================================================================
# EXTENDED CODE INTELLIGENCE EXTRACTION
# =============================================================================

# TODO extraction pattern
_TODO_PATTERN = re.compile(
    r"(?P<kind>TODO|FIXME|HACK|XXX|BUG|NOTE|OPTIMIZE)"
    r"(?:\((?P<assignee>\w+)\))?"  # TODO(alice)
    r"(?:\s*(?P<priority>!+))?"  # TODO!!
    r"(?:\s*(?P<issue>[#]?\w+-?\d+))?"  # TODO #123 or GH-456
    r"\s*:?\s*"
    r"(?P<text>.+)",
    re.IGNORECASE,
)


def extract_todos(source: str) -> list[TodoItem]:
    """Extract TODO/FIXME comments from source code."""
    todos = []
    lines = source.split("\n")

    for line_num, line in enumerate(lines, start=1):
        comment_start = -1
        for marker in ("#", "//", "/*", "*"):
            idx = line.find(marker)
            if idx != -1 and (comment_start == -1 or idx < comment_start):
                comment_start = idx

        if comment_start == -1:
            continue

        comment_text = line[comment_start:]
        match = _TODO_PATTERN.search(comment_text)
        if match:
            priority = None
            if match.group("priority"):
                priority = "high" if len(match.group("priority")) >= 2 else "low"

            todos.append(
                TodoItem(
                    kind=match.group("kind").upper(),
                    text=match.group("text").strip(),
                    line=line_num,
                    assignee=match.group("assignee"),
                    priority=priority,
                    issue_ref=match.group("issue"),
                )
            )

    return todos


# Section marker pattern
_SECTION_PATTERN = re.compile(
    r"^\s*[#/]+\s*"
    r"(?P<style_start>={3,}|-{3,})?\s*"
    r"(?P<label>[A-Z][A-Z0-9 _]+)"
    r"\s*(?P<style_end>={3,}|-{3,})?\s*$",
)

# Impure function call patterns by language
IMPURE_CALLS: dict[str, dict[str, list[str]]] = {
    "python": {
        "io_file": [
            "open", "read", "write", "Path.write_text", "Path.read_text",
            "Path.mkdir", "os.remove", "shutil.copy", "shutil.move",
        ],
        "io_network": [
            "requests.get", "requests.post", "requests.put", "requests.delete",
            "httpx.get", "httpx.post", "urllib.request.urlopen", "socket.connect",
        ],
        "console": [
            "print", "logging.info", "logging.debug", "logging.warning",
            "logging.error", "logger.info", "logger.debug", "logger.warning",
        ],
        "subprocess": [
            "subprocess.run", "subprocess.call", "subprocess.Popen",
            "os.system", "os.popen",
        ],
        "database": [
            "cursor.execute", "session.commit", "session.add", "session.delete",
        ],
    },
    "javascript": {
        "io_file": [
            "fs.readFile", "fs.writeFile", "fs.readFileSync", "fs.writeFileSync",
            "fs.appendFile", "fs.unlink", "fs.mkdir",
        ],
        "io_network": [
            "fetch", "axios.get", "axios.post", "http.get", "http.request",
        ],
        "console": [
            "console.log", "console.error", "console.warn", "console.info",
        ],
        "subprocess": [
            "child_process.exec", "child_process.spawn", "child_process.execSync",
        ],
    },
    "typescript": {},  # Falls back to javascript
    "php": {
        "io_file": [
            "file_get_contents", "file_put_contents", "fopen", "fwrite", "fread",
        ],
        "io_network": [
            "curl_exec", "file_get_contents",
        ],
        "console": [
            "echo", "print", "var_dump", "print_r",
        ],
    },
    "rust": {
        "io_file": [
            "File::open", "File::create", "fs::read", "fs::write",
        ],
        "io_network": [
            "TcpStream::connect", "reqwest::get",
        ],
        "console": [
            "println!", "eprintln!", "print!", "dbg!",
        ],
        "subprocess": [
            "Command::new", "process::Command",
        ],
    },
}


def extract_section_markers(source: str) -> list[SectionMarker]:
    """Extract section marker comments from source code."""
    markers = []
    lines = source.split("\n")

    for line_num, line in enumerate(lines, start=1):
        match = _SECTION_PATTERN.match(line)
        if match:
            style_start = match.group("style_start") or ""
            style_end = match.group("style_end") or ""

            if "=" in style_start or "=" in style_end:
                style = "equals"
            elif "-" in style_start or "-" in style_end:
                style = "dashes"
            else:
                style = "hash"

            markers.append(
                SectionMarker(
                    label=match.group("label").strip(),
                    line=line_num,
                    style=style,
                )
            )

    return markers


def extract_comments(source: str) -> list[Comment]:
    """Extract all comments from source code."""
    comments = []
    lines = source.split("\n")

    for line_num, line in enumerate(lines, start=1):
        stripped = line.strip()

        if not stripped:
            continue

        # Python/shell style comments
        if "#" in line:
            hash_pos = line.find("#")
            code_before = line[:hash_pos].strip()
            kind = "inline" if code_before else "block"
            position = "inline" if kind == "inline" else "above"

            comments.append(
                Comment(
                    text=line[hash_pos + 1 :].strip(),
                    line=line_num,
                    kind=kind,
                    position=position,
                )
            )
        # C-style single line comments
        elif "//" in line:
            slash_pos = line.find("//")
            code_before = line[:slash_pos].strip()
            kind = "inline" if code_before else "block"
            position = "inline" if kind == "inline" else "above"

            comments.append(
                Comment(
                    text=line[slash_pos + 2 :].strip(),
                    line=line_num,
                    kind=kind,
                    position=position,
                )
            )

    return comments


def associate_comments(
    element_line: int,
    all_comments: list[Comment],
    max_distance: int = 3,
) -> list[Comment]:
    """Associate comments with an element based on proximity."""
    associated = []

    for comment in all_comments:
        if element_line - max_distance <= comment.line < element_line:
            comment.position = "above"
            associated.append(comment)
        elif comment.line == element_line and comment.kind == "inline":
            comment.position = "inline"
            associated.append(comment)

    return associated


# =============================================================================
# PURITY ANALYSIS
# =============================================================================


def analyze_purity(
    calls: list[ExtractedCall],
    mutations: list[str],
    language: str,
) -> PurityInfo:
    """Analyze function purity based on calls and mutations.

    Args:
        calls: List of function calls within the function.
        mutations: List of mutated attributes (e.g., ["self.cache"]).
        language: Programming language.

    Returns:
        PurityInfo with level, confidence, and reasons.
    """
    reasons = []
    level = "pure"

    # Get impure patterns for this language (fall back to python for unknown)
    lang_patterns = IMPURE_CALLS.get(language, IMPURE_CALLS.get("python", {}))

    # If language not found and no fallback, use python patterns
    if not lang_patterns and language == "typescript":
        lang_patterns = IMPURE_CALLS.get("javascript", {})

    # Check for impure calls
    for call in calls:
        call_name = call.name
        if call.receiver:
            call_name = f"{call.receiver}.{call.name}"

        for effect_kind, patterns in lang_patterns.items():
            for pattern in patterns:
                if call_name == pattern or call.name == pattern:
                    reasons.append(f"calls {call_name} ({effect_kind})")
                    level = "mutates_external"
                    break

    # Check for self mutations
    for mutation in mutations:
        if mutation.startswith("self.") or mutation.startswith("this."):
            reasons.append(f"modifies {mutation}")
            if level == "pure":
                level = "mutates_self"

    # Determine confidence
    if level == "pure" and not calls:
        confidence = "high"
    elif level == "pure" and calls:
        # Has calls but none matched impure patterns - lower confidence
        confidence = "medium"
    else:
        confidence = "high"

    return PurityInfo(level=level, confidence=confidence, reasons=reasons)


def extract_side_effects(
    calls: list[ExtractedCall],
    mutations: list[str],
    language: str,
) -> list[SideEffect]:
    """Extract individual side effects from calls and mutations.

    Args:
        calls: List of function calls.
        mutations: List of mutations.
        language: Programming language.

    Returns:
        List of SideEffect objects.
    """
    effects = []
    lang_patterns = IMPURE_CALLS.get(language, IMPURE_CALLS.get("python", {}))

    if not lang_patterns and language == "typescript":
        lang_patterns = IMPURE_CALLS.get("javascript", {})

    for call in calls:
        call_name = call.name
        if call.receiver:
            call_name = f"{call.receiver}.{call.name}"

        for effect_kind, patterns in lang_patterns.items():
            for pattern in patterns:
                if call_name == pattern or call.name == pattern:
                    effects.append(
                        SideEffect(kind=effect_kind, target=call_name, line=call.line)
                    )
                    break

    for mutation in mutations:
        effects.append(
            SideEffect(kind="state_mutation", target=mutation, line=0)
        )

    return effects


# =============================================================================
# GLOBAL SINGLETON
# =============================================================================

_manager: TreeSitterManager | None = None


def get_manager() -> TreeSitterManager:
    """Get or create global TreeSitterManager instance."""
    global _manager
    if _manager is None:
        _manager = TreeSitterManager()
    return _manager
