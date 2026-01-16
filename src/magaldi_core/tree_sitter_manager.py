"""Tree-sitter parser manager for multiple languages.

This module provides:
- Lazy loading of tree-sitter language parsers
- Unified interface for parsing source code into ASTs
- Tree traversal utilities for extracting code elements
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

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
class ExtractedElement:
    """A code element extracted from the AST."""

    element_type: str  # 'class', 'function', 'method', 'constant', 'variable'
    name: str
    line_start: int  # 1-indexed
    line_end: int  # 1-indexed
    raw_code: str
    signature: str | None = None
    decorators: list[str] | None = None
    is_async: bool = False
    parent_node: Node | None = None  # For tracking hierarchy
    node: Node | None = None  # The AST node itself


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
                elements.append(
                    _extract_python_class(inner, lines, decorators=_get_decorators(node), decorated_node=node)
                )
            elif inner and inner.type == "function_definition":
                elements.append(
                    _extract_python_function(inner, lines, decorators=_get_decorators(node), decorated_node=node)
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
    node: Node, lines: list[str], decorators: list[str] | None = None, decorated_node: Node | None = None
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
        node=node,
    )

    return elem


def _extract_python_function(
    node: Node,
    lines: list[str],
    decorators: list[str] | None = None,
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
        is_async=is_async,
        node=node,
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


def _get_decorators(decorated_node: Node) -> list[str]:
    """Extract decorator names from a decorated_definition node.

    Handles both simple decorators (@foo) and call decorators (@foo.bar(...)).
    """
    decorators = []
    for child in decorated_node.children:
        if child.type == "decorator":
            # Get the name part (after @, before ())
            for deco_child in child.children:
                if deco_child.type == "identifier":
                    # Simple decorator: @foo
                    decorators.append(get_node_text(deco_child))
                    break
                elif deco_child.type == "attribute":
                    # Attribute decorator: @foo.bar
                    decorators.append(get_node_text(deco_child))
                    break
                elif deco_child.type == "call":
                    # Call decorator: @foo(...) or @foo.bar(...)
                    func_node = get_child_by_field(deco_child, "function")
                    if func_node:
                        decorators.append(get_node_text(func_node))
                    break
    return decorators


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
                methods.append(
                    _extract_python_function(
                        inner, lines, decorators=_get_decorators(child), is_method=True, decorated_node=child
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


def _extract_js_function(node: Node, lines: list[str]) -> ExtractedElement:
    """Extract a JavaScript function."""
    name_node = get_child_by_field(node, "name")
    name = get_node_text(name_node) if name_node else "unknown"

    params_node = get_child_by_field(node, "parameters")
    params = get_node_text(params_node) if params_node else "()"

    # Check for async
    is_async = any(child.type == "async" for child in node.children)

    line_start = node.start_point[0] + 1
    line_end = node.end_point[0] + 1
    raw_code = "\n".join(lines[line_start - 1 : line_end])

    signature = f"{'async ' if is_async else ''}function {name}{params}"

    return ExtractedElement(
        element_type="function",
        name=name,
        line_start=line_start,
        line_end=line_end,
        raw_code=raw_code,
        signature=signature,
        is_async=is_async,
        node=node,
    )


def _extract_js_arrow_function(node: Node, name: str, lines: list[str]) -> ExtractedElement:
    """Extract a JavaScript arrow function."""
    line_start = node.start_point[0] + 1
    line_end = node.end_point[0] + 1
    raw_code = "\n".join(lines[line_start - 1 : line_end])

    return ExtractedElement(
        element_type="function",
        name=name,
        line_start=line_start,
        line_end=line_end,
        raw_code=raw_code,
        node=node,
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

            is_async = any(c.type == "async" for c in child.children)

            line_start = child.start_point[0] + 1
            line_end = child.end_point[0] + 1
            raw_code = "\n".join(lines[line_start - 1 : line_end])

            methods.append(
                ExtractedElement(
                    element_type="method",
                    name=name,
                    line_start=line_start,
                    line_end=line_end,
                    raw_code=raw_code,
                    signature=f"{name}{params}",
                    is_async=is_async,
                    node=child,
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
# GLOBAL SINGLETON
# =============================================================================

_manager: TreeSitterManager | None = None


def get_manager() -> TreeSitterManager:
    """Get or create global TreeSitterManager instance."""
    global _manager
    if _manager is None:
        _manager = TreeSitterManager()
    return _manager
