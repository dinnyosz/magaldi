"""Tree-sitter parser manager for multiple languages.

This module provides:
- Lazy loading of tree-sitter language parsers
- Unified interface for parsing source code into ASTs
- Tree traversal utilities for extracting code elements
"""

from __future__ import annotations

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
# GLOBAL SINGLETON
# =============================================================================

_manager: TreeSitterManager | None = None


def get_manager() -> TreeSitterManager:
    """Get or create global TreeSitterManager instance."""
    global _manager
    if _manager is None:
        _manager = TreeSitterManager()
    return _manager
