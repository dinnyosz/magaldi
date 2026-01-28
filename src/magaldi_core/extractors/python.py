"""Python language extractor using tree-sitter.

This module provides the PythonExtractor class and standalone functions
for extracting code elements, imports, references, call graph data,
and enhanced context from Python source code.

Uses SCM queries for pattern matching with Python post-processing.
"""

from __future__ import annotations

from tree_sitter import Node, Tree

from magaldi_core.extractors.base import (
    BaseExtractor,
    get_child_by_field,
    get_children_by_type,
    get_node_text,
    walk_tree,
)
from magaldi_core.extractors.types import (
    DecoratorInfo,
    ExtractedCall,
    ExtractedElement,
    ExtractedImport,
    ExtractedReference,
    ParameterInfo,
)
from magaldi_core.query_runner import QUERIES_DIR

# =============================================================================
# PYTHON EXTRACTOR CLASS
# =============================================================================


class PythonExtractor(BaseExtractor):
    """Extractor for Python source code."""

    @property
    def language(self) -> str:
        return "python"

    def extract_elements(self, tree: Tree, lines: list[str]) -> list[ExtractedElement]:
        """Extract code elements from a Python AST."""
        return extract_python_elements(tree, lines)

    def extract_class_members(
        self, class_node: Node, lines: list[str]
    ) -> tuple[list[ExtractedElement], list[ExtractedElement]]:
        """Extract methods and class variables from a Python class."""
        return extract_python_class_members(class_node, lines)

    def extract_imports(self, tree: Tree, lines: list[str]) -> list[ExtractedImport]:
        """Extract import statements from Python AST."""
        return extract_python_imports(tree, lines)

    def extract_references(
        self, tree: Tree, lines: list[str]
    ) -> list[ExtractedReference]:
        """Extract all references from Python AST."""
        return extract_python_references(tree, lines)

    def extract_calls(self, function_node: Node) -> list[ExtractedCall]:
        """Extract function/method calls from within a Python function body."""
        return extract_python_calls(function_node)

    def extract_class_attributes(
        self, class_node: Node
    ) -> list[dict[str, str | int]]:
        """Extract instance attributes from __init__."""
        return extract_python_class_attributes(class_node)

    def extract_base_classes(self, class_node: Node) -> list[str]:
        """Extract base class names from Python class definition."""
        return extract_python_base_classes(class_node)

    def extract_raised_exceptions(self, function_node: Node) -> list[str]:
        """Extract exception types from raise statements."""
        return extract_python_raised_exceptions(function_node)

    def extract_modified_attributes(self, method_node: Node) -> list[str]:
        """Extract self.X attributes that are modified."""
        return extract_python_modified_attributes(method_node)


# =============================================================================
# PYTHON ELEMENT EXTRACTION
# =============================================================================


def _extract_nested_functions(func_node: Node, lines: list[str]) -> list[ExtractedElement]:
    """Extract nested function definitions from within a function body.

    Args:
        func_node: A function_definition node.
        lines: Source code lines.

    Returns:
        List of nested function elements.
    """
    nested: list[ExtractedElement] = []
    body_node = get_child_by_field(func_node, "body")
    if not body_node:
        return nested

    for child in body_node.children:
        if child.type == "function_definition":
            elem = _extract_python_function(child, lines, is_nested=True)
            elem.parent_node = func_node
            nested.append(elem)
            # Recursively extract nested functions from this nested function
            nested.extend(_extract_nested_functions(child, lines))
        elif child.type == "decorated_definition":
            inner = get_child_by_field(child, "definition")
            if inner and inner.type == "function_definition":
                deco_names, deco_details = _get_decorators(child)
                elem = _extract_python_function(
                    inner, lines, decorators=deco_names, decorator_details=deco_details,
                    decorated_node=child, is_nested=True
                )
                elem.parent_node = func_node
                nested.append(elem)
                # Recursively extract nested functions
                nested.extend(_extract_nested_functions(inner, lines))

    return nested


def extract_python_elements(tree: Tree, lines: list[str]) -> list[ExtractedElement]:
    """Extract code elements from a Python AST using SCM queries.

    Uses declarative SCM patterns for matching, with Python post-processing
    to build ExtractedElement objects.

    Args:
        tree: Parsed tree-sitter Tree.
        lines: Source code lines for raw code extraction.

    Returns:
        List of extracted elements.
    """
    # Check if we have SCM queries available
    query_dir = QUERIES_DIR / "python"
    if not query_dir.exists() or not (query_dir / "elements.scm").exists():
        # Fall back to imperative extraction
        return _extract_python_elements_imperative(tree, lines)

    return _extract_python_elements_scm(tree, lines)


def _extract_python_elements_scm(tree: Tree, lines: list[str]) -> list[ExtractedElement]:
    """Extract elements using SCM queries.

    This is the query-based implementation that uses declarative patterns.
    """
    from magaldi_core.tree_sitter_manager import get_manager

    elements: list[ExtractedElement] = []
    seen_nodes: set[int] = set()  # Track by node id to avoid duplicates

    manager = get_manager()
    result = manager.run_query_on_tree(tree, "python", "elements")

    # Process decorated definitions first (they include the definition inside)
    for match in result.filter_by_capture("decorated.def"):
        decorated_node = match.get("decorated.def")
        if not decorated_node or id(decorated_node) in seen_nodes:
            continue

        func_node = match.get("decorated.function")
        class_node = match.get("decorated.class")

        if func_node:
            seen_nodes.add(id(func_node))
            seen_nodes.add(id(decorated_node))
            deco_names, deco_details = _get_decorators(decorated_node)
            elem = _extract_python_function(
                func_node, lines,
                decorators=deco_names,
                decorator_details=deco_details,
                decorated_node=decorated_node
            )
            elements.append(elem)
            # Extract nested functions
            elements.extend(_extract_nested_functions(func_node, lines))

        elif class_node:
            seen_nodes.add(id(class_node))
            seen_nodes.add(id(decorated_node))
            deco_names, deco_details = _get_decorators(decorated_node)
            elements.append(_extract_python_class(
                class_node, lines,
                decorators=deco_names,
                decorator_details=deco_details,
                decorated_node=decorated_node
            ))

    # Process standalone functions (not already seen as decorated)
    for match in result.filter_by_capture("function.def"):
        func_node = match.get("function.def")
        if not func_node or id(func_node) in seen_nodes:
            continue

        # Skip if this is inside a class (methods are handled separately)
        if _is_inside_class(func_node):
            continue

        seen_nodes.add(id(func_node))
        elem = _extract_python_function(func_node, lines)
        elements.append(elem)
        elements.extend(_extract_nested_functions(func_node, lines))

    # Process async functions (not already seen)
    for match in result.filter_by_capture("async_function.def"):
        func_node = match.get("async_function.def")
        if not func_node or id(func_node) in seen_nodes:
            continue

        if _is_inside_class(func_node):
            continue

        seen_nodes.add(id(func_node))
        elem = _extract_python_function(func_node, lines)
        elements.append(elem)
        elements.extend(_extract_nested_functions(func_node, lines))

    # Process standalone classes (not already seen as decorated)
    for match in result.filter_by_capture("class.def"):
        class_node = match.get("class.def")
        if not class_node or id(class_node) in seen_nodes:
            continue

        seen_nodes.add(id(class_node))
        elements.append(_extract_python_class(class_node, lines))

    # Process module-level assignments
    for match in result.filter_by_capture("assignment.module_level"):
        name_node = match.get("assignment.name")
        if not name_node:
            continue

        # Find the actual assignment node
        assign_node = name_node.parent
        if assign_node and assign_node.type == "assignment":
            elem = _extract_python_assignment(assign_node, lines, is_module_level=True)
            if elem:
                elements.append(elem)

    return elements


def _is_inside_class(node: Node) -> bool:
    """Check if a node is inside a class definition (i.e., it's a method)."""
    current = node.parent
    while current:
        if current.type == "class_definition":
            return True
        if current.type == "function_definition":
            # Nested inside a function, not a class
            return False
        current = current.parent
    return False


def _extract_python_elements_imperative(tree: Tree, lines: list[str]) -> list[ExtractedElement]:
    """Extract elements using imperative tree-walking (fallback).

    This is the original implementation kept for backwards compatibility.
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
                func_elem = _extract_python_function(inner, lines, decorators=deco_names, decorator_details=deco_details, decorated_node=node)
                elements.append(func_elem)
                # Extract nested functions
                elements.extend(_extract_nested_functions(inner, lines))
        elif node.type == "function_definition":
            func_elem = _extract_python_function(node, lines)
            elements.append(func_elem)
            # Extract nested functions
            elements.extend(_extract_nested_functions(node, lines))
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
    is_nested: bool = False,  # noqa: ARG001
) -> ExtractedElement:
    """Extract a function/method definition.

    Args:
        node: The function_definition node.
        lines: Source code lines.
        decorators: List of decorator names (if any).
        is_method: Whether this is a method (vs standalone function).
        decorated_node: The outer decorated_definition node (if function is decorated).
                       Used to include decorator lines in raw_code.
        is_nested: Whether this is a nested function (kept for future use).
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


def extract_python_imports(
    tree: Tree, lines: list[str]  # noqa: ARG001
) -> list[ExtractedImport]:
    """Extract import statements from a Python AST.

    Handles:
    - import os -> Import(name="os", module="os", alias=None)
    - import pandas as pd -> Import(name="pandas", module="pandas", alias="pd")
    - from utils import process -> Import(name="process", module="utils", alias=None)
    - from utils import process as p -> Import(name="process", module="utils", alias="p")

    Args:
        tree: Parsed tree-sitter Tree.
        lines: Source code lines (unused but kept for API consistency).

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
