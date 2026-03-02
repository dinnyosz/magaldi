"""Python element extraction.

This module handles extraction of code elements from Python source code:
- Classes (including dataclasses, enums)
- Functions (regular and async)
- Methods
- Variables and constants
- Import statements as elements
- Nested functions
"""

from __future__ import annotations

import logging

from tree_sitter import Node, Tree

from magaldi_core.extractors.base import (
    get_child_by_field,
    get_children_by_type,
    get_node_text,
)
from magaldi_core.extractors.python.context_extractor import (
    extract_python_base_classes,
)
from magaldi_core.extractors.python.utils import (
    extract_python_parameters,
    get_decorators,
    is_inside_class,
    is_python_enum,
)
from magaldi_core.extractors.types import (
    DecoratorInfo,
    ExtractedElement,
)
from magaldi_core.query_runner import QUERIES_DIR

logger = logging.getLogger(__name__)


def extract_python_elements(
    tree: Tree, lines: list[str], _file_path: str | None = None
) -> list[ExtractedElement]:
    """Extract code elements from a Python AST using SCM queries.

    Uses declarative SCM patterns for matching, with Python post-processing
    to build ExtractedElement objects.

    Args:
        tree: Parsed tree-sitter Tree.
        lines: Source code lines for raw code extraction.
        file_path: Optional file path for logging purposes.

    Returns:
        List of extracted elements.
    """
    # Check if we have SCM queries available
    query_dir = QUERIES_DIR / "python"
    if not query_dir.exists() or not (query_dir / "elements.scm").exists():
        # Fall back to imperative extraction
        elements = _extract_python_elements_imperative(tree, lines)
    else:
        elements = _extract_python_elements_scm(tree, lines)

    return elements


def _extract_python_elements_scm(tree: Tree, lines: list[str]) -> list[ExtractedElement]:
    """Extract elements using SCM queries.

    This is the query-based implementation that uses declarative patterns.
    """
    from magaldi_core.tree_sitter_manager import get_manager

    elements: list[ExtractedElement] = []
    # Track by (start_byte, node_type) to avoid duplicates.
    # Note: Can't use id(node) because SCM queries create different node objects
    # for the same source position when matched by different patterns.
    seen_positions: set[tuple[int, str]] = set()

    manager = get_manager()
    result = manager.run_query_on_tree(tree, "python", "elements")

    # Process decorated definitions first (they include the definition inside)
    for match in result.filter_by_capture("decorated.def"):
        decorated_node = match.get("decorated.def")
        if not decorated_node:
            continue
        deco_pos = (decorated_node.start_byte, decorated_node.type)
        if deco_pos in seen_positions:
            continue

        func_node = match.get("decorated.function")
        class_node = match.get("decorated.class")

        if func_node:
            # Skip if this decorated function is inside a class —
            # class member extractor handles decorated methods
            if is_inside_class(func_node):
                seen_positions.add((func_node.start_byte, func_node.type))
                seen_positions.add(deco_pos)
                continue

            seen_positions.add((func_node.start_byte, func_node.type))
            seen_positions.add(deco_pos)
            deco_names, deco_details = get_decorators(decorated_node)
            elem = _extract_python_function(
                func_node, lines,
                decorators=deco_names,
                decorator_details=deco_details,
                decorated_node=decorated_node
            )
            elements.append(elem)
            # Extract nested functions and track their positions
            nested = _extract_nested_functions(func_node, lines)
            for nested_elem in nested:
                if nested_elem.node:
                    seen_positions.add((nested_elem.node.start_byte, nested_elem.node.type))
            elements.extend(nested)

        elif class_node:
            seen_positions.add((class_node.start_byte, class_node.type))
            seen_positions.add(deco_pos)
            deco_names, deco_details = get_decorators(decorated_node)
            elements.append(_extract_python_class(
                class_node, lines,
                decorators=deco_names,
                decorator_details=deco_details,
                decorated_node=decorated_node
            ))

    # Process standalone functions (not already seen as decorated)
    for match in result.filter_by_capture("function.def"):
        func_node = match.get("function.def")
        if not func_node:
            continue
        func_pos = (func_node.start_byte, func_node.type)
        if func_pos in seen_positions:
            continue

        # Skip if this is inside a class (methods are handled separately)
        if is_inside_class(func_node):
            continue

        seen_positions.add(func_pos)
        elem = _extract_python_function(func_node, lines)
        elements.append(elem)
        # Extract nested functions and track their positions
        nested = _extract_nested_functions(func_node, lines)
        for nested_elem in nested:
            if nested_elem.node:
                seen_positions.add((nested_elem.node.start_byte, nested_elem.node.type))
        elements.extend(nested)

    # Process async functions (not already seen)
    for match in result.filter_by_capture("async_function.def"):
        func_node = match.get("async_function.def")
        if not func_node:
            continue
        func_pos = (func_node.start_byte, func_node.type)
        if func_pos in seen_positions:
            continue

        if is_inside_class(func_node):
            continue

        seen_positions.add(func_pos)
        elem = _extract_python_function(func_node, lines)
        elements.append(elem)
        # Extract nested functions and track their positions
        nested = _extract_nested_functions(func_node, lines)
        for nested_elem in nested:
            if nested_elem.node:
                seen_positions.add((nested_elem.node.start_byte, nested_elem.node.type))
        elements.extend(nested)

    # Process standalone classes (not already seen as decorated)
    for match in result.filter_by_capture("class.def"):
        class_node = match.get("class.def")
        if not class_node:
            continue
        class_pos = (class_node.start_byte, class_node.type)
        if class_pos in seen_positions:
            continue

        seen_positions.add(class_pos)
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

    # Extract variables from inside function bodies (all depths)
    for elem in list(elements):
        if elem.element_type in ("function", "method") and elem.node:
            body_vars = _extract_function_body_variables(elem.node, lines)
            elements.extend(body_vars)

    # Process import statements (simple top-level walk, no SCM query needed)
    for node in tree.root_node.children:
        if node.type in ("import_statement", "import_from_statement"):
            elements.append(_extract_python_import(node, lines))

    return elements


def _extract_python_elements_imperative(tree: Tree, lines: list[str]) -> list[ExtractedElement]:
    """Extract elements using imperative tree-walking (fallback).

    This is the original implementation kept for backwards compatibility.
    """
    elements: list[ExtractedElement] = []
    root = tree.root_node

    # Extract imports and code elements
    for node in root.children:
        # Import statements
        if node.type in ("import_statement", "import_from_statement"):
            elements.append(_extract_python_import(node, lines))
        elif node.type == "class_definition":
            elements.append(_extract_python_class(node, lines))
        elif node.type == "decorated_definition":
            inner = get_child_by_field(node, "definition")
            if inner and inner.type == "class_definition":
                deco_names, deco_details = get_decorators(node)
                elements.append(
                    _extract_python_class(inner, lines, decorators=deco_names, decorator_details=deco_details, decorated_node=node)
                )
            elif inner and inner.type == "function_definition":
                deco_names, deco_details = get_decorators(node)
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

    # Extract variables from inside function bodies (all depths)
    for elem in list(elements):
        if elem.element_type in ("function", "method") and elem.node:
            body_vars = _extract_function_body_variables(elem.node, lines)
            elements.extend(body_vars)

    return elements


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
                deco_names, deco_details = get_decorators(child)
                elem = _extract_python_function(
                    inner, lines, decorators=deco_names, decorator_details=deco_details,
                    decorated_node=child, is_nested=True
                )
                elem.parent_node = func_node
                nested.append(elem)
                # Recursively extract nested functions
                nested.extend(_extract_nested_functions(inner, lines))

    return nested


def _extract_python_import(node: Node, lines: list[str]) -> ExtractedElement:  # noqa: ARG001
    """Extract an import statement as an element.

    Args:
        node: An import_statement or import_from_statement node.
        lines: Source code lines.

    Returns:
        ExtractedElement with element_type="import".
    """
    line_start = node.start_point[0] + 1
    line_end = node.end_point[0] + 1
    raw_code = node.text.decode("utf-8") if node.text else ""

    # Get module name based on import type
    if node.type == "import_statement":
        # import os, sys -> use first module name
        for child in node.children:
            if child.type == "dotted_name":
                module = get_node_text(child)
                break
            elif child.type == "aliased_import":
                name_node = get_child_by_field(child, "name")
                module = get_node_text(name_node) if name_node else ""
                break
        else:
            module = ""
    else:  # import_from_statement
        # from flask import ... -> use "flask"
        module = ""
        for child in node.children:
            if child.type == "dotted_name":
                module = get_node_text(child)
                break
            elif child.type == "relative_import":
                # from . import x or from ..utils import y
                module = get_node_text(child)
                break

    return ExtractedElement(
        element_type="import",
        name=module,
        line_start=line_start,
        line_end=line_end,
        raw_code=raw_code,
        byte_offset=node.start_byte,
        signature=raw_code.strip(),
        node=node,
    )


def _extract_python_class(
    node: Node,
    lines: list[str],  # noqa: ARG001
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
    # Use byte-based extraction for precise raw_code (handles minified files)
    raw_code = start_node.text.decode('utf-8') if start_node.text else ""

    # Check if this is an Enum class
    base_classes = extract_python_base_classes(node)
    element_type = "class"
    if is_python_enum(base_classes):
        element_type = "enum"

    elem = ExtractedElement(
        element_type=element_type,
        name=name,
        line_start=line_start,
        line_end=line_end,
        raw_code=raw_code,
        byte_offset=start_node.start_byte,
        decorators=decorators,
        decorator_details=decorator_details,
        node=node,
    )

    return elem


def _extract_python_function(
    node: Node,
    lines: list[str],  # noqa: ARG001
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
    parameters = extract_python_parameters(params_node) if params_node else []

    # Check for async
    is_async = any(child.type == "async" for child in node.children)

    # Use decorated_node's start if available (to include decorators in raw_code)
    start_node = decorated_node if decorated_node else node
    line_start = start_node.start_point[0] + 1
    line_end = node.end_point[0] + 1
    # Use byte-based extraction for precise raw_code (handles minified files)
    raw_code = start_node.text.decode('utf-8') if start_node.text else ""

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
        byte_offset=start_node.start_byte,
        signature=signature,
        decorators=decorators,
        decorator_details=decorator_details,
        is_async=is_async,
        node=node,
        return_type=return_type,
        parameters=parameters or None,
    )


def _extract_function_body_variables(
    func_node: Node, lines: list[str]
) -> list[ExtractedElement]:
    """Extract variable assignments from inside a function body.

    Recursively walks the function body to find all assignments,
    including those nested inside if/for/with/try blocks.

    Args:
        func_node: A function_definition node.
        lines: Source code lines.

    Returns:
        List of variable elements found inside the function.
    """
    from magaldi_core.extractors.base import walk_tree

    variables: list[ExtractedElement] = []
    body_node = get_child_by_field(func_node, "body")
    if not body_node:
        return variables

    for node in walk_tree(body_node):
        if node.type == "assignment":
            # Skip assignments inside nested function/class definitions
            parent = node.parent
            in_nested_scope = False
            while parent and parent != body_node:
                if parent.type in ("function_definition", "class_definition"):
                    in_nested_scope = True
                    break
                parent = parent.parent
            if in_nested_scope:
                continue

            elem = _extract_python_assignment(node, lines, is_module_level=False)
            if elem:
                variables.append(elem)

    return variables


def _extract_python_assignment(
    node: Node,
    lines: list[str],  # noqa: ARG001
    is_module_level: bool = False,
    parent_class: Node | None = None,
) -> ExtractedElement | None:
    """Extract a variable/constant assignment.

    Extracts all variables without filtering — the LLM-based variable scoring
    phase (Phase 4) handles usefulness determination downstream.
    """
    left_node = get_child_by_field(node, "left")
    if not left_node or left_node.type != "identifier":
        return None

    name = get_node_text(left_node)

    # Get the right-hand side value
    right_node = get_child_by_field(node, "right")
    if not right_node:
        return None

    line_start = node.start_point[0] + 1
    line_end = node.end_point[0] + 1
    # Use byte-based extraction for precise raw_code (handles minified files)
    raw_code = node.text.decode('utf-8') if node.text else ""

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
        byte_offset=node.start_byte,
        parent_node=parent_class,
        node=node,
    )


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
                deco_names, deco_details = get_decorators(child)
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
