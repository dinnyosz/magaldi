"""JavaScript/TypeScript element extraction.

This module handles extraction of code elements from JavaScript and TypeScript:
- Classes and their decorators
- Functions (regular and async)
- Arrow functions
- Variables and constants
- TypeScript interfaces, type aliases, and enums
- Import statements as elements
"""

from __future__ import annotations

from tree_sitter import Node, Tree

from magaldi_core.extractors.base import (
    get_child_by_field,
    get_children_by_type,
    get_node_text,
    walk_tree,
)
from magaldi_core.extractors.types import (
    DecoratorInfo,
    ExtractedElement,
)
from magaldi_core.extractors.javascript.utils import (
    extract_js_parameters,
    extract_js_return_type,
    get_js_decorators,
    get_preceding_decorators,
    is_react_hook,
    REACT_WRAPPERS,
)


def extract_javascript_elements(
    tree: Tree, lines: list[str], file_path: str | None = None
) -> list[ExtractedElement]:
    """Extract code elements from a JavaScript/TypeScript AST.

    Args:
        tree: Parsed tree-sitter Tree.
        lines: Source code lines for raw code extraction.
        file_path: Optional file path for logging purposes.

    Returns:
        List of extracted elements.
    """
    elements: list[ExtractedElement] = []
    root = tree.root_node

    for node in walk_tree(root):
        if node.type == "class_declaration":
            # Check for decorators in parent node or as siblings
            decorators: list[str] | None = None
            decorator_details: list[DecoratorInfo] | None = None
            decorated_node: Node | None = None

            parent = node.parent
            if parent:
                # Check export_statement for decorators
                if parent.type == "export_statement":
                    has_decorators = any(c.type == "decorator" for c in parent.children)
                    if has_decorators:
                        decorators, decorator_details = get_js_decorators(parent)
                        decorated_node = parent
                # Check if class itself has decorator children (TypeScript)
                elif any(c.type == "decorator" for c in node.children):
                    decorators, decorator_details = get_js_decorators(node)
                    decorated_node = node
                # Check for decorated_definition wrapper or statement-level decorators
                elif parent.type in ("program", "statement_block"):
                    # Look for decorator siblings preceding this class
                    decorators, decorator_details = get_preceding_decorators(node, parent)
                    if decorators:
                        decorated_node = parent

            elements.append(_extract_js_class(
                node, lines, decorators, decorator_details, decorated_node
            ))
        elif node.type == "function_declaration":
            elements.append(_extract_js_function(node, lines))
        elif node.type == "lexical_declaration":
            # const/let declarations - extract arrow functions, React wrappers, and useful variables
            is_const = any(c.type == "const" for c in node.children)
            for decl in get_children_by_type(node, "variable_declarator"):
                name_node = get_child_by_field(decl, "name")
                value_node = get_child_by_field(decl, "value")
                name = get_node_text(name_node) if name_node else "unknown"
                if value_node and value_node.type == "arrow_function":
                    elements.append(_extract_js_arrow_function(decl, name, lines))
                elif value_node and value_node.type == "call_expression":
                    # Check for React wrapper patterns: memo(), forwardRef(), lazy()
                    elem = _extract_react_wrapped_component(decl, name, value_node, lines)
                    if elem:
                        elements.append(elem)
                    else:
                        # Not a React wrapper - extract as variable with usefulness filter
                        elem = _extract_js_variable(decl, name, value_node, lines, is_const)
                        if elem:
                            elements.append(elem)
                elif value_node:
                    # Other value types (literals, objects, arrays, etc.)
                    elem = _extract_js_variable(decl, name, value_node, lines, is_const)
                    if elem:
                        elements.append(elem)
        elif node.type == "variable_declaration":
            # var declarations (older style) - same logic as lexical_declaration
            for decl in get_children_by_type(node, "variable_declarator"):
                name_node = get_child_by_field(decl, "name")
                value_node = get_child_by_field(decl, "value")
                name = get_node_text(name_node) if name_node else "unknown"
                if value_node and value_node.type == "arrow_function":
                    elements.append(_extract_js_arrow_function(decl, name, lines))
                elif value_node:
                    elem = _extract_js_variable(decl, name, value_node, lines, is_const=False)
                    if elem:
                        elements.append(elem)
        # TypeScript-specific declarations
        elif node.type == "interface_declaration":
            elements.append(_extract_ts_interface(node, lines))
        elif node.type == "type_alias_declaration":
            elements.append(_extract_ts_type_alias(node, lines))
        elif node.type == "enum_declaration":
            elements.append(_extract_ts_enum(node, lines))
        # Import statements
        elif node.type == "import_statement":
            elements.append(_extract_js_import(node, lines))

    return elements


def _extract_js_class(
    node: Node,
    lines: list[str],
    decorators: list[str] | None = None,
    decorator_details: list[DecoratorInfo] | None = None,
    decorated_node: Node | None = None,
) -> ExtractedElement:
    """Extract a JavaScript class.

    Args:
        node: The class_declaration node.
        lines: Source code lines.
        decorators: List of decorator names (if any).
        decorator_details: Rich decorator info with args.
        decorated_node: The parent node containing decorators (e.g., export_statement).
    """
    name_node = get_child_by_field(node, "name")
    name = get_node_text(name_node) if name_node else "unknown"

    # Use decorated_node's start if available (to include decorators in raw_code)
    start_node = decorated_node if decorated_node else node
    line_start = start_node.start_point[0] + 1
    line_end = node.end_point[0] + 1
    # Use byte-based extraction for precise raw_code (handles minified files)
    raw_code = start_node.text.decode('utf-8') if start_node.text else ""

    return ExtractedElement(
        element_type="class",
        name=name,
        line_start=line_start,
        line_end=line_end,
        raw_code=raw_code,
        byte_offset=start_node.start_byte,
        node=node,
        decorators=decorators,
        decorator_details=decorator_details,
    )


def _extract_js_function(node: Node, lines: list[str]) -> ExtractedElement:
    """Extract a JavaScript function."""
    name_node = get_child_by_field(node, "name")
    name = get_node_text(name_node) if name_node else "unknown"

    params_node = get_child_by_field(node, "parameters")
    params = get_node_text(params_node) if params_node else "()"

    # Extract structured parameters and return type
    parameters = extract_js_parameters(params_node) if params_node else []
    return_type = extract_js_return_type(node)

    # Check for async
    is_async = any(child.type == "async" for child in node.children)

    line_start = node.start_point[0] + 1
    line_end = node.end_point[0] + 1
    raw_code = node.text.decode('utf-8') if node.text else ""

    signature = f"{'async ' if is_async else ''}function {name}{params}"
    if return_type:
        signature += f": {return_type}"

    # Check if this is a React hook
    decorators = ["hook"] if is_react_hook(name) else None
    decorator_details = [DecoratorInfo(name="hook", args=None, full="hook")] if decorators else None

    return ExtractedElement(
        element_type="function",
        name=name,
        line_start=line_start,
        line_end=line_end,
        raw_code=raw_code,
        byte_offset=node.start_byte,
        signature=signature,
        is_async=is_async,
        node=node,
        return_type=return_type,
        parameters=parameters or None,
        decorators=decorators,
        decorator_details=decorator_details,
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
    raw_code = node.text.decode('utf-8') if node.text else ""

    # Get the actual arrow function node for call extraction
    value_node = get_child_by_field(node, "value")
    arrow_func_node = value_node if value_node and value_node.type == "arrow_function" else node

    # Check if this is a React hook
    decorators = ["hook"] if is_react_hook(name) else None
    decorator_details = [DecoratorInfo(name="hook", args=None, full="hook")] if decorators else None

    return ExtractedElement(
        element_type="function",
        name=name,
        line_start=line_start,
        line_end=line_end,
        raw_code=raw_code,
        byte_offset=node.start_byte,
        node=arrow_func_node,
        decorators=decorators,
        decorator_details=decorator_details,
    )


def _extract_js_variable(
    decl_node: Node,
    name: str,
    value_node: Node,
    lines: list[str],
    is_const: bool = False,
) -> ExtractedElement | None:
    """Extract a JavaScript/TypeScript variable or constant.

    Extracts all variables without filtering — the LLM-based variable scoring
    phase (Phase 4) handles usefulness determination downstream.

    Args:
        decl_node: The variable_declarator node.
        name: The variable name.
        value_node: The value node (right-hand side of assignment).
        lines: Source code lines.
        is_const: Whether this is a const declaration.

    Returns:
        ExtractedElement for the variable.
    """
    line_start = decl_node.start_point[0] + 1
    value_type = value_node.type

    line_end = decl_node.end_point[0] + 1
    raw_code = decl_node.text.decode("utf-8") if decl_node.text else ""

    # Determine element type: constant if UPPER_CASE or const with literal
    if name.isupper() and len(name) > 1:
        elem_type = "constant"
    elif is_const and value_type in ("string", "number", "true", "false", "null"):
        elem_type = "constant"
    else:
        elem_type = "variable"

    return ExtractedElement(
        element_type=elem_type,
        name=name,
        line_start=line_start,
        line_end=line_end,
        raw_code=raw_code,
        byte_offset=decl_node.start_byte,
        node=decl_node,
    )


def _extract_react_wrapped_component(
    decl_node: Node, name: str, call_node: Node, lines: list[str]
) -> ExtractedElement | None:
    """Extract a React component wrapped in memo, forwardRef, or lazy.

    Handles patterns like:
        const Greeting = memo(function Greeting({ name }) {...});
        const MyInput = forwardRef(function MyInput(props, ref) {...});
        const LazyComponent = lazy(() => import('./Component'));

    Args:
        decl_node: The variable_declarator node.
        name: The variable name (component name).
        call_node: The call_expression node (e.g., memo(...)).
        lines: Source code lines.

    Returns:
        ExtractedElement if this is a React wrapper pattern, None otherwise.
    """
    # Get the wrapper function name (memo, forwardRef, etc.)
    func_node = get_child_by_field(call_node, "function")
    if not func_node:
        return None

    wrapper_name = get_node_text(func_node)
    if wrapper_name not in REACT_WRAPPERS:
        return None

    # Get the arguments to find the wrapped function
    args_node = get_child_by_field(call_node, "arguments")
    if not args_node:
        return None

    # Find the function/arrow_function in the arguments
    wrapped_func_node: Node | None = None
    for child in args_node.children:
        if child.type in ("function", "function_expression", "arrow_function"):
            wrapped_func_node = child
            break

    line_start = decl_node.start_point[0] + 1
    line_end = decl_node.end_point[0] + 1
    raw_code = decl_node.text.decode('utf-8') if decl_node.text else ""

    # Use the wrapped function node for call extraction if available
    node_for_calls = wrapped_func_node if wrapped_func_node else call_node

    # Add the wrapper as a decorator for discoverability
    decorator_info = DecoratorInfo(name=wrapper_name, args=None, full=wrapper_name)

    return ExtractedElement(
        element_type="function",
        name=name,
        line_start=line_start,
        line_end=line_end,
        raw_code=raw_code,
        byte_offset=decl_node.start_byte,
        node=node_for_calls,
        decorators=[wrapper_name],
        decorator_details=[decorator_info],
    )


# =============================================================================
# TYPESCRIPT-SPECIFIC ELEMENT EXTRACTION
# =============================================================================


def _extract_ts_interface(node: Node, lines: list[str]) -> ExtractedElement:
    """Extract a TypeScript interface declaration.

    Args:
        node: An interface_declaration node from tree-sitter.
        lines: Source code lines.

    Returns:
        ExtractedElement with element_type="interface".
    """
    name_node = get_child_by_field(node, "name")
    name = get_node_text(name_node) if name_node else "unknown"

    line_start = node.start_point[0] + 1
    line_end = node.end_point[0] + 1
    raw_code = node.text.decode('utf-8') if node.text else ""

    # Build signature with extends clause if present
    signature = f"interface {name}"
    for child in node.children:
        if child.type == "extends_type_clause":
            extends_text = get_node_text(child)
            signature += f" {extends_text}"
            break

    return ExtractedElement(
        element_type="interface",
        name=name,
        line_start=line_start,
        line_end=line_end,
        raw_code=raw_code,
        byte_offset=node.start_byte,
        signature=signature,
        node=node,
    )


def _extract_ts_type_alias(node: Node, lines: list[str]) -> ExtractedElement:
    """Extract a TypeScript type alias declaration.

    Args:
        node: A type_alias_declaration node from tree-sitter.
        lines: Source code lines.

    Returns:
        ExtractedElement with element_type="type_alias".
    """
    name_node = get_child_by_field(node, "name")
    name = get_node_text(name_node) if name_node else "unknown"

    line_start = node.start_point[0] + 1
    line_end = node.end_point[0] + 1
    raw_code = node.text.decode('utf-8') if node.text else ""

    # Get the type value for signature
    value_node = get_child_by_field(node, "value")
    value_text = get_node_text(value_node) if value_node else ""

    # Truncate long type definitions in signature
    if len(value_text) > 100:
        value_text = value_text[:97] + "..."

    signature = f"type {name} = {value_text}"

    return ExtractedElement(
        element_type="type_alias",
        name=name,
        line_start=line_start,
        line_end=line_end,
        raw_code=raw_code,
        byte_offset=node.start_byte,
        signature=signature,
        node=node,
    )


def _extract_ts_enum(node: Node, lines: list[str]) -> ExtractedElement:
    """Extract a TypeScript enum declaration.

    Args:
        node: An enum_declaration node from tree-sitter.
        lines: Source code lines.

    Returns:
        ExtractedElement with element_type="enum".
    """
    name_node = get_child_by_field(node, "name")
    name = get_node_text(name_node) if name_node else "unknown"

    line_start = node.start_point[0] + 1
    line_end = node.end_point[0] + 1
    raw_code = node.text.decode('utf-8') if node.text else ""

    # Check for const enum
    is_const = any(child.type == "const" for child in node.children)
    signature = f"{'const ' if is_const else ''}enum {name}"
    decorators = ["const"] if is_const else []

    return ExtractedElement(
        element_type="enum",
        name=name,
        line_start=line_start,
        line_end=line_end,
        raw_code=raw_code,
        byte_offset=node.start_byte,
        signature=signature,
        decorators=decorators if decorators else None,
        node=node,
    )


def _extract_js_import(node: Node, lines: list[str]) -> ExtractedElement:
    """Extract an import statement as an element.

    Args:
        node: An import_statement node from tree-sitter.
        lines: Source code lines.

    Returns:
        ExtractedElement with element_type="import".
    """
    line_start = node.start_point[0] + 1
    line_end = node.end_point[0] + 1
    raw_code = node.text.decode("utf-8") if node.text else ""

    # Get the module path (the 'from' part)
    module = ""
    for child in node.children:
        if child.type == "string":
            module = get_node_text(child).strip("'\"")
            break

    # Build a readable signature showing what's imported
    signature = raw_code.strip()

    return ExtractedElement(
        element_type="import",
        name=module,  # Use module path as the name for grouping
        line_start=line_start,
        line_end=line_end,
        raw_code=raw_code,
        byte_offset=node.start_byte,
        signature=signature,
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

            # Extract structured parameters and return type
            parameters = extract_js_parameters(params_node) if params_node else []
            return_type = extract_js_return_type(child)

            is_async = any(c.type == "async" for c in child.children)

            line_start = child.start_point[0] + 1
            line_end = child.end_point[0] + 1
            raw_code = child.text.decode('utf-8') if child.text else ""

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
                    byte_offset=child.start_byte,
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
            raw_code = child.text.decode('utf-8') if child.text else ""

            fields.append(
                ExtractedElement(
                    element_type="variable",
                    name=name,
                    line_start=line_start,
                    line_end=line_end,
                    raw_code=raw_code,
                    byte_offset=child.start_byte,
                    node=child,
                )
            )

    return methods, fields
