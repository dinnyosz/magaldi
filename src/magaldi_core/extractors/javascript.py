"""JavaScript/TypeScript language extractor using tree-sitter.

This module provides the JavaScriptExtractor class and standalone functions
for extracting code elements, imports, references, call graph data,
and enhanced context from JavaScript and TypeScript source code.
"""

from __future__ import annotations

import logging

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
from magaldi_core.extractors.usefulness_filter import (
    ExtractionStats,
    get_extraction_stats,
    reset_extraction_stats,
    should_skip_variable,
    SKIP_NAMES,
)

logger = logging.getLogger(__name__)

# =============================================================================
# JAVASCRIPT EXTRACTOR CLASS
# =============================================================================


class JavaScriptExtractor(BaseExtractor):
    """Extractor for JavaScript/TypeScript source code."""

    def __init__(self, language: str = "javascript"):
        self._language = language

    @property
    def language(self) -> str:
        return self._language

    def extract_elements(self, tree: Tree, lines: list[str]) -> list[ExtractedElement]:
        """Extract code elements from a JavaScript/TypeScript AST."""
        return extract_javascript_elements(tree, lines)

    def extract_class_members(
        self, class_node: Node, lines: list[str]
    ) -> tuple[list[ExtractedElement], list[ExtractedElement]]:
        """Extract methods and class fields from a JavaScript class."""
        return extract_javascript_class_members(class_node, lines)

    def extract_imports(self, tree: Tree, lines: list[str]) -> list[ExtractedImport]:
        """Extract import statements from JavaScript/TypeScript AST."""
        return extract_javascript_imports(tree, lines)

    def extract_references(
        self, tree: Tree, lines: list[str]
    ) -> list[ExtractedReference]:
        """Extract all references from JavaScript/TypeScript AST."""
        return extract_javascript_references(tree, lines)

    def extract_calls(self, function_node: Node) -> list[ExtractedCall]:
        """Extract function/method calls from within a JavaScript function body."""
        return extract_javascript_calls(function_node)

    def extract_class_attributes(
        self, class_node: Node
    ) -> list[dict[str, str | int]]:
        """Extract class fields from JavaScript/TypeScript class."""
        return extract_javascript_class_fields(class_node)

    def extract_base_classes(self, class_node: Node) -> list[str]:
        """Extract base class name from JavaScript class definition."""
        return extract_javascript_base_class(class_node)

    def extract_raised_exceptions(self, function_node: Node) -> list[str]:
        """Extract exception types from throw statements."""
        return extract_javascript_thrown_exceptions(function_node)

    def extract_modified_attributes(self, method_node: Node) -> list[str]:
        """Extract this.X properties that are modified."""
        return extract_javascript_modified_properties(method_node)


def _is_likely_class_name(name: str) -> bool:
    """Check if name looks like a class (PascalCase)."""
    if not name:
        return False
    # Starts with uppercase, not all uppercase (to exclude constants like HTTP)
    return name[0].isupper() and not name.isupper()


# =============================================================================
# JAVASCRIPT ELEMENT EXTRACTION
# =============================================================================


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
    # Reset stats for this file
    reset_extraction_stats()

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
                        decorators, decorator_details = _get_js_decorators(parent)
                        decorated_node = parent
                # Check if class itself has decorator children (TypeScript)
                elif any(c.type == "decorator" for c in node.children):
                    decorators, decorator_details = _get_js_decorators(node)
                    decorated_node = node
                # Check for decorated_definition wrapper or statement-level decorators
                elif parent.type in ("program", "statement_block"):
                    # Look for decorator siblings preceding this class
                    decorators, decorator_details = _get_preceding_decorators(node, parent)
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

    # Log extraction stats if we have any skipped variables
    stats = reset_extraction_stats()
    if stats and stats.skipped_count > 0:
        stats.log_summary(file_path or "<unknown>")

    return elements


def _get_js_decorators(parent_node: Node) -> tuple[list[str], list[DecoratorInfo]]:
    """Extract decorator names and details from TypeScript/JavaScript decorators.

    In TS/JS, decorators are siblings of the class/function inside an export_statement
    or directly preceding the declaration.

    Args:
        parent_node: The parent node (typically export_statement) containing decorators.

    Returns:
        Tuple of (decorator_names, decorator_details).
    """
    decorators: list[str] = []
    details: list[DecoratorInfo] = []

    for child in parent_node.children:
        if child.type == "decorator":
            # Get the full decorator text (excluding @)
            full_text = get_node_text(child)
            if full_text.startswith("@"):
                full_text = full_text[1:].strip()

            # Find the decorator content (identifier or call_expression)
            for deco_child in child.children:
                if deco_child.type == "@":
                    continue
                elif deco_child.type == "identifier":
                    # Simple decorator: @Injectable
                    name = get_node_text(deco_child)
                    decorators.append(name)
                    details.append(DecoratorInfo(name=name, args=None, full=full_text))
                    break
                elif deco_child.type == "call_expression":
                    # Call decorator: @Controller('cats') or @UseGuards(AuthGuard)
                    func_node = get_child_by_field(deco_child, "function")
                    args_node = get_child_by_field(deco_child, "arguments")
                    if func_node:
                        name = get_node_text(func_node)
                        args = get_node_text(args_node) if args_node else None
                        decorators.append(name)
                        details.append(DecoratorInfo(name=name, args=args, full=full_text))
                    break
                elif deco_child.type == "member_expression":
                    # Member decorator: @Module.forRoot()
                    name = get_node_text(deco_child)
                    decorators.append(name)
                    details.append(DecoratorInfo(name=name, args=None, full=full_text))
                    break

    return decorators, details


def _get_preceding_decorators(
    node: Node, parent: Node
) -> tuple[list[str] | None, list[DecoratorInfo] | None]:
    """Extract decorators that precede a class/function declaration.

    In TypeScript, decorators can be siblings preceding the class at statement level:
    @Injectable()
    class AuthService { }

    Args:
        node: The class/function declaration node.
        parent: The parent node (program or statement_block).

    Returns:
        Tuple of (decorator_names, decorator_details) or (None, None) if no decorators.
    """
    # Find the index of our node in parent's children
    node_index = -1
    for i, child in enumerate(parent.children):
        if child is node or child.id == node.id:
            node_index = i
            break

    if node_index <= 0:
        return None, None

    # Look backwards for consecutive decorators
    decorators: list[str] = []
    details: list[DecoratorInfo] = []

    for i in range(node_index - 1, -1, -1):
        sibling = parent.children[i]
        if sibling.type == "decorator":
            full_text = get_node_text(sibling)
            if full_text.startswith("@"):
                full_text = full_text[1:].strip()

            # Extract decorator info
            for deco_child in sibling.children:
                if deco_child.type == "@":
                    continue
                elif deco_child.type == "identifier":
                    name = get_node_text(deco_child)
                    decorators.insert(0, name)  # Insert at beginning to preserve order
                    details.insert(0, DecoratorInfo(name=name, args=None, full=full_text))
                    break
                elif deco_child.type == "call_expression":
                    func_node = get_child_by_field(deco_child, "function")
                    args_node = get_child_by_field(deco_child, "arguments")
                    if func_node:
                        name = get_node_text(func_node)
                        args = get_node_text(args_node) if args_node else None
                        decorators.insert(0, name)
                        details.insert(0, DecoratorInfo(name=name, args=args, full=full_text))
                    break
                elif deco_child.type == "member_expression":
                    name = get_node_text(deco_child)
                    decorators.insert(0, name)
                    details.insert(0, DecoratorInfo(name=name, args=None, full=full_text))
                    break
        else:
            # Stop when we hit a non-decorator
            break

    if decorators:
        return decorators, details
    return None, None


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
    raw_code = node.text.decode('utf-8') if node.text else ""

    signature = f"{'async ' if is_async else ''}function {name}{params}"
    if return_type:
        signature += f": {return_type}"

    # Check if this is a React hook
    decorators = ["hook"] if _is_react_hook(name) else None
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
    decorators = ["hook"] if _is_react_hook(name) else None
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

    Applies usefulness filter to skip transient variables like instance
    creations and function call results.

    Args:
        decl_node: The variable_declarator node.
        name: The variable name.
        value_node: The value node (right-hand side of assignment).
        lines: Source code lines.
        is_const: Whether this is a const declaration.

    Returns:
        ExtractedElement if the variable is useful, None otherwise.
    """
    # Skip short/temp names early
    if name in SKIP_NAMES:
        return None

    line_start = decl_node.start_point[0] + 1
    value_type = value_node.type
    value_text = value_node.text.decode("utf-8") if value_node.text else ""

    # Determine function name if this is a call
    func_name: str | None = None
    if value_type == "call_expression":
        func_node = get_child_by_field(value_node, "function")
        if func_node:
            func_name = get_node_text(func_node)
    elif value_type == "new_expression":
        # For new expressions, the class name is in the "constructor" field
        constructor_node = get_child_by_field(value_node, "constructor")
        if constructor_node:
            func_name = get_node_text(constructor_node)

    # Apply usefulness filter
    # Use "typescript" for language since JS and TS have same AST types
    skip, reason = should_skip_variable(
        name=name,
        value_type=value_type,
        func_name=func_name,
        language="typescript",
        is_module_level=True,  # JS doesn't have module-level distinction in our context
    )

    if skip:
        # Record the skip for reporting
        stats = get_extraction_stats()
        stats.record_skip(name, line_start, reason, value_type, value_text)
        return None

    # Record that we kept this one
    stats = get_extraction_stats()
    stats.record_keep()

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


# React wrapper functions that wrap components
REACT_WRAPPERS = {"memo", "forwardRef", "lazy", "React.memo", "React.forwardRef", "React.lazy"}


def _is_react_hook(name: str) -> bool:
    """Check if a function name follows React hook naming convention.

    React hooks must start with "use" followed by an uppercase letter.
    Examples: useCounter, useState, useWindowSize
    Non-hooks: useless, user, use (too short)
    """
    if len(name) <= 3:
        return False
    return name.startswith("use") and name[3].isupper()


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
            parameters = _extract_js_parameters(params_node) if params_node else []
            return_type = _extract_js_return_type(child)

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


# =============================================================================
# JAVASCRIPT IMPORT EXTRACTION
# =============================================================================


def extract_javascript_imports(
    tree: Tree, lines: list[str]  # noqa: ARG001
) -> list[ExtractedImport]:
    """Extract import statements from a JavaScript/TypeScript AST.

    Handles:
    - import { foo } from './utils' -> Import(name="foo", module="./utils", alias=None)
    - import { foo as bar } from './utils' -> Import(name="foo", module="./utils", alias="bar")
    - import utils from './utils' -> Import(name="utils", module="./utils", alias=None)
    - import * as utils from './utils' -> Import(name="*", module="./utils", alias="utils")
    - const bar = require('lib') -> Import(name="bar", module="lib", alias=None)

    Args:
        tree: Parsed tree-sitter Tree.
        lines: Source code lines (unused but kept for API consistency).

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
    """Extract base classes and implemented interfaces from a JavaScript/TypeScript class.

    For: class Foo extends Bar implements IBaz, IQux { ... }
    Returns: ["Bar", "IBaz", "IQux"]

    Args:
        class_node: A class_declaration node from tree-sitter.

    Returns:
        List of base class and interface names.
    """
    bases: list[str] = []

    if class_node.type != "class_declaration":
        return bases

    for child in class_node.children:
        # Handle class_heritage which contains extends and implements clauses
        if child.type == "class_heritage":
            for heritage_child in child.children:
                # Simple identifier (JavaScript extends)
                if heritage_child.type == "identifier":
                    bases.append(get_node_text(heritage_child))
                elif heritage_child.type == "member_expression":
                    # Handle qualified names like module.ClassName
                    bases.append(get_node_text(heritage_child))
                # TypeScript: extends clause
                elif heritage_child.type == "extends_clause":
                    for ext_child in heritage_child.children:
                        if ext_child.type in ("identifier", "type_identifier"):
                            bases.append(get_node_text(ext_child))
                        elif ext_child.type == "member_expression":
                            bases.append(get_node_text(ext_child))
                        elif ext_child.type == "generic_type":
                            # Generic<T> - get the base type
                            for gen_child in ext_child.children:
                                if gen_child.type in ("type_identifier", "identifier"):
                                    bases.append(get_node_text(gen_child))
                                    break
                # TypeScript: implements clause (inside class_heritage)
                elif heritage_child.type == "implements_clause":
                    for impl_child in heritage_child.children:
                        if impl_child.type in ("type_identifier", "identifier"):
                            bases.append(get_node_text(impl_child))
                        elif impl_child.type == "generic_type":
                            # Extract base type from generic like Comparable<T>
                            for gen_child in impl_child.children:
                                if gen_child.type in ("type_identifier", "identifier"):
                                    bases.append(get_node_text(gen_child))
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
