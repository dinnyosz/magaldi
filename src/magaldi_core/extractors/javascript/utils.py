"""Shared utilities for JavaScript/TypeScript extraction.

This module provides common helper functions used across the JavaScript
extractor modules.
"""

from __future__ import annotations

from tree_sitter import Node

from magaldi_core.extractors.base import (
    get_child_by_field,
    get_node_text,
)
from magaldi_core.extractors.types import (
    DecoratorInfo,
    ParameterInfo,
)


def is_likely_class_name(name: str) -> bool:
    """Check if name looks like a class (PascalCase).

    Args:
        name: The name to check.

    Returns:
        True if the name starts with uppercase but is not all uppercase.
    """
    if not name:
        return False
    # Starts with uppercase, not all uppercase (to exclude constants like HTTP)
    return name[0].isupper() and not name.isupper()


def is_react_hook(name: str) -> bool:
    """Check if a function name follows React hook naming convention.

    React hooks must start with "use" followed by an uppercase letter.
    Examples: useCounter, useState, useWindowSize
    Non-hooks: useless, user, use (too short)

    Args:
        name: The function name to check.

    Returns:
        True if the name follows the React hook naming convention.
    """
    if len(name) <= 3:
        return False
    return name.startswith("use") and name[3].isupper()


# React wrapper functions that wrap components
REACT_WRAPPERS = {"memo", "forwardRef", "lazy", "React.memo", "React.forwardRef", "React.lazy"}


def get_js_decorators(parent_node: Node) -> tuple[list[str], list[DecoratorInfo]]:
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


def get_preceding_decorators(
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


def extract_js_parameters(params_node: Node) -> list[ParameterInfo]:
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


def extract_js_return_type(node: Node) -> str | None:
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
