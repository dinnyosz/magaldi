"""Shared utilities for Python extraction.

This module provides common helper functions used across the Python
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


def is_inside_class(node: Node) -> bool:
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


def find_containing_element(node: Node) -> str | None:
    """Walk up AST to find the containing function/method/class name."""
    current = node.parent
    while current:
        if current.type in ("function_definition", "class_definition"):
            name_node = get_child_by_field(current, "name")
            if name_node:
                return get_node_text(name_node)  # type: ignore[no-any-return]
        current = current.parent
    return None


def get_call_context(node: Node, lines: list[str]) -> str:
    """Get a brief context snippet for a call expression."""
    line_idx = node.start_point[0]
    if line_idx < len(lines):
        line = lines[line_idx].strip()
        # Truncate long lines
        if len(line) > 80:
            return line[:77] + "..."
        return line
    return ""


def get_decorators(decorated_node: Node) -> tuple[list[str], list[DecoratorInfo]]:
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


def extract_python_parameters(params_node: Node) -> list[ParameterInfo]:
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



def is_python_enum(base_classes: list[str]) -> bool:
    """Check if a class is an Enum based on its base classes."""
    enum_types = {
        "Enum", "IntEnum", "StrEnum", "Flag", "IntFlag",
        "enum.Enum", "enum.IntEnum", "enum.StrEnum", "enum.Flag", "enum.IntFlag",
    }
    return any(base in enum_types for base in base_classes)
