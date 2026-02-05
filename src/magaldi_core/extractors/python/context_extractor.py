"""Python enhanced context extraction.

This module handles extraction of contextual information from Python code:
- Instance attributes from __init__
- Base classes from class definitions
- Raised exceptions from functions/methods
- Modified attributes (self.X assignments)
"""

from __future__ import annotations

from tree_sitter import Node

from magaldi_core.extractors.base import (
    get_child_by_field,
    get_node_text,
    walk_tree,
)


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
