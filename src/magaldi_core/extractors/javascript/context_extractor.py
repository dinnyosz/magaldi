"""JavaScript/TypeScript context extraction.

This module handles extraction of enhanced context information:
- Class fields (both field definitions and constructor assignments)
- Base classes and implemented interfaces
- Thrown exceptions
- Modified properties (this.X = ...)
"""

from __future__ import annotations

from tree_sitter import Node

from magaldi_core.extractors.base import (
    get_child_by_field,
    get_node_text,
    walk_tree,
)


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
                        if ext_child.type in ("identifier", "type_identifier") or ext_child.type == "member_expression":
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
