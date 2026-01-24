"""PHP tree-sitter extractors.

This module provides functions for extracting code elements from PHP source files
using tree-sitter parsing:
- Classes and functions (extract_php_elements)
- Class members: methods and properties (extract_php_class_members)
- Import statements via 'use' (extract_php_imports)
- Function and method calls (extract_php_calls)
- Class properties (extract_php_class_properties)
- Base classes and interfaces (extract_php_base_class)
- Thrown exceptions (extract_php_thrown_exceptions)
- Modified properties (extract_php_modified_properties)
"""

from __future__ import annotations

from tree_sitter import Node, Tree

from magaldi_core.tree_sitter_manager import (
    ExtractedCall,
    ExtractedElement,
    ExtractedImport,
    get_node_text,
    walk_tree,
)


def extract_php_elements(tree: Tree, lines: list[str]) -> list[ExtractedElement]:
    """Extract classes and functions from PHP code.

    Args:
        tree: Parsed tree-sitter tree.
        lines: Source code lines.

    Returns:
        List of extracted elements (classes, functions).
    """
    elements: list[ExtractedElement] = []
    root = tree.root_node

    for node in root.children:
        if node.type == "class_declaration":
            elem = _extract_php_class(node, lines)
            if elem:
                elements.append(elem)
        elif node.type == "function_definition":
            elem = _extract_php_function(node, lines)
            if elem:
                elements.append(elem)

    return elements


def _extract_php_class(node: Node, lines: list[str]) -> ExtractedElement | None:
    """Extract a PHP class definition."""
    name = None
    for child in node.children:
        if child.type == "name":
            name = get_node_text(child)
            break

    if not name:
        return None

    return ExtractedElement(
        element_type="class",
        name=name,
        line_start=node.start_point[0] + 1,
        line_end=node.end_point[0] + 1,
        raw_code="\n".join(lines[node.start_point[0]:node.end_point[0] + 1]),
        node=node,
    )


def _extract_php_function(node: Node, lines: list[str]) -> ExtractedElement | None:
    """Extract a PHP function definition."""
    name = None
    for child in node.children:
        if child.type == "name":
            name = get_node_text(child)
            break

    if not name:
        return None

    # Build signature from formal_parameters
    signature = f"function {name}"
    for child in node.children:
        if child.type == "formal_parameters":
            signature += get_node_text(child)
            break

    return ExtractedElement(
        element_type="function",
        name=name,
        line_start=node.start_point[0] + 1,
        line_end=node.end_point[0] + 1,
        raw_code="\n".join(lines[node.start_point[0]:node.end_point[0] + 1]),
        signature=signature,
        node=node,
    )


def extract_php_class_members(
    class_node: Node, lines: list[str]
) -> tuple[list[ExtractedElement], list[ExtractedElement]]:
    """Extract methods and properties from a PHP class.

    Args:
        class_node: A class_declaration node.
        lines: Source code lines.

    Returns:
        Tuple of (methods, properties).
    """
    methods: list[ExtractedElement] = []
    properties: list[ExtractedElement] = []

    # Find declaration_list (class body)
    decl_list = None
    for child in class_node.children:
        if child.type == "declaration_list":
            decl_list = child
            break

    if not decl_list:
        return methods, properties

    for child in decl_list.children:
        if child.type == "method_declaration":
            elem = _extract_php_method(child, lines)
            if elem:
                methods.append(elem)
        elif child.type == "property_declaration":
            elems = _extract_php_property(child, lines)
            properties.extend(elems)

    return methods, properties


def _extract_php_method(node: Node, lines: list[str]) -> ExtractedElement | None:
    """Extract a PHP method definition."""
    name = None
    visibility = "public"
    decorators: list[str] = []

    for child in node.children:
        if child.type == "name":
            name = get_node_text(child)
        elif child.type == "visibility_modifier":
            visibility = get_node_text(child)
        elif child.type == "static_modifier":
            decorators.append("static")

    if not name:
        return None

    # Build signature
    signature = f"{visibility} function {name}"
    for child in node.children:
        if child.type == "formal_parameters":
            signature += get_node_text(child)
            break

    return ExtractedElement(
        element_type="method",
        name=name,
        line_start=node.start_point[0] + 1,
        line_end=node.end_point[0] + 1,
        raw_code="\n".join(lines[node.start_point[0]:node.end_point[0] + 1]),
        signature=signature,
        decorators=decorators,
        node=node,
    )


def _extract_php_property(node: Node, lines: list[str]) -> list[ExtractedElement]:
    """Extract PHP property declarations."""
    properties: list[ExtractedElement] = []
    visibility = "public"

    for child in node.children:
        if child.type == "visibility_modifier":
            visibility = get_node_text(child)
        elif child.type == "property_element":
            for prop_child in child.children:
                if prop_child.type == "variable_name":
                    for name_child in prop_child.children:
                        if name_child.type == "name":
                            name = get_node_text(name_child)
                            properties.append(ExtractedElement(
                                element_type="variable",
                                name=name,
                                line_start=node.start_point[0] + 1,
                                line_end=node.end_point[0] + 1,
                                raw_code="\n".join(lines[node.start_point[0]:node.end_point[0] + 1]),
                                decorators=[visibility],
                            ))

    return properties


def extract_php_imports(tree: Tree, lines: list[str]) -> list[ExtractedImport]:  # noqa: ARG001
    """Extract use statements from PHP code.

    Args:
        tree: Parsed tree-sitter tree.
        lines: Source code lines.

    Returns:
        List of extracted imports.
    """
    imports: list[ExtractedImport] = []
    root = tree.root_node

    for node in walk_tree(root):
        if node.type == "namespace_use_declaration":
            # use Foo\Bar\Baz;
            for child in node.children:
                if child.type == "namespace_use_clause":
                    name = None
                    alias = None
                    for clause_child in child.children:
                        if clause_child.type == "qualified_name":
                            name = get_node_text(clause_child)
                        elif clause_child.type == "namespace_aliasing_clause":
                            for alias_child in clause_child.children:
                                if alias_child.type == "name":
                                    alias = get_node_text(alias_child)

                    if name:
                        # Extract the class name from full path
                        parts = name.split("\\")
                        class_name = parts[-1]
                        imports.append(ExtractedImport(
                            name=alias or class_name,
                            module=name,
                            alias=alias,
                            line=node.start_point[0] + 1,
                        ))

    return imports


def extract_php_calls(function_node: Node) -> list[ExtractedCall]:
    """Extract function/method calls from PHP code.

    Args:
        function_node: A function_definition or method_declaration node.

    Returns:
        List of extracted calls.
    """
    calls: list[ExtractedCall] = []
    seen: set[tuple[str, str | None, int]] = set()

    for node in walk_tree(function_node):
        if node.type == "function_call_expression":
            # Regular function call: func()
            func_node = node.children[0] if node.children else None
            if func_node and func_node.type == "name":
                name = get_node_text(func_node)
                line = node.start_point[0] + 1
                key = (name, None, line)
                if key not in seen:
                    seen.add(key)
                    calls.append(ExtractedCall(name=name, receiver=None, line=line))

        elif node.type == "member_call_expression":
            # Method call: $obj->method()
            line = node.start_point[0] + 1
            receiver = None
            method_name = None

            for child in node.children:
                if child.type == "variable_name":
                    for vn_child in child.children:
                        if vn_child.type == "name":
                            receiver = get_node_text(vn_child)
                elif child.type == "name":
                    method_name = get_node_text(child)

            if method_name:
                key = (method_name, receiver, line)
                if key not in seen:
                    seen.add(key)
                    calls.append(ExtractedCall(name=method_name, receiver=receiver, line=line))

        elif node.type == "scoped_call_expression":
            # Static call: Class::method()
            line = node.start_point[0] + 1
            receiver = None
            method_name = None

            for child in node.children:
                if child.type in ("name", "qualified_name", "relative_scope"):
                    receiver = get_node_text(child)
                elif child.type == "name" and receiver:
                    method_name = get_node_text(child)

            # Try getting method name differently
            if not method_name:
                for i, child in enumerate(node.children):
                    if child.type == "::" and i + 1 < len(node.children):
                        next_child = node.children[i + 1]
                        if next_child.type == "name":
                            method_name = get_node_text(next_child)

            if method_name:
                key = (method_name, receiver, line)
                if key not in seen:
                    seen.add(key)
                    calls.append(ExtractedCall(name=method_name, receiver=receiver, line=line))

    return calls


# =============================================================================
# PHP ENHANCED CONTEXT EXTRACTION
# =============================================================================


def extract_php_class_properties(class_node: Node) -> list[dict[str, str | int]]:
    """Extract class properties from a PHP class.

    Finds:
    - Property declarations: `private $config;`
    - Constructor assignments: `$this->x = $x;`

    Args:
        class_node: A class_declaration node from tree-sitter.

    Returns:
        List of dicts with "name" and "line" keys.
    """
    properties: list[dict[str, str | int]] = []
    seen: set[str] = set()

    if class_node.type != "class_declaration":
        return properties

    # Find declaration_list (class body)
    decl_list = None
    for child in class_node.children:
        if child.type == "declaration_list":
            decl_list = child
            break

    if not decl_list:
        return properties

    for child in decl_list.children:
        # Property declarations: `private $config;`
        if child.type == "property_declaration":
            for prop_child in child.children:
                if prop_child.type == "property_element":
                    for var_child in prop_child.children:
                        if var_child.type == "variable_name":
                            # Get the property name (without $)
                            for name_child in var_child.children:
                                if name_child.type == "name":
                                    name = get_node_text(name_child)
                                    if name and name not in seen:
                                        seen.add(name)
                                        properties.append({
                                            "name": name,
                                            "line": var_child.start_point[0] + 1,
                                        })

        # Also check constructor for $this->x = ... assignments
        elif child.type == "method_declaration":
            method_name = None
            for method_child in child.children:
                if method_child.type == "name":
                    method_name = get_node_text(method_child)
                    break

            if method_name == "__construct":
                # Find assignments in constructor body
                for node in walk_tree(child):
                    if node.type == "assignment_expression":
                        left = node.children[0] if node.children else None
                        if left and left.type == "member_access_expression":
                            # Check if it's $this->property
                            obj = None
                            prop = None
                            for mac_child in left.children:
                                if mac_child.type == "variable_name":
                                    for vn_child in mac_child.children:
                                        if vn_child.type == "name" and get_node_text(vn_child) == "this":
                                            obj = mac_child
                                elif mac_child.type == "name":
                                    prop = mac_child

                            if obj and prop:
                                prop_name = get_node_text(prop)
                                if prop_name and prop_name not in seen:
                                    seen.add(prop_name)
                                    properties.append({
                                        "name": prop_name,
                                        "line": left.start_point[0] + 1,
                                    })

    return properties


def extract_php_base_class(class_node: Node) -> list[str]:
    """Extract the base class and interfaces from a PHP class.

    For: class Foo extends Bar implements Baz { ... }
    Returns: ["Bar", "Baz"]

    Args:
        class_node: A class_declaration node from tree-sitter.

    Returns:
        List of base class and interface names.
    """
    bases: list[str] = []

    if class_node.type != "class_declaration":
        return bases

    for child in class_node.children:
        # Base class: extends BaseClass
        if child.type == "base_clause":
            for bc_child in child.children:
                if bc_child.type == "name":
                    bases.append(get_node_text(bc_child))

        # Interfaces: implements Interface1, Interface2
        elif child.type == "class_interface_clause":
            for ic_child in child.children:
                if ic_child.type == "name":
                    bases.append(get_node_text(ic_child))

    return bases


def extract_php_thrown_exceptions(method_node: Node) -> list[str]:
    """Extract exception types from throw statements in a PHP method.

    For: throw new ValidationException("msg")
    Returns: ["ValidationException"]

    Args:
        method_node: A method_declaration or function_definition node.

    Returns:
        List of exception type names (deduplicated).
    """
    exceptions: list[str] = []
    seen: set[str] = set()

    for node in walk_tree(method_node):
        if node.type == "throw_expression":
            # throw new Exception("msg") -> find the exception type
            for child in node.children:
                if child.type == "object_creation_expression":
                    # Find the class name being instantiated
                    for oc_child in child.children:
                        if oc_child.type in ("name", "qualified_name"):
                            exc_name = get_node_text(oc_child)
                            if exc_name and exc_name not in seen:
                                seen.add(exc_name)
                                exceptions.append(exc_name)
                            break
                    break
                elif child.type == "variable_name":
                    # throw $exception (re-throwing)
                    for vn_child in child.children:
                        if vn_child.type == "name":
                            exc_name = get_node_text(vn_child)
                            if exc_name and exc_name not in seen:
                                seen.add(exc_name)
                                exceptions.append(exc_name)
                    break

    return exceptions


def extract_php_modified_properties(method_node: Node) -> list[str]:
    """Extract $this->X properties that are assigned to in a PHP method.

    Args:
        method_node: A method_declaration node.

    Returns:
        List of property names that are modified (deduplicated).
    """
    modified: list[str] = []
    seen: set[str] = set()

    for node in walk_tree(method_node):
        if node.type in ("assignment_expression", "augmented_assignment_expression"):
            left = node.children[0] if node.children else None
            if left and left.type == "member_access_expression":
                # Check if it's $this->property
                is_this = False
                prop_name = None

                for mac_child in left.children:
                    if mac_child.type == "variable_name":
                        for vn_child in mac_child.children:
                            if vn_child.type == "name" and get_node_text(vn_child) == "this":
                                is_this = True
                    elif mac_child.type == "name":
                        prop_name = get_node_text(mac_child)

                if is_this and prop_name and prop_name not in seen:
                    seen.add(prop_name)
                    modified.append(prop_name)

    return modified
