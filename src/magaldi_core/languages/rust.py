"""Rust language tree-sitter extractors.

This module provides functions for extracting code elements from Rust source code:
- Structs, enums, functions, and impl blocks
- Imports (use statements)
- Method calls and function calls
- Enhanced context (struct fields, impl traits, panics, modified fields)
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

# Unused imports that might be needed in the future are commented out:
# from magaldi_core.tree_sitter_manager import find_nodes, get_child_by_field, get_children_by_type


# =============================================================================
# RUST ELEMENT EXTRACTION
# =============================================================================


def extract_rust_elements(tree: Tree, lines: list[str]) -> list[ExtractedElement]:
    """Extract structs, enums, and functions from Rust code.

    Args:
        tree: Parsed tree-sitter tree.
        lines: Source code lines.

    Returns:
        List of extracted elements.
    """
    elements: list[ExtractedElement] = []
    root = tree.root_node

    for node in root.children:
        if node.type == "struct_item":
            elem = _extract_rust_struct(node, lines)
            if elem:
                elements.append(elem)
        elif node.type == "enum_item":
            elem = _extract_rust_enum(node, lines)
            if elem:
                elements.append(elem)
        elif node.type == "function_item":
            elem = _extract_rust_function(node, lines)
            if elem:
                elements.append(elem)
        elif node.type == "impl_item":
            # impl blocks become "class" elements for consistency
            elem = _extract_rust_impl(node, lines)
            if elem:
                elements.append(elem)

    return elements


def _extract_rust_struct(node: Node, lines: list[str]) -> ExtractedElement | None:
    """Extract a Rust struct definition."""
    name = None
    for child in node.children:
        if child.type == "type_identifier":
            name = get_node_text(child)
            break

    if not name:
        return None

    return ExtractedElement(
        element_type="class",  # Treat struct as class for consistency
        name=name,
        line_start=node.start_point[0] + 1,
        line_end=node.end_point[0] + 1,
        raw_code="\n".join(lines[node.start_point[0]:node.end_point[0] + 1]),
        node=node,
    )


def _extract_rust_enum(node: Node, lines: list[str]) -> ExtractedElement | None:
    """Extract a Rust enum definition."""
    name = None
    for child in node.children:
        if child.type == "type_identifier":
            name = get_node_text(child)
            break

    if not name:
        return None

    return ExtractedElement(
        element_type="class",  # Treat enum as class for consistency
        name=name,
        line_start=node.start_point[0] + 1,
        line_end=node.end_point[0] + 1,
        raw_code="\n".join(lines[node.start_point[0]:node.end_point[0] + 1]),
        decorators=["enum"],
        node=node,
    )


def _extract_rust_function(node: Node, lines: list[str]) -> ExtractedElement | None:
    """Extract a Rust function definition."""
    name = None
    is_async = False
    decorators: list[str] = []

    for child in node.children:
        if child.type == "identifier":
            name = get_node_text(child)
        elif child.type == "async":
            is_async = True
            decorators.append("async")
        # visibility_modifier is parsed but not used yet

    if not name:
        return None

    # Build signature
    signature = f"fn {name}"
    for child in node.children:
        if child.type == "parameters":
            signature += get_node_text(child)
            break

    # Add return type
    for i, child in enumerate(node.children):
        if child.type == "->":
            if i + 1 < len(node.children):
                ret_type = node.children[i + 1]
                signature += f" -> {get_node_text(ret_type)}"
            break

    return ExtractedElement(
        element_type="function",
        name=name,
        line_start=node.start_point[0] + 1,
        line_end=node.end_point[0] + 1,
        raw_code="\n".join(lines[node.start_point[0]:node.end_point[0] + 1]),
        signature=signature,
        is_async=is_async,
        decorators=decorators,
        node=node,
    )


def _extract_rust_impl(node: Node, lines: list[str]) -> ExtractedElement | None:
    """Extract a Rust impl block."""
    # Find the type being implemented for
    impl_type = None
    trait_name = None
    has_for = False

    for child in node.children:
        if child.type == "for":
            has_for = True
        elif child.type == "type_identifier":
            if has_for:
                impl_type = get_node_text(child)
            else:
                # This could be the trait or the type
                if impl_type is None:
                    impl_type = get_node_text(child)
                else:
                    trait_name = impl_type
                    impl_type = get_node_text(child)
        elif child.type == "generic_type":
            # For impl Trait for Type or impl<T> Type
            for gt_child in child.children:
                if gt_child.type == "type_identifier":
                    if has_for:
                        impl_type = get_node_text(gt_child)
                    elif trait_name is None:
                        trait_name = get_node_text(gt_child)
                    break

    if not impl_type:
        return None

    name = impl_type
    if trait_name:
        name = f"{impl_type}::{trait_name}"

    return ExtractedElement(
        element_type="class",  # Treat impl as class for consistency
        name=name,
        line_start=node.start_point[0] + 1,
        line_end=node.end_point[0] + 1,
        raw_code="\n".join(lines[node.start_point[0]:node.end_point[0] + 1]),
        decorators=["impl"] + ([trait_name] if trait_name else []),
        node=node,
    )


def extract_rust_impl_members(
    impl_node: Node, lines: list[str]
) -> tuple[list[ExtractedElement], list[ExtractedElement]]:
    """Extract methods and associated items from a Rust impl block.

    Args:
        impl_node: An impl_item node.
        lines: Source code lines.

    Returns:
        Tuple of (methods, constants).
    """
    methods: list[ExtractedElement] = []
    constants: list[ExtractedElement] = []

    # Find declaration_list
    decl_list = None
    for child in impl_node.children:
        if child.type == "declaration_list":
            decl_list = child
            break

    if not decl_list:
        return methods, constants

    for child in decl_list.children:
        if child.type == "function_item":
            elem = _extract_rust_method(child, lines)
            if elem:
                methods.append(elem)
        elif child.type == "const_item":
            elem = _extract_rust_const(child, lines)
            if elem:
                constants.append(elem)

    return methods, constants


def _extract_rust_method(node: Node, lines: list[str]) -> ExtractedElement | None:
    """Extract a Rust method from an impl block."""
    name = None
    is_async = False
    decorators: list[str] = []
    has_self = False

    for child in node.children:
        if child.type == "identifier":
            name = get_node_text(child)
        elif child.type == "async":
            is_async = True
            decorators.append("async")
        elif child.type == "parameters":
            # visibility_modifier is parsed but not used yet
            # Check for self parameter
            for param in child.children:
                if param.type in ("self_parameter", "self"):
                    has_self = True
                    break
                if param.type == "parameter":
                    for pc in param.children:
                        if pc.type == "self":
                            has_self = True
                            break

    if not name:
        return None

    # Build signature
    signature = f"fn {name}"
    for child in node.children:
        if child.type == "parameters":
            signature += get_node_text(child)
            break

    # Add return type
    for i, child in enumerate(node.children):
        if child.type == "->":
            if i + 1 < len(node.children):
                ret_type = node.children[i + 1]
                signature += f" -> {get_node_text(ret_type)}"
            break

    # Method vs associated function
    elem_type = "method" if has_self else "function"

    return ExtractedElement(
        element_type=elem_type,
        name=name,
        line_start=node.start_point[0] + 1,
        line_end=node.end_point[0] + 1,
        raw_code="\n".join(lines[node.start_point[0]:node.end_point[0] + 1]),
        signature=signature,
        is_async=is_async,
        decorators=decorators,
        node=node,
    )


def _extract_rust_const(node: Node, lines: list[str]) -> ExtractedElement | None:
    """Extract a Rust const from an impl block."""
    name = None
    for child in node.children:
        if child.type == "identifier":
            name = get_node_text(child)
            break

    if not name:
        return None

    return ExtractedElement(
        element_type="constant",
        name=name,
        line_start=node.start_point[0] + 1,
        line_end=node.end_point[0] + 1,
        raw_code="\n".join(lines[node.start_point[0]:node.end_point[0] + 1]),
    )


# =============================================================================
# RUST IMPORT EXTRACTION
# =============================================================================


def extract_rust_imports(tree: Tree, lines: list[str]) -> list[ExtractedImport]:  # noqa: ARG001
    """Extract use statements from Rust code.

    Args:
        tree: Parsed tree-sitter tree.
        lines: Source code lines.

    Returns:
        List of extracted imports.
    """
    imports: list[ExtractedImport] = []
    root = tree.root_node

    for node in walk_tree(root):
        if node.type == "use_declaration":
            # Extract imports from use statement
            for child in node.children:
                if child.type == "use_clause" or child.type == "scoped_identifier":
                    path = get_node_text(child)
                    if path:
                        # Get the last component as the name
                        parts = path.replace("::", ".").split(".")
                        name = parts[-1] if parts else path
                        imports.append(ExtractedImport(
                            name=name,
                            module=path,
                            alias=None,
                            line=node.start_point[0] + 1,
                        ))
                elif child.type == "use_as_clause":
                    # use foo as bar
                    original = None
                    alias = None
                    for ac_child in child.children:
                        if ac_child.type == "scoped_identifier":
                            original = get_node_text(ac_child)
                        elif ac_child.type == "identifier" and original:
                            alias = get_node_text(ac_child)
                    if original:
                        parts = original.replace("::", ".").split(".")
                        name = alias or (parts[-1] if parts else original)
                        imports.append(ExtractedImport(
                            name=name,
                            module=original,
                            alias=alias,
                            line=node.start_point[0] + 1,
                        ))

    return imports


# =============================================================================
# RUST CALL EXTRACTION
# =============================================================================


def extract_rust_calls(function_node: Node) -> list[ExtractedCall]:
    """Extract function/method calls from Rust code.

    Args:
        function_node: A function_item node.

    Returns:
        List of extracted calls.
    """
    calls: list[ExtractedCall] = []
    seen: set[tuple[str, str | None, int]] = set()

    # Find function body (block)
    body = None
    for child in function_node.children:
        if child.type == "block":
            body = child
            break

    if not body:
        return calls

    for node in walk_tree(body):
        if node.type == "call_expression":
            func_node = node.children[0] if node.children else None
            line = node.start_point[0] + 1

            if func_node:
                if func_node.type == "identifier":
                    # Direct call: func()
                    name = get_node_text(func_node)
                    key = (name, None, line)
                    if key not in seen:
                        seen.add(key)
                        calls.append(ExtractedCall(name=name, receiver=None, line=line))

                elif func_node.type == "field_expression":
                    # Method call: self.method() or obj.method()
                    receiver = None
                    method = None
                    for fe_child in func_node.children:
                        if fe_child.type == "identifier":
                            receiver = get_node_text(fe_child)
                        elif fe_child.type == "field_identifier":
                            method = get_node_text(fe_child)

                    if method:
                        key = (method, receiver, line)
                        if key not in seen:
                            seen.add(key)
                            calls.append(ExtractedCall(name=method, receiver=receiver, line=line))

                elif func_node.type == "scoped_identifier":
                    # Static/associated call: Type::method()
                    text = get_node_text(func_node)
                    if "::" in text:
                        parts = text.split("::")
                        receiver = parts[0] if len(parts) > 1 else None
                        method = parts[-1]
                        key = (method, receiver, line)
                        if key not in seen:
                            seen.add(key)
                            calls.append(ExtractedCall(name=method, receiver=receiver, line=line))

    return calls


# =============================================================================
# RUST ENHANCED CONTEXT EXTRACTION
# =============================================================================


def extract_rust_struct_fields(struct_node: Node) -> list[dict[str, str | int]]:
    """Extract fields from a Rust struct.

    For: struct Foo { x: i32, y: String }
    Returns: [{"name": "x", "line": 2}, {"name": "y", "line": 3}]

    Args:
        struct_node: A struct_item node from tree-sitter.

    Returns:
        List of dicts with "name" and "line" keys.
    """
    fields: list[dict[str, str | int]] = []
    seen: set[str] = set()

    if struct_node.type != "struct_item":
        return fields

    # Find field_declaration_list
    for child in struct_node.children:
        if child.type == "field_declaration_list":
            for field_child in child.children:
                if field_child.type == "field_declaration":
                    # Find field_identifier
                    for fd_child in field_child.children:
                        if fd_child.type == "field_identifier":
                            name = get_node_text(fd_child)
                            if name and name not in seen:
                                seen.add(name)
                                fields.append({
                                    "name": name,
                                    "line": fd_child.start_point[0] + 1,
                                })
                            break

    return fields


def extract_rust_impl_traits(impl_node: Node) -> list[str]:
    """Extract trait implementations from a Rust impl block.

    For: impl From<Config> for MyService { ... }
    Returns: ["From"]

    For: impl MyService { ... }
    Returns: []

    Args:
        impl_node: An impl_item node from tree-sitter.

    Returns:
        List of trait names being implemented.
    """
    traits: list[str] = []

    if impl_node.type != "impl_item":
        return traits

    # Check if this is a trait impl (has 'for' keyword)
    has_for = False
    for child in impl_node.children:
        if child.type == "for":
            has_for = True
            break

    if not has_for:
        return traits

    # Find the trait type (before 'for')
    for child in impl_node.children:
        if child.type == "for":
            break
        if child.type == "type_identifier":
            traits.append(get_node_text(child))
        elif child.type == "generic_type":
            # From<Config> -> get "From"
            for gt_child in child.children:
                if gt_child.type == "type_identifier":
                    traits.append(get_node_text(gt_child))
                    break

    return traits


def extract_rust_panics(function_node: Node) -> list[str]:
    """Extract panic/error types from a Rust function.

    Finds:
    - panic!("msg") -> ["panic"]
    - Err(Error::ValidationError(...)) -> inside Err call
    - return Err(...) -> ["Err"]

    Args:
        function_node: A function_item node from tree-sitter.

    Returns:
        List of panic/error type names (deduplicated).
    """
    errors: list[str] = []
    seen: set[str] = set()

    # Find the function body (block)
    body_node = None
    for child in function_node.children:
        if child.type == "block":
            body_node = child
            break

    if not body_node:
        return errors

    for node in walk_tree(body_node):
        # panic!(...) macro
        if node.type == "macro_invocation":
            for mac_child in node.children:
                if mac_child.type == "identifier":
                    macro_name = get_node_text(mac_child)
                    if macro_name in ("panic", "unreachable", "unimplemented", "todo") and macro_name not in seen:
                        seen.add(macro_name)
                        errors.append(macro_name)
                    break

        # Err(...) call
        elif node.type == "call_expression":
            func = node.children[0] if node.children else None
            if func:
                if func.type == "identifier":
                    name = get_node_text(func)
                    if name == "Err":
                        if name not in seen:
                            seen.add(name)
                            errors.append(name)
                        # Also check for error type inside Err(...)
                        args = None
                        for child in node.children:
                            if child.type == "arguments":
                                args = child
                                break
                        if args:
                            for arg in walk_tree(args):
                                if arg.type == "scoped_identifier":
                                    text = get_node_text(arg)
                                    if "::" in text and "Error" in text:
                                        parts = text.split("::")
                                        variant = parts[-1]
                                        if variant not in seen:
                                            seen.add(variant)
                                            errors.append(variant)
                elif func.type == "scoped_identifier":
                    # Only capture if it looks like an error (contains "Error")
                    text = get_node_text(func)
                    if "::" in text and "Error" in text:
                        parts = text.split("::")
                        variant = parts[-1]
                        if variant not in seen:
                            seen.add(variant)
                            errors.append(variant)

    return errors


def extract_rust_modified_fields(function_node: Node) -> list[str]:
    """Extract self.X fields that are assigned to in a Rust method.

    Args:
        function_node: A function_item node.

    Returns:
        List of field names that are modified (deduplicated).
    """
    modified: list[str] = []
    seen: set[str] = set()

    # Find the function body (block)
    body_node = None
    for child in function_node.children:
        if child.type == "block":
            body_node = child
            break

    if not body_node:
        return modified

    for node in walk_tree(body_node):
        if node.type == "assignment_expression":
            left = node.children[0] if node.children else None
            if left and left.type == "field_expression":
                # Check if it's self.field
                obj = None
                field = None
                for fe_child in left.children:
                    if fe_child.type == "identifier" and get_node_text(fe_child) == "self":
                        obj = fe_child
                    elif fe_child.type == "field_identifier":
                        field = fe_child

                if obj and field:
                    field_name = get_node_text(field)
                    if field_name and field_name not in seen:
                        seen.add(field_name)
                        modified.append(field_name)

    return modified
