"""Rust language extractor using tree-sitter.

This module provides the RustExtractor class and standalone functions
for extracting code elements, imports, references, call graph data,
and enhanced context from Rust source code.
"""

from __future__ import annotations

from tree_sitter import Node, Tree

from magaldi_core.extractors.base import (
    BaseExtractor,
    get_child_by_field,
    get_node_text,
    walk_top_level,
    walk_tree,
)
from magaldi_core.extractors.types import (
    ExtractedCall,
    ExtractedElement,
    ExtractedImport,
    ExtractedReference,
    ParameterInfo,
)

# =============================================================================
# RUST EXTRACTOR CLASS
# =============================================================================


class RustExtractor(BaseExtractor):
    """Extractor for Rust source code."""

    @property
    def language(self) -> str:
        return "rust"

    def extract_elements(self, tree: Tree, lines: list[str]) -> list[ExtractedElement]:
        """Extract code elements from a Rust AST."""
        return extract_rust_elements(tree, lines)

    def extract_class_members(
        self, class_node: Node, lines: list[str]
    ) -> tuple[list[ExtractedElement], list[ExtractedElement]]:
        """Extract methods and constants from a Rust impl block."""
        return extract_rust_impl_members(class_node, lines)

    def extract_imports(self, tree: Tree, lines: list[str]) -> list[ExtractedImport]:
        """Extract use statements from Rust AST."""
        return extract_rust_imports(tree, lines)

    def extract_references(
        self,
        tree: Tree,  # noqa: ARG002
        lines: list[str],  # noqa: ARG002
    ) -> list[ExtractedReference]:
        """Extract references from Rust AST (stub - returns empty list)."""
        return []

    def extract_calls(self, function_node: Node) -> list[ExtractedCall]:
        """Extract function/method calls from within a Rust function body."""
        return extract_rust_calls(function_node)

    def extract_class_attributes(
        self, class_node: Node
    ) -> list[dict[str, str | int]]:
        """Extract struct fields from Rust struct."""
        return extract_rust_struct_fields(class_node)

    def extract_base_classes(self, class_node: Node) -> list[str]:
        """Extract trait implementations from Rust impl block."""
        return extract_rust_impl_traits(class_node)

    def extract_raised_exceptions(self, function_node: Node) -> list[str]:
        """Extract panic/error types from Rust function."""
        return extract_rust_panics(function_node)

    def extract_modified_attributes(self, method_node: Node) -> list[str]:
        """Extract self.X fields that are modified."""
        return extract_rust_modified_fields(method_node)


# =============================================================================
# ATTRIBUTE HELPERS
# =============================================================================


def _get_preceding_attributes(node: Node) -> list[str]:
    """Extract attribute names from attribute_item siblings immediately preceding a node.

    In Rust's AST, attributes like #[test] or #[cfg(test)] are sibling nodes
    that appear immediately before the annotated item in the parent's children list.

    Args:
        node: The AST node to find preceding attributes for.

    Returns:
        List of attribute name strings (e.g., ["test", "cfg"]).
    """
    attrs: list[str] = []
    parent = node.parent
    if not parent:
        return attrs

    # Walk backwards through siblings to collect consecutive attribute_items
    found_self = False
    preceding: list[Node] = []
    for child in reversed(parent.children):
        if child.id == node.id:
            found_self = True
            continue
        if found_self:
            if child.type == "attribute_item":
                preceding.append(child)
            else:
                break  # Stop at first non-attribute sibling

    for attr_node in preceding:
        attr_name = _extract_attribute_name(attr_node)
        if attr_name:
            attrs.append(attr_name)

    return attrs


def _extract_attribute_name(attr_item_node: Node) -> str | None:
    """Extract the attribute name from an attribute_item node.

    For #[test] returns "test".
    For #[cfg(test)] returns "cfg".
    For #[tokio::test] returns "tokio::test".

    Args:
        attr_item_node: An attribute_item AST node.

    Returns:
        The attribute name string, or None.
    """
    for child in attr_item_node.children:
        if child.type == "attribute":
            for attr_child in child.children:
                if attr_child.type == "identifier" or attr_child.type == "scoped_identifier":
                    return get_node_text(attr_child)  # type: ignore[no-any-return]
    return None


# =============================================================================
# RUST VISIBILITY HELPERS
# =============================================================================


def _extract_rust_visibility(node: Node) -> str:
    """Extract visibility from a Rust AST node.

    Rust visibility rules:
    - No visibility_modifier -> "private" (module-private)
    - `pub` -> "public"
    - `pub(crate)` -> "pub(crate)"
    - `pub(super)` -> "pub(super)"
    - `pub(in path)` -> "pub(in path)"

    Args:
        node: A function_item, struct_item, enum_item, trait_item, etc.

    Returns:
        Visibility string.
    """
    for child in node.children:
        if child.type == "visibility_modifier":
            text = get_node_text(child).strip()
            if text == "pub":
                return "public"
            # pub(crate), pub(super), pub(in path::to::mod)
            return text
    return "private"


# =============================================================================
# RUST ELEMENT EXTRACTION
# =============================================================================


def extract_rust_elements(
    tree: Tree, lines: list[str], _file_path: str | None = None
) -> list[ExtractedElement]:
    """Extract structs, enums, functions, and module-level variables from Rust code.

    Args:
        tree: Parsed tree-sitter tree.
        lines: Source code lines.
        file_path: Optional file path for logging purposes.

    Returns:
        List of extracted elements.
    """
    return _extract_elements_from_children(tree.root_node.children, lines)


def _extract_elements_from_children(
    children: list[Node] | tuple[Node, ...], lines: list[str]
) -> list[ExtractedElement]:
    """Extract elements from a list of AST child nodes.

    Shared logic for extracting from both the file root and from inside
    mod_item declaration_list blocks.

    Args:
        children: List of AST child nodes to process.
        lines: Source code lines.

    Returns:
        List of extracted elements.
    """
    elements: list[ExtractedElement] = []

    for node in children:
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
                # Extract variables from function body
                elements.extend(_extract_rust_function_body_variables(node, lines))
        elif node.type == "impl_item":
            # impl blocks become "class" elements for consistency
            elem = _extract_rust_impl(node, lines)
            if elem:
                elements.append(elem)
        elif node.type == "trait_item":
            elem = _extract_rust_trait(node, lines)
            if elem:
                elements.append(elem)
        elif node.type == "use_declaration":
            elem = _extract_rust_use(node, lines)
            if elem:
                elements.append(elem)
        elif node.type == "const_item":
            # Module-level const: const FOO: i32 = 42;
            elem = _extract_rust_module_const(node, lines)
            if elem:
                elements.append(elem)
        elif node.type == "static_item":
            # Module-level static: static FOO: i32 = 42;
            elem = _extract_rust_static(node, lines)
            if elem:
                elements.append(elem)
        elif node.type == "type_item":
            elem = _extract_rust_type_alias(node, lines)
            if elem:
                elements.append(elem)
        elif node.type == "let_declaration":
            # Module-level let (rare, but handle it)
            elem = _extract_rust_variable(node, lines)
            if elem:
                elements.append(elem)
        elif node.type == "mod_item":
            # Inline module: mod tests { ... }
            mod_elem = _extract_rust_mod(node, lines)
            if mod_elem:
                elements.append(mod_elem)
                # Recursively extract child elements from the mod's body
                mod_children = _extract_mod_body_elements(node, lines)
                elements.extend(mod_children)

    return elements


def _extract_rust_mod(node: Node, _lines: list[str]) -> ExtractedElement | None:
    """Extract a Rust inline mod block as a class-like element.

    Handles: mod tests { ... }
    Also handles: #[cfg(test)] mod tests { ... } (attribute is on preceding sibling)

    Args:
        node: A mod_item AST node.
        lines: Source code lines.

    Returns:
        ExtractedElement for the module, or None if it has no body (extern mod).
    """
    name = None
    has_body = False

    for child in node.children:
        if child.type == "identifier":
            name = get_node_text(child)
        elif child.type == "declaration_list":
            has_body = True

    if not name or not has_body:
        return None  # Skip `mod foo;` (extern module declarations without body)

    # Collect preceding attributes (e.g., #[cfg(test)])
    preceding_attrs = _get_preceding_attributes(node)
    decorators = ["mod"] + preceding_attrs

    return ExtractedElement(
        element_type="class",  # Treat mod as class for consistency (like impl)
        name=name,
        line_start=node.start_point[0] + 1,
        line_end=node.end_point[0] + 1,
        raw_code=node.text.decode('utf-8') if node.text else "",
        byte_offset=node.start_byte,
        decorators=decorators,
        node=node,
    )


def _extract_mod_body_elements(
    mod_node: Node, lines: list[str]
) -> list[ExtractedElement]:
    """Extract elements from inside a mod block's declaration_list.

    Args:
        mod_node: A mod_item AST node.
        lines: Source code lines.

    Returns:
        List of elements found inside the module body.
    """
    # Find declaration_list
    decl_list = None
    for child in mod_node.children:
        if child.type == "declaration_list":
            decl_list = child
            break

    if not decl_list:
        return []

    return _extract_elements_from_children(decl_list.children, lines)


def _extract_rust_struct(node: Node, _lines: list[str]) -> ExtractedElement | None:
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
        raw_code=node.text.decode('utf-8') if node.text else "",
        byte_offset=node.start_byte,
        visibility=_extract_rust_visibility(node),
        node=node,
    )


def _extract_rust_enum(node: Node, _lines: list[str]) -> ExtractedElement | None:
    """Extract a Rust enum definition."""
    name = None
    for child in node.children:
        if child.type == "type_identifier":
            name = get_node_text(child)
            break

    if not name:
        return None

    return ExtractedElement(
        element_type="enum",
        name=name,
        line_start=node.start_point[0] + 1,
        line_end=node.end_point[0] + 1,
        raw_code=node.text.decode('utf-8') if node.text else "",
        byte_offset=node.start_byte,
        visibility=_extract_rust_visibility(node),
        node=node,
    )


def _extract_rust_trait(node: Node, _lines: list[str]) -> ExtractedElement | None:
    """Extract a Rust trait definition."""
    name = None
    for child in node.children:
        if child.type == "type_identifier":
            name = get_node_text(child)
            break

    if not name:
        return None

    return ExtractedElement(
        element_type="trait",
        name=name,
        line_start=node.start_point[0] + 1,
        line_end=node.end_point[0] + 1,
        raw_code=node.text.decode('utf-8') if node.text else "",
        byte_offset=node.start_byte,
        visibility=_extract_rust_visibility(node),
        node=node,
    )


def _extract_rust_use(node: Node, _lines: list[str]) -> ExtractedElement | None:
    """Extract a Rust use declaration (import)."""
    raw_code = node.text.decode('utf-8') if node.text else ""

    # Extract module name from use statement
    # Examples: use std::fmt::Debug; -> std::fmt::Debug
    #           use std::fmt::{Debug, Display}; -> std::fmt
    #           use std::collections::HashMap; -> std::collections::HashMap
    module = ""
    for child in node.children:
        if child.type == "use_tree":
            module = _extract_use_tree_path(child)
            break
        elif child.type == "scoped_identifier":
            module = get_node_text(child)
            break
        elif child.type == "scoped_use_list":
            # Handle: use std::fmt::{Debug, Display};
            module = _extract_scoped_use_list_path(child)
            break
        elif child.type == "identifier":
            module = get_node_text(child)
            break

    if not module:
        return None

    return ExtractedElement(
        element_type="import",
        name=module,
        line_start=node.start_point[0] + 1,
        line_end=node.end_point[0] + 1,
        raw_code=raw_code,
        byte_offset=node.start_byte,
        node=node,
    )


def _extract_scoped_use_list_path(node: Node) -> str:
    """Extract the base path from a scoped_use_list node.

    For: use std::fmt::{Debug, Display};
    Returns: std::fmt
    """
    for child in node.children:
        if child.type == "scoped_identifier" or child.type == "identifier":
            return get_node_text(child)  # type: ignore[no-any-return]
    return ""


def _extract_use_tree_path(node: Node) -> str:
    """Extract the path from a use_tree node."""
    parts = []
    for child in node.children:
        if child.type == "identifier" or child.type == "scoped_identifier":
            parts.append(get_node_text(child))
        elif child.type == "use_tree":
            inner = _extract_use_tree_path(child)
            if inner:
                parts.append(inner)
    return "::".join(parts) if parts else ""


def _extract_rust_module_const(node: Node, _lines: list[str]) -> ExtractedElement | None:
    """Extract a module-level const declaration.

    Handles patterns like:
        const MAX_SIZE: usize = 1024;
        const CONFIG: Config = Config::new();

    Args:
        node: A const_item node.
        lines: Source code lines.

    Returns:
        ExtractedElement if the constant is useful, None otherwise.
    """
    name = None
    type_node = None

    for child in node.children:
        if child.type == "identifier":
            name = get_node_text(child)
        elif child.type in ("primitive_type", "type_identifier", "generic_type", "scoped_type_identifier"):
            type_node = child
        elif child.type in (
            "integer_literal", "float_literal", "string_literal", "char_literal",
            "boolean_literal", "array_expression", "tuple_expression",
            "call_expression", "macro_invocation",
        ):
            pass

    if not name:
        return None

    line_start = node.start_point[0] + 1
    line_end = node.end_point[0] + 1
    raw_code = node.text.decode("utf-8") if node.text else ""
    const_type = get_node_text(type_node) if type_node else None

    return ExtractedElement(
        element_type="constant",
        name=name,
        line_start=line_start,
        line_end=line_end,
        raw_code=raw_code,
        byte_offset=node.start_byte,
        visibility=_extract_rust_visibility(node),
        node=node,
        return_type=const_type,  # Store type in return_type field
    )


def _extract_rust_static(node: Node, _lines: list[str]) -> ExtractedElement | None:
    """Extract a module-level static declaration.

    Handles patterns like:
        static INSTANCE: Mutex<Option<Service>> = Mutex::new(None);

    Args:
        node: A static_item node.
        lines: Source code lines.

    Returns:
        ExtractedElement for the static declaration.
    """
    name = None
    type_node = None
    is_mutable = False

    for child in node.children:
        if child.type == "identifier":
            name = get_node_text(child)
        elif child.type == "mutable_specifier":
            is_mutable = True
        elif child.type in ("primitive_type", "type_identifier", "generic_type", "scoped_type_identifier"):
            type_node = child

    if not name:
        return None

    line_start = node.start_point[0] + 1
    line_end = node.end_point[0] + 1
    raw_code = node.text.decode("utf-8") if node.text else ""
    static_type = get_node_text(type_node) if type_node else None

    decorators = ["static"]
    if is_mutable:
        decorators.append("mut")

    return ExtractedElement(
        element_type="variable",
        name=name,
        line_start=line_start,
        line_end=line_end,
        raw_code=raw_code,
        byte_offset=node.start_byte,
        visibility=_extract_rust_visibility(node),
        node=node,
        return_type=static_type,
        decorators=decorators,
    )


def _extract_rust_variable(node: Node, _lines: list[str]) -> ExtractedElement | None:
    """Extract a Rust let declaration.

    Applies usefulness filter to skip transient variables like object creations
    and function call results.

    Args:
        node: A let_declaration node.
        lines: Source code lines.

    Returns:
        ExtractedElement if the variable is useful, None otherwise.
    """
    name = None
    value_node = None
    is_mutable = False

    # Find pattern (name) and value
    pattern_node = get_child_by_field(node, "pattern")
    value_node = get_child_by_field(node, "value")

    if pattern_node:
        if pattern_node.type == "identifier":
            name = get_node_text(pattern_node)
        elif pattern_node.type == "mut_pattern":
            is_mutable = True
            for child in pattern_node.children:
                if child.type == "identifier":
                    name = get_node_text(child)
                    break

    if not name or not value_node:
        return None

    line_start = node.start_point[0] + 1
    line_end = node.end_point[0] + 1
    raw_code = node.text.decode("utf-8") if node.text else ""

    decorators = []
    if is_mutable:
        decorators.append("mut")

    return ExtractedElement(
        element_type="variable",
        name=name,
        line_start=line_start,
        line_end=line_end,
        raw_code=raw_code,
        byte_offset=node.start_byte,
        node=node,
        decorators=decorators if decorators else None,
    )


def _extract_rust_type_alias(node: Node, _lines: list[str]) -> ExtractedElement | None:
    """Extract a Rust type alias (type Foo = Bar;)."""
    name = None
    for child in node.children:
        if child.type == "type_identifier":
            name = get_node_text(child)
            break

    if not name:
        return None

    raw_code = node.text.decode('utf-8') if node.text else ""

    # Build signature: type Name = <value>
    # Truncate long type definitions
    signature = raw_code.rstrip(";").strip()
    if len(signature) > 100:
        signature = signature[:97] + "..."

    return ExtractedElement(
        element_type="type_alias",
        name=name,
        line_start=node.start_point[0] + 1,
        line_end=node.end_point[0] + 1,
        raw_code=raw_code,
        byte_offset=node.start_byte,
        visibility=_extract_rust_visibility(node),
        signature=signature,
        node=node,
    )


def _extract_rust_function_body_variables(
    func_node: Node, lines: list[str]
) -> list[ExtractedElement]:
    """Extract let/const declarations from inside a Rust function body.

    Walks the function body to find all let_declaration and const_item nodes,
    skipping nested function definitions.

    Args:
        func_node: A function_item node.
        lines: Source code lines.

    Returns:
        List of variable/constant elements found inside the function.
    """
    variables: list[ExtractedElement] = []

    # Find the block (function body)
    body_node = None
    for child in func_node.children:
        if child.type == "block":
            body_node = child
            break

    if not body_node:
        return variables

    for node in walk_tree(body_node):
        # Skip nested function definitions
        if node.type == "function_item":
            continue

        if node.type == "let_declaration":
            elem = _extract_rust_variable(node, lines)
            if elem:
                variables.append(elem)
        elif node.type == "const_item":
            elem = _extract_rust_module_const(node, lines)
            if elem:
                variables.append(elem)

    return variables


def _extract_rust_parameters(params_node: Node) -> list[ParameterInfo]:
    """Extract structured parameter info from Rust parameters node.

    Args:
        params_node: The 'parameters' AST node.

    Returns:
        List of ParameterInfo with name, type, and default value.
    """
    parameters: list[ParameterInfo] = []
    if not params_node:
        return parameters

    for child in params_node.children:
        if child.type == "self_parameter":
            # &self, &mut self, self
            self_text = get_node_text(child)
            parameters.append(ParameterInfo(name=self_text, type=None, default=None))
        elif child.type == "parameter":
            param_name: str | None = None
            param_type: str | None = None

            # Find pattern (name) and type
            pattern_node = get_child_by_field(child, "pattern")
            type_node = get_child_by_field(child, "type")

            if pattern_node:
                param_name = get_node_text(pattern_node)
            if type_node:
                param_type = get_node_text(type_node)

            if param_name:
                parameters.append(ParameterInfo(
                    name=param_name,
                    type=param_type,
                    default=None,  # Rust doesn't have default parameters
                ))

    return parameters


def _extract_rust_return_type(node: Node) -> str | None:
    """Extract return type from Rust function/method node."""
    for i, child in enumerate(node.children):
        if child.type == "->" and i + 1 < len(node.children):
            ret_type = node.children[i + 1]
            return get_node_text(ret_type)  # type: ignore[no-any-return]
    return None


def _extract_rust_function(node: Node, _lines: list[str]) -> ExtractedElement | None:
    """Extract a Rust function definition."""
    name = None
    is_async = False
    decorators: list[str] = []
    params_node = None

    # Collect preceding attributes (e.g., #[test], #[tokio::test])
    preceding_attrs = _get_preceding_attributes(node)
    decorators.extend(preceding_attrs)

    for child in node.children:
        if child.type == "identifier":
            name = get_node_text(child)
        elif child.type == "async":
            is_async = True
            if "async" not in decorators:
                decorators.append("async")
        elif child.type == "function_modifiers":
            mod_text = get_node_text(child)
            if "async" in mod_text:
                is_async = True
                if "async" not in decorators:
                    decorators.append("async")
        elif child.type == "parameters":
            params_node = child

    if not name:
        return None

    # Extract structured parameters and return type
    parameters = _extract_rust_parameters(params_node) if params_node else []
    return_type = _extract_rust_return_type(node)

    # Build signature
    signature = f"fn {name}"
    if params_node:
        signature += get_node_text(params_node)
    if return_type:
        signature += f" -> {return_type}"

    return ExtractedElement(
        element_type="function",
        name=name,
        line_start=node.start_point[0] + 1,
        line_end=node.end_point[0] + 1,
        raw_code=node.text.decode('utf-8') if node.text else "",
        byte_offset=node.start_byte,
        visibility=_extract_rust_visibility(node),
        signature=signature,
        is_async=is_async,
        decorators=decorators if decorators else None,
        node=node,
        return_type=return_type,
        parameters=parameters or None,
    )


def _extract_rust_impl(node: Node, _lines: list[str]) -> ExtractedElement | None:
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
            # For impl Trait for Type<'a> or impl<'a> Type<'a>
            for gt_child in child.children:
                if gt_child.type == "type_identifier":
                    if has_for:
                        # "impl Trait for GenericType<...>" — preserve trait name
                        if impl_type is not None:
                            trait_name = impl_type
                        impl_type = get_node_text(gt_child)
                    elif impl_type is None:
                        impl_type = get_node_text(gt_child)
                    else:
                        trait_name = impl_type
                        impl_type = get_node_text(gt_child)
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
        raw_code=node.text.decode('utf-8') if node.text else "",
        byte_offset=node.start_byte,
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


def extract_rust_trait_members(
    trait_node: Node, lines: list[str]
) -> list[ExtractedElement]:
    """Extract method signatures and default methods from a Rust trait block.

    Args:
        trait_node: A trait_item node.
        lines: Source code lines.

    Returns:
        List of method elements.
    """
    methods: list[ExtractedElement] = []

    # Find declaration_list
    decl_list = None
    for child in trait_node.children:
        if child.type == "declaration_list":
            decl_list = child
            break

    if not decl_list:
        return methods

    for child in decl_list.children:
        if child.type in ("function_item", "function_signature_item"):
            elem = _extract_rust_method(child, lines)
            if elem:
                methods.append(elem)

    return methods


def _extract_rust_method(node: Node, _lines: list[str]) -> ExtractedElement | None:
    """Extract a Rust method from an impl or trait block."""
    name = None
    is_async = False
    decorators: list[str] = []
    has_self = False
    params_node = None

    # Collect preceding attributes (e.g., #[test], #[tokio::test])
    preceding_attrs = _get_preceding_attributes(node)
    decorators.extend(preceding_attrs)

    for child in node.children:
        if child.type == "identifier":
            name = get_node_text(child)
        elif child.type == "async":
            is_async = True
            if "async" not in decorators:
                decorators.append("async")
        elif child.type == "function_modifiers":
            # function_signature_item wraps modifiers (e.g. async) in this node
            mod_text = get_node_text(child)
            if "async" in mod_text:
                is_async = True
                if "async" not in decorators:
                    decorators.append("async")
        elif child.type == "parameters":
            params_node = child
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

    # Extract structured parameters and return type
    parameters = _extract_rust_parameters(params_node) if params_node else []
    return_type = _extract_rust_return_type(node)

    # Build signature
    signature = f"fn {name}"
    if params_node:
        signature += get_node_text(params_node)
    if return_type:
        signature += f" -> {return_type}"

    # Method vs associated function
    elem_type = "method" if has_self else "function"

    return ExtractedElement(
        element_type=elem_type,
        name=name,
        line_start=node.start_point[0] + 1,
        line_end=node.end_point[0] + 1,
        raw_code=node.text.decode('utf-8') if node.text else "",
        byte_offset=node.start_byte,
        visibility=_extract_rust_visibility(node),
        signature=signature,
        is_async=is_async,
        decorators=decorators,
        node=node,
        return_type=return_type,
        parameters=parameters or None,
    )


def _extract_rust_const(node: Node, _lines: list[str]) -> ExtractedElement | None:
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
        raw_code=node.text.decode('utf-8') if node.text else "",
        byte_offset=node.start_byte,
    )


# =============================================================================
# RUST IMPORT EXTRACTION
# =============================================================================


def extract_rust_imports(
    tree: Tree, lines: list[str]  # noqa: ARG001
) -> list[ExtractedImport]:
    """Extract use statements from Rust code.

    Args:
        tree: Parsed tree-sitter tree.
        lines: Source code lines (unused but kept for API consistency).

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
                        if fe_child.type == "self":
                            receiver = "self"
                        elif fe_child.type == "identifier":
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


# Node types that define function/class scopes in Rust
_RUST_SCOPE_TYPES = frozenset({
    "function_item",
    "impl_item",
    "struct_item",
    "enum_item",
    "trait_item",
})


def extract_top_level_rust_calls(tree: Tree) -> list[ExtractedCall]:
    """Extract function/method calls from the top level of a Rust module.

    Walks file-scope statements (skipping function/impl/struct subtrees)
    and collects calls using the same extraction logic as function-body calls.

    Args:
        tree: A parsed tree-sitter Tree for a Rust file.

    Returns:
        List of extracted calls found at module scope.
    """
    calls: list[ExtractedCall] = []
    seen: set[tuple[str, str | None, int]] = set()

    for node in walk_top_level(tree.root_node, _RUST_SCOPE_TYPES):
        if node.type == "call_expression":
            func_node = node.children[0] if node.children else None
            line = node.start_point[0] + 1

            if func_node:
                if func_node.type == "identifier":
                    name = get_node_text(func_node)
                    key = (name, None, line)
                    if key not in seen:
                        seen.add(key)
                        calls.append(ExtractedCall(name=name, receiver=None, line=line))

                elif func_node.type == "field_expression":
                    receiver = None
                    method = None
                    for fe_child in func_node.children:
                        if fe_child.type == "self":
                            receiver = "self"
                        elif fe_child.type == "identifier":
                            receiver = get_node_text(fe_child)
                        elif fe_child.type == "field_identifier":
                            method = get_node_text(fe_child)

                    if method:
                        key = (method, receiver, line)
                        if key not in seen:
                            seen.add(key)
                            calls.append(ExtractedCall(name=method, receiver=receiver, line=line))

                elif func_node.type == "scoped_identifier":
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
