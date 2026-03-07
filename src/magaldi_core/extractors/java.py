"""Java language extractor using tree-sitter.

This module provides the JavaExtractor class and standalone functions
for extracting code elements, imports, references, call graph data,
and enhanced context from Java source code.
"""

from __future__ import annotations

from tree_sitter import Node, Tree

from magaldi_core.extractors.base import (
    BaseExtractor,
    get_child_by_field,
    get_node_text,
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
# JAVA EXTRACTOR CLASS
# =============================================================================


class JavaExtractor(BaseExtractor):
    """Extractor for Java source code."""

    @property
    def language(self) -> str:
        return "java"

    def extract_elements(self, tree: Tree, lines: list[str]) -> list[ExtractedElement]:
        """Extract code elements from a Java AST."""
        return extract_java_elements(tree, lines)

    def extract_class_members(
        self, class_node: Node, lines: list[str]
    ) -> tuple[list[ExtractedElement], list[ExtractedElement]]:
        """Extract methods and fields from a Java class body."""
        return extract_java_class_members(class_node, lines)

    def extract_imports(self, tree: Tree, lines: list[str]) -> list[ExtractedImport]:
        """Extract import statements from Java AST."""
        return extract_java_imports(tree, lines)

    def extract_references(
        self,
        tree: Tree,  # noqa: ARG002
        lines: list[str],  # noqa: ARG002
    ) -> list[ExtractedReference]:
        """Extract references from Java AST (stub - returns empty list)."""
        return []

    def extract_calls(self, function_node: Node) -> list[ExtractedCall]:
        """Extract method calls from within a Java method body."""
        return extract_java_calls(function_node)

    def extract_class_attributes(
        self, class_node: Node
    ) -> list[dict[str, str | int]]:
        """Extract fields from a Java class."""
        return extract_java_class_fields(class_node)

    def extract_base_classes(self, class_node: Node) -> list[str]:
        """Extract extends/implements from Java class."""
        return extract_java_base_classes(class_node)

    def extract_raised_exceptions(self, function_node: Node) -> list[str]:
        """Extract throw statement exception types from Java method."""
        return extract_java_thrown_exceptions(function_node)

    def extract_modified_attributes(self, method_node: Node) -> list[str]:
        """Extract this.field assignments in a method."""
        return extract_java_modified_attributes(method_node)


# =============================================================================
# JAVA ELEMENT EXTRACTION
# =============================================================================


def extract_java_elements(
    tree: Tree, lines: list[str], _file_path: str | None = None
) -> list[ExtractedElement]:
    """Extract classes, interfaces, enums, and top-level elements from Java code.

    Args:
        tree: Parsed tree-sitter tree.
        lines: Source code lines.
        _file_path: Optional file path for logging purposes.

    Returns:
        List of extracted elements.
    """
    elements: list[ExtractedElement] = []
    root = tree.root_node

    for node in root.children:
        if node.type == "class_declaration":
            elem = _extract_java_class(node, lines)
            if elem:
                elements.append(elem)
        elif node.type == "interface_declaration":
            elem = _extract_java_interface(node, lines)
            if elem:
                elements.append(elem)
        elif node.type == "enum_declaration":
            elem = _extract_java_enum(node, lines)
            if elem:
                elements.append(elem)
        elif node.type == "record_declaration":
            elem = _extract_java_record(node, lines)
            if elem:
                elements.append(elem)
        elif node.type == "annotation_type_declaration":
            elem = _extract_java_annotation_type(node, lines)
            if elem:
                elements.append(elem)
        elif node.type == "import_declaration":
            elem = _extract_java_import_element(node, lines)
            if elem:
                elements.append(elem)

    return elements


def _extract_java_class(node: Node, _lines: list[str]) -> ExtractedElement | None:
    """Extract a Java class declaration."""
    name = _get_identifier(node)
    if not name:
        return None

    modifiers = _get_modifiers(node)
    annotations = _get_annotations(node)
    signature = _build_class_signature(node, name, "class")

    return ExtractedElement(
        element_type="class",
        name=name,
        line_start=node.start_point[0] + 1,
        line_end=node.end_point[0] + 1,
        raw_code=node.text.decode("utf-8") if node.text else "",
        byte_offset=node.start_byte,
        node=node,
        signature=signature,
        decorators=annotations + modifiers,
    )


def _extract_java_interface(node: Node, _lines: list[str]) -> ExtractedElement | None:
    """Extract a Java interface declaration."""
    name = _get_identifier(node)
    if not name:
        return None

    modifiers = _get_modifiers(node)
    annotations = _get_annotations(node)
    signature = _build_class_signature(node, name, "interface")

    return ExtractedElement(
        element_type="interface",
        name=name,
        line_start=node.start_point[0] + 1,
        line_end=node.end_point[0] + 1,
        raw_code=node.text.decode("utf-8") if node.text else "",
        byte_offset=node.start_byte,
        node=node,
        signature=signature,
        decorators=annotations + modifiers,
    )


def _extract_java_enum(node: Node, _lines: list[str]) -> ExtractedElement | None:
    """Extract a Java enum declaration."""
    name = _get_identifier(node)
    if not name:
        return None

    modifiers = _get_modifiers(node)
    annotations = _get_annotations(node)

    return ExtractedElement(
        element_type="enum",
        name=name,
        line_start=node.start_point[0] + 1,
        line_end=node.end_point[0] + 1,
        raw_code=node.text.decode("utf-8") if node.text else "",
        byte_offset=node.start_byte,
        node=node,
        decorators=annotations + modifiers,
    )


def _extract_java_record(node: Node, _lines: list[str]) -> ExtractedElement | None:
    """Extract a Java record declaration (Java 16+)."""
    name = _get_identifier(node)
    if not name:
        return None

    modifiers = _get_modifiers(node)
    annotations = _get_annotations(node)

    # Extract record components as parameters
    params = _extract_formal_parameters(node)
    signature = f"record {name}({', '.join(f'{p.type} {p.name}' for p in params)})"

    return ExtractedElement(
        element_type="class",
        name=name,
        line_start=node.start_point[0] + 1,
        line_end=node.end_point[0] + 1,
        raw_code=node.text.decode("utf-8") if node.text else "",
        byte_offset=node.start_byte,
        node=node,
        signature=signature,
        parameters=params,
        decorators=annotations + modifiers + ["record"],
    )


def _extract_java_annotation_type(node: Node, _lines: list[str]) -> ExtractedElement | None:
    """Extract a Java annotation type declaration (@interface)."""
    name = _get_identifier(node)
    if not name:
        return None

    modifiers = _get_modifiers(node)

    return ExtractedElement(
        element_type="interface",
        name=name,
        line_start=node.start_point[0] + 1,
        line_end=node.end_point[0] + 1,
        raw_code=node.text.decode("utf-8") if node.text else "",
        byte_offset=node.start_byte,
        node=node,
        decorators=modifiers + ["@interface"],
    )


def _extract_java_import_element(node: Node, _lines: list[str]) -> ExtractedElement | None:
    """Extract an import declaration as an element."""
    raw_code = node.text.decode("utf-8") if node.text else ""
    module = _extract_import_path(node)
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


# =============================================================================
# CLASS MEMBER EXTRACTION
# =============================================================================


def extract_java_class_members(
    class_node: Node, lines: list[str]
) -> tuple[list[ExtractedElement], list[ExtractedElement]]:
    """Extract methods and fields from a Java class/interface/enum body.

    Args:
        class_node: A class/interface/enum declaration node.
        lines: Source code lines.

    Returns:
        Tuple of (methods, fields/constants).
    """
    methods: list[ExtractedElement] = []
    fields: list[ExtractedElement] = []

    body = _get_body_node(class_node)
    if not body:
        return methods, fields

    for child in body.children:
        if child.type == "method_declaration":
            elem = _extract_java_method(child, lines)
            if elem:
                methods.append(elem)
        elif child.type == "constructor_declaration":
            elem = _extract_java_constructor(child, lines)
            if elem:
                methods.append(elem)
        elif child.type == "field_declaration":
            field_elems = _extract_java_field(child, lines)
            fields.extend(field_elems)
        elif child.type == "enum_body_declarations":
            # Methods inside enum body
            for enum_child in child.children:
                if enum_child.type == "method_declaration":
                    elem = _extract_java_method(enum_child, lines)
                    if elem:
                        methods.append(elem)
                elif enum_child.type == "constructor_declaration":
                    elem = _extract_java_constructor(enum_child, lines)
                    if elem:
                        methods.append(elem)
                elif enum_child.type == "field_declaration":
                    field_elems = _extract_java_field(enum_child, lines)
                    fields.extend(field_elems)

    return methods, fields


def _extract_java_method(node: Node, _lines: list[str]) -> ExtractedElement | None:
    """Extract a Java method declaration."""
    name = _get_identifier(node)
    if not name:
        return None

    modifiers = _get_modifiers(node)
    annotations = _get_annotations(node)
    params = _extract_formal_parameters(node)
    return_type = _extract_return_type(node)
    is_async = False  # Java doesn't have async keyword

    # Build signature
    param_str = ", ".join(
        f"{p.type} {p.name}" if p.type else p.name for p in params
    )
    ret_str = return_type or "void"
    signature = f"{ret_str} {name}({param_str})"

    return ExtractedElement(
        element_type="method",
        name=name,
        line_start=node.start_point[0] + 1,
        line_end=node.end_point[0] + 1,
        raw_code=node.text.decode("utf-8") if node.text else "",
        byte_offset=node.start_byte,
        node=node,
        signature=signature,
        parameters=params,
        return_type=return_type,
        is_async=is_async,
        decorators=annotations + modifiers,
    )


def _extract_java_constructor(node: Node, _lines: list[str]) -> ExtractedElement | None:
    """Extract a Java constructor declaration."""
    name = _get_identifier(node)
    if not name:
        return None

    modifiers = _get_modifiers(node)
    annotations = _get_annotations(node)
    params = _extract_formal_parameters(node)

    param_str = ", ".join(
        f"{p.type} {p.name}" if p.type else p.name for p in params
    )
    signature = f"{name}({param_str})"

    return ExtractedElement(
        element_type="method",
        name=name,
        line_start=node.start_point[0] + 1,
        line_end=node.end_point[0] + 1,
        raw_code=node.text.decode("utf-8") if node.text else "",
        byte_offset=node.start_byte,
        node=node,
        signature=signature,
        parameters=params,
        decorators=annotations + modifiers,
    )


def _extract_java_field(
    node: Node, _lines: list[str]
) -> list[ExtractedElement]:
    """Extract field declarations from a Java class.

    A single field_declaration can declare multiple variables:
    private int x, y;
    """
    fields: list[ExtractedElement] = []
    modifiers = _get_modifiers(node)
    annotations = _get_annotations(node)

    is_constant = "static" in modifiers and "final" in modifiers

    for child in node.children:
        if child.type == "variable_declarator":
            var_name = get_child_by_field(child, "name")
            if var_name:
                name = get_node_text(var_name)
                fields.append(
                    ExtractedElement(
                        element_type="constant" if is_constant else "variable",
                        name=name,
                        line_start=node.start_point[0] + 1,
                        line_end=node.end_point[0] + 1,
                        raw_code=node.text.decode("utf-8") if node.text else "",
                        byte_offset=node.start_byte,
                        node=node,
                        decorators=annotations + modifiers,
                    )
                )

    return fields


# =============================================================================
# IMPORT EXTRACTION
# =============================================================================


def extract_java_imports(
    tree: Tree, _lines: list[str]
) -> list[ExtractedImport]:
    """Extract import statements from a Java AST.

    Handles:
    - import java.util.List;  (regular import)
    - import java.util.*;  (wildcard import)
    - import static java.util.Arrays.asList;  (static import)

    Args:
        tree: Parsed tree-sitter tree.
        _lines: Source code lines (unused).

    Returns:
        List of extracted imports.
    """
    imports: list[ExtractedImport] = []

    for node in tree.root_node.children:
        if node.type != "import_declaration":
            continue

        full_path = _extract_import_path(node)
        if not full_path:
            continue

        is_static = any(c.type == "static" for c in node.children)
        is_wildcard = full_path.endswith(".*") or any(
            c.type == "asterisk" for c in node.children
        )

        if is_wildcard:
            # import java.util.* → name="*", module="java.util"
            module = full_path.rstrip(".*") if full_path.endswith(".*") else full_path
            name = "*"
        else:
            # import java.util.List → name="List", module="java.util.List"
            parts = full_path.rsplit(".", 1)
            name = parts[-1] if len(parts) > 1 else full_path
            module = full_path

        # Use alias field to mark static imports
        alias = "static" if is_static else None

        imports.append(
            ExtractedImport(
                name=name,
                module=module,
                alias=alias,
                line=node.start_point[0] + 1,
            )
        )

    return imports


# =============================================================================
# CALL EXTRACTION
# =============================================================================


def extract_java_calls(function_node: Node) -> list[ExtractedCall]:
    """Extract method calls from within a Java method body.

    Handles:
    - obj.method()  → receiver=obj, name=method
    - method()  → receiver=None, name=method
    - new ClassName()  → receiver=None, name=ClassName
    - super.method()  → receiver=super, name=method
    - this.method()  → receiver=this, name=method
    - ClassName.staticMethod()  → receiver=ClassName, name=staticMethod

    Args:
        function_node: A method/constructor declaration node.

    Returns:
        List of extracted calls.
    """
    calls: list[ExtractedCall] = []
    seen: set[tuple[str, str | None, int]] = set()

    for node in walk_tree(function_node):
        if node.type == "method_invocation":
            call = _extract_method_invocation(node)
            if call:
                key = (call.name, call.receiver, call.line)
                if key not in seen:
                    seen.add(key)
                    calls.append(call)

        elif node.type == "object_creation_expression":
            call = _extract_constructor_call(node)
            if call:
                key = (call.name, call.receiver, call.line)
                if key not in seen:
                    seen.add(key)
                    calls.append(call)

    return calls


def extract_top_level_java_calls(tree: Tree) -> list[ExtractedCall]:
    """Extract calls at the file/class level (outside methods).

    In Java, most code is inside methods, so this typically returns empty.
    But static initializers or field initializers can contain calls.
    """
    calls: list[ExtractedCall] = []
    seen: set[tuple[str, str | None, int]] = set()

    for node in tree.root_node.children:
        if node.type in (
            "class_declaration",
            "interface_declaration",
            "enum_declaration",
            "record_declaration",
        ):
            body = _get_body_node(node)
            if not body:
                continue
            for child in body.children:
                if child.type == "field_declaration":
                    for call_node in walk_tree(child):
                        if call_node.type == "method_invocation":
                            call = _extract_method_invocation(call_node)
                            if call:
                                key = (call.name, call.receiver, call.line)
                                if key not in seen:
                                    seen.add(key)
                                    calls.append(call)
                        elif call_node.type == "object_creation_expression":
                            call = _extract_constructor_call(call_node)
                            if call:
                                key = (call.name, call.receiver, call.line)
                                if key not in seen:
                                    seen.add(key)
                                    calls.append(call)

    return calls


def _extract_method_invocation(node: Node) -> ExtractedCall | None:
    """Extract a method invocation call.

    Patterns:
    - method_invocation: method()  → identifier.identifier(args)
    - method_invocation: obj.method()  → identifier.identifier(args)
    - method_invocation: super.method()  → super.identifier(args)
    - method_invocation: this.method()  → this.identifier(args)
    - method_invocation: Cls.method()  → identifier.identifier(args)
    - method_invocation: expr.method()  → chained
    """
    # Find the method name (last identifier before argument_list)
    method_name = None
    receiver = None

    children = list(node.children)

    # Find the identifier that is the method name
    # It's typically the identifier right before the argument_list
    # In tree-sitter-java, method_invocation has structure:
    #   [receiver.] identifier argument_list
    for i, child in enumerate(children):
        if child.type == "argument_list":
            # The identifier right before this is the method name
            if i > 0 and children[i - 1].type == "identifier":
                method_name = get_node_text(children[i - 1])
            break

    if not method_name:
        return None

    # Find receiver: everything before the dot
    for i, child in enumerate(children):
        if child.type == ".":
            # The part before the dot is the receiver
            if i > 0:
                prev = children[i - 1]
                if prev.type == "identifier":
                    receiver = get_node_text(prev)
                elif prev.type == "this":
                    receiver = "this"
                elif prev.type == "super":
                    receiver = "super"
                elif prev.type == "field_access":
                    # e.g., System.out.println → receiver = "System.out"
                    receiver = get_node_text(prev)
                elif prev.type == "method_invocation":
                    # Chained: foo().bar() → receiver extracted from chain
                    receiver = _get_chained_receiver(prev)
            break

    return ExtractedCall(
        name=method_name,
        receiver=receiver,
        line=node.start_point[0] + 1,
    )


def _get_chained_receiver(node: Node) -> str | None:
    """Get receiver from a chained method invocation for display."""
    # For chained calls like a.b().c(), get the first identifier
    for child in node.children:
        if child.type == "identifier":
            return get_node_text(child)
        if child.type == "this":
            return "this"
        if child.type == "super":
            return "super"
    return None


def _extract_constructor_call(node: Node) -> ExtractedCall | None:
    """Extract a constructor invocation (new ClassName(...))."""
    # object_creation_expression: new type_identifier argument_list
    type_name = None
    for child in node.children:
        if child.type == "type_identifier":
            type_name = get_node_text(child)
            break
        if child.type == "generic_type":
            # new ArrayList<String>()
            for gc in child.children:
                if gc.type == "type_identifier":
                    type_name = get_node_text(gc)
                    break
            break
        if child.type == "scoped_type_identifier":
            type_name = get_node_text(child)
            break

    if not type_name:
        return None

    return ExtractedCall(
        name=type_name,
        receiver=None,
        line=node.start_point[0] + 1,
    )


# =============================================================================
# CLASS ATTRIBUTE EXTRACTION
# =============================================================================


def extract_java_class_fields(class_node: Node) -> list[dict[str, str | int]]:
    """Extract field declarations from a Java class body.

    Args:
        class_node: A class declaration node.

    Returns:
        List of dicts with "name" and "line" keys.
    """
    fields: list[dict[str, str | int]] = []
    body = _get_body_node(class_node)
    if not body:
        return fields

    for child in body.children:
        if child.type == "field_declaration":
            for vc in child.children:
                if vc.type == "variable_declarator":
                    var_name = get_child_by_field(vc, "name")
                    if var_name:
                        fields.append({
                            "name": get_node_text(var_name),
                            "line": child.start_point[0] + 1,
                        })

    return fields


# =============================================================================
# BASE CLASS / INTERFACE EXTRACTION
# =============================================================================


def extract_java_base_classes(class_node: Node) -> list[str]:
    """Extract extends and implements from a Java class/interface.

    Args:
        class_node: A class/interface declaration node.

    Returns:
        List of base class/interface names.
    """
    bases: list[str] = []

    for child in class_node.children:
        if child.type == "superclass":
            # extends ClassName
            for sc in child.children:
                if sc.type == "type_identifier":
                    bases.append(get_node_text(sc))
                elif sc.type == "generic_type":
                    for gc in sc.children:
                        if gc.type == "type_identifier":
                            bases.append(get_node_text(gc))
                            break

        elif child.type == "super_interfaces":
            # implements Interface1, Interface2
            for sc in child.children:
                if sc.type == "type_list":
                    for tc in sc.children:
                        if tc.type == "type_identifier":
                            bases.append(get_node_text(tc))
                        elif tc.type == "generic_type":
                            for gc in tc.children:
                                if gc.type == "type_identifier":
                                    bases.append(get_node_text(gc))
                                    break

        elif child.type == "extends_interfaces":
            # interface Foo extends Bar, Baz
            for sc in child.children:
                if sc.type == "type_list":
                    for tc in sc.children:
                        if tc.type == "type_identifier":
                            bases.append(get_node_text(tc))
                        elif tc.type == "generic_type":
                            for gc in tc.children:
                                if gc.type == "type_identifier":
                                    bases.append(get_node_text(gc))
                                    break

    return bases


# =============================================================================
# EXCEPTION EXTRACTION
# =============================================================================


def extract_java_thrown_exceptions(function_node: Node) -> list[str]:
    """Extract exception types from throw statements in a Java method.

    Args:
        function_node: A method/constructor declaration node.

    Returns:
        List of exception type names.
    """
    exceptions: list[str] = []
    seen: set[str] = set()

    for node in walk_tree(function_node):
        if node.type == "throw_statement":
            # throw new ExceptionType(...)
            for child in node.children:
                if child.type == "object_creation_expression":
                    for gc in child.children:
                        if gc.type == "type_identifier":
                            exc_name = get_node_text(gc)
                            if exc_name not in seen:
                                seen.add(exc_name)
                                exceptions.append(exc_name)
                            break

    return exceptions


# =============================================================================
# MODIFIED ATTRIBUTE EXTRACTION
# =============================================================================


def extract_java_modified_attributes(method_node: Node) -> list[str]:
    """Extract this.field assignments in a Java method.

    Finds patterns like: this.name = value;

    Args:
        method_node: A method/constructor declaration node.

    Returns:
        List of field names that are modified.
    """
    modified: list[str] = []
    seen: set[str] = set()

    for node in walk_tree(method_node):
        if node.type == "assignment_expression":
            left = get_child_by_field(node, "left")
            if left and left.type == "field_access":
                # Check if it's this.field
                obj = None
                field_name = None
                for child in left.children:
                    if child.type == "this":
                        obj = "this"
                    elif child.type == "identifier" and obj == "this":
                        field_name = get_node_text(child)

                if field_name and field_name not in seen:
                    seen.add(field_name)
                    modified.append(field_name)

    return modified


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _get_identifier(node: Node) -> str | None:
    """Get the identifier (name) from a declaration node."""
    for child in node.children:
        if child.type == "identifier":
            return get_node_text(child)
    return None


def _get_body_node(node: Node) -> Node | None:
    """Get the body node (class_body, interface_body, enum_body) from a declaration."""
    for child in node.children:
        if child.type in ("class_body", "interface_body", "enum_body"):
            return child
    return None


def _get_modifiers(node: Node) -> list[str]:
    """Extract modifiers (public, private, static, final, etc.) from a declaration."""
    modifiers: list[str] = []
    for child in node.children:
        if child.type == "modifiers":
            for mod in child.children:
                if mod.type in (
                    "public", "private", "protected", "static", "final",
                    "abstract", "synchronized", "native", "transient", "volatile",
                    "default", "strictfp", "sealed", "non-sealed",
                ):
                    modifiers.append(mod.type)
                elif mod.type == "annotation":
                    # Skip annotations here — handled by _get_annotations
                    pass
    return modifiers


def _get_annotations(node: Node) -> list[str]:
    """Extract annotation names from a declaration's modifiers.

    Returns annotation names prefixed with @, e.g., ["@Override", "@Deprecated"].
    """
    annotations: list[str] = []
    for child in node.children:
        if child.type == "modifiers":
            for mod in child.children:
                if mod.type in ("annotation", "marker_annotation"):
                    ann_name = _extract_annotation_name(mod)
                    if ann_name:
                        annotations.append(f"@{ann_name}")
    return annotations


def _extract_annotation_name(node: Node) -> str | None:
    """Extract the name from an annotation node."""
    for child in node.children:
        if child.type == "identifier":
            return get_node_text(child)
        if child.type == "scoped_identifier":
            return get_node_text(child)
    return None


def _extract_formal_parameters(node: Node) -> list[ParameterInfo]:
    """Extract formal parameters from a method/constructor/record declaration."""
    params: list[ParameterInfo] = []
    for child in node.children:
        if child.type == "formal_parameters":
            for param in child.children:
                if param.type == "formal_parameter":
                    p = _extract_single_parameter(param)
                    if p:
                        params.append(p)
                elif param.type == "spread_parameter":
                    # varargs: String... args
                    p = _extract_single_parameter(param)
                    if p:
                        params.append(p)
    return params


def _extract_single_parameter(param_node: Node) -> ParameterInfo | None:
    """Extract a single parameter from a formal_parameter node."""
    param_type = None
    param_name = None

    for child in param_node.children:
        if child.type == "identifier":
            param_name = get_node_text(child)
        elif child.type in (
            "type_identifier", "integral_type", "floating_point_type",
            "boolean_type", "void_type", "generic_type",
            "array_type", "scoped_type_identifier",
        ):
            param_type = get_node_text(child)
        elif child.type == "modifiers":
            # Skip modifiers like final
            pass
        elif child.type == "..." and param_type:
            # varargs marker
            param_type = param_type + "..."

    if param_name:
        return ParameterInfo(name=param_name, type=param_type, default=None)
    return None


def _extract_return_type(node: Node) -> str | None:
    """Extract the return type from a method declaration."""
    for child in node.children:
        if child.type in (
            "type_identifier", "integral_type", "floating_point_type",
            "boolean_type", "void_type", "generic_type",
            "array_type", "scoped_type_identifier",
        ):
            return get_node_text(child)
    return None


def _extract_import_path(node: Node) -> str | None:
    """Extract the full dotted path from an import declaration."""
    for child in node.children:
        if child.type == "scoped_identifier":
            return get_node_text(child)
        if child.type == "identifier":
            return get_node_text(child)
    return None


def _build_class_signature(node: Node, name: str, kind: str) -> str:
    """Build a class/interface signature string.

    E.g., "class Stage implements Performable, Lightable"
    """
    parts = [kind, name]

    for child in node.children:
        if child.type == "type_parameters":
            parts.append(get_node_text(child))
        elif child.type == "superclass":
            for sc in child.children:
                if sc.type != "extends":
                    parts.append(f"extends {get_node_text(sc)}")
        elif child.type == "super_interfaces":
            iface_names = []
            for sc in child.children:
                if sc.type == "type_list":
                    for tc in sc.children:
                        if tc.type in ("type_identifier", "generic_type"):
                            iface_names.append(get_node_text(tc))
            if iface_names:
                parts.append(f"implements {', '.join(iface_names)}")
        elif child.type == "extends_interfaces":
            iface_names = []
            for sc in child.children:
                if sc.type == "type_list":
                    for tc in sc.children:
                        if tc.type in ("type_identifier", "generic_type"):
                            iface_names.append(get_node_text(tc))
            if iface_names:
                parts.append(f"extends {', '.join(iface_names)}")

    return " ".join(parts)
