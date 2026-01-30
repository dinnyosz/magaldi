"""PHP language extractor using tree-sitter.

This module provides the PHPExtractor class and standalone functions
for extracting code elements, imports, references, call graph data,
and enhanced context from PHP source code.
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
    HttpRoute,
    ParameterInfo,
)

# =============================================================================
# PHP EXTRACTOR CLASS
# =============================================================================


class PHPExtractor(BaseExtractor):
    """Extractor for PHP source code."""

    @property
    def language(self) -> str:
        return "php"

    def extract_elements(self, tree: Tree, lines: list[str]) -> list[ExtractedElement]:
        """Extract code elements from a PHP AST."""
        return extract_php_elements(tree, lines)

    def extract_class_members(
        self, class_node: Node, lines: list[str]
    ) -> tuple[list[ExtractedElement], list[ExtractedElement]]:
        """Extract methods and properties from a PHP class."""
        return extract_php_class_members(class_node, lines)

    def extract_imports(self, tree: Tree, lines: list[str]) -> list[ExtractedImport]:
        """Extract use statements from PHP AST."""
        return extract_php_imports(tree, lines)

    def extract_references(
        self,
        tree: Tree,  # noqa: ARG002
        lines: list[str],  # noqa: ARG002
    ) -> list[ExtractedReference]:
        """Extract references from PHP AST (stub - returns empty list)."""
        return []

    def extract_calls(self, function_node: Node) -> list[ExtractedCall]:
        """Extract function/method calls from within a PHP function body."""
        return extract_php_calls(function_node)

    def extract_class_attributes(
        self, class_node: Node
    ) -> list[dict[str, str | int]]:
        """Extract class properties from PHP class."""
        return extract_php_class_properties(class_node)

    def extract_base_classes(self, class_node: Node) -> list[str]:
        """Extract base class and interfaces from PHP class definition."""
        return extract_php_base_class(class_node)

    def extract_raised_exceptions(self, function_node: Node) -> list[str]:
        """Extract exception types from throw statements."""
        return extract_php_thrown_exceptions(function_node)

    def extract_modified_attributes(self, method_node: Node) -> list[str]:
        """Extract $this->X properties that are modified."""
        return extract_php_modified_properties(method_node)


# =============================================================================
# PHP ELEMENT EXTRACTION
# =============================================================================


def extract_php_elements(tree: Tree, lines: list[str]) -> list[ExtractedElement]:
    """Extract classes, functions, and imports from PHP code.

    Args:
        tree: Parsed tree-sitter tree.
        lines: Source code lines.

    Returns:
        List of extracted elements (classes, functions, imports).
    """
    elements: list[ExtractedElement] = []
    root = tree.root_node

    for node in root.children:
        if node.type == "namespace_use_declaration":
            # PHP use statements (imports)
            elem = _extract_php_use_statement(node, lines)
            if elem:
                elements.append(elem)
        elif node.type == "class_declaration":
            elem = _extract_php_class(node, lines)
            if elem:
                elements.append(elem)
        elif node.type == "function_definition":
            elem = _extract_php_function(node, lines)
            if elem:
                elements.append(elem)

    return elements


def _extract_php_use_statement(node: Node, lines: list[str]) -> ExtractedElement | None:
    """Extract a PHP use statement (import) as an element.

    Args:
        node: A namespace_use_declaration node.
        lines: Source code lines.

    Returns:
        ExtractedElement with element_type="import".
    """
    line_start = node.start_point[0] + 1
    line_end = node.end_point[0] + 1
    raw_code = node.text.decode("utf-8") if node.text else ""

    # Get the module name from the use clause
    module = ""
    for child in node.children:
        if child.type == "namespace_use_clause":
            for clause_child in child.children:
                if clause_child.type == "qualified_name":
                    module = get_node_text(clause_child)
                    break
            break

    if not module:
        return None

    return ExtractedElement(
        element_type="import",
        name=module,
        line_start=line_start,
        line_end=line_end,
        raw_code=raw_code,
        byte_offset=node.start_byte,
        signature=raw_code.strip(),
        node=node,
    )


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
        raw_code=node.text.decode('utf-8') if node.text else "",
        byte_offset=node.start_byte,
        node=node,
    )


def _extract_php_parameters(params_node: Node) -> list[ParameterInfo]:
    """Extract structured parameter info from PHP formal_parameters node.

    Args:
        params_node: The 'formal_parameters' AST node.

    Returns:
        List of ParameterInfo with name, type, and default value.
    """
    parameters: list[ParameterInfo] = []
    if not params_node:
        return parameters

    for child in params_node.children:
        if child.type not in ("simple_parameter", "variadic_parameter", "property_promotion_parameter"):
            continue

        param_name: str | None = None
        param_type: str | None = None
        param_default: str | None = None

        # Get variable name
        for c in child.children:
            if c.type == "variable_name":
                for nc in c.children:
                    if nc.type == "name":
                        param_name = get_node_text(nc)
                        break
            elif c.type in ("named_type", "primitive_type", "nullable_type", "union_type"):
                param_type = get_node_text(c)
            elif c.type == "variadic_placeholder":
                param_name = "..." + (param_name or "")

        # Get default value if present
        default_node = get_child_by_field(child, "default_value")
        if default_node:
            param_default = get_node_text(default_node)

        if param_name:
            parameters.append(ParameterInfo(
                name=param_name,
                type=param_type,
                default=param_default,
            ))

    return parameters


def _extract_php_return_type(node: Node) -> str | None:
    """Extract return type from PHP function/method node."""
    return_type_node = get_child_by_field(node, "return_type")
    if return_type_node:
        return get_node_text(return_type_node)
    return None


def _extract_php_function(node: Node, lines: list[str]) -> ExtractedElement | None:
    """Extract a PHP function definition."""
    name = None
    params_node = None

    for child in node.children:
        if child.type == "name":
            name = get_node_text(child)
        elif child.type == "formal_parameters":
            params_node = child

    if not name:
        return None

    # Extract structured parameters and return type
    parameters = _extract_php_parameters(params_node) if params_node else []
    return_type = _extract_php_return_type(node)

    # Build signature from formal_parameters
    signature = f"function {name}"
    if params_node:
        signature += get_node_text(params_node)
    if return_type:
        signature += f": {return_type}"

    return ExtractedElement(
        element_type="function",
        name=name,
        line_start=node.start_point[0] + 1,
        line_end=node.end_point[0] + 1,
        raw_code=node.text.decode('utf-8') if node.text else "",
        byte_offset=node.start_byte,
        signature=signature,
        node=node,
        return_type=return_type,
        parameters=parameters or None,
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
    params_node = None

    for child in node.children:
        if child.type == "name":
            name = get_node_text(child)
        elif child.type == "visibility_modifier":
            visibility = get_node_text(child)
        elif child.type == "static_modifier":
            decorators.append("static")
        elif child.type == "formal_parameters":
            params_node = child

    if not name:
        return None

    # Extract structured parameters and return type
    parameters = _extract_php_parameters(params_node) if params_node else []
    return_type = _extract_php_return_type(node)

    # Build signature
    signature = f"{visibility} function {name}"
    if params_node:
        signature += get_node_text(params_node)
    if return_type:
        signature += f": {return_type}"

    return ExtractedElement(
        element_type="method",
        name=name,
        line_start=node.start_point[0] + 1,
        line_end=node.end_point[0] + 1,
        raw_code=node.text.decode('utf-8') if node.text else "",
        byte_offset=node.start_byte,
        signature=signature,
        decorators=decorators,
        node=node,
        return_type=return_type,
        parameters=parameters or None,
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
                                raw_code=node.text.decode('utf-8') if node.text else "",
                                byte_offset=node.start_byte,
                                decorators=[visibility],
                            ))

    return properties


# =============================================================================
# PHP IMPORT EXTRACTION
# =============================================================================


def extract_php_imports(
    tree: Tree, lines: list[str]  # noqa: ARG001
) -> list[ExtractedImport]:
    """Extract use statements from PHP code.

    Args:
        tree: Parsed tree-sitter tree.
        lines: Source code lines (unused but kept for API consistency).

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


# =============================================================================
# PHP SLIM ROUTE DETECTION (delegated to frameworks module)
# =============================================================================

# Re-export for backwards compatibility
from magaldi_core.extractors.patterns.php.slim import (  # noqa: E402
    extract_slim_routes,
    extract_slim_route_groups,
)

__all__ = [
    "PHPExtractor",
    "extract_php_elements",
    "extract_php_class_members",
    "extract_php_imports",
    "extract_php_calls",
    "extract_php_class_properties",
    "extract_php_base_class",
    "extract_php_thrown_exceptions",
    "extract_php_modified_properties",
    # Slim framework support
    "extract_slim_routes",
    "extract_slim_route_groups",
]


# =============================================================================
# PHP CALL EXTRACTION
# =============================================================================


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
                        if oc_child.type == "name" or oc_child.type == "qualified_name":
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
