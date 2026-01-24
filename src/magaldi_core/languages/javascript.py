"""JavaScript and TypeScript tree-sitter extractors.

This module provides functions for extracting code elements from JavaScript
and TypeScript source files using tree-sitter parsing:

- Elements (classes, functions, arrow functions)
- Class members (methods, fields)
- Import statements (ES6 imports, CommonJS require)
- References (instantiations, function calls, type annotations)
- Call graph data
- Enhanced context (base classes, class fields, exceptions, modified properties)
"""

from __future__ import annotations

from tree_sitter import Node, Tree

from magaldi_core.tree_sitter_manager import (
    DecoratorInfo,
    ExtractedCall,
    ExtractedElement,
    ExtractedImport,
    ExtractedReference,
    find_nodes,
    get_child_by_field,
    get_children_by_type,
    get_node_text,
    walk_tree,
)


# =============================================================================
# JAVASCRIPT ELEMENT EXTRACTION
# =============================================================================


def extract_javascript_elements(tree: Tree, lines: list[str]) -> list[ExtractedElement]:
    """Extract code elements from a JavaScript/TypeScript AST."""
    elements: list[ExtractedElement] = []
    root = tree.root_node

    for node in walk_tree(root):
        if node.type == "class_declaration":
            elements.append(_extract_js_class(node, lines))
        elif node.type == "function_declaration":
            elements.append(_extract_js_function(node, lines))
        elif node.type == "lexical_declaration":
            # const/let declarations - check for arrow functions
            for decl in get_children_by_type(node, "variable_declarator"):
                name_node = get_child_by_field(decl, "name")
                value_node = get_child_by_field(decl, "value")
                if value_node and value_node.type == "arrow_function":
                    name = get_node_text(name_node) if name_node else "unknown"
                    elements.append(_extract_js_arrow_function(decl, name, lines))

    return elements


def _extract_js_class(node: Node, lines: list[str]) -> ExtractedElement:
    """Extract a JavaScript class."""
    name_node = get_child_by_field(node, "name")
    name = get_node_text(name_node) if name_node else "unknown"

    line_start = node.start_point[0] + 1
    line_end = node.end_point[0] + 1
    raw_code = "\n".join(lines[line_start - 1 : line_end])

    return ExtractedElement(
        element_type="class",
        name=name,
        line_start=line_start,
        line_end=line_end,
        raw_code=raw_code,
        node=node,
    )


def _extract_js_function(node: Node, lines: list[str]) -> ExtractedElement:
    """Extract a JavaScript function."""
    name_node = get_child_by_field(node, "name")
    name = get_node_text(name_node) if name_node else "unknown"

    params_node = get_child_by_field(node, "parameters")
    params = get_node_text(params_node) if params_node else "()"

    # Check for async
    is_async = any(child.type == "async" for child in node.children)

    line_start = node.start_point[0] + 1
    line_end = node.end_point[0] + 1
    raw_code = "\n".join(lines[line_start - 1 : line_end])

    signature = f"{'async ' if is_async else ''}function {name}{params}"

    return ExtractedElement(
        element_type="function",
        name=name,
        line_start=line_start,
        line_end=line_end,
        raw_code=raw_code,
        signature=signature,
        is_async=is_async,
        node=node,
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
    raw_code = "\n".join(lines[line_start - 1 : line_end])

    # Get the actual arrow function node for call extraction
    value_node = get_child_by_field(node, "value")
    arrow_func_node = value_node if value_node and value_node.type == "arrow_function" else node

    return ExtractedElement(
        element_type="function",
        name=name,
        line_start=line_start,
        line_end=line_end,
        raw_code=raw_code,
        node=arrow_func_node,
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

            is_async = any(c.type == "async" for c in child.children)

            line_start = child.start_point[0] + 1
            line_end = child.end_point[0] + 1
            raw_code = "\n".join(lines[line_start - 1 : line_end])

            methods.append(
                ExtractedElement(
                    element_type="method",
                    name=name,
                    line_start=line_start,
                    line_end=line_end,
                    raw_code=raw_code,
                    signature=f"{name}{params}",
                    is_async=is_async,
                    node=child,
                )
            )
        elif child.type == "field_definition":
            name_node = get_child_by_field(child, "property")
            name = get_node_text(name_node) if name_node else "unknown"

            line_start = child.start_point[0] + 1
            line_end = child.end_point[0] + 1
            raw_code = lines[line_start - 1].strip() if line_start <= len(lines) else ""

            fields.append(
                ExtractedElement(
                    element_type="variable",
                    name=name,
                    line_start=line_start,
                    line_end=line_end,
                    raw_code=raw_code,
                    node=child,
                )
            )

    return methods, fields


# =============================================================================
# JAVASCRIPT IMPORT EXTRACTION
# =============================================================================


def extract_javascript_imports(tree: Tree, lines: list[str]) -> list[ExtractedImport]:
    """Extract import statements from a JavaScript/TypeScript AST.

    Handles:
    - import { foo } from './utils' -> Import(name="foo", module="./utils", alias=None)
    - import { foo as bar } from './utils' -> Import(name="foo", module="./utils", alias="bar")
    - import utils from './utils' -> Import(name="utils", module="./utils", alias=None)
    - import * as utils from './utils' -> Import(name="*", module="./utils", alias="utils")
    - const bar = require('lib') -> Import(name="bar", module="lib", alias=None)

    Args:
        tree: Parsed tree-sitter Tree.
        lines: Source code lines.

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


def _is_likely_class_name(name: str) -> bool:
    """Check if name looks like a class (PascalCase)."""
    if not name:
        return False
    # Starts with uppercase, not all uppercase (to exclude constants like HTTP)
    return name[0].isupper() and not name.isupper()


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
    """Extract the base class name from a JavaScript/TypeScript class.

    For: class Foo extends Bar { ... }
    Returns: ["Bar"]

    Args:
        class_node: A class_declaration node from tree-sitter.

    Returns:
        List with single base class name, or empty list.
    """
    bases: list[str] = []

    if class_node.type != "class_declaration":
        return bases

    # Find class_heritage node which contains extends clause
    for child in class_node.children:
        if child.type == "class_heritage":
            # The base class identifier is directly under class_heritage
            for heritage_child in child.children:
                if heritage_child.type == "identifier":
                    bases.append(get_node_text(heritage_child))
                    break
                elif heritage_child.type == "member_expression":
                    # Handle qualified names like module.ClassName
                    bases.append(get_node_text(heritage_child))
                    break
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
