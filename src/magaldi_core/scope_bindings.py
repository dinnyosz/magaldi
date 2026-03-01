"""AST-based variable binding extraction for scope-aware type resolution.

Strategy 5.7: Extracts variable->type bindings from tree-sitter AST,
going beyond the regex-based approaches in Strategies 5.5 and 5.6.

Supported binding patterns per language:

Python:
- var = func()              (bare call assignment)
- var = receiver.method()   (method call assignment)
- var = ClassName()         (constructor assignment)
- with expr() as var:       (context manager)
- for var in expr:          (loop variable)
- except ExcType as var:    (exception handler)

JavaScript/TypeScript:
- const/let/var x = func()  (variable declaration with call)
- const x = obj.method()    (method call assignment)
- const x = new Class()     (constructor via new)
- for (const x of expr)     (for-of loop)
- for (const x in expr)     (for-in loop)
- catch (e) {}              (catch clause — no type info)

PHP:
- $var = func()             (assignment with call)
- $var = $obj->method()     (method call assignment)
- $var = new Class()        (constructor via new)
- foreach ($items as $item) (foreach loop)
- catch (Type $var) {}      (catch clause — HAS type info)

Rust:
- let x = func()            (let binding with call)
- let x = obj.method()      (method call binding)
- let x = Type::new()       (constructor binding)
- for x in expr             (for-in loop)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from tree_sitter import Node

from magaldi_core.extractors.base import (
    get_child_by_field,
    get_node_text,
    walk_tree,
)

logger = logging.getLogger(__name__)


# =============================================================================
# DATA MODEL
# =============================================================================


@dataclass
class BindingInfo:
    """A variable binding extracted from AST scope analysis.

    Represents a variable assignment pattern where we can infer the
    variable's type from the right-hand side expression.
    """

    variable: str  # variable name (e.g., "conn", "handle")
    source: str  # how the binding was created (see SOURCE_* constants)

    # For call-based bindings:
    call_name: str | None = None  # "connect", "open", "get_items"
    call_receiver: str | None = None  # "db", "client", None for bare calls

    # For direct type bindings (except_as, constructor):
    type_name: str | None = None  # "ValueError", "Repository"

    line: int = 0  # 1-indexed line number


# Binding source constants
SOURCE_ASSIGNMENT_CALL = "assignment_call"
SOURCE_ASSIGNMENT_METHOD_CALL = "assignment_method_call"
SOURCE_CONSTRUCTOR = "constructor"
SOURCE_WITH_AS = "with_as"
SOURCE_FOR_IN = "for_in"
SOURCE_EXCEPT_AS = "except_as"


# =============================================================================
# PER-LANGUAGE CONFIGURATION
# =============================================================================


# Node types that define nested scopes — bindings inside these
# are NOT part of the enclosing function's scope.
_SCOPE_TYPES = frozenset({"function_definition", "class_definition"})

_SCOPE_TYPES_BY_LANG: dict[str, frozenset[str]] = {
    "python": frozenset({"function_definition", "class_definition"}),
    "javascript": frozenset({
        "function_declaration", "class_declaration", "arrow_function",
        "method_definition", "function",
    }),
    "typescript": frozenset({
        "function_declaration", "class_declaration", "arrow_function",
        "method_definition", "function",
    }),
    "tsx": frozenset({
        "function_declaration", "class_declaration", "arrow_function",
        "method_definition", "function",
    }),
    "php": frozenset({
        "function_definition", "class_declaration", "method_declaration",
        "anonymous_function_creation_expression",
    }),
    "rust": frozenset({
        "function_item", "impl_item", "closure_expression",
    }),
}

# Tree-sitter grammar names for parsing (tsx uses typescript grammar)
_TS_GRAMMAR: dict[str, str] = {
    "tsx": "typescript",
}

# Function definition node types per language
_FUNC_DEF_TYPES: dict[str, frozenset[str]] = {
    "python": frozenset({"function_definition"}),
    "javascript": frozenset({
        "function_declaration", "arrow_function", "method_definition",
        "function", "generator_function_declaration",
    }),
    "typescript": frozenset({
        "function_declaration", "arrow_function", "method_definition",
        "function", "generator_function_declaration",
    }),
    "tsx": frozenset({
        "function_declaration", "arrow_function", "method_definition",
        "function", "generator_function_declaration",
    }),
    "php": frozenset({
        "function_definition", "method_declaration",
    }),
    "rust": frozenset({
        "function_item",
    }),
}

# Decorated/exported wrapper types per language
_WRAPPER_TYPES: dict[str, frozenset[str]] = {
    "python": frozenset({"decorated_definition"}),
    "javascript": frozenset({"export_statement"}),
    "typescript": frozenset({"export_statement"}),
    "tsx": frozenset({"export_statement"}),
    "php": frozenset(),
    "rust": frozenset(),
}


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================


def extract_variable_bindings(
    raw_code: str,
    language: str,
) -> list[BindingInfo]:
    """Extract variable bindings from a function's raw code using tree-sitter.

    Parses the raw_code (which is a complete function/method definition)
    with tree-sitter and walks the AST to find all variable binding
    patterns that could be used for type inference.

    Args:
        raw_code: Complete source code of a function/method element.
        language: Programming language.

    Returns:
        List of BindingInfo objects, ordered by line number.
        Later bindings for the same variable override earlier ones.
    """
    if language not in _SCOPE_TYPES_BY_LANG:
        return []

    if not raw_code or not raw_code.strip():
        return []

    # Lazy import to avoid circular imports and heavy init at module load
    from magaldi_core.tree_sitter_manager import get_manager

    # Use correct grammar name (tsx -> typescript)
    grammar = _TS_GRAMMAR.get(language, language)

    try:
        manager = get_manager()
        tree = manager.parse(raw_code.encode("utf-8"), grammar)
    except Exception:
        logger.debug("Failed to parse raw_code for scope bindings", exc_info=True)
        return []

    body_node = _find_function_body(tree.root_node, language)
    if not body_node:
        return []

    scope_types = _SCOPE_TYPES_BY_LANG.get(language, _SCOPE_TYPES)
    bindings: list[BindingInfo] = []

    for node in walk_tree(body_node):
        # Skip nodes inside nested function/class definitions
        if _is_inside_nested_scope(node, body_node, scope_types):
            continue

        binding = _extract_binding(node, language)
        if isinstance(binding, list):
            bindings.extend(binding)
        elif binding:
            bindings.append(binding)

    return bindings


# =============================================================================
# FUNCTION BODY FINDER (per-language)
# =============================================================================


def _find_function_body(root: Node, language: str = "python") -> Node | None:
    """Navigate parsed raw_code to the function body block.

    raw_code contains a complete function definition, so the parse tree is:
    root_node -> [wrapper ->] function_def -> body

    Args:
        root: Root node of the parsed tree.
        language: Programming language.

    Returns:
        The body (block) node of the function, or None if not found.
    """
    func_types = _FUNC_DEF_TYPES.get(language, _FUNC_DEF_TYPES["python"])
    wrapper_types = _WRAPPER_TYPES.get(language, frozenset())

    for child in root.children:
        if child.type in func_types:
            return get_child_by_field(child, "body")
        if child.type in wrapper_types:
            for sub in child.children:
                if sub.type in func_types:
                    return get_child_by_field(sub, "body")

    return None


def _is_inside_nested_scope(
    node: Node,
    body_node: Node,
    scope_types: frozenset[str] | None = None,
) -> bool:
    """Check if a node is inside a nested function/class definition.

    We only want bindings from the immediate function scope, not
    from nested definitions which have their own scope.
    """
    if scope_types is None:
        scope_types = _SCOPE_TYPES
    parent = node.parent
    while parent and parent != body_node:
        if parent.type in scope_types:
            return True
        parent = parent.parent
    return False


# =============================================================================
# BINDING DISPATCH (per-language)
# =============================================================================


def _extract_binding(
    node: Node, language: str
) -> BindingInfo | list[BindingInfo] | None:
    """Dispatch binding extraction based on language and node type."""
    if language == "python":
        return _extract_binding_python(node)
    if language in ("javascript", "typescript", "tsx"):
        return _extract_binding_js(node)
    if language == "php":
        return _extract_binding_php(node)
    if language == "rust":
        return _extract_binding_rust(node)
    return None


# =============================================================================
# PYTHON BINDINGS
# =============================================================================


def _extract_binding_python(
    node: Node,
) -> BindingInfo | list[BindingInfo] | None:
    """Extract Python-specific bindings."""
    if node.type == "assignment":
        return _extract_assignment_binding_python(node)
    if node.type == "with_statement":
        return _extract_with_bindings_python(node)
    if node.type == "for_statement":
        return _extract_for_binding_python(node)
    if node.type == "except_clause":
        return _extract_except_binding_python(node)
    return None


def _extract_call_info(
    call_node: Node,
) -> tuple[str | None, str | None, str | None]:
    """Extract call name, receiver, and constructor type from a Python call node.

    Returns:
        Tuple of (call_name, call_receiver, type_name).
        type_name is set only for PascalCase constructors.
    """
    func_node = get_child_by_field(call_node, "function")
    if not func_node:
        return None, None, None

    if func_node.type == "identifier":
        name = get_node_text(func_node)
        # Check if it's a constructor (PascalCase)
        if name and len(name) >= 2 and name[0].isupper() and not name.isupper():
            return name, None, name
        return name, None, None

    if func_node.type == "attribute":
        attr_node = get_child_by_field(func_node, "attribute")
        obj_node = get_child_by_field(func_node, "object")
        if attr_node and obj_node:
            method_name = get_node_text(attr_node)
            if obj_node.type == "identifier":
                receiver = get_node_text(obj_node)
                return method_name, receiver, None
            if obj_node.type == "attribute":
                # Nested attribute: a.b.method()
                receiver = get_node_text(obj_node)
                return method_name, receiver, None

    return None, None, None


def _unwrap_await(node: Node) -> Node:
    """Unwrap an await expression to get the inner expression.

    If the node is an `await` expression, returns the child expression.
    Otherwise returns the node unchanged.
    """
    if node.type in ("await", "await_expression"):
        for child in node.children:
            if child.type not in ("await", "await_expression"):
                return child
    return node


def _extract_assignment_binding_python(node: Node) -> BindingInfo | None:
    """Extract binding from a Python assignment node.

    Handles: var = func(), var = obj.method(), var = Class(),
    var = await func(), var = await obj.method()
    """
    left = get_child_by_field(node, "left")
    if not left or left.type != "identifier":
        return None

    right = get_child_by_field(node, "right")
    if not right:
        return None

    var_name = get_node_text(left)
    line = node.start_point[0] + 1

    right = _unwrap_await(right)

    if right.type != "call":
        return None

    call_name, call_receiver, type_name = _extract_call_info(right)
    if not call_name:
        return None

    if type_name:
        return BindingInfo(
            variable=var_name, source=SOURCE_CONSTRUCTOR,
            call_name=call_name, type_name=type_name, line=line,
        )

    if call_receiver:
        return BindingInfo(
            variable=var_name, source=SOURCE_ASSIGNMENT_METHOD_CALL,
            call_name=call_name, call_receiver=call_receiver, line=line,
        )

    return BindingInfo(
        variable=var_name, source=SOURCE_ASSIGNMENT_CALL,
        call_name=call_name, line=line,
    )


def _extract_with_bindings_python(node: Node) -> list[BindingInfo]:
    """Extract bindings from a Python with_statement."""
    bindings: list[BindingInfo] = []

    for child in walk_tree(node):
        if child.type == "as_pattern" and child.parent and child.parent.type == "with_item":
            value_node = None
            alias_node = None

            for sub in child.children:
                if sub.type == "call":
                    value_node = sub
                elif sub.type == "as_pattern_target":
                    for inner in sub.children:
                        if inner.type == "identifier":
                            alias_node = inner
                            break

            if alias_node:
                var_name = get_node_text(alias_node)
                line = child.start_point[0] + 1

                if value_node:
                    call_name, call_receiver, type_name = _extract_call_info(value_node)
                    bindings.append(BindingInfo(
                        variable=var_name, source=SOURCE_WITH_AS,
                        call_name=call_name, call_receiver=call_receiver,
                        type_name=type_name, line=line,
                    ))
                else:
                    bindings.append(BindingInfo(
                        variable=var_name, source=SOURCE_WITH_AS, line=line,
                    ))

    return bindings


def _extract_for_binding_python(node: Node) -> BindingInfo | None:
    """Extract binding from a Python for_statement."""
    left = get_child_by_field(node, "left")
    if not left or left.type != "identifier":
        return None

    right = get_child_by_field(node, "right")
    if not right:
        return None

    var_name = get_node_text(left)
    line = node.start_point[0] + 1

    if right.type == "call":
        call_name, call_receiver, type_name = _extract_call_info(right)
        return BindingInfo(
            variable=var_name, source=SOURCE_FOR_IN,
            call_name=call_name, call_receiver=call_receiver,
            type_name=type_name, line=line,
        )

    if right.type == "identifier":
        collection_name = get_node_text(right)
        return BindingInfo(
            variable=var_name, source=SOURCE_FOR_IN,
            call_receiver=collection_name, line=line,
        )

    return None


def _extract_except_binding_python(node: Node) -> BindingInfo | None:
    """Extract binding from a Python except_clause."""
    as_pattern = get_child_by_field(node, "value")
    if not as_pattern or as_pattern.type != "as_pattern":
        return None

    exc_type = None
    alias = None

    for child in as_pattern.children:
        if child.type in ("identifier", "attribute") and exc_type is None:
            exc_type = get_node_text(child)
        elif child.type == "as_pattern_target":
            for inner in child.children:
                if inner.type == "identifier":
                    alias = get_node_text(inner)
                    break

    if not alias or not exc_type:
        return None

    return BindingInfo(
        variable=alias, source=SOURCE_EXCEPT_AS,
        type_name=exc_type, line=node.start_point[0] + 1,
    )


# =============================================================================
# JAVASCRIPT / TYPESCRIPT BINDINGS
# =============================================================================


def _extract_call_info_js(
    call_node: Node,
) -> tuple[str | None, str | None, str | None]:
    """Extract call name, receiver, and constructor type from a JS call/new node.

    Returns:
        Tuple of (call_name, call_receiver, type_name).
    """
    # new ClassName() — the node IS the new_expression
    if call_node.type == "new_expression":
        constructor_node = get_child_by_field(call_node, "constructor")
        if constructor_node and constructor_node.type == "identifier":
            name = get_node_text(constructor_node)
            return name, None, name
        return None, None, None

    # call_expression
    func_node = get_child_by_field(call_node, "function")
    if not func_node:
        return None, None, None

    if func_node.type == "identifier":
        name = get_node_text(func_node)
        # PascalCase without 'new' = regular function in JS
        return name, None, None

    if func_node.type == "member_expression":
        prop_node = get_child_by_field(func_node, "property")
        obj_node = get_child_by_field(func_node, "object")
        if prop_node and obj_node:
            method_name = get_node_text(prop_node)
            if obj_node.type == "identifier":
                receiver = get_node_text(obj_node)
                return method_name, receiver, None
            if obj_node.type == "this":
                return method_name, "this", None
            if obj_node.type == "member_expression":
                receiver = get_node_text(obj_node)
                return method_name, receiver, None

    return None, None, None


def _extract_binding_js(
    node: Node,
) -> BindingInfo | list[BindingInfo] | None:
    """Extract JS/TS-specific bindings."""
    # Variable declarations: const x = func(), let x = new Class()
    if node.type in ("variable_declaration", "lexical_declaration"):
        return _extract_var_decl_binding_js(node)
    # Assignment expression (reassignment): x = func()
    if node.type == "assignment_expression":
        return _extract_assignment_binding_js(node)
    # for (const x of items) / for (const x in obj)
    if node.type in ("for_in_statement", "for_of_statement"):
        return _extract_for_binding_js(node)
    # catch (e) {}
    if node.type == "catch_clause":
        return _extract_catch_binding_js(node)
    return None


def _extract_var_decl_binding_js(node: Node) -> list[BindingInfo]:
    """Extract bindings from JS variable declarations.

    Handles: const x = func(), let x = new Class(), var x = obj.method()
    """
    bindings: list[BindingInfo] = []

    for child in node.children:
        if child.type != "variable_declarator":
            continue

        name_node = get_child_by_field(child, "name")
        value_node = get_child_by_field(child, "value")

        if not name_node or name_node.type != "identifier":
            continue
        if not value_node:
            continue

        var_name = get_node_text(name_node)
        line = child.start_point[0] + 1

        # Unwrap await
        value_node = _unwrap_await(value_node)

        # new_expression: const x = new Foo()
        if value_node.type == "new_expression":
            call_name, _, type_name = _extract_call_info_js(value_node)
            if type_name:
                bindings.append(BindingInfo(
                    variable=var_name, source=SOURCE_CONSTRUCTOR,
                    call_name=call_name, type_name=type_name, line=line,
                ))
            continue

        # call_expression: const x = func() or const x = obj.method()
        if value_node.type == "call_expression":
            call_name, call_receiver, _ = _extract_call_info_js(value_node)
            if not call_name:
                continue

            if call_receiver:
                bindings.append(BindingInfo(
                    variable=var_name, source=SOURCE_ASSIGNMENT_METHOD_CALL,
                    call_name=call_name, call_receiver=call_receiver, line=line,
                ))
            else:
                bindings.append(BindingInfo(
                    variable=var_name, source=SOURCE_ASSIGNMENT_CALL,
                    call_name=call_name, line=line,
                ))

    return bindings


def _extract_assignment_binding_js(node: Node) -> BindingInfo | None:
    """Extract binding from JS assignment expression (reassignment)."""
    left = get_child_by_field(node, "left")
    if not left or left.type != "identifier":
        return None

    right = get_child_by_field(node, "right")
    if not right:
        return None

    var_name = get_node_text(left)
    line = node.start_point[0] + 1

    right = _unwrap_await(right)

    if right.type == "new_expression":
        call_name, _, type_name = _extract_call_info_js(right)
        if type_name:
            return BindingInfo(
                variable=var_name, source=SOURCE_CONSTRUCTOR,
                call_name=call_name, type_name=type_name, line=line,
            )
        return None

    if right.type == "call_expression":
        call_name, call_receiver, _ = _extract_call_info_js(right)
        if not call_name:
            return None
        if call_receiver:
            return BindingInfo(
                variable=var_name, source=SOURCE_ASSIGNMENT_METHOD_CALL,
                call_name=call_name, call_receiver=call_receiver, line=line,
            )
        return BindingInfo(
            variable=var_name, source=SOURCE_ASSIGNMENT_CALL,
            call_name=call_name, line=line,
        )

    return None


def _extract_for_binding_js(node: Node) -> BindingInfo | None:
    """Extract binding from JS for-of/for-in statements.

    for (const item of getItems()) or for (const key in obj)
    """
    # The "left" field contains the declaration (e.g., "const item")
    left = get_child_by_field(node, "left")
    right = get_child_by_field(node, "right")

    if not left or not right:
        return None

    # Extract the variable name from the left side
    var_name = None
    if left.type == "identifier":
        var_name = get_node_text(left)
    elif left.type in ("variable_declaration", "lexical_declaration"):
        # const/let/var declaration inside for
        for child in left.children:
            if child.type == "variable_declarator":
                name_node = get_child_by_field(child, "name")
                if name_node and name_node.type == "identifier":
                    var_name = get_node_text(name_node)
                break

    if not var_name:
        return None

    line = node.start_point[0] + 1

    if right.type == "call_expression":
        call_name, call_receiver, type_name = _extract_call_info_js(right)
        return BindingInfo(
            variable=var_name, source=SOURCE_FOR_IN,
            call_name=call_name, call_receiver=call_receiver,
            type_name=type_name, line=line,
        )

    if right.type == "identifier":
        collection_name = get_node_text(right)
        return BindingInfo(
            variable=var_name, source=SOURCE_FOR_IN,
            call_receiver=collection_name, line=line,
        )

    return None


def _extract_catch_binding_js(node: Node) -> BindingInfo | None:
    """Extract binding from JS catch clause. No type info available."""
    # catch (e) {} — parameter is the error variable
    param_node = get_child_by_field(node, "parameter")
    if not param_node:
        return None

    if param_node.type == "identifier":
        var_name = get_node_text(param_node)
        return BindingInfo(
            variable=var_name, source=SOURCE_EXCEPT_AS,
            line=node.start_point[0] + 1,
        )

    return None


# =============================================================================
# PHP BINDINGS
# =============================================================================


def _extract_call_info_php(
    call_node: Node,
) -> tuple[str | None, str | None, str | None]:
    """Extract call info from PHP call nodes.

    Handles:
    - function_call_expression: func()
    - member_call_expression: $obj->method()
    - object_creation_expression: new ClassName()
    """
    if call_node.type == "object_creation_expression":
        # new ClassName()
        for child in call_node.children:
            if child.type == "name":
                name = get_node_text(child)
                return name, None, name
            if child.type == "qualified_name":
                full = get_node_text(child)
                name = full.split("\\")[-1]
                return name, None, name
        return None, None, None

    if call_node.type == "function_call_expression":
        func_node = get_child_by_field(call_node, "function")
        if func_node:
            if func_node.type == "name":
                name = get_node_text(func_node)
                return name, None, None
            if func_node.type == "qualified_name":
                full = get_node_text(func_node)
                name = full.split("\\")[-1]
                return name, None, None
        return None, None, None

    if call_node.type == "member_call_expression":
        obj_node = get_child_by_field(call_node, "object")
        name_node = get_child_by_field(call_node, "name")
        if obj_node and name_node:
            method_name = get_node_text(name_node)
            if obj_node.type == "variable_name":
                receiver = get_node_text(obj_node).lstrip("$")
                return method_name, receiver, None
        return None, None, None

    return None, None, None


def _extract_binding_php(
    node: Node,
) -> BindingInfo | list[BindingInfo] | None:
    """Extract PHP-specific bindings."""
    if node.type == "assignment_expression":
        return _extract_assignment_binding_php(node)
    if node.type == "foreach_statement":
        return _extract_foreach_binding_php(node)
    if node.type == "catch_clause":
        return _extract_catch_binding_php(node)
    return None


def _extract_assignment_binding_php(node: Node) -> BindingInfo | None:
    """Extract binding from PHP assignment_expression.

    Handles: $var = func(), $var = $obj->method(), $var = new Class()
    """
    left = get_child_by_field(node, "left")
    right = get_child_by_field(node, "right")

    if not left or not right:
        return None

    # Left should be a variable_name ($var)
    if left.type != "variable_name":
        return None

    var_name = get_node_text(left).lstrip("$")
    line = node.start_point[0] + 1

    # new ClassName()
    if right.type == "object_creation_expression":
        call_name, _, type_name = _extract_call_info_php(right)
        if type_name:
            return BindingInfo(
                variable=var_name, source=SOURCE_CONSTRUCTOR,
                call_name=call_name, type_name=type_name, line=line,
            )
        return None

    # function_call_expression: $var = func()
    if right.type == "function_call_expression":
        call_name, call_receiver, _ = _extract_call_info_php(right)
        if not call_name:
            return None
        return BindingInfo(
            variable=var_name, source=SOURCE_ASSIGNMENT_CALL,
            call_name=call_name, line=line,
        )

    # member_call_expression: $var = $obj->method()
    if right.type == "member_call_expression":
        call_name, call_receiver, _ = _extract_call_info_php(right)
        if not call_name:
            return None
        return BindingInfo(
            variable=var_name, source=SOURCE_ASSIGNMENT_METHOD_CALL,
            call_name=call_name, call_receiver=call_receiver, line=line,
        )

    return None


def _extract_foreach_binding_php(node: Node) -> BindingInfo | None:
    """Extract binding from PHP foreach statement.

    foreach ($items as $item) or foreach ($items as $key => $value)
    """
    # Find the value variable — it's typically the last variable_name
    # before the body (compound_statement)
    # The foreach_statement structure varies; look for direct children
    variables: list[Node] = []
    collection_node = None

    for child in node.children:
        if child.type == "variable_name":
            variables.append(child)
        elif child.type in (
            "function_call_expression", "member_call_expression",
            "subscript_expression",
        ):
            collection_node = child

    if not variables:
        return None

    # Last variable is the iteration value
    value_var = variables[-1]
    var_name = get_node_text(value_var).lstrip("$")
    line = node.start_point[0] + 1

    if collection_node and collection_node.type in (
        "function_call_expression", "member_call_expression",
    ):
        call_name, call_receiver, type_name = _extract_call_info_php(collection_node)
        return BindingInfo(
            variable=var_name, source=SOURCE_FOR_IN,
            call_name=call_name, call_receiver=call_receiver,
            type_name=type_name, line=line,
        )

    return BindingInfo(
        variable=var_name, source=SOURCE_FOR_IN, line=line,
    )


def _extract_catch_binding_php(node: Node) -> BindingInfo | None:
    """Extract binding from PHP catch clause. Has type info.

    catch (ExceptionType $e) {}
    """
    # Find the type and variable in the catch clause
    type_name = None
    var_name = None

    for child in node.children:
        if child.type == "type_list":
            # Get first type from type list
            for type_child in child.children:
                if type_child.type in ("name", "qualified_name", "named_type"):
                    full_name = get_node_text(type_child)
                    type_name = full_name.split("\\")[-1]
                    break
        elif child.type == "variable_name":
            var_name = get_node_text(child).lstrip("$")

    if not var_name:
        return None

    return BindingInfo(
        variable=var_name, source=SOURCE_EXCEPT_AS,
        type_name=type_name, line=node.start_point[0] + 1,
    )


# =============================================================================
# RUST BINDINGS
# =============================================================================


def _extract_call_info_rust(
    call_node: Node,
) -> tuple[str | None, str | None, str | None]:
    """Extract call info from Rust call nodes.

    Handles:
    - call_expression with identifier: func()
    - call_expression with scoped_identifier: Type::new()
    - call_expression with field_expression: obj.method()
    """
    if call_node.type != "call_expression":
        return None, None, None

    func_node = get_child_by_field(call_node, "function")
    if not func_node:
        return None, None, None

    if func_node.type == "identifier":
        name = get_node_text(func_node)
        return name, None, None

    if func_node.type == "scoped_identifier":
        # Type::new(), Type::from(), etc.
        full_text = get_node_text(func_node)
        parts = full_text.split("::")
        if len(parts) >= 2:
            type_part = parts[-2]
            method_part = parts[-1]
            if method_part == "new" and type_part[0:1].isupper():
                return method_part, None, type_part
            # Type::from(), Type::default() — treat as constructor too
            if method_part in ("from", "default", "with_capacity") and type_part[0:1].isupper():
                return method_part, None, type_part
            # Otherwise it's a static method call
            return method_part, type_part, None
        return None, None, None

    if func_node.type == "field_expression":
        # obj.method()
        field_node = get_child_by_field(func_node, "field")
        value_node = get_child_by_field(func_node, "value")
        if field_node and value_node:
            method_name = get_node_text(field_node)
            if value_node.type == "identifier":
                receiver = get_node_text(value_node)
                return method_name, receiver, None

    return None, None, None


def _extract_binding_rust(
    node: Node,
) -> BindingInfo | None:
    """Extract Rust-specific bindings."""
    if node.type == "let_declaration":
        return _extract_let_binding_rust(node)
    if node.type == "for_expression":
        return _extract_for_binding_rust(node)
    return None


def _extract_let_binding_rust(node: Node) -> BindingInfo | None:
    """Extract binding from Rust let declaration.

    Handles: let x = func(), let mut x = obj.method(), let x = Type::new()
    """
    pattern_node = get_child_by_field(node, "pattern")
    value_node = get_child_by_field(node, "value")

    if not pattern_node or not value_node:
        return None

    # Pattern should be a simple identifier (skip destructuring)
    if pattern_node.type not in ("identifier", "mut_pattern"):
        return None

    if pattern_node.type == "mut_pattern":
        # let mut x = ... — find the identifier inside
        for child in pattern_node.children:
            if child.type == "identifier":
                pattern_node = child
                break
        else:
            return None

    var_name = get_node_text(pattern_node)
    line = node.start_point[0] + 1

    if value_node.type == "call_expression":
        call_name, call_receiver, type_name = _extract_call_info_rust(value_node)
        if not call_name:
            return None

        if type_name:
            return BindingInfo(
                variable=var_name, source=SOURCE_CONSTRUCTOR,
                call_name=call_name, type_name=type_name, line=line,
            )

        if call_receiver:
            return BindingInfo(
                variable=var_name, source=SOURCE_ASSIGNMENT_METHOD_CALL,
                call_name=call_name, call_receiver=call_receiver, line=line,
            )

        return BindingInfo(
            variable=var_name, source=SOURCE_ASSIGNMENT_CALL,
            call_name=call_name, line=line,
        )

    return None


def _extract_for_binding_rust(node: Node) -> BindingInfo | None:
    """Extract binding from Rust for expression.

    for item in items.iter() { }
    """
    pattern_node = get_child_by_field(node, "pattern")
    value_node = get_child_by_field(node, "value")

    if not pattern_node or not value_node:
        return None

    if pattern_node.type != "identifier":
        return None

    var_name = get_node_text(pattern_node)
    line = node.start_point[0] + 1

    if value_node.type == "call_expression":
        call_name, call_receiver, type_name = _extract_call_info_rust(value_node)
        return BindingInfo(
            variable=var_name, source=SOURCE_FOR_IN,
            call_name=call_name, call_receiver=call_receiver,
            type_name=type_name, line=line,
        )

    if value_node.type == "identifier":
        collection_name = get_node_text(value_node)
        return BindingInfo(
            variable=var_name, source=SOURCE_FOR_IN,
            call_receiver=collection_name, line=line,
        )

    return None
