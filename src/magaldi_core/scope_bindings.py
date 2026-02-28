"""AST-based variable binding extraction for scope-aware type resolution.

Strategy 5.7: Extracts variable->type bindings from tree-sitter AST,
going beyond the regex-based approaches in Strategies 5.5 and 5.6.

Supported Python binding patterns:
- var = func()              (bare call assignment)
- var = receiver.method()   (method call assignment) [beyond regex]
- var = ClassName()         (constructor assignment)
- with expr() as var:       (context manager) [beyond regex]
- for var in expr:          (loop variable) [beyond regex]
- except ExcType as var:    (exception handler) [beyond regex]
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

# Node types that define function/class scopes — bindings inside these
# are NOT part of the enclosing function's scope.
_SCOPE_TYPES = frozenset({"function_definition", "class_definition"})


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
        language: Programming language (currently only "python" supported).

    Returns:
        List of BindingInfo objects, ordered by line number.
        Later bindings for the same variable override earlier ones.
    """
    if language != "python":
        return []

    if not raw_code or not raw_code.strip():
        return []

    # Lazy import to avoid circular imports and heavy init at module load
    from magaldi_core.tree_sitter_manager import get_manager

    try:
        manager = get_manager()
        tree = manager.parse(raw_code.encode("utf-8"), language)
    except Exception:
        logger.debug("Failed to parse raw_code for scope bindings", exc_info=True)
        return []

    body_node = _find_function_body(tree.root_node)
    if not body_node:
        return []

    bindings: list[BindingInfo] = []

    for node in walk_tree(body_node):
        # Skip nodes inside nested function/class definitions
        if _is_inside_nested_scope(node, body_node):
            continue

        binding = None
        if node.type == "assignment":
            binding = _extract_assignment_binding(node)
        elif node.type == "with_statement":
            bindings.extend(_extract_with_bindings(node))
            continue  # with_statement handled as a group
        elif node.type == "for_statement":
            binding = _extract_for_binding(node)
        elif node.type == "except_clause":
            binding = _extract_except_binding(node)

        if binding:
            bindings.append(binding)

    return bindings


def _find_function_body(root: Node) -> Node | None:
    """Navigate parsed raw_code to the function body block.

    raw_code contains a complete function definition, so the parse tree is:
    module -> function_definition -> body (block)
    or:
    module -> decorated_definition -> function_definition -> body (block)

    Args:
        root: Root node of the parsed tree (module node).

    Returns:
        The body (block) node of the function, or None if not found.
    """
    for child in root.children:
        if child.type == "function_definition":
            return get_child_by_field(child, "body")
        if child.type == "decorated_definition":
            for sub in child.children:
                if sub.type == "function_definition":
                    return get_child_by_field(sub, "body")
    return None


def _is_inside_nested_scope(node: Node, body_node: Node) -> bool:
    """Check if a node is inside a nested function/class definition.

    We only want bindings from the immediate function scope, not
    from nested definitions which have their own scope.
    """
    parent = node.parent
    while parent and parent != body_node:
        if parent.type in _SCOPE_TYPES:
            return True
        parent = parent.parent
    return False


def _extract_call_info(
    call_node: Node,
) -> tuple[str | None, str | None, str | None]:
    """Extract call name, receiver, and constructor type from a call node.

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
    if node.type == "await":
        # await node has children: 'await' keyword + the expression
        for child in node.children:
            if child.type != "await":  # skip the keyword itself
                return child
    return node


def _extract_assignment_binding(node: Node) -> BindingInfo | None:
    """Extract binding from an assignment node.

    Handles: var = func(), var = obj.method(), var = Class(),
    var = await func(), var = await obj.method()

    Skips: var = other_var, var = literal, a, b = func() (tuple unpack)
    """
    left = get_child_by_field(node, "left")
    if not left or left.type != "identifier":
        return None

    right = get_child_by_field(node, "right")
    if not right:
        return None

    var_name = get_node_text(left)
    line = node.start_point[0] + 1

    # Unwrap await if present
    right = _unwrap_await(right)

    # Must be a call expression
    if right.type != "call":
        return None

    call_name, call_receiver, type_name = _extract_call_info(right)
    if not call_name:
        return None

    if type_name:
        return BindingInfo(
            variable=var_name,
            source=SOURCE_CONSTRUCTOR,
            call_name=call_name,
            type_name=type_name,
            line=line,
        )

    if call_receiver:
        return BindingInfo(
            variable=var_name,
            source=SOURCE_ASSIGNMENT_METHOD_CALL,
            call_name=call_name,
            call_receiver=call_receiver,
            line=line,
        )

    return BindingInfo(
        variable=var_name,
        source=SOURCE_ASSIGNMENT_CALL,
        call_name=call_name,
        line=line,
    )


def _extract_with_bindings(node: Node) -> list[BindingInfo]:
    """Extract bindings from a with_statement.

    Handles: with expr() as var:, with a() as x, b() as y:

    AST structure:
    with_statement -> with_clause -> with_item -> as_pattern
        as_pattern has: value (the call) and alias (as_pattern_target -> identifier)
    """
    bindings: list[BindingInfo] = []

    for child in walk_tree(node):
        if child.type == "as_pattern" and child.parent and child.parent.type == "with_item":
            # The as_pattern contains: expression, 'as' keyword, as_pattern_target
            value_node = None
            alias_node = None

            for sub in child.children:
                if sub.type == "call":
                    value_node = sub
                elif sub.type == "as_pattern_target":
                    # as_pattern_target contains the identifier
                    for inner in sub.children:
                        if inner.type == "identifier":
                            alias_node = inner
                            break

            if alias_node:
                var_name = get_node_text(alias_node)
                line = child.start_point[0] + 1

                if value_node:
                    call_name, call_receiver, type_name = _extract_call_info(
                        value_node
                    )
                    bindings.append(
                        BindingInfo(
                            variable=var_name,
                            source=SOURCE_WITH_AS,
                            call_name=call_name,
                            call_receiver=call_receiver,
                            type_name=type_name,
                            line=line,
                        )
                    )
                else:
                    # with expression is not a call (e.g., with lock:)
                    bindings.append(
                        BindingInfo(
                            variable=var_name,
                            source=SOURCE_WITH_AS,
                            line=line,
                        )
                    )

    return bindings


def _extract_for_binding(node: Node) -> BindingInfo | None:
    """Extract binding from a for_statement.

    Handles: for var in expr(): and for var in obj.method():

    AST structure:
    for_statement -> left (identifier), right (call or identifier)
    """
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
            variable=var_name,
            source=SOURCE_FOR_IN,
            call_name=call_name,
            call_receiver=call_receiver,
            type_name=type_name,
            line=line,
        )

    if right.type == "identifier":
        # for item in items: — record the collection name
        collection_name = get_node_text(right)
        return BindingInfo(
            variable=var_name,
            source=SOURCE_FOR_IN,
            call_receiver=collection_name,
            line=line,
        )

    return None


def _extract_except_binding(node: Node) -> BindingInfo | None:
    """Extract binding from an except_clause.

    Handles: except ExcType as var:

    AST structure (Python tree-sitter):
    except_clause -> as_pattern (field=value)
        as_pattern -> identifier (exception type), 'as', as_pattern_target -> identifier (alias)
    """
    # The except_clause uses an as_pattern with field name 'value' when there's 'as'
    as_pattern = get_child_by_field(node, "value")
    if not as_pattern or as_pattern.type != "as_pattern":
        return None

    # Find the exception type and the alias
    exc_type = None
    alias = None

    for child in as_pattern.children:
        if child.type == "identifier" and exc_type is None:
            exc_type = get_node_text(child)
        elif child.type == "attribute" and exc_type is None:
            # e.g., except module.ExcType as e
            exc_type = get_node_text(child)
        elif child.type == "as_pattern_target":
            for inner in child.children:
                if inner.type == "identifier":
                    alias = get_node_text(inner)
                    break

    if not alias or not exc_type:
        return None

    return BindingInfo(
        variable=alias,
        source=SOURCE_EXCEPT_AS,
        type_name=exc_type,
        line=node.start_point[0] + 1,
    )
