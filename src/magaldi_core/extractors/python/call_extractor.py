"""Python call extraction.

This module handles extraction of function/method calls from within Python functions:
- Direct function calls (process(x))
- Method calls on self (self.validate())
- Method calls on objects (utils.run())
- Chained calls (obj.method1().method2())
"""

from __future__ import annotations

from tree_sitter import Node, Tree

from magaldi_core.extractors.base import (
    get_child_by_field,
    get_node_text,
    walk_top_level,
    walk_tree,
)
from magaldi_core.extractors.types import ExtractedCall

# Node types that define function/class scopes in Python
_PYTHON_SCOPE_TYPES = frozenset({
    "function_definition",
    "class_definition",
    "decorated_definition",
})


def extract_python_calls(function_node: Node) -> list[ExtractedCall]:
    """Extract all function/method calls from within a Python function/method body.

    Handles:
    - process(x) -> ExtractedCall(name="process", receiver=None, line=45)
    - self.validate() -> ExtractedCall(name="validate", receiver="self", line=48)
    - utils.run() -> ExtractedCall(name="run", receiver="utils", line=52)
    - obj.method().chain() -> Extracts each call in the chain

    Args:
        function_node: A function_definition node from tree-sitter.

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
        if node.type == "call":
            extracted = _extract_python_call(node)
            if extracted:
                for call in extracted:
                    key = (call.name, call.receiver, call.line)
                    if key not in seen:
                        seen.add(key)
                        calls.append(call)

    return calls


def _extract_python_call(node: Node) -> list[ExtractedCall]:
    """Extract call information from a Python call node.

    Handles method chaining by extracting each call in the chain.

    Args:
        node: A 'call' node from tree-sitter.

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

    elif func_node.type == "attribute":
        # Method call: obj.method() or self.method() or utils.run()
        attr_node = get_child_by_field(func_node, "attribute")
        obj_node = get_child_by_field(func_node, "object")

        if attr_node:
            method_name = get_node_text(attr_node)
            receiver = None

            if obj_node:
                if obj_node.type == "identifier":
                    # Simple receiver: self.method(), utils.run()
                    receiver = get_node_text(obj_node)
                elif obj_node.type == "call":
                    # Chained call: obj.method1().method2()
                    # Get the receiver text for context (first identifier in chain)
                    receiver = _get_chain_root(obj_node)
                elif obj_node.type == "attribute":
                    # Nested attribute: a.b.method()
                    receiver = get_node_text(obj_node)

            calls.append(ExtractedCall(name=method_name, receiver=receiver, line=line))

    return calls


def extract_top_level_python_calls(tree: Tree) -> list[ExtractedCall]:
    """Extract function/method calls from the top level of a Python module.

    Walks file-scope statements (skipping function/class definition subtrees)
    and collects calls using the same extraction logic as function-body calls.

    Args:
        tree: A parsed tree-sitter Tree for a Python file.

    Returns:
        List of extracted calls found at module scope.
    """
    calls: list[ExtractedCall] = []
    seen: set[tuple[str, str | None, int]] = set()

    for node in walk_top_level(tree.root_node, _PYTHON_SCOPE_TYPES):
        if node.type == "call":
            extracted = _extract_python_call(node)
            if extracted:
                for call in extracted:
                    key = (call.name, call.receiver, call.line)
                    if key not in seen:
                        seen.add(key)
                        calls.append(call)

    return calls


def _get_chain_root(node: Node) -> str | None:
    """Get the root identifier from a call chain.

    For obj.method1().method2(), returns "obj".
    """
    current = node
    while current:
        if current.type == "identifier":
            return get_node_text(current)  # type: ignore[no-any-return]
        elif current.type == "call":
            func = get_child_by_field(current, "function")
            current = func
        elif current.type == "attribute":
            obj = get_child_by_field(current, "object")
            current = obj
        else:
            break
    return None
