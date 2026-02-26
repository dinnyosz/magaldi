"""JavaScript/TypeScript call extraction.

This module handles extraction of function/method calls from within function bodies:
- Direct function calls: process(x)
- Method calls: this.validate(), utils.run()
- Chained method calls: obj.method1().method2()
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

# Node types that define function/class scopes in JavaScript/TypeScript
_JS_SCOPE_TYPES = frozenset({
    "function_declaration",
    "class_declaration",
    "method_definition",
    "arrow_function",
    "function",
})


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


def extract_top_level_javascript_calls(tree: Tree) -> list[ExtractedCall]:
    """Extract function/method calls from the top level of a JS/TS module.

    Walks file-scope statements (skipping function/class definition subtrees)
    and collects calls using the same extraction logic as function-body calls.

    Args:
        tree: A parsed tree-sitter Tree for a JavaScript/TypeScript file.

    Returns:
        List of extracted calls found at module scope.
    """
    calls: list[ExtractedCall] = []
    seen: set[tuple[str, str | None, int]] = set()

    for node in walk_top_level(tree.root_node, _JS_SCOPE_TYPES):
        if node.type == "call_expression":
            extracted = _extract_js_call(node)
            if extracted:
                for call in extracted:
                    key = (call.name, call.receiver, call.line)
                    if key not in seen:
                        seen.add(key)
                        calls.append(call)

    return calls


def _get_js_chain_root(node: Node) -> str | None:
    """Get the root identifier from a JavaScript call chain.

    For obj.method1().method2(), returns "obj".
    """
    current = node
    while current:
        if current.type == "identifier":
            return get_node_text(current)  # type: ignore[no-any-return]
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
