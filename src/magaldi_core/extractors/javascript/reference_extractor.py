"""JavaScript/TypeScript reference extraction.

This module handles extraction of cross-file references from JavaScript and TypeScript:
- Class instantiations (new MyClass())
- Function calls (myFunction())
- Method calls (obj.method())
- Type annotations in TypeScript (x: MyClass)
"""

from __future__ import annotations

from tree_sitter import Node, Tree

from magaldi_core.extractors.base import (
    get_child_by_field,
    get_node_text,
    walk_tree,
)
from magaldi_core.extractors.types import ExtractedReference
from magaldi_core.extractors.javascript.utils import is_likely_class_name


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
                    if is_likely_class_name(name):
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
