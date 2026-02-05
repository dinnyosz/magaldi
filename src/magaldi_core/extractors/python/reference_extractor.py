"""Python reference extraction.

This module handles extraction of cross-file references from Python source code:
- Class instantiation (MyClass())
- Function calls (my_function())
- Method calls (obj.method())
- Type hints (x: MyClass, -> MyClass)
"""

from __future__ import annotations

from tree_sitter import Tree

from magaldi_core.extractors.base import (
    get_child_by_field,
    get_node_text,
    walk_tree,
)
from magaldi_core.extractors.python.utils import (
    find_containing_element,
    get_call_context,
    is_likely_class_name,
)
from magaldi_core.extractors.types import ExtractedReference


def extract_python_references(tree: Tree, lines: list[str]) -> list[ExtractedReference]:
    """Extract all references (calls, type hints) from Python AST.

    This captures cross-file usage patterns:
    - Class instantiation: MyClass()
    - Function calls: my_function()
    - Method calls: obj.method()
    - Type hints: x: MyClass, -> MyClass

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
            if parent.type in ("import_statement", "import_from_statement"):
                in_import = True
                break
            parent = parent.parent
        if in_import:
            continue

        if node.type == "call":
            func_node = get_child_by_field(node, "function")
            if not func_node:
                continue

            line = node.start_point[0] + 1
            containing = find_containing_element(node)
            context = get_call_context(node, lines)

            if func_node.type == "identifier":
                # Direct call: func() or MyClass()
                name = get_node_text(func_node)
                ref_type = "instantiation" if is_likely_class_name(name) else "function_call"

                key = (name, line, ref_type)
                if key not in seen:
                    seen.add(key)
                    refs.append(ExtractedReference(
                        ref_type=ref_type,
                        target_name=name,
                        line=line,
                        containing_element=containing,
                        context_snippet=context,
                    ))

            elif func_node.type == "attribute":
                # Method call: obj.method()
                attr_node = get_child_by_field(func_node, "attribute")
                if attr_node:
                    method_name = get_node_text(attr_node)
                    # Get the object being called on
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
                            context_snippet=f"called on {obj_name}" if obj_name else context,
                        ))

        elif node.type == "type":
            # Type hints in function parameters or return types
            # The type node contains the actual type name
            type_text = get_node_text(node)
            line = node.start_point[0] + 1
            containing = find_containing_element(node)

            # Extract simple type names (handle List[X], Optional[X], etc.)
            # For now, just get identifiers that look like class names
            for type_node in walk_tree(node):
                if type_node.type == "identifier":
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
