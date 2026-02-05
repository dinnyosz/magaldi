"""Python import extraction.

This module handles extraction of import statements from Python source code:
- import statements (import os, import pandas as pd)
- from imports (from utils import process, from utils import process as p)
- relative imports (from . import foo, from ..utils import bar)
- wildcard imports (from utils import *)
"""

from __future__ import annotations

from tree_sitter import Node, Tree

from magaldi_core.extractors.base import (
    get_child_by_field,
    get_node_text,
)
from magaldi_core.extractors.types import ExtractedImport


def extract_python_imports(
    tree: Tree, lines: list[str]  # noqa: ARG001
) -> list[ExtractedImport]:
    """Extract import statements from a Python AST.

    Handles:
    - import os -> Import(name="os", module="os", alias=None)
    - import pandas as pd -> Import(name="pandas", module="pandas", alias="pd")
    - from utils import process -> Import(name="process", module="utils", alias=None)
    - from utils import process as p -> Import(name="process", module="utils", alias="p")

    Args:
        tree: Parsed tree-sitter Tree.
        lines: Source code lines (unused but kept for API consistency).

    Returns:
        List of extracted imports.
    """
    imports: list[ExtractedImport] = []
    root = tree.root_node

    for node in root.children:
        if node.type == "import_statement":
            # Handle: import os, import pandas as pd
            imports.extend(_extract_python_import_statement(node))
        elif node.type == "import_from_statement":
            # Handle: from utils import process, from utils import process as p
            imports.extend(_extract_python_import_from_statement(node))

    return imports


def _extract_python_import_statement(node: Node) -> list[ExtractedImport]:
    """Extract imports from 'import x' or 'import x as y' statements."""
    imports: list[ExtractedImport] = []
    line = node.start_point[0] + 1

    for child in node.children:
        if child.type == "dotted_name":
            # Simple import: import os
            name = get_node_text(child)
            imports.append(ExtractedImport(
                name=name,
                module=name,
                alias=None,
                line=line,
            ))
        elif child.type == "aliased_import":
            # Import with alias: import pandas as pd
            name_node = get_child_by_field(child, "name")
            alias_node = get_child_by_field(child, "alias")
            if name_node:
                name = get_node_text(name_node)
                alias = get_node_text(alias_node) if alias_node else None
                imports.append(ExtractedImport(
                    name=name,
                    module=name,
                    alias=alias,
                    line=line,
                ))

    return imports


def _extract_python_import_from_statement(node: Node) -> list[ExtractedImport]:
    """Extract imports from 'from x import y' or 'from x import y as z' statements."""
    imports: list[ExtractedImport] = []
    line = node.start_point[0] + 1

    # Get the module name
    module_node = get_child_by_field(node, "module_name")
    module = get_node_text(module_node) if module_node else ""

    # Handle relative imports (dots before module name)
    # e.g., 'from . import foo' or 'from ..utils import bar'
    relative_prefix = ""
    for child in node.children:
        if child.type == "relative_import":
            # Get all dots and the module name from relative import
            for rel_child in child.children:
                if rel_child.type == "import_prefix":
                    relative_prefix = get_node_text(rel_child)
                elif rel_child.type == "dotted_name":
                    module = relative_prefix + get_node_text(rel_child)
            break

    # If we only have dots (from . import foo), module is empty
    if not module and relative_prefix:
        module = relative_prefix

    # Get imported names
    for child in node.children:
        if child.type == "dotted_name" and child != module_node:
            # Simple import: from utils import process
            name = get_node_text(child)
            imports.append(ExtractedImport(
                name=name,
                module=module,
                alias=None,
                line=line,
            ))
        elif child.type == "aliased_import":
            # Import with alias: from utils import process as p
            name_node = get_child_by_field(child, "name")
            alias_node = get_child_by_field(child, "alias")
            if name_node:
                name = get_node_text(name_node)
                alias = get_node_text(alias_node) if alias_node else None
                imports.append(ExtractedImport(
                    name=name,
                    module=module,
                    alias=alias,
                    line=line,
                ))
        elif child.type == "wildcard_import":
            # from utils import *
            imports.append(ExtractedImport(
                name="*",
                module=module,
                alias=None,
                line=line,
            ))

    return imports
