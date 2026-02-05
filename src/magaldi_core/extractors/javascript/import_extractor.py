"""JavaScript/TypeScript import extraction.

This module handles extraction of import statements from JavaScript and TypeScript:
- ES6 import statements (named, default, namespace imports)
- CommonJS require() statements
"""

from __future__ import annotations

from tree_sitter import Node, Tree

from magaldi_core.extractors.base import (
    get_child_by_field,
    get_node_text,
    walk_tree,
)
from magaldi_core.extractors.types import ExtractedImport


def extract_javascript_imports(
    tree: Tree, lines: list[str]  # noqa: ARG001
) -> list[ExtractedImport]:
    """Extract import statements from a JavaScript/TypeScript AST.

    Handles:
    - import { foo } from './utils' -> Import(name="foo", module="./utils", alias=None)
    - import { foo as bar } from './utils' -> Import(name="foo", module="./utils", alias="bar")
    - import utils from './utils' -> Import(name="utils", module="./utils", alias=None)
    - import * as utils from './utils' -> Import(name="*", module="./utils", alias="utils")
    - const bar = require('lib') -> Import(name="bar", module="lib", alias=None)

    Args:
        tree: Parsed tree-sitter Tree.
        lines: Source code lines (unused but kept for API consistency).

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
