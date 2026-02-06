"""Bash script extractor - extracts functions, constants, calls, and imports."""

from __future__ import annotations

from tree_sitter import Node, Tree

from magaldi_core.extractors.base import get_children_by_type, get_node_text, walk_tree
from magaldi_core.extractors.types import ExtractedCall, ExtractedElement, ExtractedImport

# Shell builtins that should not be recorded as calls
_BASH_BUILTINS = frozenset({
    "echo", "printf", "cd", "local", "return", "exit", "shift", "set",
    "unset", "export", "readonly", "declare", "typeset", "read", "eval",
    "exec", "trap", "wait", "true", "false", "test", "[", "[[", "let",
    "pushd", "popd", "dirs", "break", "continue", "getopts", "hash",
    "umask", "ulimit", "shopt", "enable", "builtin", "command", "type",
})


class BashExtractor:
    """Extractor for Bash shell scripts."""

    @property
    def language(self) -> str:
        return "bash"

    def extract_elements(self, tree: Tree, lines: list[str]) -> list[ExtractedElement]:
        """Extract functions and top-level constants from a Bash AST."""
        elements: list[ExtractedElement] = []
        root = tree.root_node

        for child in root.children:
            if child.type == "function_definition":
                elem = _extract_function(child, lines)
                if elem:
                    elements.append(elem)
            elif child.type == "variable_assignment":
                elem = _extract_variable(child, lines)
                if elem:
                    elements.append(elem)
            elif child.type == "declaration_command":
                elem = _extract_declaration(child, lines)
                if elem:
                    elements.append(elem)

        return elements

    def extract_imports(self, tree: Tree, lines: list[str]) -> list[ExtractedImport]:
        """Extract source/. imports from a Bash AST."""
        return extract_bash_imports(tree, lines)

    def extract_calls(self, function_node: Node) -> list[ExtractedCall]:
        """Extract calls from within a function body."""
        return extract_bash_calls(function_node)


def _extract_function(node: Node, lines: list[str]) -> ExtractedElement | None:
    """Extract a function definition."""
    # Find the function name - first 'word' child
    name = None
    for child in node.children:
        if child.type == "word":
            name = get_node_text(child)
            break

    if not name:
        return None

    line_start = node.start_point[0] + 1
    line_end = node.end_point[0] + 1
    raw_code = "\n".join(lines[line_start - 1:line_end])

    return ExtractedElement(
        element_type="function",
        name=name,
        line_start=line_start,
        line_end=line_end,
        raw_code=raw_code,
        node=node,
    )


def _extract_variable(node: Node, lines: list[str]) -> ExtractedElement | None:
    """Extract a top-level variable assignment (e.g. MY_VAR="value")."""
    var_name_node = None
    for child in node.children:
        if child.type == "variable_name":
            var_name_node = child
            break

    if not var_name_node:
        return None

    name = get_node_text(var_name_node)
    line_start = node.start_point[0] + 1
    line_end = node.end_point[0] + 1
    raw_code = "\n".join(lines[line_start - 1:line_end])

    return ExtractedElement(
        element_type="constant",
        name=name,
        line_start=line_start,
        line_end=line_end,
        raw_code=raw_code,
        node=node,
    )


def _extract_declaration(node: Node, lines: list[str]) -> ExtractedElement | None:
    """Extract a declaration_command (readonly, export, declare).

    Determines element_type based on the keyword:
    - readonly/export -> constant
    - declare -r -> constant
    - declare (other) -> variable
    """
    keyword = None
    has_readonly_flag = False

    for child in node.children:
        if child.type == "word":
            text = get_node_text(child)
            if text in ("readonly", "export", "declare", "typeset"):
                keyword = text
            elif text.startswith("-") and "r" in text:
                has_readonly_flag = True

    if keyword == "local":
        return None

    # Find the variable assignment child
    var_name = None
    for child in node.children:
        if child.type == "variable_assignment":
            for sub in child.children:
                if sub.type == "variable_name":
                    var_name = get_node_text(sub)
                    break
            break
        # Sometimes the variable_name is a direct child (export without =)
        if child.type == "variable_name":
            var_name = get_node_text(child)
            break

    if not var_name:
        return None

    # Determine type
    if keyword in ("readonly", "export") or has_readonly_flag:
        element_type = "constant"
    else:
        element_type = "variable"

    line_start = node.start_point[0] + 1
    line_end = node.end_point[0] + 1
    raw_code = "\n".join(lines[line_start - 1:line_end])

    return ExtractedElement(
        element_type=element_type,
        name=var_name,
        line_start=line_start,
        line_end=line_end,
        raw_code=raw_code,
        node=node,
    )


def extract_bash_imports(tree: Tree, lines: list[str]) -> list[ExtractedImport]:
    """Extract source/. import commands from a Bash AST."""
    imports: list[ExtractedImport] = []

    for node in walk_tree(tree.root_node):
        if node.type != "command":
            continue

        # Get command name
        cmd_name_node = None
        for child in node.children:
            if child.type == "command_name":
                cmd_name_node = child
                break

        if not cmd_name_node:
            continue

        word_nodes = get_children_by_type(cmd_name_node, "word")
        if not word_nodes:
            continue

        cmd_name = get_node_text(word_nodes[0])
        if cmd_name not in ("source", "."):
            continue

        # Get the sourced file path (next argument after command_name)
        source_path = None
        found_cmd = False
        for child in node.children:
            if child.type == "command_name":
                found_cmd = True
                continue
            if found_cmd and child.type in ("word", "string", "raw_string", "concatenation"):
                source_path = get_node_text(child).strip("\"'")
                break

        if source_path:
            imports.append(ExtractedImport(
                name=source_path.rsplit("/", 1)[-1],
                module=source_path,
                alias=None,
                line=node.start_point[0] + 1,
            ))

    return imports


def extract_bash_calls(function_node: Node) -> list[ExtractedCall]:
    """Extract command calls from within a function body.

    Excludes shell builtins to reduce noise.
    """
    calls: list[ExtractedCall] = []
    seen: set[tuple[str, int]] = set()

    # Find the compound_statement (function body)
    body = None
    for child in function_node.children:
        if child.type == "compound_statement":
            body = child
            break

    if not body:
        return calls

    for node in walk_tree(body):
        if node.type != "command":
            continue

        # Skip nested function definitions
        if _is_inside_nested_function(node, body):
            continue

        cmd_name_node = None
        for child in node.children:
            if child.type == "command_name":
                cmd_name_node = child
                break

        if not cmd_name_node:
            continue

        word_nodes = get_children_by_type(cmd_name_node, "word")
        if not word_nodes:
            continue

        cmd_name = get_node_text(word_nodes[0])

        # Skip builtins and source/. (imports)
        if cmd_name in _BASH_BUILTINS or cmd_name in ("source", "."):
            continue

        line = node.start_point[0] + 1
        key = (cmd_name, line)
        if key in seen:
            continue
        seen.add(key)

        calls.append(ExtractedCall(
            name=cmd_name,
            receiver=None,
            line=line,
        ))

    return calls


def _is_inside_nested_function(node: Node, body: Node) -> bool:
    """Check if a node is inside a nested function definition."""
    parent = node.parent
    while parent and parent != body:
        if parent.type == "function_definition":
            return True
        parent = parent.parent
    return False
