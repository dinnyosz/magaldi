"""Bash script extractor - extracts functions, constants, calls, and imports."""

from __future__ import annotations

import re

from tree_sitter import Node, Tree

from magaldi_core.extractors.base import (
    get_children_by_type,
    get_node_text,
    walk_tree,
)
from magaldi_core.extractors.types import ExtractedCall, ExtractedElement, ExtractedImport

# Shell builtins that should not be recorded as calls
_BASH_BUILTINS = frozenset({
    "echo", "printf", "cd", "local", "return", "exit", "shift", "set",
    "unset", "export", "readonly", "declare", "typeset", "read", "eval",
    "exec", "trap", "wait", "true", "false", "test", "[", "[[", "let",
    "pushd", "popd", "dirs", "break", "continue", "getopts", "hash",
    "umask", "ulimit", "shopt", "enable", "builtin", "command", "type",
})

# Node types that wrap script content without adding semantic meaning.
# Scripts using download guards (e.g., `{ ... }`) produce these at root level.
_TRANSPARENT_CONTAINERS = frozenset({"compound_statement", "subshell"})


def _iter_top_level_nodes(root: Node):
    """Iterate top-level nodes, descending into transparent containers.

    Bash scripts sometimes wrap all content in { ... } (download guard) or
    ( ... ) (subshell). This yields the actual function/variable nodes inside
    those wrappers instead of stopping at the container boundary.
    """
    for child in root.children:
        if child.type in _TRANSPARENT_CONTAINERS:
            yield from _iter_top_level_nodes(child)
        else:
            yield child


def _walk_top_level_bash(root: Node, skip_types: frozenset[str]):
    """Walk file-scope nodes, unwrapping transparent containers first.

    Combines transparent container unwrapping (``_iter_top_level_nodes``) with
    scope-skipping (skip ``function_definition`` subtrees).  This ensures that
    brace-group wrappers ``{ ... }`` and subshell wrappers ``( ... )`` are
    treated as transparent, while function bodies are properly skipped.

    Args:
        root: The root node (``tree.root_node``).
        skip_types: Node types whose subtrees should be skipped entirely.

    Yields:
        All descendant nodes at file scope (inside transparent containers but
        outside function bodies).
    """
    for child in _iter_top_level_nodes(root):
        if child.type in skip_types:
            continue
        yield from _walk_skip_bash(child, skip_types)


def _walk_skip_bash(node: Node, skip_types: frozenset[str]):
    """Walk a subtree, skipping nodes whose type is in *skip_types*."""
    yield node
    for child in node.children:
        if child.type in skip_types:
            continue
        yield from _walk_skip_bash(child, skip_types)


class BashExtractor:
    """Extractor for Bash shell scripts."""

    @property
    def language(self) -> str:
        return "bash"

    def extract_elements(self, tree: Tree, lines: list[str]) -> list[ExtractedElement]:
        """Extract functions, variables, and constants from a Bash AST.

        Extracts at all depths — including inside function bodies.
        Recurses into compound_statement and subshell nodes to handle
        scripts wrapped in { ... } or ( ... ) guards.

        When tree-sitter fails to parse certain Bash syntax (e.g., ANSI-C
        quoting in parameter expansions, glob patterns in variable
        substitutions), function_definition nodes may be destroyed.  A
        regex-based recovery pass detects these broken functions and
        reconstructs them.
        """
        elements: list[ExtractedElement] = []
        root = tree.root_node

        for child in _iter_top_level_nodes(root):
            if child.type == "function_definition":
                elem = _extract_function(child, lines)
                if elem:
                    elements.append(elem)
                    # Extract variables from function body
                    elements.extend(_extract_function_body_variables(child, lines))
            elif child.type == "variable_assignment":
                elem = _extract_variable(child, lines)
                if elem:
                    elements.append(elem)
            elif child.type == "declaration_command":
                elem = _extract_declaration(child, lines)
                if elem:
                    elements.append(elem)

        # Recovery pass: find functions that tree-sitter failed to parse
        if root.has_error:
            recovered = _recover_broken_functions(lines, elements)
            elements.extend(recovered)

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


# Regex matching top-level function definitions: name() { (with optional
# 'function' keyword).  Anchored to start of line to avoid matching function
# calls or definitions inside heredocs / strings.
_FUNC_DEF_RE = re.compile(
    r"^(?:function\s+)?(\w+)\s*\(\)\s*\{",
    re.MULTILINE,
)


def _recover_broken_functions(
    lines: list[str], existing: list[ExtractedElement]
) -> list[ExtractedElement]:
    """Recover function definitions that tree-sitter failed to parse.

    When tree-sitter-bash encounters syntax it cannot handle (ANSI-C quoting
    inside parameter expansions, glob character classes in variable
    substitutions, etc.), it produces ERROR nodes that destroy
    ``function_definition`` AST nodes.  This causes functions — and often
    several subsequent functions — to be silently dropped.

    This recovery pass uses a regex scan + brace-counting to detect and
    reconstruct missing functions.

    Args:
        lines: Source code lines.
        existing: Elements already extracted from the tree-sitter AST.

    Returns:
        List of recovered function elements not already present in *existing*.
    """
    source = "\n".join(lines)
    # Build a set of (name, line_start) already extracted
    known: set[tuple[str, int]] = {
        (e.name, e.line_start) for e in existing if e.element_type == "function"
    }

    recovered: list[ExtractedElement] = []

    for match in _FUNC_DEF_RE.finditer(source):
        name = match.group(1)
        line_start = source[: match.start()].count("\n") + 1

        if (name, line_start) in known:
            continue

        # Find the closing brace by counting braces from the opening {
        brace_pos = match.end() - 1  # Position of the opening {
        line_end = _find_closing_brace(source, brace_pos)
        if line_end is None:
            continue

        raw_code = "\n".join(lines[line_start - 1 : line_end])
        byte_offset = match.start()

        recovered.append(
            ExtractedElement(
                element_type="function",
                name=name,
                line_start=line_start,
                line_end=line_end,
                raw_code=raw_code,
                byte_offset=byte_offset,
                node=None,
            )
        )

    return recovered


def _find_closing_brace(source: str, open_pos: int) -> int | None:
    """Find the line number of the closing ``}`` matching the ``{`` at *open_pos*.

    Performs brace counting while skipping:
    - Single-quoted strings (``'...'``)
    - Double-quoted strings (``"..."`` with backslash escapes)
    - ANSI-C strings (``$'...'`` with backslash escapes)
    - Comments (``# ...`` to end of line)
    - Heredocs (``<<[-]'?WORD'?`` ... ``WORD``)

    Returns the 1-indexed line number of the closing brace, or ``None`` if
    not found.
    """
    depth = 1
    i = open_pos + 1
    length = len(source)

    while i < length and depth > 0:
        ch = source[i]

        if ch == "#":
            # Skip to end of line (comment)
            nl = source.find("\n", i)
            i = nl + 1 if nl != -1 else length
            continue

        if ch == "'":
            # Single-quoted string: no escapes, find closing '
            end = source.find("'", i + 1)
            i = end + 1 if end != -1 else length
            continue

        if ch == "$" and i + 1 < length and source[i + 1] == "'":
            # ANSI-C string $'...' with backslash escapes
            i += 2
            while i < length:
                if source[i] == "\\" and i + 1 < length:
                    i += 2
                elif source[i] == "'":
                    i += 1
                    break
                else:
                    i += 1
            continue

        if ch == '"':
            # Double-quoted string with backslash escapes
            i += 1
            while i < length:
                if source[i] == "\\" and i + 1 < length:
                    i += 2
                elif source[i] == '"':
                    i += 1
                    break
                else:
                    i += 1
            continue

        if ch == "<" and i + 1 < length and source[i + 1] == "<":
            # Possible heredoc
            heredoc_end = _skip_heredoc(source, i)
            if heredoc_end is not None:
                i = heredoc_end
                continue

        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return source[: i + 1].count("\n") + 1

        i += 1

    return None


def _skip_heredoc(source: str, pos: int) -> int | None:
    """Skip a heredoc starting at ``pos`` (pointing to the first ``<``).

    Handles ``<<WORD``, ``<<-WORD``, ``<<'WORD'``, ``<<"WORD"``.

    Returns the position *after* the heredoc terminator line, or ``None`` if
    this is not a valid heredoc.
    """
    # Must start with <<
    if source[pos : pos + 2] != "<<":
        return None

    i = pos + 2
    length = len(source)

    # Optional - for <<- (strip leading tabs)
    if i < length and source[i] == "-":
        i += 1

    # Skip whitespace (but not newline)
    while i < length and source[i] in (" ", "\t"):
        i += 1

    if i >= length or source[i] == "\n":
        return None

    # Extract delimiter word (possibly quoted)
    quote_char = None
    if source[i] in ("'", '"'):
        quote_char = source[i]
        i += 1

    delim_start = i
    while i < length and source[i] not in ("\n", " ", "\t"):
        if quote_char and source[i] == quote_char:
            break
        i += 1

    delimiter = source[delim_start:i]
    if not delimiter:
        return None

    if quote_char and i < length and source[i] == quote_char:
        i += 1

    # Find the heredoc terminator line
    # Move to next line
    nl = source.find("\n", i)
    if nl == -1:
        return None
    i = nl + 1

    while i < length:
        nl = source.find("\n", i)
        if nl == -1:
            line_text = source[i:]
            i = length
        else:
            line_text = source[i:nl]
            i = nl + 1

        # Check if this line is the terminator (possibly with leading tabs for <<-)
        stripped = line_text.lstrip("\t") if pos + 2 < length and source[pos + 2] == "-" else line_text
        if stripped.strip() == delimiter:
            return i

    return None


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
        # Keywords like readonly/export/declare/local have their own node types
        if child.type in ("readonly", "export", "declare", "typeset", "local"):
            keyword = child.type
        elif child.type == "word":
            text = get_node_text(child)
            # declare/typeset flags like -r, -a, -A
            if text.startswith("-") and "r" in text:
                has_readonly_flag = True

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


def _extract_function_body_variables(
    func_node: Node, lines: list[str]
) -> list[ExtractedElement]:
    """Extract variable assignments and declarations from inside a Bash function body.

    Args:
        func_node: A function_definition node.
        lines: Source code lines.

    Returns:
        List of variable/constant elements found inside the function.
    """
    variables: list[ExtractedElement] = []

    # Find the compound_statement (function body)
    body = None
    for child in func_node.children:
        if child.type == "compound_statement":
            body = child
            break

    if not body:
        return variables

    for node in walk_tree(body):
        # Skip nested function definitions
        if node.type == "function_definition":
            continue

        if node.type == "variable_assignment":
            # Skip variable_assignment nodes inside declaration_command
            # (e.g. `local x="hello"` has both a declaration_command and
            # a child variable_assignment — avoid double extraction)
            if node.parent and node.parent.type == "declaration_command":
                continue
            elem = _extract_variable(node, lines)
            if elem:
                variables.append(elem)
        elif node.type == "declaration_command":
            elem = _extract_declaration(node, lines)
            if elem:
                variables.append(elem)

    return variables


def extract_bash_imports(tree: Tree, _lines: list[str]) -> list[ExtractedImport]:
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


# Node types that define function scopes in Bash
_BASH_SCOPE_TYPES = frozenset({
    "function_definition",
})


def extract_top_level_bash_calls(tree: Tree) -> list[ExtractedCall]:
    """Extract command calls from the top level of a Bash script.

    Walks file-scope statements (skipping function definition subtrees)
    and collects command calls, excluding shell builtins.

    Args:
        tree: A parsed tree-sitter Tree for a Bash file.

    Returns:
        List of extracted calls found at file scope.
    """
    calls: list[ExtractedCall] = []
    seen: set[tuple[str, int]] = set()

    for node in _walk_top_level_bash(tree.root_node, _BASH_SCOPE_TYPES):
        if node.type != "command":
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
