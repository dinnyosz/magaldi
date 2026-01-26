"""Semantic analysis functions for code.

This module provides functions for extracting semantic information:
- TODO/FIXME comment extraction
- Section marker extraction
- Comment association with elements
- Purity analysis
- Side effect detection
- Type annotation extraction
"""

from __future__ import annotations

import re

from tree_sitter import Node

from magaldi_core.extractors.base import (
    find_nodes,
    get_child_by_field,
    get_node_text,
)
from magaldi_core.extractors.types import (
    Comment,
    ExtractedCall,
    PurityInfo,
    SectionMarker,
    SideEffect,
    TodoItem,
    TypeAnnotation,
)

# =============================================================================
# TODO EXTRACTION
# =============================================================================

# TODO extraction pattern
_TODO_PATTERN = re.compile(
    r"(?P<kind>TODO|FIXME|HACK|XXX|BUG|NOTE|OPTIMIZE)"
    r"(?:\((?P<assignee>\w+)\))?"  # TODO(alice)
    r"(?:\s*(?P<priority>!+))?"  # TODO!!
    r"(?:\s*(?P<issue>[#]?\w+-?\d+))?"  # TODO #123 or GH-456
    r"\s*:?\s*"
    r"(?P<text>.+)",
    re.IGNORECASE,
)


def extract_todos(source: str) -> list[TodoItem]:
    """Extract TODO/FIXME comments from source code."""
    todos = []
    lines = source.split("\n")

    for line_num, line in enumerate(lines, start=1):
        comment_start = -1
        for marker in ("#", "//", "/*", "*"):
            idx = line.find(marker)
            if idx != -1 and (comment_start == -1 or idx < comment_start):
                comment_start = idx

        if comment_start == -1:
            continue

        comment_text = line[comment_start:]
        match = _TODO_PATTERN.search(comment_text)
        if match:
            priority = None
            if match.group("priority"):
                priority = "high" if len(match.group("priority")) >= 2 else "low"

            todos.append(
                TodoItem(
                    kind=match.group("kind").upper(),
                    text=match.group("text").strip(),
                    line=line_num,
                    assignee=match.group("assignee"),
                    priority=priority,
                    issue_ref=match.group("issue"),
                )
            )

    return todos


# =============================================================================
# SECTION MARKER EXTRACTION
# =============================================================================

# Section marker pattern
_SECTION_PATTERN = re.compile(
    r"^\s*[#/]+\s*"
    r"(?P<style_start>={3,}|-{3,})?\s*"
    r"(?P<label>[A-Z][A-Z0-9 _]+)"
    r"\s*(?P<style_end>={3,}|-{3,})?\s*$",
)


def extract_section_markers(source: str) -> list[SectionMarker]:
    """Extract section marker comments from source code."""
    markers = []
    lines = source.split("\n")

    for line_num, line in enumerate(lines, start=1):
        match = _SECTION_PATTERN.match(line)
        if match:
            style_start = match.group("style_start") or ""
            style_end = match.group("style_end") or ""

            if "=" in style_start or "=" in style_end:
                style = "equals"
            elif "-" in style_start or "-" in style_end:
                style = "dashes"
            else:
                style = "hash"

            markers.append(
                SectionMarker(
                    label=match.group("label").strip(),
                    line=line_num,
                    style=style,
                )
            )

    return markers


# =============================================================================
# COMMENT EXTRACTION
# =============================================================================


def extract_comments(source: str) -> list[Comment]:
    """Extract all comments from source code."""
    comments = []
    lines = source.split("\n")

    for line_num, line in enumerate(lines, start=1):
        stripped = line.strip()

        if not stripped:
            continue

        # Python/shell style comments
        if "#" in line:
            hash_pos = line.find("#")
            code_before = line[:hash_pos].strip()
            kind = "inline" if code_before else "block"
            position = "inline" if kind == "inline" else "above"

            comments.append(
                Comment(
                    text=line[hash_pos + 1 :].strip(),
                    line=line_num,
                    kind=kind,
                    position=position,
                )
            )
        # C-style single line comments
        elif "//" in line:
            slash_pos = line.find("//")
            code_before = line[:slash_pos].strip()
            kind = "inline" if code_before else "block"
            position = "inline" if kind == "inline" else "above"

            comments.append(
                Comment(
                    text=line[slash_pos + 2 :].strip(),
                    line=line_num,
                    kind=kind,
                    position=position,
                )
            )

    return comments


def associate_comments(
    element_line: int,
    all_comments: list[Comment],
    max_distance: int = 3,
) -> list[Comment]:
    """Associate comments with an element based on proximity."""
    associated = []

    for comment in all_comments:
        if element_line - max_distance <= comment.line < element_line:
            comment.position = "above"
            associated.append(comment)
        elif comment.line == element_line and comment.kind == "inline":
            comment.position = "inline"
            associated.append(comment)

    return associated


# =============================================================================
# PURITY ANALYSIS
# =============================================================================

# Impure function call patterns by language
IMPURE_CALLS: dict[str, dict[str, list[str]]] = {
    "python": {
        "io_file": [
            "open", "read", "write", "Path.write_text", "Path.read_text",
            "Path.mkdir", "os.remove", "shutil.copy", "shutil.move",
        ],
        "io_network": [
            "requests.get", "requests.post", "requests.put", "requests.delete",
            "httpx.get", "httpx.post", "urllib.request.urlopen", "socket.connect",
        ],
        "console": [
            "print", "logging.info", "logging.debug", "logging.warning",
            "logging.error", "logger.info", "logger.debug", "logger.warning",
        ],
        "subprocess": [
            "subprocess.run", "subprocess.call", "subprocess.Popen",
            "os.system", "os.popen",
        ],
        "database": [
            "cursor.execute", "session.commit", "session.add", "session.delete",
        ],
    },
    "javascript": {
        "io_file": [
            "fs.readFile", "fs.writeFile", "fs.readFileSync", "fs.writeFileSync",
            "fs.appendFile", "fs.unlink", "fs.mkdir",
        ],
        "io_network": [
            "fetch", "axios.get", "axios.post", "http.get", "http.request",
        ],
        "console": [
            "console.log", "console.error", "console.warn", "console.info",
        ],
        "subprocess": [
            "child_process.exec", "child_process.spawn", "child_process.execSync",
        ],
    },
    "typescript": {},  # Falls back to javascript
    "php": {
        "io_file": [
            "file_get_contents", "file_put_contents", "fopen", "fwrite", "fread",
        ],
        "io_network": [
            "curl_exec", "file_get_contents",
        ],
        "console": [
            "echo", "print", "var_dump", "print_r",
        ],
    },
    "rust": {
        "io_file": [
            "File::open", "File::create", "fs::read", "fs::write",
        ],
        "io_network": [
            "TcpStream::connect", "reqwest::get",
        ],
        "console": [
            "println!", "eprintln!", "print!", "dbg!",
        ],
        "subprocess": [
            "Command::new", "process::Command",
        ],
    },
}


def analyze_purity(
    calls: list[ExtractedCall],
    mutations: list[str],
    language: str,
) -> PurityInfo:
    """Analyze function purity based on calls and mutations.

    Args:
        calls: List of function calls within the function.
        mutations: List of mutated attributes (e.g., ["self.cache"]).
        language: Programming language.

    Returns:
        PurityInfo with level, confidence, and reasons.
    """
    reasons = []
    level = "pure"

    # Get impure patterns for this language (fall back to python for unknown)
    lang_patterns = IMPURE_CALLS.get(language, IMPURE_CALLS.get("python", {}))

    # If language not found and no fallback, use python patterns
    if not lang_patterns and language == "typescript":
        lang_patterns = IMPURE_CALLS.get("javascript", {})

    # Check for impure calls
    for call in calls:
        call_name = call.name
        if call.receiver:
            call_name = f"{call.receiver}.{call.name}"

        for effect_kind, patterns in lang_patterns.items():
            for pattern in patterns:
                if call_name == pattern or call.name == pattern:
                    reasons.append(f"calls {call_name} ({effect_kind})")
                    level = "mutates_external"
                    break

    # Check for self mutations
    for mutation in mutations:
        if mutation.startswith("self.") or mutation.startswith("this."):
            reasons.append(f"modifies {mutation}")
            if level == "pure":
                level = "mutates_self"

    # Determine confidence
    if level == "pure" and not calls:
        confidence = "high"
    elif level == "pure" and calls:
        # Has calls but none matched impure patterns - lower confidence
        confidence = "medium"
    else:
        confidence = "high"

    return PurityInfo(level=level, confidence=confidence, reasons=reasons)


def extract_side_effects(
    calls: list[ExtractedCall],
    mutations: list[str],
    language: str,
) -> list[SideEffect]:
    """Extract individual side effects from calls and mutations.

    Args:
        calls: List of function calls.
        mutations: List of mutations.
        language: Programming language.

    Returns:
        List of SideEffect objects.
    """
    effects = []
    lang_patterns = IMPURE_CALLS.get(language, IMPURE_CALLS.get("python", {}))

    if not lang_patterns and language == "typescript":
        lang_patterns = IMPURE_CALLS.get("javascript", {})

    for call in calls:
        call_name = call.name
        if call.receiver:
            call_name = f"{call.receiver}.{call.name}"

        for effect_kind, patterns in lang_patterns.items():
            for pattern in patterns:
                if call_name == pattern or call.name == pattern:
                    effects.append(
                        SideEffect(kind=effect_kind, target=call_name, line=call.line)
                    )
                    break

    for mutation in mutations:
        effects.append(
            SideEffect(kind="state_mutation", target=mutation, line=0)
        )

    return effects


# =============================================================================
# TYPE ANNOTATION EXTRACTION
# =============================================================================


def _parse_generic_type(type_str: str) -> tuple[str, list[str] | None]:
    """Parse a generic type like List[str] into base and args.

    Args:
        type_str: Type string like "List[str]" or "Dict[str, int]"

    Returns:
        Tuple of (full_type, generic_args or None)
    """
    if "[" not in type_str:
        return type_str, None

    bracket_pos = type_str.index("[")
    # Extract the content between brackets
    inner = type_str[bracket_pos + 1 : -1]  # Remove outer brackets
    # Simple split by comma (doesn't handle nested generics perfectly)
    args = [arg.strip() for arg in inner.split(",")]
    return type_str, args


def extract_type_annotations(node: Node, language: str) -> list[TypeAnnotation]:
    """Extract type annotations from an AST node.

    Args:
        node: Root AST node to search.
        language: Programming language.

    Returns:
        List of TypeAnnotation objects.
    """
    annotations = []

    if language == "python":
        annotations.extend(_extract_python_type_annotations(node))
    elif language in ("typescript", "javascript"):
        annotations.extend(_extract_typescript_type_annotations(node))

    return annotations


def _extract_python_type_annotations(node: Node) -> list[TypeAnnotation]:
    """Extract type annotations from Python AST."""
    annotations = []

    for func_node in find_nodes(node, "function_definition"):
        func_line = func_node.start_point[0] + 1

        # Get parameters
        params = get_child_by_field(func_node, "parameters")
        if params:
            for param in params.children:
                if param.type == "typed_parameter":
                    name_node = param.children[0] if param.children else None
                    type_node = get_child_by_field(param, "type")

                    if name_node and type_node:
                        param_name = get_node_text(name_node)
                        type_str = get_node_text(type_node)
                        full_type, generic_args = _parse_generic_type(type_str)

                        annotations.append(
                            TypeAnnotation(
                                name=full_type,
                                kind="parameter",
                                location=f"param:{param_name}",
                                line=func_line,
                                generic_args=generic_args,
                            )
                        )

        # Get return type
        return_type = get_child_by_field(func_node, "return_type")
        if return_type:
            type_str = get_node_text(return_type)
            full_type, generic_args = _parse_generic_type(type_str)

            annotations.append(
                TypeAnnotation(
                    name=full_type,
                    kind="return",
                    location="return",
                    line=func_line,
                    generic_args=generic_args,
                )
            )

    return annotations


def _extract_typescript_type_annotations(node: Node) -> list[TypeAnnotation]:
    """Extract type annotations from TypeScript AST."""
    annotations = []

    # Find function declarations and arrow functions
    for func_type in ("function_declaration", "arrow_function", "method_definition"):
        for func_node in find_nodes(node, func_type):
            func_line = func_node.start_point[0] + 1

            # Look for type annotations in parameters
            params = get_child_by_field(func_node, "parameters")
            if params:
                for param in params.children:
                    if param.type in ("required_parameter", "optional_parameter"):
                        # Look for type_annotation child
                        for child in param.children:
                            if child.type == "type_annotation":
                                type_node = child.children[-1] if child.children else None
                                name_node = param.children[0] if param.children else None

                                if type_node and name_node:
                                    param_name = get_node_text(name_node)
                                    type_str = get_node_text(type_node)
                                    full_type, generic_args = _parse_generic_type(type_str)

                                    annotations.append(
                                        TypeAnnotation(
                                            name=full_type,
                                            kind="parameter",
                                            location=f"param:{param_name}",
                                            line=func_line,
                                            generic_args=generic_args,
                                        )
                                    )

            # Look for return type annotation
            for child in func_node.children:
                if child.type == "type_annotation":
                    type_node = child.children[-1] if child.children else None
                    if type_node:
                        type_str = get_node_text(type_node)
                        full_type, generic_args = _parse_generic_type(type_str)

                        annotations.append(
                            TypeAnnotation(
                                name=full_type,
                                kind="return",
                                location="return",
                                line=func_line,
                                generic_args=generic_args,
                            )
                        )

    return annotations
