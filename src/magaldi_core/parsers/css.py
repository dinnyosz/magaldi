"""CSS and SCSS parsers — no tree-sitter, uses tinycss2 + regex.

Extracts logical sections from stylesheets:
- Rule blocks (.class { ... })
- @media queries
- @keyframes animations
- @font-face declarations
- CSS custom properties (:root { --var: ...; })
- SCSS variables ($var: value;)
- SCSS mixins (@mixin name { ... })
- SCSS functions (@function name { ... })
- SCSS nested rules

Summaries are generated heuristically (no LLM) via css_summary module.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

import tinycss2

from magaldi_core.parsers.base import CodeElement, generate_element_id
from magaldi_core.parsers.css_summary import (
    summarize_at_rule,
    summarize_custom_properties,
    summarize_file,
    summarize_font_face,
    summarize_keyframes,
    summarize_media,
    summarize_rule,
)

if TYPE_CHECKING:
    from magaldi_core.change_detection import FileInfo


# =============================================================================
# HELPERS
# =============================================================================


def _source_line_to_offset(content: str, line: int) -> int:
    """Convert 1-indexed line number to byte offset in content."""
    offset = 0
    for i, text_line in enumerate(content.split("\n"), 1):
        if i == line:
            return offset
        offset += len(text_line.encode("utf-8")) + 1  # +1 for \n
    return offset


def _count_lines_in(text: str) -> int:
    """Count lines in a text block."""
    return text.count("\n") + 1


def _serialize_node(node: tinycss2.ast.Node) -> str:
    """Serialize a tinycss2 node to CSS text."""
    return tinycss2.serialize([node])


def _serialize_nodes(nodes: list) -> str:
    """Serialize a list of tinycss2 nodes to CSS text."""
    return tinycss2.serialize(nodes)


def _has_custom_properties(block_text: str) -> bool:
    """Check if a block contains CSS custom properties (--*)."""
    return bool(re.search(r"--[\w-]+\s*:", block_text))


def _find_end_line(content: str, start_line: int, raw_code: str) -> int:  # noqa: ARG001
    """Find the end line of a raw_code block starting at start_line."""
    return start_line + raw_code.count("\n")


# =============================================================================
# CSS PARSER
# =============================================================================


class CSSParser:
    """Parse CSS files without tree-sitter.

    Uses tinycss2 for tokenization and creates CodeElement per logical section.
    Has the same parse() interface as PlainTextParser.
    """

    def __init__(self, language: str = "css") -> None:
        self._language = language

    def parse(
        self,
        content: str,
        file_info: FileInfo,
        scope: str,
        repository: str,
        username: str,
    ) -> list[CodeElement]:
        """Parse CSS content into logical sections."""
        lines = content.split("\n")
        line_count = len(lines)
        elements: list[CodeElement] = []

        # Parse with tinycss2
        rules = tinycss2.parse_stylesheet(
            content, skip_whitespace=True, skip_comments=False
        )

        # Create file-level element first
        file_element = CodeElement(
            scope=scope,
            repository=repository,
            username=username,
            relative_path=file_info.relative_path,
            element_type="file",
            name=Path(file_info.relative_path).name,
            language=self._language,
            line_start=1,
            line_end=line_count,
            level=0,
            raw_code=content,
        )
        file_element.element_id = generate_element_id(
            scope, repository, username, file_info.relative_path,
            "file", file_element.name, 0,
        )

        # Track preceding comments to attach as docstrings
        pending_comment: str | None = None

        # Extract elements from top-level rules
        for rule in rules:
            if rule.type == "comment":
                pending_comment = rule.value.strip()
                continue

            element = self._process_rule(
                rule, content, file_info, scope, repository, username,
                parent_id=file_element.element_id,
            )
            if element is not None:
                if pending_comment:
                    element.docstring = pending_comment
                elements.append(element)

            pending_comment = None

        # Set file summary with child count
        child_count = len(elements)
        file_element.summary = summarize_file(
            file_element.name, child_count, self._language
        )

        return [file_element, *elements]

    def _process_rule(
        self,
        rule: tinycss2.ast.Node,
        content: str,
        file_info: FileInfo,
        scope: str,
        repository: str,
        username: str,
        parent_id: str | None = None,
    ) -> CodeElement | None:
        """Process a single tinycss2 rule into a CodeElement."""
        if rule.type == "qualified-rule":
            return self._process_qualified_rule(
                rule, content, file_info, scope, repository, username, parent_id
            )
        if rule.type == "at-rule":
            return self._process_at_rule(
                rule, content, file_info, scope, repository, username, parent_id
            )
        return None

    def _process_qualified_rule(
        self,
        rule: tinycss2.ast.QualifiedRule,
        content: str,
        file_info: FileInfo,
        scope: str,
        repository: str,
        username: str,
        parent_id: str | None,
    ) -> CodeElement:
        """Process a qualified rule (selector { declarations })."""
        selector = _serialize_nodes(rule.prelude).strip()
        block_text = _serialize_nodes(rule.content)
        raw_code = _serialize_node(rule)
        line_start = rule.source_line
        line_end = _find_end_line(content, line_start, raw_code)
        byte_offset = _source_line_to_offset(content, line_start)

        # Check if this is a custom property block (:root, [data-theme], etc.)
        if _has_custom_properties(block_text):
            summary = summarize_custom_properties(selector, block_text)
            element_type = "variable"
        else:
            summary = summarize_rule(selector, block_text)
            element_type = "css_rule"

        element = CodeElement(
            scope=scope,
            repository=repository,
            username=username,
            relative_path=file_info.relative_path,
            element_type=element_type,
            name=selector,
            language=self._language,
            line_start=line_start,
            line_end=line_end,
            byte_offset=byte_offset,
            level=1,
            parent_id=parent_id,
            raw_code=raw_code,
            signature=selector,
            summary=summary,
        )
        element.element_id = generate_element_id(
            scope, repository, username, file_info.relative_path,
            element_type, selector, byte_offset,
        )
        return element

    def _process_at_rule(
        self,
        rule: tinycss2.ast.AtRule,
        content: str,
        file_info: FileInfo,
        scope: str,
        repository: str,
        username: str,
        parent_id: str | None,
    ) -> CodeElement:
        """Process an at-rule (@media, @keyframes, @font-face, etc.)."""
        at_name = rule.lower_at_keyword  # e.g., "media", "keyframes"
        prelude = _serialize_nodes(rule.prelude).strip()
        block_text = _serialize_nodes(rule.content) if rule.content is not None else ""
        raw_code = _serialize_node(rule)
        line_start = rule.source_line
        line_end = _find_end_line(content, line_start, raw_code)
        byte_offset = _source_line_to_offset(content, line_start)

        # Determine element type and summary based on at-rule kind
        if at_name == "media":
            summary = summarize_media(prelude, block_text)
            element_type = "css_rule"
            name = f"@media {prelude}"
        elif at_name == "keyframes":
            summary = summarize_keyframes(prelude, block_text)
            element_type = "css_rule"
            name = f"@keyframes {prelude}"
        elif at_name == "font-face":
            summary = summarize_font_face(block_text)
            element_type = "css_rule"
            name = "@font-face"
        else:
            summary = summarize_at_rule(at_name, prelude, block_text or None)
            element_type = "css_rule"
            name = f"@{at_name} {prelude}".strip()

        element = CodeElement(
            scope=scope,
            repository=repository,
            username=username,
            relative_path=file_info.relative_path,
            element_type=element_type,
            name=name,
            language=self._language,
            line_start=line_start,
            line_end=line_end,
            byte_offset=byte_offset,
            level=1,
            parent_id=parent_id,
            raw_code=raw_code,
            signature=name,
            summary=summary,
        )
        element.element_id = generate_element_id(
            scope, repository, username, file_info.relative_path,
            element_type, name, byte_offset,
        )
        return element


# =============================================================================
# SCSS PARSER
# =============================================================================

# Regex patterns for SCSS-specific constructs (not handled by tinycss2)

# Matches both single-line and multi-line variable declarations:
#   $color: red;
#   $map: (key: value, ...);
#   $list: a, b, c !default;
_SCSS_VARIABLE_RE = re.compile(
    r"^(\$[\w-]+)\s*:\s*("
    r"\([^)]*\)"  # Parenthesized map/list value: (key: val, ...)
    r"|"
    r"[^;]+"  # Non-paren value: everything up to ;
    r")(?:\s*!default)?\s*;",
    re.MULTILINE | re.DOTALL,
)
_SCSS_MIXIN_RE = re.compile(
    r"@mixin\s+([\w-]+)\s*(?:\(([^)]*)\))?\s*\{",
    re.MULTILINE,
)
_SCSS_FUNCTION_RE = re.compile(
    r"@function\s+([\w-]+)\s*(?:\(([^)]*)\))?\s*\{",
    re.MULTILINE,
)
_SCSS_INCLUDE_RE = re.compile(r"@include\s+([\w-]+)")
_SCSS_USE_RE = re.compile(
    r'^@use\s+"([^"]+)"(?:\s+as\s+([\w*]+))?',
    re.MULTILINE,
)
_SCSS_FORWARD_RE = re.compile(
    r'^@forward\s+"([^"]+)"',
    re.MULTILINE,
)
# Full-line versions (including semicolons) for stripping
_SCSS_USE_STRIP_RE = re.compile(r"^@use\s+[^\n]*;?", re.MULTILINE)
_SCSS_FORWARD_STRIP_RE = re.compile(r"^@forward\s+[^\n]*;?", re.MULTILINE)
# SCSS single-line comments (// ...) — must be stripped before tinycss2
_SCSS_LINE_COMMENT_RE = re.compile(r"//[^\n]*")
# SCSS control flow (@each, @for, @while, @if, @else) with braces
_SCSS_CONTROL_RE = re.compile(
    r"@(?:each|for|while|if|else\s+if|else)\b[^{]*\{",
    re.MULTILINE,
)


def _find_matching_brace(content: str, start: int) -> int:
    """Find the matching closing brace for the opening brace at `start`.

    Args:
        content: Full file content.
        start: Index of the opening '{'.

    Returns:
        Index of the matching '}', or len(content) if not found.
    """
    depth = 0
    i = start
    while i < len(content):
        if content[i] == "{":
            depth += 1
        elif content[i] == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return len(content)


def _line_number_at(content: str, pos: int) -> int:
    """Get 1-indexed line number for a byte position."""
    return content[:pos].count("\n") + 1


class SCSSParser(CSSParser):
    """Parse SCSS files — extends CSSParser with SCSS-specific constructs.

    SCSS has constructs that tinycss2 can't parse (variables, mixins, functions,
    nesting with &). We pre-extract those with regex, then let tinycss2 handle
    the CSS-compatible parts.
    """

    def __init__(self) -> None:
        super().__init__(language="scss")

    def parse(
        self,
        content: str,
        file_info: FileInfo,
        scope: str,
        repository: str,
        username: str,
    ) -> list[CodeElement]:
        """Parse SCSS content — extracts SCSS constructs + CSS rules."""
        lines = content.split("\n")
        line_count = len(lines)

        # Create file-level element
        file_element = CodeElement(
            scope=scope,
            repository=repository,
            username=username,
            relative_path=file_info.relative_path,
            element_type="file",
            name=Path(file_info.relative_path).name,
            language="scss",
            line_start=1,
            line_end=line_count,
            level=0,
            raw_code=content,
        )
        file_element.element_id = generate_element_id(
            scope, repository, username, file_info.relative_path,
            "file", file_element.name, 0,
        )

        elements: list[CodeElement] = []

        # 1. Extract SCSS variables (single-line and multi-line maps)
        elements.extend(
            self._extract_scss_variables(
                content, file_info, scope, repository, username,
                file_element.element_id,
            )
        )

        # 2. Extract SCSS mixins
        elements.extend(
            self._extract_scss_mixins(
                content, file_info, scope, repository, username,
                file_element.element_id,
            )
        )

        # 3. Extract SCSS functions
        elements.extend(
            self._extract_scss_functions(
                content, file_info, scope, repository, username,
                file_element.element_id,
            )
        )

        # 3b. Extract @use/@forward as imports on the file element
        self._extract_scss_imports(content, file_element)

        # 4. Extract CSS-compatible rules via tinycss2
        # Strip SCSS-only syntax that would confuse tinycss2
        css_compatible = self._strip_scss_syntax(content)
        try:
            css_rules = tinycss2.parse_stylesheet(
                css_compatible, skip_whitespace=True, skip_comments=False
            )
            pending_comment: str | None = None
            for rule in css_rules:
                if rule.type == "comment":
                    pending_comment = rule.value.strip()
                    continue
                if rule.type == "error":
                    pending_comment = None
                    continue
                element = self._process_rule(
                    rule, css_compatible, file_info, scope, repository, username,
                    parent_id=file_element.element_id,
                )
                if element is not None:
                    # Fix language to scss (base class sets it from self._language)
                    if pending_comment:
                        element.docstring = pending_comment
                    elements.append(element)
                pending_comment = None
        except Exception:
            # If tinycss2 chokes on SCSS remnants, that's OK — we have the
            # SCSS-specific elements already extracted above
            pass

        # Set file summary
        file_element.summary = summarize_file(
            file_element.name, len(elements), "scss"
        )

        return [file_element, *elements]

    def _extract_scss_variables(
        self,
        content: str,
        file_info: FileInfo,
        scope: str,
        repository: str,
        username: str,
        parent_id: str,
    ) -> list[CodeElement]:
        """Extract top-level SCSS variables ($name: value;).

        Handles both single-line and multi-line map/list variables:
          $color: red;
          $breakpoints: (sm: 640px, md: 768px, lg: 1024px);
        """
        from magaldi_core.parsers.css_summary import summarize_scss_variable

        elements: list[CodeElement] = []
        for m in _SCSS_VARIABLE_RE.finditer(content):
            name = m.group(1)
            value = m.group(2).strip()
            line_start = _line_number_at(content, m.start())
            raw_code = m.group(0)
            line_end = _line_number_at(content, m.end() - 1)  # End of full match
            byte_offset = m.start()

            element = CodeElement(
                scope=scope,
                repository=repository,
                username=username,
                relative_path=file_info.relative_path,
                element_type="variable",
                name=name,
                language="scss",
                line_start=line_start,
                line_end=line_end,
                byte_offset=byte_offset,
                level=1,
                parent_id=parent_id,
                raw_code=raw_code,
                signature=f"{name}: {value}",
                summary=summarize_scss_variable(name, value),
            )
            element.element_id = generate_element_id(
                scope, repository, username, file_info.relative_path,
                "variable", name, byte_offset,
            )
            elements.append(element)
        return elements

    def _extract_scss_mixins(
        self,
        content: str,
        file_info: FileInfo,
        scope: str,
        repository: str,
        username: str,
        parent_id: str,
    ) -> list[CodeElement]:
        """Extract SCSS @mixin definitions."""
        from magaldi_core.parsers.css_summary import summarize_scss_mixin

        elements: list[CodeElement] = []
        for m in _SCSS_MIXIN_RE.finditer(content):
            name = m.group(1)
            params = (m.group(2) or "").strip()
            brace_start = m.end() - 1  # Position of '{'
            brace_end = _find_matching_brace(content, brace_start)
            raw_code = content[m.start() : brace_end + 1]
            block_content = content[brace_start + 1 : brace_end]
            line_start = _line_number_at(content, m.start())
            line_end = _line_number_at(content, brace_end)
            byte_offset = m.start()

            element = CodeElement(
                scope=scope,
                repository=repository,
                username=username,
                relative_path=file_info.relative_path,
                element_type="function",
                name=name,
                language="scss",
                line_start=line_start,
                line_end=line_end,
                byte_offset=byte_offset,
                level=1,
                parent_id=parent_id,
                raw_code=raw_code,
                signature=f"@mixin {name}({params})" if params else f"@mixin {name}",
                summary=summarize_scss_mixin(name, params, block_content),
            )
            element.element_id = generate_element_id(
                scope, repository, username, file_info.relative_path,
                "function", name, byte_offset,
            )
            elements.append(element)
        return elements

    def _extract_scss_functions(
        self,
        content: str,
        file_info: FileInfo,
        scope: str,
        repository: str,
        username: str,
        parent_id: str,
    ) -> list[CodeElement]:
        """Extract SCSS @function definitions."""
        from magaldi_core.parsers.css_summary import summarize_scss_function

        elements: list[CodeElement] = []
        for m in _SCSS_FUNCTION_RE.finditer(content):
            name = m.group(1)
            params = (m.group(2) or "").strip()
            brace_start = m.end() - 1
            brace_end = _find_matching_brace(content, brace_start)
            raw_code = content[m.start() : brace_end + 1]
            block_content = content[brace_start + 1 : brace_end]
            line_start = _line_number_at(content, m.start())
            line_end = _line_number_at(content, brace_end)
            byte_offset = m.start()

            element = CodeElement(
                scope=scope,
                repository=repository,
                username=username,
                relative_path=file_info.relative_path,
                element_type="function",
                name=name,
                language="scss",
                line_start=line_start,
                line_end=line_end,
                byte_offset=byte_offset,
                level=1,
                parent_id=parent_id,
                raw_code=raw_code,
                signature=f"@function {name}({params})" if params else f"@function {name}",
                summary=summarize_scss_function(name, params, block_content),
            )
            element.element_id = generate_element_id(
                scope, repository, username, file_info.relative_path,
                "function", name, byte_offset,
            )
            elements.append(element)
        return elements

    def _extract_scss_imports(
        self,
        content: str,
        file_element: CodeElement,
    ) -> None:
        """Extract @use and @forward statements as imports on the file element."""
        from magaldi_core.parsers.base import Import

        for m in _SCSS_USE_RE.finditer(content):
            module = m.group(1)
            alias = m.group(2)  # May be None
            line = _line_number_at(content, m.start())
            # Module name is the last segment (e.g., "sass:math" -> "math")
            name = module.rsplit("/", 1)[-1].rsplit(":", 1)[-1].lstrip("_")
            file_element.imports.append(Import(
                name=alias or name,
                module=module,
                alias=alias,
                line=line,
            ))

        for m in _SCSS_FORWARD_RE.finditer(content):
            module = m.group(1)
            line = _line_number_at(content, m.start())
            name = module.rsplit("/", 1)[-1].lstrip("_")
            file_element.imports.append(Import(
                name=name,
                module=module,
                alias=None,
                line=line,
            ))

    def _strip_scss_syntax(self, content: str) -> str:
        """Strip SCSS-only syntax so tinycss2 can parse the CSS-compatible parts.

        Removes:
        - // single-line comments (tinycss2 only understands /* */ comments)
        - $variable declarations (single-line and multi-line maps)
        - @mixin ... { ... } blocks
        - @function ... { ... } blocks
        - @include statements
        - @use/@forward statements
        - @each/@for/@while/@if control flow blocks
        - #{interpolation} → placeholder text
        - %placeholder selectors → .placeholder

        Preserves line count by replacing with blank lines where possible.
        """
        result = content

        # 1. Strip // comments (MUST be first — they confuse everything else)
        result = _SCSS_LINE_COMMENT_RE.sub(lambda m: " " * len(m.group(0)), result)

        # 2. Replace $variable declarations (including multi-line maps) with blanks
        result = _SCSS_VARIABLE_RE.sub(
            lambda m: re.sub(r"[^\n]", " ", m.group(0)), result
        )

        # 3. Replace @mixin blocks with blanks
        # Work backwards to avoid position shifting
        mixin_matches = list(_SCSS_MIXIN_RE.finditer(result))
        for m in reversed(mixin_matches):
            brace_start = m.end() - 1
            brace_end = _find_matching_brace(result, brace_start)
            block = result[m.start() : brace_end + 1]
            replacement = re.sub(r"[^\n]", " ", block)
            result = result[:m.start()] + replacement + result[brace_end + 1:]

        # 4. Replace @function blocks with blanks
        func_matches = list(_SCSS_FUNCTION_RE.finditer(result))
        for m in reversed(func_matches):
            brace_start = m.end() - 1
            brace_end = _find_matching_brace(result, brace_start)
            block = result[m.start() : brace_end + 1]
            replacement = re.sub(r"[^\n]", " ", block)
            result = result[:m.start()] + replacement + result[brace_end + 1:]

        # 5. Replace @each/@for/@while/@if control flow blocks
        ctrl_matches = list(_SCSS_CONTROL_RE.finditer(result))
        for m in reversed(ctrl_matches):
            brace_start = m.end() - 1
            brace_end = _find_matching_brace(result, brace_start)
            block = result[m.start() : brace_end + 1]
            replacement = re.sub(r"[^\n]", " ", block)
            result = result[:m.start()] + replacement + result[brace_end + 1:]

        # 6. Replace @include with spaces
        result = _SCSS_INCLUDE_RE.sub(lambda m: " " * len(m.group(0)), result)

        # 7. Replace @use/@forward full lines with spaces
        result = _SCSS_USE_STRIP_RE.sub(lambda m: " " * len(m.group(0)), result)
        result = _SCSS_FORWARD_STRIP_RE.sub(lambda m: " " * len(m.group(0)), result)

        # 8. Replace %placeholder selectors with .placeholder (valid CSS)
        result = re.sub(r"%(?=[\w-])", "._ph_", result)

        # 9. Replace #{...} interpolation with a safe placeholder
        result = re.sub(r"#\{[^}]+\}", "INTERP", result)

        return result
