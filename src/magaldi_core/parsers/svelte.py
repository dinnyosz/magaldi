"""Svelte Single File Component (.svelte) parser.

Svelte files differ from Vue in that the template (HTML) is at the top level,
not wrapped in a <template> tag. <script> and <style> sections are extracted
and the remaining content is treated as the template.

- template (top-level HTML) → strip Svelte syntax, then HtmlParser
- script → JavaScriptParser (or TypeScript if lang="ts")
- style → CSSParser or SCSSParser (if lang="scss")
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

from magaldi_core.parsers.base import CodeElement
from magaldi_core.parsers.css import CSSParser, SCSSParser
from magaldi_core.parsers.html import HtmlParser
from magaldi_core.parsers.javascript import JavaScriptParser
from magaldi_core.parsers.sfc_splitter import split_sfc, extract_template_from_svelte

if TYPE_CHECKING:
    from magaldi_core.change_detection import FileInfo


# =============================================================================
# SVELTE TEMPLATE SYNTAX STRIPPING
# =============================================================================

_SVELTE_PATTERNS: list[re.Pattern[str]] = [
    # Block logic: {#if expr}, {:else if expr}, {:else}, {/if}
    re.compile(r"\{[#:/]\s*(?:if|else\s*if|else|each|await|then|catch|key)\b[^}]*\}", re.DOTALL),
    # Special tags: {@html expr}, {@debug vars}, {@const x = ...}
    re.compile(r"\{@\w+\b[^}]*\}", re.DOTALL),
]

_SVELTE_REPLACEMENTS: list[tuple[re.Pattern[str], str]] = [
    # Expression interpolation: {expr} → expression text
    # Must come after block removal to avoid matching {#if} etc.
    (re.compile(r"\{([^#:/@ ][^}]*)\}"), r"\1"),
    # Event handlers: on:click={handler} or on:click|modifiers={handler} → remove
    (re.compile(r'on:\w+(?:\|[\w|]+)?=\{[^}]*\}'), ""),
    # Bind directives: bind:value={expr} → remove
    (re.compile(r'bind:\w+=\{[^}]*\}'), ""),
    # Transitions/animations: transition:fade, animate:flip, in:fly, out:fade
    (re.compile(r'(?:transition|animate|in|out):\w+(?:=\{[^}]*\})?'), ""),
    # Use directives: use:action={params}
    (re.compile(r'use:\w+(?:=\{[^}]*\})?'), ""),
    # Class directive: class:name={expr} → class="name"
    (re.compile(r'class:(\w[\w-]*)(?:=\{[^}]*\})?'), r'class="\1"'),
]


def _strip_svelte_syntax(html: str) -> str:
    """Strip Svelte-specific template syntax from HTML content.

    Removes block logic ({#if}, {#each}), event handlers, bind directives,
    and converts interpolation {expr} to plain text.

    Args:
        html: Raw Svelte template content.

    Returns:
        Cleaned HTML content.
    """
    result = html
    for pattern in _SVELTE_PATTERNS:
        result = pattern.sub("", result)
    for pattern, repl in _SVELTE_REPLACEMENTS:
        result = pattern.sub(repl, result)
    return result


# Language mapping (same as Vue)
_SCRIPT_LANG_MAP: dict[str, str] = {
    "ts": "typescript",
    "typescript": "typescript",
    "js": "javascript",
    "javascript": "javascript",
    "": "javascript",
}

_STYLE_LANG_MAP: dict[str, str] = {
    "scss": "scss",
    "css": "css",
    "": "css",
}


# =============================================================================
# SVELTE PARSER
# =============================================================================


class SvelteParser:
    """Parse Svelte Single File Components (.svelte files).

    In Svelte, the template is the top-level HTML (not in a <template> tag).
    <script> and <style> sections are extracted separately.
    """

    def __init__(self) -> None:
        self._html_parser = HtmlParser()
        self._js_parsers: dict[str, JavaScriptParser] = {
            "javascript": JavaScriptParser("javascript"),
            "typescript": JavaScriptParser("typescript"),
        }
        self._css_parser = CSSParser()
        self._scss_parser = SCSSParser()

    def parse(
        self,
        content: str,
        file_info: FileInfo,
        scope: str,
        repository: str,
        username: str,
    ) -> list[CodeElement]:
        """Parse a Svelte SFC and extract elements from all sections.

        Args:
            content: Full .svelte file content.
            file_info: File metadata from change detection.
            scope: Repository scope.
            repository: Repository name.
            username: Username (branch).

        Returns:
            List of CodeElements: file element + script/style child elements.
        """
        sections = split_sfc(content)

        # Extract script and style sections
        script_sections = [s for s in sections if s.tag == "script"]
        style_sections = [s for s in sections if s.tag == "style"]

        # Template is everything outside <script> and <style>
        template_html, _ = extract_template_from_svelte(content)
        template_html = _strip_svelte_syntax(template_html)

        # --- Template → file element via HtmlParser ---
        html_fi = MagicMock()
        html_fi.relative_path = file_info.relative_path
        html_fi.absolute_path = file_info.absolute_path
        html_fi.language = "html"

        elements = self._html_parser.parse(
            template_html, html_fi, scope, repository, username,
        )

        # Fix the file element
        file_element = elements[0]
        file_element.language = "svelte"
        file_element.line_end = content.count("\n") + 1

        # --- Script sections → child elements ---
        for section in script_sections:
            lang_key = _SCRIPT_LANG_MAP.get(
                section.attrs.get("lang", ""), "javascript"
            )
            parser = self._js_parsers.get(lang_key)
            if parser and section.content.strip():
                script_fi = MagicMock()
                script_fi.relative_path = file_info.relative_path
                script_fi.absolute_path = file_info.absolute_path
                script_fi.language = lang_key

                script_elements = parser.parse(
                    section.content, script_fi, scope, repository, username,
                )
                for elem in script_elements:
                    if elem.element_type == "file":
                        continue
                    elem.line_start += section.line_start - 1
                    elem.line_end += section.line_start - 1
                    elem.language = "svelte"
                    elements.append(elem)

        # --- Style sections → child elements ---
        for section in style_sections:
            style_lang = _STYLE_LANG_MAP.get(
                section.attrs.get("lang", ""), "css"
            )
            if section.content.strip():
                style_fi = MagicMock()
                style_fi.relative_path = file_info.relative_path
                style_fi.absolute_path = file_info.absolute_path
                style_fi.language = style_lang

                if style_lang == "scss":
                    style_elements = self._scss_parser.parse(
                        section.content, style_fi, scope, repository, username,
                    )
                else:
                    style_elements = self._css_parser.parse(
                        section.content, style_fi, scope, repository, username,
                    )
                for elem in style_elements:
                    if elem.element_type == "file":
                        continue
                    elem.line_start += section.line_start - 1
                    elem.line_end += section.line_start - 1
                    elem.language = "svelte"
                    elements.append(elem)

        return elements
