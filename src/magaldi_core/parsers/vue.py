"""Vue Single File Component (.vue) parser.

Splits .vue files into <template>, <script>, and <style> sections,
then delegates each to the appropriate parser:
- template → strip Vue syntax, then HtmlParser for markdown conversion
- script → JavaScriptParser (or TypeScript if lang="ts")
- style → CSSParser or SCSSParser (if lang="scss")

Elements from script/style sections get their line numbers adjusted
to reflect their position in the original .vue file.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

from magaldi_core.parsers.base import CodeElement, generate_element_id
from magaldi_core.parsers.css import CSSParser, SCSSParser
from magaldi_core.parsers.html import HtmlParser
from magaldi_core.parsers.javascript import JavaScriptParser
from magaldi_core.parsers.sfc_splitter import SFCSection, split_sfc

if TYPE_CHECKING:
    from magaldi_core.change_detection import FileInfo


# =============================================================================
# VUE TEMPLATE SYNTAX STRIPPING
# =============================================================================

# Vue-specific template syntax to strip before HTML parsing
_VUE_PATTERNS: list[re.Pattern[str]] = [
    # Vue interpolation: {{ expr }} → keep expression text
    # (handled as replacement below)
]

_VUE_REPLACEMENTS: list[tuple[re.Pattern[str], str]] = [
    # Interpolation: {{ expr }} → expression text
    (re.compile(r"\{\{\s*(.*?)\s*\}\}", re.DOTALL), r"\1"),
    # v-bind shorthand: :attr="expr" → attr="expr"
    (re.compile(r':(\w[\w-]*)="([^"]*)"'), r'\1="\2"'),
    # v-on shorthand: @event="handler" → remove entirely
    (re.compile(r'@\w[\w.-]*="[^"]*"'), ""),
    # v-directives: v-if="expr", v-for="item in items", v-show, v-model, etc.
    (re.compile(r'v-[\w.-]+="[^"]*"'), ""),
    # Valueless v-directives: v-else, v-once, v-cloak, v-pre
    (re.compile(r"\bv-(?:else|once|cloak|pre)\b"), ""),
    # Slot syntax: #slotName="props" or v-slot:name="props"
    (re.compile(r'(?:#|v-slot:)\w+(?:="[^"]*")?'), ""),
]


def _strip_vue_syntax(html: str) -> str:
    """Strip Vue-specific template syntax from HTML content.

    Removes directives (v-if, v-for, @click, :class), interpolation {{ }},
    and slot syntax, leaving clean HTML for markdownify.

    Args:
        html: Raw Vue template content.

    Returns:
        Cleaned HTML content.
    """
    result = html
    for pattern in _VUE_PATTERNS:
        result = pattern.sub("", result)
    for pattern, repl in _VUE_REPLACEMENTS:
        result = pattern.sub(repl, result)
    return result


# =============================================================================
# PARSER LANGUAGE MAPPING
# =============================================================================

# Map <script lang="..."> values to parser language keys
_SCRIPT_LANG_MAP: dict[str, str] = {
    "ts": "typescript",
    "typescript": "typescript",
    "tsx": "tsx",
    "js": "javascript",
    "javascript": "javascript",
    "": "javascript",  # default
}

# Map <style lang="..."> values to parser types
_STYLE_LANG_MAP: dict[str, str] = {
    "scss": "scss",
    "css": "css",
    "": "css",  # default
}


# =============================================================================
# VUE PARSER
# =============================================================================


class VueParser:
    """Parse Vue Single File Components (.vue files).

    Splits the file into template/script/style sections and delegates
    each to the appropriate sub-parser. The template becomes the file
    element's raw_code (converted to markdown), while script and style
    produce child elements with adjusted line numbers.
    """

    def __init__(self) -> None:
        self._html_parser = HtmlParser()
        self._js_parsers: dict[str, JavaScriptParser] = {
            "javascript": JavaScriptParser("javascript"),
            "typescript": JavaScriptParser("typescript"),
            "tsx": JavaScriptParser("tsx"),
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
        """Parse a Vue SFC and extract elements from all sections.

        Args:
            content: Full .vue file content.
            file_info: File metadata from change detection.
            scope: Repository scope.
            repository: Repository name.
            username: Username (branch).

        Returns:
            List of CodeElements: file element + script/style child elements.
        """
        sections = split_sfc(content)

        # Find each section type
        template_section: SFCSection | None = None
        script_sections: list[SFCSection] = []
        style_sections: list[SFCSection] = []

        for section in sections:
            if section.tag == "template":
                template_section = section
            elif section.tag == "script":
                script_sections.append(section)
            elif section.tag == "style":
                style_sections.append(section)

        # --- Template → file element via HtmlParser ---
        template_html = ""
        if template_section:
            template_html = _strip_vue_syntax(template_section.content)

        # Create a mock file_info for HtmlParser (it uses file_info.language)
        html_fi = MagicMock()
        html_fi.relative_path = file_info.relative_path
        html_fi.absolute_path = file_info.absolute_path
        html_fi.language = "html"

        elements = self._html_parser.parse(
            template_html, html_fi, scope, repository, username,
        )

        # Fix the file element: set language to "vue"
        file_element = elements[0]
        file_element.language = "vue"
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
                # Adjust line numbers and filter out the script's own file element
                for elem in script_elements:
                    if elem.element_type == "file":
                        continue
                    elem.line_start += section.line_start - 1
                    elem.line_end += section.line_start - 1
                    elem.language = "vue"
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
                # Adjust line numbers and filter out the style's own file element
                for elem in style_elements:
                    if elem.element_type == "file":
                        continue
                    elem.line_start += section.line_start - 1
                    elem.line_end += section.line_start - 1
                    elem.language = "vue"
                    elements.append(elem)

        return elements
