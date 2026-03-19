"""SFC (Single File Component) section splitter for Vue and Svelte files.

Splits `.vue` and `.svelte` files into their constituent sections
(template, script, style) with accurate line offset tracking so that
sub-parsers can extract elements with correct line numbers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class SFCSection:
    """A section extracted from an SFC file.

    Attributes:
        tag: Section type — "template", "script", or "style".
        content: Inner content of the section (between open/close tags).
        line_start: 1-indexed line number where inner content starts
                    in the original file.
        attrs: Tag attributes as key-value pairs.
              e.g. {"lang": "ts", "setup": ""} for <script setup lang="ts">.
    """

    tag: str
    content: str
    line_start: int
    attrs: dict[str, str] = field(default_factory=dict)


# Regex to match top-level SFC tags: <template>, <script>, <style>
# Captures: tag name, attributes, inner content
_SFC_TAG_RE = re.compile(
    r"<(template|script|style)\b([^>]*)>"  # Opening tag with attrs
    r"(.*?)"                                # Inner content (non-greedy)
    r"</\1\s*>",                            # Matching closing tag
    re.DOTALL,
)

# Regex to parse attributes from the opening tag
_ATTR_RE = re.compile(
    r"""(\w[\w-]*)"""             # Attribute name
    r"""(?:\s*=\s*"""             # Optional = value
    r"""(?:"([^"]*)"|'([^']*)')"""  # Quoted value
    r""")?""",
)


def _parse_attrs(attr_str: str) -> dict[str, str]:
    """Parse HTML-style attributes from a tag's attribute string.

    Args:
        attr_str: Raw attribute string, e.g. ' setup lang="ts" scoped'.

    Returns:
        Dict of attribute name → value. Valueless attributes get empty string.
    """
    attrs: dict[str, str] = {}
    for match in _ATTR_RE.finditer(attr_str.strip()):
        name = match.group(1)
        value = match.group(2) if match.group(2) is not None else (match.group(3) or "")
        attrs[name] = value
    return attrs


def _count_newlines_before(text: str, pos: int) -> int:
    """Count newlines in text before position `pos`."""
    return text[:pos].count("\n")


def split_sfc(content: str) -> list[SFCSection]:
    """Split an SFC file into its constituent sections.

    Extracts <template>, <script>, and <style> sections from Vue or Svelte
    files. Each section includes its inner content and the line offset
    where that content starts in the original file.

    Args:
        content: Full file content of a .vue or .svelte file.

    Returns:
        List of SFCSection objects, one per matched section.
    """
    sections: list[SFCSection] = []

    for match in _SFC_TAG_RE.finditer(content):
        tag = match.group(1)
        attr_str = match.group(2)
        inner = match.group(3)

        # Calculate line offset: lines before the match + the opening tag line
        # The inner content starts on the line after the opening tag
        opening_tag_end = match.start(3)
        line_start = _count_newlines_before(content, opening_tag_end) + 1

        sections.append(SFCSection(
            tag=tag,
            content=inner,
            line_start=line_start,
            attrs=_parse_attrs(attr_str),
        ))

    return sections


def extract_template_from_svelte(content: str) -> tuple[str, int]:
    """Extract template content from a Svelte file.

    In Svelte, the template is everything outside <script> and <style> tags.
    This extracts those sections, removes them, and returns the remaining
    HTML as the template.

    Args:
        content: Full Svelte file content.

    Returns:
        Tuple of (template_content, template_line_start).
        template_line_start is always 1 since template is at the top level.
    """
    # Remove all <script> and <style> blocks
    result = content
    for match in reversed(list(_SFC_TAG_RE.finditer(content))):
        if match.group(1) in ("script", "style"):
            result = result[:match.start()] + result[match.end():]

    return result.strip(), 1
