"""Markdown extractor - extracts heading sections from markdown files."""

from __future__ import annotations

from tree_sitter import Tree

from magaldi_core.extractors.base import get_node_text


def extract_markdown_sections(tree: Tree, lines: list[str]) -> list[dict]:
    """Extract heading sections from a markdown AST.

    Returns a list of section dicts: {title, level, line_start, line_end}.
    Each section spans from its heading to the line before the next
    same-or-higher-level heading (or EOF).
    """
    sections: list[dict] = []
    _collect_headings(tree.root_node, sections)

    # Compute line_end for each section
    total_lines = len(lines)
    for i, section in enumerate(sections):
        # Find next section at same or higher level
        next_start = total_lines
        for j in range(i + 1, len(sections)):
            if sections[j]["level"] <= section["level"]:
                next_start = sections[j]["line_start"] - 1
                break
        section["line_end"] = next_start

    return sections


def _collect_headings(node: object, sections: list[dict]) -> None:
    """Recursively collect atx_heading nodes from the AST."""
    # tree-sitter-markdown wraps content in nested section nodes
    if getattr(node, "type", "") == "atx_heading":
        level = _heading_level(node)
        title = _heading_title(node)
        line_start = node.start_point[0] + 1  # 1-indexed
        sections.append({
            "title": title,
            "level": level,
            "line_start": line_start,
            "line_end": line_start,  # Will be computed later
        })

    for child in getattr(node, "children", []):
        _collect_headings(child, sections)


def _heading_level(node: object) -> int:
    """Determine heading level from the marker node (atx_h1_marker, etc.)."""
    for child in getattr(node, "children", []):
        node_type = getattr(child, "type", "")
        if node_type.startswith("atx_h") and node_type.endswith("_marker"):
            # atx_h1_marker -> 1, atx_h2_marker -> 2, etc.
            try:
                return int(node_type[5])
            except (IndexError, ValueError):
                pass
    return 1


def _heading_title(node: object) -> str:
    """Extract the heading title text from an atx_heading node."""
    for child in getattr(node, "children", []):
        if getattr(child, "type", "") == "inline":
            return get_node_text(child).strip()
    return ""


class MarkdownExtractor:
    """Extractor for markdown files."""

    @property
    def language(self) -> str:
        return "markdown"

    def extract_elements(self, tree: Tree, lines: list[str]) -> list:
        """Extract elements from markdown. Returns empty list.

        Markdown sections are extracted separately via extract_markdown_sections()
        and stored as document_sections on the file element, not as sub-elements.
        """
        return []

    def extract_sections(self, tree: Tree, lines: list[str]) -> list[dict]:
        """Extract heading sections for document_sections metadata."""
        return extract_markdown_sections(tree, lines)
