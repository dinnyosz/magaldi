"""TOML extractor - extracts tables and keys from TOML files."""

from __future__ import annotations

from tree_sitter import Tree

from magaldi_core.extractors.base import get_node_text
from magaldi_core.extractors.types import ExtractedElement


class TomlExtractor:
    """Extractor for TOML files.

    Extracts all tables (including dotted sub-tables) as constant elements.
    """

    @property
    def language(self) -> str:
        return "toml"

    def extract_elements(self, tree: Tree, lines: list[str]) -> list[ExtractedElement]:
        """Extract tables from TOML."""
        elements: list[ExtractedElement] = []
        root = tree.root_node

        for child in root.children:
            if child.type == "table":
                self._extract_table(child, lines, elements)

        return elements

    def _extract_table(
        self,
        node: object,
        lines: list[str],
        elements: list[ExtractedElement],
    ) -> None:
        """Extract a TOML table as a constant element."""
        table_name = self._get_table_name(node)
        if not table_name:
            return

        line_start = node.start_point[0] + 1  # type: ignore[attr-defined]
        line_end = node.end_point[0] + 1  # type: ignore[attr-defined]
        raw_code = "\n".join(lines[line_start - 1:line_end])

        elements.append(ExtractedElement(
            element_type="constant",
            name=table_name,
            line_start=line_start,
            line_end=line_end,
            raw_code=raw_code,
            node=node,
        ))

    def _get_table_name(self, node: object) -> str:
        """Extract table header name from a table node.

        Handles both simple keys ([project]) and dotted keys ([tool.ruff.lint]).
        """
        for child in getattr(node, "children", []):
            child_type = getattr(child, "type", "")
            if child_type == "bare_key":
                return get_node_text(child)  # type: ignore[no-any-return]
            if child_type == "dotted_key":
                return self._get_dotted_key(child)
        return ""

    def _get_dotted_key(self, node: object) -> str:
        """Reconstruct a dotted key like tool.ruff.lint."""
        parts: list[str] = []
        for child in getattr(node, "children", []):
            child_type = getattr(child, "type", "")
            if child_type == "bare_key":
                parts.append(get_node_text(child))
            elif child_type == "dotted_key":
                parts.append(self._get_dotted_key(child))
        return ".".join(parts)
