"""YAML extractor - extracts mapping keys recursively from YAML files."""

from __future__ import annotations

from tree_sitter import Tree

from magaldi_core.extractors.base import get_node_text
from magaldi_core.extractors.types import ExtractedElement


class YamlExtractor:
    """Extractor for YAML files.

    Recursively extracts mapping keys that have nested mappings as values.
    Leaf scalar values are not extracted as separate elements.
    """

    @property
    def language(self) -> str:
        return "yaml"

    def extract_elements(self, tree: Tree, lines: list[str]) -> list[ExtractedElement]:
        """Extract structural mapping keys from YAML."""
        elements: list[ExtractedElement] = []
        root = tree.root_node

        # YAML AST: stream -> document -> block_node -> block_mapping -> block_mapping_pair*
        for doc_node in root.children:
            if doc_node.type != "document":
                continue
            for child in doc_node.children:
                self._extract_from_node(child, lines, elements, prefix="")

        return elements

    def _extract_from_node(
        self,
        node: object,
        lines: list[str],
        elements: list[ExtractedElement],
        prefix: str,
    ) -> None:
        """Recursively extract mapping pairs."""
        node_type = getattr(node, "type", "")

        if node_type == "block_mapping":
            for child in getattr(node, "children", []):
                if getattr(child, "type", "") == "block_mapping_pair":
                    self._extract_mapping_pair(child, lines, elements, prefix)

        elif node_type == "block_node":
            for child in getattr(node, "children", []):
                self._extract_from_node(child, lines, elements, prefix)

    def _extract_mapping_pair(
        self,
        node: object,
        lines: list[str],
        elements: list[ExtractedElement],
        prefix: str,
    ) -> None:
        """Extract a single mapping pair and recurse into nested mappings.

        Only extracts keys that have nested mappings as values (structural nodes).
        Leaf scalar pairs are not extracted as separate elements — they're
        captured in their parent's raw_code.
        """
        children = getattr(node, "children", [])
        key_name = ""
        value_node = None

        for child in children:
            child_type = getattr(child, "type", "")
            if child_type in ("flow_node", "block_scalar"):
                # Key node
                if not key_name:
                    key_name = get_node_text(child).strip()
            elif child_type == "block_node":
                value_node = child

        if not key_name:
            return

        full_name = f"{prefix}{key_name}" if not prefix else f"{prefix}.{key_name}"
        has_nested_mapping = _has_nested_mapping(value_node) if value_node else False

        # Only extract keys that have nested mappings (structural nodes)
        # Skip leaf scalars/sequences — they're part of parent's raw_code
        if not has_nested_mapping:
            return

        line_start = node.start_point[0] + 1  # type: ignore[attr-defined]
        line_end = node.end_point[0] + 1  # type: ignore[attr-defined]
        raw_code = "\n".join(lines[line_start - 1:line_end])

        elements.append(ExtractedElement(
            element_type="constant",
            name=full_name,
            line_start=line_start,
            line_end=line_end,
            raw_code=raw_code,
            node=node,
        ))

        # Recurse into nested mappings
        self._extract_from_node(value_node, lines, elements, prefix=full_name)


def _has_nested_mapping(node: object) -> bool:
    """Check if a node contains a nested block_mapping."""
    if getattr(node, "type", "") == "block_mapping":
        return True
    return any(_has_nested_mapping(child) for child in getattr(node, "children", []))
