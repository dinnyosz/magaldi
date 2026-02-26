"""Markdown parser using tree-sitter.

Extracts document structure (headings) from markdown files.
Headings are stored as document_sections metadata on the file element.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from magaldi_core.extractors.markdown import MarkdownExtractor
from magaldi_core.parsers.base import CodeElement, TreeSitterParser

if TYPE_CHECKING:
    from magaldi_core.change_detection import FileInfo


class MarkdownParser(TreeSitterParser):
    """Parse markdown files using tree-sitter."""

    def __init__(self) -> None:
        super().__init__("markdown")
        self._extractor = MarkdownExtractor()

    def parse(
        self,
        content: str,
        file_info: FileInfo,
        scope: str,
        repository: str,
        username: str,
    ) -> list[CodeElement]:
        """Parse markdown content and extract document structure."""
        lines = content.split("\n")

        # Create file-level element
        file_element = self._create_file_element(
            content, file_info, scope, repository, username, "markdown"
        )

        # Extract heading sections as metadata on file element
        tree = self.manager.parse(content.encode("utf-8"), "markdown")
        file_element.document_sections = self._extractor.extract_sections(tree, lines)

        return [file_element]
