"""HTML parser — converts HTML to markdown, then extracts document structure.

Creates a file-level element with the markdown-converted content as raw_code,
making summarization input 3-10x smaller than raw HTML. Heading sections are
extracted from the converted markdown using the existing MarkdownExtractor.
CSS classes and element IDs are extracted as code_metrics metadata.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from bs4 import BeautifulSoup
from markdownify import markdownify

from magaldi_core.extractors.markdown import MarkdownExtractor
from magaldi_core.parsers.base import CodeElement, generate_element_id
from magaldi_core.tree_sitter_manager import get_manager

if TYPE_CHECKING:
    from magaldi_core.change_detection import FileInfo


def _extract_css_classes_and_ids(html: str) -> tuple[list[str], list[str]]:
    """Extract unique CSS class names and element IDs from HTML.

    Args:
        html: Raw HTML content.

    Returns:
        Tuple of (sorted unique classes, sorted unique IDs).
    """
    soup = BeautifulSoup(html, "html.parser")
    classes: set[str] = set()
    ids: set[str] = set()

    for tag in soup.find_all(True):
        # Classes: each element can have multiple
        tag_classes = tag.get("class", [])
        if isinstance(tag_classes, list):
            classes.update(tag_classes)
        elif isinstance(tag_classes, str):
            classes.update(tag_classes.split())

        # IDs: one per element
        tag_id = tag.get("id")
        if tag_id:
            ids.add(tag_id)

    return sorted(classes), sorted(ids)


class HtmlParser:
    """Parse HTML files by converting to markdown first.

    Converts HTML → markdown via markdownify, stores the markdown as raw_code,
    and extracts heading sections using tree-sitter-markdown. Does not extend
    TreeSitterParser since HTML has no tree-sitter grammar — the tree-sitter
    usage is only for section extraction on the converted markdown.
    """

    def __init__(self) -> None:
        self._manager = get_manager()
        self._extractor = MarkdownExtractor()

    def parse(
        self,
        content: str,
        file_info: FileInfo,
        scope: str,
        repository: str,
        username: str,
    ) -> list[CodeElement]:
        """Parse HTML content by converting to markdown.

        Args:
            content: Raw HTML content.
            file_info: File metadata from change detection.
            scope: Repository scope.
            repository: Repository name.
            username: Username (branch).

        Returns:
            List with a single file-level CodeElement containing
            markdown-converted content as raw_code.
        """
        # Extract CSS classes and IDs from original HTML before conversion
        css_classes: list[str] = []
        css_ids: list[str] = []
        if content.strip():
            css_classes, css_ids = _extract_css_classes_and_ids(content)

        # Convert HTML → markdown
        md_content = markdownify(
            content,
            heading_style="ATX",
            escape_asterisks=False,
            escape_underscores=False,
            newline_style="BACKSLASH",
            table_infer_header=True,
        )
        # Clean up excessive blank lines from conversion
        while "\n\n\n" in md_content:
            md_content = md_content.replace("\n\n\n", "\n\n")
        md_content = md_content.strip()

        # Prepend CSS classes/IDs inline so the LLM sees them in context
        meta_lines: list[str] = []
        if css_classes:
            meta_lines.append(f"CSS classes: {', '.join(css_classes)}")
        if css_ids:
            meta_lines.append(f"Element IDs: {', '.join(css_ids)}")
        if meta_lines and md_content:
            md_content = "\n".join(meta_lines) + "\n\n" + md_content
        elif meta_lines:
            md_content = "\n".join(meta_lines)

        lines = md_content.split("\n")
        line_count = len(lines)

        # Create file-level element with markdown as raw_code
        file_element = CodeElement(
            scope=scope,
            repository=repository,
            username=username,
            relative_path=file_info.relative_path,
            element_type="file",
            name=Path(file_info.relative_path).name,
            language=file_info.language,
            line_start=1,
            line_end=line_count,
            level=0,
            raw_code=md_content,
        )
        file_element.element_id = generate_element_id(
            scope, repository, username, file_info.relative_path,
            "file", file_element.name, 1,
        )

        # Store CSS classes and IDs as code_metrics for MCP/search
        if css_classes or css_ids:
            file_element.code_metrics = {}
            if css_classes:
                file_element.code_metrics["css_classes"] = css_classes
            if css_ids:
                file_element.code_metrics["css_ids"] = css_ids

        # Extract heading sections from the converted markdown
        if md_content:
            tree = self._manager.parse(md_content.encode("utf-8"), "markdown")
            file_element.document_sections = self._extractor.extract_sections(
                tree, lines
            )

        return [file_element]
