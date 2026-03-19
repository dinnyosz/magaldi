"""HTML parser — converts HTML to markdown, then extracts document structure.

Creates a file-level element with the markdown-converted content as raw_code,
making summarization input 3-10x smaller than raw HTML. Heading sections are
extracted from the converted markdown using the existing MarkdownExtractor.
CSS classes and element IDs are annotated inline next to their elements.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from bs4 import BeautifulSoup
from markdownify import MarkdownConverter, markdownify

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


# Tags whose classes/IDs are meaningful enough to annotate inline.
# Structural tags like html, head, body, meta are noise.
_ANNOTATABLE_TAGS = frozenset({
    "div", "section", "article", "aside", "main", "header", "footer",
    "nav", "form", "fieldset", "details", "dialog", "figure",
    "a", "button", "input", "select", "textarea", "label",
    "ul", "ol", "li", "dl", "dt", "dd",
    "table", "thead", "tbody", "tfoot", "tr", "th", "td",
    "h1", "h2", "h3", "h4", "h5", "h6",
    "p", "span", "img", "video", "audio", "canvas", "svg",
    "iframe", "embed", "object",
})


def _build_attr_suffix(el: Any) -> str:
    """Build a compact CSS-selector-style suffix for a tag's class/id.

    Examples:
        <div id="app" class="container flex"> → ' {#app .container .flex}'
        <nav class="sidebar"> → ' {.sidebar}'
        <p> → ''

    Args:
        el: BeautifulSoup Tag element.

    Returns:
        Annotation string (empty if no class/id on this tag).
    """
    if el.name not in _ANNOTATABLE_TAGS:
        return ""

    parts: list[str] = []
    tag_id = el.get("id")
    if tag_id:
        parts.append(f"#{tag_id}")

    tag_classes = el.get("class", [])
    if isinstance(tag_classes, str):
        tag_classes = tag_classes.split()
    for cls in tag_classes:
        parts.append(f".{cls}")

    if not parts:
        return ""
    return " {" + " ".join(parts) + "}"


class _AnnotatingConverter(MarkdownConverter):
    """Custom markdownify converter that annotates elements with CSS classes/IDs.

    Overrides structural tag converters to append {#id .class} suffixes
    so the LLM sees styling context inline with the content.
    """

    def convert_div(self, el: Any, text: str, parent_tags: Any) -> str:
        suffix = _build_attr_suffix(el)
        if suffix and text.strip():
            # Annotation on its own line before the block content
            return f"\n\n{suffix}\n{text.strip()}\n\n"
        return f"\n\n{text}\n\n" if text.strip() else ""

    def convert_section(self, el: Any, text: str, parent_tags: Any) -> str:
        return self.convert_div(el, text, parent_tags)

    def convert_article(self, el: Any, text: str, parent_tags: Any) -> str:
        return self.convert_div(el, text, parent_tags)

    def convert_main(self, el: Any, text: str, parent_tags: Any) -> str:
        return self.convert_div(el, text, parent_tags)

    def convert_header(self, el: Any, text: str, parent_tags: Any) -> str:
        return self.convert_div(el, text, parent_tags)

    def convert_footer(self, el: Any, text: str, parent_tags: Any) -> str:
        return self.convert_div(el, text, parent_tags)

    def convert_aside(self, el: Any, text: str, parent_tags: Any) -> str:
        return self.convert_div(el, text, parent_tags)

    def convert_nav(self, el: Any, text: str, parent_tags: Any) -> str:
        return self.convert_div(el, text, parent_tags)

    def convert_form(self, el: Any, text: str, parent_tags: Any) -> str:
        return self.convert_div(el, text, parent_tags)

    def convert_a(self, el: Any, text: str, parent_tags: Any) -> str:
        # Let parent handle the markdown link conversion, then append attrs
        result = super().convert_a(el, text, parent_tags)
        suffix = _build_attr_suffix(el)
        if suffix and result.strip():
            return result.rstrip() + suffix
        return result

    def convert_li(self, el: Any, text: str, parent_tags: Any) -> str:
        result = super().convert_li(el, text, parent_tags)
        suffix = _build_attr_suffix(el)
        if suffix and result.strip():
            # Insert suffix before the trailing newline
            return result.rstrip("\n") + suffix + "\n"
        return result


def _convert_html_to_markdown(html: str) -> str:
    """Convert HTML to markdown with inline CSS class/ID annotations.

    Args:
        html: Raw HTML content.

    Returns:
        Markdown string with {#id .class} annotations on structural elements.
    """
    converter = _AnnotatingConverter(
        heading_style="ATX",
        escape_asterisks=False,
        escape_underscores=False,
        newline_style="BACKSLASH",
        table_infer_header=True,
    )
    return converter.convert(html)


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
        # Extract CSS classes and IDs from original HTML
        css_classes: list[str] = []
        css_ids: list[str] = []
        if content.strip():
            css_classes, css_ids = _extract_css_classes_and_ids(content)

        # Convert HTML → markdown with inline class/ID annotations
        md_content = _convert_html_to_markdown(content)
        # Clean up excessive blank lines from conversion
        while "\n\n\n" in md_content:
            md_content = md_content.replace("\n\n\n", "\n\n")
        md_content = md_content.strip()

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
