"""Tests for the markdown heading extractor."""

from magaldi_core.extractors.markdown import MarkdownExtractor, extract_markdown_sections
from magaldi_core.tree_sitter_manager import get_manager


class TestMarkdownExtractor:
    """Tests for markdown heading section extraction."""

    def _parse_and_extract(self, content: str) -> list[dict]:
        manager = get_manager()
        tree = manager.parse(content.encode("utf-8"), "markdown")
        return extract_markdown_sections(tree, content.split("\n"))

    def test_extracts_single_heading(self):
        sections = self._parse_and_extract("# Title\n\nSome text\n")
        assert len(sections) == 1
        assert sections[0]["title"] == "Title"
        assert sections[0]["level"] == 1
        assert sections[0]["line_start"] == 1

    def test_extracts_multiple_headings(self):
        content = "# Title\n\nIntro\n\n## Section A\n\nContent A\n\n## Section B\n\nContent B\n"
        sections = self._parse_and_extract(content)
        assert len(sections) == 3
        assert sections[0]["title"] == "Title"
        assert sections[0]["level"] == 1
        assert sections[1]["title"] == "Section A"
        assert sections[1]["level"] == 2
        assert sections[2]["title"] == "Section B"
        assert sections[2]["level"] == 2

    def test_nested_heading_levels(self):
        content = "# H1\n\n## H2\n\n### H3\n\n## Another H2\n"
        sections = self._parse_and_extract(content)
        assert len(sections) == 4
        levels = [s["level"] for s in sections]
        assert levels == [1, 2, 3, 2]

    def test_section_line_end_calculation(self):
        content = "# Title\n\nSome text\n\n## Section A\n\nContent A\n\n## Section B\n\nContent B\n"
        lines = content.split("\n")
        sections = self._parse_and_extract(content)

        # Title section ends before Section A
        assert sections[0]["line_end"] == len(lines)  # H1 spans to EOF
        # Section A ends before Section B (same level)
        assert sections[1]["line_end"] == sections[2]["line_start"] - 1
        # Section B spans to EOF
        assert sections[2]["line_end"] == len(lines)

    def test_empty_document(self):
        sections = self._parse_and_extract("")
        assert sections == []

    def test_no_headings(self):
        sections = self._parse_and_extract("Just some plain text.\nNo headings here.\n")
        assert sections == []

    def test_extractor_class_extract_elements_returns_empty(self):
        """MarkdownExtractor.extract_elements returns empty — sections go via extract_sections."""
        manager = get_manager()
        tree = manager.parse(b"# Title\n\n## Section\n", "markdown")
        extractor = MarkdownExtractor()
        elements = extractor.extract_elements(tree, ["# Title", "", "## Section", ""])
        assert elements == []

    def test_extractor_class_extract_sections(self):
        manager = get_manager()
        content = "# Title\n\n## Section\n"
        tree = manager.parse(content.encode("utf-8"), "markdown")
        extractor = MarkdownExtractor()
        sections = extractor.extract_sections(tree, content.split("\n"))
        assert len(sections) == 2
        assert sections[0]["title"] == "Title"
        assert sections[1]["title"] == "Section"
