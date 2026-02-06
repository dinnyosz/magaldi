"""Tests for the markdown parser."""

from pathlib import Path

from magaldi_core.change_detection import FileInfo
from magaldi_core.parsers.markdown import MarkdownParser


class TestMarkdownParser:
    """Tests for markdown file parsing."""

    def _parse(self, content: str, filename: str = "README.md") -> list:
        parser = MarkdownParser()
        file_info = FileInfo(
            relative_path=filename,
            absolute_path=Path(f"/fake/{filename}"),
            language="markdown",
        )
        return parser.parse(content, file_info, "scope", "repo", "main")

    def test_creates_file_element(self):
        elements = self._parse("# Title\n\nSome text\n")
        assert len(elements) == 1
        assert elements[0].element_type == "file"
        assert elements[0].name == "README.md"
        assert elements[0].language == "markdown"
        assert elements[0].level == 0

    def test_file_element_has_document_sections(self):
        content = "# Title\n\n## Section A\n\nContent\n\n## Section B\n\nMore\n"
        elements = self._parse(content)
        file_elem = elements[0]
        assert len(file_elem.document_sections) == 3
        assert file_elem.document_sections[0]["title"] == "Title"
        assert file_elem.document_sections[0]["level"] == 1
        assert file_elem.document_sections[1]["title"] == "Section A"
        assert file_elem.document_sections[1]["level"] == 2
        assert file_elem.document_sections[2]["title"] == "Section B"

    def test_no_sub_elements(self):
        """Markdown headings go into document_sections, not as sub-elements."""
        content = "# Title\n\n## Section\n\n### Subsection\n"
        elements = self._parse(content)
        assert len(elements) == 1  # Only file element

    def test_empty_markdown(self):
        elements = self._parse("")
        assert len(elements) == 1
        assert elements[0].element_type == "file"
        assert elements[0].document_sections == []

    def test_file_element_has_raw_code(self):
        content = "# Title\n\nSome text\n"
        elements = self._parse(content)
        assert elements[0].raw_code == content

    def test_element_id_is_set(self):
        elements = self._parse("# Title\n")
        assert elements[0].element_id is not None
        assert len(elements[0].element_id) > 0

    def test_custom_filename(self):
        elements = self._parse("# Changelog\n", filename="CHANGELOG.md")
        assert elements[0].name == "CHANGELOG.md"
