"""Tests for the plain text parser."""

from pathlib import Path

from magaldi_core.change_detection import FileInfo
from magaldi_core.parsers.plaintext import PlainTextParser


class TestPlainTextParser:
    """Tests for plain text file parsing."""

    def _parse(self, content: str, filename: str = "notes.txt") -> list:
        parser = PlainTextParser()
        file_info = FileInfo(
            relative_path=filename,
            absolute_path=Path(f"/fake/{filename}"),
            language="text",
        )
        return parser.parse(content, file_info, "scope", "repo", "main")

    def test_creates_file_element_only(self):
        elements = self._parse("Just some plain text.\nLine two.\n")
        assert len(elements) == 1
        assert elements[0].element_type == "file"
        assert elements[0].name == "notes.txt"
        assert elements[0].language == "text"
        assert elements[0].level == 0

    def test_file_element_has_raw_code(self):
        content = "Line one.\nLine two.\nLine three.\n"
        elements = self._parse(content)
        assert elements[0].raw_code == content

    def test_line_count(self):
        content = "Line 1\nLine 2\nLine 3"
        elements = self._parse(content)
        assert elements[0].line_start == 1
        assert elements[0].line_end == 3

    def test_empty_file(self):
        elements = self._parse("")
        assert len(elements) == 1
        assert elements[0].element_type == "file"

    def test_element_id_is_set(self):
        elements = self._parse("content")
        assert elements[0].element_id is not None
        assert len(elements[0].element_id) > 0

    def test_nfo_file(self):
        elements = self._parse("Some info content", filename="info.nfo")
        assert elements[0].name == "info.nfo"

    def test_no_sub_elements(self):
        """Plain text should never produce sub-elements."""
        content = "# This looks like markdown\ndef this_looks_like_python():\n    pass\n"
        elements = self._parse(content)
        assert len(elements) == 1
