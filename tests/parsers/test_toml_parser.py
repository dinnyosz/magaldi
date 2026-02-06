"""Tests for the TOML parser."""

from pathlib import Path

from magaldi_core.change_detection import FileInfo
from magaldi_core.parsers.toml_lang import TomlParser


class TestTomlParser:
    """Tests for TOML file parsing."""

    def _parse(self, content: str, filename: str = "pyproject.toml") -> list:
        parser = TomlParser()
        file_info = FileInfo(
            relative_path=filename,
            absolute_path=Path(f"/fake/{filename}"),
            language="toml",
        )
        return parser.parse(content, file_info, "scope", "repo", "main")

    def test_creates_file_element(self):
        content = "[project]\nname = \"test\"\n"
        elements = self._parse(content)
        file_elems = [e for e in elements if e.element_type == "file"]
        assert len(file_elems) == 1
        assert file_elems[0].name == "pyproject.toml"
        assert file_elems[0].language == "toml"
        assert file_elems[0].level == 0

    def test_creates_constant_sub_elements(self):
        content = "[project]\nname = \"test\"\n\n[dependencies]\nfoo = \"1.0\"\n"
        elements = self._parse(content)
        constants = [e for e in elements if e.element_type == "constant"]
        names = [c.name for c in constants]
        assert "project" in names
        assert "dependencies" in names

    def test_constants_have_parent_id(self):
        content = "[project]\nname = \"test\"\n"
        elements = self._parse(content)
        file_elem = next(e for e in elements if e.element_type == "file")
        constants = [e for e in elements if e.element_type == "constant"]
        for const in constants:
            assert const.parent_id == file_elem.element_id
            assert const.level == 1

    def test_dotted_table_names(self):
        content = "[tool.ruff]\nline-length = 120\n\n[tool.ruff.lint]\nselect = [\"E\"]\n"
        elements = self._parse(content)
        names = [e.name for e in elements if e.element_type == "constant"]
        assert "tool.ruff" in names
        assert "tool.ruff.lint" in names

    def test_element_ids_are_unique(self):
        content = "[project]\nname = \"test\"\n\n[dependencies]\nfoo = \"1.0\"\n"
        elements = self._parse(content)
        ids = [e.element_id for e in elements]
        assert len(ids) == len(set(ids))

    def test_empty_toml(self):
        elements = self._parse("")
        assert len(elements) == 1
        assert elements[0].element_type == "file"
