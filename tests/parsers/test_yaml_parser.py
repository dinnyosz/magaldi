"""Tests for the YAML parser."""

from pathlib import Path

from magaldi_core.change_detection import FileInfo
from magaldi_core.parsers.yaml_lang import YamlParser


class TestYamlParser:
    """Tests for YAML file parsing."""

    def _parse(self, content: str, filename: str = "config.yaml") -> list:
        parser = YamlParser()
        file_info = FileInfo(
            relative_path=filename,
            absolute_path=Path(f"/fake/{filename}"),
            language="yaml",
        )
        return parser.parse(content, file_info, "scope", "repo", "main")

    def test_creates_file_element(self):
        content = "name: test\nservices:\n  web:\n    build: .\n"
        elements = self._parse(content)
        file_elems = [e for e in elements if e.element_type == "file"]
        assert len(file_elems) == 1
        assert file_elems[0].name == "config.yaml"
        assert file_elems[0].language == "yaml"
        assert file_elems[0].level == 0

    def test_creates_constant_sub_elements(self):
        content = "services:\n  web:\n    build: .\n"
        elements = self._parse(content)
        constants = [e for e in elements if e.element_type == "constant"]
        names = [c.name for c in constants]
        assert "services" in names
        assert "services.web" in names

    def test_constants_have_parent_id(self):
        content = "services:\n  web:\n    build: .\n"
        elements = self._parse(content)
        file_elem = next(e for e in elements if e.element_type == "file")
        constants = [e for e in elements if e.element_type == "constant"]
        for const in constants:
            assert const.parent_id == file_elem.element_id
            assert const.level == 1

    def test_leaf_keys_not_extracted(self):
        content = "name: test\nversion: 1.0\n"
        elements = self._parse(content)
        assert len(elements) == 1  # Only file element

    def test_element_ids_are_unique(self):
        content = "services:\n  web:\n    build: .\n  db:\n    image: postgres\n"
        elements = self._parse(content)
        ids = [e.element_id for e in elements]
        assert len(ids) == len(set(ids))

    def test_empty_yaml(self):
        elements = self._parse("")
        assert len(elements) == 1
        assert elements[0].element_type == "file"

    def test_deep_nesting(self):
        content = "a:\n  b:\n    c:\n      d: value\n"
        elements = self._parse(content)
        names = [e.name for e in elements if e.element_type == "constant"]
        assert "a" in names
        assert "a.b" in names
        assert "a.b.c" in names
        assert "a.b.c.d" not in names  # Leaf
