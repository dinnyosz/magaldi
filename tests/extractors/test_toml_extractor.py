"""Tests for the TOML extractor."""

from magaldi_core.extractors.toml_lang import TomlExtractor
from magaldi_core.tree_sitter_manager import get_manager


class TestTomlExtractor:
    """Tests for TOML table/key extraction."""

    def _parse_and_extract(self, content: str) -> list:
        manager = get_manager()
        tree = manager.parse(content.encode("utf-8"), "toml")
        extractor = TomlExtractor()
        return extractor.extract_elements(tree, content.split("\n"))

    def test_extracts_simple_table(self):
        content = "[project]\nname = \"magaldi\"\nversion = \"1.0\"\n"
        elements = self._parse_and_extract(content)
        names = [e.name for e in elements]
        assert "project" in names

    def test_extracts_dotted_table(self):
        content = "[tool.ruff]\nline-length = 120\n\n[tool.ruff.lint]\nselect = [\"E\"]\n"
        elements = self._parse_and_extract(content)
        names = [e.name for e in elements]
        assert "tool.ruff" in names
        assert "tool.ruff.lint" in names

    def test_element_type_is_constant(self):
        content = "[project]\nname = \"test\"\n"
        elements = self._parse_and_extract(content)
        for elem in elements:
            assert elem.element_type == "constant"

    def test_line_numbers(self):
        content = "[project]\nname = \"test\"\nversion = \"1.0\"\n\n[dependencies]\nfoo = \"1.0\"\n"
        elements = self._parse_and_extract(content)
        project = next(e for e in elements if e.name == "project")
        assert project.line_start == 1

        deps = next(e for e in elements if e.name == "dependencies")
        assert deps.line_start == 5

    def test_raw_code_contains_table_content(self):
        content = "[project]\nname = \"magaldi\"\nversion = \"1.0\"\n"
        elements = self._parse_and_extract(content)
        project = next(e for e in elements if e.name == "project")
        assert "name" in project.raw_code
        assert "version" in project.raw_code

    def test_empty_document(self):
        elements = self._parse_and_extract("")
        assert elements == []

    def test_pyproject_toml_style(self):
        content = (
            "[build-system]\n"
            "requires = [\"hatchling\"]\n"
            "build-backend = \"hatchling.build\"\n"
            "\n"
            "[project]\n"
            "name = \"myproject\"\n"
            "version = \"0.1.0\"\n"
            "\n"
            "[tool.ruff]\n"
            "line-length = 120\n"
        )
        elements = self._parse_and_extract(content)
        names = [e.name for e in elements]
        assert "build-system" in names
        assert "project" in names
        assert "tool.ruff" in names
