"""Tests for the Dockerfile parser."""

from pathlib import Path

from magaldi_core.change_detection import FileInfo
from magaldi_core.parsers.dockerfile import DockerfileParser


class TestDockerfileParser:
    """Tests for Dockerfile parsing."""

    def _parse(self, content: str, filename: str = "Dockerfile") -> list:
        parser = DockerfileParser()
        file_info = FileInfo(
            relative_path=filename,
            absolute_path=Path(f"/fake/{filename}"),
            language="dockerfile",
        )
        return parser.parse(content, file_info, "scope", "repo", "main")

    def test_creates_file_element(self):
        content = "FROM python:3.12\nRUN pip install foo\n"
        elements = self._parse(content)
        file_elems = [e for e in elements if e.element_type == "file"]
        assert len(file_elems) == 1
        assert file_elems[0].name == "Dockerfile"
        assert file_elems[0].language == "dockerfile"
        assert file_elems[0].level == 0

    def test_creates_constant_sub_elements(self):
        content = "FROM python:3.12\nWORKDIR /app\nCOPY . .\nRUN pip install -r requirements.txt\n"
        elements = self._parse(content)
        constants = [e for e in elements if e.element_type == "constant"]
        assert len(constants) >= 3
        names = [c.name for c in constants]
        assert any("FROM" in n for n in names)
        assert any("WORKDIR" in n for n in names)
        assert any("RUN" in n for n in names)

    def test_constants_have_parent_id(self):
        content = "FROM ubuntu\nRUN apt-get update\n"
        elements = self._parse(content)
        file_elem = next(e for e in elements if e.element_type == "file")
        constants = [e for e in elements if e.element_type == "constant"]
        for const in constants:
            assert const.parent_id == file_elem.element_id
            assert const.level == 1

    def test_element_ids_are_unique(self):
        content = "FROM python:3.12\nRUN pip install foo\nEXPOSE 8080\n"
        elements = self._parse(content)
        ids = [e.element_id for e in elements]
        assert len(ids) == len(set(ids))

    def test_empty_dockerfile(self):
        elements = self._parse("")
        assert len(elements) == 1
        assert elements[0].element_type == "file"

    def test_custom_filename(self):
        elements = self._parse("FROM nginx\n", filename="Dockerfile.prod")
        file_elem = next(e for e in elements if e.element_type == "file")
        assert file_elem.name == "Dockerfile.prod"
