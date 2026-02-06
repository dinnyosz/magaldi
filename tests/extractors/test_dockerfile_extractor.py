"""Tests for the Dockerfile extractor."""

from magaldi_core.extractors.dockerfile import DockerfileExtractor
from magaldi_core.tree_sitter_manager import get_manager


class TestDockerfileExtractor:
    """Tests for Dockerfile instruction extraction."""

    def _parse_and_extract(self, content: str) -> list:
        manager = get_manager()
        tree = manager.parse(content.encode("utf-8"), "dockerfile")
        extractor = DockerfileExtractor()
        return extractor.extract_elements(tree, content.split("\n"))

    def test_extracts_from_instruction(self):
        content = "FROM python:3.12-slim\n"
        elements = self._parse_and_extract(content)
        assert len(elements) >= 1
        from_elem = elements[0]
        assert from_elem.element_type == "constant"
        assert "FROM" in from_elem.name

    def test_extracts_multiple_instructions(self):
        content = (
            "FROM python:3.12\n"
            "WORKDIR /app\n"
            "COPY . .\n"
            "RUN pip install -r requirements.txt\n"
            "EXPOSE 8080\n"
            "CMD [\"python\", \"app.py\"]\n"
        )
        elements = self._parse_and_extract(content)
        assert len(elements) >= 5
        names = [e.name for e in elements]
        assert any("FROM" in n for n in names)
        assert any("WORKDIR" in n for n in names)
        assert any("RUN" in n for n in names)
        assert any("EXPOSE" in n for n in names)

    def test_element_type_is_constant(self):
        content = "FROM ubuntu:22.04\nRUN apt-get update\n"
        elements = self._parse_and_extract(content)
        for elem in elements:
            assert elem.element_type == "constant"

    def test_line_numbers(self):
        content = "FROM python:3.12\nWORKDIR /app\nCOPY . .\n"
        elements = self._parse_and_extract(content)
        from_elem = next(e for e in elements if "FROM" in e.name)
        assert from_elem.line_start == 1

    def test_env_instruction(self):
        content = "FROM ubuntu\nENV APP_ENV=production\n"
        elements = self._parse_and_extract(content)
        names = [e.name for e in elements]
        assert any("ENV" in n for n in names)

    def test_empty_dockerfile(self):
        elements = self._parse_and_extract("")
        assert elements == []

    def test_multiline_run(self):
        content = "FROM ubuntu\nRUN apt-get update && \\\n    apt-get install -y curl\n"
        elements = self._parse_and_extract(content)
        run_elem = next(e for e in elements if "RUN" in e.name)
        assert run_elem.line_start == 2

    def test_multi_stage_build(self):
        content = (
            "FROM node:18 AS builder\n"
            "WORKDIR /build\n"
            "COPY . .\n"
            "RUN npm run build\n"
            "\n"
            "FROM nginx:alpine\n"
            "COPY --from=builder /build/dist /usr/share/nginx/html\n"
        )
        elements = self._parse_and_extract(content)
        from_elems = [e for e in elements if "FROM" in e.name]
        assert len(from_elems) == 2
