"""Tests for the YAML extractor."""

from magaldi_core.extractors.yaml_lang import YamlExtractor
from magaldi_core.tree_sitter_manager import get_manager


class TestYamlExtractor:
    """Tests for YAML structural key extraction."""

    def _parse_and_extract(self, content: str) -> list:
        manager = get_manager()
        tree = manager.parse(content.encode("utf-8"), "yaml")
        extractor = YamlExtractor()
        return extractor.extract_elements(tree, content.split("\n"))

    def test_extracts_top_level_structural_keys(self):
        content = "services:\n  web:\n    build: .\n  db:\n    image: postgres\nname: test\n"
        elements = self._parse_and_extract(content)
        names = [e.name for e in elements]
        assert "services" in names
        assert "services.web" in names
        assert "services.db" in names
        # Leaf keys should NOT be extracted
        assert "name" not in names
        assert "services.web.build" not in names

    def test_deep_nesting(self):
        content = "magaldi:\n  config:\n    test:\n      whatever: value\n"
        elements = self._parse_and_extract(content)
        names = [e.name for e in elements]
        assert "magaldi" in names
        assert "magaldi.config" in names
        assert "magaldi.config.test" in names
        # 'whatever' is a leaf — not extracted
        assert "magaldi.config.test.whatever" not in names

    def test_leaf_only_document(self):
        """Document with only leaf scalar pairs — no structural elements."""
        content = "name: test\nversion: 1.0\nauthor: someone\n"
        elements = self._parse_and_extract(content)
        assert elements == []

    def test_mixed_structural_and_leaf(self):
        content = "name: myproject\nservices:\n  web:\n    port: 8080\nversion: 1.0\n"
        elements = self._parse_and_extract(content)
        names = [e.name for e in elements]
        assert "services" in names
        assert "services.web" in names
        assert "name" not in names
        assert "version" not in names
        assert "services.web.port" not in names

    def test_element_type_is_constant(self):
        content = "services:\n  web:\n    build: .\n"
        elements = self._parse_and_extract(content)
        for elem in elements:
            assert elem.element_type == "constant"

    def test_line_numbers(self):
        content = "services:\n  web:\n    build: .\n  db:\n    image: postgres\n"
        elements = self._parse_and_extract(content)
        services = next(e for e in elements if e.name == "services")
        assert services.line_start == 1
        # tree-sitter end_point includes trailing newline
        assert services.line_end >= 5

        web = next(e for e in elements if e.name == "services.web")
        assert web.line_start == 2

    def test_raw_code_includes_children(self):
        content = "services:\n  web:\n    build: .\n    ports:\n      - 8080\n"
        elements = self._parse_and_extract(content)
        services = next(e for e in elements if e.name == "services")
        assert "build: ." in services.raw_code
        assert "ports:" in services.raw_code

    def test_empty_document(self):
        elements = self._parse_and_extract("")
        assert elements == []

    def test_docker_compose_style(self):
        content = (
            "version: '3'\n"
            "services:\n"
            "  web:\n"
            "    build: .\n"
            "    ports:\n"
            "      - '8080:80'\n"
            "  redis:\n"
            "    image: redis:alpine\n"
            "volumes:\n"
            "  data:\n"
            "    driver: local\n"
        )
        elements = self._parse_and_extract(content)
        names = [e.name for e in elements]
        assert "services" in names
        assert "services.web" in names
        assert "services.redis" in names
        assert "volumes" in names
        assert "volumes.data" in names
