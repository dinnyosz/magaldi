"""Dockerfile parser using tree-sitter.

Extracts instructions as constant elements from Dockerfiles.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from magaldi_core.extractors.dockerfile import DockerfileExtractor
from magaldi_core.parsers.base import CodeElement, TreeSitterParser, generate_element_id

if TYPE_CHECKING:
    from magaldi_core.change_detection import FileInfo


class DockerfileParser(TreeSitterParser):
    """Parse Dockerfiles using tree-sitter."""

    def __init__(self) -> None:
        super().__init__("dockerfile")
        self._extractor = DockerfileExtractor()

    def parse(
        self,
        content: str,
        file_info: "FileInfo",
        scope: str,
        repository: str,
        username: str,
    ) -> list[CodeElement]:
        """Parse Dockerfile content and extract instructions."""
        elements: list[CodeElement] = []
        lines = content.split("\n")

        # Create file-level element
        file_element = self._create_file_element(
            content, file_info, scope, repository, username, "dockerfile"
        )
        elements.append(file_element)

        # Parse and extract
        tree = self.manager.parse(content.encode("utf-8"), "dockerfile")
        extracted = self._extractor.extract_elements(tree, lines)

        # Convert to CodeElements
        for ext in extracted:
            elem = CodeElement(
                scope=scope,
                repository=repository,
                username=username,
                relative_path=file_info.relative_path,
                element_type=ext.element_type,
                name=ext.name,
                language="dockerfile",
                line_start=ext.line_start,
                line_end=ext.line_end,
                raw_code=ext.raw_code,
                level=1,
                parent_id=file_element.element_id,
            )
            elem.element_id = generate_element_id(
                scope, repository, username, file_info.relative_path,
                ext.element_type, ext.name, ext.line_start,
            )
            elements.append(elem)

        return elements
