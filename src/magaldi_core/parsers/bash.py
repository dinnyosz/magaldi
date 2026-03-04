"""Bash parser using tree-sitter.

Extracts functions, constants/variables, calls, and source imports from Bash scripts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from magaldi_core.extractors.bash import (
    BashExtractor,
    extract_bash_calls,
    extract_top_level_bash_calls,
)
from magaldi_core.parsers.base import (
    Call,
    CodeElement,
    Import,
    TreeSitterParser,
    extract_preceding_doc_comment,
    generate_element_id,
)

if TYPE_CHECKING:
    from magaldi_core.change_detection import FileInfo


class BashParser(TreeSitterParser):
    """Parse Bash scripts using tree-sitter."""

    def __init__(self) -> None:
        super().__init__("bash")
        self._extractor = BashExtractor()

    def parse(
        self,
        content: str,
        file_info: FileInfo,
        scope: str,
        repository: str,
        username: str,
    ) -> list[CodeElement]:
        """Parse Bash content and extract elements."""
        elements: list[CodeElement] = []
        lines = content.split("\n")

        # Create file-level element
        file_element = self._create_file_element(
            content, file_info, scope, repository, username, "bash"
        )
        elements.append(file_element)

        # Parse and extract
        tree = self.manager.parse(content.encode("utf-8"), "bash")
        extracted = self._extractor.extract_elements(tree, lines)

        # Extract imports and attach to file element
        extracted_imports = self._extractor.extract_imports(tree, lines)
        file_element.imports = [
            Import(name=imp.name, module=imp.module, alias=imp.alias, line=imp.line)
            for imp in extracted_imports
        ]

        # Extract top-level calls and populate on file element
        top_level_calls = extract_top_level_bash_calls(tree)
        file_element.calls = [
            Call(name=c.name, receiver=c.receiver, line=c.line)
            for c in top_level_calls
        ]

        # Convert to CodeElements
        for ext in extracted:
            if ext.element_type == "function":
                # Extract calls within the function
                calls: list[Call] = []
                if ext.node:
                    extracted_calls = extract_bash_calls(ext.node)
                    calls = [Call(name=c.name, receiver=c.receiver, line=c.line) for c in extracted_calls]

                elem = CodeElement(
                    scope=scope,
                    repository=repository,
                    username=username,
                    relative_path=file_info.relative_path,
                    element_type="function",
                    name=ext.name,
                    language="bash",
                    line_start=ext.line_start,
                    line_end=ext.line_end,
                    raw_code=ext.raw_code,
                    docstring=extract_preceding_doc_comment(lines, ext.line_start, "bash"),
                    level=2,
                    parent_id=file_element.element_id,
                    calls=calls,
                )
            else:
                # constant or variable
                elem = CodeElement(
                    scope=scope,
                    repository=repository,
                    username=username,
                    relative_path=file_info.relative_path,
                    element_type=ext.element_type,
                    name=ext.name,
                    language="bash",
                    line_start=ext.line_start,
                    line_end=ext.line_end,
                    raw_code=ext.raw_code,
                    docstring=extract_preceding_doc_comment(lines, ext.line_start, "bash"),
                    level=3,
                    parent_id=file_element.element_id,
                )

            elem.element_id = generate_element_id(
                scope, repository, username, file_info.relative_path,
                ext.element_type, ext.name, ext.get_byte_offset(),
            )
            elements.append(elem)

        # Resolve same-file function calls
        self._resolve_calls_in_file(elements)

        return elements
