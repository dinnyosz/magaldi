"""Plain text parser for .txt and .nfo files.

Creates a file-level element only — no tree-sitter grammar available.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from magaldi_core.parsers.base import CodeElement, generate_element_id

if TYPE_CHECKING:
    from magaldi_core.change_detection import FileInfo


class PlainTextParser:
    """Parse plain text files without tree-sitter.

    Creates only a file-level CodeElement with the raw content.
    Has the same parse() interface as TreeSitterParser but does not
    extend it since there's no tree-sitter grammar for plain text.
    """

    def parse(
        self,
        content: str,
        file_info: "FileInfo",
        scope: str,
        repository: str,
        username: str,
    ) -> list[CodeElement]:
        """Parse plain text — creates a single file-level element."""
        lines = content.split("\n")
        line_count = len(lines)

        file_element = CodeElement(
            scope=scope,
            repository=repository,
            username=username,
            relative_path=file_info.relative_path,
            element_type="file",
            name=Path(file_info.relative_path).name,
            language=file_info.language,
            line_start=1,
            line_end=line_count,
            level=0,
            raw_code=content,
        )
        file_element.element_id = generate_element_id(
            scope, repository, username, file_info.relative_path,
            "file", file_element.name, 1,
        )

        return [file_element]
