"""Dockerfile extractor - extracts instructions from Dockerfiles."""

from __future__ import annotations

from tree_sitter import Tree

from magaldi_core.extractors.base import get_node_text
from magaldi_core.extractors.types import ExtractedElement

# Instruction node types in tree-sitter-dockerfile
INSTRUCTION_TYPES = {
    "from_instruction",
    "run_instruction",
    "copy_instruction",
    "add_instruction",
    "cmd_instruction",
    "entrypoint_instruction",
    "env_instruction",
    "expose_instruction",
    "volume_instruction",
    "workdir_instruction",
    "arg_instruction",
    "label_instruction",
    "user_instruction",
    "healthcheck_instruction",
    "stopsignal_instruction",
    "shell_instruction",
    "maintainer_instruction",
}


class DockerfileExtractor:
    """Extractor for Dockerfile instructions."""

    @property
    def language(self) -> str:
        return "dockerfile"

    def extract_elements(self, tree: Tree, lines: list[str]) -> list[ExtractedElement]:
        """Extract Dockerfile instructions as constant elements."""
        elements: list[ExtractedElement] = []
        root = tree.root_node

        for child in root.children:
            if child.type in INSTRUCTION_TYPES:
                name = _instruction_name(child)
                line_start = child.start_point[0] + 1
                line_end = child.end_point[0] + 1
                raw_code = "\n".join(lines[line_start - 1:line_end])

                elements.append(ExtractedElement(
                    element_type="constant",
                    name=name,
                    line_start=line_start,
                    line_end=line_end,
                    raw_code=raw_code,
                    node=child,
                ))

        return elements


def _instruction_name(node: object) -> str:
    """Build a descriptive name for a Dockerfile instruction.

    Examples:
        FROM python:3.12-slim AS builder
        RUN apt-get update
        EXPOSE 8080
        ENV APP_ENV=production
    """
    text = get_node_text(node).strip()
    # Take the first line only (multi-line RUN can be very long)
    first_line = text.split("\n")[0]
    # Trim to reasonable length
    if len(first_line) > 80:
        return first_line[:77] + "..."  # type: ignore[no-any-return]
    return first_line  # type: ignore[no-any-return]
