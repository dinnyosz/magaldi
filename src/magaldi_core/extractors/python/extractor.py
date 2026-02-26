"""Python extractor class.

This module provides the main PythonExtractor class that implements
the BaseExtractor interface for Python code.
"""

from __future__ import annotations

from tree_sitter import Node, Tree

from magaldi_core.extractors.base import BaseExtractor
from magaldi_core.extractors.python.call_extractor import (
    extract_python_calls,
)
from magaldi_core.extractors.python.context_extractor import (
    extract_python_base_classes,
    extract_python_class_attributes,
    extract_python_modified_attributes,
    extract_python_raised_exceptions,
)
from magaldi_core.extractors.python.element_extractor import (
    extract_python_class_members,
    extract_python_elements,
)
from magaldi_core.extractors.python.import_extractor import (
    extract_python_imports,
)
from magaldi_core.extractors.python.reference_extractor import (
    extract_python_references,
)
from magaldi_core.extractors.types import (
    ExtractedCall,
    ExtractedElement,
    ExtractedImport,
    ExtractedReference,
)


class PythonExtractor(BaseExtractor):
    """Extractor for Python source code."""

    @property
    def language(self) -> str:
        return "python"

    def extract_elements(self, tree: Tree, lines: list[str]) -> list[ExtractedElement]:
        """Extract code elements from a Python AST."""
        return extract_python_elements(tree, lines)  # type: ignore[no-any-return]

    def extract_class_members(
        self, class_node: Node, lines: list[str]
    ) -> tuple[list[ExtractedElement], list[ExtractedElement]]:
        """Extract methods and class variables from a Python class."""
        return extract_python_class_members(class_node, lines)  # type: ignore[no-any-return]

    def extract_imports(self, tree: Tree, lines: list[str]) -> list[ExtractedImport]:
        """Extract import statements from Python AST."""
        return extract_python_imports(tree, lines)  # type: ignore[no-any-return]

    def extract_references(
        self, tree: Tree, lines: list[str]
    ) -> list[ExtractedReference]:
        """Extract all references from Python AST."""
        return extract_python_references(tree, lines)  # type: ignore[no-any-return]

    def extract_calls(self, function_node: Node) -> list[ExtractedCall]:
        """Extract function/method calls from within a Python function body."""
        return extract_python_calls(function_node)  # type: ignore[no-any-return]

    def extract_class_attributes(
        self, class_node: Node
    ) -> list[dict[str, str | int]]:
        """Extract instance attributes from __init__."""
        return extract_python_class_attributes(class_node)  # type: ignore[no-any-return]

    def extract_base_classes(self, class_node: Node) -> list[str]:
        """Extract base class names from Python class definition."""
        return extract_python_base_classes(class_node)  # type: ignore[no-any-return]

    def extract_raised_exceptions(self, function_node: Node) -> list[str]:
        """Extract exception types from raise statements."""
        return extract_python_raised_exceptions(function_node)  # type: ignore[no-any-return]

    def extract_modified_attributes(self, method_node: Node) -> list[str]:
        """Extract self.X attributes that are modified."""
        return extract_python_modified_attributes(method_node)  # type: ignore[no-any-return]
