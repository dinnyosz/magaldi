"""JavaScript/TypeScript extractor class.

This module provides the main JavaScriptExtractor class that implements
the BaseExtractor interface for JavaScript and TypeScript code.
"""

from __future__ import annotations

from tree_sitter import Node, Tree

from magaldi_core.extractors.base import BaseExtractor
from magaldi_core.extractors.types import (
    ExtractedCall,
    ExtractedElement,
    ExtractedImport,
    ExtractedReference,
)
from magaldi_core.extractors.javascript.call_extractor import (
    extract_javascript_calls,
)
from magaldi_core.extractors.javascript.context_extractor import (
    extract_javascript_base_class,
    extract_javascript_class_fields,
    extract_javascript_modified_properties,
    extract_javascript_thrown_exceptions,
)
from magaldi_core.extractors.javascript.element_extractor import (
    extract_javascript_class_members,
    extract_javascript_elements,
)
from magaldi_core.extractors.javascript.import_extractor import (
    extract_javascript_imports,
)
from magaldi_core.extractors.javascript.reference_extractor import (
    extract_javascript_references,
)


class JavaScriptExtractor(BaseExtractor):
    """Extractor for JavaScript/TypeScript source code."""

    def __init__(self, language: str = "javascript"):
        self._language = language

    @property
    def language(self) -> str:
        return self._language

    def extract_elements(self, tree: Tree, lines: list[str]) -> list[ExtractedElement]:
        """Extract code elements from a JavaScript/TypeScript AST."""
        return extract_javascript_elements(tree, lines)

    def extract_class_members(
        self, class_node: Node, lines: list[str]
    ) -> tuple[list[ExtractedElement], list[ExtractedElement]]:
        """Extract methods and class fields from a JavaScript class."""
        return extract_javascript_class_members(class_node, lines)

    def extract_imports(self, tree: Tree, lines: list[str]) -> list[ExtractedImport]:
        """Extract import statements from JavaScript/TypeScript AST."""
        return extract_javascript_imports(tree, lines)

    def extract_references(
        self, tree: Tree, lines: list[str]
    ) -> list[ExtractedReference]:
        """Extract all references from JavaScript/TypeScript AST."""
        return extract_javascript_references(tree, lines)

    def extract_calls(self, function_node: Node) -> list[ExtractedCall]:
        """Extract function/method calls from within a JavaScript function body."""
        return extract_javascript_calls(function_node)

    def extract_class_attributes(
        self, class_node: Node
    ) -> list[dict[str, str | int]]:
        """Extract class fields from JavaScript/TypeScript class."""
        return extract_javascript_class_fields(class_node)

    def extract_base_classes(self, class_node: Node) -> list[str]:
        """Extract base class name from JavaScript class definition."""
        return extract_javascript_base_class(class_node)

    def extract_raised_exceptions(self, function_node: Node) -> list[str]:
        """Extract exception types from throw statements."""
        return extract_javascript_thrown_exceptions(function_node)

    def extract_modified_attributes(self, method_node: Node) -> list[str]:
        """Extract this.X properties that are modified."""
        return extract_javascript_modified_properties(method_node)
