"""Base extractor class and common utilities for language extractors.

This module provides:
- BaseExtractor abstract class defining the common extraction interface
- Shared utility functions used across all language extractors
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import TYPE_CHECKING

from tree_sitter import Node, Tree

if TYPE_CHECKING:
    from magaldi_core.extractors.types import (
        ExtractedCall,
        ExtractedElement,
        ExtractedImport,
        ExtractedReference,
    )


# =============================================================================
# TREE UTILITIES
# =============================================================================


def walk_tree(node: Node) -> Iterator[Node]:
    """Walk all nodes in the tree depth-first."""

    yield node
    for child in node.children:
        yield from walk_tree(child)


def find_nodes(root: Node, node_type: str) -> Iterator[Node]:
    """Find all nodes of a specific type."""

    for node in walk_tree(root):
        if node.type == node_type:
            yield node


def get_node_text(node: Node) -> str:
    """Get the text content of a node."""
    return node.text.decode("utf-8") if node.text else ""


def get_child_by_field(node: Node, field_name: str) -> Node | None:
    """Get a child node by its field name."""
    return node.child_by_field_name(field_name)


def get_children_by_type(node: Node, node_type: str) -> list[Node]:
    """Get all direct children of a specific type."""
    return [child for child in node.children if child.type == node_type]


# =============================================================================
# BASE EXTRACTOR
# =============================================================================


class BaseExtractor(ABC):
    """Abstract base class for language-specific extractors.

    Each language extractor implements methods to extract various code
    elements from the tree-sitter AST.
    """

    @property
    @abstractmethod
    def language(self) -> str:
        """Return the language name this extractor handles."""
        ...

    @abstractmethod
    def extract_elements(self, tree: Tree, lines: list[str]) -> list[ExtractedElement]:
        """Extract top-level code elements (classes, functions, etc.) from an AST.

        Args:
            tree: Parsed tree-sitter Tree.
            lines: Source code lines for raw code extraction.

        Returns:
            List of extracted elements.
        """
        ...

    @abstractmethod
    def extract_class_members(
        self, class_node: Node, lines: list[str]
    ) -> tuple[list[ExtractedElement], list[ExtractedElement]]:
        """Extract methods and class variables from a class.

        Args:
            class_node: The class definition node.
            lines: Source code lines.

        Returns:
            Tuple of (methods, class_variables).
        """
        ...

    @abstractmethod
    def extract_imports(self, tree: Tree, lines: list[str]) -> list[ExtractedImport]:
        """Extract import statements from an AST.

        Args:
            tree: Parsed tree-sitter Tree.
            lines: Source code lines.

        Returns:
            List of extracted imports.
        """
        ...

    @abstractmethod
    def extract_references(
        self, tree: Tree, lines: list[str]
    ) -> list[ExtractedReference]:
        """Extract all references (calls, type hints) from an AST.

        Args:
            tree: Parsed tree-sitter Tree.
            lines: Source code lines.

        Returns:
            List of extracted references.
        """
        ...

    @abstractmethod
    def extract_calls(self, function_node: Node) -> list[ExtractedCall]:
        """Extract function/method calls from within a function body.

        Args:
            function_node: A function definition node.

        Returns:
            List of extracted calls.
        """
        ...

    def extract_class_attributes(
        self, class_node: Node  # noqa: ARG002
    ) -> list[dict[str, str | int]]:
        """Extract instance attributes defined in the class constructor.

        Default implementation returns empty list. Override in subclasses.

        Args:
            class_node: A class definition node.

        Returns:
            List of dicts with "name" and "line" keys.
        """
        return []

    def extract_base_classes(self, class_node: Node) -> list[str]:  # noqa: ARG002
        """Extract base class names from class definition.

        Default implementation returns empty list. Override in subclasses.

        Args:
            class_node: A class definition node.

        Returns:
            List of base class names.
        """
        return []

    def extract_raised_exceptions(self, function_node: Node) -> list[str]:  # noqa: ARG002
        """Extract exception types from raise/throw statements in a function.

        Default implementation returns empty list. Override in subclasses.

        Args:
            function_node: A function definition node.

        Returns:
            List of exception type names.
        """
        return []

    def extract_modified_attributes(self, method_node: Node) -> list[str]:  # noqa: ARG002
        """Extract self/this attributes that are assigned to in a method.

        Default implementation returns empty list. Override in subclasses.

        Args:
            method_node: A method definition node.

        Returns:
            List of attribute names that are modified.
        """
        return []
