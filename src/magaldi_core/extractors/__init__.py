"""Code element extractors for multiple programming languages.

This package provides language-specific extractors for parsing source code
using tree-sitter and extracting code elements (classes, functions, methods,
imports, references, etc.).

Main exports:
- get_extractor(language): Get an extractor instance for a language
- ExtractedElement, ExtractedImport, etc.: Data classes for extracted elements
- BaseExtractor: Abstract base class for implementing extractors
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from magaldi_core.extractors.base import (
    BaseExtractor,
    find_nodes,
    get_child_by_field,
    get_children_by_type,
    get_node_text,
    walk_tree,
)
from magaldi_core.extractors.java import (
    JavaExtractor,
    extract_java_base_classes,
    extract_java_calls,
    extract_java_class_fields,
    extract_java_class_members,
    extract_java_elements,
    extract_java_imports,
    extract_java_modified_attributes,
    extract_java_thrown_exceptions,
    extract_top_level_java_calls,
)
from magaldi_core.extractors.javascript import (
    JavaScriptExtractor,
    extract_javascript_base_class,
    extract_javascript_calls,
    extract_javascript_class_fields,
    extract_javascript_class_members,
    extract_javascript_elements,
    extract_javascript_imports,
    extract_javascript_modified_properties,
    extract_javascript_references,
    extract_javascript_thrown_exceptions,
)
from magaldi_core.extractors.php import (
    PHPExtractor,
    extract_php_base_class,
    extract_php_calls,
    extract_php_class_members,
    extract_php_class_properties,
    extract_php_elements,
    extract_php_imports,
    extract_php_modified_properties,
    extract_php_thrown_exceptions,
)
from magaldi_core.extractors.python import (
    PythonExtractor,
    extract_python_base_classes,
    extract_python_calls,
    extract_python_class_attributes,
    extract_python_class_members,
    extract_python_elements,
    extract_python_imports,
    extract_python_modified_attributes,
    extract_python_raised_exceptions,
    extract_python_references,
)
from magaldi_core.extractors.rust import (
    RustExtractor,
    extract_rust_calls,
    extract_rust_elements,
    extract_rust_impl_members,
    extract_rust_impl_traits,
    extract_rust_imports,
    extract_rust_modified_fields,
    extract_rust_panics,
    extract_rust_struct_fields,
)
from magaldi_core.extractors.types import (
    CliCommand,
    Comment,
    DecoratorInfo,
    ExtractedCall,
    ExtractedElement,
    ExtractedImport,
    ExtractedReference,
    HttpRoute,
    ParameterInfo,
    PurityInfo,
    SectionMarker,
    SideEffect,
    TodoItem,
    TreeSitterError,
    TypeAnnotation,
)

if TYPE_CHECKING:
    pass


# Extractor registry
_EXTRACTORS: dict[str, type[BaseExtractor]] = {
    "python": PythonExtractor,
    "javascript": JavaScriptExtractor,
    "typescript": JavaScriptExtractor,
    "php": PHPExtractor,
    "rust": RustExtractor,
    "java": JavaExtractor,
}


def get_extractor(language: str) -> BaseExtractor | None:
    """Get an extractor instance for the specified language.

    Args:
        language: Language name (e.g., "python", "javascript", "typescript", "php", "rust")

    Returns:
        An extractor instance, or None if language is not supported.
    """
    extractor_class = _EXTRACTORS.get(language)
    if extractor_class is None:
        return None

    if language == "typescript":
        return JavaScriptExtractor(language="typescript")

    return extractor_class()


__all__ = [
    # Factory function
    "get_extractor",
    # Base class
    "BaseExtractor",
    # Tree utilities
    "walk_tree",
    "find_nodes",
    "get_node_text",
    "get_child_by_field",
    "get_children_by_type",
    # Data classes
    "TreeSitterError",
    "ExtractedImport",
    "DecoratorInfo",
    "ParameterInfo",
    "ExtractedElement",
    "ExtractedReference",
    "ExtractedCall",
    "TypeAnnotation",
    "TodoItem",
    "SectionMarker",
    "Comment",
    "HttpRoute",
    "CliCommand",
    "PurityInfo",
    "SideEffect",
    # Extractor classes
    "PythonExtractor",
    "JavaScriptExtractor",
    "PHPExtractor",
    "RustExtractor",
    "JavaExtractor",
    # Python extraction functions
    "extract_python_elements",
    "extract_python_class_members",
    "extract_python_imports",
    "extract_python_references",
    "extract_python_calls",
    "extract_python_class_attributes",
    "extract_python_base_classes",
    "extract_python_raised_exceptions",
    "extract_python_modified_attributes",
    # JavaScript extraction functions
    "extract_javascript_elements",
    "extract_javascript_class_members",
    "extract_javascript_imports",
    "extract_javascript_references",
    "extract_javascript_calls",
    "extract_javascript_class_fields",
    "extract_javascript_base_class",
    "extract_javascript_thrown_exceptions",
    "extract_javascript_modified_properties",
    # PHP extraction functions
    "extract_php_elements",
    "extract_php_class_members",
    "extract_php_imports",
    "extract_php_calls",
    "extract_php_class_properties",
    "extract_php_base_class",
    "extract_php_thrown_exceptions",
    "extract_php_modified_properties",
    # Rust extraction functions
    "extract_rust_elements",
    "extract_rust_impl_members",
    "extract_rust_imports",
    "extract_rust_calls",
    "extract_rust_struct_fields",
    "extract_rust_impl_traits",
    "extract_rust_panics",
    "extract_rust_modified_fields",
    # Java extraction functions
    "extract_java_elements",
    "extract_java_class_members",
    "extract_java_imports",
    "extract_java_calls",
    "extract_top_level_java_calls",
    "extract_java_class_fields",
    "extract_java_base_classes",
    "extract_java_thrown_exceptions",
    "extract_java_modified_attributes",
]
