"""Language-specific tree-sitter extractors.

Each language module provides functions for extracting:
- Elements (classes, functions, methods, variables)
- Imports
- References (calls, type hints)
- Call graph data
- Enhanced context (base classes, attributes, exceptions)
"""

from __future__ import annotations

# Re-export all language extractors for convenience
from magaldi_core.languages.javascript import (
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
from magaldi_core.languages.php import (
    extract_php_base_class,
    extract_php_calls,
    extract_php_class_members,
    extract_php_class_properties,
    extract_php_elements,
    extract_php_imports,
    extract_php_modified_properties,
    extract_php_thrown_exceptions,
)
from magaldi_core.languages.python import (
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
from magaldi_core.languages.rust import (
    extract_rust_calls,
    extract_rust_elements,
    extract_rust_impl_members,
    extract_rust_impl_traits,
    extract_rust_imports,
    extract_rust_modified_fields,
    extract_rust_panics,
    extract_rust_struct_fields,
)

__all__ = [
    # Python
    "extract_python_elements",
    "extract_python_class_members",
    "extract_python_imports",
    "extract_python_references",
    "extract_python_calls",
    "extract_python_class_attributes",
    "extract_python_base_classes",
    "extract_python_raised_exceptions",
    "extract_python_modified_attributes",
    # JavaScript
    "extract_javascript_elements",
    "extract_javascript_class_members",
    "extract_javascript_imports",
    "extract_javascript_references",
    "extract_javascript_calls",
    "extract_javascript_class_fields",
    "extract_javascript_base_class",
    "extract_javascript_thrown_exceptions",
    "extract_javascript_modified_properties",
    # PHP
    "extract_php_elements",
    "extract_php_class_members",
    "extract_php_imports",
    "extract_php_calls",
    "extract_php_class_properties",
    "extract_php_base_class",
    "extract_php_thrown_exceptions",
    "extract_php_modified_properties",
    # Rust
    "extract_rust_elements",
    "extract_rust_impl_members",
    "extract_rust_imports",
    "extract_rust_calls",
    "extract_rust_struct_fields",
    "extract_rust_impl_traits",
    "extract_rust_panics",
    "extract_rust_modified_fields",
]
