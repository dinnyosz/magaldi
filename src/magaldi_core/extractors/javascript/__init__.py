"""JavaScript/TypeScript language extractor using tree-sitter.

This package provides the JavaScriptExtractor class and standalone functions
for extracting code elements, imports, references, call graph data,
and enhanced context from JavaScript and TypeScript source code.

The functionality is split into focused modules:
- element_extractor: Class, function, variable, and TypeScript type extraction
- import_extractor: ES6 imports and CommonJS require extraction
- reference_extractor: Cross-file reference extraction (calls, instantiations, type hints)
- call_extractor: Function/method call extraction within functions
- context_extractor: Class fields, base classes, exceptions, modified properties
"""

from __future__ import annotations

from magaldi_core.extractors.javascript.call_extractor import (
    extract_javascript_calls,
    extract_top_level_javascript_calls,
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
from magaldi_core.extractors.javascript.extractor import JavaScriptExtractor
from magaldi_core.extractors.javascript.import_extractor import (
    extract_javascript_imports,
)
from magaldi_core.extractors.javascript.reference_extractor import (
    extract_javascript_references,
)

__all__ = [
    # Main extractor class
    "JavaScriptExtractor",
    # Element extraction
    "extract_javascript_elements",
    "extract_javascript_class_members",
    # Import extraction
    "extract_javascript_imports",
    # Reference extraction
    "extract_javascript_references",
    # Call extraction
    "extract_javascript_calls",
    "extract_top_level_javascript_calls",
    # Context extraction
    "extract_javascript_class_fields",
    "extract_javascript_base_class",
    "extract_javascript_thrown_exceptions",
    "extract_javascript_modified_properties",
]
