"""Python language extractor using tree-sitter.

This package provides the PythonExtractor class and standalone functions
for extracting code elements, imports, references, call graph data,
and enhanced context from Python source code.

The functionality is split into focused modules:
- element_extractor: Class, function, variable, and nested function extraction
- import_extractor: Import statement extraction
- reference_extractor: Cross-file reference extraction (calls, instantiations, type hints)
- call_extractor: Function/method call extraction within functions
- context_extractor: Instance attributes, base classes, exceptions, modified attributes
"""

from __future__ import annotations

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
from magaldi_core.extractors.python.extractor import PythonExtractor
from magaldi_core.extractors.python.import_extractor import (
    extract_python_imports,
)
from magaldi_core.extractors.python.reference_extractor import (
    extract_python_references,
)

__all__ = [
    # Main extractor class
    "PythonExtractor",
    # Element extraction
    "extract_python_elements",
    "extract_python_class_members",
    # Import extraction
    "extract_python_imports",
    # Reference extraction
    "extract_python_references",
    # Call extraction
    "extract_python_calls",
    # Context extraction
    "extract_python_class_attributes",
    "extract_python_base_classes",
    "extract_python_raised_exceptions",
    "extract_python_modified_attributes",
]
