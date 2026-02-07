"""PHP parser using tree-sitter.

Extracts code elements from PHP source files including:
- Classes with properties and base classes
- Functions and methods with calls and exceptions
- Variables and imports (use statements)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from magaldi_core.parsers.base import (
    Call,
    CodeElement,
    Import,
    TreeSitterParser,
    generate_element_id,
)
from magaldi_core.tree_sitter_manager import (
    ExtractedElement,
    extract_php_base_class,
    extract_php_calls,
    extract_top_level_php_calls,
    extract_php_class_members,
    extract_php_class_properties,
    extract_php_elements,
    extract_php_imports,
    extract_php_modified_properties,
    extract_php_thrown_exceptions,
)

if TYPE_CHECKING:
    from magaldi_core.change_detection import FileInfo


class PhpParser(TreeSitterParser):
    """Parse PHP files using tree-sitter."""

    def __init__(self) -> None:
        super().__init__("php")

    def parse(
        self,
        content: str,
        file_info: "FileInfo",
        scope: str,
        repository: str,
        username: str,
    ) -> list[CodeElement]:
        """Parse PHP content and extract elements."""
        elements: list[CodeElement] = []
        lines = content.split("\n")

        # Create file-level element
        file_element = self._create_file_element(
            content, file_info, scope, repository, username, "php"
        )
        elements.append(file_element)

        # Parse with tree-sitter
        tree = self.manager.parse(content.encode("utf-8"), "php")
        extracted = extract_php_elements(tree, lines)

        # Extract imports
        extracted_imports = extract_php_imports(tree, lines)
        file_element.imports = [
            Import(name=imp.name, module=imp.module, alias=imp.alias, line=imp.line)
            for imp in extracted_imports
        ]

        # Extract top-level calls and populate on file element
        top_level_calls = extract_top_level_php_calls(tree)
        file_element.calls = [
            Call(name=c.name, receiver=c.receiver, line=c.line)
            for c in top_level_calls
        ]

        # Convert ExtractedElements to CodeElements
        for ext in extracted:
            if ext.element_type == "class":
                class_elem = self._convert_class(ext, file_info, scope, repository, username, lines)
                elements.append(class_elem)

                # Extract class members
                if ext.node:
                    methods, properties = extract_php_class_members(ext.node, lines)

                    for method_ext in methods:
                        method_elem = self._convert_method(
                            method_ext, class_elem, file_info, scope, repository, username, lines
                        )
                        elements.append(method_elem)

                    for prop_ext in properties:
                        prop_elem = self._convert_variable(
                            prop_ext, file_info, scope, repository, username, lines, parent=class_elem
                        )
                        elements.append(prop_elem)

            elif ext.element_type == "function":
                func_elem = self._convert_function(ext, file_info, scope, repository, username, lines)
                elements.append(func_elem)

            elif ext.element_type == "import":
                import_elem = self._convert_import(ext, file_info, scope, repository, username)
                elements.append(import_elem)

        # Set parent IDs
        self._set_hierarchy(elements, file_element)

        # Resolve same-file and $this->method() calls (Phase 1)
        # PHP uses both '$this' and 'this' for self-reference
        self._resolve_calls_in_file_php(elements)

        return elements

    def _resolve_calls_in_file_php(self, elements: list[CodeElement]) -> None:
        """Resolve calls that can be determined at parse time (Phase 1).

        Resolves:
        - Same-file bare function calls: func() -> file-level function
        - $this->method() calls: -> sibling method in same class

        Also categorizes unresolved calls.
        """
        from magaldi_core.extractors.call_categorizer import categorize_calls

        # Build lookup for file-level functions
        file_functions: dict[str, str] = {
            e.name: e.element_id
            for e in elements
            if e.element_type == "function"
        }

        # Build lookup for methods grouped by parent class
        class_methods: dict[str, dict[str, str]] = {}
        for e in elements:
            if e.element_type == "method" and e.parent_id:
                if e.parent_id not in class_methods:
                    class_methods[e.parent_id] = {}
                class_methods[e.parent_id][e.name] = e.element_id

        # Resolve calls in each element
        for elem in elements:
            if not elem.calls:
                continue

            for call in elem.calls:
                if call.resolved_id:
                    continue

                # Strategy 1: Same-file bare function call
                if call.receiver is None and call.name in file_functions:
                    call.resolved_id = file_functions[call.name]
                    continue

                # Strategy 2: $this->method() call within class (PHP uses '$this' or 'this')
                if call.receiver in ("$this", "this") and elem.parent_id:
                    sibling_methods = class_methods.get(elem.parent_id, {})
                    if call.name in sibling_methods:
                        call.resolved_id = sibling_methods[call.name]

            # Categorize all calls (resolved and unresolved)
            categorize_calls(elem.calls, "php", elem.parameters)

    def _convert_import(
        self,
        ext: ExtractedElement,
        file_info: "FileInfo",
        scope: str,
        repository: str,
        username: str,
    ) -> CodeElement:
        """Convert PHP use statement (import) to CodeElement."""
        elem = CodeElement(
            scope=scope,
            repository=repository,
            username=username,
            relative_path=file_info.relative_path,
            element_type="import",
            name=ext.name,  # Full namespace path
            language="php",
            line_start=ext.line_start,
            line_end=ext.line_end,
            raw_code=ext.raw_code,
            signature=ext.signature,
            level=0,  # Top level, same as file
        )
        elem.element_id = generate_element_id(
            scope, repository, username, file_info.relative_path,
            "import", ext.name, ext.get_byte_offset()
        )
        return elem

    def _convert_class(
        self,
        ext: ExtractedElement,
        file_info: "FileInfo",
        scope: str,
        repository: str,
        username: str,
        lines: list[str],
    ) -> CodeElement:
        """Convert extracted class to CodeElement."""
        class_attributes = None
        base_classes = None
        if ext.node:
            props = extract_php_class_properties(ext.node)
            class_attributes = props if props else None
            base_classes = extract_php_base_class(ext.node) or None

        elem = CodeElement(
            scope=scope,
            repository=repository,
            username=username,
            relative_path=file_info.relative_path,
            element_type="class",
            name=ext.name,
            language="php",
            line_start=ext.line_start,
            line_end=ext.line_end,
            raw_code=ext.raw_code,
            level=1,
            class_attributes=class_attributes,
            base_classes=base_classes,
        )
        elem.element_id = generate_element_id(
            scope, repository, username, file_info.relative_path, "class", ext.name, ext.get_byte_offset()
        )
        return elem

    def _convert_function(
        self,
        ext: ExtractedElement,
        file_info: "FileInfo",
        scope: str,
        repository: str,
        username: str,
        lines: list[str],
    ) -> CodeElement:
        """Convert extracted function to CodeElement."""
        calls: list[Call] = []
        exceptions_raised = None
        if ext.node:
            extracted_calls = extract_php_calls(ext.node)
            calls = [Call(name=c.name, receiver=c.receiver, line=c.line) for c in extracted_calls]
            exceptions_raised = extract_php_thrown_exceptions(ext.node) or None

        # Convert parameters to dicts for storage
        parameters = None
        if ext.parameters:
            parameters = [
                {"name": p.name, "type": p.type, "default": p.default}
                for p in ext.parameters
            ]

        elem = CodeElement(
            scope=scope,
            repository=repository,
            username=username,
            relative_path=file_info.relative_path,
            element_type="function",
            name=ext.name,
            language="php",
            line_start=ext.line_start,
            line_end=ext.line_end,
            raw_code=ext.raw_code,
            signature=ext.signature,
            level=2,
            calls=calls,
            exceptions_raised=exceptions_raised,
            return_type=ext.return_type,
            parameters=parameters or [],
        )
        elem.element_id = generate_element_id(
            scope, repository, username, file_info.relative_path, "function", ext.name, ext.get_byte_offset()
        )
        return elem

    def _convert_method(
        self,
        ext: ExtractedElement,
        parent_class: CodeElement,
        file_info: "FileInfo",
        scope: str,
        repository: str,
        username: str,
        lines: list[str],
    ) -> CodeElement:
        """Convert extracted method to CodeElement."""
        calls: list[Call] = []
        exceptions_raised = None
        attributes_modified = None
        if ext.node:
            extracted_calls = extract_php_calls(ext.node)
            calls = [Call(name=c.name, receiver=c.receiver, line=c.line) for c in extracted_calls]
            exceptions_raised = extract_php_thrown_exceptions(ext.node) or None
            attributes_modified = extract_php_modified_properties(ext.node) or None

        # Convert parameters to dicts for storage
        parameters = None
        if ext.parameters:
            parameters = [
                {"name": p.name, "type": p.type, "default": p.default}
                for p in ext.parameters
            ]

        elem = CodeElement(
            scope=scope,
            repository=repository,
            username=username,
            relative_path=file_info.relative_path,
            element_type="method",
            name=ext.name,
            language="php",
            line_start=ext.line_start,
            line_end=ext.line_end,
            raw_code=ext.raw_code,
            signature=ext.signature,
            decorators=ext.decorators or [],
            level=2,
            parent_id=parent_class.element_id,
            calls=calls,
            exceptions_raised=exceptions_raised,
            attributes_modified=attributes_modified,
            return_type=ext.return_type,
            parameters=parameters or [],
        )
        elem.element_id = generate_element_id(
            scope, repository, username, file_info.relative_path, "method", ext.name, ext.get_byte_offset()
        )
        return elem

    def _convert_variable(
        self,
        ext: ExtractedElement,
        file_info: "FileInfo",
        scope: str,
        repository: str,
        username: str,
        lines: list[str],
        parent: CodeElement | None = None,
    ) -> CodeElement:
        """Convert extracted variable/property to CodeElement."""
        elem = CodeElement(
            scope=scope,
            repository=repository,
            username=username,
            relative_path=file_info.relative_path,
            element_type="variable",
            name=ext.name,
            language="php",
            line_start=ext.line_start,
            line_end=ext.line_end,
            raw_code=ext.raw_code,
            decorators=ext.decorators or [],
            level=3,
            parent_id=parent.element_id if parent else None,
        )
        elem.element_id = generate_element_id(
            scope, repository, username, file_info.relative_path, "variable", ext.name, ext.get_byte_offset()
        )
        return elem
