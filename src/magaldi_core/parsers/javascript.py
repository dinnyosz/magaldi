"""JavaScript/TypeScript parser using tree-sitter.

Extracts code elements from JavaScript and TypeScript source files including:
- Classes with fields and base classes
- Functions and methods with calls and exceptions
- TypeScript interfaces and type aliases
- Variables and imports
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
    extract_javascript_base_class,
    extract_javascript_calls,
    extract_top_level_javascript_calls,
    extract_javascript_class_fields,
    extract_javascript_class_members,
    extract_javascript_elements,
    extract_javascript_imports,
    extract_javascript_modified_properties,
    extract_javascript_thrown_exceptions,
)

if TYPE_CHECKING:
    from magaldi_core.change_detection import FileInfo


class JavaScriptParser(TreeSitterParser):
    """Parse JavaScript/TypeScript files using tree-sitter."""

    def __init__(self, language: str = "javascript") -> None:
        super().__init__(language)

    def parse(
        self,
        content: str,
        file_info: "FileInfo",
        scope: str,
        repository: str,
        username: str,
    ) -> list[CodeElement]:
        """Parse JavaScript content and extract elements."""
        elements: list[CodeElement] = []
        lines = content.split("\n")

        # Create file-level element
        file_element = self._create_file_element(
            content, file_info, scope, repository, username, file_info.language
        )
        elements.append(file_element)

        # Parse with tree-sitter
        tree = self.manager.parse(content.encode("utf-8"), self.language)
        extracted = extract_javascript_elements(tree, lines)

        # Extract imports and populate on file element
        extracted_imports = extract_javascript_imports(tree, lines)
        file_element.imports = [
            Import(
                name=imp.name,
                module=imp.module,
                alias=imp.alias,
                line=imp.line,
            )
            for imp in extracted_imports
        ]

        # Extract top-level calls and populate on file element
        top_level_calls = extract_top_level_javascript_calls(tree)
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
                    methods, fields = extract_javascript_class_members(ext.node, lines)

                    for method_ext in methods:
                        method_elem = self._convert_method(
                            method_ext, class_elem, file_info, scope, repository, username, lines
                        )
                        elements.append(method_elem)

                    for field_ext in fields:
                        field_elem = self._convert_variable(
                            field_ext, file_info, scope, repository, username, lines, parent=class_elem
                        )
                        elements.append(field_elem)

            elif ext.element_type == "function":
                func_elem = self._convert_function(ext, file_info, scope, repository, username, lines)
                elements.append(func_elem)

            elif ext.element_type in ("interface", "type_alias"):
                # TypeScript interfaces and type aliases are type-level constructs
                type_elem = self._convert_type_definition(ext, file_info, scope, repository, username)
                elements.append(type_elem)

            elif ext.element_type == "import":
                # Import statements
                import_elem = self._convert_import(ext, file_info, scope, repository, username)
                elements.append(import_elem)

        # Set parent IDs
        self._set_hierarchy(elements, file_element)

        # Resolve same-file and this-method calls (Phase 1)
        self._resolve_calls_in_file(elements, self_keyword="this")

        return elements

    def _convert_type_definition(
        self,
        ext: ExtractedElement,
        file_info: "FileInfo",
        scope: str,
        repository: str,
        username: str,
    ) -> CodeElement:
        """Convert TypeScript interface or type alias to CodeElement."""
        elem = CodeElement(
            scope=scope,
            repository=repository,
            username=username,
            relative_path=file_info.relative_path,
            element_type=ext.element_type,  # 'interface' or 'type_alias'
            name=ext.name,
            language=file_info.language,
            line_start=ext.line_start,
            line_end=ext.line_end,
            raw_code=ext.raw_code,
            signature=ext.signature,
            level=1,  # Same level as classes
        )
        elem.element_id = generate_element_id(
            scope, repository, username, file_info.relative_path,
            ext.element_type, ext.name, ext.get_byte_offset()
        )
        return elem

    def _convert_import(
        self,
        ext: ExtractedElement,
        file_info: "FileInfo",
        scope: str,
        repository: str,
        username: str,
    ) -> CodeElement:
        """Convert import statement to CodeElement."""
        elem = CodeElement(
            scope=scope,
            repository=repository,
            username=username,
            relative_path=file_info.relative_path,
            element_type="import",
            name=ext.name,  # Module path
            language=file_info.language,
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
        # Extract class fields and base class from AST
        class_attributes = None
        base_classes = None
        if ext.node:
            fields = extract_javascript_class_fields(ext.node)
            class_attributes = fields if fields else None
            base_classes = extract_javascript_base_class(ext.node) or None

        elem = CodeElement(
            scope=scope,
            repository=repository,
            username=username,
            relative_path=file_info.relative_path,
            element_type="class",
            name=ext.name,
            language=file_info.language,
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
        # Extract calls and exceptions from function body
        calls: list[Call] = []
        exceptions_raised = None
        if ext.node:
            extracted_calls = extract_javascript_calls(ext.node)
            calls = [
                Call(name=c.name, receiver=c.receiver, line=c.line)
                for c in extracted_calls
            ]
            exceptions_raised = extract_javascript_thrown_exceptions(ext.node) or None

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
            language=file_info.language,
            line_start=ext.line_start,
            line_end=ext.line_end,
            raw_code=ext.raw_code,
            signature=ext.signature,
            is_async=ext.is_async,
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
        # Extract calls, exceptions, and modified properties from method body
        calls: list[Call] = []
        exceptions_raised = None
        attributes_modified = None
        if ext.node:
            extracted_calls = extract_javascript_calls(ext.node)
            calls = [
                Call(name=c.name, receiver=c.receiver, line=c.line)
                for c in extracted_calls
            ]
            exceptions_raised = extract_javascript_thrown_exceptions(ext.node) or None
            attributes_modified = extract_javascript_modified_properties(ext.node) or None

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
            language=file_info.language,
            line_start=ext.line_start,
            line_end=ext.line_end,
            raw_code=ext.raw_code,
            signature=ext.signature,
            is_async=ext.is_async,
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
        """Convert extracted variable/field to CodeElement."""
        elem = CodeElement(
            scope=scope,
            repository=repository,
            username=username,
            relative_path=file_info.relative_path,
            element_type="variable",
            name=ext.name,
            language=file_info.language,
            line_start=ext.line_start,
            line_end=ext.line_end,
            raw_code=ext.raw_code,
            level=3,
            parent_id=parent.element_id if parent else None,
        )
        elem.element_id = generate_element_id(
            scope, repository, username, file_info.relative_path, "variable", ext.name, ext.get_byte_offset()
        )
        return elem
