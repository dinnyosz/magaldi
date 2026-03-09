"""Rust parser using tree-sitter.

Extracts code elements from Rust source files including:
- Structs with fields
- Enums
- Impl blocks with methods and associated functions
- Free functions
- Constants
- Imports (use statements)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from magaldi_core.parsers.base import (
    Call,
    CodeElement,
    Import,
    TreeSitterParser,
    extract_preceding_doc_comment,
    generate_element_id,
)
from magaldi_core.tree_sitter_manager import (
    ExtractedElement,
    extract_rust_calls,
    extract_rust_elements,
    extract_rust_impl_members,
    extract_rust_impl_traits,
    extract_rust_imports,
    extract_rust_modified_fields,
    extract_rust_panics,
    extract_rust_struct_fields,
    extract_rust_trait_members,
    extract_top_level_rust_calls,
)

if TYPE_CHECKING:
    from magaldi_core.change_detection import FileInfo

# Rust test attribute names that mark test functions
_TEST_ATTRIBUTES = frozenset({"test", "tokio::test", "async_std::test"})


def _is_test_function(decorators: list[str] | None) -> bool:
    """Check if a Rust function/method has a test attribute."""
    if not decorators:
        return False
    return any(d in _TEST_ATTRIBUTES for d in decorators)


class RustParser(TreeSitterParser):
    """Parse Rust files using tree-sitter."""

    def __init__(self) -> None:
        super().__init__("rust")

    def parse(
        self,
        content: str,
        file_info: FileInfo,
        scope: str,
        repository: str,
        username: str,
    ) -> list[CodeElement]:
        """Parse Rust content and extract elements."""
        elements: list[CodeElement] = []
        lines = content.split("\n")

        # Create file-level element
        file_element = self._create_file_element(
            content, file_info, scope, repository, username, "rust"
        )
        elements.append(file_element)

        # Parse with tree-sitter
        tree = self.manager.parse(content.encode("utf-8"), "rust")
        extracted = extract_rust_elements(tree, lines)

        # Extract imports
        extracted_imports = extract_rust_imports(tree, lines)
        file_element.imports = [
            Import(name=imp.name, module=imp.module, alias=imp.alias, line=imp.line)
            for imp in extracted_imports
        ]

        # Extract top-level calls and populate on file element
        top_level_calls = extract_top_level_rust_calls(tree)
        file_element.calls = [
            Call(name=c.name, receiver=c.receiver, line=c.line)
            for c in top_level_calls
        ]

        # Convert ExtractedElements to CodeElements
        for ext in extracted:
            if ext.element_type == "class":
                # Could be struct, enum, or impl
                class_elem = self._convert_class(ext, file_info, scope, repository, username, lines)
                elements.append(class_elem)

                # For impl blocks, extract methods
                if ext.node and ext.node.type == "impl_item":
                    methods, constants = extract_rust_impl_members(ext.node, lines)

                    for method_ext in methods:
                        method_elem = self._convert_method(
                            method_ext, class_elem, file_info, scope, repository, username, lines
                        )
                        elements.append(method_elem)

                    for const_ext in constants:
                        const_elem = self._convert_constant(
                            const_ext, file_info, scope, repository, username, lines, parent=class_elem
                        )
                        elements.append(const_elem)

            elif ext.element_type == "function":
                func_elem = self._convert_function(ext, file_info, scope, repository, username, lines)
                elements.append(func_elem)

            elif ext.element_type == "trait":
                trait_elem = self._convert_type_definition(ext, file_info, scope, repository, username, lines)
                elements.append(trait_elem)

                # Extract trait methods (abstract signatures + default impls)
                if ext.node:
                    trait_methods = extract_rust_trait_members(ext.node, lines)
                    for method_ext in trait_methods:
                        method_elem = self._convert_method(
                            method_ext, trait_elem, file_info, scope, repository, username, lines
                        )
                        elements.append(method_elem)

            elif ext.element_type in ("enum", "type_alias"):
                type_elem = self._convert_type_definition(ext, file_info, scope, repository, username, lines)
                elements.append(type_elem)

            elif ext.element_type == "import":
                import_elem = self._convert_import(ext, file_info, scope, repository, username)
                elements.append(import_elem)

            elif ext.element_type in ("constant", "variable"):
                const_elem = self._convert_constant(ext, file_info, scope, repository, username, lines, parent=None)
                elements.append(const_elem)

        # Set parent IDs
        self._set_hierarchy(elements, file_element)

        # Resolve same-file and self.method() calls (Phase 1)
        self._resolve_calls_in_file(elements, self_keyword="self")

        return elements

    def _convert_class(
        self,
        ext: ExtractedElement,
        file_info: FileInfo,
        scope: str,
        repository: str,
        username: str,
        lines: list[str],
    ) -> CodeElement:
        """Convert extracted struct/enum/impl to CodeElement."""
        class_attributes = None
        base_classes = None

        if ext.node:
            if ext.node.type == "struct_item":
                fields = extract_rust_struct_fields(ext.node)
                class_attributes = fields if fields else None
            elif ext.node.type == "impl_item":
                traits = extract_rust_impl_traits(ext.node)
                base_classes = traits if traits else None

        elem = CodeElement(
            scope=scope,
            repository=repository,
            username=username,
            relative_path=file_info.relative_path,
            element_type="class",
            name=ext.name,
            language="rust",
            line_start=ext.line_start,
            line_end=ext.line_end,
            raw_code=ext.raw_code,
            docstring=extract_preceding_doc_comment(lines, ext.line_start, "rust"),
            decorators=ext.decorators or [],
            visibility=ext.visibility or "public",
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
        file_info: FileInfo,
        scope: str,
        repository: str,
        username: str,
        lines: list[str],
    ) -> CodeElement:
        """Convert extracted function to CodeElement."""
        calls: list[Call] = []
        exceptions_raised = None
        if ext.node:
            extracted_calls = extract_rust_calls(ext.node)
            calls = [Call(name=c.name, receiver=c.receiver, line=c.line) for c in extracted_calls]
            exceptions_raised = extract_rust_panics(ext.node) or None

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
            language="rust",
            line_start=ext.line_start,
            line_end=ext.line_end,
            raw_code=ext.raw_code,
            signature=ext.signature,
            docstring=extract_preceding_doc_comment(lines, ext.line_start, "rust"),
            is_async=ext.is_async,
            is_test=_is_test_function(ext.decorators),
            decorators=ext.decorators or [],
            visibility=ext.visibility or "public",
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
        file_info: FileInfo,
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
            extracted_calls = extract_rust_calls(ext.node)
            calls = [Call(name=c.name, receiver=c.receiver, line=c.line) for c in extracted_calls]
            exceptions_raised = extract_rust_panics(ext.node) or None
            attributes_modified = extract_rust_modified_fields(ext.node) or None

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
            element_type=ext.element_type,  # method or function (associated)
            name=ext.name,
            language="rust",
            line_start=ext.line_start,
            line_end=ext.line_end,
            raw_code=ext.raw_code,
            signature=ext.signature,
            docstring=extract_preceding_doc_comment(lines, ext.line_start, "rust"),
            is_async=ext.is_async,
            is_test=_is_test_function(ext.decorators),
            decorators=ext.decorators or [],
            visibility=ext.visibility or "public",
            level=2,
            parent_id=parent_class.element_id,
            calls=calls,
            exceptions_raised=exceptions_raised,
            attributes_modified=attributes_modified,
            return_type=ext.return_type,
            parameters=parameters or [],
        )
        elem.element_id = generate_element_id(
            scope, repository, username, file_info.relative_path, ext.element_type, ext.name, ext.get_byte_offset()
        )
        return elem

    def _convert_constant(
        self,
        ext: ExtractedElement,
        file_info: FileInfo,
        scope: str,
        repository: str,
        username: str,
        lines: list[str],
        parent: CodeElement | None = None,
    ) -> CodeElement:
        """Convert extracted constant to CodeElement."""
        elem = CodeElement(
            scope=scope,
            repository=repository,
            username=username,
            relative_path=file_info.relative_path,
            element_type="constant",
            name=ext.name,
            language="rust",
            line_start=ext.line_start,
            line_end=ext.line_end,
            raw_code=ext.raw_code,
            docstring=extract_preceding_doc_comment(lines, ext.line_start, "rust"),
            visibility=ext.visibility or "public",
            level=3,
            parent_id=parent.element_id if parent else None,
        )
        elem.element_id = generate_element_id(
            scope, repository, username, file_info.relative_path, "constant", ext.name, ext.get_byte_offset()
        )
        return elem

    def _convert_type_definition(
        self,
        ext: ExtractedElement,
        file_info: FileInfo,
        scope: str,
        repository: str,
        username: str,
        lines: list[str],
    ) -> CodeElement:
        """Convert extracted enum, trait, or type_alias to CodeElement."""
        elem = CodeElement(
            scope=scope,
            repository=repository,
            username=username,
            relative_path=file_info.relative_path,
            element_type=ext.element_type,
            name=ext.name,
            language="rust",
            line_start=ext.line_start,
            line_end=ext.line_end,
            raw_code=ext.raw_code,
            signature=ext.signature,
            docstring=extract_preceding_doc_comment(lines, ext.line_start, "rust"),
            visibility=ext.visibility or "public",
            level=1,
        )
        elem.element_id = generate_element_id(
            scope, repository, username, file_info.relative_path,
            ext.element_type, ext.name, ext.get_byte_offset()
        )
        return elem

    def _convert_import(
        self,
        ext: ExtractedElement,
        file_info: FileInfo,
        scope: str,
        repository: str,
        username: str,
    ) -> CodeElement:
        """Convert extracted use declaration to CodeElement."""
        elem = CodeElement(
            scope=scope,
            repository=repository,
            username=username,
            relative_path=file_info.relative_path,
            element_type="import",
            name=ext.name,
            language="rust",
            line_start=ext.line_start,
            line_end=ext.line_end,
            raw_code=ext.raw_code,
            signature=ext.raw_code.strip(),
            level=2,
        )
        elem.element_id = generate_element_id(
            scope, repository, username, file_info.relative_path,
            "import", ext.name, ext.get_byte_offset()
        )
        return elem
