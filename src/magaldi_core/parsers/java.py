"""Java parser using tree-sitter.

Extracts code elements from Java source files including:
- Classes with fields and base classes
- Interfaces with methods
- Enums with methods
- Records (Java 16+)
- Constructors and methods with calls and exceptions
- Imports (regular, static, wildcard)
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

if TYPE_CHECKING:
    from magaldi_core.change_detection import FileInfo

_VISIBILITY_KEYWORDS = frozenset({"public", "private", "protected"})


def _is_test_method(decorators: list[str] | None) -> bool:
    """Check if method has a JUnit test annotation."""
    if not decorators:
        return False
    return "@Test" in decorators or "@ParameterizedTest" in decorators


def _extract_visibility(decorators: list[str] | None) -> str:
    """Extract visibility modifier from decorator list."""
    if decorators:
        for d in decorators:
            if d in _VISIBILITY_KEYWORDS:
                return d
    return "public"  # Java default is package-private, but we map to public for consistency


class JavaParser(TreeSitterParser):
    """Parse Java files using tree-sitter."""

    def __init__(self) -> None:
        super().__init__("java")

    def parse(
        self,
        content: str,
        file_info: FileInfo,
        scope: str,
        repository: str,
        username: str,
    ) -> list[CodeElement]:
        """Parse Java content and extract elements."""
        elements: list[CodeElement] = []
        lines = content.split("\n")

        # Create file-level element
        file_element = self._create_file_element(
            content, file_info, scope, repository, username, "java"
        )
        elements.append(file_element)

        # Parse with tree-sitter
        tree = self.manager.parse(content.encode("utf-8"), "java")
        extracted = extract_java_elements(tree, lines)

        # Extract imports
        extracted_imports = extract_java_imports(tree, lines)
        file_element.imports = [
            Import(name=imp.name, module=imp.module, alias=imp.alias, line=imp.line)
            for imp in extracted_imports
        ]

        # Extract top-level calls (field initializers, etc.)
        top_level_calls = extract_top_level_java_calls(tree)
        file_element.calls = [
            Call(name=c.name, receiver=c.receiver, line=c.line)
            for c in top_level_calls
        ]

        # Convert ExtractedElements to CodeElements
        for ext in extracted:
            if ext.element_type in ("class", "interface", "enum"):
                class_elem = self._convert_class(
                    ext, file_info, scope, repository, username, lines
                )
                elements.append(class_elem)

                # Extract class members
                if ext.node:
                    methods, fields = extract_java_class_members(ext.node, lines)

                    for method_ext in methods:
                        method_elem = self._convert_method(
                            method_ext, class_elem, file_info, scope, repository, username, lines
                        )
                        elements.append(method_elem)

                    for field_ext in fields:
                        field_elem = self._convert_constant(
                            field_ext, file_info, scope, repository, username, lines,
                            parent=class_elem,
                        )
                        elements.append(field_elem)

            elif ext.element_type == "import":
                import_elem = self._convert_import(
                    ext, file_info, scope, repository, username
                )
                elements.append(import_elem)

        # Set parent IDs
        self._set_hierarchy(elements, file_element)

        # Resolve same-file and this.method() calls (Phase 1)
        self._resolve_calls_in_file(elements, self_keyword="this")

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
        """Convert extracted class/interface/enum to CodeElement."""
        class_attributes = None
        base_classes = None

        if ext.node:
            fields = extract_java_class_fields(ext.node)
            class_attributes = fields if fields else None
            bases = extract_java_base_classes(ext.node)
            base_classes = bases if bases else None

        visibility = _extract_visibility(ext.decorators)

        elem = CodeElement(
            scope=scope,
            repository=repository,
            username=username,
            relative_path=file_info.relative_path,
            element_type=ext.element_type,
            name=ext.name,
            language="java",
            line_start=ext.line_start,
            line_end=ext.line_end,
            raw_code=ext.raw_code,
            signature=ext.signature,
            docstring=extract_preceding_doc_comment(lines, ext.line_start, "java"),
            decorators=ext.decorators or [],
            visibility=visibility,
            level=1,
            class_attributes=class_attributes,
            base_classes=base_classes,
        )
        elem.element_id = generate_element_id(
            scope, repository, username, file_info.relative_path,
            ext.element_type, ext.name, ext.get_byte_offset(),
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
        """Convert extracted method/constructor to CodeElement."""
        calls: list[Call] = []
        exceptions_raised = None
        attributes_modified = None
        if ext.node:
            extracted_calls = extract_java_calls(ext.node)
            calls = [Call(name=c.name, receiver=c.receiver, line=c.line) for c in extracted_calls]
            exceptions_raised = extract_java_thrown_exceptions(ext.node) or None
            attributes_modified = extract_java_modified_attributes(ext.node) or None

        # Convert parameters to dicts for storage
        parameters: list[dict[str, str | None]] = []
        if ext.parameters:
            parameters = [
                {"name": p.name, "type": p.type, "default": p.default}
                for p in ext.parameters
            ]

        visibility = _extract_visibility(ext.decorators)

        elem = CodeElement(
            scope=scope,
            repository=repository,
            username=username,
            relative_path=file_info.relative_path,
            element_type="method",
            name=ext.name,
            language="java",
            line_start=ext.line_start,
            line_end=ext.line_end,
            raw_code=ext.raw_code,
            signature=ext.signature,
            docstring=extract_preceding_doc_comment(lines, ext.line_start, "java"),
            is_async=ext.is_async,
            is_test=_is_test_method(ext.decorators),
            decorators=ext.decorators or [],
            visibility=visibility,
            level=2,
            parent_id=parent_class.element_id,
            calls=calls,
            exceptions_raised=exceptions_raised,
            attributes_modified=attributes_modified,
            return_type=ext.return_type,
            parameters=parameters or [],
        )
        elem.element_id = generate_element_id(
            scope, repository, username, file_info.relative_path,
            "method", ext.name, ext.get_byte_offset(),
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
        """Convert extracted field/constant to CodeElement."""
        visibility = _extract_visibility(ext.decorators)

        elem = CodeElement(
            scope=scope,
            repository=repository,
            username=username,
            relative_path=file_info.relative_path,
            element_type=ext.element_type,
            name=ext.name,
            language="java",
            line_start=ext.line_start,
            line_end=ext.line_end,
            raw_code=ext.raw_code,
            docstring=extract_preceding_doc_comment(lines, ext.line_start, "java"),
            decorators=ext.decorators or [],
            visibility=visibility,
            level=3,
            parent_id=parent.element_id if parent else None,
        )
        elem.element_id = generate_element_id(
            scope, repository, username, file_info.relative_path,
            ext.element_type, ext.name, ext.get_byte_offset(),
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
        """Convert extracted import declaration to CodeElement."""
        elem = CodeElement(
            scope=scope,
            repository=repository,
            username=username,
            relative_path=file_info.relative_path,
            element_type="import",
            name=ext.name,
            language="java",
            line_start=ext.line_start,
            line_end=ext.line_end,
            raw_code=ext.raw_code,
            signature=ext.raw_code.strip(),
            level=2,
        )
        elem.element_id = generate_element_id(
            scope, repository, username, file_info.relative_path,
            "import", ext.name, ext.get_byte_offset(),
        )
        return elem
