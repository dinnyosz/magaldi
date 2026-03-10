"""Parser tests for TypeScript namespace/module declaration extraction.

Covers:
- Exported namespace declarations
- Non-exported namespace declarations
- Module declarations (legacy syntax)
- Inner elements parented to namespace
- Namespace element_type and hierarchy
"""

from pathlib import Path

from magaldi_core.code_parser import JavaScriptParser
from magaldi_core.change_detection import FileInfo


def _parse_ts(code: str) -> list:
    """Helper to parse TypeScript code and return elements."""
    parser = JavaScriptParser("typescript")
    file_info = FileInfo(
        relative_path="test.ts",
        absolute_path=Path("/fake/test.ts"),
        language="typescript",
    )
    return parser.parse(code, file_info, "scope", "repo", "main")


class TestTypescriptNamespaceDeclaration:
    """Tests for typescript namespace/module declaration extraction."""

    def test_namespace_extracted_as_element(self):
        """Namespace declarations should be extracted as namespace elements."""
        code = (
            'export namespace util {\n'
            '  export function assertEqual<A, B>(val: HasSameShape<A, B>): void {}\n'
            '  export function assertIs<T>(_arg: T): void {}\n'
            '}'
        )
        elements = _parse_ts(code)

        ns = next(
            (e for e in elements if e.name == "util" and e.element_type == "namespace"),
            None,
        )
        assert ns is not None, (
            f"Expected namespace 'util' not found. Got: "
            f"{[(e.name, e.element_type) for e in elements]}"
        )

    def test_namespace_inner_functions_parented(self):
        """Functions inside a namespace should have parent_id pointing to the namespace."""
        code = (
            'export namespace util {\n'
            '  export function assertEqual(): void {}\n'
            '  export function assertIs(): void {}\n'
            '}'
        )
        elements = _parse_ts(code)

        ns = next(
            (e for e in elements if e.name == "util" and e.element_type == "namespace"),
            None,
        )
        assert ns is not None, "Namespace 'util' not found"

        fn_equal = next(
            (e for e in elements if e.name == "assertEqual"),
            None,
        )
        assert fn_equal is not None, "Function 'assertEqual' not found"
        assert fn_equal.parent_id == ns.element_id, (
            f"Expected parent_id={ns.element_id}, got {fn_equal.parent_id}"
        )

        fn_is = next(
            (e for e in elements if e.name == "assertIs"),
            None,
        )
        assert fn_is is not None, "Function 'assertIs' not found"
        assert fn_is.parent_id == ns.element_id, (
            f"Expected parent_id={ns.element_id}, got {fn_is.parent_id}"
        )

    def test_non_exported_namespace(self):
        """Non-exported namespaces should also be extracted."""
        code = (
            'namespace InternalModule {\n'
            '  export const VERSION = "1.0";\n'
            '  export class Helper {\n'
            '    static create(): Helper { return new Helper(); }\n'
            '  }\n'
            '}'
        )
        elements = _parse_ts(code)

        ns = next(
            (e for e in elements if e.name == "InternalModule" and e.element_type == "namespace"),
            None,
        )
        assert ns is not None, (
            f"Expected namespace 'InternalModule' not found. Got: "
            f"{[(e.name, e.element_type) for e in elements]}"
        )

        # Inner class should be parented to namespace
        helper = next(
            (e for e in elements if e.name == "Helper" and e.element_type == "class"),
            None,
        )
        assert helper is not None, "Class 'Helper' not found"
        assert helper.parent_id == ns.element_id, (
            f"Expected parent_id={ns.element_id}, got {helper.parent_id}"
        )

    def test_module_keyword_declaration(self):
        """The 'module' keyword (legacy TS syntax) should also be extracted as namespace."""
        code = (
            'export module LegacyModule {\n'
            '  export function doStuff(): void {}\n'
            '}'
        )
        elements = _parse_ts(code)

        ns = next(
            (e for e in elements if e.name == "LegacyModule" and e.element_type == "namespace"),
            None,
        )
        assert ns is not None, (
            f"Expected namespace 'LegacyModule' not found. Got: "
            f"{[(e.name, e.element_type) for e in elements]}"
        )

        fn = next(
            (e for e in elements if e.name == "doStuff" and e.element_type == "function"),
            None,
        )
        assert fn is not None, "Function 'doStuff' not found"
        assert fn.parent_id == ns.element_id, (
            f"Expected parent_id={ns.element_id}, got {fn.parent_id}"
        )

    def test_namespace_with_type_alias(self):
        """Type aliases inside a namespace should be parented to it."""
        code = (
            'export namespace util {\n'
            '  export type AssertEqual<A, B> = A extends B ? B extends A ? true : never : never;\n'
            '}'
        )
        elements = _parse_ts(code)

        ns = next(
            (e for e in elements if e.name == "util" and e.element_type == "namespace"),
            None,
        )
        assert ns is not None, "Namespace 'util' not found"

        ta = next(
            (e for e in elements if e.name == "AssertEqual" and e.element_type == "type_alias"),
            None,
        )
        assert ta is not None, "Type alias 'AssertEqual' not found"
        assert ta.parent_id == ns.element_id, (
            f"Expected parent_id={ns.element_id}, got {ta.parent_id}"
        )

    def test_full_namespace_element_count(self):
        """All elements (namespaces + inner elements) should be extracted."""
        code = (
            'export namespace util {\n'
            '  export function assertEqual<A, B>(val: HasSameShape<A, B>): void {}\n'
            '  export function assertIs<T>(_arg: T): void {}\n'
            '  export function assertNever(_x: never): never { throw new Error(); }\n'
            '  export type AssertEqual<A, B> = A extends B ? B extends A ? true : never : never;\n'
            '}\n'
            '\n'
            'namespace InternalModule {\n'
            '  export const VERSION = "1.0";\n'
            '  export class Helper {\n'
            '    static create(): Helper { return new Helper(); }\n'
            '  }\n'
            '}\n'
            '\n'
            'export module LegacyModule {\n'
            '  export function doStuff(): void {}\n'
            '}'
        )
        elements = _parse_ts(code)

        # 3 namespaces + 4 functions + 1 type_alias + 1 constant + 1 class + 1 method = 11
        # (Helper.create is a method inside the class)
        names = [(e.name, e.element_type) for e in elements]
        assert len(elements) >= 10, (
            f"Expected at least 10 elements, got {len(elements)}: {names}"
        )
