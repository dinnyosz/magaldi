"""Auto-generated parser test for rust_inline_test_module_extraction."""

from pathlib import Path

from magaldi_core.code_parser import RustParser
from magaldi_core.change_detection import FileInfo


class TestRustInlineTestModuleExtraction:
    """Tests for inline #[cfg(test)] mod tests block extraction."""

    def test_rust_inline_test_module_extraction(self):
        code = (
            'use std::collections::HashMap;\n'
            '\n'
            'pub fn public_function() -> bool {\n'
            '    true\n'
            '}\n'
            '\n'
            '#[cfg(test)]\n'
            'mod tests {\n'
            '    use super::*;\n'
            '\n'
            '    #[test]\n'
            '    fn test_something() {\n'
            '        assert_eq!(1, 1);\n'
            '    }\n'
            '\n'
            '    #[test]\n'
            '    fn test_another() {\n'
            '        assert!(true);\n'
            '    }\n'
            '}\n'
        )

        parser = RustParser()
        file_info = FileInfo(
            relative_path="test.rs",
            absolute_path=Path("/fake/test.rs"),
            language="rust",
        )

        elements = parser.parse(code, file_info, "scope", "repo", "main")
        # Exclude file element for clarity
        non_file = [e for e in elements if e.element_type != "file"]

        # The mod tests block should be extracted as a class-like element
        mod_elem = next(
            (e for e in non_file if e.name == "tests" and e.element_type == "class"),
            None,
        )
        assert mod_elem is not None, "Expected class 'tests' (mod block) not found"

        # Functions inside the mod block should be extracted
        test_something = next(
            (e for e in non_file if e.name == "test_something" and e.element_type == "function"),
            None,
        )
        assert test_something is not None, "Expected function 'test_something' not found"

        test_another = next(
            (e for e in non_file if e.name == "test_another" and e.element_type == "function"),
            None,
        )
        assert test_another is not None, "Expected function 'test_another' not found"

        # The test functions should have parent_id pointing to the mod element
        assert test_something.parent_id == mod_elem.element_id, (
            f"test_something parent_id should be mod element, got {test_something.parent_id}"
        )
        assert test_another.parent_id == mod_elem.element_id, (
            f"test_another parent_id should be mod element, got {test_another.parent_id}"
        )

    def test_nested_mod_with_impl_blocks(self):
        """Modules can contain impl blocks too, not just functions."""
        code = (
            'mod inner {\n'
            '    struct Foo;\n'
            '\n'
            '    impl Foo {\n'
            '        fn bar(&self) {}\n'
            '    }\n'
            '}\n'
        )

        parser = RustParser()
        file_info = FileInfo(
            relative_path="test.rs",
            absolute_path=Path("/fake/test.rs"),
            language="rust",
        )

        elements = parser.parse(code, file_info, "scope", "repo", "main")
        non_file = [e for e in elements if e.element_type != "file"]

        # mod inner should be extracted
        mod_elem = next(
            (e for e in non_file if e.name == "inner" and e.element_type == "class"),
            None,
        )
        assert mod_elem is not None, "Expected class 'inner' (mod block) not found"

        # struct Foo should be extracted
        struct_elem = next(
            (e for e in non_file if e.name == "Foo" and e.element_type == "class"
             and "impl" not in (e.decorators or [])),
            None,
        )
        assert struct_elem is not None, "Expected class 'Foo' (struct) not found"

        # impl Foo should be extracted
        impl_elem = next(
            (e for e in non_file if e.name == "Foo" and e.element_type == "class"
             and "impl" in (e.decorators or [])),
            None,
        )
        assert impl_elem is not None, "Expected class 'Foo' (impl block) not found"

        # method bar should be extracted
        bar_method = next(
            (e for e in non_file if e.name == "bar"),
            None,
        )
        assert bar_method is not None, "Expected method 'bar' not found"
