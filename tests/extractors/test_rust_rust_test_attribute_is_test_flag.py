"""Auto-generated parser test for rust_test_attribute_is_test_flag."""

from pathlib import Path

from magaldi_core.code_parser import RustParser
from magaldi_core.change_detection import FileInfo


class TestRustTestAttributeIsTestFlag:
    """Tests for #[test] attribute captured in decorators and is_test flag."""

    def test_test_attribute_sets_decorators_and_is_test(self):
        code = (
            '#[cfg(test)]\n'
            'mod tests {\n'
            '    use super::*;\n'
            '\n'
            '    #[test]\n'
            '    fn test_addition() {\n'
            '        assert_eq!(2 + 2, 4);\n'
            '    }\n'
            '\n'
            '    fn helper_function() -> i32 {\n'
            '        42\n'
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

        # test_addition should have #[test] in decorators
        test_elem = next(
            (e for e in elements if e.name == "test_addition"),
            None,
        )
        assert test_elem is not None, "Expected function 'test_addition' not found"
        assert "test" in test_elem.decorators, (
            f"Expected 'test' in decorators, got {test_elem.decorators}"
        )
        assert test_elem.is_test is True, (
            f"Expected is_test=True for test_addition, got {test_elem.is_test}"
        )

        # helper_function should NOT have is_test
        helper_elem = next(
            (e for e in elements if e.name == "helper_function"),
            None,
        )
        assert helper_elem is not None, "Expected function 'helper_function' not found"
        assert helper_elem.is_test is False, (
            f"Expected is_test=False for helper_function, got {helper_elem.is_test}"
        )

    def test_test_attribute_on_top_level_function(self):
        """#[test] attribute on a top-level function (outside mod)."""
        code = (
            '#[test]\n'
            'fn test_it_works() {\n'
            '    assert!(true);\n'
            '}\n'
        )

        parser = RustParser()
        file_info = FileInfo(
            relative_path="test.rs",
            absolute_path=Path("/fake/test.rs"),
            language="rust",
        )

        elements = parser.parse(code, file_info, "scope", "repo", "main")

        test_elem = next(
            (e for e in elements if e.name == "test_it_works"),
            None,
        )
        assert test_elem is not None, "Expected function 'test_it_works' not found"
        assert "test" in test_elem.decorators, (
            f"Expected 'test' in decorators, got {test_elem.decorators}"
        )
        assert test_elem.is_test is True, (
            f"Expected is_test=True, got {test_elem.is_test}"
        )

    def test_impl_method_with_test_attribute(self):
        """#[test] on a method inside an impl block."""
        code = (
            'impl MyStruct {\n'
            '    #[test]\n'
            '    fn test_method(&self) {\n'
            '        assert!(true);\n'
            '    }\n'
            '\n'
            '    fn regular_method(&self) {}\n'
            '}\n'
        )

        parser = RustParser()
        file_info = FileInfo(
            relative_path="test.rs",
            absolute_path=Path("/fake/test.rs"),
            language="rust",
        )

        elements = parser.parse(code, file_info, "scope", "repo", "main")

        test_method = next(
            (e for e in elements if e.name == "test_method"),
            None,
        )
        assert test_method is not None, "Expected method 'test_method' not found"
        assert "test" in test_method.decorators, (
            f"Expected 'test' in decorators, got {test_method.decorators}"
        )
        assert test_method.is_test is True

        regular_method = next(
            (e for e in elements if e.name == "regular_method"),
            None,
        )
        assert regular_method is not None, "Expected method 'regular_method' not found"
        assert regular_method.is_test is False
