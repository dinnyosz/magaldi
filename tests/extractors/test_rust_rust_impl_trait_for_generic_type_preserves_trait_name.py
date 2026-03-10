"""Auto-generated parser test for rust_impl_trait_for_generic_type_preserves_trait_name."""

import pytest
from pathlib import Path

from magaldi_core.code_parser import RustParser
from magaldi_core.change_detection import FileInfo


class TestRustImplTraitForGenericTypePreservesTraitName:
    """Tests for rust_impl_trait_for_generic_type_preserves_trait_name extraction."""

    def test_rust_impl_trait_for_generic_type_preserves_trait_name(self):
        code = 'trait Display {\n    fn fmt(&self) -> String;\n}\n\nimpl Display for Vec<String> {\n    fn fmt(&self) -> String {\n        String::new()\n    }\n}\n\nimpl Display for HashMap<String, i32> {\n    fn fmt(&self) -> String {\n        String::new()\n    }\n}'

        parser = RustParser()
        file_info = FileInfo(
            relative_path="test.rs",
            absolute_path=Path("/fake/test.rs"),
            language="rust",
        )

        elements = parser.parse(code, file_info, "scope", "repo", "main")

        # Check for Display
        rust_impl_trait_for_generic_type_preserves_trait_name_elem = next(
            (e for e in elements if e.name == "Display" and e.element_type == "trait"),
            None
        )
        assert rust_impl_trait_for_generic_type_preserves_trait_name_elem is not None, f"Expected trait 'Display' not found"

        # Check for Vec::Display
        rust_impl_trait_for_generic_type_preserves_trait_name_elem = next(
            (e for e in elements if e.name == "Vec::Display" and e.element_type == "class"),
            None
        )
        assert rust_impl_trait_for_generic_type_preserves_trait_name_elem is not None, f"Expected class 'Vec::Display' not found"

        # Check for HashMap::Display
        rust_impl_trait_for_generic_type_preserves_trait_name_elem = next(
            (e for e in elements if e.name == "HashMap::Display" and e.element_type == "class"),
            None
        )
        assert rust_impl_trait_for_generic_type_preserves_trait_name_elem is not None, f"Expected class 'HashMap::Display' not found"
