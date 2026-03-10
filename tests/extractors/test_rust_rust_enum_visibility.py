"""Auto-generated parser test for rust_enum_visibility."""

import pytest
from pathlib import Path

from magaldi_core.code_parser import RustParser
from magaldi_core.change_detection import FileInfo


class TestRustEnumVisibility:
    """Tests for rust_enum_visibility extraction."""

    def test_rust_enum_visibility(self):
        code = 'enum PrivateColor {\n    Red,\n    Green,\n    Blue,\n}\n\npub enum PublicShape {\n    Circle,\n    Square,\n}\n\npub(crate) enum CrateVisible {\n    Alpha,\n    Beta,\n}\n\npub(super) enum SuperVisible {\n    One,\n    Two,\n}\n'

        parser = RustParser()
        file_info = FileInfo(
            relative_path="test.rs",
            absolute_path=Path("/fake/test.rs"),
            language="rust",
        )

        elements = parser.parse(code, file_info, "scope", "repo", "main")

        # Check for PrivateColor
        rust_enum_visibility_elem = next(
            (e for e in elements if e.name == "PrivateColor" and e.element_type == "enum"),
            None
        )
        assert rust_enum_visibility_elem is not None, f"Expected enum 'PrivateColor' not found"

        # Check for PublicShape
        rust_enum_visibility_elem = next(
            (e for e in elements if e.name == "PublicShape" and e.element_type == "enum"),
            None
        )
        assert rust_enum_visibility_elem is not None, f"Expected enum 'PublicShape' not found"

        # Check for CrateVisible
        rust_enum_visibility_elem = next(
            (e for e in elements if e.name == "CrateVisible" and e.element_type == "enum"),
            None
        )
        assert rust_enum_visibility_elem is not None, f"Expected enum 'CrateVisible' not found"

        # Check for SuperVisible
        rust_enum_visibility_elem = next(
            (e for e in elements if e.name == "SuperVisible" and e.element_type == "enum"),
            None
        )
        assert rust_enum_visibility_elem is not None, f"Expected enum 'SuperVisible' not found"
