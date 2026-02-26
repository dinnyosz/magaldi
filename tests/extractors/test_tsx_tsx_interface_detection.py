"""Auto-generated parser test for tsx_interface_detection."""

from pathlib import Path

from magaldi_core.change_detection import FileInfo
from magaldi_core.code_parser import JavaScriptParser


class TestTsxInterfaceDetection:
    """Tests for tsx_interface_detection extraction."""

    def test_tsx_interface_detection(self):
        code = 'interface Props {\n  name: string;\n  age: number;\n}\n\ninterface ExtendedProps extends Props {\n  email: string;\n}\n'

        parser = JavaScriptParser("tsx")
        file_info = FileInfo(
            relative_path="test.tsx",
            absolute_path=Path("/fake/test.tsx"),
            language="tsx",
        )

        elements = parser.parse(code, file_info, "scope", "repo", "main")

        # Check for Props
        props_elem = next(
            (e for e in elements if e.name == "Props" and e.element_type == "interface"),
            None
        )
        assert props_elem is not None, f"Expected interface 'Props' not found. Got: {[(e.name, e.element_type) for e in elements]}"

        # Check for ExtendedProps
        extended_elem = next(
            (e for e in elements if e.name == "ExtendedProps" and e.element_type == "interface"),
            None
        )
        assert extended_elem is not None, f"Expected interface 'ExtendedProps' not found. Got: {[(e.name, e.element_type) for e in elements]}"
