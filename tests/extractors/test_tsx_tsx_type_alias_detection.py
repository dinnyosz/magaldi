"""Auto-generated parser test for tsx_type_alias_detection."""

import pytest
from pathlib import Path

from magaldi_core.code_parser import JavaScriptParser
from magaldi_core.change_detection import FileInfo


class TestTsxTypeAliasDetection:
    """Tests for tsx_type_alias_detection extraction."""

    def test_tsx_type_alias_detection(self):
        code = "type Status = 'pending' | 'active' | 'completed';\n\ntype UserWithStatus = {\n  name: string;\n  status: Status;\n};\n\ntype Callback = (value: number) => void;\n"

        parser = JavaScriptParser("tsx")
        file_info = FileInfo(
            relative_path="test.tsx",
            absolute_path=Path("/fake/test.tsx"),
            language="tsx",
        )

        elements = parser.parse(code, file_info, "scope", "repo", "main")

        # Check for Status
        status_elem = next(
            (e for e in elements if e.name == "Status" and e.element_type == "type_alias"),
            None
        )
        assert status_elem is not None, f"Expected type_alias 'Status' not found. Got: {[(e.name, e.element_type) for e in elements]}"

        # Check for UserWithStatus
        user_elem = next(
            (e for e in elements if e.name == "UserWithStatus" and e.element_type == "type_alias"),
            None
        )
        assert user_elem is not None, f"Expected type_alias 'UserWithStatus' not found. Got: {[(e.name, e.element_type) for e in elements]}"

        # Check for Callback
        callback_elem = next(
            (e for e in elements if e.name == "Callback" and e.element_type == "type_alias"),
            None
        )
        assert callback_elem is not None, f"Expected type_alias 'Callback' not found. Got: {[(e.name, e.element_type) for e in elements]}"
