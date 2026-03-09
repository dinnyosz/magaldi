"""Auto-generated parser test for java_enum_constants."""

from pathlib import Path

from magaldi_core.change_detection import FileInfo
from magaldi_core.code_parser import JavaParser


class TestJavaEnumConstants:
    """Tests for java_enum_constants extraction."""

    def test_java_enum_constants(self):
        code = 'public enum JsonToken {\n    BEGIN_ARRAY,\n    END_ARRAY,\n    BEGIN_OBJECT,\n    END_OBJECT;\n\n    public boolean isValue() {\n        return true;\n    }\n}'

        parser = JavaParser()
        file_info = FileInfo(
            relative_path="test.java",
            absolute_path=Path("/fake/test.java"),
            language="java",
        )

        elements = parser.parse(code, file_info, "scope", "repo", "main")

        # Check for JsonToken
        java_enum_constants_elem = next(
            (e for e in elements if e.name == "JsonToken" and e.element_type == "enum"),
            None
        )
        assert java_enum_constants_elem is not None, "Expected enum 'JsonToken' not found"

        # Check for BEGIN_ARRAY
        java_enum_constants_elem = next(
            (e for e in elements if e.name == "BEGIN_ARRAY" and e.element_type == "constant"),
            None
        )
        assert java_enum_constants_elem is not None, "Expected constant 'BEGIN_ARRAY' not found"

        # Check for END_ARRAY
        java_enum_constants_elem = next(
            (e for e in elements if e.name == "END_ARRAY" and e.element_type == "constant"),
            None
        )
        assert java_enum_constants_elem is not None, "Expected constant 'END_ARRAY' not found"

        # Check for BEGIN_OBJECT
        java_enum_constants_elem = next(
            (e for e in elements if e.name == "BEGIN_OBJECT" and e.element_type == "constant"),
            None
        )
        assert java_enum_constants_elem is not None, "Expected constant 'BEGIN_OBJECT' not found"

        # Check for END_OBJECT
        java_enum_constants_elem = next(
            (e for e in elements if e.name == "END_OBJECT" and e.element_type == "constant"),
            None
        )
        assert java_enum_constants_elem is not None, "Expected constant 'END_OBJECT' not found"

        # Check for isValue
        java_enum_constants_elem = next(
            (e for e in elements if e.name == "isValue" and e.element_type == "method"),
            None
        )
        assert java_enum_constants_elem is not None, "Expected method 'isValue' not found"
