"""Auto-generated parser test for java_annotation_members."""

from pathlib import Path

from magaldi_core.change_detection import FileInfo
from magaldi_core.code_parser import JavaParser


class TestJavaAnnotationMembers:
    """Tests for java_annotation_members extraction."""

    def test_java_annotation_members(self):
        code = 'public @interface JsonAdapter {\n    Class<?> value();\n    boolean nullSafe() default true;\n}'

        parser = JavaParser()
        file_info = FileInfo(
            relative_path="test.java",
            absolute_path=Path("/fake/test.java"),
            language="java",
        )

        elements = parser.parse(code, file_info, "scope", "repo", "main")

        # Check for JsonAdapter
        java_annotation_members_elem = next(
            (e for e in elements if e.name == "JsonAdapter" and e.element_type == "interface"),
            None
        )
        assert java_annotation_members_elem is not None, "Expected interface 'JsonAdapter' not found"
        assert "public" in java_annotation_members_elem.decorators, "Missing decorator public"
        assert "@interface" in java_annotation_members_elem.decorators, "Missing decorator @interface"

        # Check for value
        java_annotation_members_elem = next(
            (e for e in elements if e.name == "value" and e.element_type == "method"),
            None
        )
        assert java_annotation_members_elem is not None, "Expected method 'value' not found"

        # Check for nullSafe
        java_annotation_members_elem = next(
            (e for e in elements if e.name == "nullSafe" and e.element_type == "method"),
            None
        )
        assert java_annotation_members_elem is not None, "Expected method 'nullSafe' not found"
