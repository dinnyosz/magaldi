"""Auto-generated parser test for java_enum_constant_class_body_methods."""

from pathlib import Path

from magaldi_core.code_parser import JavaParser
from magaldi_core.change_detection import FileInfo


class TestJavaEnumConstantClassBodyMethods:
    """Tests for enum constants with anonymous class body methods."""

    def test_java_enum_constant_class_body_methods(self):
        code = (
            "public enum FieldNamingPolicy {\n"
            "    IDENTITY {\n"
            "        @Override\n"
            "        public String translateName(Field f) {\n"
            "            return f.getName();\n"
            "        }\n"
            "    },\n"
            "    UPPER_CAMEL_CASE {\n"
            "        @Override\n"
            "        public String translateName(Field f) {\n"
            "            return upperCaseFirstLetter(f.getName());\n"
            "        }\n"
            "    };\n"
            "\n"
            "    public abstract String translateName(Field f);\n"
            "}"
        )

        parser = JavaParser()
        file_info = FileInfo(
            relative_path="test.java",
            absolute_path=Path("/fake/test.java"),
            language="java",
        )

        elements = parser.parse(code, file_info, "scope", "repo", "main")

        # Collect all translateName methods
        translate_methods = [
            e for e in elements
            if e.name == "translateName" and e.element_type == "method"
        ]

        # Should have 3 translateName methods:
        # 1) IDENTITY's override (line 4)
        # 2) UPPER_CAMEL_CASE's override (line 10)
        # 3) Abstract declaration in the enum body (line 15)
        assert len(translate_methods) == 3, (
            f"Expected 3 translateName methods (2 in enum constants + 1 abstract), "
            f"got {len(translate_methods)}. "
            f"Lines: {[m.line_start for m in translate_methods]}"
        )

        # Verify we got the enum and its constants
        enum_elem = next(
            (e for e in elements if e.name == "FieldNamingPolicy" and e.element_type == "enum"),
            None,
        )
        assert enum_elem is not None, "Expected enum 'FieldNamingPolicy' not found"

        constants = [
            e for e in elements if e.element_type == "constant"
        ]
        constant_names = {c.name for c in constants}
        assert "IDENTITY" in constant_names, "Expected constant 'IDENTITY' not found"
        assert "UPPER_CAMEL_CASE" in constant_names, "Expected constant 'UPPER_CAMEL_CASE' not found"
