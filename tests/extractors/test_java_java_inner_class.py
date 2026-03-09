"""Auto-generated parser test for java_inner_class."""

import pytest
from pathlib import Path

from magaldi_core.code_parser import JavaParser
from magaldi_core.change_detection import FileInfo


class TestJavaInnerClass:
    """Tests for java_inner_class extraction."""

    def test_java_inner_class(self):
        code = 'public class Outer {\n    private String name;\n\n    public static class InnerStatic {\n        void doStuff() {}\n    }\n\n    interface InnerInterface {\n        void act();\n    }\n\n    enum InnerEnum {\n        A, B, C;\n    }\n}'

        parser = JavaParser()
        file_info = FileInfo(
            relative_path="test.java",
            absolute_path=Path("/fake/test.java"),
            language="java",
        )

        elements = parser.parse(code, file_info, "scope", "repo", "main")

        # Check for Outer
        java_inner_class_elem = next(
            (e for e in elements if e.name == "Outer" and e.element_type == "class"),
            None
        )
        assert java_inner_class_elem is not None, f"Expected class 'Outer' not found"

        # Check for InnerStatic
        java_inner_class_elem = next(
            (e for e in elements if e.name == "InnerStatic" and e.element_type == "class"),
            None
        )
        assert java_inner_class_elem is not None, f"Expected class 'InnerStatic' not found"

        # Check for InnerInterface
        java_inner_class_elem = next(
            (e for e in elements if e.name == "InnerInterface" and e.element_type == "interface"),
            None
        )
        assert java_inner_class_elem is not None, f"Expected interface 'InnerInterface' not found"

        # Check for InnerEnum
        java_inner_class_elem = next(
            (e for e in elements if e.name == "InnerEnum" and e.element_type == "enum"),
            None
        )
        assert java_inner_class_elem is not None, f"Expected enum 'InnerEnum' not found"
