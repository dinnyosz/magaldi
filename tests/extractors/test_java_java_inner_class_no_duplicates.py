"""Auto-generated parser test for java_inner_class_no_duplicates."""

import pytest
from pathlib import Path

from magaldi_core.code_parser import JavaParser
from magaldi_core.change_detection import FileInfo


class TestJavaInnerClassNoDuplicates:
    """Tests for java_inner_class_no_duplicates extraction."""

    def test_java_inner_class_no_duplicates(self):
        code = 'public class Outer {\n    private int outerField;\n\n    public void outerMethod() {\n        System.out.println("outer");\n    }\n\n    public static class Inner {\n        private int innerField;\n\n        public void innerMethod() {\n            System.out.println("inner");\n        }\n    }\n\n    public interface InnerInterface {\n        void doSomething();\n    }\n}'

        parser = JavaParser()
        file_info = FileInfo(
            relative_path="test.java",
            absolute_path=Path("/fake/test.java"),
            language="java",
        )

        elements = parser.parse(code, file_info, "scope", "repo", "main")

        # Check for Outer
        java_inner_class_no_duplicates_elem = next(
            (e for e in elements if e.name == "Outer" and e.element_type == "class"),
            None
        )
        assert java_inner_class_no_duplicates_elem is not None, f"Expected class 'Outer' not found"

        # Check for outerField
        java_inner_class_no_duplicates_elem = next(
            (e for e in elements if e.name == "outerField" and e.element_type == "variable"),
            None
        )
        assert java_inner_class_no_duplicates_elem is not None, f"Expected variable 'outerField' not found"

        # Check for outerMethod
        java_inner_class_no_duplicates_elem = next(
            (e for e in elements if e.name == "outerMethod" and e.element_type == "method"),
            None
        )
        assert java_inner_class_no_duplicates_elem is not None, f"Expected method 'outerMethod' not found"

        # Check for Inner
        java_inner_class_no_duplicates_elem = next(
            (e for e in elements if e.name == "Inner" and e.element_type == "class"),
            None
        )
        assert java_inner_class_no_duplicates_elem is not None, f"Expected class 'Inner' not found"

        # Check for innerField
        java_inner_class_no_duplicates_elem = next(
            (e for e in elements if e.name == "innerField" and e.element_type == "variable"),
            None
        )
        assert java_inner_class_no_duplicates_elem is not None, f"Expected variable 'innerField' not found"

        # Check for innerMethod
        java_inner_class_no_duplicates_elem = next(
            (e for e in elements if e.name == "innerMethod" and e.element_type == "method"),
            None
        )
        assert java_inner_class_no_duplicates_elem is not None, f"Expected method 'innerMethod' not found"

        # Check for InnerInterface
        java_inner_class_no_duplicates_elem = next(
            (e for e in elements if e.name == "InnerInterface" and e.element_type == "interface"),
            None
        )
        assert java_inner_class_no_duplicates_elem is not None, f"Expected interface 'InnerInterface' not found"

        # Check for doSomething
        java_inner_class_no_duplicates_elem = next(
            (e for e in elements if e.name == "doSomething" and e.element_type == "method"),
            None
        )
        assert java_inner_class_no_duplicates_elem is not None, f"Expected method 'doSomething' not found"

        # Check element count
        assert len(elements) == 9, f"Expected 9 elements, got {len(elements)}"
