"""Auto-generated parser test for java_inner_classes."""

from pathlib import Path

from magaldi_core.change_detection import FileInfo
from magaldi_core.code_parser import JavaParser


class TestJavaInnerClasses:
    """Tests for java_inner_classes extraction."""

    def test_java_inner_classes(self):
        code = 'public class Outer {\n    private int x;\n\n    static class Inner {\n        private String name;\n\n        public void doStuff() {}\n    }\n\n    interface Callback<R> {\n        R execute();\n    }\n\n    public void outerMethod() {}\n}'

        parser = JavaParser()
        file_info = FileInfo(
            relative_path="test.java",
            absolute_path=Path("/fake/test.java"),
            language="java",
        )

        elements = parser.parse(code, file_info, "scope", "repo", "main")

        # Check for Outer
        java_inner_classes_elem = next(
            (e for e in elements if e.name == "Outer" and e.element_type == "class"),
            None
        )
        assert java_inner_classes_elem is not None, "Expected class 'Outer' not found"

        # Check for outerMethod
        java_inner_classes_elem = next(
            (e for e in elements if e.name == "outerMethod" and e.element_type == "method"),
            None
        )
        assert java_inner_classes_elem is not None, "Expected method 'outerMethod' not found"

        # Check for x
        java_inner_classes_elem = next(
            (e for e in elements if e.name == "x" and e.element_type == "variable"),
            None
        )
        assert java_inner_classes_elem is not None, "Expected variable 'x' not found"

        # Check for Inner
        java_inner_classes_elem = next(
            (e for e in elements if e.name == "Inner" and e.element_type == "class"),
            None
        )
        assert java_inner_classes_elem is not None, "Expected class 'Inner' not found"

        # Check for doStuff
        java_inner_classes_elem = next(
            (e for e in elements if e.name == "doStuff" and e.element_type == "method"),
            None
        )
        assert java_inner_classes_elem is not None, "Expected method 'doStuff' not found"

        # Check for name
        java_inner_classes_elem = next(
            (e for e in elements if e.name == "name" and e.element_type == "variable"),
            None
        )
        assert java_inner_classes_elem is not None, "Expected variable 'name' not found"

        # Check for Callback
        java_inner_classes_elem = next(
            (e for e in elements if e.name == "Callback" and e.element_type == "interface"),
            None
        )
        assert java_inner_classes_elem is not None, "Expected interface 'Callback' not found"

        # Check for execute
        java_inner_classes_elem = next(
            (e for e in elements if e.name == "execute" and e.element_type == "method"),
            None
        )
        assert java_inner_classes_elem is not None, "Expected method 'execute' not found"
