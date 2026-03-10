"""Auto-generated parser test for java_inner_class_extraction."""

from pathlib import Path

from magaldi_core.code_parser import JavaParser
from magaldi_core.change_detection import FileInfo


def _parse(code: str) -> list:
    parser = JavaParser()
    file_info = FileInfo(
        relative_path="test.java",
        absolute_path=Path("/fake/test.java"),
        language="java",
    )
    return parser.parse(code, file_info, "scope", "repo", "main")


class TestJavaInnerClassExtraction:
    """Tests for java_inner_class_extraction extraction."""

    def test_inner_class_and_methods_extracted(self):
        code = (
            "public class Gson {\n"
            "    private TypeAdapter delegate;\n"
            "\n"
            "    public String toJson(Object src) {\n"
            "        return src.toString();\n"
            "    }\n"
            "\n"
            "    static class FutureTypeAdapter extends TypeAdapter {\n"
            "        private TypeAdapter delegate;\n"
            "\n"
            "        public void setDelegate(TypeAdapter typeAdapter) {\n"
            "            delegate = typeAdapter;\n"
            "        }\n"
            "\n"
            "        public Object read() {\n"
            "            return delegate.read();\n"
            "        }\n"
            "\n"
            "        public void write(Object value) {\n"
            "            delegate.write(value);\n"
            "        }\n"
            "    }\n"
            "}"
        )

        elements = _parse(code)

        # Outer class exists at level 1
        gson = next(
            (e for e in elements if e.name == "Gson" and e.element_type == "class"),
            None,
        )
        assert gson is not None, "Expected class 'Gson' not found"
        assert gson.level == 1

        # Inner class exists at level 2 with correct parent
        future = next(
            (e for e in elements if e.name == "FutureTypeAdapter" and e.element_type == "class"),
            None,
        )
        assert future is not None, "Expected class 'FutureTypeAdapter' not found"
        assert future.level == 2, f"Inner class should be level 2, got {future.level}"
        assert future.parent_id == gson.element_id

        # Outer method is child of outer class
        to_json = next(
            (e for e in elements if e.name == "toJson" and e.element_type == "method"),
            None,
        )
        assert to_json is not None, "Expected method 'toJson' not found"
        assert to_json.parent_id == gson.element_id

        # Inner-class methods are children of the inner class
        for method_name in ("setDelegate", "read", "write"):
            method = next(
                (e for e in elements if e.name == method_name and e.element_type == "method"),
                None,
            )
            assert method is not None, f"Expected method '{method_name}' not found"
            assert method.parent_id == future.element_id, (
                f"Method '{method_name}' should be child of FutureTypeAdapter"
            )

    def test_deeply_nested_inner_class(self):
        """LinkedTreeMap.EntrySet style double-nesting."""
        code = (
            "public class LinkedTreeMap {\n"
            "    public Object get(Object key) { return null; }\n"
            "\n"
            "    static class Node {\n"
            "        Object key;\n"
            "        public Object getKey() { return key; }\n"
            "    }\n"
            "\n"
            "    class EntrySet {\n"
            "        public int size() { return 0; }\n"
            "\n"
            "        class EntryIterator {\n"
            "            public Object next() { return null; }\n"
            "        }\n"
            "    }\n"
            "}"
        )

        elements = _parse(code)

        # Top-level class
        ltm = next(
            (e for e in elements if e.name == "LinkedTreeMap" and e.element_type == "class"),
            None,
        )
        assert ltm is not None

        # First-level nested classes
        node = next(
            (e for e in elements if e.name == "Node" and e.element_type == "class"),
            None,
        )
        assert node is not None, "Expected inner class 'Node' not found"
        assert node.parent_id == ltm.element_id
        assert node.level == 2, f"Node should be level 2, got {node.level}"

        entry_set = next(
            (e for e in elements if e.name == "EntrySet" and e.element_type == "class"),
            None,
        )
        assert entry_set is not None, "Expected inner class 'EntrySet' not found"
        assert entry_set.parent_id == ltm.element_id
        assert entry_set.level == 2, f"EntrySet should be level 2, got {entry_set.level}"

        # Second-level nested class
        entry_iter = next(
            (e for e in elements if e.name == "EntryIterator" and e.element_type == "class"),
            None,
        )
        assert entry_iter is not None, "Expected inner class 'EntryIterator' not found"
        assert entry_iter.parent_id == entry_set.element_id
        assert entry_iter.level == 2, f"EntryIterator should be level 2, got {entry_iter.level}"

        # Methods of deeply nested class
        next_method = next(
            (e for e in elements if e.name == "next" and e.element_type == "method"),
            None,
        )
        assert next_method is not None, "Expected method 'next' inside EntryIterator"
        assert next_method.parent_id == entry_iter.element_id

    def test_interface_inside_class(self):
        """Inner interface extraction."""
        code = (
            "public class Streams {\n"
            "    public interface AppendableWriter {\n"
            "        void write(String s);\n"
            "        void flush();\n"
            "    }\n"
            "}"
        )

        elements = _parse(code)

        streams = next(
            (e for e in elements if e.name == "Streams" and e.element_type == "class"),
            None,
        )
        assert streams is not None

        writer = next(
            (e for e in elements if e.name == "AppendableWriter" and e.element_type == "interface"),
            None,
        )
        assert writer is not None, "Expected inner interface 'AppendableWriter' not found"
        assert writer.parent_id == streams.element_id
        assert writer.level == 2

    def test_enum_inside_class(self):
        """Inner enum extraction."""
        code = (
            "public class JsonToken {\n"
            "    public enum Kind {\n"
            "        BEGIN_OBJECT,\n"
            "        END_OBJECT;\n"
            "\n"
            "        public boolean isValue() { return false; }\n"
            "    }\n"
            "}"
        )

        elements = _parse(code)

        token = next(
            (e for e in elements if e.name == "JsonToken" and e.element_type == "class"),
            None,
        )
        assert token is not None

        kind = next(
            (e for e in elements if e.name == "Kind" and e.element_type == "enum"),
            None,
        )
        assert kind is not None, "Expected inner enum 'Kind' not found"
        assert kind.parent_id == token.element_id
        assert kind.level == 2
