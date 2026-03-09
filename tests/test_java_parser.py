"""Tests for Java language support (extractor + parser).

Covers:
- Class, interface, enum, record extraction
- Method and constructor extraction
- Import extraction (regular, static, wildcard)
- Call extraction (method calls, constructor calls, chained calls)
- Visibility detection (public, private, protected)
- Annotations as decorators
- Base classes (extends + implements)
- Thrown exceptions
- Modified attributes (this.field)
- Field/constant extraction
- Javadoc docstring extraction
"""

from pathlib import Path

import pytest

from magaldi_core.change_detection import FileInfo
from magaldi_core.code_parser import JavaParser
from magaldi_core.extractors.java import (
    JavaExtractor,
    extract_java_base_classes,
    extract_java_calls,
    extract_java_class_fields,
    extract_java_elements,
    extract_java_imports,
    extract_java_modified_attributes,
    extract_java_thrown_exceptions,
)
from magaldi_core.parsers.base import CodeElement  # noqa: F401
from magaldi_core.tree_sitter_manager import get_manager

# =============================================================================
# Helpers
# =============================================================================

def _parse_java(code: str) -> list[CodeElement]:
    """Parse Java code and return elements."""
    parser = JavaParser()
    file_info = FileInfo(
        relative_path="Test.java",
        absolute_path=Path("/fake/Test.java"),
        language="java",
    )
    return parser.parse(code, file_info, "scope", "repo", "main")


def _get_tree(code: str):
    """Parse Java code into a tree-sitter tree."""
    mgr = get_manager()
    return mgr.parse(code.encode("utf-8"), "java")


# =============================================================================
# Class extraction
# =============================================================================


class TestJavaClassExtraction:
    """Test class/interface/enum/record extraction."""

    def test_basic_class(self):
        elements = _parse_java("public class Foo {}")
        classes = [e for e in elements if e.element_type == "class"]
        assert len(classes) == 1
        assert classes[0].name == "Foo"

    def test_interface(self):
        elements = _parse_java("interface Bar { void run(); }")
        ifaces = [e for e in elements if e.element_type == "interface"]
        assert len(ifaces) == 1
        assert ifaces[0].name == "Bar"

    def test_enum(self):
        elements = _parse_java("enum Color { RED, GREEN, BLUE; }")
        enums = [e for e in elements if e.element_type == "enum"]
        assert len(enums) == 1
        assert enums[0].name == "Color"

    def test_record(self):
        elements = _parse_java("record Point(int x, int y) {}")
        records = [e for e in elements if e.element_type == "class"]
        assert len(records) == 1
        assert records[0].name == "Point"
        assert "record" in records[0].decorators

    def test_annotation_type(self):
        elements = _parse_java("@interface MyAnnotation { String value(); }")
        annots = [e for e in elements if e.element_type == "interface"]
        assert len(annots) == 1
        assert annots[0].name == "MyAnnotation"

    def test_class_with_extends_and_implements(self):
        code = "public class Dog extends Animal implements Runnable, Comparable<Dog> {}"
        elements = _parse_java(code)
        cls = [e for e in elements if e.element_type == "class"][0]
        assert cls.name == "Dog"
        assert "Animal" in cls.base_classes
        assert "Runnable" in cls.base_classes
        assert "Comparable" in cls.base_classes

    def test_interface_extends(self):
        code = "interface Sortable extends Comparable<String>, Serializable {}"
        elements = _parse_java(code)
        iface = [e for e in elements if e.element_type == "interface"][0]
        assert "Comparable" in iface.base_classes
        assert "Serializable" in iface.base_classes


# =============================================================================
# Method extraction
# =============================================================================


class TestJavaMethodExtraction:
    """Test method and constructor extraction."""

    def test_basic_method(self):
        code = """\
public class Foo {
    public String getName() {
        return "foo";
    }
}"""
        elements = _parse_java(code)
        methods = [e for e in elements if e.element_type == "method"]
        assert len(methods) == 1
        assert methods[0].name == "getName"
        assert methods[0].return_type == "String"

    def test_constructor(self):
        code = """\
public class Foo {
    public Foo(int x, String y) {
        this.x = x;
    }
}"""
        elements = _parse_java(code)
        constructors = [e for e in elements if e.element_type == "method" and e.name == "Foo"]
        assert len(constructors) == 1
        assert len(constructors[0].parameters) == 2
        assert constructors[0].parameters[0]["name"] == "x"
        assert constructors[0].parameters[0]["type"] == "int"
        assert constructors[0].parameters[1]["name"] == "y"
        assert constructors[0].parameters[1]["type"] == "String"

    def test_method_with_params_and_return_type(self):
        code = """\
public class Foo {
    public int add(int a, int b) {
        return a + b;
    }
}"""
        elements = _parse_java(code)
        method = [e for e in elements if e.name == "add"][0]
        assert method.return_type == "int"
        assert len(method.parameters) == 2
        assert method.signature == "int add(int a, int b)"

    def test_visibility_detection(self):
        code = """\
public class Foo {
    public void pub() {}
    private void priv() {}
    protected void prot() {}
    void pkg() {}
}"""
        elements = _parse_java(code)
        methods = {e.name: e for e in elements if e.element_type == "method"}
        assert methods["pub"].visibility == "public"
        assert methods["priv"].visibility == "private"
        assert methods["prot"].visibility == "protected"
        assert methods["pkg"].visibility == "package"  # no modifier = package-private

    def test_static_method(self):
        code = """\
public class Foo {
    public static int count() { return 0; }
}"""
        elements = _parse_java(code)
        method = [e for e in elements if e.name == "count"][0]
        assert "static" in method.decorators

    def test_abstract_method(self):
        code = """\
public abstract class Foo {
    public abstract void doSomething();
}"""
        elements = _parse_java(code)
        method = [e for e in elements if e.name == "doSomething"][0]
        assert "abstract" in method.decorators


# =============================================================================
# Import extraction
# =============================================================================


class TestJavaImportExtraction:
    """Test import statement extraction."""

    def test_regular_import(self):
        code = "import java.util.List;\nclass Foo {}"
        elements = _parse_java(code)
        file_elem = elements[0]
        imports = file_elem.imports
        assert len(imports) == 1
        assert imports[0].name == "List"
        assert imports[0].module == "java.util.List"

    def test_static_import(self):
        code = "import static java.util.Arrays.asList;\nclass Foo {}"
        elements = _parse_java(code)
        imports = elements[0].imports
        assert len(imports) == 1
        assert imports[0].name == "asList"
        assert imports[0].alias == "static"

    def test_wildcard_import(self):
        code = "import java.io.*;\nclass Foo {}"
        elements = _parse_java(code)
        imports = elements[0].imports
        assert len(imports) == 1
        assert imports[0].name == "*"
        # Module should be the path without .*
        assert "java.io" in imports[0].module

    def test_multiple_imports(self):
        code = """\
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
class Foo {}"""
        elements = _parse_java(code)
        imports = elements[0].imports
        assert len(imports) == 3


# =============================================================================
# Call extraction
# =============================================================================


class TestJavaCallExtraction:
    """Test method call extraction."""

    def test_simple_method_call(self):
        code = """\
public class Foo {
    void run() {
        doSomething();
    }
}"""
        elements = _parse_java(code)
        method = [e for e in elements if e.name == "run"][0]
        call_names = [c.name for c in method.calls]
        assert "doSomething" in call_names

    def test_method_call_with_receiver(self):
        code = """\
public class Foo {
    void run() {
        obj.process();
    }
}"""
        elements = _parse_java(code)
        method = [e for e in elements if e.name == "run"][0]
        calls = {c.name: c for c in method.calls}
        assert "process" in calls
        assert calls["process"].receiver == "obj"

    def test_this_method_call(self):
        code = """\
public class Foo {
    void run() {
        this.setup();
    }
}"""
        elements = _parse_java(code)
        method = [e for e in elements if e.name == "run"][0]
        calls = {c.name: c for c in method.calls}
        assert "setup" in calls
        assert calls["setup"].receiver == "this"

    def test_constructor_call(self):
        code = """\
public class Foo {
    void run() {
        Bar b = new Bar(1, 2);
    }
}"""
        elements = _parse_java(code)
        method = [e for e in elements if e.name == "run"][0]
        call_names = [c.name for c in method.calls]
        assert "Bar" in call_names

    def test_chained_calls(self):
        code = """\
public class Foo {
    void run() {
        items.stream().filter(x -> true).collect(Collectors.toList());
    }
}"""
        elements = _parse_java(code)
        method = [e for e in elements if e.name == "run"][0]
        call_names = [c.name for c in method.calls]
        assert "stream" in call_names
        assert "filter" in call_names
        assert "collect" in call_names

    def test_static_method_call(self):
        code = """\
public class Foo {
    void run() {
        List.of("a", "b");
    }
}"""
        elements = _parse_java(code)
        method = [e for e in elements if e.name == "run"][0]
        calls = {c.name: c for c in method.calls}
        assert "of" in calls
        assert calls["of"].receiver == "List"


# =============================================================================
# Annotation extraction
# =============================================================================


class TestJavaAnnotations:
    """Test annotation extraction as decorators."""

    def test_override_annotation(self):
        code = """\
public class Foo {
    @Override
    public String toString() { return "foo"; }
}"""
        elements = _parse_java(code)
        method = [e for e in elements if e.name == "toString"][0]
        assert "@Override" in method.decorators

    def test_class_annotation(self):
        code = """\
@Deprecated
public class Old {}"""
        elements = _parse_java(code)
        cls = [e for e in elements if e.element_type == "class"][0]
        assert "@Deprecated" in cls.decorators

    def test_multiple_annotations(self):
        code = """\
public class Foo {
    @Override
    @SuppressWarnings
    public void run() {}
}"""
        elements = _parse_java(code)
        method = [e for e in elements if e.name == "run"][0]
        assert "@Override" in method.decorators
        assert "@SuppressWarnings" in method.decorators


# =============================================================================
# Field and constant extraction
# =============================================================================


class TestJavaFieldExtraction:
    """Test field and constant extraction."""

    def test_static_final_constant(self):
        code = """\
public class Foo {
    public static final int MAX = 100;
}"""
        elements = _parse_java(code)
        constants = [e for e in elements if e.element_type == "constant"]
        assert len(constants) == 1
        assert constants[0].name == "MAX"

    def test_instance_field(self):
        code = """\
public class Foo {
    private String name;
}"""
        elements = _parse_java(code)
        fields = [e for e in elements if e.element_type == "variable"]
        assert len(fields) == 1
        assert fields[0].name == "name"
        assert fields[0].visibility == "private"

    def test_multiple_fields(self):
        code = """\
public class Foo {
    private int x;
    protected int y;
    public static final String Z = "z";
}"""
        elements = _parse_java(code)
        vars_ = [e for e in elements if e.element_type in ("variable", "constant")]
        names = {v.name for v in vars_}
        assert names == {"x", "y", "Z"}


# =============================================================================
# Exception extraction
# =============================================================================


class TestJavaExceptionExtraction:
    """Test throw statement exception type extraction."""

    def test_throw_new_exception(self):
        code = """\
public class Foo {
    void run() {
        throw new IllegalArgumentException("bad");
    }
}"""
        elements = _parse_java(code)
        method = [e for e in elements if e.name == "run"][0]
        assert method.exceptions_raised is not None
        assert "IllegalArgumentException" in method.exceptions_raised

    def test_multiple_throws(self):
        code = """\
public class Foo {
    void run() {
        if (true) throw new IOException("io");
        throw new RuntimeException("rt");
    }
}"""
        elements = _parse_java(code)
        method = [e for e in elements if e.name == "run"][0]
        assert "IOException" in method.exceptions_raised
        assert "RuntimeException" in method.exceptions_raised


# =============================================================================
# Modified attributes
# =============================================================================


class TestJavaModifiedAttributes:
    """Test this.field = ... detection."""

    def test_this_assignment(self):
        code = """\
public class Foo {
    private String name;
    public Foo(String name) {
        this.name = name;
    }
}"""
        elements = _parse_java(code)
        ctor = [e for e in elements if e.element_type == "method" and e.name == "Foo"][0]
        assert ctor.attributes_modified is not None
        assert "name" in ctor.attributes_modified

    def test_multiple_this_assignments(self):
        code = """\
public class Foo {
    void init() {
        this.x = 1;
        this.y = 2;
    }
}"""
        elements = _parse_java(code)
        method = [e for e in elements if e.name == "init"][0]
        assert "x" in method.attributes_modified
        assert "y" in method.attributes_modified


# =============================================================================
# Docstring (Javadoc) extraction
# =============================================================================


class TestJavaDocstring:
    """Test Javadoc extraction."""

    def test_class_javadoc(self):
        code = """\
/**
 * A simple class.
 */
public class Foo {}"""
        elements = _parse_java(code)
        cls = [e for e in elements if e.element_type == "class"][0]
        assert cls.docstring is not None
        assert "A simple class." in cls.docstring

    def test_method_javadoc(self):
        code = """\
public class Foo {
    /**
     * Do the thing.
     */
    public void doIt() {}
}"""
        elements = _parse_java(code)
        method = [e for e in elements if e.name == "doIt"][0]
        assert method.docstring is not None
        assert "Do the thing." in method.docstring

    def test_javadoc_with_annotation_between(self):
        """Javadoc before @Override should still be found."""
        code = """\
public class Foo {
    /**
     * Overridden method.
     */
    @Override
    public String toString() { return ""; }
}"""
        elements = _parse_java(code)
        method = [e for e in elements if e.name == "toString"][0]
        assert method.docstring is not None
        assert "Overridden method." in method.docstring


# =============================================================================
# Hierarchy
# =============================================================================


class TestJavaHierarchy:
    """Test parent-child hierarchy."""

    def test_method_has_class_parent(self):
        code = """\
public class Foo {
    void run() {}
}"""
        elements = _parse_java(code)
        method = [e for e in elements if e.name == "run"][0]
        cls = [e for e in elements if e.name == "Foo"][0]
        assert method.parent_id == cls.element_id

    def test_field_has_class_parent(self):
        code = """\
public class Foo {
    private int x;
}"""
        elements = _parse_java(code)
        field = [e for e in elements if e.name == "x"][0]
        cls = [e for e in elements if e.name == "Foo"][0]
        assert field.parent_id == cls.element_id

    def test_file_is_root(self):
        elements = _parse_java("public class Foo {}")
        file_elem = [e for e in elements if e.element_type == "file"][0]
        assert file_elem.parent_id is None
        assert file_elem.level == 0


# =============================================================================
# Fixture-based integration test
# =============================================================================


class TestJavaFixture:
    """Integration tests using the teatro_stage.java fixture."""

    @pytest.fixture
    def elements(self):
        parser = JavaParser()
        fixture = Path(__file__).parent / "fixtures" / "languages" / "teatro_stage.java"
        content = fixture.read_text()
        file_info = FileInfo(
            relative_path="teatro_stage.java",
            absolute_path=fixture,
            language="java",
        )
        return parser.parse(content, file_info, "scope", "repo", "main")

    def test_element_count(self, elements):
        """Fixture should produce a reasonable number of elements."""
        assert len(elements) >= 30

    def test_all_expected_classes(self, elements):
        class_names = {e.name for e in elements if e.element_type in ("class", "interface", "enum")}
        assert "Stage" in class_names
        assert "Performable" in class_names
        assert "Lightable" in class_names
        assert "ActType" in class_names
        assert "Ticket" in class_names
        assert "BackstageAccess" in class_names
        assert "StageException" in class_names
        assert "Theater" in class_names

    def test_stage_base_classes(self, elements):
        stage = [e for e in elements if e.name == "Stage" and e.element_type == "class"][0]
        assert stage.base_classes is not None
        assert "Performable" in stage.base_classes
        assert "Lightable" in stage.base_classes

    def test_stage_methods(self, elements):
        stage_methods = [
            e for e in elements
            if e.element_type == "method"
            and e.parent_id
            and "Stage" in e.parent_id
            and "StageException" not in e.parent_id
        ]
        method_names = {m.name for m in stage_methods}
        assert "Stage" in method_names  # constructor
        assert "perform" in method_names
        assert "getCapacity" in method_names
        assert "getStageName" in method_names
        assert "getPerformers" in method_names

    def test_stage_constants(self, elements):
        constants = [
            e for e in elements
            if e.element_type == "constant"
            and e.parent_id
            and "Stage" in e.parent_id
        ]
        names = {c.name for c in constants}
        assert "MAX_CAPACITY" in names
        assert "THEATER_NAME" in names

    def test_enum_methods(self, elements):
        enum_methods = [
            e for e in elements
            if e.element_type == "method"
            and e.parent_id
            and "ActType" in e.parent_id
        ]
        names = {m.name for m in enum_methods}
        assert "ActType" in names  # constructor
        assert "getDisplayName" in names

    def test_record_method(self, elements):
        record_methods = [
            e for e in elements
            if e.element_type == "method"
            and e.parent_id
            and "Ticket" in e.parent_id
        ]
        assert any(m.name == "display" for m in record_methods)

    def test_perform_has_calls(self, elements):
        perform = [
            e for e in elements
            if e.name == "perform"
            and e.element_type == "method"
            and e.calls
        ][0]
        call_names = {c.name for c in perform.calls}
        assert "println" in call_names
        assert "StageException" in call_names  # new StageException()

    def test_perform_throws_exception(self, elements):
        perform = [
            e for e in elements
            if e.name == "perform"
            and e.element_type == "method"
            and e.exceptions_raised
        ][0]
        assert "StageException" in perform.exceptions_raised

    def test_imports(self, elements):
        file_elem = elements[0]
        assert len(file_elem.imports) >= 5
        import_names = {i.name for i in file_elem.imports}
        assert "List" in import_names
        assert "ArrayList" in import_names

    def test_language_is_java(self, elements):
        for e in elements:
            assert e.language == "java"


# =============================================================================
# Extractor unit tests
# =============================================================================


class TestJavaExtractorUnit:
    """Unit tests for standalone extraction functions."""

    def test_extractor_language(self):
        ext = JavaExtractor()
        assert ext.language == "java"

    def test_extract_elements_from_tree(self):
        code = "class Foo {} interface Bar {}"
        tree = _get_tree(code)
        elements = extract_java_elements(tree, code.split("\n"))
        names = {e.name for e in elements}
        assert "Foo" in names
        assert "Bar" in names

    def test_extract_imports(self):
        code = "import java.util.List;\nimport java.util.Map;\nclass Foo {}"
        tree = _get_tree(code)
        imports = extract_java_imports(tree, code.split("\n"))
        assert len(imports) == 2
        assert imports[0].name == "List"
        assert imports[1].name == "Map"

    def test_extract_class_fields(self):
        code = """\
public class Foo {
    private int x;
    public String y;
}"""
        tree = _get_tree(code)
        root = tree.root_node
        class_node = root.children[0]
        fields = extract_java_class_fields(class_node)
        names = {f["name"] for f in fields}
        assert "x" in names
        assert "y" in names

    def test_extract_base_classes(self):
        code = "class Foo extends Bar implements Baz {}"
        tree = _get_tree(code)
        class_node = tree.root_node.children[0]
        bases = extract_java_base_classes(class_node)
        assert "Bar" in bases
        assert "Baz" in bases

    def test_extract_calls_from_method(self):
        code = """\
public class Foo {
    void run() {
        bar();
        obj.baz();
        new Qux();
    }
}"""
        tree = _get_tree(code)
        class_node = tree.root_node.children[0]
        # Find the method node
        body = None
        for c in class_node.children:
            if c.type == "class_body":
                body = c
                break
        method_node = None
        for c in body.children:
            if c.type == "method_declaration":
                method_node = c
                break

        calls = extract_java_calls(method_node)
        call_names = {c.name for c in calls}
        assert "bar" in call_names
        assert "baz" in call_names
        assert "Qux" in call_names

    def test_extract_thrown_exceptions(self):
        code = """\
public class Foo {
    void run() {
        throw new IllegalStateException("oops");
    }
}"""
        tree = _get_tree(code)
        class_node = tree.root_node.children[0]
        body = [c for c in class_node.children if c.type == "class_body"][0]
        method = [c for c in body.children if c.type == "method_declaration"][0]
        exceptions = extract_java_thrown_exceptions(method)
        assert "IllegalStateException" in exceptions

    def test_extract_modified_attributes(self):
        code = """\
public class Foo {
    void init() {
        this.x = 1;
        this.y = 2;
        z = 3;
    }
}"""
        tree = _get_tree(code)
        class_node = tree.root_node.children[0]
        body = [c for c in class_node.children if c.type == "class_body"][0]
        method = [c for c in body.children if c.type == "method_declaration"][0]
        modified = extract_java_modified_attributes(method)
        assert "x" in modified
        assert "y" in modified
        assert "z" not in modified  # Not this.z
