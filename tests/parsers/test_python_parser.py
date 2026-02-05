"""Tests for the Python parser."""

from pathlib import Path

import pytest

from magaldi_core.change_detection import FileInfo
from magaldi_core.code_parser import PythonParser


# =============================================================================
# PYTHON PARSER
# =============================================================================


class TestPythonParser:
    """Tests for Python parsing."""

    def test_extracts_file_element(self, python_code: str):
        parser = PythonParser()
        file_info = FileInfo(
            relative_path="module.py",
            absolute_path=Path("/fake/module.py"),
            language="python",
        )

        elements = parser.parse(python_code, file_info, "scope", "repo", "main")

        file_elements = [e for e in elements if e.element_type == "file"]
        assert len(file_elements) == 1
        assert file_elements[0].name == "module.py"
        assert file_elements[0].level == 0

    def test_extracts_classes(self, python_code: str):
        parser = PythonParser()
        file_info = FileInfo(
            relative_path="module.py",
            absolute_path=Path("/fake/module.py"),
            language="python",
        )

        elements = parser.parse(python_code, file_info, "scope", "repo", "main")

        classes = [e for e in elements if e.element_type == "class"]
        class_names = [c.name for c in classes]

        assert "MyClass" in class_names
        assert "_PrivateClass" in class_names

    def test_extracts_class_docstring(self, python_code: str):
        parser = PythonParser()
        file_info = FileInfo(
            relative_path="module.py",
            absolute_path=Path("/fake/module.py"),
            language="python",
        )

        elements = parser.parse(python_code, file_info, "scope", "repo", "main")

        my_class = next(e for e in elements if e.name == "MyClass" and e.element_type == "class")
        assert my_class.docstring is not None
        assert "sample class" in my_class.docstring.lower()

    def test_extracts_standalone_functions(self, python_code: str):
        parser = PythonParser()
        file_info = FileInfo(
            relative_path="module.py",
            absolute_path=Path("/fake/module.py"),
            language="python",
        )

        elements = parser.parse(python_code, file_info, "scope", "repo", "main")

        functions = [e for e in elements if e.element_type == "function"]
        func_names = [f.name for f in functions]

        assert "standalone_function" in func_names
        assert "async_function" in func_names

    def test_extracts_methods(self, python_code: str):
        parser = PythonParser()
        file_info = FileInfo(
            relative_path="module.py",
            absolute_path=Path("/fake/module.py"),
            language="python",
        )

        elements = parser.parse(python_code, file_info, "scope", "repo", "main")

        methods = [e for e in elements if e.element_type == "method"]
        method_names = [m.name for m in methods]

        assert "__init__" in method_names
        assert "greet" in method_names
        assert "static_method" in method_names
        assert "_private_method" in method_names

    def test_extracts_constants(self, python_code: str):
        parser = PythonParser()
        file_info = FileInfo(
            relative_path="module.py",
            absolute_path=Path("/fake/module.py"),
            language="python",
        )

        elements = parser.parse(python_code, file_info, "scope", "repo", "main")

        # CONSTANT and MAX_SIZE are uppercase, so they're extracted as 'constant' type
        constants = [e for e in elements if e.element_type == "constant"]
        const_names = [c.name for c in constants]

        assert "CONSTANT" in const_names
        assert "MAX_SIZE" in const_names

    def test_async_function_detection(self, python_code: str):
        parser = PythonParser()
        file_info = FileInfo(
            relative_path="module.py",
            absolute_path=Path("/fake/module.py"),
            language="python",
        )

        elements = parser.parse(python_code, file_info, "scope", "repo", "main")

        async_func = next(e for e in elements if e.name == "async_function")
        assert async_func.is_async is True

        sync_func = next(e for e in elements if e.name == "standalone_function")
        assert sync_func.is_async is False

    def test_visibility_detection(self, python_code: str):
        parser = PythonParser()
        file_info = FileInfo(
            relative_path="module.py",
            absolute_path=Path("/fake/module.py"),
            language="python",
        )

        elements = parser.parse(python_code, file_info, "scope", "repo", "main")

        # Single underscore = protected (by Python convention)
        protected_class = next(e for e in elements if e.name == "_PrivateClass")
        assert protected_class.visibility == "protected"

        public_class = next(e for e in elements if e.name == "MyClass")
        assert public_class.visibility == "public"

        protected_method = next(e for e in elements if e.name == "_private_method")
        assert protected_method.visibility == "protected"

    def test_decorator_extraction(self, python_code: str):
        parser = PythonParser()
        file_info = FileInfo(
            relative_path="module.py",
            absolute_path=Path("/fake/module.py"),
            language="python",
        )

        elements = parser.parse(python_code, file_info, "scope", "repo", "main")

        static_method = next(e for e in elements if e.name == "static_method")
        assert "staticmethod" in static_method.decorators

    def test_signature_extraction(self, python_code: str):
        parser = PythonParser()
        file_info = FileInfo(
            relative_path="module.py",
            absolute_path=Path("/fake/module.py"),
            language="python",
        )

        elements = parser.parse(python_code, file_info, "scope", "repo", "main")

        func = next(e for e in elements if e.name == "standalone_function")
        assert "def standalone_function" in func.signature
        assert "int" in func.signature

    def test_method_parent_id_set(self, python_code: str):
        parser = PythonParser()
        file_info = FileInfo(
            relative_path="module.py",
            absolute_path=Path("/fake/module.py"),
            language="python",
        )

        elements = parser.parse(python_code, file_info, "scope", "repo", "main")

        greet_method = next(e for e in elements if e.name == "greet")
        my_class = next(e for e in elements if e.name == "MyClass" and e.element_type == "class")

        assert greet_method.parent_id == my_class.element_id

    def test_hierarchy_levels(self, python_code: str):
        parser = PythonParser()
        file_info = FileInfo(
            relative_path="module.py",
            absolute_path=Path("/fake/module.py"),
            language="python",
        )

        elements = parser.parse(python_code, file_info, "scope", "repo", "main")

        file_elem = next(e for e in elements if e.element_type == "file")
        assert file_elem.level == 0

        class_elem = next(e for e in elements if e.element_type == "class")
        assert class_elem.level == 1

        method_elem = next(e for e in elements if e.element_type == "method")
        assert method_elem.level == 2

        func_elem = next(e for e in elements if e.element_type == "function")
        assert func_elem.level == 2

        # CONSTANT and MAX_SIZE are uppercase, so they're 'constant' type
        const_elem = next(e for e in elements if e.element_type == "constant")
        assert const_elem.level == 3


# =============================================================================
# IMPORT EXTRACTION TESTS
# =============================================================================


class TestPythonImportExtraction:
    """Tests for Python import extraction."""

    def test_simple_import(self):
        """Test extracting simple import statements."""
        code = """import os
import sys
"""
        parser = PythonParser()
        file_info = FileInfo(
            relative_path="module.py",
            absolute_path=Path("/fake/module.py"),
            language="python",
        )

        elements = parser.parse(code, file_info, "scope", "repo", "main")
        file_elem = next(e for e in elements if e.element_type == "file")

        assert len(file_elem.imports) == 2

        os_import = next(i for i in file_elem.imports if i.name == "os")
        assert os_import.module == "os"
        assert os_import.alias is None
        assert os_import.line == 1

        sys_import = next(i for i in file_elem.imports if i.name == "sys")
        assert sys_import.module == "sys"
        assert sys_import.alias is None
        assert sys_import.line == 2

    def test_import_with_alias(self):
        """Test extracting import with alias."""
        code = """import pandas as pd
import numpy as np
"""
        parser = PythonParser()
        file_info = FileInfo(
            relative_path="module.py",
            absolute_path=Path("/fake/module.py"),
            language="python",
        )

        elements = parser.parse(code, file_info, "scope", "repo", "main")
        file_elem = next(e for e in elements if e.element_type == "file")

        assert len(file_elem.imports) == 2

        pd_import = next(i for i in file_elem.imports if i.name == "pandas")
        assert pd_import.module == "pandas"
        assert pd_import.alias == "pd"
        assert pd_import.line == 1

        np_import = next(i for i in file_elem.imports if i.name == "numpy")
        assert np_import.module == "numpy"
        assert np_import.alias == "np"
        assert np_import.line == 2

    def test_from_import(self):
        """Test extracting from import statements."""
        code = """from pathlib import Path
from utils import process
"""
        parser = PythonParser()
        file_info = FileInfo(
            relative_path="module.py",
            absolute_path=Path("/fake/module.py"),
            language="python",
        )

        elements = parser.parse(code, file_info, "scope", "repo", "main")
        file_elem = next(e for e in elements if e.element_type == "file")

        assert len(file_elem.imports) == 2

        path_import = next(i for i in file_elem.imports if i.name == "Path")
        assert path_import.module == "pathlib"
        assert path_import.alias is None
        assert path_import.line == 1

        process_import = next(i for i in file_elem.imports if i.name == "process")
        assert process_import.module == "utils"
        assert process_import.alias is None
        assert process_import.line == 2

    def test_from_import_with_alias(self):
        """Test extracting from import with alias."""
        code = """from utils import process as p
from collections import OrderedDict as OD
"""
        parser = PythonParser()
        file_info = FileInfo(
            relative_path="module.py",
            absolute_path=Path("/fake/module.py"),
            language="python",
        )

        elements = parser.parse(code, file_info, "scope", "repo", "main")
        file_elem = next(e for e in elements if e.element_type == "file")

        assert len(file_elem.imports) == 2

        p_import = next(i for i in file_elem.imports if i.name == "process")
        assert p_import.module == "utils"
        assert p_import.alias == "p"
        assert p_import.line == 1

        od_import = next(i for i in file_elem.imports if i.name == "OrderedDict")
        assert od_import.module == "collections"
        assert od_import.alias == "OD"
        assert od_import.line == 2

    def test_dotted_module_import(self):
        """Test extracting import of dotted module names."""
        code = """import os.path
from collections.abc import Callable
"""
        parser = PythonParser()
        file_info = FileInfo(
            relative_path="module.py",
            absolute_path=Path("/fake/module.py"),
            language="python",
        )

        elements = parser.parse(code, file_info, "scope", "repo", "main")
        file_elem = next(e for e in elements if e.element_type == "file")

        assert len(file_elem.imports) == 2

        ospath_import = next(i for i in file_elem.imports if i.name == "os.path")
        assert ospath_import.module == "os.path"
        assert ospath_import.alias is None

        callable_import = next(i for i in file_elem.imports if i.name == "Callable")
        assert callable_import.module == "collections.abc"
        assert callable_import.alias is None

    def test_mixed_imports(self):
        """Test extracting various import patterns together."""
        code = """import os
import pandas as pd
from pathlib import Path
from utils import process as p
"""
        parser = PythonParser()
        file_info = FileInfo(
            relative_path="module.py",
            absolute_path=Path("/fake/module.py"),
            language="python",
        )

        elements = parser.parse(code, file_info, "scope", "repo", "main")
        file_elem = next(e for e in elements if e.element_type == "file")

        assert len(file_elem.imports) == 4

        imports_by_name = {i.name: i for i in file_elem.imports}

        assert imports_by_name["os"].alias is None
        assert imports_by_name["pandas"].alias == "pd"
        assert imports_by_name["Path"].module == "pathlib"
        assert imports_by_name["process"].alias == "p"


# =============================================================================
# CALL EXTRACTION TESTS
# =============================================================================


class TestPythonCallExtraction:
    """Tests for Python call extraction."""

    def test_bare_function_call(self):
        """Test extracting bare function calls."""
        code = """def main():
    process(x)
    validate(data)
"""
        parser = PythonParser()
        file_info = FileInfo(
            relative_path="module.py",
            absolute_path=Path("/fake/module.py"),
            language="python",
        )

        elements = parser.parse(code, file_info, "scope", "repo", "main")
        main_func = next(e for e in elements if e.name == "main" and e.element_type == "function")

        assert len(main_func.calls) == 2

        process_call = next(c for c in main_func.calls if c.name == "process")
        assert process_call.receiver is None
        assert process_call.line == 2

        validate_call = next(c for c in main_func.calls if c.name == "validate")
        assert validate_call.receiver is None
        assert validate_call.line == 3

    def test_method_call_on_self(self):
        """Test extracting method calls on self."""
        code = """class MyClass:
    def process(self):
        self.validate()
        self.helper(x, y)
"""
        parser = PythonParser()
        file_info = FileInfo(
            relative_path="module.py",
            absolute_path=Path("/fake/module.py"),
            language="python",
        )

        elements = parser.parse(code, file_info, "scope", "repo", "main")
        process_method = next(e for e in elements if e.name == "process" and e.element_type == "method")

        assert len(process_method.calls) == 2

        validate_call = next(c for c in process_method.calls if c.name == "validate")
        assert validate_call.receiver == "self"
        assert validate_call.line == 3

        helper_call = next(c for c in process_method.calls if c.name == "helper")
        assert helper_call.receiver == "self"
        assert helper_call.line == 4

    def test_method_call_on_object(self):
        """Test extracting method calls on objects."""
        code = """def process():
    utils.run()
    config.get('key')
    db.query(sql)
"""
        parser = PythonParser()
        file_info = FileInfo(
            relative_path="module.py",
            absolute_path=Path("/fake/module.py"),
            language="python",
        )

        elements = parser.parse(code, file_info, "scope", "repo", "main")
        process_func = next(e for e in elements if e.name == "process" and e.element_type == "function")

        assert len(process_func.calls) == 3

        run_call = next(c for c in process_func.calls if c.name == "run")
        assert run_call.receiver == "utils"

        get_call = next(c for c in process_func.calls if c.name == "get")
        assert get_call.receiver == "config"

        query_call = next(c for c in process_func.calls if c.name == "query")
        assert query_call.receiver == "db"

    def test_chained_calls(self):
        """Test extracting chained method calls."""
        code = """def process():
    obj.method1().method2()
    data.filter(x).map(y).reduce(z)
"""
        parser = PythonParser()
        file_info = FileInfo(
            relative_path="module.py",
            absolute_path=Path("/fake/module.py"),
            language="python",
        )

        elements = parser.parse(code, file_info, "scope", "repo", "main")
        process_func = next(e for e in elements if e.name == "process" and e.element_type == "function")

        # Should extract method2 (from the outermost call)
        # The chained calls appear as separate call nodes
        call_names = [c.name for c in process_func.calls]
        assert "method2" in call_names or "method1" in call_names

    def test_nested_attribute_call(self):
        """Test extracting calls on nested attributes."""
        code = """def process():
    a.b.method()
    config.db.connect()
"""
        parser = PythonParser()
        file_info = FileInfo(
            relative_path="module.py",
            absolute_path=Path("/fake/module.py"),
            language="python",
        )

        elements = parser.parse(code, file_info, "scope", "repo", "main")
        process_func = next(e for e in elements if e.name == "process" and e.element_type == "function")

        assert len(process_func.calls) == 2

        method_call = next(c for c in process_func.calls if c.name == "method")
        assert method_call.receiver == "a.b"

        connect_call = next(c for c in process_func.calls if c.name == "connect")
        assert connect_call.receiver == "config.db"

    def test_mixed_calls(self):
        """Test extracting various call patterns together."""
        code = """def process(self, data):
    validate(data)
    self.transform(data)
    utils.process(data)
    return result
"""
        parser = PythonParser()
        file_info = FileInfo(
            relative_path="module.py",
            absolute_path=Path("/fake/module.py"),
            language="python",
        )

        elements = parser.parse(code, file_info, "scope", "repo", "main")
        process_func = next(e for e in elements if e.name == "process" and e.element_type == "function")

        assert len(process_func.calls) == 3

        calls_by_name = {c.name: c for c in process_func.calls}

        assert calls_by_name["validate"].receiver is None
        assert calls_by_name["transform"].receiver == "self"
        assert calls_by_name["process"].receiver == "utils"

    def test_no_calls_in_function(self):
        """Test function with no calls."""
        code = """def simple():
    x = 1 + 2
    return x
"""
        parser = PythonParser()
        file_info = FileInfo(
            relative_path="module.py",
            absolute_path=Path("/fake/module.py"),
            language="python",
        )

        elements = parser.parse(code, file_info, "scope", "repo", "main")
        simple_func = next(e for e in elements if e.name == "simple" and e.element_type == "function")

        assert len(simple_func.calls) == 0


# =============================================================================
# PATTERN DETECTION TESTS
# =============================================================================


class TestPythonPatternDetection:
    """Tests for pattern detection during parsing."""

    def test_extracts_class_variables_for_singleton_pattern(self):
        """Test that class variables are detected for singleton pattern detection."""
        code = '''
class Singleton:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def do_something(self):
        pass
'''
        parser = PythonParser()
        file_info = FileInfo(
            relative_path="singleton.py",
            absolute_path=Path("/fake/singleton.py"),
            language="python",
        )

        elements = parser.parse(code, file_info, "scope", "repo", "main")
        class_elem = next((e for e in elements if e.element_type == "class"), None)

        assert class_elem is not None
        assert class_elem.detected_patterns is not None
        assert "singleton" in class_elem.detected_patterns

    def test_singleton_pattern_with_get_instance_method(self):
        """Test singleton pattern detection with get_instance method."""
        code = '''
class MySingleton:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
'''
        parser = PythonParser()
        file_info = FileInfo(
            relative_path="singleton.py",
            absolute_path=Path("/fake/singleton.py"),
            language="python",
        )

        elements = parser.parse(code, file_info, "scope", "repo", "main")
        class_elem = next((e for e in elements if e.element_type == "class"), None)

        assert class_elem is not None
        assert class_elem.detected_patterns is not None
        assert "singleton" in class_elem.detected_patterns

    def test_class_without_class_variables_no_singleton(self):
        """Test that a class without _instance class variable is not detected as singleton."""
        code = '''
class RegularClass:
    def __init__(self):
        self.value = 42

    def do_something(self):
        pass
'''
        parser = PythonParser()
        file_info = FileInfo(
            relative_path="regular.py",
            absolute_path=Path("/fake/regular.py"),
            language="python",
        )

        elements = parser.parse(code, file_info, "scope", "repo", "main")
        class_elem = next((e for e in elements if e.element_type == "class"), None)

        assert class_elem is not None
        # Should not detect singleton pattern
        if class_elem.detected_patterns:
            assert "singleton" not in class_elem.detected_patterns
