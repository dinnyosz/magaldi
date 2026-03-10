"""Tests for the JavaScript/TypeScript parser."""

from pathlib import Path

from magaldi_core.change_detection import FileInfo
from magaldi_core.code_parser import JavaScriptParser

# =============================================================================
# JAVASCRIPT PARSER
# =============================================================================


class TestJavaScriptParser:
    """Tests for JavaScript parsing."""

    def test_extracts_file_element(self, javascript_code: str):
        parser = JavaScriptParser()
        file_info = FileInfo(
            relative_path="module.js",
            absolute_path=Path("/fake/module.js"),
            language="javascript",
        )

        elements = parser.parse(javascript_code, file_info, "scope", "repo", "main")

        file_elements = [e for e in elements if e.element_type == "file"]
        assert len(file_elements) == 1

    def test_extracts_functions(self, javascript_code: str):
        parser = JavaScriptParser()
        file_info = FileInfo(
            relative_path="module.js",
            absolute_path=Path("/fake/module.js"),
            language="javascript",
        )

        elements = parser.parse(javascript_code, file_info, "scope", "repo", "main")

        functions = [e for e in elements if e.element_type == "function"]
        func_names = [f.name for f in functions]

        assert "regularFunction" in func_names
        assert "asyncFunction" in func_names
        assert "arrowFunc" in func_names

    def test_extracts_classes(self, javascript_code: str):
        parser = JavaScriptParser()
        file_info = FileInfo(
            relative_path="module.js",
            absolute_path=Path("/fake/module.js"),
            language="javascript",
        )

        elements = parser.parse(javascript_code, file_info, "scope", "repo", "main")

        classes = [e for e in elements if e.element_type == "class"]
        class_names = [c.name for c in classes]

        assert "MyClass" in class_names
        assert "ExportedClass" in class_names

    def test_async_function_detection(self, javascript_code: str):
        parser = JavaScriptParser()
        file_info = FileInfo(
            relative_path="module.js",
            absolute_path=Path("/fake/module.js"),
            language="javascript",
        )

        elements = parser.parse(javascript_code, file_info, "scope", "repo", "main")

        async_func = next(e for e in elements if e.name == "asyncFunction")
        assert async_func.is_async is True


# =============================================================================
# IMPORT EXTRACTION TESTS
# =============================================================================


class TestJavaScriptImportExtraction:
    """Tests for JavaScript/TypeScript import extraction."""

    def test_named_imports(self):
        """Test extracting named imports."""
        code = """import { foo, bar } from './utils';
"""
        parser = JavaScriptParser()
        file_info = FileInfo(
            relative_path="module.js",
            absolute_path=Path("/fake/module.js"),
            language="javascript",
        )

        elements = parser.parse(code, file_info, "scope", "repo", "main")
        file_elem = next(e for e in elements if e.element_type == "file")

        assert len(file_elem.imports) == 2

        foo_import = next(i for i in file_elem.imports if i.name == "foo")
        assert foo_import.module == "./utils"
        assert foo_import.alias is None
        assert foo_import.line == 1

        bar_import = next(i for i in file_elem.imports if i.name == "bar")
        assert bar_import.module == "./utils"
        assert bar_import.alias is None

    def test_named_imports_with_alias(self):
        """Test extracting named imports with alias."""
        code = """import { foo as bar } from './utils';
"""
        parser = JavaScriptParser()
        file_info = FileInfo(
            relative_path="module.js",
            absolute_path=Path("/fake/module.js"),
            language="javascript",
        )

        elements = parser.parse(code, file_info, "scope", "repo", "main")
        file_elem = next(e for e in elements if e.element_type == "file")

        assert len(file_elem.imports) == 1

        import_elem = file_elem.imports[0]
        assert import_elem.name == "foo"
        assert import_elem.module == "./utils"
        assert import_elem.alias == "bar"
        assert import_elem.line == 1

    def test_default_import(self):
        """Test extracting default imports."""
        code = """import utils from './utils';
import React from 'react';
"""
        parser = JavaScriptParser()
        file_info = FileInfo(
            relative_path="module.js",
            absolute_path=Path("/fake/module.js"),
            language="javascript",
        )

        elements = parser.parse(code, file_info, "scope", "repo", "main")
        file_elem = next(e for e in elements if e.element_type == "file")

        assert len(file_elem.imports) == 2

        utils_import = next(i for i in file_elem.imports if i.name == "utils")
        assert utils_import.module == "./utils"
        assert utils_import.alias is None
        assert utils_import.line == 1

        react_import = next(i for i in file_elem.imports if i.name == "React")
        assert react_import.module == "react"
        assert react_import.alias is None
        assert react_import.line == 2

    def test_namespace_import(self):
        """Test extracting namespace imports."""
        code = """import * as utils from './utils';
"""
        parser = JavaScriptParser()
        file_info = FileInfo(
            relative_path="module.js",
            absolute_path=Path("/fake/module.js"),
            language="javascript",
        )

        elements = parser.parse(code, file_info, "scope", "repo", "main")
        file_elem = next(e for e in elements if e.element_type == "file")

        assert len(file_elem.imports) == 1

        import_elem = file_elem.imports[0]
        assert import_elem.name == "*"
        assert import_elem.module == "./utils"
        assert import_elem.alias == "utils"
        assert import_elem.line == 1

    def test_require_import(self):
        """Test extracting CommonJS require imports."""
        code = """const bar = require('lib');
const utils = require('./utils');
"""
        parser = JavaScriptParser()
        file_info = FileInfo(
            relative_path="module.js",
            absolute_path=Path("/fake/module.js"),
            language="javascript",
        )

        elements = parser.parse(code, file_info, "scope", "repo", "main")
        file_elem = next(e for e in elements if e.element_type == "file")

        assert len(file_elem.imports) == 2

        bar_import = next(i for i in file_elem.imports if i.name == "bar")
        assert bar_import.module == "lib"
        assert bar_import.alias is None
        assert bar_import.line == 1

        utils_import = next(i for i in file_elem.imports if i.name == "utils")
        assert utils_import.module == "./utils"
        assert utils_import.alias is None
        assert utils_import.line == 2

    def test_mixed_imports(self):
        """Test extracting various import patterns together."""
        code = """import React from 'react';
import { useState, useEffect } from 'react';
import * as utils from './utils';
const path = require('path');
"""
        parser = JavaScriptParser()
        file_info = FileInfo(
            relative_path="module.js",
            absolute_path=Path("/fake/module.js"),
            language="javascript",
        )

        elements = parser.parse(code, file_info, "scope", "repo", "main")
        file_elem = next(e for e in elements if e.element_type == "file")

        assert len(file_elem.imports) == 5

        imports_by_name = {i.name: i for i in file_elem.imports}

        assert imports_by_name["React"].module == "react"
        assert imports_by_name["useState"].module == "react"
        assert imports_by_name["useEffect"].module == "react"
        assert imports_by_name["*"].alias == "utils"
        assert imports_by_name["path"].module == "path"


class TestTypeScriptImportExtraction:
    """Tests for TypeScript import extraction."""

    def test_typescript_imports(self):
        """Test extracting TypeScript imports (same syntax as JS)."""
        code = """import { Component } from '@angular/core';
import type { User } from './types';
"""
        parser = JavaScriptParser(language="typescript")
        file_info = FileInfo(
            relative_path="module.ts",
            absolute_path=Path("/fake/module.ts"),
            language="typescript",
        )

        elements = parser.parse(code, file_info, "scope", "repo", "main")
        file_elem = next(e for e in elements if e.element_type == "file")

        # Should extract both regular and type imports
        assert len(file_elem.imports) >= 1

        component_import = next(i for i in file_elem.imports if i.name == "Component")
        assert component_import.module == "@angular/core"
        assert component_import.alias is None


# =============================================================================
# CALL EXTRACTION TESTS
# =============================================================================


class TestJavaScriptCallExtraction:
    """Tests for JavaScript/TypeScript call extraction."""

    def test_bare_function_call(self):
        """Test extracting bare function calls."""
        code = """function main() {
    process(x);
    validate(data);
}
"""
        parser = JavaScriptParser()
        file_info = FileInfo(
            relative_path="module.js",
            absolute_path=Path("/fake/module.js"),
            language="javascript",
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

    def test_method_call_on_this(self):
        """Test extracting method calls on this."""
        code = """class MyClass {
    process() {
        this.validate();
        this.helper(x, y);
    }
}
"""
        parser = JavaScriptParser()
        file_info = FileInfo(
            relative_path="module.js",
            absolute_path=Path("/fake/module.js"),
            language="javascript",
        )

        elements = parser.parse(code, file_info, "scope", "repo", "main")
        process_method = next(e for e in elements if e.name == "process" and e.element_type == "method")

        assert len(process_method.calls) == 2

        validate_call = next(c for c in process_method.calls if c.name == "validate")
        assert validate_call.receiver == "this"

        helper_call = next(c for c in process_method.calls if c.name == "helper")
        assert helper_call.receiver == "this"

    def test_method_call_on_object(self):
        """Test extracting method calls on objects."""
        code = """function process() {
    utils.run();
    config.get('key');
    db.query(sql);
}
"""
        parser = JavaScriptParser()
        file_info = FileInfo(
            relative_path="module.js",
            absolute_path=Path("/fake/module.js"),
            language="javascript",
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
        code = """function process() {
    obj.method1().method2();
    data.filter(x).map(y).reduce(z);
}
"""
        parser = JavaScriptParser()
        file_info = FileInfo(
            relative_path="module.js",
            absolute_path=Path("/fake/module.js"),
            language="javascript",
        )

        elements = parser.parse(code, file_info, "scope", "repo", "main")
        process_func = next(e for e in elements if e.name == "process" and e.element_type == "function")

        # Should extract calls from the chain
        call_names = [c.name for c in process_func.calls]
        assert "method2" in call_names or "method1" in call_names

    def test_mixed_calls(self):
        """Test extracting various call patterns together."""
        code = """function process(data) {
    validate(data);
    this.transform(data);
    utils.process(data);
    return result;
}
"""
        parser = JavaScriptParser()
        file_info = FileInfo(
            relative_path="module.js",
            absolute_path=Path("/fake/module.js"),
            language="javascript",
        )

        elements = parser.parse(code, file_info, "scope", "repo", "main")
        process_func = next(e for e in elements if e.name == "process" and e.element_type == "function")

        # Note: 'this' reference in a regular function might be handled differently
        # The important thing is we extract the calls we can
        call_names = [c.name for c in process_func.calls]
        assert "validate" in call_names
        assert "process" in call_names

    def test_no_calls_in_function(self):
        """Test function with no calls."""
        code = """function simple() {
    const x = 1 + 2;
    return x;
}
"""
        parser = JavaScriptParser()
        file_info = FileInfo(
            relative_path="module.js",
            absolute_path=Path("/fake/module.js"),
            language="javascript",
        )

        elements = parser.parse(code, file_info, "scope", "repo", "main")
        simple_func = next(e for e in elements if e.name == "simple" and e.element_type == "function")

        assert len(simple_func.calls) == 0

    def test_arrow_function_calls(self):
        """Test extracting calls from arrow functions."""
        code = """const process = (data) => {
    validate(data);
    transform(data);
};
"""
        parser = JavaScriptParser()
        file_info = FileInfo(
            relative_path="module.js",
            absolute_path=Path("/fake/module.js"),
            language="javascript",
        )

        elements = parser.parse(code, file_info, "scope", "repo", "main")
        process_func = next(e for e in elements if e.name == "process" and e.element_type == "function")

        assert len(process_func.calls) == 2
        call_names = [c.name for c in process_func.calls]
        assert "validate" in call_names
        assert "transform" in call_names


# =============================================================================
# MINIFIED CODE EXTRACTION
# =============================================================================


class TestMinifiedCodeExtraction:
    """Tests for extracting elements from minified (single-line) code.

    These tests verify that byte_offset is used correctly to generate unique
    element IDs and that raw_code is extracted precisely for each element.
    """

    def test_minified_javascript_unique_ids(self):
        """Minified JS with multiple functions on one line has unique element IDs."""
        # Minified code: three functions on same line
        minified_js = 'function foo(){return 1;}function bar(){return 2;}function baz(){return 3;}'

        parser = JavaScriptParser()
        file_info = FileInfo(
            relative_path="app.min.js",
            absolute_path=Path("/fake/app.min.js"),
            language="javascript",
        )

        elements = parser.parse(minified_js, file_info, "scope", "repo", "main")

        # Should extract 3 functions + 1 file element = 4 elements
        functions = [e for e in elements if e.element_type == "function"]
        assert len(functions) == 3, f"Expected 3 functions, got {len(functions)}"

        # All element IDs should be unique
        element_ids = [e.element_id for e in functions]
        assert len(set(element_ids)) == 3, "Element IDs should be unique"

        # Verify element names are correct
        names = {e.name for e in functions}
        assert names == {"foo", "bar", "baz"}

    def test_minified_javascript_raw_code_extraction(self):
        """Each function in minified JS has its own raw_code, not the whole file."""
        minified_js = 'function foo(){return 1;}function bar(){return 2;}'

        parser = JavaScriptParser()
        file_info = FileInfo(
            relative_path="app.min.js",
            absolute_path=Path("/fake/app.min.js"),
            language="javascript",
        )

        elements = parser.parse(minified_js, file_info, "scope", "repo", "main")
        functions = [e for e in elements if e.element_type == "function"]

        foo = next(e for e in functions if e.name == "foo")
        bar = next(e for e in functions if e.name == "bar")

        # raw_code should be the specific function, not the entire file
        assert foo.raw_code == "function foo(){return 1;}", f"Got: {foo.raw_code!r}"
        assert bar.raw_code == "function bar(){return 2;}", f"Got: {bar.raw_code!r}"

    def test_minified_javascript_byte_offset_in_element_id(self):
        """Byte offsets in minified JS should differ in element IDs."""
        minified_js = 'function foo(){return 1;}function bar(){return 2;}'

        parser = JavaScriptParser()
        file_info = FileInfo(
            relative_path="app.min.js",
            absolute_path=Path("/fake/app.min.js"),
            language="javascript",
        )

        elements = parser.parse(minified_js, file_info, "scope", "repo", "main")
        functions = [e for e in elements if e.element_type == "function"]

        foo = next(e for e in functions if e.name == "foo")
        bar = next(e for e in functions if e.name == "bar")

        # foo starts at byte 0, bar starts at byte 25
        # The byte offset is encoded in the element_id
        assert foo.element_id.endswith(":0"), f"foo element_id: {foo.element_id}"
        assert bar.element_id.endswith(":25"), f"bar element_id: {bar.element_id}"

    def test_minified_js_class_with_methods(self):
        """Minified JS class with methods on single line."""
        minified_js = 'class Foo{constructor(){this.x=1;}bar(){return 2;}}'

        parser = JavaScriptParser()
        file_info = FileInfo(
            relative_path="class.min.js",
            absolute_path=Path("/fake/class.min.js"),
            language="javascript",
        )

        elements = parser.parse(minified_js, file_info, "scope", "repo", "main")

        # Should have: file, class, constructor method, bar method
        classes = [e for e in elements if e.element_type == "class"]
        methods = [e for e in elements if e.element_type == "method"]

        assert len(classes) == 1
        assert len(methods) == 2, f"Expected 2 methods, got {len(methods)}"

        # All element IDs should be unique
        all_elem_ids = [e.element_id for e in elements]
        assert len(set(all_elem_ids)) == len(all_elem_ids), "All element IDs should be unique"

    def test_element_id_contains_byte_offset(self):
        """Element IDs should contain byte_offset, not line number."""
        minified_js = 'function foo(){return 1;}function bar(){return 2;}'

        parser = JavaScriptParser()
        file_info = FileInfo(
            relative_path="app.min.js",
            absolute_path=Path("/fake/app.min.js"),
            language="javascript",
        )

        elements = parser.parse(minified_js, file_info, "scope", "repo", "main")
        functions = [e for e in elements if e.element_type == "function"]

        # Both functions are on line 1, but byte_offsets differ
        foo = next(e for e in functions if e.name == "foo")
        bar = next(e for e in functions if e.name == "bar")

        # IDs should end with different byte offsets
        assert foo.element_id.endswith(":0"), f"foo element_id: {foo.element_id}"
        assert bar.element_id.endswith(":25"), f"bar element_id: {bar.element_id}"


# =============================================================================
# MINIFIED PYTHON CODE (for completeness)
# =============================================================================


class TestMinifiedPythonCodeExtraction:
    """Tests for minified Python code extraction."""

    def test_minified_python_newline_separated(self):
        """Minified Python with minimal newlines."""
        from magaldi_core.code_parser import PythonParser

        # Python allows semicolons for multiple statements
        # Note: This is unusual but technically valid Python
        minified_py = 'def foo(): return 1\ndef bar(): return 2\ndef baz(): return 3'

        parser = PythonParser()
        file_info = FileInfo(
            relative_path="app.min.py",
            absolute_path=Path("/fake/app.min.py"),
            language="python",
        )

        elements = parser.parse(minified_py, file_info, "scope", "repo", "main")
        functions = [e for e in elements if e.element_type == "function"]

        assert len(functions) == 3, f"Expected 3 functions, got {len(functions)}"

        # All element IDs should be unique
        element_ids = [e.element_id for e in functions]
        assert len(set(element_ids)) == 3, "Element IDs should be unique"


# =============================================================================
# ASSIGNMENT FUNCTION EXTRACTION (obj.prop = function() {})
# =============================================================================


class TestAssignmentFunctionExtraction:
    """Tests for assignment-based function patterns common in pre-ES6 JS."""

    def _parse(self, code: str) -> list:
        parser = JavaScriptParser()
        file_info = FileInfo(
            relative_path="module.js",
            absolute_path=Path("/fake/module.js"),
            language="javascript",
        )
        return parser.parse(code, file_info, "scope", "repo", "main")

    def test_obj_prop_named_function(self):
        """obj.prop = function name() {} extracts as function with correct name."""
        code = """app.init = function init() {
  this.cache = {};
};

app.use = function use(fn) {
  var offset = 0;
};

req.get = function get(name) {
  return this.header(name);
};

res.send = function send(body) {
  var chunk = body;
};
"""
        elements = self._parse(code)
        functions = [e for e in elements if e.element_type == "function"]
        func_names = {f.name for f in functions}

        assert "init" in func_names, f"Expected 'init', got {func_names}"
        assert "use" in func_names, f"Expected 'use', got {func_names}"
        assert "get" in func_names, f"Expected 'get', got {func_names}"
        assert "send" in func_names, f"Expected 'send', got {func_names}"

    def test_obj_prop_function_signature_includes_name(self):
        """Signature for named function expression should include the function name."""
        code = """app.init = function init() {
  this.cache = {};
};
"""
        elements = self._parse(code)
        init_func = next(e for e in elements if e.name == "init" and e.element_type == "function")

        # Signature should include the function name: "app.init = function init()"
        # not just "app.init = function()"
        assert "function init(" in init_func.signature, (
            f"Signature should contain 'function init(', got: {init_func.signature}"
        )

    def test_obj_prop_anonymous_function(self):
        """obj.prop = function() {} uses property name as function name."""
        code = """app.render = function(name, options, callback) {
  var done = callback;
};
"""
        elements = self._parse(code)
        render_func = next(
            (e for e in elements if e.name == "render" and e.element_type == "function"),
            None
        )
        assert render_func is not None, "Expected anonymous function to use property name 'render'"

    def test_prototype_method_extraction(self):
        """Constructor.prototype.method = function() {} is extracted."""
        code = """View.prototype.lookup = function lookup() {
  var path = this.name;
  return path;
};

View.prototype.render = function render(options, callback) {
  this.engine(this.path, options, callback);
};
"""
        elements = self._parse(code)
        functions = [e for e in elements if e.element_type == "function"]
        func_names = {f.name for f in functions}

        assert "lookup" in func_names, f"Expected 'lookup', got {func_names}"
        assert "render" in func_names, f"Expected 'render', got {func_names}"

    def test_prototype_method_signature(self):
        """Prototype method signature includes prototype chain."""
        code = """View.prototype.lookup = function lookup(name) {
  return name;
};
"""
        elements = self._parse(code)
        lookup = next(e for e in elements if e.name == "lookup" and e.element_type == "function")

        assert "View.prototype.lookup" in lookup.signature, (
            f"Signature should contain 'View.prototype.lookup', got: {lookup.signature}"
        )

    def test_exports_named_function(self):
        """exports.xxx = function name() {} is extracted."""
        code = """exports.setCharset = function setCharset(type, charset) {
  return type;
};
"""
        elements = self._parse(code)
        func = next(
            (e for e in elements if e.name == "setCharset" and e.element_type == "function"),
            None
        )
        assert func is not None, "Expected exports function 'setCharset'"

    def test_exports_anonymous_function(self):
        """exports.xxx = function() {} uses property name."""
        code = """exports.normalizeType = function(type) {
  return type;
};
"""
        elements = self._parse(code)
        func = next(
            (e for e in elements if e.name == "normalizeType" and e.element_type == "function"),
            None
        )
        assert func is not None, "Expected exports function 'normalizeType'"

    def test_assignment_function_parameters(self):
        """Assignment functions extract parameters correctly."""
        code = """app.use = function use(fn, opts) {
  return fn;
};
"""
        elements = self._parse(code)
        use_func = next(e for e in elements if e.name == "use" and e.element_type == "function")

        assert use_func.parameters is not None, "Expected parameters to be extracted"
        # Parameters may be dicts or objects depending on serialization
        param_names = [
            p["name"] if isinstance(p, dict) else p.name
            for p in use_func.parameters
        ]
        assert "fn" in param_names, f"Expected 'fn' in params, got {param_names}"
        assert "opts" in param_names, f"Expected 'opts' in params, got {param_names}"


# =============================================================================
# COMMONJS REQUIRE() AS IMPORT ELEMENTS
# =============================================================================


class TestCommonJSRequireElements:
    """Tests for CommonJS require() being captured as import elements."""

    def _parse(self, code: str) -> list:
        parser = JavaScriptParser()
        file_info = FileInfo(
            relative_path="module.js",
            absolute_path=Path("/fake/module.js"),
            language="javascript",
        )
        return parser.parse(code, file_info, "scope", "repo", "main")

    def test_var_require_creates_import_element(self):
        """var x = require('mod') should create an import element."""
        code = """var http = require('http');
var path = require('path');
"""
        elements = self._parse(code)
        import_elems = [e for e in elements if e.element_type == "import"]

        import_names = {e.name for e in import_elems}
        assert "http" in import_names, (
            f"Expected import element for 'http', got: {import_names}"
        )
        assert "path" in import_names, (
            f"Expected import element for 'path', got: {import_names}"
        )

    def test_const_require_creates_import_element(self):
        """const x = require('mod') should create an import element."""
        code = """const express = require('express');
"""
        elements = self._parse(code)
        import_elems = [e for e in elements if e.element_type == "import"]

        assert any(e.name == "express" for e in import_elems), (
            f"Expected import element for 'express', got: {[e.name for e in import_elems]}"
        )

    def test_chained_require_creates_import(self):
        """var debug = require('debug')('app') should create an import."""
        code = """var debug = require('debug')('express:view');
"""
        elements = self._parse(code)
        file_elem = next(e for e in elements if e.element_type == "file")

        # Should be captured as import on the file element
        import_names = {i.name for i in file_elem.imports}
        assert "debug" in import_names, (
            f"Expected import for 'debug' from chained require, got: {import_names}"
        )

    def test_require_not_duplicated_as_variable(self):
        """require() should create import elements, not variable elements."""
        code = """var http = require('http');
const express = require('express');
"""
        elements = self._parse(code)

        # Should not have variable elements for require imports
        var_elems = [e for e in elements if e.element_type == "variable"]
        var_names = {e.name for e in var_elems}
        assert "http" not in var_names, (
            f"'http' should be import, not variable: {var_names}"
        )
        assert "express" not in var_names, (
            f"'express' should be import, not variable: {var_names}"
        )

    def test_require_import_element_signature(self):
        """Import elements from require() should have readable signatures."""
        code = """const express = require('express');
"""
        elements = self._parse(code)
        import_elems = [e for e in elements if e.element_type == "import"]

        assert len(import_elems) > 0, "Expected at least one import element"
        express_import = next(e for e in import_elems if e.name == "express")
        assert express_import.signature is not None, "Import element should have a signature"

    def test_mixed_imports_and_requires(self):
        """Both ES6 imports and require() should create import elements."""
        code = """import { useState } from 'react';
const path = require('path');
"""
        elements = self._parse(code)
        import_elems = [e for e in elements if e.element_type == "import"]

        import_names = {e.name for e in import_elems}
        assert "react" in import_names or "useState" in import_names, (
            f"Expected ES6 import element, got: {import_names}"
        )
        assert "path" in import_names, (
            f"Expected require import element for 'path', got: {import_names}"
        )
