"""Tests for SCSS parser (src/magaldi_core/parsers/css.py::SCSSParser)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from magaldi_core.parsers.css import SCSSParser

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def parser():
    return SCSSParser()


@pytest.fixture
def file_info():
    fi = MagicMock()
    fi.relative_path = "styles/theme.scss"
    fi.absolute_path = Path("/tmp/test.scss")
    fi.language = "scss"
    return fi


def _parse(parser, file_info, scss: str):
    return parser.parse(scss, file_info, "test", "repo", "user")


# =============================================================================
# FILE ELEMENT
# =============================================================================


class TestSCSSFileElement:
    def test_file_element_created(self, parser, file_info):
        elements = _parse(parser, file_info, "$color: red;")
        file_elem = elements[0]
        assert file_elem.element_type == "file"
        assert file_elem.name == "theme.scss"
        assert file_elem.language == "scss"
        assert "SCSS stylesheet" in file_elem.summary

    def test_empty_file(self, parser, file_info):
        elements = _parse(parser, file_info, "")
        assert len(elements) == 1
        assert elements[0].summary == "Empty SCSS stylesheet"


# =============================================================================
# SCSS VARIABLES
# =============================================================================


class TestSCSSVariables:
    def test_simple_variable(self, parser, file_info):
        scss = "$primary-color: #007bff;"
        elements = _parse(parser, file_info, scss)
        var = elements[1]
        assert var.element_type == "variable"
        assert var.name == "$primary-color"
        assert "#007bff" in var.summary
        assert "SCSS variable" in var.summary

    def test_variable_with_default(self, parser, file_info):
        scss = "$spacing: 16px !default;"
        elements = _parse(parser, file_info, scss)
        var = elements[1]
        assert var.element_type == "variable"
        assert var.name == "$spacing"
        assert "16px" in var.summary

    def test_multiple_variables(self, parser, file_info):
        scss = "$a: red;\n$b: blue;\n$c: green;"
        elements = _parse(parser, file_info, scss)
        vars_ = [e for e in elements if e.element_type == "variable"]
        assert len(vars_) == 3
        assert vars_[0].name == "$a"
        assert vars_[1].name == "$b"
        assert vars_[2].name == "$c"

    def test_variable_line_numbers(self, parser, file_info):
        scss = "\n\n$color: red;\n"
        elements = _parse(parser, file_info, scss)
        var = elements[1]
        assert var.line_start == 3

    def test_variable_has_signature(self, parser, file_info):
        scss = "$font-stack: 'Helvetica', sans-serif;"
        elements = _parse(parser, file_info, scss)
        var = elements[1]
        assert var.signature is not None
        assert "$font-stack" in var.signature

    def test_variable_long_value_truncated(self, parser, file_info):
        long_val = ", ".join([f"item-{i}" for i in range(50)])
        scss = f"$list: {long_val};"
        elements = _parse(parser, file_info, scss)
        var = elements[1]
        assert "..." in var.summary


# =============================================================================
# SCSS MIXINS
# =============================================================================


class TestSCSSMixins:
    def test_mixin_no_params(self, parser, file_info):
        scss = "@mixin clearfix {\n    &::after { content: ''; display: table; clear: both; }\n}"
        elements = _parse(parser, file_info, scss)
        mixin = [e for e in elements if e.element_type == "function" and e.name == "clearfix"][0]
        assert "Mixin clearfix" in mixin.summary
        assert mixin.signature == "@mixin clearfix"

    def test_mixin_with_params(self, parser, file_info):
        scss = "@mixin responsive($breakpoint) {\n    @media (max-width: $breakpoint) {\n        @content;\n    }\n}"
        elements = _parse(parser, file_info, scss)
        mixin = [e for e in elements if e.element_type == "function" and e.name == "responsive"][0]
        assert "Mixin responsive($breakpoint)" in mixin.summary
        assert "media query" in mixin.summary

    def test_mixin_line_range(self, parser, file_info):
        scss = "\n@mixin foo {\n    color: red;\n}\n"
        elements = _parse(parser, file_info, scss)
        mixin = [e for e in elements if e.element_type == "function"][0]
        assert mixin.line_start == 2
        assert mixin.line_end == 4

    def test_mixin_raw_code(self, parser, file_info):
        scss = "@mixin box($size) {\n    width: $size;\n    height: $size;\n}"
        elements = _parse(parser, file_info, scss)
        mixin = [e for e in elements if e.element_type == "function"][0]
        assert "@mixin box" in mixin.raw_code
        assert "width" in mixin.raw_code


# =============================================================================
# SCSS FUNCTIONS
# =============================================================================


class TestSCSSFunctions:
    def test_function_with_return(self, parser, file_info):
        scss = "@function rem($px) {\n    @return $px / 16px * 1rem;\n}"
        elements = _parse(parser, file_info, scss)
        func = [e for e in elements if e.element_type == "function" and e.name == "rem"][0]
        assert "Function rem($px)" in func.summary
        assert "returns" in func.summary

    def test_function_signature(self, parser, file_info):
        scss = "@function double($n) {\n    @return $n * 2;\n}"
        elements = _parse(parser, file_info, scss)
        func = [e for e in elements if e.element_type == "function"][0]
        assert func.signature == "@function double($n)"


# =============================================================================
# CSS RULES IN SCSS
# =============================================================================


class TestSCSSCSSRules:
    def test_plain_css_rule(self, parser, file_info):
        scss = ".card {\n    border-radius: 8px;\n    padding: 1rem;\n}"
        elements = _parse(parser, file_info, scss)
        rules = [e for e in elements if e.element_type == "css_rule"]
        assert len(rules) >= 1
        assert "border radius" in rules[0].summary

    def test_scss_with_mixed_constructs(self, parser, file_info):
        scss = """$color: blue;

@mixin flex-center {
    display: flex;
    justify-content: center;
    align-items: center;
}

.container {
    max-width: 1200px;
    margin: 0 auto;
}
"""
        elements = _parse(parser, file_info, scss)
        types = {e.element_type for e in elements}
        assert "file" in types
        assert "variable" in types
        assert "function" in types  # mixin stored as function
        assert "css_rule" in types


# =============================================================================
# EDGE CASES
# =============================================================================


class TestSCSSEdgeCases:
    def test_nested_braces_in_mixin(self, parser, file_info):
        scss = """@mixin responsive($bp) {
    @media (max-width: $bp) {
        .sidebar {
            display: none;
        }
    }
}"""
        elements = _parse(parser, file_info, scss)
        mixin = [e for e in elements if e.element_type == "function"][0]
        # Mixin should capture the full nested block
        assert "display: none" in mixin.raw_code

    def test_scss_interpolation(self, parser, file_info):
        scss = """.icon-#{$name} { color: red; }"""
        # Should not crash — interpolation replaced with placeholder
        elements = _parse(parser, file_info, scss)
        assert len(elements) >= 1  # At least file element

    def test_include_statements_stripped(self, parser, file_info):
        scss = """.box {
    @include flex-center;
    padding: 1rem;
}"""
        # Should parse without crashing even though @include is SCSS-only
        elements = _parse(parser, file_info, scss)
        assert len(elements) >= 1

    def test_parent_ids_set(self, parser, file_info):
        scss = "$color: red;\n.foo { color: $color; }"
        elements = _parse(parser, file_info, scss)
        file_id = elements[0].element_id
        for e in elements[1:]:
            assert e.parent_id == file_id
