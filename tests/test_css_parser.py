"""Tests for CSS parser (src/magaldi_core/parsers/css.py)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from magaldi_core.parsers.css import CSSParser

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def parser():
    return CSSParser()


@pytest.fixture
def file_info():
    fi = MagicMock()
    fi.relative_path = "styles/main.css"
    fi.absolute_path = Path("/tmp/test.css")
    fi.language = "css"
    return fi


def _parse(parser, file_info, css: str):
    return parser.parse(css, file_info, "test", "repo", "user")


# =============================================================================
# FILE ELEMENT
# =============================================================================


class TestFileElement:
    def test_file_element_created(self, parser, file_info):
        elements = _parse(parser, file_info, ".foo { color: red; }")
        file_elem = elements[0]
        assert file_elem.element_type == "file"
        assert file_elem.name == "main.css"
        assert file_elem.language == "css"
        assert file_elem.level == 0

    def test_file_summary_with_sections(self, parser, file_info):
        css = ".a { color: red; }\n.b { color: blue; }"
        elements = _parse(parser, file_info, css)
        assert elements[0].summary == "CSS stylesheet with 2 sections"

    def test_empty_file(self, parser, file_info):
        elements = _parse(parser, file_info, "")
        assert len(elements) == 1  # File element only
        assert elements[0].element_type == "file"
        assert elements[0].summary == "Empty CSS stylesheet"

    def test_comments_only_file(self, parser, file_info):
        elements = _parse(parser, file_info, "/* Just a comment */")
        assert len(elements) == 1  # File element only


# =============================================================================
# QUALIFIED RULES
# =============================================================================


class TestQualifiedRules:
    def test_simple_rule(self, parser, file_info):
        css = ".navbar { display: flex; padding: 1rem; }"
        elements = _parse(parser, file_info, css)
        rule = elements[1]
        assert rule.element_type == "css_rule"
        assert rule.name == ".navbar"
        assert "flex layout" in rule.summary
        assert "padding" in rule.summary

    def test_rule_with_multiple_properties(self, parser, file_info):
        css = """.card {
    box-shadow: 0 2px 4px rgba(0,0,0,.1);
    border-radius: 8px;
    padding: 1.5rem;
    background-color: white;
}"""
        elements = _parse(parser, file_info, css)
        rule = elements[1]
        assert rule.element_type == "css_rule"
        assert rule.name == ".card"
        assert "box shadow" in rule.summary
        assert "border radius" in rule.summary

    def test_rule_line_numbers(self, parser, file_info):
        css = "\n\n.foo {\n    color: red;\n}\n"
        elements = _parse(parser, file_info, css)
        rule = elements[1]
        assert rule.line_start == 3
        assert rule.line_end == 5

    def test_complex_selector(self, parser, file_info):
        css = "header > nav .menu-item:hover { color: blue; }"
        elements = _parse(parser, file_info, css)
        rule = elements[1]
        assert "header > nav .menu-item:hover" in rule.name

    def test_multiple_selectors(self, parser, file_info):
        css = "h1, h2, h3 { font-weight: bold; }"
        elements = _parse(parser, file_info, css)
        rule = elements[1]
        assert "h1" in rule.name

    def test_rule_has_parent_id(self, parser, file_info):
        css = ".foo { color: red; }"
        elements = _parse(parser, file_info, css)
        rule = elements[1]
        assert rule.parent_id == elements[0].element_id
        assert rule.level == 1

    def test_rule_raw_code(self, parser, file_info):
        css = ".foo { color: red; }"
        elements = _parse(parser, file_info, css)
        rule = elements[1]
        assert "color" in rule.raw_code
        assert ".foo" in rule.raw_code

    def test_empty_rule(self, parser, file_info):
        css = ".empty { }"
        elements = _parse(parser, file_info, css)
        rule = elements[1]
        assert rule.element_type == "css_rule"
        assert "Empty rule" in rule.summary

    def test_rule_element_id_generated(self, parser, file_info):
        css = ".foo { color: red; }"
        elements = _parse(parser, file_info, css)
        rule = elements[1]
        assert rule.element_id
        assert "css_rule" in rule.element_id


# =============================================================================
# @MEDIA RULES
# =============================================================================


class TestMediaRules:
    def test_media_query(self, parser, file_info):
        css = """@media (max-width: 768px) {
    .navbar { flex-direction: column; }
    .sidebar { display: none; }
}"""
        elements = _parse(parser, file_info, css)
        media = elements[1]
        assert media.element_type == "css_rule"
        assert "@media" in media.name
        assert "Responsive" in media.summary
        assert ".navbar" in media.summary

    def test_media_query_name(self, parser, file_info):
        css = "@media print { body { font-size: 12pt; } }"
        elements = _parse(parser, file_info, css)
        media = elements[1]
        assert "@media" in media.name
        assert "print" in media.name


# =============================================================================
# @KEYFRAMES RULES
# =============================================================================


class TestKeyframesRules:
    def test_keyframes(self, parser, file_info):
        css = """@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}"""
        elements = _parse(parser, file_info, css)
        kf = elements[1]
        assert kf.element_type == "css_rule"
        assert "fadeIn" in kf.name
        assert "Animation" in kf.summary
        assert "opacity" in kf.summary

    def test_keyframes_with_percentages(self, parser, file_info):
        css = """@keyframes slide {
    0% { transform: translateX(0); }
    50% { transform: translateX(50%); }
    100% { transform: translateX(100%); }
}"""
        elements = _parse(parser, file_info, css)
        kf = elements[1]
        assert "Animation" in kf.summary
        assert "transform" in kf.summary


# =============================================================================
# CSS CUSTOM PROPERTIES
# =============================================================================


class TestCustomProperties:
    def test_root_custom_properties(self, parser, file_info):
        css = """:root {
    --color-primary: #007bff;
    --spacing-md: 16px;
    --font-size-base: 1rem;
}"""
        elements = _parse(parser, file_info, css)
        root = elements[1]
        assert root.element_type == "variable"
        assert ":root" in root.name
        assert "--color-primary" in root.summary
        assert "--spacing-md" in root.summary

    def test_theme_custom_properties(self, parser, file_info):
        css = """[data-theme="dark"] {
    --bg-color: #1a1a2e;
    --text-color: #eee;
}"""
        elements = _parse(parser, file_info, css)
        theme = elements[1]
        assert theme.element_type == "variable"
        assert "--bg-color" in theme.summary


# =============================================================================
# @FONT-FACE
# =============================================================================


class TestFontFace:
    def test_font_face(self, parser, file_info):
        css = """@font-face {
    font-family: 'Inter';
    src: url('/fonts/inter.woff2') format('woff2');
    font-weight: 400;
}"""
        elements = _parse(parser, file_info, css)
        ff = elements[1]
        assert ff.element_type == "css_rule"
        assert "Font face" in ff.summary
        assert "Inter" in ff.summary


# =============================================================================
# COMMENTS AS DOCSTRINGS
# =============================================================================


class TestComments:
    def test_comment_attached_as_docstring(self, parser, file_info):
        css = """/* Navigation styles */
.navbar { display: flex; }"""
        elements = _parse(parser, file_info, css)
        rule = elements[1]
        assert rule.docstring == "Navigation styles"

    def test_comment_not_carried_to_wrong_rule(self, parser, file_info):
        css = """/* First comment */
.a { color: red; }
.b { color: blue; }"""
        elements = _parse(parser, file_info, css)
        assert elements[1].docstring == "First comment"
        assert elements[2].docstring is None


# =============================================================================
# OTHER AT-RULES
# =============================================================================


class TestOtherAtRules:
    def test_import_rule(self, parser, file_info):
        css = "@import 'normalize.css';"
        elements = _parse(parser, file_info, css)
        # @import without a block is a statement at-rule
        imp = elements[1]
        assert imp.element_type == "css_rule"
        assert "@import" in imp.summary

    def test_supports_rule(self, parser, file_info):
        css = "@supports (display: grid) { .container { display: grid; } }"
        elements = _parse(parser, file_info, css)
        sup = elements[1]
        assert "@supports" in sup.summary


# =============================================================================
# EDGE CASES
# =============================================================================


class TestEdgeCases:
    def test_minified_single_line(self, parser, file_info):
        css = ".a{color:red;}.b{color:blue;}.c{display:flex;}"
        elements = _parse(parser, file_info, css)
        # Should extract at least the rules (file + rules)
        assert len(elements) >= 3

    def test_large_file_many_rules(self, parser, file_info):
        rules = "\n".join(f".rule-{i} {{ color: red; }}" for i in range(50))
        elements = _parse(parser, file_info, rules)
        # 1 file + 50 rules
        assert len(elements) == 51

    def test_unicode_content(self, parser, file_info):
        css = '.emoji::before { content: "🎉"; color: red; }'
        elements = _parse(parser, file_info, css)
        assert len(elements) >= 2

    def test_vendor_prefix_properties(self, parser, file_info):
        css = ".box { -webkit-transform: rotate(45deg); transform: rotate(45deg); }"
        elements = _parse(parser, file_info, css)
        rule = elements[1]
        assert "transform" in rule.summary
