"""Tests for CSS summary heuristics (src/magaldi_core/parsers/css_summary.py)."""

from __future__ import annotations

from magaldi_core.parsers.css_summary import (
    _join_list,
    _property_to_concept,
    summarize_at_rule,
    summarize_custom_properties,
    summarize_file,
    summarize_font_face,
    summarize_keyframes,
    summarize_media,
    summarize_rule,
    summarize_scss_function,
    summarize_scss_mixin,
    summarize_scss_variable,
)

# =============================================================================
# PROPERTY → CONCEPT MAPPING
# =============================================================================


class TestPropertyToConcept:
    def test_display_flex(self):
        assert _property_to_concept("display", "flex") == "flex layout"

    def test_display_grid(self):
        assert _property_to_concept("display", "grid") == "grid layout"

    def test_display_none(self):
        assert _property_to_concept("display", "none") == "hidden"

    def test_position_fixed(self):
        assert _property_to_concept("position", "fixed") == "fixed positioning"

    def test_position_sticky(self):
        assert _property_to_concept("position", "sticky") == "sticky positioning"

    def test_simple_property(self):
        assert _property_to_concept("padding") == "padding"
        assert _property_to_concept("margin") == "margin"
        assert _property_to_concept("border-radius") == "border radius"

    def test_typography(self):
        assert _property_to_concept("font-size") == "font size"
        assert _property_to_concept("text-align") == "text alignment"

    def test_vendor_prefix(self):
        assert _property_to_concept("-webkit-transform") == "transform"
        assert _property_to_concept("-moz-animation") == "animation"

    def test_unknown_property(self):
        assert _property_to_concept("nonexistent-property") is None

    def test_value_with_spaces(self):
        assert _property_to_concept("display", "  flex  ") == "flex layout"

    def test_value_with_semicolon(self):
        assert _property_to_concept("display", "flex;") == "flex layout"


# =============================================================================
# RULE SUMMARIES
# =============================================================================


class TestSummarizeRule:
    def test_simple_rule(self):
        result = summarize_rule(".navbar", "display: flex; padding: 1rem;")
        assert "Styles .navbar" in result
        assert "flex layout" in result
        assert "padding" in result

    def test_empty_rule(self):
        result = summarize_rule(".empty", "")
        assert "Empty rule for .empty" in result

    def test_many_properties(self):
        props = "; ".join([
            "display: flex", "padding: 1rem", "margin: 0",
            "border-radius: 4px", "box-shadow: none",
            "color: red", "background-color: white",
        ])
        result = summarize_rule(".box", props)
        assert "Styles .box" in result
        # Should truncate with "and N more"
        assert "more" in result or "and" in result

    def test_property_value_combo(self):
        result = summarize_rule(".flex", "display: flex;")
        assert "flex layout" in result
        # Should NOT have both "flex layout" and "hidden"

    def test_deduplicates_concepts(self):
        result = summarize_rule(".box", "padding: 1rem; padding-top: 2rem;")
        # "padding" should appear only once
        assert result.count("padding") == 1

    def test_custom_properties_skipped(self):
        result = summarize_rule(".root", "--color: red; display: flex;")
        # Custom properties should be ignored in rule summary
        assert "flex layout" in result


# =============================================================================
# @MEDIA SUMMARIES
# =============================================================================


class TestSummarizeMedia:
    def test_media_with_selectors(self):
        block = ".navbar { flex-direction: column; } .sidebar { display: none; }"
        result = summarize_media("(max-width: 768px)", block)
        assert "Responsive" in result
        assert ".navbar" in result

    def test_media_without_selectors(self):
        result = summarize_media("print", "")
        assert "Responsive styles for print" in result

    def test_media_many_selectors_truncated(self):
        selectors = " ".join(f".s{i} {{ color: red; }}" for i in range(10))
        result = summarize_media("(max-width: 768px)", selectors)
        assert "more" in result


# =============================================================================
# @KEYFRAMES SUMMARIES
# =============================================================================


class TestSummarizeKeyframes:
    def test_keyframes_with_properties(self):
        block = "from { opacity: 0; } to { opacity: 1; }"
        result = summarize_keyframes("fadeIn", block)
        assert "Animation 'fadeIn'" in result
        assert "opacity" in result

    def test_keyframes_no_properties(self):
        result = summarize_keyframes("empty", "")
        assert "Animation 'empty'" in result


# =============================================================================
# @FONT-FACE SUMMARIES
# =============================================================================


class TestSummarizeFontFace:
    def test_font_face_basic(self):
        block = "font-family: 'Inter'; src: url('/fonts/inter.woff2') format('woff2');"
        result = summarize_font_face(block)
        assert "Font face" in result
        assert "Inter" in result
        assert "woff2" in result

    def test_font_face_with_weight(self):
        block = "font-family: 'Roboto'; font-weight: 700; src: url('/fonts/roboto.ttf');"
        result = summarize_font_face(block)
        assert "Roboto" in result
        assert "700" in result

    def test_font_face_unknown_family(self):
        block = "src: url('/fonts/custom.woff');"
        result = summarize_font_face(block)
        assert "unknown" in result


# =============================================================================
# CUSTOM PROPERTIES SUMMARIES
# =============================================================================


class TestSummarizeCustomProperties:
    def test_root_properties(self):
        block = "--color-primary: #007bff; --spacing-md: 16px;"
        result = summarize_custom_properties(":root", block)
        assert "CSS custom properties on :root" in result
        assert "--color-primary" in result
        assert "--spacing-md" in result

    def test_many_properties_truncated(self):
        props = "; ".join(f"--var-{i}: val" for i in range(10))
        result = summarize_custom_properties(":root", props)
        assert "more" in result

    def test_empty_properties(self):
        result = summarize_custom_properties(":root", "")
        assert "Custom properties on :root" in result


# =============================================================================
# SCSS SUMMARIES
# =============================================================================


class TestSCSSVariableSummary:
    def test_simple_variable(self):
        result = summarize_scss_variable("$color", "#007bff")
        assert "SCSS variable $color" in result
        assert "#007bff" in result

    def test_long_value_truncated(self):
        long_val = "a" * 100
        result = summarize_scss_variable("$x", long_val)
        assert "..." in result


class TestSCSSMixinSummary:
    def test_mixin_with_properties(self):
        result = summarize_scss_mixin("flex-center", "", "display: flex; align-items: center;")
        assert "Mixin flex-center" in result
        assert "flex layout" in result

    def test_mixin_with_media(self):
        result = summarize_scss_mixin("responsive", "$bp", "@media (max-width: $bp) { @content; }")
        assert "Mixin responsive($bp)" in result
        assert "media query" in result

    def test_mixin_with_includes(self):
        result = summarize_scss_mixin("card", "", "@include shadow; padding: 1rem;")
        assert "includes shadow" in result

    def test_mixin_no_body(self):
        result = summarize_scss_mixin("empty", "", "")
        assert "Mixin empty" in result


class TestSCSSFunctionSummary:
    def test_function_with_return(self):
        result = summarize_scss_function("rem", "$px", "@return $px / 16 * 1rem;")
        assert "Function rem($px)" in result
        assert "returns" in result

    def test_function_no_return(self):
        result = summarize_scss_function("noop", "", "")
        assert "Function noop" in result


# =============================================================================
# FILE SUMMARY
# =============================================================================


class TestSummarizeFile:
    def test_css_file(self):
        assert summarize_file("styles.css", 5) == "CSS stylesheet with 5 sections"

    def test_scss_file(self):
        assert summarize_file("theme.scss", 3, "scss") == "SCSS stylesheet with 3 sections"

    def test_empty_file(self):
        assert summarize_file("empty.css", 0) == "Empty CSS stylesheet"

    def test_single_section(self):
        assert summarize_file("one.css", 1) == "CSS stylesheet with 1 section"


# =============================================================================
# JOIN LIST HELPER
# =============================================================================


class TestJoinList:
    def test_empty(self):
        assert _join_list([]) == ""

    def test_single(self):
        assert _join_list(["a"]) == "a"

    def test_two(self):
        assert _join_list(["a", "b"]) == "a and b"

    def test_three(self):
        assert _join_list(["a", "b", "c"]) == "a, b, and c"

    def test_many_truncated(self):
        result = _join_list(["a", "b", "c", "d", "e", "f", "g"])
        assert "and 2 more" in result


# =============================================================================
# AT-RULE SUMMARIES
# =============================================================================


class TestSummarizeAtRule:
    def test_import(self):
        result = summarize_at_rule("import", "'normalize.css'", None)
        assert "@import" in result

    def test_charset(self):
        result = summarize_at_rule("charset", "\"UTF-8\"", None)
        assert "@charset" in result

    def test_layer_with_content(self):
        result = summarize_at_rule("layer", "base", ".reset { margin: 0; }")
        assert "@layer base" in result
        assert "1 rule" in result

    def test_supports(self):
        result = summarize_at_rule("supports", "(display: grid)", ".grid { display: grid; }")
        assert "@supports" in result
        assert ".grid" in result

    def test_unknown_at_rule(self):
        result = summarize_at_rule("custom", "foo", "bar")
        assert "@custom foo" in result
