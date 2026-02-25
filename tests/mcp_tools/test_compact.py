"""Tests for shared compact formatting utilities."""

from __future__ import annotations

from magaldi_mcp.formatters._compact import (
    HASH_LEN,
    SUMMARY_MAX,
    SUMMARY_SHORT,
    TYPE_ABBREV,
    abbrev_type,
    compact_element,
    compact_element_header,
    compact_ref,
    compact_unresolved,
    file_group,
    file_path,
    indented_summary,
    line_range,
    truncate_hash,
    truncate_summary,
)


class TestTypeAbbrev:
    """Tests for TYPE_ABBREV dict and abbrev_type()."""

    def test_all_standard_types(self):
        expected = {
            "function": "fn",
            "method": "mth",
            "class": "cls",
            "constant": "const",
            "variable": "var",
            "interface": "iface",
            "enum": "enum",
            "type_alias": "type",
            "import": "imp",
            "file": "file",
            "trait": "trait",
        }
        for full_type, abbrev in expected.items():
            assert abbrev_type(full_type) == abbrev

    def test_unknown_type_passes_through(self):
        assert abbrev_type("decorator") == "decorator"
        assert abbrev_type("unknown_thing") == "unknown_thing"

    def test_none_type(self):
        assert abbrev_type(None) == "?"

    def test_empty_string(self):
        assert abbrev_type("") == "?"


class TestTruncateHash:
    def test_normal_hash(self):
        assert truncate_hash("abcdef1234567890") == "abcdef12"
        assert len(truncate_hash("abcdef1234567890")) == HASH_LEN

    def test_short_hash(self):
        assert truncate_hash("abc") == "abc"

    def test_exact_length(self):
        assert truncate_hash("12345678") == "12345678"

    def test_none(self):
        assert truncate_hash(None) == ""

    def test_empty(self):
        assert truncate_hash("") == ""


class TestLineRange:
    def test_with_line_and_line_end(self):
        assert line_range({"line": 42, "line_end": 78}) == "L42-78"

    def test_with_line_start_variant(self):
        assert line_range({"line_start": 10, "line_end": 20}) == "L10-20"

    def test_line_preferred_over_line_start(self):
        assert line_range({"line": 42, "line_start": 10, "line_end": 78}) == "L42-78"

    def test_without_line_end(self):
        assert line_range({"line": 42}) == "L42"

    def test_same_start_and_end(self):
        assert line_range({"line": 42, "line_end": 42}) == "L42"

    def test_missing_line(self):
        assert line_range({}) == "L?"


class TestFilePath:
    def test_file_key(self):
        assert file_path({"file": "src/app.py"}) == "src/app.py"

    def test_relative_path_key(self):
        assert file_path({"relative_path": "src/app.py"}) == "src/app.py"

    def test_file_preferred(self):
        assert file_path({"file": "a.py", "relative_path": "b.py"}) == "a.py"

    def test_missing(self):
        assert file_path({}) == "unknown"


class TestCompactElement:
    def _make(self, **kwargs):
        base = {
            "name": "process",
            "type": "function",
            "file": "src/app.py",
            "line": 42,
            "line_end": 78,
            "hash_id": "abcd1234",
        }
        base.update(kwargs)
        return base

    def test_basic_format(self):
        result = compact_element(self._make())
        assert result == "  fn:process:L42-78|abcd1234"

    def test_with_score(self):
        result = compact_element(self._make(score=0.85), score=True)
        assert result == "  fn:process:L42-78:0.85|abcd1234"

    def test_custom_indent(self):
        result = compact_element(self._make(), indent=4)
        assert result.startswith("    fn:")

    def test_zero_indent(self):
        result = compact_element(self._make(), indent=0)
        assert result == "fn:process:L42-78|abcd1234"

    def test_element_type_variant(self):
        """Handles 'element_type' field name."""
        r = self._make()
        del r["type"]
        r["element_type"] = "method"
        result = compact_element(r)
        assert "mth:process:" in result

    def test_no_line_end(self):
        result = compact_element(self._make(line_end=None))
        assert "L42|" in result
        assert "L42-" not in result

    def test_missing_hash(self):
        r = self._make()
        del r["hash_id"]
        result = compact_element(r)
        assert result.endswith("|")


class TestCompactRef:
    def _make(self, **kwargs):
        base = {
            "name": "caller_fn",
            "type": "function",
            "file": "src/caller.py",
            "line": 15,
            "hash_id": "eeee5555",
        }
        base.update(kwargs)
        return base

    def test_caller_arrow(self):
        result = compact_ref(self._make(), arrow="←")
        assert result == "  ← fn:caller_fn:L15|eeee5555"

    def test_callee_arrow(self):
        result = compact_ref(self._make(), arrow="→")
        assert result == "  → fn:caller_fn:L15|eeee5555"

    def test_similar_arrow(self):
        result = compact_ref(self._make(score=0.92), arrow="~", score=True)
        assert result == "  ~ fn:caller_fn:L15:0.92|eeee5555"

    def test_no_arrow(self):
        result = compact_ref(self._make())
        assert result == "  fn:caller_fn:L15|eeee5555"

    def test_custom_indent(self):
        result = compact_ref(self._make(), arrow="←", indent=4)
        assert result.startswith("    ← ")


class TestCompactUnresolved:
    def test_with_receiver(self):
        r = {"receiver": "MyClass", "name": "do_thing", "line": 42}
        result = compact_unresolved(r)
        assert result == "  → ?:MyClass.do_thing():L42"

    def test_without_receiver(self):
        r = {"name": "standalone", "line": 10}
        result = compact_unresolved(r)
        assert result == "  → ?:standalone():L10"

    def test_custom_indent(self):
        r = {"name": "foo", "line": 1}
        result = compact_unresolved(r, indent=4)
        assert result.startswith("    → ?:")


class TestCompactElementHeader:
    def test_no_indent(self):
        r = {"name": "X", "type": "class", "line": 1, "hash_id": "aabb1122"}
        result = compact_element_header(r)
        assert result == "cls:X:L1|aabb1122"
        assert not result.startswith(" ")


class TestTruncateSummary:
    def test_short_summary(self):
        assert truncate_summary("hello") == "hello"

    def test_long_summary(self):
        long = "x" * 250
        result = truncate_summary(long)
        assert len(result) == SUMMARY_MAX + 3  # +3 for "..."
        assert result.endswith("...")

    def test_custom_max(self):
        result = truncate_summary("x" * 150, max_len=SUMMARY_SHORT)
        assert len(result) == SUMMARY_SHORT + 3

    def test_none(self):
        assert truncate_summary(None) is None

    def test_empty(self):
        assert truncate_summary("") is None


class TestIndentedSummary:
    def test_basic(self):
        r = {"summary": "Process data and return results"}
        result = indented_summary(r)
        assert result == "    Process data and return results"

    def test_custom_indent(self):
        r = {"summary": "Hello"}
        result = indented_summary(r, indent=2)
        assert result == "  Hello"

    def test_truncation(self):
        r = {"summary": "x" * 250}
        result = indented_summary(r)
        assert result is not None
        assert result.endswith("...")
        # 4 indent + 200 chars + "..." = 207
        assert len(result) == 207

    def test_no_summary(self):
        assert indented_summary({}) is None
        assert indented_summary({"summary": ""}) is None
        assert indented_summary({"summary": None}) is None


class TestFileGroup:
    def _make(self, name, file, score=0.5, hash_id="aaaa1111"):
        return {
            "name": name,
            "type": "function",
            "file": file,
            "line": 1,
            "score": score,
            "hash_id": hash_id,
        }

    def test_groups_by_file(self):
        results = [
            self._make("a", "src/x.py", hash_id="a1"),
            self._make("b", "src/y.py", hash_id="b1"),
            self._make("c", "src/x.py", hash_id="c1"),
        ]
        groups = file_group(results)
        paths = [g[0] for g in groups]
        assert "src/x.py" in paths
        assert "src/y.py" in paths
        # x.py group should have 2 elements
        x_group = next(g for g in groups if g[0] == "src/x.py")
        assert len(x_group[2]) == 2

    def test_sorted_by_avg_score_desc(self):
        results = [
            self._make("low", "src/low.py", score=0.2),
            self._make("high", "src/high.py", score=0.9),
        ]
        groups = file_group(results)
        assert groups[0][0] == "src/high.py"
        assert groups[1][0] == "src/low.py"

    def test_elements_sorted_within_group(self):
        results = [
            self._make("low", "src/x.py", score=0.3, hash_id="lo"),
            self._make("high", "src/x.py", score=0.9, hash_id="hi"),
        ]
        groups = file_group(results)
        elements = groups[0][2]
        assert elements[0]["name"] == "high"
        assert elements[1]["name"] == "low"

    def test_avg_score_calculated(self):
        results = [
            self._make("a", "src/x.py", score=0.8, hash_id="a1"),
            self._make("b", "src/x.py", score=0.6, hash_id="b1"),
        ]
        groups = file_group(results)
        avg = groups[0][1]
        assert abs(avg - 0.7) < 0.001

    def test_empty_results(self):
        assert file_group([]) == []
