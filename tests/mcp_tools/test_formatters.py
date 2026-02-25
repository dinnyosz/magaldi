"""Tests for MCP result formatters - GroupedSearchFormatter."""

from __future__ import annotations

from magaldi_mcp.formatters.search import GroupedSearchFormatter


class TestGroupedSearchFormatter:
    """Tests for the compact file-grouped search formatter."""

    def setup_method(self):
        self.formatter = GroupedSearchFormatter()

    def _make_result(
        self,
        code_results: list | None = None,
        test_results: list | None = None,
    ) -> dict:
        return {
            "code_results": code_results or [],
            "test_results": test_results or [],
            "total_code": len(code_results or []),
            "total_tests": len(test_results or []),
        }

    def _make_entry(
        self,
        name: str = "foo",
        elem_type: str = "function",
        file: str = "src/app.py",
        line: int = 10,
        line_end: int | None = 20,
        score: float = 0.85,
        hash_id: str = "abcd1234",
        **kwargs,
    ) -> dict:
        entry = {
            "name": name,
            "type": elem_type,
            "file": file,
            "line": line,
            "line_end": line_end,
            "score": score,
            "hash_id": hash_id,
            "is_test": False,
            **kwargs,
        }
        return entry

    def test_can_format_valid_result(self):
        result = self._make_result(code_results=[self._make_entry()])
        assert self.formatter.can_format(result) is True

    def test_can_format_rejects_non_dict(self):
        assert self.formatter.can_format([]) is False
        assert self.formatter.can_format("string") is False

    def test_can_format_rejects_target_key(self):
        """Results with 'target' key belong to FindCallersFormatter."""
        result = self._make_result(code_results=[self._make_entry()])
        result["target"] = "something"
        assert self.formatter.can_format(result) is False

    def test_empty_results(self):
        result = self._make_result()
        output = self.formatter.format(result)
        assert output == "No results found."

    def test_header_format(self):
        result = self._make_result(code_results=[self._make_entry()])
        output = self.formatter.format(result)
        lines = output.split("\n")
        assert lines[0] == "# search_code: 1 results (scored 0-1, desc)"
        assert lines[1] == "# type:name:L<start>[-end]:score|<hash8>"

    def test_file_grouping(self):
        """Elements from the same file are grouped together."""
        entries = [
            self._make_entry(name="foo", file="src/a.py", score=0.9, hash_id="aaaa1111"),
            self._make_entry(name="bar", file="src/b.py", score=0.8, hash_id="bbbb2222"),
            self._make_entry(name="baz", file="src/a.py", score=0.7, hash_id="cccc3333"),
        ]
        result = self._make_result(code_results=entries)
        output = self.formatter.format(result)

        # src/a.py should appear once (not twice)
        assert output.count("src/a.py") == 1
        assert output.count("src/b.py") == 1

        # Both elements should be under src/a.py
        a_section = output.split("src/a.py")[1].split("src/b.py")[0]
        assert "foo" in a_section
        assert "baz" in a_section

    def test_score_sorting_files_by_avg(self):
        """Files are sorted by average score descending."""
        entries = [
            self._make_entry(name="low", file="src/low.py", score=0.3),
            self._make_entry(name="high", file="src/high.py", score=0.9),
        ]
        result = self._make_result(code_results=entries)
        output = self.formatter.format(result)

        # high.py should appear before low.py
        assert output.index("src/high.py") < output.index("src/low.py")

    def test_score_sorting_elements_within_file(self):
        """Elements within a file are sorted by score descending."""
        entries = [
            self._make_entry(name="low", file="src/app.py", score=0.3, hash_id="aaaa1111"),
            self._make_entry(name="high", file="src/app.py", score=0.9, hash_id="bbbb2222"),
        ]
        result = self._make_result(code_results=entries)
        output = self.formatter.format(result)

        # high should appear before low within the file
        assert output.index("high") < output.index("low")

    def test_avg_score_displayed(self):
        """File groups show average score."""
        entries = [
            self._make_entry(name="a", file="src/app.py", score=0.80, hash_id="aaaa1111"),
            self._make_entry(name="b", file="src/app.py", score=0.60, hash_id="bbbb2222"),
        ]
        result = self._make_result(code_results=entries)
        output = self.formatter.format(result)
        assert "avg:0.70" in output

    def test_type_abbreviations(self):
        """All element types are abbreviated correctly."""
        test_cases = {
            "function": "fn",
            "method": "mth",
            "class": "cls",
            "constant": "const",
            "variable": "var",
            "interface": "iface",
            "enum": "enum",
            "type_alias": "type",
            "import": "imp",
            "trait": "trait",
        }
        for full_type, abbrev in test_cases.items():
            entries = [self._make_entry(name="x", elem_type=full_type)]
            result = self._make_result(code_results=entries)
            output = self.formatter.format(result)
            assert f"  {abbrev}:x:" in output, f"Expected {abbrev} for {full_type}"

    def test_unknown_type_passes_through(self):
        """Unknown types are not abbreviated, just passed through."""
        entries = [self._make_entry(name="x", elem_type="unknown_thing")]
        result = self._make_result(code_results=entries)
        output = self.formatter.format(result)
        assert "  unknown_thing:x:" in output

    def test_line_range_with_end(self):
        """Line range shows start-end when line_end is present."""
        entries = [self._make_entry(line=42, line_end=78)]
        result = self._make_result(code_results=entries)
        output = self.formatter.format(result)
        assert "L42-78" in output

    def test_line_range_without_end(self):
        """Line range shows just start when line_end is None."""
        entries = [self._make_entry(line=42, line_end=None)]
        result = self._make_result(code_results=entries)
        output = self.formatter.format(result)
        assert "L42:" in output
        assert "L42-" not in output

    def test_hash_id_in_output(self):
        """8-char hash ID appears after score."""
        entries = [self._make_entry(hash_id="b1d0e591", score=0.91)]
        result = self._make_result(code_results=entries)
        output = self.formatter.format(result)
        assert "0.91|b1d0e591" in output

    def test_brief_mode_no_summary(self):
        """Brief mode (no summary key) doesn't show summary lines."""
        entries = [self._make_entry()]  # No summary key
        result = self._make_result(code_results=entries)
        output = self.formatter.format(result)
        lines = output.strip().split("\n")
        # Should be: header(2) + blank + file_header + element = 5 lines
        # No indented summary line
        indented_lines = [line for line in lines if line.startswith("    ")]
        assert len(indented_lines) == 0

    def test_non_brief_with_summary(self):
        """Non-brief results show summary indented under element."""
        entries = [self._make_entry(summary="Process input data and validate")]
        result = self._make_result(code_results=entries)
        output = self.formatter.format(result)
        assert "    Process input data and validate" in output

    def test_non_brief_omits_signature(self):
        """Compact format omits signature (available via get_element)."""
        entries = [self._make_entry(signature="def foo(x: int) -> str")]
        result = self._make_result(code_results=entries)
        output = self.formatter.format(result)
        # Signature is NOT included in compact search output
        assert "def foo(x: int) -> str" not in output

    def test_summary_truncation(self):
        """Summaries longer than 200 chars are truncated."""
        long_summary = "x" * 250
        entries = [self._make_entry(summary=long_summary)]
        result = self._make_result(code_results=entries)
        output = self.formatter.format(result)
        assert "..." in output
        # Find the summary line
        summary_line = [line for line in output.split("\n") if line.startswith("    x")][0]
        # 4 spaces indent + 200 chars + "..." = 207
        assert len(summary_line) == 207

    def test_code_omitted_in_compact(self):
        """Compact format omits code blocks (available via get_element)."""
        entries = [self._make_entry(code="def foo():\n    return 42")]
        result = self._make_result(code_results=entries)
        output = self.formatter.format(result)
        # Code is NOT included in compact search output
        assert "```" not in output
        assert "def foo():" not in output

    def test_mixed_code_and_test_results(self):
        """Code and test results are merged into same file groups."""
        code = [self._make_entry(name="process", file="src/app.py", hash_id="aaaa1111")]
        tests = [self._make_entry(
            name="test_process", file="tests/test_app.py",
            hash_id="bbbb2222", is_test=True,
        )]
        result = self._make_result(code_results=code, test_results=tests)
        output = self.formatter.format(result)

        assert "# search_code: 2 results" in output
        assert "src/app.py" in output
        assert "tests/test_app.py" in output
        # No separate "Test Results" section
        assert "Test Results" not in output

    def test_compact_element_format(self):
        """Verify the exact compact format of an element line."""
        entries = [self._make_entry(
            name="Parser.extract",
            elem_type="method",
            file="src/parser.py",
            line=128,
            line_end=156,
            score=0.80,
            hash_id="7baec454",
        )]
        result = self._make_result(code_results=entries)
        output = self.formatter.format(result)
        assert "  mth:Parser.extract:L128-156:0.80|7baec454" in output

    def test_pattern_search_field_names(self):
        """Handles pattern_search field names (element_type, line_start)."""
        entries = [{
            "element_id": "test:id",
            "hash_id": "aabb1122",
            "file": "src/app.py",
            "name": "process",
            "element_type": "function",
            "line_start": 10,
            "raw_code": "def process(): pass",
            "is_test": False,
        }]
        result = {
            "code_results": entries,
            "test_results": [],
            "totals": {"code": 1, "tests": 0},
            "mode": "regexp",
            "pattern": "process",
        }
        output = self.formatter.format(result)
        assert "fn:process:L10:" in output
        # Compact format omits code blocks
        assert "```" not in output
