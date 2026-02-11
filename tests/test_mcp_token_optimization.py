"""Tests for MCP token optimization features.

Covers:
- readOnlyHint annotations on all tool schemas
- max_tokens truncation behavior
- filename save-to-file behavior
- Smart truncation preserves result completeness
- File summary generation for different result types
"""

from __future__ import annotations

import os
import tempfile

from magaldi_mcp.server import (
    _apply_filename,
    _apply_max_tokens,
    _build_result_hint,
    _estimate_tokens,
    _generate_file_summary,
)
from magaldi_mcp.tools.schemas import ALL_TOOL_SCHEMAS
from magaldi_mcp.tools.schemas._annotations import TOOLS_WITH_OUTPUT_CONTROL

# =============================================================================
# ANNOTATION TESTS
# =============================================================================


class TestToolAnnotations:
    """Verify readOnlyHint annotations are set on all tools."""

    def test_all_tools_have_annotations(self):
        """Every tool should have annotations set."""
        for tool in ALL_TOOL_SCHEMAS:
            assert tool.annotations is not None, (
                f"Tool '{tool.name}' is missing annotations"
            )

    def test_readonly_tools_have_correct_hints(self):
        """Read-only tools should have readOnlyHint=True."""
        write_tools = {"generate_skill", "generate_config", "parser_lab_create_test"}
        for tool in ALL_TOOL_SCHEMAS:
            if tool.name not in write_tools:
                assert tool.annotations is not None
                assert tool.annotations.readOnlyHint is True, (
                    f"Tool '{tool.name}' should have readOnlyHint=True"
                )
                assert tool.annotations.destructiveHint is False, (
                    f"Tool '{tool.name}' should have destructiveHint=False"
                )

    def test_write_tools_have_correct_hints(self):
        """Write tools should have readOnlyHint=False."""
        write_tools = {"generate_skill", "generate_config", "parser_lab_create_test"}
        for tool in ALL_TOOL_SCHEMAS:
            if tool.name in write_tools:
                assert tool.annotations is not None
                assert tool.annotations.readOnlyHint is False, (
                    f"Tool '{tool.name}' should have readOnlyHint=False"
                )

    def test_all_tools_are_idempotent(self):
        """All tools should be marked as idempotent."""
        for tool in ALL_TOOL_SCHEMAS:
            assert tool.annotations is not None
            assert tool.annotations.idempotentHint is True, (
                f"Tool '{tool.name}' should have idempotentHint=True"
            )

    def test_all_tools_are_closed_world(self):
        """All tools should be marked as closed-world (no external interaction)."""
        for tool in ALL_TOOL_SCHEMAS:
            assert tool.annotations is not None
            assert tool.annotations.openWorldHint is False, (
                f"Tool '{tool.name}' should have openWorldHint=False"
            )


# =============================================================================
# OUTPUT CONTROL PARAMETER TESTS
# =============================================================================


class TestOutputControlParams:
    """Verify max_tokens and filename params are on correct tools."""

    def test_output_control_tools_have_params(self):
        """All high-volume tools should have max_tokens and filename params."""
        for tool in ALL_TOOL_SCHEMAS:
            if tool.name in TOOLS_WITH_OUTPUT_CONTROL:
                props = tool.inputSchema.get("properties", {})
                assert "max_tokens" in props, (
                    f"Tool '{tool.name}' should have max_tokens parameter"
                )
                assert "filename" in props, (
                    f"Tool '{tool.name}' should have filename parameter"
                )

    def test_non_output_control_tools_lack_params(self):
        """Tools not in the output control set should NOT have these params."""
        for tool in ALL_TOOL_SCHEMAS:
            if tool.name not in TOOLS_WITH_OUTPUT_CONTROL:
                props = tool.inputSchema.get("properties", {})
                assert "max_tokens" not in props, (
                    f"Tool '{tool.name}' should NOT have max_tokens parameter"
                )
                assert "filename" not in props, (
                    f"Tool '{tool.name}' should NOT have filename parameter"
                )

    def test_output_control_tools_match_expected_set(self):
        """The TOOLS_WITH_OUTPUT_CONTROL set should match the planned tools."""
        expected = {
            "search_code",
            "search_features",
            "find_usages",
            "pattern_search",
            "find_callers",
            "find_call_chain",
            "explain_element",
            "find_dead_code",
            "find_entry_points",
        }
        assert expected == TOOLS_WITH_OUTPUT_CONTROL


# =============================================================================
# TOKEN ESTIMATION TESTS
# =============================================================================


class TestEstimateTokens:
    """Test token estimation function."""

    def test_empty_string(self):
        assert _estimate_tokens("") == 0

    def test_short_string(self):
        # 12 chars / 4 = 3 tokens
        assert _estimate_tokens("hello world!") == 3

    def test_longer_string(self):
        text = "a" * 400
        assert _estimate_tokens(text) == 100


# =============================================================================
# MAX_TOKENS TRUNCATION TESTS
# =============================================================================


class TestApplyMaxTokens:
    """Test max_tokens truncation behavior."""

    def test_no_truncation_when_under_budget(self):
        """Short text should pass through unchanged."""
        text = "hello world"
        result = _apply_max_tokens(text, 1000)
        assert result == text

    def test_truncation_drops_trailing_blocks(self):
        """Should drop trailing blocks to fit budget."""
        blocks = [f"Block {i}: content here padding text" for i in range(10)]
        text = "\n\n".join(blocks)
        # Budget for roughly 2 blocks
        result = _apply_max_tokens(text, 20)
        assert "Block 0" in result
        assert "more results omitted" in result

    def test_truncation_keeps_complete_blocks(self):
        """Each kept block should be complete, not cut mid-text."""
        blocks = [f"Block {i}: " + "x" * 50 for i in range(10)]
        text = "\n\n".join(blocks)
        result = _apply_max_tokens(text, 30)
        # Should not end mid-block content
        for block in result.split("\n\n"):
            if "more results omitted" not in block:
                assert block.startswith("Block ")

    def test_truncation_message_includes_budget(self):
        """Truncation message should include the token budget."""
        blocks = ["block " * 20 for _ in range(10)]
        text = "\n\n".join(blocks)
        result = _apply_max_tokens(text, 50)
        assert "token budget: 50" in result

    def test_single_block_truncation(self):
        """Single large block with no separators should truncate by chars."""
        text = "a" * 2000  # ~500 tokens
        result = _apply_max_tokens(text, 100)
        assert len(result) < 2000
        assert "truncated" in result

    def test_exact_budget_no_truncation(self):
        """Text exactly at budget should not be truncated."""
        text = "a" * 400  # 100 tokens
        result = _apply_max_tokens(text, 100)
        assert result == text

    def test_first_block_always_kept(self):
        """Even if the first block exceeds budget, it should be kept."""
        blocks = ["x" * 200, "y" * 10]  # First block is ~50 tokens
        text = "\n\n".join(blocks)
        result = _apply_max_tokens(text, 10)
        assert "x" * 200 in result


# =============================================================================
# FILENAME SAVE-TO-FILE TESTS
# =============================================================================


class TestApplyFilename:
    """Test filename save-to-file behavior."""

    def test_file_is_written(self):
        """Full formatted result should be written to the file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            tmpfile = f.name
        try:
            formatted = "# Results\n\nSome formatted content here"
            _apply_filename(formatted, tmpfile, "search_code", [])
            with open(tmpfile) as f:
                content = f.read()
            assert content == formatted
        finally:
            os.unlink(tmpfile)

    def test_returns_summary_not_content(self):
        """Should return a summary, not the full content."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            tmpfile = f.name
        try:
            formatted = "x" * 5000
            result = _apply_filename(formatted, tmpfile, "search_code", [{"name": "foo"}])
            assert len(result) < 500  # Much shorter than original
            assert "saved to" in result
            assert "Read tool" in result
        finally:
            os.unlink(tmpfile)

    def test_creates_parent_directories(self):
        """Should create parent directories if they don't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "sub", "dir", "results.md")
            _apply_filename("content", filepath, "search_code", [])
            assert os.path.exists(filepath)

    def test_summary_includes_filename(self):
        """Summary should reference the file path."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            tmpfile = f.name
        try:
            result = _apply_filename("content", tmpfile, "search_code", [])
            assert tmpfile in result
        finally:
            os.unlink(tmpfile)


# =============================================================================
# FILE SUMMARY GENERATION TESTS
# =============================================================================


class TestGenerateFileSummary:
    """Test file summary generation for different result types."""

    def test_list_result_shows_count_and_names(self):
        """List results should show count and top names."""
        result = [
            {"name": "foo_handler"},
            {"name": "bar_service"},
            {"name": "baz_util"},
            {"name": "qux_model"},
        ]
        summary = _generate_file_summary("search_code", result, "/tmp/test.md", "x" * 400)
        assert "4 results" in summary
        assert "foo_handler" in summary
        assert "bar_service" in summary
        assert "baz_util" in summary

    def test_explain_element_result(self):
        """Explain element results should show element name and counts."""
        result = {
            "element": {"name": "my_function", "type": "function"},
            "callers": [{"name": "a"}, {"name": "b"}],
            "callees": [{"name": "c"}],
        }
        summary = _generate_file_summary("explain_element", result, "/tmp/test.md", "x" * 400)
        assert "my_function" in summary
        assert "Callers: 2" in summary
        assert "Callees: 1" in summary

    def test_dead_code_result(self):
        """Dead code results should show count of dead functions."""
        result = {
            "potentially_dead": [{"name": "unused1"}, {"name": "unused2"}],
            "stats": {"total": 100},
        }
        summary = _generate_file_summary("find_dead_code", result, "/tmp/test.md", "x" * 400)
        assert "2 potentially dead" in summary

    def test_grouped_result(self):
        """Grouped results (code + test) should show both counts."""
        result = {
            "code_results": [{"name": "a"}, {"name": "b"}],
            "test_results": [{"name": "c"}],
        }
        summary = _generate_file_summary("find_callers", result, "/tmp/test.md", "x" * 400)
        assert "3 results" in summary
        assert "2 code" in summary
        assert "1 test" in summary

    def test_summary_includes_token_estimate(self):
        """Summary should include approximate token count."""
        result = []
        formatted = "x" * 2000  # ~500 tokens
        summary = _generate_file_summary("search_code", result, "/tmp/test.md", formatted)
        assert "~500 tokens" in summary

    def test_summary_always_includes_read_instruction(self):
        """Every summary should tell the agent to use Read tool."""
        result = {"some": "data"}
        summary = _generate_file_summary("search_code", result, "/tmp/test.md", "content")
        assert "Read tool" in summary

    def test_empty_list_result(self):
        """Empty list should still produce valid summary."""
        summary = _generate_file_summary("search_code", [], "/tmp/test.md", "")
        assert "0 results" in summary

    def test_list_with_element_name_key(self):
        """Should handle results with 'element_name' key instead of 'name'."""
        result = [{"element_name": "alpha"}, {"element_name": "beta"}]
        summary = _generate_file_summary("search_code", result, "/tmp/test.md", "x" * 100)
        assert "alpha" in summary
        assert "beta" in summary


# =============================================================================
# RESULT HINT TESTS
# =============================================================================


class TestBuildResultHint:
    """Test narrowing hints when results hit the limit."""

    def test_hint_when_list_hits_limit(self):
        """Should return hint when list result count equals limit."""
        result = [{"name": f"item_{i}"} for i in range(30)]
        hint = _build_result_hint("find_usages", result, {"limit": 30})
        assert hint is not None
        assert "30 results" in hint
        assert "limit reached" in hint

    def test_no_hint_when_under_limit(self):
        """Should return None when results are under the limit."""
        result = [{"name": "item_0"}]
        hint = _build_result_hint("find_usages", result, {"limit": 30})
        assert hint is None

    def test_no_hint_for_unknown_tool(self):
        """Should return None for tools not in the narrowing map."""
        result = [{"name": "item"} for _ in range(50)]
        hint = _build_result_hint("get_element", result, {"limit": 50})
        assert hint is None

    def test_no_hint_when_no_limit_arg(self):
        """Should return None when no limit was specified."""
        result = [{"name": "item"} for _ in range(50)]
        hint = _build_result_hint("search_code", result, {})
        assert hint is None

    def test_hint_for_grouped_results(self):
        """Should count code_results + test_results for grouped tools."""
        result = {
            "code_results": [{"name": f"c{i}"} for i in range(40)],
            "test_results": [{"name": f"t{i}"} for i in range(10)],
        }
        hint = _build_result_hint("pattern_search", result, {"limit": 50})
        assert hint is not None
        assert "50 results" in hint

    def test_hint_includes_tool_specific_params(self):
        """Hint should mention the right narrowing params for each tool."""
        result = [{"name": f"item_{i}"} for i in range(50)]
        hint = _build_result_hint("pattern_search", result, {"limit": 50})
        assert "glob" in hint

    def test_hint_for_dead_code(self):
        """Should handle dead code result shape."""
        result = {"potentially_dead": [{"name": f"fn{i}"} for i in range(30)]}
        hint = _build_result_hint("find_dead_code", result, {"limit": 30})
        assert hint is not None
        assert "30 results" in hint
