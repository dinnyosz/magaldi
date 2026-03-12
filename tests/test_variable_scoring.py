"""Tests for variable scoring module."""

from __future__ import annotations

from magaldi_core.variable_scoring import _parse_scores, _strip_think_tags
from magaldi_core.variable_scoring.models import VariableScore
from magaldi_core.variable_scoring.prompts import SYSTEM_PROMPT, build_user_prompt

# =============================================================================
# PROMPT TESTS
# =============================================================================


class TestBuildUserPrompt:
    """Tests for build_user_prompt function."""

    def test_includes_no_think_directive(self):
        """Prompt should start with /no_think to disable Qwen3 thinking."""
        variables = [(1, "src/main.py", "MAX_RETRIES", "MAX_RETRIES = 3")]
        prompt = build_user_prompt(variables)
        assert prompt.startswith("/no_think\n")

    def test_formats_variables_with_index_and_path(self):
        """Variables should be formatted as 'idx. [path] code'."""
        variables = [
            (1, "src/main.py", "MAX_RETRIES", "MAX_RETRIES = 3"),
            (2, "src/utils.py", "tmp", "tmp = []"),
        ]
        prompt = build_user_prompt(variables)
        assert '1. [src/main.py] MAX_RETRIES = 3' in prompt
        assert '2. [src/utils.py] tmp = []' in prompt

    def test_no_truncation_within_budget(self):
        """Code under the token budget should NOT be truncated."""
        long_code = "x = " + "a" * 400
        variables = [(1, "test.py", "x", long_code)]
        prompt = build_user_prompt(variables)
        lines = prompt.split("\n")
        code_line = lines[-1]
        # 400 chars is well within default budget (1200 tokens ≈ 4600 chars)
        assert not code_line.endswith("...")
        code_part = code_line.split("] ", 1)[1]
        assert code_part == long_code

    def test_truncates_when_exceeding_budget(self):
        """Code exceeding the token budget should be truncated."""
        # Use a tiny budget to force truncation
        long_code = "x = " + "a" * 2000
        variables = [(1, "test.py", "x", long_code)]
        prompt = build_user_prompt(variables, token_budget=200)
        lines = prompt.split("\n")
        code_line = lines[-1]
        assert code_line.endswith("...")
        code_part = code_line.split("] ", 1)[1]
        assert len(code_part) <= 200 * 4  # token_budget * 4 chars/token

    def test_replaces_newlines_in_code(self):
        """Newlines in raw code should be replaced with spaces."""
        variables = [(1, "test.py", "x", "x = {\n  'a': 1,\n  'b': 2\n}")]
        prompt = build_user_prompt(variables)
        assert "\n  'a'" not in prompt  # newline replaced


class TestSystemPrompt:
    """Tests for the system prompt template."""

    def test_target_keep_rate_mentioned(self):
        """System prompt should mention ~30% keep rate target."""
        assert "30%" in SYSTEM_PROMPT

    def test_has_binary_format(self):
        """System prompt should use KEEP/DROP binary format."""
        assert "KEEP" in SYSTEM_PROMPT
        assert "DROP" in SYSTEM_PROMPT

    def test_has_keep_examples(self):
        """System prompt should have KEEP examples."""
        assert "MAX_RETRIES" in SYSTEM_PROMPT
        assert "1. KEEP" in SYSTEM_PROMPT

    def test_has_drop_examples(self):
        """System prompt should have DROP examples."""
        assert "2. DROP" in SYSTEM_PROMPT
        assert "result = process_items(data)" in SYSTEM_PROMPT


# =============================================================================
# THINK TAG STRIPPING TESTS
# =============================================================================


class TestStripThinkTags:
    """Tests for _strip_think_tags function."""

    def test_strips_simple_think_block(self):
        """Should strip a simple <think>...</think> block."""
        text = "<think>reasoning here</think>actual output"
        assert _strip_think_tags(text) == "actual output"

    def test_strips_multiline_think_block(self):
        """Should strip think blocks spanning multiple lines."""
        text = "<think>\nline 1\nline 2\nline 3\n</think>\n1. KEEP"
        result = _strip_think_tags(text)
        assert "<think>" not in result
        assert "1. KEEP" in result

    def test_strips_multiple_think_blocks(self):
        """Should strip multiple think blocks."""
        text = "<think>first</think>A<think>second</think>B"
        assert _strip_think_tags(text) == "AB"

    def test_preserves_text_without_think_tags(self):
        """Should preserve text that has no think tags."""
        text = "1. KEEP\n2. DROP"
        assert _strip_think_tags(text) == text

    def test_returns_empty_for_only_think_content(self):
        """Should return empty string if output is only think tags."""
        text = "<think>all thinking, no output</think>"
        assert _strip_think_tags(text) == ""

    def test_handles_think_with_numbered_patterns(self):
        """Think blocks with decision-like patterns should be stripped."""
        text = (
            "<think>Let me decide:\n"
            "1. MAX_RETRIES = 3 -> config constant, KEEP\n"
            "2. tmp = [] -> temp var, DROP\n"
            "</think>\n"
            "1. KEEP\n"
            "2. DROP"
        )
        result = _strip_think_tags(text)
        assert "Let me decide" not in result
        # Only the actual decisions should remain
        lines = [line for line in result.strip().split("\n") if line.strip()]
        assert len(lines) == 2


# =============================================================================
# SCORE PARSING TESTS
# =============================================================================


class TestParseScores:
    """Tests for _parse_scores function."""

    def test_parses_keep_drop_format(self):
        """Should parse binary KEEP/DROP format."""
        output = "1. KEEP\n2. DROP\n3. KEEP"
        scores = _parse_scores(output, 3)
        assert scores[0] == VariableScore(keep=True)
        assert scores[1] == VariableScore(keep=False)
        assert scores[2] == VariableScore(keep=True)

    def test_case_insensitive(self):
        """Should handle case variations."""
        output = "1. keep\n2. Drop\n3. KEEP"
        scores = _parse_scores(output, 3)
        assert scores[0] is not None and scores[0].keep is True
        assert scores[1] is not None and scores[1].keep is False
        assert scores[2] is not None and scores[2].keep is True

    def test_returns_none_for_missing_scores(self):
        """Should return None for unparseable lines."""
        output = "1. KEEP\n3. DROP"  # missing #2
        scores = _parse_scores(output, 3)
        assert scores[0] is not None
        assert scores[1] is None  # missing
        assert scores[2] is not None

    def test_handles_think_tags_in_output(self):
        """Should strip think tags before parsing decisions."""
        output = (
            "<think>Reasoning:\n"
            "1. looks like a config constant\n"
            "2. just a temp var</think>\n"
            "1. KEEP\n"
            "2. DROP"
        )
        scores = _parse_scores(output, 2)
        assert scores[0] is not None and scores[0].keep is True
        assert scores[1] is not None and scores[1].keep is False

    def test_handles_extra_whitespace(self):
        """Should handle extra whitespace in format."""
        output = "1.  KEEP\n2.   DROP"
        scores = _parse_scores(output, 2)
        assert scores[0] is not None and scores[0].keep is True
        assert scores[1] is not None and scores[1].keep is False

    def test_empty_output_returns_all_none(self):
        """Empty output should return all None scores."""
        scores = _parse_scores("", 3)
        assert all(s is None for s in scores)


# =============================================================================
# VARIABLE SCORE MODEL TESTS
# =============================================================================


class TestVariableScore:
    """Tests for VariableScore model."""

    def test_default_is_keep(self):
        """Default should be keep=True (safe default)."""
        score = VariableScore()
        assert score.keep is True

    def test_keep_passes_threshold(self):
        """KEEP variables should pass threshold."""
        score = VariableScore(keep=True)
        assert score.passes_threshold() is True

    def test_drop_fails_threshold(self):
        """DROP variables should fail threshold."""
        score = VariableScore(keep=False)
        assert score.passes_threshold() is False

    def test_threshold_arg_ignored(self):
        """Threshold argument should be ignored in binary scoring."""
        assert VariableScore(keep=True).passes_threshold(99) is True
        assert VariableScore(keep=False).passes_threshold(0) is False

    def test_verdict_property(self):
        """verdict should return 'KEEP' or 'DROP'."""
        assert VariableScore(keep=True).verdict == "KEEP"
        assert VariableScore(keep=False).verdict == "DROP"
