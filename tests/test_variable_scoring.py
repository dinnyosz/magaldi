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

    def test_has_high_examples(self):
        """System prompt should have HIGH scoring examples."""
        assert "MAX_RETRIES" in SYSTEM_PROMPT
        assert "9,1,1,8" in SYSTEM_PROMPT

    def test_has_low_examples(self):
        """System prompt should have LOW scoring examples (1,1,1,1)."""
        assert "1,1,1,1" in SYSTEM_PROMPT
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
        text = "<think>\nline 1\nline 2\nline 3\n</think>\n1. 9,1,1,8"
        result = _strip_think_tags(text)
        assert "<think>" not in result
        assert "1. 9,1,1,8" in result

    def test_strips_multiple_think_blocks(self):
        """Should strip multiple think blocks."""
        text = "<think>first</think>A<think>second</think>B"
        assert _strip_think_tags(text) == "AB"

    def test_preserves_text_without_think_tags(self):
        """Should preserve text that has no think tags."""
        text = "1. 9,1,1,8\n2. 1,1,1,1"
        assert _strip_think_tags(text) == text

    def test_returns_empty_for_only_think_content(self):
        """Should return empty string if output is only think tags."""
        text = "<think>all thinking, no output</think>"
        assert _strip_think_tags(text) == ""

    def test_handles_think_with_numbered_patterns(self):
        """Think blocks with score-like patterns should be stripped."""
        text = (
            "<think>Let me score:\n"
            "1. MAX_RETRIES = 3 -> high config value, score 8,1,1,7\n"
            "2. tmp = [] -> low, score 1,1,1,1\n"
            "</think>\n"
            "1. 8,1,1,7\n"
            "2. 1,1,1,1"
        )
        result = _strip_think_tags(text)
        assert "Let me score" not in result
        # Only the actual scores should remain
        lines = [line for line in result.strip().split("\n") if line.strip()]
        assert len(lines) == 2


# =============================================================================
# SCORE PARSING TESTS
# =============================================================================


class TestParseScores:
    """Tests for _parse_scores function."""

    def test_parses_standard_format(self):
        """Should parse standard numbered score format."""
        output = "1. 9,2,1,8\n2. 1,1,1,1\n3. 1,9,1,9"
        scores = _parse_scores(output, 3)
        assert scores[0] == VariableScore(9, 2, 1, 8)
        assert scores[1] == VariableScore(1, 1, 1, 1)
        assert scores[2] == VariableScore(1, 9, 1, 9)

    def test_returns_none_for_missing_scores(self):
        """Should return None for unparseable lines."""
        output = "1. 9,2,1,8\n3. 1,1,1,1"  # missing #2
        scores = _parse_scores(output, 3)
        assert scores[0] is not None
        assert scores[1] is None  # missing
        assert scores[2] is not None

    def test_clamps_scores_to_range(self):
        """Scores should be clamped to 1-10."""
        output = "1. 15,0,11,1"
        scores = _parse_scores(output, 1)
        assert scores[0] is not None
        assert scores[0].config_value == 10  # clamped from 15
        assert scores[0].architectural_role == 1  # clamped from 0 (min 1 via max(1,...))
        assert scores[0].data_definition == 10  # clamped from 11
        assert scores[0].general_usefulness == 1

    def test_handles_think_tags_in_output(self):
        """Should strip think tags before parsing scores."""
        output = (
            "<think>Reasoning:\n"
            "1. looks like a config constant\n"
            "2. just a temp var</think>\n"
            "1. 9,1,1,8\n"
            "2. 1,1,1,1"
        )
        scores = _parse_scores(output, 2)
        # After stripping think tags, should parse correctly
        assert scores[0] is not None
        assert scores[0].config_value == 9
        assert scores[1] is not None
        assert scores[1].config_value == 1

    def test_handles_extra_whitespace(self):
        """Should handle extra whitespace in score format."""
        output = "1.  9 , 2 , 1 , 8"
        scores = _parse_scores(output, 1)
        assert scores[0] == VariableScore(9, 2, 1, 8)

    def test_empty_output_returns_all_none(self):
        """Empty output should return all None scores."""
        scores = _parse_scores("", 3)
        assert all(s is None for s in scores)


# =============================================================================
# VARIABLE SCORE MODEL TESTS
# =============================================================================


class TestVariableScore:
    """Tests for VariableScore model."""

    def test_max_score(self):
        """max_score should return highest dimension."""
        score = VariableScore(3, 7, 2, 5)
        assert score.max_score == 7

    def test_passes_threshold_any_high(self):
        """Should pass if any dimension >= threshold."""
        score = VariableScore(1, 1, 1, 5)
        assert score.passes_threshold(5) is True

    def test_fails_threshold_all_low(self):
        """Should fail if all dimensions < threshold."""
        score = VariableScore(1, 1, 1, 4)
        assert score.passes_threshold(5) is False

    def test_default_threshold_is_5(self):
        """Default threshold should be 5."""
        score_pass = VariableScore(general_usefulness=5)
        assert score_pass.passes_threshold() is True

        score_fail = VariableScore(config_value=4, architectural_role=4,
                                   data_definition=4, general_usefulness=4)
        assert score_fail.passes_threshold() is False

    def test_as_tuple(self):
        """as_tuple should return 4-element tuple."""
        score = VariableScore(9, 2, 1, 8)
        assert score.as_tuple() == (9, 2, 1, 8)
