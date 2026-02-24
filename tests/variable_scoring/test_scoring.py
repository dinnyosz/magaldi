"""Tests for the variable scoring module.

Tests cover models, prompt building, score parsing, batching,
and the main score_variables orchestrator.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from magaldi_core.variable_scoring import (
    _build_batches,
    _estimate_tokens,
    _parse_scores,
    _score_batch,
    score_variables,
)
from magaldi_core.variable_scoring.models import (
    ScoringResult,
    VariableScore,
    VariableScoringConfig,
)
from magaldi_core.variable_scoring.prompts import SYSTEM_PROMPT, build_user_prompt

# =============================================================================
# VariableScore model tests
# =============================================================================


class TestVariableScore:
    """Tests for VariableScore dataclass."""

    def test_default_scores_are_one(self):
        score = VariableScore()
        assert score.config_value == 1
        assert score.architectural_role == 1
        assert score.data_definition == 1
        assert score.general_usefulness == 1

    def test_max_score(self):
        score = VariableScore(config_value=3, architectural_role=7, data_definition=2, general_usefulness=5)
        assert score.max_score == 7

    def test_max_score_all_equal(self):
        score = VariableScore(config_value=4, architectural_role=4, data_definition=4, general_usefulness=4)
        assert score.max_score == 4

    def test_passes_threshold_default(self):
        # Default threshold is 5; any dimension >= 5 passes
        assert VariableScore(general_usefulness=5).passes_threshold()
        assert not VariableScore(general_usefulness=4).passes_threshold()

    def test_passes_threshold_custom(self):
        score = VariableScore(config_value=3)
        assert score.passes_threshold(threshold=3)
        assert not score.passes_threshold(threshold=4)

    def test_passes_threshold_any_dimension(self):
        # Only config_value is high, rest are 1
        assert VariableScore(config_value=8).passes_threshold()
        # Only architectural_role is high
        assert VariableScore(architectural_role=6).passes_threshold()
        # Only data_definition is high
        assert VariableScore(data_definition=10).passes_threshold()

    def test_fails_threshold_all_low(self):
        score = VariableScore(config_value=2, architectural_role=3, data_definition=4, general_usefulness=4)
        assert not score.passes_threshold()

    def test_as_tuple(self):
        score = VariableScore(config_value=1, architectural_role=2, data_definition=3, general_usefulness=4)
        assert score.as_tuple() == (1, 2, 3, 4)


class TestVariableScoringConfig:
    """Tests for VariableScoringConfig defaults."""

    def test_defaults(self):
        config = VariableScoringConfig()
        assert config.threshold == 5
        assert config.temperature == 0.1
        assert config.token_budget == 800
        assert config.max_retries == 2


class TestScoringResult:
    """Tests for ScoringResult dataclass."""

    def test_defaults(self):
        result = ScoringResult()
        assert result.total_variables == 0
        assert result.kept == 0
        assert result.dropped == 0
        assert result.batch_count == 0
        assert result.elapsed == 0.0
        assert result.errors == 0
        assert result.scores == {}


# =============================================================================
# Prompt building tests
# =============================================================================


class TestBuildUserPrompt:
    """Tests for build_user_prompt."""

    def test_basic_prompt(self):
        variables = [
            (1, "src/config.py", "MAX_RETRIES", "MAX_RETRIES = 3"),
        ]
        prompt = build_user_prompt(variables)
        assert "Score these variables:" in prompt
        assert "1. [src/config.py] MAX_RETRIES = 3" in prompt

    def test_multiple_variables(self):
        variables = [
            (1, "src/a.py", "x", "x = 1"),
            (2, "src/b.py", "y", "y = 'hello'"),
        ]
        prompt = build_user_prompt(variables)
        assert "1. [src/a.py] x = 1" in prompt
        assert "2. [src/b.py] y = 'hello'" in prompt

    def test_truncates_long_code(self):
        long_code = "x = " + "a" * 200
        variables = [(1, "f.py", "x", long_code)]
        prompt = build_user_prompt(variables)
        # Should be truncated to 120 chars (117 + "...")
        line = prompt.split("\n")[1]  # Second line (after "Score these variables:")
        code_part = line.split("] ", 1)[1]
        assert code_part.endswith("...")
        assert len(code_part) <= 120

    def test_multiline_code_collapsed(self):
        code = "x = {\n    'a': 1,\n    'b': 2\n}"
        variables = [(1, "f.py", "x", code)]
        prompt = build_user_prompt(variables)
        # Newlines should be replaced with spaces
        assert "\n    'a'" not in prompt.split("\n")[1]


class TestSystemPrompt:
    """Tests for system prompt content."""

    def test_has_scoring_dimensions(self):
        assert "config_value" in SYSTEM_PROMPT
        assert "architectural_role" in SYSTEM_PROMPT
        assert "data_definition" in SYSTEM_PROMPT
        assert "general_usefulness" in SYSTEM_PROMPT

    def test_has_rules(self):
        assert "One line per variable" in SYSTEM_PROMPT
        assert "no explanations" in SYSTEM_PROMPT


# =============================================================================
# Token estimation tests
# =============================================================================


class TestEstimateTokens:
    """Tests for _estimate_tokens."""

    def test_short_variable(self):
        tokens = _estimate_tokens("x = 1", "f.py")
        # "x = 1" = 5 chars / 4 = 1, "f.py" = 4 chars / 4 = 1, + 20 overhead
        assert tokens == 22

    def test_longer_variable(self):
        tokens = _estimate_tokens("DATABASE_URL = 'postgres://localhost:5432/db'", "src/config.py")
        assert tokens > 20  # Should be meaningful

    def test_empty_code(self):
        tokens = _estimate_tokens("", "")
        assert tokens == 20  # Just overhead


# =============================================================================
# Batch building tests
# =============================================================================


class TestBuildBatches:
    """Tests for _build_batches."""

    def test_single_variable_single_batch(self):
        variables = [("id1", "f.py", "x", "x = 1")]
        batches = _build_batches(variables, token_budget=800)
        assert len(batches) == 1
        assert len(batches[0]) == 1
        # Check structure: (index, element_id, file_path, name, raw_code)
        assert batches[0][0] == (1, "id1", "f.py", "x", "x = 1")

    def test_multiple_variables_fit_one_batch(self):
        variables = [
            ("id1", "f.py", "x", "x = 1"),
            ("id2", "f.py", "y", "y = 2"),
            ("id3", "f.py", "z", "z = 3"),
        ]
        batches = _build_batches(variables, token_budget=800)
        assert len(batches) == 1
        assert len(batches[0]) == 3

    def test_splits_when_budget_exceeded(self):
        # Each variable will use ~22 tokens. Budget of 40 should force split.
        variables = [
            ("id1", "f.py", "x", "x = 1"),
            ("id2", "f.py", "y", "y = 2"),
            ("id3", "f.py", "z", "z = 3"),
        ]
        batches = _build_batches(variables, token_budget=40)
        assert len(batches) >= 2

    def test_indices_reset_per_batch(self):
        # With tiny budget, each variable gets its own batch
        variables = [
            ("id1", "f.py", "a", "a" * 200),
            ("id2", "f.py", "b", "b" * 200),
        ]
        batches = _build_batches(variables, token_budget=30)
        assert len(batches) == 2
        # First variable in each batch should have index 1
        assert batches[0][0][0] == 1
        assert batches[1][0][0] == 1

    def test_empty_variables(self):
        batches = _build_batches([], token_budget=800)
        assert batches == []

    def test_very_large_single_variable(self):
        # A variable that exceeds the budget alone should still be in a batch
        variables = [("id1", "f.py", "big", "x = " + "a" * 10000)]
        batches = _build_batches(variables, token_budget=100)
        assert len(batches) == 1
        assert len(batches[0]) == 1


# =============================================================================
# Score parsing tests
# =============================================================================


class TestParseScores:
    """Tests for _parse_scores."""

    def test_basic_parsing(self):
        output = "1. 9,2,1,8\n2. 3,7,1,5"
        scores = _parse_scores(output, batch_size=2)
        assert scores[0] is not None
        assert scores[0].config_value == 9
        assert scores[0].architectural_role == 2
        assert scores[0].data_definition == 1
        assert scores[0].general_usefulness == 8
        assert scores[1] is not None
        assert scores[1].config_value == 3
        assert scores[1].architectural_role == 7

    def test_extra_whitespace(self):
        output = "1.  9 , 2 , 1 , 8"
        scores = _parse_scores(output, batch_size=1)
        assert scores[0] is not None
        assert scores[0].config_value == 9

    def test_missing_line(self):
        # Only line 1 present, batch expects 2
        output = "1. 5,5,5,5"
        scores = _parse_scores(output, batch_size=2)
        assert scores[0] is not None
        assert scores[1] is None

    def test_out_of_range_index(self):
        # Index 3 but batch_size is 2 — should be ignored
        output = "3. 5,5,5,5"
        scores = _parse_scores(output, batch_size=2)
        assert scores[0] is None
        assert scores[1] is None

    def test_zero_index_ignored(self):
        output = "0. 5,5,5,5"
        scores = _parse_scores(output, batch_size=1)
        assert scores[0] is None

    def test_scores_clamped_to_range(self):
        # Scores > 10 should be clamped to 10
        output = "1. 15,0,1,10"
        scores = _parse_scores(output, batch_size=1)
        assert scores[0] is not None
        assert scores[0].config_value == 10  # clamped from 15
        assert scores[0].architectural_role == 1  # min(10, max(1, 0)) = 1

    def test_llm_noise_ignored(self):
        # LLM might add explanations — only scored lines should match
        output = """Here are the scores:
1. 8,2,1,7
2. 1,1,1,1
Done!"""
        scores = _parse_scores(output, batch_size=2)
        assert scores[0] is not None
        assert scores[0].config_value == 8
        assert scores[1] is not None
        assert scores[1].config_value == 1

    def test_empty_output(self):
        scores = _parse_scores("", batch_size=3)
        assert all(s is None for s in scores)

    def test_batch_size_matches_output_length(self):
        output = "1. 5,5,5,5\n2. 3,3,3,3\n3. 7,7,7,7"
        scores = _parse_scores(output, batch_size=3)
        assert len(scores) == 3
        assert all(s is not None for s in scores)


# =============================================================================
# Score batch tests
# =============================================================================


class TestScoreBatch:
    """Tests for _score_batch."""

    def test_successful_scoring(self):
        batch = [
            (1, "eid1", "src/config.py", "MAX_RETRIES", "MAX_RETRIES = 3"),
            (2, "eid2", "src/app.py", "i", "i = 0"),
        ]
        mock_client = MagicMock()
        mock_client.generate_from_messages.return_value = "1. 9,2,1,8\n2. 1,1,1,1"

        config = VariableScoringConfig()
        result = _score_batch(batch, mock_client, config, num_ctx=1024)

        assert "eid1" in result
        assert result["eid1"].config_value == 9
        assert "eid2" in result
        assert result["eid2"].config_value == 1

    def test_llm_error_defaults_to_keep(self):
        batch = [
            (1, "eid1", "f.py", "x", "x = 1"),
        ]
        mock_client = MagicMock()
        mock_client.generate_from_messages.side_effect = RuntimeError("LLM unavailable")

        config = VariableScoringConfig()
        result = _score_batch(batch, mock_client, config, num_ctx=1024)

        assert "eid1" in result
        assert result["eid1"].general_usefulness == 5  # Default keep score

    def test_unparseable_output_defaults_to_keep(self):
        batch = [
            (1, "eid1", "f.py", "x", "x = 1"),
        ]
        mock_client = MagicMock()
        mock_client.generate_from_messages.return_value = "I cannot score this variable."

        config = VariableScoringConfig()
        result = _score_batch(batch, mock_client, config, num_ctx=1024)

        assert "eid1" in result
        assert result["eid1"].general_usefulness == 5  # Default keep score

    def test_passes_correct_temperature(self):
        batch = [(1, "eid1", "f.py", "x", "x = 1")]
        mock_client = MagicMock()
        mock_client.generate_from_messages.return_value = "1. 5,5,5,5"

        config = VariableScoringConfig(temperature=0.3)
        _score_batch(batch, mock_client, config, num_ctx=1024)

        call_kwargs = mock_client.generate_from_messages.call_args
        assert call_kwargs.kwargs["temperature"] == 0.3


# =============================================================================
# Main score_variables orchestrator tests
# =============================================================================


class TestScoreVariables:
    """Tests for score_variables orchestrator."""

    def test_empty_input(self):
        result = score_variables([], MagicMock())
        assert result.total_variables == 0
        assert result.kept == 0
        assert result.dropped == 0
        assert result.batch_count == 0

    def test_all_kept(self):
        variables = [
            ("eid1", "f.py", "MAX_RETRIES", "MAX_RETRIES = 3"),
            ("eid2", "f.py", "DB_URL", "DB_URL = 'postgres://...'"),
        ]
        mock_client = MagicMock()
        mock_client.generate_from_messages.return_value = "1. 9,2,1,8\n2. 8,9,1,7"

        result = score_variables(variables, mock_client, max_workers=1)

        assert result.total_variables == 2
        assert result.kept == 2
        assert result.dropped == 0
        assert result.batch_count >= 1
        assert result.elapsed > 0

    def test_some_dropped(self):
        variables = [
            ("eid1", "f.py", "MAX_RETRIES", "MAX_RETRIES = 3"),
            ("eid2", "f.py", "i", "i = 0"),
        ]
        mock_client = MagicMock()
        # First var scores high, second all 1s
        mock_client.generate_from_messages.return_value = "1. 9,2,1,8\n2. 1,1,1,1"

        result = score_variables(variables, mock_client, max_workers=1)

        assert result.total_variables == 2
        assert result.kept == 1
        assert result.dropped == 1

    def test_custom_threshold(self):
        variables = [("eid1", "f.py", "x", "x = 1")]
        mock_client = MagicMock()
        mock_client.generate_from_messages.return_value = "1. 3,3,3,3"

        # Default threshold=5 → dropped
        result = score_variables(variables, mock_client, max_workers=1)
        assert result.dropped == 1

        # Lower threshold → kept
        config = VariableScoringConfig(threshold=3)
        result = score_variables(variables, mock_client, config=config, max_workers=1)
        assert result.kept == 1

    def test_scores_dict_populated(self):
        variables = [("eid1", "f.py", "x", "x = 1")]
        mock_client = MagicMock()
        mock_client.generate_from_messages.return_value = "1. 7,3,2,6"

        result = score_variables(variables, mock_client, max_workers=1)

        assert "eid1" in result.scores
        assert result.scores["eid1"].config_value == 7
        assert result.scores["eid1"].general_usefulness == 6

    def test_parallel_processing(self):
        """Test that parallel processing works with multiple batches."""
        # Create enough variables to span multiple batches (tiny budget)
        variables = [
            (f"eid{i}", "f.py", f"var_{i}", f"var_{i} = {i}" * 50)
            for i in range(10)
        ]
        mock_client = MagicMock()

        # Each batch gets its own call — mock returns valid scores
        def mock_generate(**kwargs):
            # Parse the prompt to figure out how many variables
            content = kwargs.get("messages", [{}])[-1].get("content", "")
            lines = [line for line in content.split("\n") if line and line[0].isdigit()]
            return "\n".join(f"{i+1}. 5,5,5,5" for i in range(len(lines)))

        mock_client.generate_from_messages.side_effect = mock_generate

        config = VariableScoringConfig(token_budget=100)  # Very small → many batches
        result = score_variables(variables, mock_client, config=config, max_workers=4)

        assert result.total_variables == 10
        assert result.batch_count > 1  # Should have split
        assert result.kept == 10  # All scored 5 → pass threshold
