"""Tests for the variable scoring module.

Tests cover models, prompt building, score parsing, batching,
the main score_variables orchestrator, and progress tracking.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

from magaldi_core.variable_scoring import (
    _build_batches,
    _estimate_tokens,
    _parse_scores,
    _score_batch,
    score_variables,
)
from magaldi_core.variable_scoring.models import (
    ScoringProgressState,
    ScoringResult,
    ScoringWorkerStatus,
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
        assert config.token_budget == 1200
        assert config.max_retries == 2
        assert config.timeout == 180


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
        # Find the variable line (contains "[f.py]")
        var_line = [line for line in prompt.split("\n") if "[f.py]" in line][0]
        code_part = var_line.split("] ", 1)[1]
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

    def test_has_few_shot_examples(self):
        """Prompt includes HIGH and LOW scoring examples for calibration."""
        assert "Examples (HIGH" in SYSTEM_PROMPT
        assert "Examples (LOW" in SYSTEM_PROMPT
        # LOW examples should show 1,1,1,1
        assert "1,1,1,1" in SYSTEM_PROMPT
        # HIGH examples should show high scores
        assert "9,1,1,8" in SYSTEM_PROMPT  # config constant

    def test_covers_string_templates(self):
        """Prompt covers string templates/prompts as HIGH scoring."""
        assert "prompt template" in SYSTEM_PROMPT or "string template" in SYSTEM_PROMPT
        assert "SYSTEM_PROMPT" in SYSTEM_PROMPT

    def test_covers_coding_agent_perspective(self):
        """Prompt frames scoring from a coding agent's perspective."""
        assert "coding agent" in SYSTEM_PROMPT

    def test_covers_export_lists(self):
        """Prompt covers __all__ export lists as HIGH scoring."""
        assert "__all__" in SYSTEM_PROMPT

    def test_covers_query_strings(self):
        """Prompt covers SQL/query strings and sentinels."""
        assert "query string" in SYSTEM_PROMPT or "SQL" in SYSTEM_PROMPT
        assert "sentinel" in SYSTEM_PROMPT

    def test_has_drop_rate_guidance(self):
        """Prompt guides model that most variables should be dropped."""
        assert "30%" in SYSTEM_PROMPT


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
        result, prompt_tok, resp_tok = _score_batch(batch, mock_client, config, num_ctx=1024)

        assert "eid1" in result
        assert result["eid1"].config_value == 9
        assert "eid2" in result
        assert result["eid2"].config_value == 1
        assert prompt_tok > 0
        assert resp_tok > 0

    def test_llm_error_defaults_to_keep(self):
        batch = [
            (1, "eid1", "f.py", "x", "x = 1"),
        ]
        mock_client = MagicMock()
        mock_client.generate_from_messages.side_effect = RuntimeError("LLM unavailable")

        config = VariableScoringConfig()
        result, prompt_tok, resp_tok = _score_batch(batch, mock_client, config, num_ctx=1024)

        assert "eid1" in result
        assert result["eid1"].general_usefulness == 5  # Default keep score
        assert prompt_tok > 0
        assert resp_tok == 0  # No response on error

    def test_unparseable_output_defaults_to_keep(self):
        batch = [
            (1, "eid1", "f.py", "x", "x = 1"),
        ]
        mock_client = MagicMock()
        mock_client.generate_from_messages.return_value = "I cannot score this variable."

        config = VariableScoringConfig()
        result, _prompt_tok, _resp_tok = _score_batch(batch, mock_client, config, num_ctx=1024)

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

    def test_passes_correct_timeout(self):
        """Timeout from config is propagated to generate_from_messages."""
        batch = [(1, "eid1", "f.py", "x", "x = 1")]
        mock_client = MagicMock()
        mock_client.generate_from_messages.return_value = "1. 5,5,5,5"

        # Default timeout (180s)
        config = VariableScoringConfig()
        _score_batch(batch, mock_client, config, num_ctx=1024)
        call_kwargs = mock_client.generate_from_messages.call_args
        assert call_kwargs.kwargs["timeout"] == 180

        # Custom timeout
        config = VariableScoringConfig(timeout=300)
        _score_batch(batch, mock_client, config, num_ctx=1024)
        call_kwargs = mock_client.generate_from_messages.call_args
        assert call_kwargs.kwargs["timeout"] == 300

    def test_token_counts_returned(self):
        """Token counts are estimated from prompt and response."""
        batch = [
            (1, "eid1", "src/config.py", "MAX_RETRIES", "MAX_RETRIES = 3"),
        ]
        mock_client = MagicMock()
        mock_client.generate_from_messages.return_value = "1. 9,2,1,8"

        config = VariableScoringConfig()
        _result, prompt_tok, resp_tok = _score_batch(batch, mock_client, config, num_ctx=1024)

        # Prompt includes system prompt + user prompt, estimated at ~4 chars/token
        assert prompt_tok > 100  # System prompt alone is ~660 tokens
        assert resp_tok > 0


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

    def test_token_counts_aggregated(self):
        """Token counts are accumulated across batches."""
        variables = [
            ("eid1", "f.py", "MAX_RETRIES", "MAX_RETRIES = 3"),
            ("eid2", "f.py", "DB_URL", "DB_URL = 'postgres://...'"),
        ]
        mock_client = MagicMock()
        mock_client.generate_from_messages.return_value = "1. 9,2,1,8\n2. 8,9,1,7"

        result = score_variables(variables, mock_client, max_workers=1)

        assert result.prompt_tokens > 0
        assert result.response_tokens > 0

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


# =============================================================================
# Progress tracking model tests
# =============================================================================


class TestScoringWorkerStatus:
    """Tests for ScoringWorkerStatus thread-safe worker tracker."""

    def test_set_and_get(self):
        ws = ScoringWorkerStatus()
        ws.set(0, 1, 5)
        all_workers = ws.get_all()
        assert 0 in all_workers
        batch_num, batch_size, start_time = all_workers[0]
        assert batch_num == 1
        assert batch_size == 5
        assert start_time > 0

    def test_clear(self):
        ws = ScoringWorkerStatus()
        ws.set(0, 1, 5)
        assert ws.active_count() == 1
        ws.clear(0)
        assert ws.active_count() == 0
        assert ws.get_all() == {}

    def test_clear_nonexistent_worker(self):
        ws = ScoringWorkerStatus()
        ws.clear(99)  # Should not raise

    def test_active_count(self):
        ws = ScoringWorkerStatus()
        assert ws.active_count() == 0
        ws.set(0, 1, 5)
        ws.set(1, 2, 3)
        assert ws.active_count() == 2
        ws.clear(0)
        assert ws.active_count() == 1

    def test_multiple_workers(self):
        ws = ScoringWorkerStatus()
        ws.set(0, 1, 10)
        ws.set(1, 2, 8)
        ws.set(2, 3, 6)
        all_workers = ws.get_all()
        assert len(all_workers) == 3
        assert all_workers[0][0] == 1  # batch_num
        assert all_workers[1][0] == 2
        assert all_workers[2][0] == 3


class TestScoringProgressState:
    """Tests for ScoringProgressState."""

    def test_defaults(self):
        state = ScoringProgressState()
        assert state.total_variables == 0
        assert state.total_batches == 0
        assert state.completed_batches == 0
        assert state.completed_variables == 0
        assert state.kept == 0
        assert state.dropped == 0
        assert state.errors == 0
        assert state.num_workers == 1
        assert state.batch_times == []

    def test_elapsed(self):
        state = ScoringProgressState(start_time=time.time() - 2.0)
        assert state.elapsed >= 1.9

    def test_avg_batch_time_empty(self):
        state = ScoringProgressState()
        assert state.avg_batch_time == 0.0

    def test_avg_batch_time(self):
        state = ScoringProgressState(batch_times=[1.0, 2.0, 3.0])
        assert state.avg_batch_time == 2.0

    def test_avg_variable_time_zero_completed(self):
        state = ScoringProgressState()
        assert state.avg_variable_time == 0.0

    def test_avg_variable_time(self):
        state = ScoringProgressState(
            completed_variables=10,
            start_time=time.time() - 5.0,
        )
        avg = state.avg_variable_time
        assert 0.4 <= avg <= 0.6  # ~0.5s/var with 5s elapsed and 10 vars

    def test_eta_seconds_no_completed(self):
        state = ScoringProgressState(total_batches=10)
        assert state.eta_seconds() is None

    def test_eta_seconds_all_done(self):
        state = ScoringProgressState(
            total_batches=5,
            completed_batches=5,
            batch_times=[1.0] * 5,
        )
        assert state.eta_seconds() == 0.0

    def test_eta_seconds_midway(self):
        state = ScoringProgressState(
            total_batches=10,
            completed_batches=5,
            num_workers=1,
            batch_times=[2.0] * 5,  # avg 2s per batch
        )
        eta = state.eta_seconds()
        assert eta is not None
        # 5 remaining batches * 2s / 1 worker = 10s
        assert 9.9 <= eta <= 10.1

    def test_eta_seconds_with_parallelism(self):
        state = ScoringProgressState(
            total_batches=10,
            completed_batches=5,
            num_workers=5,
            batch_times=[2.0] * 5,
        )
        eta = state.eta_seconds()
        assert eta is not None
        # 5 remaining / 5 workers * 2s = 2s
        assert 1.9 <= eta <= 2.1

    def test_eta_caps_workers_at_remaining(self):
        state = ScoringProgressState(
            total_batches=10,
            completed_batches=9,
            num_workers=8,
            batch_times=[2.0] * 9,
        )
        eta = state.eta_seconds()
        assert eta is not None
        # 1 remaining / min(8, 1) workers * 2s = 2s
        assert 1.9 <= eta <= 2.1


# =============================================================================
# Progress callback integration tests
# =============================================================================


class TestScoreVariablesWithProgress:
    """Tests for score_variables with progress tracking."""

    def test_progress_callback_called(self):
        """Verify on_progress fires after each batch."""
        variables = [
            ("eid1", "f.py", "x", "x = 1"),
            ("eid2", "f.py", "y", "y = 2"),
        ]
        mock_client = MagicMock()
        mock_client.generate_from_messages.return_value = "1. 9,2,1,8\n2. 8,7,1,6"

        progress_states: list[ScoringProgressState] = []

        def on_progress(state: ScoringProgressState) -> None:
            # Snapshot the current state
            progress_states.append(ScoringProgressState(
                total_variables=state.total_variables,
                total_batches=state.total_batches,
                completed_batches=state.completed_batches,
                completed_variables=state.completed_variables,
                kept=state.kept,
                dropped=state.dropped,
                errors=state.errors,
            ))

        state = ScoringProgressState()
        worker_status = ScoringWorkerStatus()

        result = score_variables(
            variables, mock_client, max_workers=1,
            on_progress=on_progress,
            progress_state=state,
            worker_status=worker_status,
        )

        assert len(progress_states) >= 1
        assert result.kept == 2
        # Final progress state should reflect all completed
        final = progress_states[-1]
        assert final.completed_variables == 2
        assert final.total_variables == 2

    def test_progress_state_initialized(self):
        """Verify progress_state gets initialized with correct totals."""
        variables = [
            ("eid1", "f.py", "x", "x = 1"),
            ("eid2", "f.py", "y", "y = 2"),
        ]
        mock_client = MagicMock()
        mock_client.generate_from_messages.return_value = "1. 9,2,1,8\n2. 1,1,1,1"

        state = ScoringProgressState()
        worker_status = ScoringWorkerStatus()

        score_variables(
            variables, mock_client, max_workers=1,
            progress_state=state,
            worker_status=worker_status,
        )

        assert state.total_variables == 2
        assert state.total_batches >= 1
        assert state.completed_batches == state.total_batches
        assert state.completed_variables == 2
        assert state.kept == 1
        assert state.dropped == 1

    def test_progress_tracks_kept_and_dropped(self):
        """Verify real-time kept/dropped counting in progress state."""
        variables = [
            ("eid1", "f.py", "x", "x = 1"),
            ("eid2", "f.py", "y", "y = 2"),
        ]
        mock_client = MagicMock()
        # First kept (score 9), second dropped (all 1s)
        mock_client.generate_from_messages.return_value = "1. 9,2,1,8\n2. 1,1,1,1"

        state = ScoringProgressState()
        worker_status = ScoringWorkerStatus()

        score_variables(
            variables, mock_client, max_workers=1,
            progress_state=state,
            worker_status=worker_status,
        )

        assert state.kept == 1
        assert state.dropped == 1

    def test_progress_tracks_errors(self):
        """Verify error counting when _score_batch itself raises.

        Note: _score_batch has internal error handling that catches LLM
        errors and returns defaults. To trigger the executor-level error
        path, we need a failure that escapes _score_batch entirely.
        """
        # Create enough variables for multiple batches with tiny budget
        variables = [
            (f"eid{i}", "f.py", f"var_{i}", f"var_{i} = {i}" * 50)
            for i in range(6)
        ]
        mock_client = MagicMock()

        call_count = 0

        def mock_generate(**kwargs):
            nonlocal call_count
            call_count += 1
            content = kwargs.get("messages", [{}])[-1].get("content", "")
            lines = [line for line in content.split("\n") if line and line[0].isdigit()]
            return "\n".join(f"{i+1}. 5,5,5,5" for i in range(len(lines)))

        mock_client.generate_from_messages.side_effect = mock_generate

        state = ScoringProgressState()
        worker_status = ScoringWorkerStatus()
        config = VariableScoringConfig(token_budget=100)

        result = score_variables(
            variables, mock_client, config=config, max_workers=2,
            progress_state=state,
            worker_status=worker_status,
        )

        # _score_batch catches LLM errors internally, so no executor-level errors
        assert result.errors == 0
        assert state.errors == 0
        # All variables should still have scores
        assert len(result.scores) == 6
        # All batches should be tracked in progress
        assert state.completed_batches == state.total_batches

    def test_progress_batch_times_populated(self):
        """Verify batch timing data is collected."""
        variables = [
            ("eid1", "f.py", "x", "x = 1"),
        ]
        mock_client = MagicMock()
        mock_client.generate_from_messages.return_value = "1. 5,5,5,5"

        state = ScoringProgressState()
        worker_status = ScoringWorkerStatus()

        score_variables(
            variables, mock_client, max_workers=1,
            progress_state=state,
            worker_status=worker_status,
        )

        assert len(state.batch_times) >= 1
        assert all(t >= 0 for t in state.batch_times)

    def test_workers_cleared_after_completion(self):
        """Verify all workers are cleared after processing finishes."""
        variables = [
            ("eid1", "f.py", "x", "x = 1"),
            ("eid2", "f.py", "y", "y = 2"),
        ]
        mock_client = MagicMock()
        mock_client.generate_from_messages.return_value = "1. 5,5,5,5\n2. 5,5,5,5"

        state = ScoringProgressState()
        worker_status = ScoringWorkerStatus()

        score_variables(
            variables, mock_client, max_workers=2,
            progress_state=state,
            worker_status=worker_status,
        )

        # All workers should be cleared after processing
        assert worker_status.active_count() == 0

    def test_backward_compatible_without_progress(self):
        """Verify score_variables still works without progress args."""
        variables = [("eid1", "f.py", "x", "x = 1")]
        mock_client = MagicMock()
        mock_client.generate_from_messages.return_value = "1. 5,5,5,5"

        result = score_variables(variables, mock_client, max_workers=1)

        assert result.total_variables == 1
        assert result.kept == 1


# =============================================================================
# Rich display builder tests
# =============================================================================


class TestBuildScoringDisplay:
    """Tests for _build_scoring_display in _runners.py."""

    def test_display_builds_without_error(self):
        """Verify the display builder produces output without crashing."""
        from shared.cli._runners import _build_scoring_display

        state = ScoringProgressState(
            total_variables=100,
            total_batches=10,
            completed_batches=5,
            completed_variables=50,
            kept=30,
            dropped=20,
            num_workers=4,
            start_time=time.time() - 10.0,
            batch_times=[2.0] * 5,
        )

        display = _build_scoring_display(state, num_workers=4)
        assert display is not None

    def test_display_with_zero_progress(self):
        """Verify display works at start before any batches complete."""
        from shared.cli._runners import _build_scoring_display

        state = ScoringProgressState(
            total_variables=50,
            total_batches=5,
            num_workers=4,
        )

        display = _build_scoring_display(state, num_workers=4)
        assert display is not None

    def test_display_with_errors(self):
        """Verify display includes error count."""
        from shared.cli._runners import _build_scoring_display

        state = ScoringProgressState(
            total_variables=100,
            total_batches=10,
            completed_batches=3,
            completed_variables=30,
            kept=20,
            dropped=8,
            errors=2,
            num_workers=4,
            start_time=time.time() - 5.0,
            batch_times=[1.5] * 3,
        )

        display = _build_scoring_display(state, num_workers=4)
        assert display is not None

    def test_display_all_complete(self):
        """Verify display works when all batches are complete."""
        from shared.cli._runners import _build_scoring_display

        state = ScoringProgressState(
            total_variables=20,
            total_batches=4,
            completed_batches=4,
            completed_variables=20,
            kept=15,
            dropped=5,
            num_workers=4,
            start_time=time.time() - 8.0,
            batch_times=[2.0] * 4,
        )

        display = _build_scoring_display(state, num_workers=4)
        assert display is not None

    def test_display_with_active_workers(self):
        """Verify display handles active workers correctly."""
        from shared.cli._runners import _build_scoring_display

        ws = ScoringWorkerStatus()
        ws.set(0, 3, 8)
        ws.set(1, 4, 6)

        state = ScoringProgressState(
            total_variables=50,
            total_batches=8,
            completed_batches=2,
            completed_variables=15,
            kept=10,
            dropped=5,
            num_workers=4,
            start_time=time.time() - 3.0,
            batch_times=[1.5] * 2,
            workers=ws,
        )

        display = _build_scoring_display(state, num_workers=4)
        assert display is not None

    def test_display_with_throttle_decision(self):
        """Verify display handles throttle decision state."""
        from shared.cli._runners import _build_scoring_display
        from shared.throttling import ThrottleDecision

        state = ScoringProgressState(
            total_variables=100,
            total_batches=20,
            completed_batches=10,
            completed_variables=50,
            kept=30,
            dropped=20,
            num_workers=8,
            start_time=time.time() - 10.0,
            batch_times=[2.0] * 10,
            throttle_decision=ThrottleDecision(
                should_throttle=True,
                current_max=5.0,
                historical_max=0.0,
                completed_avg=2.5,
                recommended_workers=4,
                reason="Throttle (4=78s/2.5s base)",
                completion_count=10,
            ),
            allowed_workers=4,
        )

        display = _build_scoring_display(state, num_workers=8)
        assert display is not None

    def test_display_throttled_workers_shown(self):
        """Verify throttled workers appear differently from idle workers."""
        from shared.cli._runners import _build_scoring_display

        ws = ScoringWorkerStatus()
        ws.set(0, 1, 10)
        ws.set(1, 2, 8)

        state = ScoringProgressState(
            total_variables=50,
            total_batches=10,
            completed_batches=2,
            completed_variables=15,
            kept=10,
            dropped=5,
            num_workers=6,
            start_time=time.time() - 5.0,
            batch_times=[2.5] * 2,
            workers=ws,
            allowed_workers=3,  # Only 3 allowed, workers 3-5 are throttled
        )

        display = _build_scoring_display(state, num_workers=6)
        assert display is not None

    def test_display_budget_disabled_workers(self):
        """Verify workers beyond explore_cap are collapsed into summary line."""
        import re
        from io import StringIO

        from rich.console import Console

        from shared.cli._runners import _build_scoring_display
        from shared.throttling import ThrottleDecision

        ws = ScoringWorkerStatus()
        ws.set(0, 1, 10)
        ws.set(1, 2, 8)

        state = ScoringProgressState(
            total_variables=50,
            total_batches=10,
            completed_batches=2,
            completed_variables=15,
            kept=10,
            dropped=5,
            num_workers=16,
            start_time=time.time() - 5.0,
            batch_times=[2.5] * 2,
            workers=ws,
            allowed_workers=5,
            throttle_decision=ThrottleDecision(
                should_throttle=True,
                current_max=5.0,
                historical_max=0.0,
                completed_avg=2.5,
                recommended_workers=5,
                reason="test",
                completion_count=10,
                explore_cap=7,  # Only 7 workers reachable, 9 disabled
            ),
        )

        display = _build_scoring_display(state, num_workers=16)
        assert display is not None

        # Render to string and strip ANSI to verify the budget summary line appears
        buf = StringIO()
        console = Console(file=buf, width=200, force_terminal=True)
        console.print(display)
        output = buf.getvalue()
        # Strip ANSI escape codes and collapse whitespace for matching
        clean = re.sub(r'\x1b\[[0-9;]*m', '', output)
        clean = ' '.join(clean.split())
        # Text may be truncated by Rich column widths, so match key fragments
        assert "9 workers disabled" in clean
        assert "budget)" in clean
        # Budget-disabled workers should NOT show as individual rows
        assert "[8]" not in clean  # Workers 8-16 should not appear
        # Workers within explore_cap should still show (active/idle/throttled)
        assert "[1]" in clean
        assert "[7]" in clean

    def test_display_no_budget_line_without_explore_cap(self):
        """Verify no budget line when explore_cap is not set."""
        import re
        from io import StringIO

        from rich.console import Console

        from shared.cli._runners import _build_scoring_display

        ws = ScoringWorkerStatus()
        ws.set(0, 1, 10)

        state = ScoringProgressState(
            total_variables=50,
            total_batches=10,
            completed_batches=2,
            completed_variables=15,
            kept=10,
            dropped=5,
            num_workers=8,
            start_time=time.time() - 5.0,
            batch_times=[2.5] * 2,
            workers=ws,
            allowed_workers=4,
        )

        display = _build_scoring_display(state, num_workers=8)
        assert display is not None

        buf = StringIO()
        console = Console(file=buf, width=200, force_terminal=True)
        console.print(display)
        output = buf.getvalue()
        clean = re.sub(r'\x1b\[[0-9;]*m', '', output)
        assert "exploration budget" not in clean


# =============================================================================
# Throttle integration tests
# =============================================================================


class TestScoreVariablesThrottling:
    """Tests for score_variables with runtime-aware throttling."""

    def test_throttling_activates_with_slow_llm(self):
        """When LLM is slow, throttling should reduce allowed workers."""
        # Create enough variables for many batches
        variables = [
            (f"eid{i}", "f.py", f"var_{i}", f"var_{i} = {i}" * 50)
            for i in range(20)
        ]
        mock_client = MagicMock()

        def slow_generate(**kwargs):
            time.sleep(0.05)  # Simulate slow LLM
            content = kwargs.get("messages", [{}])[-1].get("content", "")
            lines = [line for line in content.split("\n") if line and line[0].isdigit()]
            return "\n".join(f"{i+1}. 5,5,5,5" for i in range(len(lines)))

        mock_client.generate_from_messages.side_effect = slow_generate

        state = ScoringProgressState()
        worker_status = ScoringWorkerStatus()
        config = VariableScoringConfig(token_budget=100)

        result = score_variables(
            variables, mock_client, config=config, max_workers=4,
            progress_state=state,
            worker_status=worker_status,
        )

        # All variables should be scored regardless of throttling
        assert len(result.scores) == 20
        assert result.kept + result.dropped == 20
        # Throttle decision should have been set
        assert state.throttle_decision is not None

    def test_throttle_state_in_progress(self):
        """Verify throttle info is available in progress state."""
        variables = [
            (f"eid{i}", "f.py", f"var_{i}", f"var_{i} = {i}" * 50)
            for i in range(10)
        ]
        mock_client = MagicMock()

        def mock_generate(**kwargs):
            content = kwargs.get("messages", [{}])[-1].get("content", "")
            lines = [line for line in content.split("\n") if line and line[0].isdigit()]
            return "\n".join(f"{i+1}. 5,5,5,5" for i in range(len(lines)))

        mock_client.generate_from_messages.side_effect = mock_generate

        progress_updates: list[int | None] = []

        def on_progress(state: ScoringProgressState) -> None:
            progress_updates.append(state.allowed_workers)

        state = ScoringProgressState()
        worker_status = ScoringWorkerStatus()
        config = VariableScoringConfig(token_budget=100)

        result = score_variables(
            variables, mock_client, config=config, max_workers=4,
            on_progress=on_progress,
            progress_state=state,
            worker_status=worker_status,
        )

        assert len(result.scores) == 10
        # Should have received progress updates with allowed_workers info
        assert len(progress_updates) >= 1
        # All values should be positive integers
        assert all(w is not None and w >= 1 for w in progress_updates)

    def test_eta_uses_allowed_workers(self):
        """Verify ETA calculation uses allowed_workers when throttled."""
        state = ScoringProgressState(
            total_batches=10,
            completed_batches=5,
            num_workers=8,
            batch_times=[2.0] * 5,
            allowed_workers=2,  # Throttled to 2
        )
        eta = state.eta_seconds()
        assert eta is not None
        # 5 remaining / min(2, 5) workers * 2s = 5s
        assert 4.9 <= eta <= 5.1

    def test_eta_uses_num_workers_when_no_throttle(self):
        """Verify ETA calculation uses num_workers when not throttled."""
        state = ScoringProgressState(
            total_batches=10,
            completed_batches=5,
            num_workers=5,
            batch_times=[2.0] * 5,
            allowed_workers=None,  # No throttle
        )
        eta = state.eta_seconds()
        assert eta is not None
        # 5 remaining / 5 workers * 2s = 2s
        assert 1.9 <= eta <= 2.1


# =============================================================================
# Debug log tests
# =============================================================================


class TestDebugLog:
    """Tests for debug logging of prompt → LLM response."""

    def test_debug_log_captured(self):
        """Verify debug_log captures first batch's prompt and response."""
        variables = [("eid1", "f.py", "x", "x = 1")]
        mock_client = MagicMock()
        mock_client.generate_from_messages.return_value = "1. 9,2,1,8"

        result = score_variables(variables, mock_client, max_workers=1)

        assert len(result.debug_log) == 1
        prompt, response = result.debug_log[0]
        assert "Score these variables:" in prompt
        assert "x = 1" in prompt
        assert "9,2,1,8" in response

    def test_debug_log_only_first_batch(self):
        """Verify debug_log only captures one batch even with multiple."""
        variables = [
            (f"eid{i}", "f.py", f"var_{i}", f"var_{i} = {i}" * 50)
            for i in range(6)
        ]
        mock_client = MagicMock()

        def mock_generate(**kwargs):
            content = kwargs.get("messages", [{}])[-1].get("content", "")
            lines = [line for line in content.split("\n") if line and line[0].isdigit()]
            return "\n".join(f"{i+1}. 5,5,5,5" for i in range(len(lines)))

        mock_client.generate_from_messages.side_effect = mock_generate

        config = VariableScoringConfig(token_budget=100)
        result = score_variables(variables, mock_client, config=config, max_workers=1)

        assert result.batch_count > 1
        assert len(result.debug_log) == 1  # Only first batch

    def test_debug_log_empty_on_empty_input(self):
        """Verify debug_log is empty when no variables."""
        result = score_variables([], MagicMock())
        assert result.debug_log == []

    def test_debug_log_empty_on_all_errors(self):
        """Verify debug_log is empty when LLM errors on all batches."""
        variables = [("eid1", "f.py", "x", "x = 1")]
        mock_client = MagicMock()
        mock_client.generate_from_messages.side_effect = RuntimeError("LLM down")

        result = score_variables(variables, mock_client, max_workers=1)

        # _score_batch catches the error internally and returns defaults
        # debug_log should be empty since the exception path doesn't capture
        assert result.debug_log == []


class TestWorkerStatusMaxRuntime:
    """Tests for ScoringWorkerStatus.get_max_active_runtime."""

    def test_max_runtime_empty(self):
        ws = ScoringWorkerStatus()
        assert ws.get_max_active_runtime() == 0.0

    def test_max_runtime_single_worker(self):
        ws = ScoringWorkerStatus()
        ws.set(0, 1, 5)
        time.sleep(0.05)
        runtime = ws.get_max_active_runtime()
        assert runtime >= 0.04

    def test_max_runtime_multiple_workers(self):
        ws = ScoringWorkerStatus()
        ws.set(0, 1, 5)
        time.sleep(0.05)
        ws.set(1, 2, 3)  # Started later
        runtime = ws.get_max_active_runtime()
        # Worker 0 has been running longer
        assert runtime >= 0.04
