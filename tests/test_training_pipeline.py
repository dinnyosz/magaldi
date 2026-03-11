"""Tests for the variable scorer training pipeline.

Tests the data generation, quality filtering, parsing, and formatting
components of tools/training/generate_scoring_data.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add training tools to path
_project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_project_root / "tools" / "training"))
sys.path.insert(0, str(_project_root / "src"))

from generate_scoring_data import (
    NUM_SCORES,
    RawScoringResult,
    TEACHER_SYSTEM_PROMPT,
    _format_duration,
    _parse_teacher_scores,
    _result_hash,
    build_training_batches,
    format_training_example,
    load_existing_results,
    save_raw_result,
    validate_batch_scores,
    validate_scores,
)


# =============================================================================
# _parse_teacher_scores
# =============================================================================


class TestParseTeacherScores:
    """Tests for parsing 7-dimension teacher output."""

    def test_single_variable(self):
        output = "1. 9,1,1,8,2,8,9"
        result = _parse_teacher_scores(output, 1)
        assert result == [(9, 1, 1, 8, 2, 8, 9)]

    def test_multiple_variables(self):
        output = "1. 9,1,1,8,2,8,9\n2. 1,1,1,1,1,1,1\n3. 1,10,1,9,3,7,9"
        result = _parse_teacher_scores(output, 3)
        assert result[0] == (9, 1, 1, 8, 2, 8, 9)
        assert result[1] == (1, 1, 1, 1, 1, 1, 1)
        assert result[2] == (1, 10, 1, 9, 3, 7, 9)

    def test_clamps_scores_to_range(self):
        output = "1. 0,11,1,1,1,1,1"
        result = _parse_teacher_scores(output, 1)
        assert result[0] == (1, 10, 1, 1, 1, 1, 1)

    def test_missing_variable(self):
        output = "1. 9,1,1,8,2,8,9\n3. 1,1,1,1,1,1,1"
        result = _parse_teacher_scores(output, 3)
        assert result[0] == (9, 1, 1, 8, 2, 8, 9)
        assert result[1] is None
        assert result[2] == (1, 1, 1, 1, 1, 1, 1)

    def test_empty_output(self):
        result = _parse_teacher_scores("", 3)
        assert result == [None, None, None]

    def test_strips_think_tags(self):
        output = "<think>reasoning here</think>\n1. 5,5,5,5,5,5,5"
        result = _parse_teacher_scores(output, 1)
        assert result[0] == (5, 5, 5, 5, 5, 5, 5)

    def test_multiline_think_tags(self):
        output = "<think>\nlong\nreasoning\n</think>\n1. 3,3,3,3,3,3,3"
        result = _parse_teacher_scores(output, 1)
        assert result[0] == (3, 3, 3, 3, 3, 3, 3)

    def test_ignores_extra_text(self):
        output = "Here are the scores:\n1. 7,7,7,7,7,7,7\nSome explanation"
        result = _parse_teacher_scores(output, 1)
        assert result[0] == (7, 7, 7, 7, 7, 7, 7)

    def test_with_spaces_around_commas(self):
        output = "1. 9 , 1 , 1 , 8 , 2 , 8 , 9"
        result = _parse_teacher_scores(output, 1)
        assert result[0] == (9, 1, 1, 8, 2, 8, 9)

    def test_out_of_bounds_index_ignored(self):
        output = "0. 1,1,1,1,1,1,1\n5. 1,1,1,1,1,1,1"
        result = _parse_teacher_scores(output, 3)
        assert result == [None, None, None]

    def test_only_4_scores_does_not_match(self):
        """Ensure 4-score format from old production model doesn't match."""
        output = "1. 9,1,1,8"
        result = _parse_teacher_scores(output, 1)
        assert result[0] is None


# =============================================================================
# validate_scores
# =============================================================================


class TestValidateScores:
    """Tests for individual score validation."""

    def test_valid_scores(self):
        var = {"name": "MAX_RETRIES", "raw_code": "MAX_RETRIES = 3"}
        valid, reason = validate_scores(var, (9, 1, 1, 8, 2, 8, 9))
        assert valid
        assert reason == "ok"

    def test_wrong_dimension_count(self):
        var = {"name": "x", "raw_code": "x = 1"}
        valid, reason = validate_scores(var, (1, 1, 1, 1))
        assert not valid
        assert "wrong_dim_count" in reason

    def test_out_of_range(self):
        var = {"name": "x", "raw_code": "x = 1"}
        valid, reason = validate_scores(var, (0, 1, 1, 1, 1, 1, 1))
        assert not valid
        assert reason == "out_of_range"

    def test_out_of_range_high(self):
        var = {"name": "x", "raw_code": "x = 1"}
        valid, reason = validate_scores(var, (11, 1, 1, 1, 1, 1, 1))
        assert not valid
        assert reason == "out_of_range"

    def test_heuristic_disagreement(self):
        """If heuristic says drop but teacher scored high, reject."""
        var = {"name": "_", "raw_code": "_ = something()"}
        valid, reason = validate_scores(var, (1, 1, 1, 9, 1, 1, 1))
        assert not valid
        assert "heuristic_disagreement" in reason

    def test_uppercase_constant_scored_low(self):
        var = {"name": "DATABASE_URL", "raw_code": 'DATABASE_URL = "postgres://..."'}
        valid, reason = validate_scores(var, (2, 1, 1, 2, 1, 2, 3))
        assert not valid
        assert "constant_scored_low" in reason

    def test_short_uppercase_not_flagged(self):
        """Short UPPER names (<=3 chars) aren't flagged by the constant check."""
        var = {"name": "URL", "raw_code": 'URL = "http://..."'}
        valid, reason = validate_scores(var, (2, 1, 1, 2, 1, 2, 3))
        assert valid

    def test_all_ones_valid(self):
        var = {"name": "tmp", "raw_code": "tmp = []"}
        valid, reason = validate_scores(var, (1, 1, 1, 1, 1, 1, 1))
        assert valid
        assert reason == "ok"


# =============================================================================
# validate_batch_scores
# =============================================================================


class TestValidateBatchScores:
    """Tests for batch score validation."""

    def test_valid_batch(self):
        variables = [
            {"name": "x", "raw_code": "x = 1"},
            {"name": "y", "raw_code": "y = 2"},
        ]
        scores = [(1, 1, 1, 1, 1, 1, 1), (5, 5, 5, 5, 5, 5, 5)]
        valid, reason = validate_batch_scores(variables, scores)
        assert valid

    def test_uniform_scores_rejected(self):
        """If >3 variables all got identical scores, reject."""
        variables = [{"name": f"v{i}", "raw_code": f"v{i} = {i}"} for i in range(5)]
        scores = [(5, 5, 5, 5, 5, 5, 5)] * 5
        valid, reason = validate_batch_scores(variables, scores)
        assert not valid
        assert reason == "uniform_scores"

    def test_small_batch_uniform_ok(self):
        """<=3 variables with same scores is ok."""
        variables = [{"name": f"v{i}", "raw_code": f"v{i} = {i}"} for i in range(3)]
        scores = [(5, 5, 5, 5, 5, 5, 5)] * 3
        valid, reason = validate_batch_scores(variables, scores)
        assert valid


# =============================================================================
# format_training_example
# =============================================================================


class TestFormatTrainingExample:
    """Tests for ChatML formatting."""

    def test_basic_format(self):
        variables = [
            {"file_path": "src/config.py", "name": "MAX_RETRIES", "raw_code": "MAX_RETRIES = 3"},
            {"file_path": "src/app.py", "name": "tmp", "raw_code": "tmp = []"},
        ]
        scores = [(9, 1, 1, 8, 2, 8, 9), (1, 1, 1, 1, 1, 1, 1)]

        result = format_training_example(variables, scores)

        assert "conversations" in result
        convos = result["conversations"]
        assert len(convos) == 3
        assert convos[0]["role"] == "system"
        assert convos[1]["role"] == "user"
        assert convos[2]["role"] == "assistant"

    def test_system_prompt_is_teacher(self):
        variables = [{"file_path": "a.py", "name": "x", "raw_code": "x = 1"}]
        scores = [(1, 1, 1, 1, 1, 1, 1)]

        result = format_training_example(variables, scores)
        assert result["conversations"][0]["content"] == TEACHER_SYSTEM_PROMPT

    def test_assistant_has_7_scores(self):
        variables = [
            {"file_path": "a.py", "name": "x", "raw_code": "x = 1"},
            {"file_path": "b.py", "name": "y", "raw_code": "y = 2"},
        ]
        scores = [(9, 8, 7, 6, 5, 4, 3), (1, 2, 3, 4, 5, 6, 7)]

        result = format_training_example(variables, scores)
        assistant = result["conversations"][2]["content"]

        lines = assistant.strip().split("\n")
        assert len(lines) == 2
        assert lines[0] == "1. 9,8,7,6,5,4,3"
        assert lines[1] == "2. 1,2,3,4,5,6,7"

    def test_no_think_stripped(self):
        """The /no_think prefix should be removed from training data."""
        variables = [{"file_path": "a.py", "name": "x", "raw_code": "x = 1"}]
        scores = [(1, 1, 1, 1, 1, 1, 1)]

        result = format_training_example(variables, scores)
        user_content = result["conversations"][1]["content"]
        assert not user_content.startswith("/no_think")


# =============================================================================
# build_training_batches
# =============================================================================


class TestBuildTrainingBatches:
    """Tests for batch building with varied sizes."""

    def _make_results(self, n: int) -> list[RawScoringResult]:
        return [
            RawScoringResult(
                element_id=f"scope:repo:main:file{i}.py:variable:var{i}:{i}",
                file_path=f"file{i}.py",
                name=f"var{i}",
                raw_code=f"var{i} = {i}",
                repo_name="test-repo",
                language="python",
                scores=(1, 1, 1, 1, 1, 1, 1),
                heuristic_drop=False,
                heuristic_reason="",
                teacher_model="test",
                timestamp="2025-01-01",
            )
            for i in range(n)
        ]

    def test_all_variables_included(self):
        results = self._make_results(50)
        batches = build_training_batches(results, seed=42)

        # Sum of all batch sizes should equal total results
        total = sum(
            len(b["conversations"][2]["content"].strip().split("\n"))
            for b in batches
        )
        assert total == 50

    def test_single_variable_produces_batch(self):
        results = self._make_results(1)
        batches = build_training_batches(results, seed=42)
        assert len(batches) == 1

    def test_deterministic_with_seed(self):
        results = self._make_results(20)
        batches1 = build_training_batches(results, seed=42)
        batches2 = build_training_batches(results, seed=42)
        assert batches1 == batches2

    def test_different_seeds_different_order(self):
        results = self._make_results(20)
        batches1 = build_training_batches(results, seed=42)
        batches2 = build_training_batches(results, seed=99)
        # Very likely different (not guaranteed but extremely unlikely with 20 items)
        assert batches1 != batches2


# =============================================================================
# Resume support
# =============================================================================


class TestResumeSupport:
    """Tests for raw result save/load."""

    def test_result_hash_deterministic(self):
        h1 = _result_hash("scope:repo:main:file.py:variable:x:1")
        h2 = _result_hash("scope:repo:main:file.py:variable:x:1")
        assert h1 == h2
        assert len(h1) == 16

    def test_result_hash_different_ids(self):
        h1 = _result_hash("scope:repo:main:a.py:variable:x:1")
        h2 = _result_hash("scope:repo:main:b.py:variable:y:2")
        assert h1 != h2

    def test_save_and_load(self, tmp_path):
        raw_dir = tmp_path / "raw"
        result = RawScoringResult(
            element_id="scope:repo:main:file.py:variable:MAX:1",
            file_path="file.py",
            name="MAX",
            raw_code="MAX = 10",
            repo_name="test",
            language="python",
            scores=(9, 1, 1, 8, 2, 8, 9),
            heuristic_drop=False,
            heuristic_reason="",
            teacher_model="teacher",
            timestamp="2025-01-01T00:00:00",
        )

        save_raw_result(result, raw_dir)
        loaded = load_existing_results(raw_dir)

        assert result.element_id in loaded
        loaded_result = loaded[result.element_id]
        assert tuple(loaded_result.scores) == (9, 1, 1, 8, 2, 8, 9)
        assert loaded_result.name == "MAX"

    def test_load_empty_dir(self, tmp_path):
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        loaded = load_existing_results(raw_dir)
        assert loaded == {}

    def test_load_nonexistent_dir(self, tmp_path):
        raw_dir = tmp_path / "nonexistent"
        loaded = load_existing_results(raw_dir)
        assert loaded == {}

    def test_load_skips_corrupt_files(self, tmp_path):
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        (raw_dir / "bad.json").write_text("not json")
        loaded = load_existing_results(raw_dir)
        assert loaded == {}


# =============================================================================
# _format_duration
# =============================================================================


class TestFormatDuration:
    """Tests for time formatting helper."""

    def test_seconds(self):
        assert _format_duration(45) == "0:45"

    def test_minutes(self):
        assert _format_duration(125) == "2:05"

    def test_hours(self):
        assert _format_duration(3661) == "1:01:01"

    def test_zero(self):
        assert _format_duration(0) == "0:00"


# =============================================================================
# NUM_SCORES constant
# =============================================================================


class TestConstants:
    """Tests for module constants."""

    def test_num_scores_is_7(self):
        assert NUM_SCORES == 7

    def test_teacher_prompt_mentions_all_dimensions(self):
        dimensions = [
            "config_value",
            "architectural_role",
            "data_definition",
            "general_usefulness",
            "value_complexity",
            "naming_quality",
            "scope_significance",
        ]
        for dim in dimensions:
            assert dim in TEACHER_SYSTEM_PROMPT, f"Missing dimension: {dim}"

    def test_teacher_prompt_example_has_7_scores(self):
        """Example output in prompt should have exactly 7 comma-separated scores."""
        # Find the "Correct output" example lines
        lines = TEACHER_SYSTEM_PROMPT.split("\n")
        example_lines = [
            l for l in lines
            if l.strip() and l.strip()[0].isdigit() and "." in l and "," in l
            and not l.strip().startswith("1. [")  # skip input examples
        ]
        for line in example_lines:
            # Extract scores after "N. "
            scores_part = line.strip().split(". ", 1)[1]
            scores = scores_part.split(",")
            assert len(scores) == 7, f"Expected 7 scores in example line: {line}"
