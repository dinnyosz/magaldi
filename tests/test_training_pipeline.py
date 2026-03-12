"""Tests for the variable scorer training pipeline.

Tests the data generation, quality filtering, parsing, and formatting
components of tools/training/generate_scoring_data.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add training tools to path
_project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_project_root / "tools" / "training"))
sys.path.insert(0, str(_project_root / "src"))

from generate_scoring_data import (
    NUM_SCORES,
    RawScoringResult,
    TEACHER_SYSTEM_PROMPT,
    _batch_to_example,
    _estimate_system_prompt_tokens,
    _estimate_variable_tokens,
    _format_duration,
    _is_trivial,
    _parse_teacher_scores,
    _result_hash,
    build_training_batches,
    format_training_example,
    load_existing_results,
    rebalance_results,
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


class TestEstimateTokens:
    """Tests for token estimation helpers."""

    def _make_result(self, raw_code: str = "x = 1", file_path: str = "a.py") -> RawScoringResult:
        return RawScoringResult(
            element_id="scope:repo:main:a.py:variable:x:1",
            file_path=file_path,
            name="x",
            raw_code=raw_code,
            repo_name="test",
            language="python",
            scores=(1, 1, 1, 1, 1, 1, 1),
            heuristic_drop=False,
            heuristic_reason="",
            teacher_model="test",
            timestamp="2025-01-01",
        )

    def test_estimate_variable_tokens_positive(self):
        result = self._make_result("MAX_RETRIES = 3")
        tokens = _estimate_variable_tokens(result)
        assert tokens > 0

    def test_longer_code_more_tokens(self):
        short = self._make_result("x = 1")
        long_code = self._make_result(
            "CORS_ORIGINS = ['http://localhost:3000', 'https://myapp.com', 'https://staging.myapp.com']"
        )
        assert _estimate_variable_tokens(long_code) > _estimate_variable_tokens(short)

    def test_longer_path_more_tokens(self):
        short_path = self._make_result("x = 1", file_path="a.py")
        long_path = self._make_result("x = 1", file_path="src/very/deep/nested/module/config.py")
        assert _estimate_variable_tokens(long_path) > _estimate_variable_tokens(short_path)

    def test_multiline_code_collapsed(self):
        """Newlines in raw_code are replaced with spaces for estimation."""
        result = self._make_result("ITEMS = [\n    'a',\n    'b',\n]")
        tokens = _estimate_variable_tokens(result)
        # Should still produce a positive estimate after collapsing
        assert tokens > 0
        # Should be similar to (but not necessarily identical to) the manually
        # collapsed version, since replace("\n", " ") preserves leading spaces
        result_flat = self._make_result("ITEMS = [ 'a', 'b', ]")
        flat_tokens = _estimate_variable_tokens(result_flat)
        # Within 50% of each other
        assert abs(tokens - flat_tokens) / max(tokens, flat_tokens) < 0.5

    def test_system_prompt_tokens_positive(self):
        tokens = _estimate_system_prompt_tokens()
        assert tokens > 0

    def test_system_prompt_tokens_plausible_range(self):
        """System prompt should be roughly 300-600 tokens."""
        tokens = _estimate_system_prompt_tokens()
        assert 200 < tokens < 800


class TestBatchToExample:
    """Tests for _batch_to_example helper."""

    def test_produces_valid_chatml(self):
        results = [
            RawScoringResult(
                element_id="scope:repo:main:a.py:variable:x:1",
                file_path="a.py",
                name="x",
                raw_code="x = 1",
                repo_name="test",
                language="python",
                scores=(9, 1, 1, 8, 2, 8, 9),
                heuristic_drop=False,
                heuristic_reason="",
                teacher_model="test",
                timestamp="2025-01-01",
            )
        ]
        example = _batch_to_example(results)
        assert "conversations" in example
        assert len(example["conversations"]) == 3
        roles = [m["role"] for m in example["conversations"]]
        assert roles == ["system", "user", "assistant"]

    def test_scores_match_input(self):
        results = [
            RawScoringResult(
                element_id="scope:repo:main:a.py:variable:x:1",
                file_path="a.py",
                name="x",
                raw_code="x = 1",
                repo_name="test",
                language="python",
                scores=(9, 8, 7, 6, 5, 4, 3),
                heuristic_drop=False,
                heuristic_reason="",
                teacher_model="test",
                timestamp="2025-01-01",
            )
        ]
        example = _batch_to_example(results)
        assistant = example["conversations"][2]["content"]
        assert "9,8,7,6,5,4,3" in assistant


class TestBuildTrainingBatches:
    """Tests for batch building with token-budget packing."""

    def _make_results(self, n: int, raw_code: str | None = None) -> list[RawScoringResult]:
        return [
            RawScoringResult(
                element_id=f"scope:repo:main:file{i}.py:variable:var{i}:{i}",
                file_path=f"file{i}.py",
                name=f"var{i}",
                raw_code=raw_code or f"var{i} = {i}",
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
        batches = build_training_batches(results, seed=42, max_seq_len=1024)

        # Sum of all batch sizes should equal total results
        total = sum(
            len(b["conversations"][2]["content"].strip().split("\n"))
            for b in batches
        )
        assert total == 50

    def test_single_variable_produces_batch(self):
        results = self._make_results(1)
        batches = build_training_batches(results, seed=42, max_seq_len=1024)
        assert len(batches) == 1

    def test_deterministic_with_seed(self):
        results = self._make_results(20)
        batches1 = build_training_batches(results, seed=42, max_seq_len=1024)
        batches2 = build_training_batches(results, seed=42, max_seq_len=1024)
        assert batches1 == batches2

    def test_different_seeds_different_order(self):
        results = self._make_results(20)
        batches1 = build_training_batches(results, seed=42, max_seq_len=1024)
        batches2 = build_training_batches(results, seed=99, max_seq_len=1024)
        # Very likely different (not guaranteed but extremely unlikely with 20 items)
        assert batches1 != batches2

    def test_smaller_budget_produces_more_batches(self):
        """A tighter token budget should produce more, smaller batches."""
        results = self._make_results(30)
        batches_large = build_training_batches(results, seed=42, max_seq_len=2048)
        batches_small = build_training_batches(results, seed=42, max_seq_len=512)
        assert len(batches_small) > len(batches_large)

    def test_no_batch_exceeds_budget(self):
        """Each batch's estimated tokens should fit within max_seq_len."""
        results = self._make_results(50)
        max_seq_len = 1024
        batches = build_training_batches(results, seed=42, max_seq_len=max_seq_len)

        system_tokens = _estimate_system_prompt_tokens()
        header_tokens = 8
        available = max_seq_len - system_tokens - header_tokens

        for batch in batches:
            assistant_content = batch["conversations"][2]["content"]
            batch_size = len(assistant_content.strip().split("\n"))
            # For non-oversized batches (>1 variable), total tokens should be <= available
            # Single-variable batches are allowed even if they're oversized
            if batch_size > 1:
                # Re-estimate: sum up token costs from user+assistant lines
                user_content = batch["conversations"][1]["content"]
                # Rough estimate: total chars / 4
                total_chars = len(user_content) + len(assistant_content)
                estimated_tokens = total_chars // 4 + 5 * batch_size
                # Allow some margin for estimation imprecision
                assert estimated_tokens <= available * 1.2, (
                    f"Batch with {batch_size} vars estimated at {estimated_tokens} "
                    f"tokens exceeds budget {available}"
                )

    def test_oversized_variable_gets_solo_batch(self):
        """A variable whose code alone exceeds the budget is placed alone."""
        # Create one very long variable and several short ones
        long_code = "HUGE_DICT = {" + ", ".join(f"'key_{i}': 'value_{i}'" for i in range(200)) + "}"
        results = self._make_results(5)
        results.append(
            RawScoringResult(
                element_id="scope:repo:main:big.py:variable:HUGE_DICT:1",
                file_path="big.py",
                name="HUGE_DICT",
                raw_code=long_code,
                repo_name="test-repo",
                language="python",
                scores=(9, 1, 1, 8, 2, 8, 9),
                heuristic_drop=False,
                heuristic_reason="",
                teacher_model="test",
                timestamp="2025-01-01",
            )
        )
        # Use a small budget so the big variable definitely exceeds it
        batches = build_training_batches(results, seed=42, max_seq_len=256)

        # The big variable should be in a solo batch
        solo_batches = [
            b for b in batches
            if len(b["conversations"][2]["content"].strip().split("\n")) == 1
            and "HUGE_DICT" in b["conversations"][1]["content"]
        ]
        assert len(solo_batches) == 1

    def test_all_variables_included_with_oversized(self):
        """Even oversized variables are included (not skipped)."""
        long_code = "BIG = " + "x" * 5000
        results = self._make_results(3)
        results.append(
            RawScoringResult(
                element_id="scope:repo:main:big.py:variable:BIG:1",
                file_path="big.py",
                name="BIG",
                raw_code=long_code,
                repo_name="test-repo",
                language="python",
                scores=(5, 5, 5, 5, 5, 5, 5),
                heuristic_drop=False,
                heuristic_reason="",
                teacher_model="test",
                timestamp="2025-01-01",
            )
        )
        batches = build_training_batches(results, seed=42, max_seq_len=512)

        total = sum(
            len(b["conversations"][2]["content"].strip().split("\n"))
            for b in batches
        )
        assert total == 4

    def test_empty_results(self):
        batches = build_training_batches([], seed=42, max_seq_len=1024)
        assert batches == []

    def test_does_not_mutate_input(self):
        """The function should not modify the caller's list."""
        results = self._make_results(10)
        original_ids = [r.element_id for r in results]
        build_training_batches(results, seed=42, max_seq_len=1024)
        assert [r.element_id for r in results] == original_ids


# =============================================================================
# rebalance_results
# =============================================================================


class TestRebalanceResults:
    """Tests for training data rebalancing."""

    def _make_result(
        self, name: str, scores: tuple[int, ...] = (1, 1, 1, 1, 1, 1, 1)
    ) -> RawScoringResult:
        return RawScoringResult(
            element_id=f"scope:repo:main:a.py:variable:{name}:1",
            file_path="a.py",
            name=name,
            raw_code=f"{name} = 1",
            repo_name="test",
            language="python",
            scores=scores,
            heuristic_drop=False,
            heuristic_reason="",
            teacher_model="test",
            timestamp="2025-01-01",
        )

    def _make_mixed_results(
        self, n_trivial: int, n_non_trivial: int
    ) -> list[RawScoringResult]:
        """Create a mix of trivial and non-trivial results."""
        results = []
        for i in range(n_trivial):
            results.append(self._make_result(f"triv_{i}", (1, 1, 1, 1, 1, 1, 1)))
        for i in range(n_non_trivial):
            results.append(self._make_result(f"keep_{i}", (9, 1, 1, 8, 2, 8, 9)))
        return results

    def test_is_trivial_all_ones(self):
        r = self._make_result("x", (1, 1, 1, 1, 1, 1, 1))
        assert _is_trivial(r)

    def test_is_trivial_not_all_ones(self):
        r = self._make_result("x", (1, 1, 1, 1, 1, 1, 2))
        assert not _is_trivial(r)

    def test_no_rebalancing_when_ratio_is_1(self):
        results = self._make_mixed_results(50, 50)
        rebalanced = rebalance_results(results, max_trivial_ratio=1.0)
        assert len(rebalanced) == 100

    def test_undersamples_trivial_to_target_ratio(self):
        # 70 trivial + 30 non-trivial = 70% trivial
        results = self._make_mixed_results(70, 30)
        rebalanced = rebalance_results(results, max_trivial_ratio=0.3, seed=42)

        trivial_count = sum(1 for r in rebalanced if _is_trivial(r))
        non_trivial_count = sum(1 for r in rebalanced if not _is_trivial(r))

        # All non-trivial should be kept
        assert non_trivial_count == 30
        # Trivial ratio should be <= 0.3
        actual_ratio = trivial_count / len(rebalanced)
        assert actual_ratio <= 0.31  # small margin for rounding

    def test_preserves_all_non_trivial(self):
        results = self._make_mixed_results(80, 20)
        rebalanced = rebalance_results(results, max_trivial_ratio=0.3, seed=42)

        non_trivial_names = {r.name for r in rebalanced if not _is_trivial(r)}
        expected_names = {f"keep_{i}" for i in range(20)}
        assert non_trivial_names == expected_names

    def test_no_change_when_already_balanced(self):
        # 20 trivial + 80 non-trivial = 20% trivial (< 30%)
        results = self._make_mixed_results(20, 80)
        rebalanced = rebalance_results(results, max_trivial_ratio=0.3, seed=42)
        assert len(rebalanced) == 100

    def test_all_trivial_returns_all(self):
        results = self._make_mixed_results(50, 0)
        rebalanced = rebalance_results(results, max_trivial_ratio=0.3, seed=42)
        assert len(rebalanced) == 50

    def test_deterministic_with_seed(self):
        results = self._make_mixed_results(70, 30)
        r1 = rebalance_results(results, max_trivial_ratio=0.3, seed=42)
        r2 = rebalance_results(results, max_trivial_ratio=0.3, seed=42)
        assert [r.element_id for r in r1] == [r.element_id for r in r2]

    def test_different_seeds_different_sampling(self):
        results = self._make_mixed_results(70, 30)
        r1 = rebalance_results(results, max_trivial_ratio=0.3, seed=42)
        r2 = rebalance_results(results, max_trivial_ratio=0.3, seed=99)
        # The trivial samples kept should differ
        trivial_ids_1 = {r.element_id for r in r1 if _is_trivial(r)}
        trivial_ids_2 = {r.element_id for r in r2 if _is_trivial(r)}
        assert trivial_ids_1 != trivial_ids_2

    def test_does_not_mutate_input(self):
        results = self._make_mixed_results(50, 50)
        original_len = len(results)
        rebalance_results(results, max_trivial_ratio=0.3, seed=42)
        assert len(results) == original_len

    def test_keeps_at_least_one_trivial(self):
        """Even with extreme ratio, keep at least 1 trivial example."""
        results = self._make_mixed_results(100, 1)
        rebalanced = rebalance_results(results, max_trivial_ratio=0.01, seed=42)
        trivial_count = sum(1 for r in rebalanced if _is_trivial(r))
        assert trivial_count >= 1


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
            ln for ln in lines
            if ln.strip() and ln.strip()[0].isdigit() and "." in ln and "," in ln
            and not ln.strip().startswith("1. [")  # skip input examples
        ]
        for line in example_lines:
            # Extract scores after "N. "
            scores_part = line.strip().split(". ", 1)[1]
            scores = scores_part.split(",")
            assert len(scores) == 7, f"Expected 7 scores in example line: {line}"
