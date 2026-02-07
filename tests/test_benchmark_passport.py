"""Tests for the passport embedding benchmark module."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from shared.cli.benchmark_passport import (
    ALL_VARIANT_NAMES,
    GroundTruthPair,
    VariantResult,
    _build_breadcrumbs,
    _cosine_similarity,
    _evaluate_variant,
    _format_calls,
    _format_params,
    _save_markdown,
    build_passport_text,
)

# ---------------------------------------------------------------------------
# Helper fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_doc() -> dict:
    """A realistic ES document for a function element."""
    return {
        "element_id": "magaldi:magaldi:main:src/shared/ai/embedding.py:function:embed_single:145",
        "name": "embed_single",
        "element_type": "function",
        "relative_path": "src/shared/ai/embedding.py",
        "signature": "def embed_single(self, text: str, timeout: int = 30) -> list[float]",
        "docstring": "Generate embedding for single text.",
        "summary": "Generates a vector embedding for a single text input. Delegates to the underlying LLM client.",
        "calls": [
            {"name": "embed", "receiver": "self._client", "line": 159},
            {"name": "EmbeddingError", "receiver": None, "line": 161},
        ],
        "parameters": [
            {"name": "self", "type": None},
            {"name": "text", "type": "str"},
            {"name": "timeout", "type": "int", "default": "30"},
        ],
        "return_type": "list[float]",
        "decorators": [],
        "detected_patterns": [],
        "parent_id": "magaldi:magaldi:main:src/shared/ai/embedding.py:class:CodeEmbeddingClient:85",
        "level": 2,
        "language": "python",
    }


@pytest.fixture
def sample_doc_no_optional() -> dict:
    """A minimal doc with no optional fields."""
    return {
        "element_id": "magaldi:magaldi:main:src/utils.py:function:helper:10",
        "name": "helper",
        "element_type": "function",
        "relative_path": "src/utils.py",
    }


@pytest.fixture
def file_summaries() -> dict[str, str]:
    """File summaries cache."""
    return {
        "src/shared/ai/embedding.py": "Code embedding client and text builders.",
        "src/utils.py": "Utility functions.",
    }


@pytest.fixture
def class_summaries() -> dict[str, str]:
    """Class summaries cache."""
    return {
        "magaldi:magaldi:main:src/shared/ai/embedding.py:class:CodeEmbeddingClient:85": (
            "Manages embedding generation via Ollama."
        ),
    }


@pytest.fixture
def ground_truth_pairs() -> list[GroundTruthPair]:
    return [
        GroundTruthPair(caller_id="caller_a", callee_id="callee_x", call_name="foo"),
        GroundTruthPair(caller_id="caller_b", callee_id="callee_y", call_name="bar"),
        GroundTruthPair(caller_id="caller_c", callee_id="callee_z", call_name="baz"),
    ]


# ---------------------------------------------------------------------------
# Tests: breadcrumbs, params, calls helpers
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_build_breadcrumbs_with_class_parent(self, sample_doc: dict) -> None:
        result = _build_breadcrumbs(sample_doc)
        assert "src/shared/ai/embedding.py" in result
        assert "CodeEmbeddingClient" in result
        assert " > " in result

    def test_build_breadcrumbs_no_parent(self, sample_doc_no_optional: dict) -> None:
        result = _build_breadcrumbs(sample_doc_no_optional)
        assert result == "src/utils.py"

    def test_build_breadcrumbs_non_class_parent(self) -> None:
        doc = {
            "relative_path": "foo.py",
            "parent_id": "s:r:m:foo.py:file:foo.py:1",
        }
        # parent is a file, not class — should only have the path
        result = _build_breadcrumbs(doc)
        assert result == "foo.py"

    def test_format_params_with_types(self, sample_doc: dict) -> None:
        result = _format_params(sample_doc)
        assert "text: str" in result
        assert "timeout: int" in result
        assert "self" in result

    def test_format_params_empty(self, sample_doc_no_optional: dict) -> None:
        assert _format_params(sample_doc_no_optional) == ""

    def test_format_calls_deduplicates(self) -> None:
        doc = {
            "calls": [
                {"name": "foo", "receiver": "self"},
                {"name": "foo", "receiver": "self"},
                {"name": "bar", "receiver": None},
            ]
        }
        result = _format_calls(doc)
        assert result == "self.foo, bar"

    def test_format_calls_empty(self) -> None:
        assert _format_calls({}) == ""
        assert _format_calls({"calls": []}) == ""


# ---------------------------------------------------------------------------
# Tests: passport text variants
# ---------------------------------------------------------------------------


class TestBuildPassportText:
    """All variants start with baseline then add metadata."""

    def test_baseline_has_file_and_function(self, sample_doc: dict) -> None:
        text = build_passport_text("baseline", sample_doc, {}, {}, {}, {})
        assert "File: src/shared/ai/embedding.py" in text
        assert "Function: embed_single" in text
        assert "Summary:" in text
        assert "Signature:" in text
        assert "Docstring:" in text

    def test_baseline_includes_file_context(
        self, sample_doc: dict, file_summaries: dict, class_summaries: dict
    ) -> None:
        text = build_passport_text(
            "baseline", sample_doc, file_summaries, class_summaries, {}, {}
        )
        assert "File context: Code embedding client" in text

    def test_baseline_includes_class_context(
        self, sample_doc: dict, file_summaries: dict, class_summaries: dict
    ) -> None:
        text = build_passport_text(
            "baseline", sample_doc, file_summaries, class_summaries, {}, {}
        )
        assert "Class context: Manages embedding generation" in text

    def test_baseline_no_class_context_for_non_method(
        self, sample_doc_no_optional: dict, file_summaries: dict
    ) -> None:
        text = build_passport_text(
            "baseline", sample_doc_no_optional, file_summaries, {}, {}, {}
        )
        assert "Class context:" not in text

    def test_baseline_truncates_long_docstring(self) -> None:
        doc = {
            "element_id": "x",
            "name": "func",
            "relative_path": "a.py",
            "docstring": "x" * 600,
        }
        text = build_passport_text("baseline", doc, {}, {}, {}, {})
        assert "..." in text

    def test_plus_calls_adds_calls(
        self, sample_doc: dict, file_summaries: dict, class_summaries: dict
    ) -> None:
        text = build_passport_text(
            "plus_calls", sample_doc, file_summaries, class_summaries, {}, {}
        )
        # Should have baseline content
        assert "File: src/shared/ai/embedding.py" in text
        assert "Function: embed_single" in text
        # Plus calls
        assert "Calls:" in text
        assert "self._client.embed" in text

    def test_plus_calls_without_calls(self, sample_doc_no_optional: dict) -> None:
        text = build_passport_text("plus_calls", sample_doc_no_optional, {}, {}, {}, {})
        assert "Calls:" not in text

    def test_plus_params_adds_params_and_returns(
        self, sample_doc: dict, file_summaries: dict, class_summaries: dict
    ) -> None:
        text = build_passport_text(
            "plus_params", sample_doc, file_summaries, class_summaries, {}, {}
        )
        assert "Params:" in text
        assert "text: str" in text
        assert "Returns: list[float]" in text
        # Should NOT have calls
        assert "Calls:" not in text

    def test_plus_calls_params_adds_both(
        self, sample_doc: dict, file_summaries: dict, class_summaries: dict
    ) -> None:
        text = build_passport_text(
            "plus_calls_params", sample_doc, file_summaries, class_summaries, {}, {}
        )
        assert "Calls:" in text
        assert "Params:" in text
        assert "Returns:" in text

    def test_plus_full_adds_all_structural(
        self, sample_doc: dict, file_summaries: dict, class_summaries: dict
    ) -> None:
        text = build_passport_text(
            "plus_full", sample_doc, file_summaries, class_summaries, {}, {}
        )
        assert "Calls:" in text
        assert "Params:" in text
        assert "Returns:" in text
        assert "Location:" in text
        # No decorators/patterns since empty in sample_doc
        assert "Decorators:" not in text

    def test_plus_full_with_decorators_and_patterns(self) -> None:
        doc = {
            "element_id": "x:x:main:f.py:function:foo:1",
            "name": "foo",
            "relative_path": "f.py",
            "decorators": ["@staticmethod"],
            "detected_patterns": ["factory"],
            "calls": [{"name": "bar", "receiver": None}],
        }
        text = build_passport_text("plus_full", doc, {}, {}, {}, {})
        assert "Decorators: @staticmethod" in text
        assert "Patterns: factory" in text

    def test_plus_imports_adds_imports(
        self, sample_doc: dict, file_summaries: dict, class_summaries: dict
    ) -> None:
        imports_cache = {
            "src/shared/ai/embedding.py": [
                {"name": "CodeElement", "module": "magaldi_core.code_parser"},
                {"name": "EmbeddingClient", "module": "shared.ai.llm_client"},
            ]
        }
        text = build_passport_text(
            "plus_imports",
            sample_doc,
            file_summaries,
            class_summaries,
            imports_cache,
            {},
        )
        assert "Imports:" in text
        assert "magaldi_core.code_parser" in text
        assert "shared.ai.llm_client" in text
        # Should also have plus_full content
        assert "Calls:" in text

    def test_plus_imports_no_imports(self, sample_doc: dict) -> None:
        text = build_passport_text("plus_imports", sample_doc, {}, {}, {}, {})
        assert "Imports:" not in text

    def test_plus_siblings_adds_siblings(
        self, sample_doc: dict, file_summaries: dict, class_summaries: dict
    ) -> None:
        siblings_cache = {sample_doc["element_id"]: ["embed_batch", "verify_model"]}
        text = build_passport_text(
            "plus_siblings",
            sample_doc,
            file_summaries,
            class_summaries,
            {},
            siblings_cache,
        )
        assert "Siblings:" in text
        assert "embed_batch" in text
        assert "verify_model" in text

    def test_plus_siblings_no_siblings(self, sample_doc: dict) -> None:
        text = build_passport_text("plus_siblings", sample_doc, {}, {}, {}, {})
        assert "Siblings:" not in text

    def test_unknown_variant_still_returns_baseline(self, sample_doc: dict) -> None:
        text = build_passport_text("nonexistent_variant", sample_doc, {}, {}, {}, {})
        # Should still get baseline content (fallback)
        assert "File: src/shared/ai/embedding.py" in text
        assert "Function: embed_single" in text

    def test_all_variants_produce_nonempty(self, sample_doc: dict) -> None:
        for variant in ALL_VARIANT_NAMES:
            text = build_passport_text(variant, sample_doc, {}, {}, {}, {})
            assert len(text) > 0, f"Variant {variant} produced empty text"

    def test_all_variants_contain_baseline(self, sample_doc: dict) -> None:
        """Every variant should contain the baseline content."""
        baseline = build_passport_text("baseline", sample_doc, {}, {}, {}, {})
        for variant in ALL_VARIANT_NAMES:
            text = build_passport_text(variant, sample_doc, {}, {}, {}, {})
            assert baseline in text, (
                f"Variant {variant} doesn't contain baseline"
            )

    def test_plus_variants_are_supersets(
        self, sample_doc: dict, file_summaries: dict, class_summaries: dict
    ) -> None:
        """Each plus variant should be a superset of the baseline."""
        baseline = build_passport_text(
            "baseline", sample_doc, file_summaries, class_summaries, {}, {}
        )
        for variant in ALL_VARIANT_NAMES:
            if variant == "baseline":
                continue
            text = build_passport_text(
                variant, sample_doc, file_summaries, class_summaries, {}, {}
            )
            assert text.startswith(baseline) or baseline in text, (
                f"Variant {variant} is not a superset of baseline"
            )


# ---------------------------------------------------------------------------
# Tests: cosine similarity
# ---------------------------------------------------------------------------


class TestCosineSimilarity:
    def test_identical_vectors(self) -> None:
        v = [0.5, 0.5, 0.5, 0.5]
        assert _cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self) -> None:
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert _cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors(self) -> None:
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert _cosine_similarity(a, b) == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# Tests: evaluation logic
# ---------------------------------------------------------------------------


class TestEvaluateVariant:
    def test_perfect_ranking(self) -> None:
        """When callee is always most similar, MRR=1.0 and Hit@1=1.0."""
        pairs = [
            GroundTruthPair("A", "B", "call_b"),
        ]
        # caller A has sim=0.9 with callee B, sim=0.1 with neg_1
        embeddings = {
            "A": [1.0, 0.0],
            "B": [0.95, 0.31],  # very similar to A
            "neg_1": [0.0, 1.0],  # orthogonal to A
        }
        result = _evaluate_variant("test", pairs, embeddings, ["neg_1"])
        assert result.mrr == pytest.approx(1.0)
        assert result.hit_at_1 == pytest.approx(1.0)
        assert result.hit_at_3 == pytest.approx(1.0)
        assert result.hit_at_5 == pytest.approx(1.0)
        assert result.mean_pos_sim > result.mean_neg_sim
        assert result.separation > 0

    def test_worst_ranking(self) -> None:
        """When all negatives score higher than callee, rank = n_neg + 1."""
        pairs = [
            GroundTruthPair("A", "B", "call_b"),
        ]
        embeddings = {
            "A": [1.0, 0.0],
            "B": [0.0, 1.0],  # orthogonal to caller
            "neg_1": [0.99, 0.14],  # very similar to caller
            "neg_2": [0.95, 0.31],
        }
        result = _evaluate_variant("test", pairs, embeddings, ["neg_1", "neg_2"])
        assert result.mrr < 0.5  # rank 3 → MRR = 0.333
        assert result.hit_at_1 == pytest.approx(0.0)

    def test_missing_embeddings_skipped(self) -> None:
        """Pairs with missing embeddings are silently skipped."""
        pairs = [
            GroundTruthPair("A", "B", "call_b"),
            GroundTruthPair("C", "D", "call_d"),  # C missing
        ]
        embeddings = {
            "A": [1.0, 0.0],
            "B": [1.0, 0.0],
        }
        result = _evaluate_variant("test", pairs, embeddings, [])
        # Only one pair evaluated (A→B), with no negatives → rank=1
        assert result.mrr == pytest.approx(1.0)

    def test_no_valid_pairs_returns_zero(self) -> None:
        """When no pairs have embeddings, returns zero metrics."""
        pairs = [GroundTruthPair("X", "Y", "call")]
        result = _evaluate_variant("test", pairs, {}, [])
        assert result.mrr == 0.0

    def test_worst_pairs_sorted_descending(self) -> None:
        """worst_pairs should be sorted by rank descending."""
        pairs = [
            GroundTruthPair("A", "B", "good"),
            GroundTruthPair("A", "C", "bad"),
        ]
        embeddings = {
            "A": [1.0, 0.0],
            "B": [0.98, 0.2],  # similar
            "C": [0.0, 1.0],  # orthogonal
            "neg_1": [0.5, 0.87],
        }
        result = _evaluate_variant("test", pairs, embeddings, ["neg_1"])
        assert len(result.worst_pairs) == 2
        # First should be the worse-ranked pair
        assert result.worst_pairs[0]["rank"] >= result.worst_pairs[1]["rank"]


# ---------------------------------------------------------------------------
# Tests: markdown output
# ---------------------------------------------------------------------------


class TestSaveMarkdown:
    def test_creates_file_with_results(self) -> None:
        results = [
            VariantResult(name="baseline", mrr=0.8, hit_at_1=0.7, hit_at_3=0.85,
                          hit_at_5=0.9, mean_pos_sim=0.6, mean_neg_sim=0.3,
                          separation=0.3, embed_time_s=5.0),
            VariantResult(name="plus_calls", mrr=0.5, hit_at_1=0.4, hit_at_3=0.6,
                          hit_at_5=0.7, mean_pos_sim=0.5, mean_neg_sim=0.35,
                          separation=0.15, embed_time_s=3.0),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _save_markdown(results, n_pairs=100, n_negatives=200,
                                  n_callers=50, seed=42, output_dir=tmpdir)
            content = Path(path).read_text()
            assert "# Semantic Passport Embedding Benchmark" in content
            assert "baseline" in content
            assert "plus_calls" in content
            assert "0.8000" in content  # MRR for best
            assert "**Best variant:** `baseline`" in content
            # Analysis hints should be present
            assert "How to Read the Results" in content
            assert "Variant Descriptions" in content
            assert "Next Steps" in content

    def test_includes_error_analysis(self) -> None:
        results = [
            VariantResult(
                name="baseline",
                mrr=0.5,
                worst_pairs=[
                    {"caller_id": "s:r:m:f.py:function:foo:1",
                     "callee_id": "s:r:m:g.py:function:bar:10",
                     "call_name": "bar",
                     "pos_sim": 0.2,
                     "rank": 50},
                ],
            ),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _save_markdown(results, 10, 100, 5, 42, tmpdir)
            content = Path(path).read_text()
            assert "Error Analysis" in content
            assert "bar" in content
            assert "50" in content

    def test_creates_output_directory(self) -> None:
        results = [VariantResult(name="x", mrr=0.5)]
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_dir = str(Path(tmpdir) / "deep" / "nested")
            path = _save_markdown(results, 1, 1, 1, 42, nested_dir)
            assert Path(path).exists()

    def test_variant_descriptions_reference_baseline(self) -> None:
        results = [VariantResult(name="baseline", mrr=0.5)]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _save_markdown(results, 1, 1, 1, 42, tmpdir)
            content = Path(path).read_text()
            assert "build_summary_embedding_text" in content
            assert "plus_calls" in content
            assert "plus_full" in content


# ---------------------------------------------------------------------------
# Tests: auto-detect repo
# ---------------------------------------------------------------------------


class TestAutoDetect:
    def test_auto_detect_from_magaldi_yaml(self, tmp_path: Path) -> None:
        yaml_file = tmp_path / "magaldi.yaml"
        yaml_file.write_text("scope: myorg\nrepository: myrepo\n")
        import os

        from shared.cli.benchmark_passport import _auto_detect_repo

        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            scope, repo = _auto_detect_repo()
            assert scope == "myorg"
            assert repo == "myrepo"
        finally:
            os.chdir(original_cwd)

    def test_auto_detect_missing_yaml(self, tmp_path: Path) -> None:
        import os
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            from shared.cli.benchmark_passport import _auto_detect_repo

            scope, repo = _auto_detect_repo()
            assert scope == ""
            assert repo == ""
        finally:
            os.chdir(original_cwd)


# ---------------------------------------------------------------------------
# Tests: CLI tools entry point
# ---------------------------------------------------------------------------


class TestToolsCLI:
    def test_passport_benchmark_help(self) -> None:
        """Verify the click CLI group is properly configured."""
        from click.testing import CliRunner

        from shared.cli.tools import tools_cli

        runner = CliRunner()
        result = runner.invoke(tools_cli, ["passport-benchmark", "--help"])
        assert result.exit_code == 0
        assert "passport" in result.output.lower()
        assert "--scope" in result.output
        assert "--variants" in result.output

    def test_tools_cli_help(self) -> None:
        from click.testing import CliRunner

        from shared.cli.tools import tools_cli

        runner = CliRunner()
        result = runner.invoke(tools_cli, ["--help"])
        assert result.exit_code == 0
        assert "passport-benchmark" in result.output

    def test_passport_benchmark_no_scope_no_yaml(self) -> None:
        """Should error when no scope/repo and no magaldi.yaml."""
        from click.testing import CliRunner

        from shared.cli.tools import tools_cli

        runner = CliRunner()
        with runner.isolated_filesystem():
            result = runner.invoke(tools_cli, ["passport-benchmark"])
            assert result.exit_code != 0


# ---------------------------------------------------------------------------
# Tests: data classes
# ---------------------------------------------------------------------------


class TestDataClasses:
    def test_ground_truth_pair(self) -> None:
        pair = GroundTruthPair("a", "b", "call_b")
        assert pair.caller_id == "a"
        assert pair.callee_id == "b"
        assert pair.call_name == "call_b"

    def test_variant_result_defaults(self) -> None:
        result = VariantResult(name="test")
        assert result.mrr == 0.0
        assert result.worst_pairs == []
