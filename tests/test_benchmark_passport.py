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
    def test_name_only(self, sample_doc: dict) -> None:
        text = build_passport_text("name_only", sample_doc, {}, {})
        assert text == "embed_single"

    def test_signature_only(self, sample_doc: dict) -> None:
        text = build_passport_text("signature_only", sample_doc, {}, {})
        assert "embed_single" in text
        assert "Signature:" in text
        assert "Returns: list[float]" in text
        assert "Params:" in text

    def test_signature_only_minimal(self, sample_doc_no_optional: dict) -> None:
        text = build_passport_text("signature_only", sample_doc_no_optional, {}, {})
        assert text == "helper"

    def test_baseline_summary(self, sample_doc: dict) -> None:
        text = build_passport_text("baseline_summary", sample_doc, {}, {})
        assert "File: src/shared/ai/embedding.py" in text
        assert "Function: embed_single" in text
        assert "Summary:" in text
        assert "Signature:" in text
        assert "Docstring:" in text

    def test_baseline_summary_truncates_long_docstring(self) -> None:
        doc = {
            "element_id": "x",
            "name": "func",
            "relative_path": "a.py",
            "docstring": "x" * 600,
        }
        text = build_passport_text("baseline_summary", doc, {}, {})
        assert "..." in text

    def test_calls_fingerprint(self, sample_doc: dict) -> None:
        text = build_passport_text("calls_fingerprint", sample_doc, {}, {})
        assert "embed_single" in text
        assert "Calls:" in text
        assert "self._client.embed" in text

    def test_full_passport(self, sample_doc: dict) -> None:
        text = build_passport_text("full_passport", sample_doc, {}, {})
        assert "embed_single" in text
        assert "Location:" in text
        assert "Signature:" in text
        assert "Params:" in text
        assert "Returns:" in text
        assert "Calls:" in text
        assert "Summary:" in text
        # Summary should be first sentence only
        assert "Generates a vector embedding for a single text input." in text
        assert "Delegates to" not in text  # second sentence trimmed

    def test_passport_with_imports(self, sample_doc: dict) -> None:
        imports_cache = {
            "src/shared/ai/embedding.py": [
                {"name": "CodeElement", "module": "magaldi_core.code_parser"},
                {"name": "EmbeddingClient", "module": "shared.ai.llm_client"},
            ]
        }
        text = build_passport_text("passport_with_imports", sample_doc, imports_cache, {})
        assert "Imports:" in text
        assert "magaldi_core.code_parser" in text
        assert "shared.ai.llm_client" in text

    def test_passport_with_imports_no_imports(self, sample_doc: dict) -> None:
        text = build_passport_text("passport_with_imports", sample_doc, {}, {})
        assert "Imports:" not in text

    def test_passport_with_siblings(self, sample_doc: dict) -> None:
        siblings_cache = {
            sample_doc["element_id"]: ["embed_batch", "verify_model"]
        }
        text = build_passport_text("passport_with_siblings", sample_doc, {}, siblings_cache)
        assert "Siblings:" in text
        assert "embed_batch" in text
        assert "verify_model" in text

    def test_passport_with_siblings_no_siblings(self, sample_doc: dict) -> None:
        text = build_passport_text("passport_with_siblings", sample_doc, {}, {})
        assert "Siblings:" not in text

    def test_unknown_variant_returns_name(self, sample_doc: dict) -> None:
        text = build_passport_text("nonexistent_variant", sample_doc, {}, {})
        assert text == "embed_single"

    def test_all_variants_produce_nonempty(self, sample_doc: dict) -> None:
        for variant in ALL_VARIANT_NAMES:
            text = build_passport_text(variant, sample_doc, {}, {})
            assert len(text) > 0, f"Variant {variant} produced empty text"


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
            VariantResult(name="variant_a", mrr=0.8, hit_at_1=0.7, hit_at_3=0.85,
                          hit_at_5=0.9, mean_pos_sim=0.6, mean_neg_sim=0.3,
                          separation=0.3, embed_time_s=5.0),
            VariantResult(name="variant_b", mrr=0.5, hit_at_1=0.4, hit_at_3=0.6,
                          hit_at_5=0.7, mean_pos_sim=0.5, mean_neg_sim=0.35,
                          separation=0.15, embed_time_s=3.0),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _save_markdown(results, n_pairs=100, n_negatives=200,
                                  n_callers=50, seed=42, output_dir=tmpdir)
            content = Path(path).read_text()
            assert "# Semantic Passport Embedding Benchmark" in content
            assert "variant_a" in content
            assert "variant_b" in content
            assert "0.8000" in content  # MRR for best
            assert "**Best variant:** `variant_a`" in content
            # Analysis hints should be present
            assert "How to Read the Results" in content
            assert "Variant Descriptions" in content
            assert "Next Steps" in content

    def test_includes_error_analysis(self) -> None:
        results = [
            VariantResult(
                name="test_v",
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
