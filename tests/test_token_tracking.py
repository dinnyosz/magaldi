"""Tests for token usage tracking in Phase 7/8 timing stats classes."""

from __future__ import annotations

from shared.ai.clustering.clusterer import LabelingTimingStats
from shared.ai.clustering.feature_processor import (
    FeatureTimingStats,
    ProcessedFeature,
)
from shared.ai.clustering.subfeature_processor import (
    ProcessedSubfeature,
    SubfeatureTimingStats,
)
from shared.ai.glossary.ai_extractor import GlossaryTimingStats

# =============================================================================
# LABELING TIMING STATS TOKEN TRACKING
# =============================================================================


class TestLabelingTimingStatsTokens:
    """Tests for token tracking in LabelingTimingStats."""

    def test_record_accumulates_tokens(self):
        stats = LabelingTimingStats()
        stats.record(0.5, prompt_tokens=100, response_tokens=10, model_name="qwen3:4b")
        stats.record(0.3, prompt_tokens=200, response_tokens=20, model_name="qwen3:4b")

        assert stats.total_input_tokens == 300
        assert stats.total_output_tokens == 30

    def test_record_per_model_tracking(self):
        stats = LabelingTimingStats()
        stats.record(0.5, prompt_tokens=100, response_tokens=10, model_name="qwen3:4b")
        stats.record(0.3, prompt_tokens=200, response_tokens=20, model_name="qwen3:2b")

        assert stats.model_input_tokens == {"qwen3:4b": 100, "qwen3:2b": 200}
        assert stats.model_output_tokens == {"qwen3:4b": 10, "qwen3:2b": 20}
        assert stats.model_sample_counts == {"qwen3:4b": 1, "qwen3:2b": 1}

    def test_record_backward_compatible_without_tokens(self):
        stats = LabelingTimingStats()
        stats.record(0.5)  # No token args

        assert stats.label_count == 1
        assert stats.total_input_tokens == 0
        assert stats.total_output_tokens == 0

    def test_get_token_usage_summary_shape(self):
        stats = LabelingTimingStats()
        stats.record(0.5, prompt_tokens=100, response_tokens=10, model_name="qwen3:4b")
        stats.record(0.3, prompt_tokens=200, response_tokens=20, model_name="qwen3:4b")

        summary = stats.get_token_usage_summary()

        assert "by_type" in summary
        assert "by_model" in summary
        assert "totals" in summary

        # by_type should have "labeling" key
        assert "labeling" in summary["by_type"]
        assert summary["by_type"]["labeling"]["input"] == 300
        assert summary["by_type"]["labeling"]["output"] == 30
        assert summary["by_type"]["labeling"]["count"] == 2

        # by_model
        assert "qwen3:4b" in summary["by_model"]
        assert summary["by_model"]["qwen3:4b"]["input"] == 300
        assert summary["by_model"]["qwen3:4b"]["count"] == 2

        # totals
        assert summary["totals"]["input"] == 300
        assert summary["totals"]["output"] == 30
        assert summary["totals"]["count"] == 2

    def test_get_token_usage_summary_empty(self):
        stats = LabelingTimingStats()
        summary = stats.get_token_usage_summary()

        assert summary["by_type"] == {}
        assert summary["by_model"] == {}
        assert summary["totals"]["input"] == 0
        assert summary["totals"]["output"] == 0
        assert summary["totals"]["count"] == 0


# =============================================================================
# FEATURE TIMING STATS TOKEN TRACKING
# =============================================================================


class TestFeatureTimingStatsTokens:
    """Tests for token tracking in FeatureTimingStats."""

    def test_record_accumulates_tokens(self):
        stats = FeatureTimingStats()
        stats.record(1.0, 0.2, tier=2048, prompt_tokens=500, response_tokens=50, model_name="qwen3:4b")
        stats.record(0.8, 0.1, tier=4096, prompt_tokens=1000, response_tokens=100, model_name="qwen3:4b")

        assert stats.total_input_tokens == 1500
        assert stats.total_output_tokens == 150

    def test_record_backward_compatible_without_tokens(self):
        stats = FeatureTimingStats()
        stats.record(1.0, 0.2, tier=2048)  # No token args

        assert stats.summarize_count == 1
        assert stats.total_input_tokens == 0

    def test_get_token_usage_summary_shape(self):
        stats = FeatureTimingStats()
        stats.record(1.0, 0.2, prompt_tokens=500, response_tokens=50, model_name="model_a")
        stats.record(0.8, 0.1, prompt_tokens=300, response_tokens=30, model_name="model_b")

        summary = stats.get_token_usage_summary()

        assert "feature_summary" in summary["by_type"]
        assert summary["by_type"]["feature_summary"]["input"] == 800
        assert summary["by_type"]["feature_summary"]["output"] == 80
        assert summary["by_type"]["feature_summary"]["count"] == 2

        assert "model_a" in summary["by_model"]
        assert "model_b" in summary["by_model"]
        assert summary["totals"]["input"] == 800
        assert summary["totals"]["output"] == 80

    def test_thread_safety(self):
        """Token tracking is thread-safe (uses _lock)."""
        import threading

        stats = FeatureTimingStats()
        errors = []

        def record_many():
            try:
                for _ in range(100):
                    stats.record(0.01, 0.01, prompt_tokens=10, response_tokens=1, model_name="m")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=record_many) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert stats.summarize_count == 400
        assert stats.total_input_tokens == 4000
        assert stats.total_output_tokens == 400


# =============================================================================
# SUBFEATURE TIMING STATS TOKEN TRACKING
# =============================================================================


class TestSubfeatureTimingStatsTokens:
    """Tests for token tracking in SubfeatureTimingStats."""

    def test_record_accumulates_tokens(self):
        stats = SubfeatureTimingStats()
        stats.record(0.5, 0.1, tier=2048, prompt_tokens=250, response_tokens=25, model_name="qwen3:4b")

        assert stats.total_input_tokens == 250
        assert stats.total_output_tokens == 25

    def test_get_token_usage_summary_shape(self):
        stats = SubfeatureTimingStats()
        stats.record(0.5, 0.1, prompt_tokens=250, response_tokens=25, model_name="qwen3:4b")

        summary = stats.get_token_usage_summary()

        assert "subfeature_summary" in summary["by_type"]
        assert summary["by_type"]["subfeature_summary"]["input"] == 250
        assert summary["totals"]["input"] == 250
        assert summary["totals"]["output"] == 25


# =============================================================================
# GLOSSARY TIMING STATS TOKEN TRACKING
# =============================================================================


class TestGlossaryTimingStatsTokens:
    """Tests for token tracking in GlossaryTimingStats."""

    def test_record_api_call_with_tokens(self):
        stats = GlossaryTimingStats()
        stats.record_api_call(
            0.5, tier=2048,
            prompt_tokens=100, response_tokens=10,
            model_name="qwen3:4b", call_type="glossary_extract",
        )

        assert stats.input_tokens_by_type == {"glossary_extract": 100}
        assert stats.output_tokens_by_type == {"glossary_extract": 10}
        assert stats.model_input_tokens == {"qwen3:4b": 100}

    def test_record_api_call_backward_compatible(self):
        stats = GlossaryTimingStats()
        stats.record_api_call(0.5, tier=2048)  # No token args

        assert stats.features_processed == 1
        assert stats.input_tokens_by_type == {}

    def test_two_call_types(self):
        stats = GlossaryTimingStats()
        stats.record_api_call(
            0.5, prompt_tokens=100, response_tokens=10,
            model_name="m", call_type="glossary_extract",
        )
        stats.record_api_call(
            0.8, prompt_tokens=500, response_tokens=80,
            model_name="m", call_type="glossary_summary",
        )

        summary = stats.get_token_usage_summary()

        assert "glossary_extract" in summary["by_type"]
        assert "glossary_summary" in summary["by_type"]
        assert summary["by_type"]["glossary_extract"]["input"] == 100
        assert summary["by_type"]["glossary_summary"]["input"] == 500
        assert summary["totals"]["input"] == 600
        assert summary["totals"]["output"] == 90
        assert summary["totals"]["count"] == 2
        assert summary["by_model"]["m"]["input"] == 600

    def test_get_token_usage_summary_empty(self):
        stats = GlossaryTimingStats()
        summary = stats.get_token_usage_summary()

        assert summary["by_type"] == {}
        assert summary["by_model"] == {}
        assert summary["totals"]["input"] == 0
        assert summary["totals"]["output"] == 0
        assert summary["totals"]["count"] == 0


# =============================================================================
# PROCESSED DATA CLASSES TOKEN FIELDS
# =============================================================================


class TestProcessedDataClassTokenFields:
    """Tests for token fields on ProcessedFeature and ProcessedSubfeature."""

    def test_processed_feature_has_token_fields(self):
        pf = ProcessedFeature(
            cluster_id=1, success=True,
            summarize_time=1.0, embed_time=0.2,
            prompt_tokens=500, response_tokens=50,
            model_name="qwen3:4b",
        )
        assert pf.prompt_tokens == 500
        assert pf.response_tokens == 50
        assert pf.model_name == "qwen3:4b"

    def test_processed_feature_defaults_zero(self):
        pf = ProcessedFeature(cluster_id=1, success=True, summarize_time=1.0, embed_time=0.2)
        assert pf.prompt_tokens == 0
        assert pf.response_tokens == 0
        assert pf.model_name == ""

    def test_processed_subfeature_has_token_fields(self):
        ps = ProcessedSubfeature(
            parent_label="test", sub_label="sub",
            success=True, summarize_time=0.5, embed_time=0.1,
            prompt_tokens=250, response_tokens=25,
            model_name="qwen3:4b",
        )
        assert ps.prompt_tokens == 250
        assert ps.response_tokens == 25

    def test_processed_subfeature_defaults_zero(self):
        ps = ProcessedSubfeature(
            parent_label="test", sub_label="sub",
            success=True, summarize_time=0.5, embed_time=0.1,
        )
        assert ps.prompt_tokens == 0
        assert ps.response_tokens == 0
        assert ps.model_name == ""


# =============================================================================
# MERGE TOKEN SUMMARIES
# =============================================================================


class TestMergeTokenSummaries:
    """Tests for _merge_token_summaries helper."""

    def test_merge_all_three(self):
        from shared.cli.feature_commands import _merge_token_summaries

        labeling = LabelingTimingStats()
        labeling.record(0.5, prompt_tokens=100, response_tokens=10, model_name="m")

        feature = FeatureTimingStats()
        feature.record(1.0, 0.2, prompt_tokens=500, response_tokens=50, model_name="m")

        subfeature = SubfeatureTimingStats()
        subfeature.record(0.5, 0.1, prompt_tokens=200, response_tokens=20, model_name="m")

        merged = _merge_token_summaries(labeling, feature, subfeature)

        assert "labeling" in merged["by_type"]
        assert "feature_summary" in merged["by_type"]
        assert "subfeature_summary" in merged["by_type"]
        assert merged["totals"]["input"] == 800
        assert merged["totals"]["output"] == 80
        assert merged["totals"]["count"] == 3
        assert merged["by_model"]["m"]["input"] == 800

    def test_merge_with_none(self):
        from shared.cli.feature_commands import _merge_token_summaries

        labeling = LabelingTimingStats()
        labeling.record(0.5, prompt_tokens=100, response_tokens=10, model_name="m")

        merged = _merge_token_summaries(labeling, None, None)

        assert merged["totals"]["input"] == 100
        assert merged["totals"]["output"] == 10

    def test_merge_empty(self):
        from shared.cli.feature_commands import _merge_token_summaries

        merged = _merge_token_summaries(None, None, None)

        assert merged["totals"]["input"] == 0
        assert merged["totals"]["output"] == 0
        assert merged["totals"]["count"] == 0

    def test_merge_model_aggregation_across_phases(self):
        """Models are aggregated across labeling and feature phases."""
        from shared.cli.feature_commands import _merge_token_summaries

        labeling = LabelingTimingStats()
        labeling.record(0.5, prompt_tokens=100, response_tokens=10, model_name="model_a")

        feature = FeatureTimingStats()
        feature.record(1.0, 0.2, prompt_tokens=500, response_tokens=50, model_name="model_a")

        merged = _merge_token_summaries(labeling, feature)

        # Same model used in both phases → aggregated
        assert merged["by_model"]["model_a"]["input"] == 600
        assert merged["by_model"]["model_a"]["output"] == 60
        assert merged["by_model"]["model_a"]["count"] == 2


# =============================================================================
# PARSE LOGGER INTEGRATION
# =============================================================================


class TestParseLoggerTokenIntegration:
    """Tests verifying that Phase 7/8 token data integrates with ParseRunLogger."""

    def test_phase7_token_usage_logged(self, tmp_path):
        from shared.cli.parse_logger import ParseRunLogger

        logger = ParseRunLogger(str(tmp_path), "s", "r", "u")
        token_usage = {
            "by_type": {
                "labeling": {"input": 100, "output": 10, "count": 5},
                "feature_summary": {"input": 500, "output": 50, "count": 10},
            },
            "by_model": {"qwen3:4b": {"input": 600, "output": 60, "count": 15}},
            "totals": {"input": 600, "output": 60, "count": 15},
        }
        logger.log_token_usage("Phase 7: Feature Extraction", token_usage)

        totals = logger.get_token_totals()
        assert totals["input"] == 600
        assert totals["output"] == 60
        assert totals["count"] == 15

    def test_phase8_token_usage_logged(self, tmp_path):
        from shared.cli.parse_logger import ParseRunLogger

        logger = ParseRunLogger(str(tmp_path), "s", "r", "u")
        token_usage = {
            "by_type": {
                "glossary_extract": {"input": 200, "output": 20, "count": 30},
                "glossary_summary": {"input": 400, "output": 80, "count": 20},
            },
            "by_model": {"qwen3:4b": {"input": 600, "output": 100, "count": 50}},
            "totals": {"input": 600, "output": 100, "count": 50},
        }
        logger.log_token_usage("Phase 8: Glossary Extraction", token_usage)

        totals = logger.get_token_totals()
        assert totals["input"] == 600
        assert totals["output"] == 100

    def test_all_phases_aggregate(self, tmp_path):
        """Token totals aggregate across Phase 4, 5, 7, and 8."""
        from shared.cli.parse_logger import ParseRunLogger

        logger = ParseRunLogger(str(tmp_path), "s", "r", "u")

        # Phase 5 tokens (existing pattern)
        logger.log_token_usage("Phase 5: Processing", {
            "by_type": {"function": {"input": 10000, "output": 1000, "count": 100}},
            "by_model": {"qwen3:4b": {"input": 10000, "output": 1000, "count": 100}},
            "totals": {"input": 10000, "output": 1000, "count": 100},
        })

        # Phase 7 tokens
        logger.log_token_usage("Phase 7: Feature Extraction", {
            "by_type": {"feature_summary": {"input": 500, "output": 50, "count": 10}},
            "by_model": {"qwen3:4b": {"input": 500, "output": 50, "count": 10}},
            "totals": {"input": 500, "output": 50, "count": 10},
        })

        totals = logger.get_token_totals()
        assert totals["input"] == 10500
        assert totals["output"] == 1050
        assert totals["count"] == 110

        # Model totals should also aggregate
        model_totals = logger.get_model_totals()
        assert model_totals["qwen3:4b"]["input"] == 10500
        assert model_totals["qwen3:4b"]["output"] == 1050
