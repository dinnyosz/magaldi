"""Tests for the clusterer module."""

from __future__ import annotations

import time

import pytest

from shared.ai.clustering.clusterer import (
    ClusterConfig,
    ClusterResult,
    ClusteringResult,
    ClusteringError,
    LabelingTimingStats,
    LabelingProgressState,
)


# =============================================================================
# CLUSTER CONFIG TESTS
# =============================================================================


class TestClusterConfig:
    """Tests for ClusterConfig class."""

    def test_default_values(self):
        """Test default configuration values from ClusteringConfig."""
        from shared.config import ClusteringConfig
        defaults = ClusteringConfig()
        config = ClusterConfig()

        # HDBSCAN params should match ClusteringConfig
        assert config.min_cluster_size == defaults.min_cluster_size
        assert config.min_samples == defaults.min_samples
        assert config.metric == defaults.metric
        assert config.cluster_selection_method == defaults.cluster_selection_method
        assert config.cluster_selection_epsilon == defaults.cluster_selection_epsilon
        assert config.membership_threshold == defaults.membership_threshold
        assert config.affinity_percentile == defaults.affinity_percentile

        # ClusterConfig-specific defaults
        assert config.element_types == ["function", "method"]
        assert config.api_base == "http://localhost:11434"
        assert config.labeling_model == "qwen3:4b-instruct"
        assert config.label_temperature == 0.2
        assert config.label_max_tokens == 32
        assert config.label_timeout == 30

    def test_custom_values(self):
        """Test custom configuration values."""
        config = ClusterConfig(
            min_cluster_size=10,
            min_samples=5,
            element_types=["class"],
            labeling_model="gpt-4",
        )

        assert config.min_cluster_size == 10
        assert config.min_samples == 5
        assert config.element_types == ["class"]
        assert config.labeling_model == "gpt-4"


# =============================================================================
# CLUSTER RESULT TESTS
# =============================================================================


class TestClusterResult:
    """Tests for ClusterResult class."""

    def test_creation(self):
        """Test creating a cluster result."""
        result = ClusterResult(
            cluster_id=0,
            label="authentication",
            element_ids=["id1", "id2", "id3"],
            element_names=["login", "logout", "verify"],
        )

        assert result.cluster_id == 0
        assert result.label == "authentication"
        assert len(result.element_ids) == 3
        assert len(result.element_names) == 3

    def test_size_property(self):
        """Test the size property."""
        result = ClusterResult(
            cluster_id=0,
            label="test",
            element_ids=["id1", "id2", "id3", "id4", "id5"],
            element_names=["a", "b", "c", "d", "e"],
        )

        assert result.size == 5

    def test_size_empty_cluster(self):
        """Test size property for empty cluster."""
        result = ClusterResult(
            cluster_id=0,
            label=None,
            element_ids=[],
            element_names=[],
        )

        assert result.size == 0

    def test_optional_centroid(self):
        """Test optional centroid field."""
        result = ClusterResult(
            cluster_id=0,
            label="test",
            element_ids=["id1"],
            element_names=["func1"],
            centroid=[0.1, 0.2, 0.3],
        )

        assert result.centroid == [0.1, 0.2, 0.3]

    def test_centroid_default_none(self):
        """Test centroid defaults to None."""
        result = ClusterResult(
            cluster_id=0,
            label="test",
            element_ids=["id1"],
            element_names=["func1"],
        )

        assert result.centroid is None

    def test_element_summaries(self):
        """Test element_summaries field."""
        result = ClusterResult(
            cluster_id=0,
            label="test",
            element_ids=["id1", "id2"],
            element_names=["func1", "func2"],
            element_summaries={
                "id1": "Processes user input",
                "id2": "Validates data",
            },
        )

        assert result.element_summaries == {
            "id1": "Processes user input",
            "id2": "Validates data",
        }

    def test_element_summaries_default_empty(self):
        """Test element_summaries defaults to empty dict."""
        result = ClusterResult(
            cluster_id=0,
            label="test",
            element_ids=["id1"],
            element_names=["func1"],
        )

        assert result.element_summaries == {}


# =============================================================================
# CLUSTERING RESULT TESTS
# =============================================================================


class TestClusteringResult:
    """Tests for ClusteringResult class."""

    def test_creation(self):
        """Test creating a clustering result."""
        cluster1 = ClusterResult(
            cluster_id=0, label="auth", element_ids=["id1", "id2"], element_names=["a", "b"]
        )
        cluster2 = ClusterResult(
            cluster_id=1, label="api", element_ids=["id3", "id4"], element_names=["c", "d"]
        )

        result = ClusteringResult(
            clusters=[cluster1, cluster2],
            outlier_count=3,
            outlier_element_ids=["o1", "o2", "o3"],
            total_elements=7,
        )

        assert len(result.clusters) == 2
        assert result.outlier_count == 3
        assert result.total_elements == 7

    def test_cluster_count_property(self):
        """Test the cluster_count property."""
        cluster = ClusterResult(
            cluster_id=0, label="test", element_ids=["id1"], element_names=["a"]
        )

        result = ClusteringResult(
            clusters=[cluster, cluster, cluster],
            outlier_count=0,
            outlier_element_ids=[],
            total_elements=3,
        )

        assert result.cluster_count == 3

    def test_cluster_count_empty(self):
        """Test cluster_count with no clusters."""
        result = ClusteringResult(
            clusters=[],
            outlier_count=5,
            outlier_element_ids=["o1", "o2", "o3", "o4", "o5"],
            total_elements=5,
        )

        assert result.cluster_count == 0


# =============================================================================
# LABELING TIMING STATS TESTS
# =============================================================================


class TestLabelingTimingStats:
    """Tests for LabelingTimingStats class."""

    def test_default_values(self):
        """Test default timing values."""
        stats = LabelingTimingStats()

        assert stats.phase_start == 0.0
        assert stats.total_label_time == 0.0
        assert stats.label_count == 0

    def test_record_timing(self):
        """Test recording timing for labels."""
        stats = LabelingTimingStats()

        stats.record(0.5)
        stats.record(0.7)

        assert stats.total_label_time == 1.2
        assert stats.label_count == 2

    def test_avg_label_time(self):
        """Test average label time calculation."""
        stats = LabelingTimingStats()

        stats.record(0.4)
        stats.record(0.6)

        assert abs(stats.avg_label_time - 0.5) < 0.001

    def test_avg_label_time_empty(self):
        """Test average label time with no data."""
        stats = LabelingTimingStats()

        assert stats.avg_label_time == 0.0

    def test_elapsed_property(self):
        """Test elapsed time property."""
        stats = LabelingTimingStats()
        stats.phase_start = time.time() - 5.0  # Started 5 seconds ago

        elapsed = stats.elapsed

        assert elapsed >= 5.0

    def test_eta_seconds_no_data(self):
        """Test ETA with no data."""
        stats = LabelingTimingStats()

        assert stats.eta_seconds(0, 10) is None

    def test_eta_seconds_completed(self):
        """Test ETA when all completed."""
        stats = LabelingTimingStats()
        stats.record(1.0)

        assert stats.eta_seconds(10, 10) is None

    def test_eta_seconds_with_data(self):
        """Test ETA calculation."""
        stats = LabelingTimingStats()

        stats.record(1.0)
        stats.record(1.0)

        # 2 completed, 8 remaining, avg=1.0s per label
        eta = stats.eta_seconds(2, 10)
        assert eta is not None
        assert abs(eta - 8.0) < 0.1


# =============================================================================
# LABELING PROGRESS STATE TESTS
# =============================================================================


class TestLabelingProgressState:
    """Tests for LabelingProgressState class."""

    def test_creation(self):
        """Test creating progress state."""
        timing = LabelingTimingStats()

        state = LabelingProgressState(
            total=10,
            completed=5,
            skipped=2,
            failed=1,
            timing=timing,
            current_cluster="auth_feature",
            model="gpt-4",
        )

        assert state.total == 10
        assert state.completed == 5
        assert state.skipped == 2
        assert state.failed == 1
        assert state.current_cluster == "auth_feature"
        assert state.model == "gpt-4"


# =============================================================================
# CLUSTERING ERROR TESTS
# =============================================================================


class TestClusteringError:
    """Tests for ClusteringError exception."""

    def test_raises_with_message(self):
        """Test raising ClusteringError with message."""
        with pytest.raises(ClusteringError, match="Test error"):
            raise ClusteringError("Test error")

    def test_inherits_from_exception(self):
        """Test ClusteringError is an Exception."""
        error = ClusteringError("test")
        assert isinstance(error, Exception)


