"""Tests for the soft clustering module."""

from __future__ import annotations

import numpy as np
import pytest

from shared.ai.clustering.soft_clustering import (
    FeatureAffinity,
    FeatureMembership,
    SoftClusteringConfig,
    SoftClusteringPipeline,
    SoftClusteringProgress,
    SoftClusteringResult,
)


# =============================================================================
# CONFIG TESTS
# =============================================================================


class TestSoftClusteringProgress:
    """Tests for SoftClusteringProgress class."""

    def test_creation(self):
        """Test creating a progress state."""
        progress = SoftClusteringProgress(
            phase="hdbscan",
            phase_description="HDBSCAN soft clustering",
            current_step=1,
            total_steps=1,
            n_elements=1000,
            n_features=100,
        )

        assert progress.phase == "hdbscan"
        assert progress.current_step == 1
        assert progress.total_steps == 1

    def test_percent_property(self):
        """Test percent calculation."""
        progress = SoftClusteringProgress(
            phase="hdbscan",
            phase_description="test",
            current_step=25,
            total_steps=100,
        )
        assert progress.percent == 25.0

    def test_percent_zero_total(self):
        """Test percent with zero total."""
        progress = SoftClusteringProgress(
            phase="hdbscan",
            phase_description="test",
            current_step=0,
            total_steps=0,
        )
        assert progress.percent == 0.0

    def test_status_line(self):
        """Test status line formatting."""
        progress = SoftClusteringProgress(
            phase="hdbscan",
            phase_description="HDBSCAN soft clustering",
            current_step=1,
            total_steps=1,
        )
        assert progress.status_line == "HDBSCAN soft clustering [1/1]"


class TestSoftClusteringConfig:
    """Tests for SoftClusteringConfig class."""

    def test_default_values(self):
        """Test default configuration values."""
        config = SoftClusteringConfig()

        assert config.min_cluster_size == 5
        assert config.min_samples == 2
        assert config.metric == "euclidean"
        assert config.membership_threshold == 0.01
        assert config.affinity_threshold == 0.001

    def test_custom_values(self):
        """Test custom configuration values."""
        config = SoftClusteringConfig(
            min_cluster_size=10,
            min_samples=3,
            metric="manhattan",
            membership_threshold=0.05,
        )

        assert config.min_cluster_size == 10
        assert config.min_samples == 3
        assert config.metric == "manhattan"
        assert config.membership_threshold == 0.05


# =============================================================================
# DATA CLASS TESTS
# =============================================================================


class TestFeatureMembership:
    """Tests for FeatureMembership class."""

    def test_creation(self):
        """Test creating a feature membership."""
        membership = FeatureMembership(
            feature_idx=5,
            score=0.45,
            is_primary=True,
        )

        assert membership.feature_idx == 5
        assert membership.score == 0.45
        assert membership.is_primary is True

    def test_default_is_primary(self):
        """Test is_primary defaults to False."""
        membership = FeatureMembership(feature_idx=0, score=0.1)
        assert membership.is_primary is False


class TestFeatureAffinity:
    """Tests for FeatureAffinity class."""

    def test_creation(self):
        """Test creating a feature affinity."""
        affinity = FeatureAffinity(feature_idx=3, affinity=0.72)

        assert affinity.feature_idx == 3
        assert affinity.affinity == 0.72


class TestSoftClusteringResult:
    """Tests for SoftClusteringResult class."""

    def test_creation(self):
        """Test creating a result."""
        memberships = {
            "elem_0": [FeatureMembership(0, 0.5, True), FeatureMembership(1, 0.3, False)],
            "elem_1": [FeatureMembership(1, 0.6, True)],
        }
        affinity = np.eye(5)

        result = SoftClusteringResult(
            element_memberships=memberships,
            feature_affinity=affinity,
            n_features_found=2,
            n_elements=100,
            cooccurrence_density=0.0,  # Kept for API compatibility
        )

        assert len(result.element_memberships) == 2
        assert result.n_features_found == 2
        assert result.n_elements == 100
        assert result.cooccurrence_density == 0.0


# =============================================================================
# PIPELINE TESTS
# =============================================================================


class TestSoftClusteringPipeline:
    """Tests for SoftClusteringPipeline class."""

    @pytest.fixture
    def small_config(self):
        """Configuration for small test datasets."""
        return SoftClusteringConfig(
            min_cluster_size=3,
            min_samples=2,
            membership_threshold=0.01,
        )

    @pytest.fixture
    def clustered_embeddings(self):
        """Create synthetic embeddings with clear clusters."""
        np.random.seed(42)

        # Create 3 distinct clusters
        cluster1 = np.random.randn(20, 50) + np.array([5, 0, 0] + [0] * 47)
        cluster2 = np.random.randn(20, 50) + np.array([0, 5, 0] + [0] * 47)
        cluster3 = np.random.randn(20, 50) + np.array([0, 0, 5] + [0] * 47)

        # Add some overlap points between clusters
        overlap = np.random.randn(10, 50) + np.array([2.5, 2.5, 0] + [0] * 47)

        embeddings = np.vstack([cluster1, cluster2, cluster3, overlap])
        return embeddings.astype(np.float32)

    def test_init_default_config(self):
        """Test initialization with default config."""
        pipeline = SoftClusteringPipeline()
        assert pipeline.config is not None
        assert pipeline.config.min_cluster_size == 5

    def test_init_custom_config(self, small_config):
        """Test initialization with custom config."""
        pipeline = SoftClusteringPipeline(small_config)
        assert pipeline.config.min_cluster_size == 3

    def test_fit_empty_embeddings(self, small_config):
        """Test fit with empty embeddings raises error."""
        pipeline = SoftClusteringPipeline(small_config)

        with pytest.raises(ValueError, match="Empty embeddings"):
            pipeline.fit(np.array([]))

    def test_fit_wrong_dimensions(self, small_config):
        """Test fit with wrong dimensions raises error."""
        pipeline = SoftClusteringPipeline(small_config)

        with pytest.raises(ValueError, match="Expected 2D"):
            pipeline.fit(np.array([1, 2, 3]))

    def test_fit_basic(self, small_config, clustered_embeddings):
        """Test basic fit operation."""
        pipeline = SoftClusteringPipeline(small_config)
        result = pipeline.fit(clustered_embeddings)

        assert isinstance(result, SoftClusteringResult)
        assert result.n_elements == len(clustered_embeddings)
        assert result.feature_affinity.shape[0] == result.feature_affinity.shape[1]
        assert len(result.element_memberships) > 0

    def test_fit_with_element_ids(self, small_config, clustered_embeddings):
        """Test fit with custom element IDs."""
        pipeline = SoftClusteringPipeline(small_config)
        element_ids = [f"element_{i}" for i in range(len(clustered_embeddings))]

        result = pipeline.fit(clustered_embeddings, element_ids)

        # Check that element IDs are used as keys
        for key in result.element_memberships.keys():
            assert isinstance(key, str)
            assert key.startswith("element_")

    def test_fit_without_element_ids(self, small_config, clustered_embeddings):
        """Test fit without element IDs uses integer indices."""
        pipeline = SoftClusteringPipeline(small_config)
        result = pipeline.fit(clustered_embeddings)

        # Check that integer indices are used as keys
        for key in result.element_memberships.keys():
            assert isinstance(key, int)

    def test_memberships_have_primary(self, small_config, clustered_embeddings):
        """Test that each element has exactly one primary membership."""
        pipeline = SoftClusteringPipeline(small_config)
        result = pipeline.fit(clustered_embeddings)

        for memberships in result.element_memberships.values():
            primary_count = sum(1 for m in memberships if m.is_primary)
            assert primary_count == 1, "Each element should have exactly one primary"

    def test_memberships_sorted_by_score(self, small_config, clustered_embeddings):
        """Test that memberships are sorted by score descending."""
        pipeline = SoftClusteringPipeline(small_config)
        result = pipeline.fit(clustered_embeddings)

        for memberships in result.element_memberships.values():
            scores = [m.score for m in memberships]
            assert scores == sorted(scores, reverse=True), "Memberships should be sorted by score"

    def test_primary_is_highest_score(self, small_config, clustered_embeddings):
        """Test that primary membership has the highest score."""
        pipeline = SoftClusteringPipeline(small_config)
        result = pipeline.fit(clustered_embeddings)

        for memberships in result.element_memberships.values():
            if memberships:
                assert memberships[0].is_primary, "First (highest) should be primary"
                assert memberships[0].score >= max(m.score for m in memberships)

    def test_feature_affinity_symmetric(self, small_config, clustered_embeddings):
        """Test that feature affinity matrix is symmetric."""
        pipeline = SoftClusteringPipeline(small_config)
        result = pipeline.fit(clustered_embeddings)

        np.testing.assert_array_almost_equal(
            result.feature_affinity,
            result.feature_affinity.T,
            decimal=5,
            err_msg="Feature affinity should be symmetric",
        )

    def test_feature_affinity_diagonal_highest(self, small_config, clustered_embeddings):
        """Test that diagonal entries in affinity are >= off-diagonal."""
        pipeline = SoftClusteringPipeline(small_config)
        result = pipeline.fit(clustered_embeddings)

        for i in range(result.feature_affinity.shape[0]):
            diagonal = result.feature_affinity[i, i]
            row_max = result.feature_affinity[i].max()
            assert diagonal >= row_max - 1e-6, "Diagonal should be >= max in row"

    def test_progress_callback(self, small_config, clustered_embeddings):
        """Test that progress callback is invoked."""
        progress_states = []

        def on_progress(progress: SoftClusteringProgress):
            progress_states.append(progress)

        pipeline = SoftClusteringPipeline(small_config)
        result = pipeline.fit(clustered_embeddings, on_progress=on_progress)

        # Should have received progress callbacks
        assert len(progress_states) > 0

        # Should have HDBSCAN phase callbacks
        hdbscan_phases = [p for p in progress_states if p.phase == "hdbscan"]
        assert len(hdbscan_phases) >= 1

        # Should have complete callback
        complete_phases = [p for p in progress_states if p.phase == "complete"]
        assert len(complete_phases) == 1


class TestSoftClusteringPipelineQueries:
    """Tests for query methods."""

    @pytest.fixture
    def fitted_result(self):
        """Create a pre-fitted result for query tests."""
        np.random.seed(42)

        config = SoftClusteringConfig(
            min_cluster_size=3,
            min_samples=2,
            membership_threshold=0.01,
        )

        # Create embeddings with clear structure
        embeddings = np.random.randn(50, 30).astype(np.float32)
        element_ids = [f"elem_{i}" for i in range(50)]

        pipeline = SoftClusteringPipeline(config)
        return pipeline, pipeline.fit(embeddings, element_ids)

    def test_get_element_memberships_exists(self, fitted_result):
        """Test getting memberships for existing element."""
        pipeline, result = fitted_result

        # Find an element that has memberships
        if result.element_memberships:
            elem_id = list(result.element_memberships.keys())[0]
            memberships = pipeline.get_element_memberships(result, elem_id)
            assert len(memberships) > 0
            assert all(isinstance(m, FeatureMembership) for m in memberships)

    def test_get_element_memberships_not_exists(self, fitted_result):
        """Test getting memberships for non-existent element."""
        pipeline, result = fitted_result
        memberships = pipeline.get_element_memberships(result, "nonexistent")
        assert memberships == []

    def test_get_feature_members(self, fitted_result):
        """Test getting members of a feature."""
        pipeline, result = fitted_result

        # Find a feature that has members
        feature_idx = 0
        members = pipeline.get_feature_members(result, feature_idx)

        # Members should be tuples of (element_id, score, is_primary)
        for member in members:
            assert len(member) == 3
            elem_id, score, is_primary = member
            assert isinstance(score, float)
            assert isinstance(is_primary, bool)

    def test_get_feature_members_sorted(self, fitted_result):
        """Test that feature members are sorted by score."""
        pipeline, result = fitted_result

        members = pipeline.get_feature_members(result, 0)
        if len(members) > 1:
            scores = [m[1] for m in members]
            assert scores == sorted(scores, reverse=True)

    def test_get_feature_members_min_score(self, fitted_result):
        """Test min_score filter in get_feature_members."""
        pipeline, result = fitted_result

        all_members = pipeline.get_feature_members(result, 0, min_score=0.0)
        high_score_members = pipeline.get_feature_members(result, 0, min_score=0.5)

        assert len(high_score_members) <= len(all_members)
        for _, score, _ in high_score_members:
            assert score >= 0.5

    def test_get_connected_features(self, fitted_result):
        """Test getting connected features."""
        pipeline, result = fitted_result

        if result.feature_affinity.shape[0] > 0:
            connected = pipeline.get_connected_features(result, 0)

            # Should return FeatureAffinity objects
            for conn in connected:
                assert isinstance(conn, FeatureAffinity)
                assert conn.feature_idx != 0  # Should not include self

    def test_get_connected_features_sorted(self, fitted_result):
        """Test that connected features are sorted by affinity."""
        pipeline, result = fitted_result

        if result.feature_affinity.shape[0] > 0:
            connected = pipeline.get_connected_features(result, 0)
            if len(connected) > 1:
                affinities = [c.affinity for c in connected]
                assert affinities == sorted(affinities, reverse=True)

    def test_get_connected_features_min_affinity(self, fitted_result):
        """Test min_affinity filter in get_connected_features."""
        pipeline, result = fitted_result

        if result.feature_affinity.shape[0] > 0:
            all_connected = pipeline.get_connected_features(result, 0, min_affinity=0.0)
            high_affinity = pipeline.get_connected_features(result, 0, min_affinity=0.5)

            assert len(high_affinity) <= len(all_connected)
            for conn in high_affinity:
                assert conn.affinity >= 0.5


# =============================================================================
# EDGE CASE TESTS
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_small_dataset(self):
        """Test with small dataset."""
        config = SoftClusteringConfig(
            min_cluster_size=3,
            min_samples=2,
        )

        np.random.seed(42)
        embeddings = np.random.randn(15, 20).astype(np.float32)
        pipeline = SoftClusteringPipeline(config)

        # Should handle gracefully
        result = pipeline.fit(embeddings)
        assert result.n_elements == 15

    def test_single_element(self):
        """Test with single element (should handle gracefully)."""
        np.random.seed(42)

        config = SoftClusteringConfig(
            min_cluster_size=2,  # Won't form clusters with 1 element
        )

        embeddings = np.random.randn(1, 10).astype(np.float32)
        pipeline = SoftClusteringPipeline(config)

        # Should not crash - handles gracefully with uniform memberships
        result = pipeline.fit(embeddings)
        assert result.n_elements == 1
        assert result.feature_affinity is not None

    def test_two_elements(self):
        """Test with two elements."""
        config = SoftClusteringConfig(
            min_cluster_size=2,
        )

        np.random.seed(42)
        embeddings = np.random.randn(2, 10).astype(np.float32)
        pipeline = SoftClusteringPipeline(config)

        result = pipeline.fit(embeddings)
        assert result.n_elements == 2

    def test_high_threshold(self):
        """Test with high membership threshold (may result in few/no memberships)."""
        config = SoftClusteringConfig(
            min_cluster_size=3,
            membership_threshold=0.99,  # Very high
        )

        np.random.seed(42)
        embeddings = np.random.randn(30, 20).astype(np.float32)
        pipeline = SoftClusteringPipeline(config)

        result = pipeline.fit(embeddings)
        # High threshold should result in fewer memberships
        # (but shouldn't crash)
        assert result.n_elements == 30

    def test_identical_embeddings(self):
        """Test with all identical embeddings."""
        config = SoftClusteringConfig(
            min_cluster_size=3,
        )

        # All same embedding
        embeddings = np.ones((20, 10), dtype=np.float32)
        pipeline = SoftClusteringPipeline(config)

        # Should handle (all points should cluster together)
        result = pipeline.fit(embeddings)
        assert result.n_elements == 20

    def test_different_metrics(self):
        """Test with different distance metrics."""
        config = SoftClusteringConfig(
            min_cluster_size=3,
            metric="manhattan",  # L1 distance
        )

        # Create embeddings with some structure
        np.random.seed(42)
        embeddings = np.random.randn(30, 100).astype(np.float32)
        pipeline = SoftClusteringPipeline(config)

        result = pipeline.fit(embeddings)
        assert result.n_elements == 30

    def test_high_dimensional_embeddings(self):
        """Test with high-dimensional embeddings (like real sentence transformers)."""
        config = SoftClusteringConfig(
            min_cluster_size=3,
            min_samples=2,
        )

        # Simulate 1024-dim embeddings
        np.random.seed(42)
        embeddings = np.random.randn(50, 1024).astype(np.float32)
        pipeline = SoftClusteringPipeline(config)

        result = pipeline.fit(embeddings)
        assert result.n_elements == 50
        assert result.feature_affinity is not None

    def test_no_clusters_found(self):
        """Test handling when HDBSCAN finds no clusters."""
        config = SoftClusteringConfig(
            min_cluster_size=50,  # Very high - won't form clusters
            min_samples=10,
        )

        np.random.seed(42)
        embeddings = np.random.randn(20, 20).astype(np.float32)
        pipeline = SoftClusteringPipeline(config)

        # Should handle gracefully with uniform memberships
        result = pipeline.fit(embeddings)
        assert result.n_elements == 20
        assert result.feature_affinity is not None
