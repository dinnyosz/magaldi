"""Tests for the clustering module."""

from __future__ import annotations

import numpy as np
import pytest

from shared.ai.clustering.clusterer import (
    ClusterConfig,
    ClusterResult,
    ClusteringError,
    ClusteringResult,
    FeatureClusterer,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def sample_embeddings() -> list[dict]:
    """Create sample elements with embeddings for testing.

    Creates 3 distinct clusters:
    - Cluster 1: 10 elements around [1, 0, 0, ...]
    - Cluster 2: 10 elements around [0, 1, 0, ...]
    - Cluster 3: 10 elements around [0, 0, 1, ...]
    - 5 outliers scattered around
    """
    np.random.seed(42)
    dim = 1024
    elements = []

    # Cluster 1: auth-related functions
    center1 = np.zeros(dim)
    center1[0] = 1.0
    for i in range(10):
        embedding = center1 + np.random.normal(0, 0.1, dim)
        elements.append({
            "element_id": f"scope:repo:user:auth/login.py:function:auth_func_{i}:10",
            "embedding": embedding.tolist(),
            "name": f"auth_func_{i}",
            "element_type": "function",
            "relative_path": "auth/login.py",
        })

    # Cluster 2: database-related functions
    center2 = np.zeros(dim)
    center2[1] = 1.0
    for i in range(10):
        embedding = center2 + np.random.normal(0, 0.1, dim)
        elements.append({
            "element_id": f"scope:repo:user:db/queries.py:function:db_func_{i}:20",
            "embedding": embedding.tolist(),
            "name": f"db_func_{i}",
            "element_type": "function",
            "relative_path": "db/queries.py",
        })

    # Cluster 3: API-related functions
    center3 = np.zeros(dim)
    center3[2] = 1.0
    for i in range(10):
        embedding = center3 + np.random.normal(0, 0.1, dim)
        elements.append({
            "element_id": f"scope:repo:user:api/handlers.py:method:api_method_{i}:30",
            "embedding": embedding.tolist(),
            "name": f"api_method_{i}",
            "element_type": "method",
            "relative_path": "api/handlers.py",
        })

    # Outliers: random embeddings far from clusters
    for i in range(5):
        embedding = np.random.normal(0, 1, dim)
        elements.append({
            "element_id": f"scope:repo:user:utils/misc.py:function:util_func_{i}:40",
            "embedding": embedding.tolist(),
            "name": f"util_func_{i}",
            "element_type": "function",
            "relative_path": "utils/misc.py",
        })

    return elements


@pytest.fixture
def cluster_config() -> ClusterConfig:
    """Create a test cluster config."""
    return ClusterConfig(
        min_cluster_size=5,
        min_samples=3,
        element_types=["function", "method"],
    )


# =============================================================================
# CLUSTER CONFIG TESTS
# =============================================================================


class TestClusterConfig:
    """Tests for ClusterConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ClusterConfig()
        assert config.min_cluster_size == 3
        assert config.min_samples == 1
        assert config.element_types == ["function", "method"]
        assert config.api_base == "http://localhost:11434"

    def test_custom_values(self):
        """Test custom configuration values."""
        config = ClusterConfig(
            min_cluster_size=10,
            min_samples=5,
            element_types=["function"],
        )
        assert config.min_cluster_size == 10
        assert config.min_samples == 5
        assert config.element_types == ["function"]


# =============================================================================
# CLUSTER RESULT TESTS
# =============================================================================


class TestClusterResult:
    """Tests for ClusterResult dataclass."""

    def test_size_property(self):
        """Test that size property returns element count."""
        result = ClusterResult(
            cluster_id=0,
            label="test_cluster",
            element_ids=["id1", "id2", "id3"],
            element_names=["func1", "func2", "func3"],
        )
        assert result.size == 3

    def test_empty_cluster(self):
        """Test cluster with no elements."""
        result = ClusterResult(
            cluster_id=0,
            label=None,
            element_ids=[],
            element_names=[],
        )
        assert result.size == 0


# =============================================================================
# FEATURE CLUSTERER TESTS
# =============================================================================


class TestFeatureClusterer:
    """Tests for FeatureClusterer class."""

    def test_init_with_default_config(self):
        """Test initialization with default config."""
        clusterer = FeatureClusterer()
        assert clusterer.config is not None
        assert clusterer.config.min_cluster_size == 3

    def test_init_with_custom_config(self, cluster_config):
        """Test initialization with custom config."""
        clusterer = FeatureClusterer(cluster_config)
        assert clusterer.config == cluster_config

    def test_cluster_empty_elements(self, cluster_config):
        """Test clustering with empty element list."""
        clusterer = FeatureClusterer(cluster_config)
        with pytest.raises(ClusteringError, match="No elements provided"):
            clusterer.cluster([])

    def test_cluster_no_embeddings(self, cluster_config):
        """Test clustering with elements that have no embeddings."""
        clusterer = FeatureClusterer(cluster_config)
        elements = [
            {"element_id": "id1", "name": "func1", "embedding": None},
            {"element_id": "id2", "name": "func2"},  # Missing embedding key
        ]
        with pytest.raises(ClusteringError, match="No embeddings found"):
            clusterer.cluster(elements)

    def test_cluster_finds_clusters(self, cluster_config, sample_embeddings):
        """Test that clustering finds expected number of clusters."""
        clusterer = FeatureClusterer(cluster_config)
        result = clusterer.cluster(sample_embeddings)

        # Should find approximately 3 clusters
        assert result.cluster_count >= 2
        assert result.cluster_count <= 4

        # Should have some outliers
        assert result.outlier_count >= 0

        # Total should match input
        total_in_clusters = sum(c.size for c in result.clusters)
        assert total_in_clusters + result.outlier_count == result.total_elements

    def test_cluster_elements_assigned_correctly(self, cluster_config, sample_embeddings):
        """Test that element IDs are assigned to clusters."""
        clusterer = FeatureClusterer(cluster_config)
        result = clusterer.cluster(sample_embeddings)

        # All clusters should have element_ids
        for cluster in result.clusters:
            assert len(cluster.element_ids) > 0
            assert len(cluster.element_names) == len(cluster.element_ids)

    def test_cluster_centroids_computed(self, cluster_config, sample_embeddings):
        """Test that centroids are computed for clusters."""
        clusterer = FeatureClusterer(cluster_config)
        result = clusterer.cluster(sample_embeddings)

        for cluster in result.clusters:
            assert cluster.centroid is not None
            assert len(cluster.centroid) == 1024  # Embedding dimension

    def test_cluster_returns_clustering_result(self, cluster_config, sample_embeddings):
        """Test that cluster returns a ClusteringResult."""
        clusterer = FeatureClusterer(cluster_config)
        result = clusterer.cluster(sample_embeddings)

        assert isinstance(result, ClusteringResult)
        assert isinstance(result.clusters, list)
        assert all(isinstance(c, ClusterResult) for c in result.clusters)


class TestClusterLabeling:
    """Tests for cluster labeling functionality."""

    def test_clean_label(self, cluster_config):
        """Test label cleaning."""
        clusterer = FeatureClusterer(cluster_config)

        # Test basic cleaning
        assert clusterer._clean_label("  Authentication  ") == "authentication"
        assert clusterer._clean_label('"database"') == "database"
        assert clusterer._clean_label("'api handlers'") == "api_handlers"

        # Test prefix removal
        assert clusterer._clean_label("Label: auth") == "auth"
        assert clusterer._clean_label("The label is database") == "database"

        # Test special char removal
        assert clusterer._clean_label("auth@123") == "auth123"

        # Test length limit
        long_label = "a" * 100
        assert len(clusterer._clean_label(long_label)) <= 50


# =============================================================================
# CLUSTERING RESULT TESTS
# =============================================================================


class TestClusteringResult:
    """Tests for ClusteringResult dataclass."""

    def test_cluster_count_property(self):
        """Test cluster_count property."""
        result = ClusteringResult(
            clusters=[
                ClusterResult(0, "auth", ["id1", "id2"], ["f1", "f2"]),
                ClusterResult(1, "db", ["id3", "id4"], ["f3", "f4"]),
            ],
            outlier_count=5,
            outlier_element_ids=["out1", "out2", "out3", "out4", "out5"],
            total_elements=9,
        )

        assert result.cluster_count == 2
        assert result.outlier_count == 5
        assert result.total_elements == 9


# =============================================================================
# INTEGRATION TESTS (with mock Ollama)
# =============================================================================


class TestClusteringIntegration:
    """Integration tests for the full clustering pipeline."""

    def test_full_clustering_pipeline(self, cluster_config, sample_embeddings):
        """Test complete clustering without labeling."""
        clusterer = FeatureClusterer(cluster_config)

        # Run clustering
        result = clusterer.cluster(sample_embeddings)

        # Verify basic structure
        assert result.cluster_count > 0
        assert result.total_elements == len(sample_embeddings)

        # Verify clusters have expected structure
        for cluster in result.clusters:
            assert cluster.cluster_id >= 0
            assert cluster.label is None  # No labeling yet
            assert len(cluster.element_ids) >= cluster_config.min_cluster_size
