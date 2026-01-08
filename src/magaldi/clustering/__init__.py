"""Feature clustering module for Magaldi.

Groups semantically similar code elements into features using HDBSCAN clustering.
"""

from magaldi.clustering.clusterer import (
    ClusterConfig,
    ClusterResult,
    FeatureClusterer,
    LabelingProgressState,
    LabelingTimingStats,
)

__all__ = [
    "ClusterConfig",
    "ClusterResult",
    "FeatureClusterer",
    "LabelingProgressState",
    "LabelingTimingStats",
]
