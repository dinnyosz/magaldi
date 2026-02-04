"""Soft clustering for code element embeddings using HDBSCAN.

Implements overlapping/soft cluster memberships by running HDBSCAN
directly on embedding vectors and using all_points_membership_vectors()
to extract soft memberships.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import hdbscan
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# PROGRESS TRACKING
# =============================================================================


@dataclass
class SoftClusteringProgress:
    """Progress state for soft clustering pipeline.

    Used for CLI feedback during clustering operations.
    """

    phase: str  # "hdbscan", "complete"
    phase_description: str  # Human-readable description
    current_step: int
    total_steps: int
    # Optional details
    n_elements: int = 0
    n_features: int = 0
    cooccurrence_density: float = 0.0  # Kept for API compatibility
    # Timing for ETA
    elapsed_seconds: float = 0.0
    eta_seconds: float | None = None

    @property
    def percent(self) -> float:
        """Completion percentage for current phase."""
        if self.total_steps == 0:
            return 0.0
        return (self.current_step / self.total_steps) * 100

    @property
    def status_line(self) -> str:
        """Formatted status line for CLI display."""
        return f"{self.phase_description} [{self.current_step}/{self.total_steps}]"


# Progress callback type
ProgressCallback = Callable[[SoftClusteringProgress], None]


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class SoftClusteringConfig:
    """Configuration for soft clustering using HDBSCAN directly on embeddings.

    Runs HDBSCAN directly on the embedding space without dimensionality reduction.
    Modern embedding models produce well-structured vectors that don't need
    preprocessing like PCA or UMAP.

    Attributes:
        min_cluster_size: HDBSCAN minimum cluster size.
        min_samples: HDBSCAN min_samples (density threshold).
        metric: Distance metric for HDBSCAN. 'euclidean' works well for embeddings.
        membership_threshold: Minimum soft membership to retain.
        affinity_threshold: Minimum affinity between features to report.
    """

    min_cluster_size: int = 5
    min_samples: int = 2
    metric: str = "euclidean"  # euclidean works well for normalized embeddings
    membership_threshold: float = 0.01
    affinity_threshold: float = 0.001


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class FeatureMembership:
    """Soft membership of an element in a feature.

    Attributes:
        feature_idx: Index of the feature (cluster).
        score: Membership probability (0-1, rows sum to ~1).
        is_primary: Whether this is the element's primary (highest score) feature.
    """

    feature_idx: int
    score: float
    is_primary: bool = False


@dataclass
class FeatureAffinity:
    """Connection strength between two features.

    Attributes:
        feature_idx: Index of the connected feature.
        affinity: Connection strength (higher = more shared elements).
    """

    feature_idx: int
    affinity: float


@dataclass
class SoftClusteringResult:
    """Results from soft clustering pipeline.

    Attributes:
        element_memberships: Per-element soft memberships.
        feature_affinity: Feature-to-feature affinity matrix.
        n_features_found: Number of features with at least one member.
        n_elements: Total elements processed.
        cooccurrence_density: Kept for API compatibility (always 0.0).
    """

    element_memberships: dict[str | int, list[FeatureMembership]]
    feature_affinity: np.ndarray
    n_features_found: int
    n_elements: int
    cooccurrence_density: float = 0.0


# =============================================================================
# SOFT CLUSTERING PIPELINE
# =============================================================================


class SoftClusteringPipeline:
    """Soft clustering using HDBSCAN directly on embeddings.

    Runs HDBSCAN with soft clustering directly on the embedding vectors.
    Modern embedding models produce well-structured vectors that work well
    without dimensionality reduction preprocessing.

    Algorithm:
    1. HDBSCAN with prediction_data=True enables soft clustering
    2. all_points_membership_vectors() extracts soft memberships directly
    3. Feature affinity computed from membership matrix (W^T @ W)

    Example:
        >>> pipeline = SoftClusteringPipeline()
        >>> result = pipeline.fit(embeddings, element_ids)
        >>> memberships = result.element_memberships["element_42"]
        >>> for m in memberships:
        ...     print(f"Feature {m.feature_idx}: {m.score:.2%}")
    """

    def __init__(self, config: SoftClusteringConfig | None = None):
        """Initialize pipeline with configuration.

        Args:
            config: Clustering configuration. Uses defaults if not provided.
        """
        self.config = config or SoftClusteringConfig()
        self._clusterer: hdbscan.HDBSCAN | None = None
        self._memberships: np.ndarray | None = None
        self._feature_affinity: np.ndarray | None = None

    def fit(
        self,
        embeddings: np.ndarray,
        element_ids: list[str] | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> SoftClusteringResult:
        """Fit the pipeline using HDBSCAN soft clustering directly on embeddings.

        Args:
            embeddings: (n_elements, embedding_dim) embedding matrix.
            element_ids: Optional list of element IDs for result mapping.
                If not provided, integer indices are used as keys.
            on_progress: Optional callback for progress updates.

        Returns:
            SoftClusteringResult with element memberships and feature affinity.

        Raises:
            ValueError: If embeddings is empty or has wrong shape.
        """
        import time

        if embeddings.size == 0:
            raise ValueError("Empty embeddings array")

        if embeddings.ndim != 2:
            raise ValueError(f"Expected 2D embeddings, got {embeddings.ndim}D")

        n_elements, embedding_dim = embeddings.shape
        logger.info(
            f"Starting soft clustering: {n_elements} elements, "
            f"{embedding_dim}D embeddings"
        )

        # HDBSCAN soft clustering directly on embeddings
        hdbscan_start = time.time()
        if on_progress:
            on_progress(SoftClusteringProgress(
                phase="hdbscan",
                phase_description="HDBSCAN soft clustering",
                current_step=0,
                total_steps=1,
                n_elements=n_elements,
            ))

        self._memberships, n_clusters = self._apply_hdbscan_soft(embeddings)

        if on_progress:
            elapsed = time.time() - hdbscan_start
            on_progress(SoftClusteringProgress(
                phase="hdbscan",
                phase_description="HDBSCAN soft clustering",
                current_step=1,
                total_steps=1,
                n_elements=n_elements,
                n_features=n_clusters,
                elapsed_seconds=elapsed,
            ))

        logger.info(f"HDBSCAN: found {n_clusters} clusters")

        # Compute feature affinity from soft memberships
        self._feature_affinity = self._compute_feature_affinity()

        # Build element membership mapping
        element_memberships = self._build_element_memberships(element_ids)

        # Count features with members
        features_with_members = set()
        for memberships in element_memberships.values():
            for m in memberships:
                features_with_members.add(m.feature_idx)

        logger.info(
            f"Soft clustering complete: {len(element_memberships)} elements "
            f"assigned to {len(features_with_members)} features"
        )

        # Report completion
        if on_progress:
            on_progress(SoftClusteringProgress(
                phase="complete",
                phase_description="Soft clustering complete",
                current_step=1,
                total_steps=1,
                n_elements=n_elements,
                n_features=len(features_with_members),
            ))

        return SoftClusteringResult(
            element_memberships=element_memberships,
            feature_affinity=self._feature_affinity,
            n_features_found=len(features_with_members),
            n_elements=n_elements,
        )

    def _apply_hdbscan_soft(
        self,
        embeddings: np.ndarray,
    ) -> tuple[np.ndarray, int]:
        """Apply HDBSCAN with soft clustering support.

        Uses prediction_data=True to enable soft clustering via
        all_points_membership_vectors().

        Args:
            embeddings: Embedding vectors (n_samples, embedding_dim).

        Returns:
            Tuple of (soft_membership_matrix, n_clusters).
        """
        n_samples = embeddings.shape[0]

        # Handle edge case: too few samples for HDBSCAN
        if n_samples < 2:
            logger.warning(f"Too few samples ({n_samples}) for HDBSCAN. Returning uniform memberships.")
            return np.ones((n_samples, 1), dtype=np.float32), 1

        # Adjust min_cluster_size if dataset is small
        min_cluster_size = min(self.config.min_cluster_size, max(2, n_samples // 10))
        min_samples = min(self.config.min_samples, min_cluster_size, n_samples)

        self._clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric=self.config.metric,
            prediction_data=True,  # Required for soft clustering
        )

        # Fit the clusterer
        self._clusterer.fit(embeddings)

        # Get hard labels to count clusters
        labels = self._clusterer.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        if n_clusters == 0:
            logger.warning("HDBSCAN found no clusters. Returning uniform memberships.")
            # Return uniform memberships to a single "cluster"
            return np.ones((n_samples, 1), dtype=np.float32), 1

        # Extract soft memberships using all_points_membership_vectors
        # This gives probability of each point belonging to each cluster
        # Shape: (n_samples, n_clusters)
        soft_memberships = hdbscan.all_points_membership_vectors(self._clusterer)

        # Handle edge case where soft memberships might be empty
        if soft_memberships.size == 0:
            logger.warning("Empty soft memberships from HDBSCAN. Using hard labels.")
            # Fall back to hard labels as one-hot memberships
            soft_memberships = np.zeros((n_samples, n_clusters), dtype=np.float32)
            for i, label in enumerate(labels):
                if label >= 0:
                    soft_memberships[i, label] = 1.0

        logger.debug(
            f"HDBSCAN soft memberships: {soft_memberships.shape}, "
            f"noise points: {(labels == -1).sum()}"
        )

        return soft_memberships, n_clusters

    def _compute_feature_affinity(self) -> np.ndarray:
        """Compute feature-to-feature affinity from soft memberships.

        Affinity[i,j] = sum over elements of membership_i * membership_j
        This measures how much two features share elements.

        Returns:
            Feature affinity matrix (n_features, n_features).
        """
        if self._memberships is None:
            raise ValueError("Must fit before computing affinity")

        # Normalize memberships to sum to 1 per element
        row_sums = self._memberships.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        normalized = self._memberships / row_sums

        # Affinity = W^T @ W
        affinity: np.ndarray = normalized.T @ normalized
        return affinity

    def _build_element_memberships(
        self, element_ids: list[str] | None
    ) -> dict[str | int, list[FeatureMembership]]:
        """Build per-element membership lists from soft membership matrix.

        Args:
            element_ids: Optional element ID list. Uses indices if not provided.

        Returns:
            Dict mapping element ID/index to list of FeatureMembership objects.
        """
        if self._memberships is None:
            raise ValueError("Must fit before building memberships")

        result: dict[str | int, list[FeatureMembership]] = {}

        for elem_idx in range(self._memberships.shape[0]):
            scores = self._memberships[elem_idx]

            # Normalize to probabilities
            total = scores.sum()
            if total > 0:
                scores = scores / total

            # Find all features above threshold
            above_threshold = [
                (feat_idx, float(score))
                for feat_idx, score in enumerate(scores)
                if score >= self.config.membership_threshold
            ]

            if not above_threshold:
                continue

            # Sort by score descending
            above_threshold.sort(key=lambda x: -x[1])

            # Build membership objects
            memberships = [
                FeatureMembership(
                    feature_idx=feat_idx,
                    score=score,
                    is_primary=(i == 0),
                )
                for i, (feat_idx, score) in enumerate(above_threshold)
            ]

            key = element_ids[elem_idx] if element_ids else elem_idx
            result[key] = memberships

        return result

    def get_element_memberships(
        self, result: SoftClusteringResult, element_id: str | int
    ) -> list[FeatureMembership]:
        """Get soft memberships for a specific element."""
        return result.element_memberships.get(element_id, [])

    def get_feature_members(
        self,
        result: SoftClusteringResult,
        feature_idx: int,
        min_score: float | None = None,
    ) -> list[tuple[str | int, float, bool]]:
        """Get all elements belonging to a feature with their scores."""
        min_score = min_score or self.config.membership_threshold

        members: list[tuple[str | int, float, bool]] = []
        for elem_id, memberships in result.element_memberships.items():
            for m in memberships:
                if m.feature_idx == feature_idx and m.score >= min_score:
                    members.append((elem_id, m.score, m.is_primary))
                    break

        return sorted(members, key=lambda x: -x[1])

    def get_connected_features(
        self,
        result: SoftClusteringResult,
        feature_idx: int,
        min_affinity: float | None = None,
    ) -> list[FeatureAffinity]:
        """Get features connected to the given feature."""
        min_affinity = min_affinity or self.config.affinity_threshold

        affinities = result.feature_affinity[feature_idx]
        connected = [
            FeatureAffinity(feature_idx=i, affinity=float(affinities[i]))
            for i in range(len(affinities))
            if i != feature_idx and affinities[i] >= min_affinity
        ]

        return sorted(connected, key=lambda x: -x.affinity)
