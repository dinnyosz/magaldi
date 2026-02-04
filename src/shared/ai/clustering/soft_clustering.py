"""Soft clustering using random projection ensemble and NMF.

Implements overlapping/soft cluster memberships for high-dimensional embeddings
using random projections, ensemble clustering, and cooccurrence analysis.

See plans/random_projection_ensemble_clustering.md for detailed algorithm design.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field

import hdbscan
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.decomposition import NMF
from sklearn.random_projection import GaussianRandomProjection

logger = logging.getLogger(__name__)


# =============================================================================
# PROGRESS TRACKING
# =============================================================================


@dataclass
class SoftClusteringProgress:
    """Progress state for soft clustering pipeline.

    Used for CLI feedback during long-running clustering operations.
    """

    phase: str  # "projection", "nmf", "complete"
    phase_description: str  # Human-readable description
    current_step: int
    total_steps: int
    # Optional details
    n_elements: int = 0
    n_features: int = 0
    cooccurrence_density: float = 0.0
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
    """Configuration for soft clustering pipeline.

    Attributes:
        n_projection_runs: Number of random projection + clustering runs.
            More runs = smoother cooccurrence matrix. Diminishing returns past ~50.
        projection_dims: Target dimensions for random projection (J-L lemma).
            Lower = faster, higher = more faithful to original structure.
        min_cluster_size: HDBSCAN min_cluster_size for each projection run.
            Start with sqrt(n_elements) as a heuristic.
        n_features: Number of features (clusters) to extract via NMF.
            If 0 (default), auto-calculated as sqrt(n_elements) capped at 200.
            Use reconstruction error elbow plot or domain knowledge for manual tuning.
        membership_threshold: Minimum membership score to keep (filters numerical noise).
            Algorithm self-filters unrelated memberships; this handles floating point.
        cooccurrence_threshold: Minimum cooccurrence probability to store in sparse matrix.
        affinity_threshold: Minimum affinity to consider features connected.
    """

    n_projection_runs: int = 50
    projection_dims: int = 50
    min_cluster_size: int = 15
    min_samples: int = 3  # HDBSCAN min_samples
    n_features: int = 0  # 0 = auto-calculate based on dataset size
    membership_threshold: float = 0.01  # 1% - low because algorithm self-filters
    cooccurrence_threshold: float = 0.05  # 5% cooccurrence minimum
    affinity_threshold: float = 0.001  # Very low to capture weak connections
    nmf_max_iter: int = 1000  # Increased for better convergence
    random_state: int = 42


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
        cooccurrence_density: Density of the sparse cooccurrence matrix.
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
    """Pipeline for soft/overlapping clustering of high-dimensional embeddings.

    Designed for Magaldi feature extraction at scale (1k-30k elements, 500+ features).

    Algorithm:
    1. Random projections reduce dimensionality (J-L lemma preserves distances)
    2. Ensemble clustering across projections builds cooccurrence matrix
    3. Sparse storage enables scaling to 30k elements
    4. NMF decomposes cooccurrence into soft membership matrix
    5. Feature affinity emerges from membership overlap

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
        self._cooccurrence: csr_matrix | None = None
        self._memberships: np.ndarray | None = None  # W matrix from NMF
        self._feature_affinity: np.ndarray | None = None

    def fit(
        self,
        embeddings: np.ndarray,
        element_ids: list[str] | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> SoftClusteringResult:
        """Fit the pipeline: build cooccurrence matrix, extract soft memberships.

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
        if embeddings.size == 0:
            raise ValueError("Empty embeddings array")

        if embeddings.ndim != 2:
            raise ValueError(f"Expected 2D embeddings, got {embeddings.ndim}D")

        n_elements = embeddings.shape[0]
        logger.info(
            f"Starting soft clustering: {n_elements} elements, "
            f"{self.config.n_projection_runs} projections, "
            f"{self.config.n_features} features"
        )

        # Calculate n_features if auto (0)
        # Heuristic: sqrt(n_elements) gives reasonable granularity, capped at 200
        if self.config.n_features == 0:
            auto_n_features = max(10, min(200, int(np.sqrt(n_elements))))
            logger.info(f"Auto-calculated n_features: {auto_n_features} (sqrt of {n_elements})")
            effective_n_features = auto_n_features
        else:
            effective_n_features = self.config.n_features

        # Ensure n_features doesn't exceed what NMF can handle
        # NMF requires n_components <= min(n_samples, n_features of cooccurrence matrix)
        max_features = min(n_elements // 2, n_elements)
        if effective_n_features > max_features:
            logger.warning(
                f"Reduced n_features from {effective_n_features} to "
                f"{max_features} due to dataset size"
            )
            effective_n_features = max(1, max_features)

        import time
        phase_start_time = time.time()

        # Step 1-3: Build sparse cooccurrence matrix
        logger.info("Building cooccurrence matrix via ensemble clustering...")
        self._cooccurrence = self._build_cooccurrence_sparse(
            embeddings,
            on_progress=on_progress,
            n_elements=n_elements,
            n_features=effective_n_features,
            start_time=phase_start_time,
        )

        projection_elapsed = time.time() - phase_start_time

        cooccurrence_density = (
            self._cooccurrence.nnz / (n_elements * n_elements)
        )
        logger.info(
            f"Cooccurrence matrix: {self._cooccurrence.nnz} non-zero entries "
            f"({cooccurrence_density:.2%} density)"
        )

        # Step 4: NMF for soft memberships
        logger.info(f"Extracting memberships via NMF ({effective_n_features} features)...")
        nmf_start_time = time.time()

        # Report NMF phase start
        if on_progress:
            on_progress(SoftClusteringProgress(
                phase="nmf",
                phase_description="NMF membership extraction",
                current_step=0,
                total_steps=1,
                n_elements=n_elements,
                n_features=effective_n_features,
                cooccurrence_density=cooccurrence_density,
                elapsed_seconds=time.time() - phase_start_time,
            ))

        self._memberships, self._feature_affinity = self._extract_memberships(
            effective_n_features
        )

        # Report NMF complete
        if on_progress:
            on_progress(SoftClusteringProgress(
                phase="nmf",
                phase_description="NMF membership extraction",
                current_step=1,
                total_steps=1,
                n_elements=n_elements,
                n_features=effective_n_features,
                cooccurrence_density=cooccurrence_density,
            ))

        # Step 5: Build element membership mapping
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
                cooccurrence_density=cooccurrence_density,
            ))

        return SoftClusteringResult(
            element_memberships=element_memberships,
            feature_affinity=self._feature_affinity,
            n_features_found=len(features_with_members),
            n_elements=n_elements,
            cooccurrence_density=cooccurrence_density,
        )

    def _build_cooccurrence_sparse(
        self,
        embeddings: np.ndarray,
        on_progress: ProgressCallback | None = None,
        n_elements: int = 0,
        n_features: int = 0,
        start_time: float | None = None,
    ) -> csr_matrix:
        """Build sparse cooccurrence matrix from ensemble clustering.

        For each random projection, clusters elements with HDBSCAN and records
        which pairs land in the same cluster. The final matrix contains
        cooccurrence probabilities across all projections.

        Args:
            embeddings: (n_elements, embedding_dim) array.
            on_progress: Optional progress callback.
            n_elements: Number of elements (for progress reporting).
            n_features: Number of features (for progress reporting).

        Returns:
            Sparse (n_elements, n_elements) cooccurrence probability matrix.
        """
        n_points = embeddings.shape[0]
        embedding_dim = embeddings.shape[1]

        # Handle edge case: too few points for clustering
        if n_points < self.config.min_cluster_size:
            logger.warning(
                f"Too few points ({n_points}) for min_cluster_size "
                f"({self.config.min_cluster_size}). Returning identity cooccurrence."
            )
            # Return identity-like sparse matrix (each point only cooccurs with itself)
            return csr_matrix(np.eye(n_points, dtype=np.float32))

        # Ensure projection dims doesn't exceed embedding dims
        projection_dims = min(self.config.projection_dims, embedding_dim)

        # Use lil_matrix for efficient incremental construction
        cooccurrence = lil_matrix((n_points, n_points), dtype=np.float32)

        for run in range(self.config.n_projection_runs):
            # Random projection (different random state each run)
            rp = GaussianRandomProjection(
                n_components=projection_dims,
                random_state=self.config.random_state + run,
            )
            reduced = rp.fit_transform(embeddings)

            # Cluster in reduced space
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=self.config.min_cluster_size,
                min_samples=self.config.min_samples,
                metric="euclidean",
            )
            labels = clusterer.fit_predict(reduced)

            # Count co-occurrences (excluding noise points labeled -1)
            unique_labels = set(labels) - {-1}
            for cluster_id in unique_labels:
                indices = np.where(labels == cluster_id)[0]
                # Only update upper triangle + diagonal for efficiency
                for i_idx, i in enumerate(indices):
                    for j in indices[i_idx:]:  # j >= i
                        cooccurrence[i, j] += 1

            if (run + 1) % 10 == 0:
                logger.debug(f"Completed projection run {run + 1}/{self.config.n_projection_runs}")

            # Report progress with ETA
            if on_progress:
                import time
                elapsed = time.time() - start_time if start_time else 0
                completed = run + 1
                remaining = self.config.n_projection_runs - completed
                eta = (elapsed / completed * remaining) if completed > 0 else None

                on_progress(SoftClusteringProgress(
                    phase="projection",
                    phase_description="Random projection ensemble",
                    current_step=completed,
                    total_steps=self.config.n_projection_runs,
                    n_elements=n_elements,
                    n_features=n_features,
                    elapsed_seconds=elapsed,
                    eta_seconds=eta,
                ))

        # Normalize to probabilities
        cooccurrence = cooccurrence / self.config.n_projection_runs

        # Convert to CSR for efficient arithmetic
        cooccurrence = csr_matrix(cooccurrence)

        # Threshold small values
        cooccurrence.data[cooccurrence.data < self.config.cooccurrence_threshold] = 0
        cooccurrence.eliminate_zeros()

        # Make symmetric (copy upper triangle to lower)
        cooccurrence = cooccurrence + cooccurrence.T
        # Fix diagonal (was doubled)
        diag = cooccurrence.diagonal() / 2
        cooccurrence.setdiag(diag)

        return cooccurrence

    def _extract_memberships(
        self, n_features: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract soft memberships via NMF decomposition.

        NMF decomposes cooccurrence ≈ W @ H where W is the membership matrix.
        Non-negativity constraint forces additive, parts-based representation,
        making each element a positive sum of feature contributions.

        Args:
            n_features: Number of features to extract.

        Returns:
            Tuple of (normalized_memberships, feature_affinity) matrices.
        """
        # Convert sparse to dense for NMF
        # TODO: Consider sparse NMF variants for very large matrices
        cooccurrence_dense = self._cooccurrence.toarray()

        # Ensure non-negative (should be, but floating point)
        cooccurrence_dense = np.maximum(cooccurrence_dense, 0)

        n_samples = cooccurrence_dense.shape[0]

        # Choose init method based on dataset size
        # nndsvd requires n_components <= min(n_samples, n_features)
        if n_features <= min(n_samples, cooccurrence_dense.shape[1]):
            init = "nndsvd"
        else:
            init = "random"
            logger.debug(f"Using random init for NMF (n_features={n_features} > n_samples={n_samples})")

        nmf = NMF(
            n_components=n_features,
            init=init,
            max_iter=self.config.nmf_max_iter,
            random_state=self.config.random_state,
        )

        W = nmf.fit_transform(cooccurrence_dense)  # (n_elements, n_features)

        # Log if NMF didn't fully converge (n_iter_ == max_iter means it hit the limit)
        if hasattr(nmf, 'n_iter_') and nmf.n_iter_ >= self.config.nmf_max_iter:
            logger.warning(
                f"NMF reached max iterations ({self.config.nmf_max_iter}). "
                f"Consider increasing nmf_max_iter or reducing n_features. "
                f"Reconstruction error: {nmf.reconstruction_err_:.4f}"
            )

        logger.debug(f"NMF reconstruction error: {nmf.reconstruction_err_:.4f}")

        # Normalize rows to sum to 1 (proper probabilities)
        row_sums = W.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        W_normalized = W / row_sums

        # Compute feature affinity matrix: how much features share elements
        # feature_affinity[i,j] = sum over elements of membership_i * membership_j
        feature_affinity = W_normalized.T @ W_normalized  # (n_features, n_features)

        return W_normalized, feature_affinity

    def _build_element_memberships(
        self, element_ids: list[str] | None
    ) -> dict[str | int, list[FeatureMembership]]:
        """Build per-element membership lists from the NMF membership matrix.

        Args:
            element_ids: Optional element ID list. Uses indices if not provided.

        Returns:
            Dict mapping element ID/index to list of FeatureMembership objects,
            sorted by score descending with is_primary=True on the highest.
        """
        result: dict[str | int, list[FeatureMembership]] = {}

        for elem_idx in range(self._memberships.shape[0]):
            scores = self._memberships[elem_idx]

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

            # Build membership objects, mark first as primary
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
        """Get soft memberships for a specific element.

        Args:
            result: Result from fit().
            element_id: Element ID or index.

        Returns:
            List of FeatureMembership objects, sorted by score descending.
        """
        return result.element_memberships.get(element_id, [])

    def get_feature_members(
        self,
        result: SoftClusteringResult,
        feature_idx: int,
        min_score: float | None = None,
    ) -> list[tuple[str | int, float, bool]]:
        """Get all elements belonging to a feature with their scores.

        Args:
            result: Result from fit().
            feature_idx: Feature index to query.
            min_score: Minimum membership score. Uses config threshold if not provided.

        Returns:
            List of (element_id, score, is_primary) tuples, sorted by score descending.
        """
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
        """Get features connected to the given feature.

        Args:
            result: Result from fit().
            feature_idx: Feature index to query.
            min_affinity: Minimum affinity. Uses config threshold if not provided.

        Returns:
            List of FeatureAffinity objects, sorted by affinity descending.
        """
        min_affinity = min_affinity or self.config.affinity_threshold

        affinities = result.feature_affinity[feature_idx]
        connected = [
            FeatureAffinity(feature_idx=i, affinity=float(affinities[i]))
            for i in range(len(affinities))
            if i != feature_idx and affinities[i] >= min_affinity
        ]

        return sorted(connected, key=lambda x: -x.affinity)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def find_optimal_n_features(
    cooccurrence: csr_matrix | np.ndarray,
    k_range: range | None = None,
    max_iter: int = 500,
) -> tuple[list[int], list[float]]:
    """Find optimal number of features using reconstruction error elbow.

    Runs NMF with different k values and returns errors for elbow plot analysis.

    Args:
        cooccurrence: Cooccurrence matrix (sparse or dense).
        k_range: Range of k values to test. Defaults to range(50, 600, 50).
        max_iter: Maximum NMF iterations.

    Returns:
        Tuple of (k_values, reconstruction_errors) for plotting.
    """
    if k_range is None:
        k_range = range(50, 600, 50)

    if isinstance(cooccurrence, csr_matrix):
        cooccurrence = cooccurrence.toarray()

    cooccurrence = np.maximum(cooccurrence, 0)

    k_values = list(k_range)
    errors = []

    for k in k_values:
        nmf = NMF(n_components=k, init="nndsvd", max_iter=max_iter, random_state=42)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
            nmf.fit_transform(cooccurrence)
        errors.append(nmf.reconstruction_err_)
        logger.debug(f"k={k}: reconstruction error = {nmf.reconstruction_err_:.4f}")

    return k_values, errors
