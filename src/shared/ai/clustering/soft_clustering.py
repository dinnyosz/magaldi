"""Soft clustering using random projection ensemble and NMF.

Implements overlapping/soft cluster memberships for high-dimensional embeddings
using random projections, ensemble clustering, and cooccurrence analysis.

Two pipeline options are provided:
1. SoftClusteringPipeline: Random projection ensemble + NMF (O(n²) cooccurrence matrix)
2. ScalableSoftClusteringPipeline: PCA + UMAP + HDBSCAN soft clustering (O(n log n))

For datasets with 5k+ elements, use ScalableSoftClusteringPipeline for better performance.

See plans/random_projection_ensemble_clustering.md for detailed algorithm design.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field

import hdbscan
import numpy as np
import umap
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.decomposition import NMF, PCA
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


@dataclass
class ScalableSoftClusteringConfig:
    """Configuration for scalable soft clustering using UMAP + HDBSCAN.

    This approach is O(n log n) vs O(n²) for the cooccurrence-based approach,
    making it suitable for large codebases (5k-30k+ elements).

    Algorithm:
    1. PCA reduces dimensionality while preserving distances (critical for UMAP)
    2. UMAP further reduces to clustering-friendly dimensions
    3. HDBSCAN with soft clustering extracts memberships directly

    Attributes:
        pca_components: Number of PCA components (0 = auto: min(100, 90% variance)).
            PCA first preserves Euclidean distances that UMAP would distort.
        umap_components: Final dimensionality for HDBSCAN clustering.
        umap_n_neighbors: UMAP locality parameter. Higher = more global structure.
            15-30 is typical; higher values help preserve distances.
        umap_min_dist: UMAP minimum distance between points. Higher = less clumping.
            0.1-0.3 typical; higher helps prevent distance distortion.
        umap_metric: Distance metric for UMAP. 'cosine' for embeddings.
        min_cluster_size: HDBSCAN minimum cluster size.
        min_samples: HDBSCAN min_samples (density threshold).
        membership_threshold: Minimum soft membership to retain.
        affinity_threshold: Minimum affinity between features to report.
        random_state: Random seed for reproducibility.
    """

    pca_components: int = 0  # 0 = auto: min(100, n_components for 90% variance)
    umap_components: int = 50
    umap_n_neighbors: int = 30  # Higher for better distance preservation
    umap_min_dist: float = 0.1  # Higher to reduce clumping/distortion
    umap_metric: str = "cosine"  # Best for embedding spaces
    min_cluster_size: int = 5
    min_samples: int = 2
    membership_threshold: float = 0.01
    affinity_threshold: float = 0.001
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

        self._memberships, self._feature_affinity = self._extract_memberships(
            effective_n_features,
            on_progress=on_progress,
            n_elements=n_elements,
            cooccurrence_density=cooccurrence_density,
            start_time=nmf_start_time,
        )

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
        self,
        n_features: int,
        on_progress: ProgressCallback | None = None,
        n_elements: int = 0,
        cooccurrence_density: float = 0.0,
        start_time: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract soft memberships via NMF decomposition.

        NMF decomposes cooccurrence ≈ W @ H where W is the membership matrix.
        Non-negativity constraint forces additive, parts-based representation,
        making each element a positive sum of feature contributions.

        Uses iterative calls to non_negative_factorization with init='custom'
        to enable progress tracking between iteration batches.

        Args:
            n_features: Number of features to extract.
            on_progress: Optional progress callback.
            n_elements: Number of elements (for progress reporting).
            cooccurrence_density: Cooccurrence density (for progress reporting).
            start_time: Start time for elapsed/ETA calculation.

        Returns:
            Tuple of (normalized_memberships, feature_affinity) matrices.
        """
        import time
        from sklearn.decomposition import non_negative_factorization

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

        # Run NMF in batches of iterations to allow progress reporting
        # Each batch runs iter_per_batch iterations, then we continue with init='custom'
        # Use smaller batches for more frequent progress updates
        total_iterations = self.config.nmf_max_iter
        iter_per_batch = 20  # Small batches for responsive progress
        completed_iterations = 0

        W = None
        H = None
        converged = False
        reconstruction_err = float('inf')
        prev_reconstruction_err = float('inf')
        convergence_tol = 1e-4  # Our own convergence check between batches

        # Suppress sklearn's ConvergenceWarning during batch NMF
        # Rationale: sklearn warns when n_iter==max_iter, but we INTENTIONALLY
        # run in small batches to report progress. We check convergence ourselves
        # by comparing reconstruction error between batches. This is a legitimate
        # use case that sklearn doesn't support natively.
        import warnings
        from sklearn.exceptions import ConvergenceWarning

        while completed_iterations < total_iterations and not converged:
            remaining_iters = total_iterations - completed_iterations
            batch_iters = min(iter_per_batch, remaining_iters)

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)

                if W is None:
                    # First batch - use specified init
                    W, H, n_iter = non_negative_factorization(
                        cooccurrence_dense,
                        n_components=n_features,
                        init=init,
                        max_iter=batch_iters,
                        random_state=self.config.random_state,
                    )
                else:
                    # Subsequent batches - continue from previous W, H
                    W, H, n_iter = non_negative_factorization(
                        cooccurrence_dense,
                        W=W,
                        H=H,
                        n_components=n_features,
                        init="custom",
                        update_H=True,
                        max_iter=batch_iters,
                        random_state=self.config.random_state,
                    )

            completed_iterations += n_iter

            # Calculate reconstruction error
            reconstruction_err = np.linalg.norm(cooccurrence_dense - W @ H, 'fro')

            # Check convergence: either sklearn converged early, or error improvement is minimal
            if n_iter < batch_iters:
                converged = True
                logger.debug(f"NMF converged early at iteration {completed_iterations}")
            elif prev_reconstruction_err - reconstruction_err < convergence_tol * prev_reconstruction_err:
                converged = True
                logger.debug(f"NMF converged at iteration {completed_iterations} (error delta below tolerance)")

            prev_reconstruction_err = reconstruction_err

            # Report progress
            if on_progress:
                elapsed = time.time() - start_time if start_time else 0
                # ETA based on iterations completed
                if completed_iterations > 0:
                    remaining = total_iterations - completed_iterations
                    eta = (elapsed / completed_iterations) * remaining
                else:
                    eta = None

                on_progress(SoftClusteringProgress(
                    phase="nmf",
                    phase_description="NMF membership extraction",
                    current_step=completed_iterations,
                    total_steps=total_iterations,
                    n_elements=n_elements,
                    n_features=n_features,
                    cooccurrence_density=cooccurrence_density,
                    elapsed_seconds=elapsed,
                    eta_seconds=eta,
                ))

        # Log convergence status
        if converged:
            logger.debug(f"NMF converged at iteration {completed_iterations}")
        elif completed_iterations >= total_iterations:
            logger.warning(
                f"NMF reached max iterations ({total_iterations}). "
                f"Consider increasing nmf_max_iter or reducing n_features. "
                f"Reconstruction error: {reconstruction_err:.4f}"
            )

        logger.debug(f"NMF reconstruction error: {reconstruction_err:.4f}")

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
# SCALABLE SOFT CLUSTERING PIPELINE (UMAP + HDBSCAN)
# =============================================================================


class ScalableSoftClusteringPipeline:
    """Scalable soft clustering using PCA + UMAP + HDBSCAN.

    Designed for large codebases (5k-30k+ elements) where the O(n²) cooccurrence
    matrix approach becomes prohibitively expensive.

    Algorithm:
    1. PCA preprocessing preserves Euclidean distances (critical before UMAP)
    2. UMAP reduces to clustering-friendly dimensions with tuned parameters
    3. HDBSCAN with prediction_data=True enables soft clustering
    4. all_points_membership_vectors() extracts soft memberships directly

    Complexity: O(n log n) for UMAP, O(n) for HDBSCAN
    vs O(n²) for cooccurrence matrix construction

    Example:
        >>> pipeline = ScalableSoftClusteringPipeline()
        >>> result = pipeline.fit(embeddings, element_ids)
        >>> memberships = result.element_memberships["element_42"]
        >>> for m in memberships:
        ...     print(f"Feature {m.feature_idx}: {m.score:.2%}")
    """

    def __init__(self, config: ScalableSoftClusteringConfig | None = None):
        """Initialize pipeline with configuration.

        Args:
            config: Clustering configuration. Uses defaults if not provided.
        """
        self.config = config or ScalableSoftClusteringConfig()
        self._pca_model: PCA | None = None
        self._umap_model: umap.UMAP | None = None
        self._clusterer: hdbscan.HDBSCAN | None = None
        self._memberships: np.ndarray | None = None
        self._feature_affinity: np.ndarray | None = None

    def fit(
        self,
        embeddings: np.ndarray,
        element_ids: list[str] | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> SoftClusteringResult:
        """Fit the pipeline using UMAP + HDBSCAN soft clustering.

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
            f"Starting scalable soft clustering: {n_elements} elements, "
            f"{embedding_dim}D embeddings"
        )

        # Phase 1: PCA preprocessing
        phase_start = time.time()
        if on_progress:
            on_progress(SoftClusteringProgress(
                phase="pca",
                phase_description="PCA dimensionality reduction",
                current_step=0,
                total_steps=1,
                n_elements=n_elements,
            ))

        pca_reduced = self._apply_pca(embeddings)
        pca_dims = pca_reduced.shape[1]

        if on_progress:
            elapsed = time.time() - phase_start
            on_progress(SoftClusteringProgress(
                phase="pca",
                phase_description="PCA dimensionality reduction",
                current_step=1,
                total_steps=1,
                n_elements=n_elements,
                n_features=pca_dims,
                elapsed_seconds=elapsed,
            ))

        logger.info(f"PCA: {embedding_dim}D → {pca_dims}D")

        # Phase 2: UMAP embedding
        umap_start = time.time()
        if on_progress:
            on_progress(SoftClusteringProgress(
                phase="umap",
                phase_description="UMAP embedding",
                current_step=0,
                total_steps=1,
                n_elements=n_elements,
            ))

        umap_reduced = self._apply_umap(pca_reduced, n_elements, on_progress)
        umap_dims = umap_reduced.shape[1]

        if on_progress:
            elapsed = time.time() - umap_start
            on_progress(SoftClusteringProgress(
                phase="umap",
                phase_description="UMAP embedding",
                current_step=1,
                total_steps=1,
                n_elements=n_elements,
                n_features=umap_dims,
                elapsed_seconds=elapsed,
            ))

        logger.info(f"UMAP: {pca_dims}D → {umap_dims}D")

        # Phase 3: HDBSCAN soft clustering
        hdbscan_start = time.time()
        if on_progress:
            on_progress(SoftClusteringProgress(
                phase="hdbscan",
                phase_description="HDBSCAN soft clustering",
                current_step=0,
                total_steps=1,
                n_elements=n_elements,
            ))

        self._memberships, n_clusters = self._apply_hdbscan_soft(umap_reduced)

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

        # Phase 4: Compute feature affinity from soft memberships
        self._feature_affinity = self._compute_feature_affinity()

        # Build element membership mapping
        element_memberships = self._build_element_memberships(element_ids)

        # Count features with members
        features_with_members = set()
        for memberships in element_memberships.values():
            for m in memberships:
                features_with_members.add(m.feature_idx)

        logger.info(
            f"Scalable soft clustering complete: {len(element_memberships)} elements "
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
            cooccurrence_density=0.0,  # Not applicable for this approach
        )

    def _apply_pca(self, embeddings: np.ndarray) -> np.ndarray:
        """Apply PCA to preserve distances before UMAP.

        PCA is critical here because:
        1. It preserves Euclidean distances (unlike UMAP)
        2. Reduces computational cost of UMAP
        3. Removes noise dimensions that could confuse clustering

        Args:
            embeddings: (n_elements, embedding_dim) array.

        Returns:
            PCA-reduced embeddings.
        """
        n_samples, n_features = embeddings.shape

        if self.config.pca_components == 0:
            # Auto-calculate: use enough components for 90% variance, capped at 100
            # or n_samples-1 (whichever is smaller)
            max_components = min(100, n_samples - 1, n_features)
            self._pca_model = PCA(
                n_components=max_components,
                random_state=self.config.random_state,
            )
            reduced = self._pca_model.fit_transform(embeddings)

            # Find how many components for 90% variance
            cumsum = np.cumsum(self._pca_model.explained_variance_ratio_)
            n_for_90pct = np.searchsorted(cumsum, 0.90) + 1
            n_components = min(n_for_90pct, max_components)

            logger.debug(
                f"PCA auto-selected {n_components} components "
                f"({cumsum[n_components-1]:.1%} variance explained)"
            )

            # Return only the needed components
            return reduced[:, :n_components]
        else:
            # Use specified number of components
            n_components = min(self.config.pca_components, n_samples - 1, n_features)
            self._pca_model = PCA(
                n_components=n_components,
                random_state=self.config.random_state,
            )
            return self._pca_model.fit_transform(embeddings)

    def _apply_umap(
        self,
        embeddings: np.ndarray,
        n_elements: int,
        on_progress: ProgressCallback | None = None,
    ) -> np.ndarray:
        """Apply UMAP with tuned parameters for distance preservation.

        UMAP parameters are tuned to minimize distance distortion:
        - Higher n_neighbors (30) preserves more global structure
        - Higher min_dist (0.1) prevents excessive clumping
        - Cosine metric matches embedding space geometry

        Args:
            embeddings: PCA-reduced embeddings.
            n_elements: Total elements (for progress).
            on_progress: Progress callback.

        Returns:
            UMAP-reduced embeddings.
        """
        n_samples = embeddings.shape[0]

        # Adjust n_neighbors if dataset is small
        n_neighbors = min(self.config.umap_n_neighbors, n_samples - 1)

        # Adjust target dimensions if needed
        n_components = min(self.config.umap_components, embeddings.shape[1])

        # Create UMAP with progress callback if available
        # Note: UMAP doesn't have a native progress callback, but we can use
        # verbose mode and parse output, or just report at start/end
        self._umap_model = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=self.config.umap_min_dist,
            metric=self.config.umap_metric,
            random_state=self.config.random_state,
            verbose=False,  # We manage our own progress
        )

        return self._umap_model.fit_transform(embeddings)

    def _apply_hdbscan_soft(
        self,
        embeddings: np.ndarray,
    ) -> tuple[np.ndarray, int]:
        """Apply HDBSCAN with soft clustering support.

        Uses prediction_data=True to enable soft clustering via
        all_points_membership_vectors().

        Args:
            embeddings: UMAP-reduced embeddings.

        Returns:
            Tuple of (soft_membership_matrix, n_clusters).
        """
        n_samples = embeddings.shape[0]

        # Adjust min_cluster_size if dataset is small
        min_cluster_size = min(self.config.min_cluster_size, max(2, n_samples // 10))
        min_samples = min(self.config.min_samples, min_cluster_size)

        self._clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="euclidean",  # Euclidean works well on UMAP output
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
        return normalized.T @ normalized

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
