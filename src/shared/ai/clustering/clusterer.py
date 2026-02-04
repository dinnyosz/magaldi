"""Feature clustering using HDBSCAN on code element embeddings.

Groups semantically similar functions/methods into feature clusters
with optional Ollama-based labeling.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import hdbscan
import numpy as np

from shared.ai.summarization import SummarizationLLMClient

if TYPE_CHECKING:
    from shared.config import MagaldiConfig


class ClusteringError(Exception):
    """Raised when clustering fails."""

    pass


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class ClusterConfig:
    """Configuration for HDBSCAN clustering.

    HDBSCAN parameters default to None and are filled from ClusteringConfig
    in __post_init__ to ensure a single source of truth.
    """

    # HDBSCAN parameters - defaults from ClusteringConfig
    min_cluster_size: int | None = None
    min_samples: int | None = None
    metric: str | None = None
    cluster_selection_method: str | None = None
    cluster_selection_epsilon: float | None = None
    membership_threshold: float | None = None
    affinity_percentile: float | None = None

    # Element types to cluster
    element_types: list[str] = field(default_factory=lambda: ["function", "method"])

    # LLM settings for labeling (based on arxiv.org/html/2507.03160v2)
    api_base: str = "http://localhost:11434"
    labeling_model: str = "qwen3:4b-instruct"
    provider: str = "ollama"  # For tiered model name display
    label_temperature: float = 0.2
    label_top_p: float = 0.95
    label_max_tokens: int = 32
    label_timeout: int = 30

    # Soft clustering options
    soft_clustering: bool = True  # Enable soft/overlapping memberships

    def __post_init__(self) -> None:
        """Fill HDBSCAN defaults from ClusteringConfig."""
        from shared.config import ClusteringConfig
        defaults = ClusteringConfig()
        if self.min_cluster_size is None:
            self.min_cluster_size = defaults.min_cluster_size
        if self.min_samples is None:
            self.min_samples = defaults.min_samples
        if self.metric is None:
            self.metric = defaults.metric
        if self.cluster_selection_method is None:
            self.cluster_selection_method = defaults.cluster_selection_method
        if self.cluster_selection_epsilon is None:
            self.cluster_selection_epsilon = defaults.cluster_selection_epsilon
        if self.membership_threshold is None:
            self.membership_threshold = defaults.membership_threshold
        if self.affinity_percentile is None:
            self.affinity_percentile = defaults.affinity_percentile

    @classmethod
    def from_magaldi_config(
        cls,
        config: "MagaldiConfig",
        api_base: str | None = None,
        labeling_model: str | None = None,
        provider: str | None = None,
        # CLI overrides
        min_cluster_size: int | None = None,
        min_samples: int | None = None,
    ) -> "ClusterConfig":
        """Create ClusterConfig from MagaldiConfig.

        Args:
            config: The main Magaldi configuration.
            api_base: Override for LLM API base (usually from summarize model).
            labeling_model: Override for labeling model name.
            provider: Override for LLM provider.
            min_cluster_size: CLI override for min_cluster_size.
            min_samples: CLI override for min_samples.

        Returns:
            ClusterConfig with values from the central config.
        """
        cc = config.clustering
        return cls(
            # From ClusteringConfig (with optional CLI overrides)
            min_cluster_size=min_cluster_size if min_cluster_size is not None else cc.min_cluster_size,
            min_samples=min_samples if min_samples is not None else cc.min_samples,
            membership_threshold=cc.membership_threshold,
            affinity_percentile=cc.affinity_percentile,
            # LLM settings (passed in, typically from summarize model)
            api_base=api_base or "http://localhost:11434",
            labeling_model=labeling_model or "qwen3:4b-instruct",
            provider=provider or "ollama",
        )


@dataclass
class ElementMembership:
    """Soft membership of an element in a cluster/feature."""

    cluster_id: int
    score: float
    is_primary: bool = False


@dataclass
class ClusterAffinity:
    """Connection strength between two clusters."""

    cluster_id: int
    label: str | None
    affinity: float


@dataclass
class ClusterResult:
    """Result for a single cluster."""

    cluster_id: int
    label: str | None
    element_ids: list[str]
    element_names: list[str]
    centroid: list[float] | None = None
    # Soft clustering: connected clusters (from affinity matrix)
    connected_clusters: list[ClusterAffinity] = field(default_factory=list)
    # Element summaries for better labeling (element_id -> summary)
    element_summaries: dict[str, str] = field(default_factory=dict)

    @property
    def size(self) -> int:
        """Number of elements in this cluster."""
        return len(self.element_ids)


@dataclass
class ClusteringResult:
    """Overall clustering result."""

    clusters: list[ClusterResult]
    outlier_count: int
    outlier_element_ids: list[str]
    total_elements: int
    # Soft clustering: per-element memberships (element_id -> list of memberships)
    element_memberships: dict[str, list[ElementMembership]] = field(default_factory=dict)
    # Whether soft clustering was used
    is_soft_clustering: bool = False

    @property
    def cluster_count(self) -> int:
        """Number of clusters found (excluding outliers)."""
        return len(self.clusters)


# =============================================================================
# LABELING PROGRESS TRACKING
# =============================================================================


@dataclass
class LabelingTimingStats:
    """Timing statistics for labeling phase."""

    phase_start: float = 0.0
    total_label_time: float = 0.0
    label_count: int = 0

    def record(self, label_time: float) -> None:
        """Record timing for a completed label."""
        self.total_label_time += label_time
        self.label_count += 1

    @property
    def avg_label_time(self) -> float:
        """Average time per label API call."""
        return self.total_label_time / self.label_count if self.label_count > 0 else 0.0

    @property
    def elapsed(self) -> float:
        """Elapsed time since phase start."""
        return time.time() - self.phase_start

    def eta_seconds(self, completed: int, total: int) -> float | None:
        """Calculate ETA based on average label time."""
        if self.label_count == 0 or total <= completed:
            return None
        remaining = total - completed
        return remaining * self.avg_label_time


@dataclass
class LabelingProgressState:
    """Progress state for labeling display."""

    total: int
    completed: int
    skipped: int  # Clusters with no names to label
    failed: int
    timing: LabelingTimingStats
    current_cluster: str = ""  # Currently being labeled
    model: str = ""  # Model being used for labeling
    ctx_size: str = ""  # Context size being used (e.g., "2K")


@dataclass
class ClusteringProgressState:
    """Progress state for clustering display."""

    phase: str  # "projection", "nmf", "complete"
    phase_description: str
    current_step: int
    total_steps: int
    n_elements: int = 0
    n_features: int = 0
    cooccurrence_density: float = 0.0
    elapsed_seconds: float = 0.0
    eta_seconds: float | None = None

    @property
    def percent(self) -> float:
        """Completion percentage."""
        if self.total_steps == 0:
            return 0.0
        return (self.current_step / self.total_steps) * 100


# =============================================================================
# CLUSTER LABELING PROMPTS (Optimized for Prefix Caching)
# =============================================================================
# System message is STATIC and gets cached by Ollama's KV cache.
# User message contains VARIABLE content (function names).

LABEL_SYSTEM_PROMPT = """Given function/method names and their summaries, generate a short label (1-5 words) that describes the common feature or functionality they share.

IMPORTANT: Generate specific, descriptive labels that identify WHAT the code does, not HOW it's organized.
- Good labels: "user authentication", "database queries", "REST API endpoints", "file parsing", "input validation"
- AVOID generic labels like: "utilities", "helpers", "processing", "handling", "operations", "functionality", "management"

Write ONLY the label, nothing else."""

LABEL_USER_PROMPT = """Functions in this cluster:
{functions}"""

# Legacy single-prompt template (kept for backwards compatibility)
LABEL_PROMPT = """Given these function/method names and summaries, generate a short label (1-5 words) that describes the common feature or functionality they share.

Functions:
{functions}

IMPORTANT: Generate specific, descriptive labels that identify WHAT the code does, not HOW it's organized.
- Good labels: "user authentication", "database queries", "REST API endpoints", "file parsing", "input validation"
- AVOID generic labels like: "utilities", "helpers", "processing", "handling", "operations", "functionality", "management"

Label:"""


# =============================================================================
# FEATURE CLUSTERER
# =============================================================================


class FeatureClusterer:
    """Clusters code elements by semantic similarity using HDBSCAN."""

    def __init__(
        self,
        config: ClusterConfig | None = None,
        llm_client: SummarizationLLMClient | None = None,
    ):
        """Initialize clusterer.

        Args:
            config: Clustering configuration.
            llm_client: Optional LLM client for labeling.
        """
        self.config = config or ClusterConfig()
        self._llm_client = llm_client

    def _get_llm_client(self) -> SummarizationLLMClient:
        """Get or create LLM client."""
        if self._llm_client is None:
            self._llm_client = SummarizationLLMClient(
                url=self.config.api_base,
                model=self.config.labeling_model,
            )
        return self._llm_client

    def cluster(
        self,
        elements: list[dict[str, Any]],
        on_progress: Callable[[ClusteringProgressState], None] | None = None,
    ) -> ClusteringResult:
        """Run clustering on element embeddings.

        Uses soft clustering (random projection ensemble + NMF) when
        config.soft_clustering is True, otherwise uses standard HDBSCAN.

        Args:
            elements: List of dicts with element_id, embedding, name, element_type.
            on_progress: Optional callback for clustering progress updates.

        Returns:
            ClusteringResult with clusters and outliers.

        Raises:
            ClusteringError: If clustering fails or no embeddings provided.
        """
        if not elements:
            raise ClusteringError("No elements provided for clustering")

        # Extract embeddings as numpy array
        embeddings = []
        valid_elements = []
        for elem in elements:
            emb = elem.get("embedding")
            if emb is not None:
                embeddings.append(emb)
                valid_elements.append(elem)

        if not embeddings:
            raise ClusteringError("No embeddings found in elements")

        embedding_array = np.array(embeddings, dtype=np.float32)

        # Use soft clustering if enabled
        if self.config.soft_clustering:
            return self._cluster_soft(embedding_array, valid_elements, on_progress)

        return self._cluster_hard(embedding_array, valid_elements)

    def _cluster_hard(
        self,
        embedding_array: np.ndarray,
        valid_elements: list[dict[str, Any]],
    ) -> ClusteringResult:
        """Standard HDBSCAN clustering with hard assignments."""
        # Run HDBSCAN
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.config.min_cluster_size,
            min_samples=self.config.min_samples,
            metric="euclidean",
            cluster_selection_method="eom",
        )

        labels = clusterer.fit_predict(embedding_array)

        # Group elements by cluster label
        cluster_map: dict[int, list[dict[str, Any]]] = {}
        outliers: list[str] = []

        for idx, label in enumerate(labels):
            elem = valid_elements[idx]
            if label == -1:
                # Outlier
                outliers.append(elem["element_id"])
            else:
                if label not in cluster_map:
                    cluster_map[label] = []
                cluster_map[label].append(elem)

        # Build ClusterResult objects
        clusters: list[ClusterResult] = []
        for cluster_label, cluster_elements in sorted(cluster_map.items()):
            # Compute centroid
            cluster_embeddings = np.array(
                [e["embedding"] for e in cluster_elements], dtype=np.float32
            )
            centroid = cluster_embeddings.mean(axis=0).tolist()

            # Build element summaries dict
            elem_summaries = {
                e["element_id"]: e.get("summary", "")
                for e in cluster_elements
                if e.get("summary")
            }

            cluster = ClusterResult(
                cluster_id=cluster_label,
                label=None,  # Will be set by label_clusters
                element_ids=[e["element_id"] for e in cluster_elements],
                element_names=[e.get("name", "") for e in cluster_elements],
                centroid=centroid,
                element_summaries=elem_summaries,
            )
            clusters.append(cluster)

        return ClusteringResult(
            clusters=clusters,
            outlier_count=len(outliers),
            outlier_element_ids=outliers,
            total_elements=len(valid_elements),
            is_soft_clustering=False,
        )

    def _cluster_soft(
        self,
        embedding_array: np.ndarray,
        valid_elements: list[dict[str, Any]],
        on_progress: Callable[[ClusteringProgressState], None] | None = None,
    ) -> ClusteringResult:
        """Soft clustering with overlapping memberships using HDBSCAN."""
        from shared.ai.clustering.soft_clustering import (
            SoftClusteringPipeline,
            SoftClusteringProgress,
        )
        from shared.config import ClusteringConfig

        n_elements = len(valid_elements)

        # Create progress adapter if callback provided
        def soft_progress_callback(progress: SoftClusteringProgress) -> None:
            if on_progress:
                on_progress(ClusteringProgressState(
                    phase=progress.phase,
                    phase_description=progress.phase_description,
                    current_step=progress.current_step,
                    total_steps=progress.total_steps,
                    n_elements=progress.n_elements,
                    n_features=progress.n_features,
                    cooccurrence_density=progress.cooccurrence_density,
                    elapsed_seconds=progress.elapsed_seconds,
                    eta_seconds=progress.eta_seconds,
                ))

        element_ids = [e["element_id"] for e in valid_elements]

        # HDBSCAN soft clustering - use ClusteringConfig directly
        soft_config = ClusteringConfig(
            min_cluster_size=self.config.min_cluster_size,
            min_samples=self.config.min_samples,
            metric=self.config.metric,
            cluster_selection_method=self.config.cluster_selection_method,
            cluster_selection_epsilon=self.config.cluster_selection_epsilon,
            membership_threshold=self.config.membership_threshold,
            affinity_percentile=self.config.affinity_percentile,
        )
        pipeline = SoftClusteringPipeline(soft_config)

        # Run soft clustering
        soft_result = pipeline.fit(
            embedding_array,
            element_ids,
            on_progress=soft_progress_callback if on_progress else None,
        )

        # Convert soft memberships to ClusteringResult format
        # Group elements by their PRIMARY cluster (highest score)
        cluster_map: dict[int, list[dict[str, Any]]] = {}
        outliers: list[str] = []
        element_memberships: dict[str, list[ElementMembership]] = {}

        for idx, elem in enumerate(valid_elements):
            elem_id = elem["element_id"]
            memberships = soft_result.element_memberships.get(elem_id, [])

            if not memberships:
                # No memberships - treat as outlier
                outliers.append(elem_id)
                continue

            # Store all soft memberships
            element_memberships[elem_id] = [
                ElementMembership(
                    cluster_id=m.feature_idx,
                    score=m.score,
                    is_primary=m.is_primary,
                )
                for m in memberships
            ]

            # Primary cluster is the first (highest score)
            primary_cluster = memberships[0].feature_idx
            if primary_cluster not in cluster_map:
                cluster_map[primary_cluster] = []
            cluster_map[primary_cluster].append(elem)

        # Build ClusterResult objects with connected clusters
        clusters: list[ClusterResult] = []
        for cluster_id, cluster_elements in sorted(cluster_map.items()):
            # Compute centroid
            cluster_embeddings = np.array(
                [e["embedding"] for e in cluster_elements], dtype=np.float32
            )
            centroid = cluster_embeddings.mean(axis=0).tolist()

            # Get connected clusters from affinity matrix (uses computed percentile threshold)
            connected = pipeline.get_connected_features(soft_result, cluster_id)
            connected_clusters = [
                ClusterAffinity(
                    cluster_id=c.feature_idx,
                    label=None,  # Will be filled after labeling
                    affinity=c.affinity,
                )
                for c in connected
            ]

            # Build element summaries dict
            elem_summaries = {
                e["element_id"]: e.get("summary", "")
                for e in cluster_elements
                if e.get("summary")
            }

            cluster = ClusterResult(
                cluster_id=cluster_id,
                label=None,  # Will be set by label_clusters
                element_ids=[e["element_id"] for e in cluster_elements],
                element_names=[e.get("name", "") for e in cluster_elements],
                centroid=centroid,
                connected_clusters=connected_clusters,
                element_summaries=elem_summaries,
            )
            clusters.append(cluster)

        return ClusteringResult(
            clusters=clusters,
            outlier_count=len(outliers),
            outlier_element_ids=outliers,
            total_elements=len(valid_elements),
            element_memberships=element_memberships,
            is_soft_clustering=True,
        )

    def label_clusters(
        self,
        result: ClusteringResult,
        max_names_per_prompt: int = 15,
        on_progress: Callable[[LabelingProgressState], None] | None = None,
        timing_stats: LabelingTimingStats | None = None,
        scope: str | None = None,
        repository: str | None = None,
        username: str | None = None,
        magaldi_config: "MagaldiConfig | None" = None,
    ) -> ClusteringResult:
        """Generate labels for clusters using Ollama.

        Args:
            result: Clustering result to label.
            max_names_per_prompt: Max function names to include in prompt.
            on_progress: Optional callback for progress updates.
            timing_stats: Optional timing stats (created if not provided).
            scope: Repository scope (for Redis tracking).
            repository: Repository name (for Redis tracking).
            username: Username/branch (for Redis tracking).
            magaldi_config: Optional config for Redis job tracking.

        Returns:
            Updated ClusteringResult with labels.
        """
        llm_client = self._get_llm_client()

        # Initialize Redis job tracking if config provided
        redis_repo = None
        if magaldi_config and scope and repository and username:
            from shared.db.redis import RedisLabelingJobRepository
            redis_repo = RedisLabelingJobRepository(magaldi_config)
            # Clear stale labeling queue data
            client = redis_repo._get_client()
            for key_type in ["jobs", "running", "queue"]:
                client.delete(f"magaldi:labeling:{key_type}:{scope}:{repository}:{username}")
            # Add all clusters as pending jobs (convert numpy int64 to Python int)
            for cluster in result.clusters:
                redis_repo.add_job(int(cluster.cluster_id), scope, repository, username)

        # Initialize timing stats
        if timing_stats is None:
            timing_stats = LabelingTimingStats()
        timing_stats.phase_start = time.time()

        total = len(result.clusters)
        completed = 0
        skipped = 0
        failed = 0
        labeling_model = self.config.labeling_model

        # Helper to get tiered model display name for Ollama
        def get_display_model(num_ctx: int) -> str:
            if self.config.provider == "ollama":
                from shared.ai.ollama_models import get_tiered_model_name
                return str(get_tiered_model_name(labeling_model, num_ctx))
            return labeling_model

        for cluster in result.clusters:
            # Mark as running in Redis (convert numpy int64 to Python int)
            if redis_repo and scope and repository and username:
                redis_repo.mark_running(int(cluster.cluster_id), scope, repository, username)

            # Build function lines with names and summaries
            func_lines = []
            for i, elem_id in enumerate(cluster.element_ids[:max_names_per_prompt]):
                name = cluster.element_names[i] if i < len(cluster.element_names) else ""
                if not name:
                    continue
                summary = cluster.element_summaries.get(elem_id, "")
                if summary:
                    # Truncate long summaries to keep prompt concise
                    if len(summary) > 150:
                        summary = summary[:147] + "..."
                    func_lines.append(f"- {name}(): {summary}")
                else:
                    func_lines.append(f"- {name}()")

            functions_str = "\n".join(func_lines)

            # Build display name for progress
            current_name = f"cluster_{cluster.cluster_id} ({cluster.size} members)"

            if not functions_str:
                cluster.label = f"cluster_{cluster.cluster_id}"
                skipped += 1
                completed += 1
                # Mark as completed in Redis (skipped still counts as completed)
                if redis_repo and scope and repository and username:
                    redis_repo.mark_completed(int(cluster.cluster_id), scope, repository, username)
                if on_progress:
                    on_progress(LabelingProgressState(
                        total=total,
                        completed=completed,
                        skipped=skipped,
                        failed=failed,
                        timing=timing_stats,
                        current_cluster="",
                        model=labeling_model,
                    ))
                continue

            # Build messages optimized for prefix caching
            user_content = LABEL_USER_PROMPT.format(functions=functions_str)
            messages = [
                {"role": "system", "content": LABEL_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ]

            # Compute dynamic context size based on prompt length
            from shared.ai.context_size import compute_aggregation_num_ctx
            prompt_chars = len(LABEL_SYSTEM_PROMPT) + len(user_content)
            num_ctx = compute_aggregation_num_ctx(prompt_chars, task_type="labeling")
            ctx_size_str = f"{num_ctx // 1024}K" if num_ctx >= 1024 else str(num_ctx)

            # Report current cluster being labeled (with context size)
            if on_progress:
                on_progress(LabelingProgressState(
                    total=total,
                    completed=completed,
                    skipped=skipped,
                    failed=failed,
                    timing=timing_stats,
                    current_cluster=current_name,
                    model=get_display_model(num_ctx),
                    ctx_size=ctx_size_str,
                ))

            try:
                api_start = time.time()
                raw_label = llm_client.generate_from_messages(
                    messages=messages,
                    temperature=self.config.label_temperature,
                    top_p=self.config.label_top_p,
                    max_tokens=self.config.label_max_tokens,
                    timeout=self.config.label_timeout,
                    num_ctx=num_ctx,
                )
                label_time = time.time() - api_start
                timing_stats.record(label_time)

                # Clean label
                label = self._clean_label(raw_label)
                cluster.label = label or f"cluster_{cluster.cluster_id}"
                completed += 1
                # Mark as completed in Redis
                if redis_repo and scope and repository and username:
                    redis_repo.mark_completed(int(cluster.cluster_id), scope, repository, username)

            except Exception as e:
                # Fall back to numbered label
                cluster.label = f"cluster_{cluster.cluster_id}"
                failed += 1
                completed += 1
                # Mark as failed in Redis
                if redis_repo and scope and repository and username:
                    redis_repo.mark_failed(int(cluster.cluster_id), scope, repository, username, str(e))

            # Report progress after each cluster
            if on_progress:
                on_progress(LabelingProgressState(
                    total=total,
                    completed=completed,
                    skipped=skipped,
                    failed=failed,
                    timing=timing_stats,
                    current_cluster="",
                    model=get_display_model(num_ctx),
                ))

        return result

    def _clean_label(self, raw_label: str) -> str:
        """Clean and normalize generated label.

        Args:
            raw_label: Raw label from Ollama.

        Returns:
            Cleaned label string.
        """
        label = raw_label.strip().lower()

        # Remove quotes
        label = label.strip("\"'")

        # Remove common prefixes
        prefixes = ["label:", "the label is", "this cluster is about"]
        for prefix in prefixes:
            if label.startswith(prefix):
                label = label[len(prefix) :].strip()

        # Replace spaces with underscores for consistency
        label = label.replace(" ", "_")

        # Remove special characters except underscores
        label = "".join(c for c in label if c.isalnum() or c == "_")

        # Limit length
        if len(label) > 50:
            label = label[:50]

        return label


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def run_clustering(
    elements: list[dict[str, Any]],
    config: ClusterConfig | None = None,
    label_clusters: bool = True,
) -> ClusteringResult:
    """Run clustering pipeline on elements.

    Args:
        elements: List of elements with embeddings.
        config: Optional clustering config.
        label_clusters: Whether to generate labels with Ollama.

    Returns:
        ClusteringResult with clusters.
    """
    clusterer = FeatureClusterer(config)

    # Run HDBSCAN
    result = clusterer.cluster(elements)

    # Optionally label clusters
    if label_clusters and result.clusters:
        result = clusterer.label_clusters(result)

    return result
