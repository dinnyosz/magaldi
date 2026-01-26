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
    """Configuration for HDBSCAN clustering."""

    # HDBSCAN parameters
    min_cluster_size: int = 5
    min_samples: int = 3

    # Element types to cluster
    element_types: list[str] = field(default_factory=lambda: ["function", "method"])

    # LLM settings for labeling (based on arxiv.org/html/2507.03160v2)
    api_base: str = "http://localhost:11434"
    labeling_model: str = "qwen3:4b-instruct"
    label_temperature: float = 0.2
    label_top_p: float = 0.95
    label_max_tokens: int = 32
    label_timeout: int = 30


@dataclass
class ClusterResult:
    """Result for a single cluster."""

    cluster_id: int
    label: str | None
    element_ids: list[str]
    element_names: list[str]
    centroid: list[float] | None = None

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


# =============================================================================
# CLUSTER LABELING PROMPTS (Optimized for Prefix Caching)
# =============================================================================
# System message is STATIC and gets cached by Ollama's KV cache.
# User message contains VARIABLE content (function names).

LABEL_SYSTEM_PROMPT = """Given function/method names from a code cluster, generate a short label (1-5 words) that describes the common feature or functionality they share.

Generate ONLY a short label like: "user authentication", "database query handling", "REST API endpoints", "file processing utilities", "input validation", etc.

Write ONLY the label, nothing else."""

LABEL_USER_PROMPT = """Function names:
{names}"""

# Legacy single-prompt template (kept for backwards compatibility)
LABEL_PROMPT = """Given these function/method names from a code cluster, generate a short label (1-5 words) that describes the common feature or functionality they share.

Function names:
{names}

Generate ONLY a short label like: "user authentication", "database query handling", "REST API endpoints", "file processing utilities", "input validation", etc.

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
    ) -> ClusteringResult:
        """Run HDBSCAN clustering on element embeddings.

        Args:
            elements: List of dicts with element_id, embedding, name, element_type.

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

            cluster = ClusterResult(
                cluster_id=cluster_label,
                label=None,  # Will be set by label_clusters
                element_ids=[e["element_id"] for e in cluster_elements],
                element_names=[e.get("name", "") for e in cluster_elements],
                centroid=centroid,
            )
            clusters.append(cluster)

        return ClusteringResult(
            clusters=clusters,
            outlier_count=len(outliers),
            outlier_element_ids=outliers,
            total_elements=len(valid_elements),
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

        for cluster in result.clusters:
            # Mark as running in Redis (convert numpy int64 to Python int)
            if redis_repo and scope and repository and username:
                redis_repo.mark_running(int(cluster.cluster_id), scope, repository, username)
            # Get sample of function names
            names = cluster.element_names[:max_names_per_prompt]
            names_str = "\n".join(f"- {name}" for name in names if name)

            # Build display name for progress
            current_name = f"cluster_{cluster.cluster_id} ({cluster.size} members)"

            if not names_str:
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
            user_content = LABEL_USER_PROMPT.format(names=names_str)
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
                    model=labeling_model,
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
                    model=labeling_model,
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
