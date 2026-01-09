"""Feature processor - summarize and embed features in parallel.

Processes feature clusters:
1. Fetch member summaries from ES
2. Generate feature summary using Ollama
3. Embed feature summary
4. Index feature document to ES
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from shared.ai.clustering.clusterer import ClusteringResult, ClusterResult

if TYPE_CHECKING:
    from shared.config import MagaldiConfig
from shared.db.elasticsearch import ElasticsearchRepository
from shared.ai.embedding import (
    OllamaEmbedClient,
    normalize_vector,
    validate_vector,
)
from shared.ai.summarization import OllamaClient

# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class FeatureProcessingConfig:
    """Configuration for feature processing."""

    summarize_model: str = "qwen2.5-coder:3b"
    embed_model: str = "snowflake-arctic-embed2"
    ollama_url: str = "http://localhost:11434"

    # Summarization settings
    summarize_temperature: float = 0.3
    summarize_max_tokens: int = 512  # Longer for feature summaries
    summarize_timeout: int = 90

    # Embedding settings
    embed_dimensions: int = 1024
    embed_timeout: int = 30

    # Parallel processing
    num_workers: int = 4

    # Feature summary settings
    max_member_summaries: int = 20  # Max member summaries to include in prompt


@dataclass
class SubClusterConfig:
    """Configuration for sub-clustering large features."""

    # HDBSCAN parameters for sub-clustering (smaller than main clustering)
    min_cluster_size: int = 3
    min_samples: int = 2

    # Threshold for triggering sub-clustering
    min_members_for_subclustering: int = 20


@dataclass
class FeatureTimingStats:
    """Thread-safe timing statistics for feature processing."""

    _lock: threading.Lock = field(default_factory=threading.Lock)
    phase_start: float = 0.0

    total_summarize_time: float = 0.0
    total_embed_time: float = 0.0
    summarize_count: int = 0
    embed_count: int = 0

    def record(self, summarize_time: float, embed_time: float) -> None:
        """Record timing for a completed feature."""
        with self._lock:
            self.total_summarize_time += summarize_time
            self.total_embed_time += embed_time
            self.summarize_count += 1
            if embed_time > 0:
                self.embed_count += 1

    @property
    def avg_summarize_time(self) -> float:
        """Average summarize time per feature."""
        with self._lock:
            return self.total_summarize_time / self.summarize_count if self.summarize_count > 0 else 0.0

    @property
    def avg_embed_time(self) -> float:
        """Average embed time per feature."""
        with self._lock:
            return self.total_embed_time / self.embed_count if self.embed_count > 0 else 0.0

    @property
    def elapsed(self) -> float:
        return time.time() - self.phase_start

    def eta_seconds(self, completed: int, total: int, num_workers: int = 1) -> float | None:
        """Calculate ETA based on average times."""
        with self._lock:
            if self.summarize_count == 0:
                return None

            avg_time = (self.total_summarize_time + self.total_embed_time) / self.summarize_count
            remaining = total - completed
            if remaining <= 0:
                return 0.0

            # Divide by workers for parallel processing
            return (remaining * avg_time) / max(num_workers, 1)


@dataclass
class FeatureWorkerStatus:
    """Track what each worker is doing."""

    _lock: threading.Lock = field(default_factory=threading.Lock)
    _status: dict[int, tuple[str, str, str]] = field(default_factory=dict)  # worker_id -> (feature_name, stage, model)

    def set(self, worker_id: int, feature_name: str, stage: str, model: str = "") -> None:
        with self._lock:
            self._status[worker_id] = (feature_name, stage, model)

    def clear(self, worker_id: int) -> None:
        with self._lock:
            self._status.pop(worker_id, None)

    def get_all(self) -> dict[int, tuple[str, str, str]]:
        with self._lock:
            return dict(self._status)


@dataclass
class FeatureProgressState:
    """Combined state for display updates."""

    total: int
    completed: int
    failed: int
    timing: FeatureTimingStats
    workers: FeatureWorkerStatus
    num_workers: int = 1


@dataclass
class SubfeatureProgressState:
    """Progress state for subfeature processing."""

    total_parent_features: int
    completed_parent_features: int
    total_subfeatures: int
    completed_subfeatures: int
    failed: int
    current_parent: str = ""
    current_stage: str = ""  # "clustering", "labeling", "summarizing", "embedding", "indexing"
    elapsed: float = 0.0


@dataclass
class ProcessedFeature:
    """Result from processing a single feature."""

    cluster_id: int
    success: bool
    summarize_time: float
    embed_time: float
    label: str = ""
    summary: str = ""
    error: str | None = None


# =============================================================================
# FEATURE SUMMARY PROMPT
# =============================================================================


FEATURE_SUMMARY_PROMPT = """You are analyzing a code feature containing related functions/methods.
Based on the member function summaries below, describe what this feature does.

Feature name: {label}
Number of members: {member_count}

Member function summaries:
{member_summaries}

Write a 2-4 sentence summary describing:
1. The overall purpose of this feature
2. The key operations it provides

Summary:"""


# =============================================================================
# FEATURE PROCESSING
# =============================================================================


def _generate_feature_summary(
    cluster: ClusterResult,
    member_summaries: dict[str, str],
    ollama: OllamaClient,
    config: FeatureProcessingConfig,
) -> str:
    """Generate summary for a feature based on member summaries.

    Args:
        cluster: Cluster result with member info.
        member_summaries: Dict mapping element_id to summary.
        ollama: Ollama client for LLM.
        config: Processing configuration.

    Returns:
        Generated feature summary.
    """
    # Build member summaries text
    summaries_text = []
    for i, element_id in enumerate(cluster.element_ids[:config.max_member_summaries]):
        summary = member_summaries.get(element_id, "")
        name = cluster.element_names[i] if i < len(cluster.element_names) else "unknown"
        if summary:
            summaries_text.append(f"- {name}(): {summary}")

    if not summaries_text:
        # Fallback if no summaries available
        summaries_text = [f"- {name}()" for name in cluster.element_names[:10]]

    prompt = FEATURE_SUMMARY_PROMPT.format(
        label=cluster.label or f"cluster_{cluster.cluster_id}",
        member_count=cluster.size,
        member_summaries="\n".join(summaries_text),
    )

    raw_summary = ollama.generate(
        prompt=prompt,
        temperature=config.summarize_temperature,
        max_tokens=config.summarize_max_tokens,
        timeout=config.summarize_timeout,
    )

    # Clean summary
    summary = raw_summary.strip()
    if summary.lower().startswith("summary:"):
        summary = summary[8:].strip()
    if not summary.endswith("."):
        summary += "."

    return summary


def _embed_feature(
    summary: str,
    ollama_embed: OllamaEmbedClient,
    config: FeatureProcessingConfig,
) -> list[float]:
    """Generate embedding for feature summary.

    Args:
        summary: Feature summary text.
        ollama_embed: Ollama embedding client.
        config: Processing configuration.

    Returns:
        Embedding vector.
    """
    embedding = ollama_embed.embed_single(summary, timeout=config.embed_timeout)

    if not validate_vector(embedding, config.embed_dimensions):
        raise ValueError(
            f"Invalid embedding: expected {config.embed_dimensions} dims, "
            f"got {len(embedding)}"
        )

    return normalize_vector(embedding)


def _process_single_feature(
    cluster: ClusterResult,
    member_summaries: dict[str, str],
    scope: str,
    repository: str,
    username: str,
    es_repo: ElasticsearchRepository,
    ollama: OllamaClient,
    ollama_embed: OllamaEmbedClient,
    config: FeatureProcessingConfig,
    worker_id: int,
    worker_status: FeatureWorkerStatus,
    on_status_change: Callable[[], None] | None = None,
) -> ProcessedFeature:
    """Process a single feature: summarize -> embed -> index.

    Args:
        cluster: Cluster to process.
        member_summaries: Summaries of member elements.
        scope: Repository scope.
        repository: Repository name.
        username: Username/branch.
        es_repo: Elasticsearch repository.
        ollama: Ollama client for summarization.
        ollama_embed: Ollama client for embeddings.
        config: Processing configuration.
        worker_id: Worker thread ID.
        worker_status: Status tracker for workers.
        on_status_change: Optional callback when worker status changes.

    Returns:
        ProcessedFeature with timing info and success/error status.
    """
    summarize_time = 0.0
    embed_time = 0.0

    label = cluster.label or f"cluster_{cluster.cluster_id}"
    display_name = f"{label} ({cluster.size} members)"

    def update_status(stage: str, model: str = "") -> None:
        worker_status.set(worker_id, display_name, stage, model)
        if on_status_change:
            on_status_change()

    try:
        # Step 1: Generate feature summary
        update_status("summarizing", config.summarize_model)
        api_start = time.time()
        summary = _generate_feature_summary(cluster, member_summaries, ollama, config)
        summarize_time = time.time() - api_start

        # Step 2: Embed feature summary
        update_status("embedding")
        api_start = time.time()
        embedding = _embed_feature(summary, ollama_embed, config)
        embed_time = time.time() - api_start

        # Step 3: Index feature document
        update_status("indexing")
        feature_id = f"{scope}:{repository}:{username}:feature:{label}:{cluster.cluster_id}"
        es_repo.index_feature(
            feature_id=feature_id,
            scope=scope,
            repository=repository,
            username=username,
            cluster_id=str(cluster.cluster_id),
            label=label,
            summary=summary,
            embedding=embedding,
            member_ids=cluster.element_ids,
        )

        worker_status.clear(worker_id)

        return ProcessedFeature(
            cluster_id=cluster.cluster_id,
            success=True,
            summarize_time=summarize_time,
            embed_time=embed_time,
            label=label,
            summary=summary,
        )

    except Exception as e:
        worker_status.clear(worker_id)
        return ProcessedFeature(
            cluster_id=cluster.cluster_id,
            success=False,
            summarize_time=summarize_time,
            embed_time=embed_time,
            error=str(e),
        )


def process_features(
    clustering_result: ClusteringResult,
    scope: str,
    repository: str,
    username: str,
    es_repo: ElasticsearchRepository,
    config: FeatureProcessingConfig | None = None,
    on_progress: Callable[[FeatureProgressState], None] | None = None,
    on_status_change: Callable[[], None] | None = None,
    worker_status: FeatureWorkerStatus | None = None,
    timing_stats: FeatureTimingStats | None = None,
    magaldi_config: "MagaldiConfig | None" = None,
) -> dict[str, Any]:
    """Process features: summarize -> embed -> index (parallel).

    Args:
        clustering_result: Result from HDBSCAN clustering.
        scope: Repository scope.
        repository: Repository name.
        username: Username/branch.
        es_repo: Elasticsearch repository.
        config: Processing configuration.
        on_progress: Optional callback for progress updates.
        on_status_change: Optional callback when worker status changes.
        worker_status: Optional shared worker status tracker.
        timing_stats: Optional shared timing stats.
        magaldi_config: Optional config for Redis job tracking.

    Returns:
        Dict with processing results.
    """
    if config is None:
        config = FeatureProcessingConfig()

    clusters = clustering_result.clusters
    if not clusters:
        return {"processed": 0, "failed": 0}

    # Initialize Redis job tracking if config provided
    redis_repo = None
    if magaldi_config:
        from shared.db.redis import RedisFeatureJobRepository
        redis_repo = RedisFeatureJobRepository(magaldi_config)
        # Clear stale feature queue data
        client = redis_repo._get_client()
        for key_type in ["jobs", "running", "queue"]:
            client.delete(f"magaldi:feature:{key_type}:{scope}:{repository}:{username}")
        # Add all clusters as pending jobs
        for cluster in clusters:
            label = cluster.label or f"cluster_{cluster.cluster_id}"
            feature_id = f"{scope}:{repository}:{username}:feature:{label}:{cluster.cluster_id}"
            redis_repo.add_job(feature_id, scope, repository, username, label)

    # Delete existing feature documents
    es_repo.delete_features(scope, repository, username)

    # Fetch all member summaries in batch
    all_member_ids = []
    for cluster in clusters:
        all_member_ids.extend(cluster.element_ids)

    member_summaries = es_repo.get_summaries_batch(all_member_ids)

    # Initialize Ollama clients
    ollama = OllamaClient(config.ollama_url, config.summarize_model)
    ollama_embed = OllamaEmbedClient(config.ollama_url, config.embed_model)

    # Initialize tracking structures
    if timing_stats is None:
        timing_stats = FeatureTimingStats()
    timing_stats.phase_start = time.time()
    if worker_status is None:
        worker_status = FeatureWorkerStatus()

    total = len(clusters)
    completed_count = 0
    failed_count = 0
    errors: list[str] = []
    processed_features: dict[int, tuple[str, str]] = {}  # cluster_id -> (label, summary)

    # Worker ID pool
    available_worker_ids: list[int] = list(range(config.num_workers))
    worker_id_lock = threading.Lock()

    def acquire_worker_id() -> int:
        with worker_id_lock:
            if available_worker_ids:
                return available_worker_ids.pop(0)
            return 0

    def release_worker_id(wid: int) -> None:
        with worker_id_lock:
            if wid not in available_worker_ids:
                available_worker_ids.append(wid)

    def process_wrapper(cluster: ClusterResult) -> ProcessedFeature:
        wid = acquire_worker_id()
        label = cluster.label or f"cluster_{cluster.cluster_id}"
        feature_id = f"{scope}:{repository}:{username}:feature:{label}:{cluster.cluster_id}"

        # Mark as running in Redis
        if redis_repo:
            redis_repo.mark_running(feature_id, scope, repository, username)

        try:
            result = _process_single_feature(
                cluster=cluster,
                member_summaries=member_summaries,
                scope=scope,
                repository=repository,
                username=username,
                es_repo=es_repo,
                ollama=ollama,
                ollama_embed=ollama_embed,
                config=config,
                worker_id=wid,
                worker_status=worker_status,
                on_status_change=on_status_change,
            )
            # Mark as completed or failed in Redis based on result
            if redis_repo:
                if result.success:
                    redis_repo.mark_completed(feature_id, scope, repository, username)
                else:
                    redis_repo.mark_failed(feature_id, scope, repository, username, result.error or "Unknown error")
            return result
        finally:
            release_worker_id(wid)

    # Process features in parallel
    executor = ThreadPoolExecutor(max_workers=config.num_workers)
    future_to_cluster: dict = {}

    try:
        # Submit all features for processing
        for cluster in clusters:
            future = executor.submit(process_wrapper, cluster)
            future_to_cluster[future] = cluster

        while future_to_cluster:
            done, _ = wait(future_to_cluster.keys(), return_when=FIRST_COMPLETED)

            for future in done:
                cluster = future_to_cluster.pop(future)
                processed = future.result()

                timing_stats.record(processed.summarize_time, processed.embed_time)

                if processed.success:
                    completed_count += 1
                    # Track processed feature data for sub-feature processing
                    processed_features[processed.cluster_id] = (processed.label, processed.summary)
                else:
                    failed_count += 1
                    errors.append(f"Feature {cluster.label}: {processed.error}")

                if on_progress:
                    progress_state = FeatureProgressState(
                        total=total,
                        completed=completed_count,
                        failed=failed_count,
                        timing=timing_stats,
                        workers=worker_status,
                        num_workers=config.num_workers,
                    )
                    on_progress(progress_state)

    except KeyboardInterrupt:
        for future in future_to_cluster:
            future.cancel()
        for wid in range(config.num_workers):
            worker_status.clear(wid)
        raise
    else:
        executor.shutdown(wait=True)

    return {
        "processed": completed_count,
        "failed": failed_count,
        "errors": errors,
        "elapsed": timing_stats.elapsed,
        "avg_summarize": timing_stats.avg_summarize_time,
        "avg_embed": timing_stats.avg_embed_time,
        "processed_features": processed_features,  # cluster_id -> (label, summary)
    }


# =============================================================================
# SUBFEATURE SUMMARY PROMPT
# =============================================================================


SUBFEATURE_SUMMARY_PROMPT = """You are analyzing a sub-group of related functions/methods within a larger feature.

Parent feature: {parent_label}
Parent feature description: {parent_summary}

This sub-group contains {member_count} related functions within the parent feature.

Member function summaries:
{member_summaries}

Write a 1-2 sentence summary describing what this specific sub-group does within the context of the parent feature.

Summary:"""


# =============================================================================
# SUBFEATURE PROCESSING
# =============================================================================


def _generate_subfeature_summary(
    cluster: "ClusterResult",
    member_summaries: dict[str, str],
    parent_label: str,
    parent_summary: str,
    ollama: OllamaClient,
    config: FeatureProcessingConfig,
) -> str:
    """Generate summary for a subfeature based on member summaries.

    Args:
        cluster: Cluster result with member info.
        member_summaries: Dict mapping element_id to summary.
        parent_label: Label of the parent feature.
        parent_summary: Summary of the parent feature.
        ollama: Ollama client for LLM.
        config: Processing configuration.

    Returns:
        Generated subfeature summary.
    """
    # Build member summaries text
    summaries_text = []
    for i, element_id in enumerate(cluster.element_ids[:config.max_member_summaries]):
        summary = member_summaries.get(element_id, "")
        name = cluster.element_names[i] if i < len(cluster.element_names) else "unknown"
        if summary:
            summaries_text.append(f"- {name}(): {summary}")

    if not summaries_text:
        # Fallback if no summaries available
        summaries_text = [f"- {name}()" for name in cluster.element_names[:10]]

    prompt = SUBFEATURE_SUMMARY_PROMPT.format(
        parent_label=parent_label,
        parent_summary=parent_summary,
        member_count=cluster.size,
        member_summaries="\n".join(summaries_text),
    )

    raw_summary = ollama.generate(
        prompt=prompt,
        temperature=config.summarize_temperature,
        max_tokens=config.summarize_max_tokens,
        timeout=config.summarize_timeout,
    )

    # Clean summary
    summary = raw_summary.strip()
    if summary.lower().startswith("summary:"):
        summary = summary[8:].strip()
    if not summary.endswith("."):
        summary += "."

    return summary


def process_subfeatures(
    clustering_result: ClusteringResult,
    processed_features: dict[int, tuple[str, str]],
    scope: str,
    repository: str,
    username: str,
    es_repo: "ElasticsearchRepository",
    config: FeatureProcessingConfig | None = None,
    subcluster_config: SubClusterConfig | None = None,
    on_progress: Callable[[SubfeatureProgressState], None] | None = None,
    magaldi_config: "MagaldiConfig | None" = None,
) -> dict[str, Any]:
    """Process subfeatures for large features (>20 members).

    Args:
        clustering_result: Result from main clustering.
        processed_features: Dict mapping cluster_id to (label, summary).
        scope: Repository scope.
        repository: Repository name.
        username: Username/branch.
        es_repo: Elasticsearch repository.
        config: Feature processing configuration.
        subcluster_config: Sub-clustering configuration.
        on_progress: Optional callback for progress updates.
        magaldi_config: Optional config for Redis job tracking.

    Returns:
        Dict with processing results.
    """
    from shared.ai.clustering.clusterer import ClusterConfig, FeatureClusterer

    if config is None:
        config = FeatureProcessingConfig()
    if subcluster_config is None:
        subcluster_config = SubClusterConfig()

    # Find features that need sub-clustering
    large_clusters = [
        c for c in clustering_result.clusters
        if c.size > subcluster_config.min_members_for_subclustering
    ]

    if not large_clusters:
        return {"subfeatures_created": 0, "parent_features_processed": 0}

    # Delete existing subfeatures
    es_repo.delete_subfeatures(scope, repository, username)

    # Initialize Ollama clients
    ollama = OllamaClient(config.ollama_url, config.summarize_model)
    ollama_embed = OllamaEmbedClient(config.ollama_url, config.embed_model)

    # Sub-clustering config
    cluster_config = ClusterConfig(
        min_cluster_size=subcluster_config.min_cluster_size,
        min_samples=subcluster_config.min_samples,
        ollama_url=config.ollama_url,
        ollama_model=config.summarize_model,
    )
    sub_clusterer = FeatureClusterer(cluster_config)

    subfeatures_created = 0
    parent_features_processed = 0
    errors: list[str] = []
    start_time = time.time()

    def report_progress(current_parent: str, stage: str, total_subs: int = 0) -> None:
        if on_progress:
            on_progress(SubfeatureProgressState(
                total_parent_features=len(large_clusters),
                completed_parent_features=parent_features_processed,
                total_subfeatures=total_subs,
                completed_subfeatures=subfeatures_created,
                failed=len(errors),
                current_parent=current_parent,
                current_stage=stage,
                elapsed=time.time() - start_time,
            ))

    for cluster in large_clusters:
        parent_label, parent_summary = processed_features.get(
            cluster.cluster_id, (cluster.label or f"cluster_{cluster.cluster_id}", "")
        )

        report_progress(parent_label, "fetching")

        # Fetch embeddings for cluster members
        member_docs = es_repo.get_documents_batch(cluster.element_ids)
        elements_with_embeddings = []

        for element_id in cluster.element_ids:
            doc = member_docs.get(element_id)
            if doc and doc.get("embedding"):
                idx = cluster.element_ids.index(element_id)
                name = cluster.element_names[idx] if idx < len(cluster.element_names) else ""
                elements_with_embeddings.append({
                    "element_id": element_id,
                    "embedding": doc["embedding"],
                    "name": name,
                    "element_type": doc.get("element_type", "function"),
                })

        if len(elements_with_embeddings) < subcluster_config.min_cluster_size:
            continue

        try:
            # Run sub-clustering
            report_progress(parent_label, "clustering")
            sub_result = sub_clusterer.cluster(elements_with_embeddings)

            if not sub_result.clusters:
                continue

            # Label sub-clusters
            report_progress(parent_label, "labeling", len(sub_result.clusters))
            sub_result = sub_clusterer.label_clusters(sub_result)

            # Fetch member summaries for all sub-cluster members
            all_sub_member_ids = []
            for sub_cluster in sub_result.clusters:
                all_sub_member_ids.extend(sub_cluster.element_ids)
            member_summaries = es_repo.get_summaries_batch(all_sub_member_ids)

            # Process each sub-cluster
            for sub_cluster in sub_result.clusters:
                try:
                    sub_label = sub_cluster.label or f"subcluster_{sub_cluster.cluster_id}"
                    report_progress(parent_label, f"summarizing: {sub_label}", len(sub_result.clusters))

                    # Generate subfeature summary
                    summary = _generate_subfeature_summary(
                        sub_cluster, member_summaries,
                        parent_label, parent_summary,
                        ollama, config,
                    )

                    report_progress(parent_label, f"embedding: {sub_label}", len(sub_result.clusters))
                    # Embed subfeature
                    embedding = ollama_embed.embed_single(summary, timeout=config.embed_timeout)
                    if validate_vector(embedding, config.embed_dimensions):
                        embedding = normalize_vector(embedding)
                    else:
                        embedding = None

                    # Build subfeature ID
                    subfeature_id = f"{scope}:{repository}:{username}:subfeature:{parent_label}:{sub_label}:{sub_cluster.cluster_id}"

                    report_progress(parent_label, f"indexing: {sub_label}", len(sub_result.clusters))
                    # Index subfeature
                    es_repo.index_subfeature(
                        subfeature_id=subfeature_id,
                        scope=scope,
                        repository=repository,
                        username=username,
                        cluster_id=str(sub_cluster.cluster_id),
                        label=sub_label,
                        summary=summary,
                        embedding=embedding,
                        member_ids=sub_cluster.element_ids,
                        parent_feature_label=parent_label,
                        parent_feature_summary=parent_summary,
                    )

                    subfeatures_created += 1
                    report_progress(parent_label, "", len(sub_result.clusters))

                except Exception as e:
                    errors.append(f"Subfeature {sub_cluster.cluster_id} of {parent_label}: {e}")

            parent_features_processed += 1
            report_progress("", "", 0)

        except Exception as e:
            errors.append(f"Sub-clustering {parent_label}: {e}")

    return {
        "subfeatures_created": subfeatures_created,
        "parent_features_processed": parent_features_processed,
        "errors": errors,
    }
