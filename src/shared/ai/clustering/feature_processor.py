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
class ProcessedFeature:
    """Result from processing a single feature."""

    cluster_id: int
    success: bool
    summarize_time: float
    embed_time: float
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
    }
