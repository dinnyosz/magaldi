"""Feature processor - summarize and embed features in parallel.

Processes feature clusters:
1. Fetch member summaries from ES
2. Generate feature summary using Ollama
3. Embed feature summary
4. Index feature document to ES

Subfeature processing has been moved to subfeature_processor.py.
This module re-exports subfeature types for backwards compatibility.
"""

from __future__ import annotations

import bisect
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from shared.ai.clustering.clusterer import ClusteringResult, ClusterResult
from shared.ai.context_size import TIER_SCALING_EXPONENT
from shared.parallel_processor import ThrottleContext, ThrottleDisplayInfo, run_throttled_tier
from shared.throttling import ThroughputTracker

if TYPE_CHECKING:
    from shared.config import MagaldiConfig
from shared.ai.embedding import (
    CodeEmbeddingClient,
    normalize_vector,
    validate_vector,
)
from shared.ai.summarization import SummarizationLLMClient
from shared.db.store import Repository

# Import shared config from common module
from shared.ai.clustering.config import (
    FeatureProcessingConfig,
    SubClusterConfig,
    _get_model_display_name,
)

# Re-export subfeature types for backwards compatibility
from shared.ai.clustering.subfeature_processor import (
    ProcessedSubfeature,
    SubfeatureLabelingState,
    SubfeatureProgressState,
    SubfeatureTimingStats,
    SubfeatureWorkItem,
    SubfeatureWorkerStatus,
    SUBFEATURE_SUMMARY_PROMPT,
    SUBFEATURE_SYSTEM_PROMPT,
    SUBFEATURE_USER_PROMPT,
    _build_subfeature_prompt,
    _generate_subfeature_summary,
    _process_single_subfeature,
    process_subfeatures,
)

# Re-export config types for backwards compatibility
__all__ = [
    # Config types (from config.py)
    "FeatureProcessingConfig",
    "SubClusterConfig",
    "_get_model_display_name",
    # Feature types
    "FeatureTimingStats",
    "FeatureWorkerStatus",
    "FeatureProgressState",
    "ProcessedFeature",
    "FEATURE_SYSTEM_PROMPT",
    "FEATURE_USER_PROMPT",
    "FEATURE_SUMMARY_PROMPT",
    "_build_feature_prompt",
    "_generate_feature_summary",
    "_embed_feature",
    "_process_single_feature",
    "process_features",
    # Subfeature types (re-exported from subfeature_processor.py)
    "SubfeatureTimingStats",
    "SubfeatureWorkerStatus",
    "SubfeatureProgressState",
    "SubfeatureLabelingState",
    "SubfeatureWorkItem",
    "ProcessedSubfeature",
    "SUBFEATURE_SYSTEM_PROMPT",
    "SUBFEATURE_USER_PROMPT",
    "SUBFEATURE_SUMMARY_PROMPT",
    "_build_subfeature_prompt",
    "_generate_subfeature_summary",
    "_process_single_subfeature",
    "process_subfeatures",
]


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class FeatureTimingStats:
    """Thread-safe timing statistics for feature processing."""

    _lock: threading.Lock = field(default_factory=threading.Lock)
    phase_start: float = 0.0

    total_summarize_time: float = 0.0
    total_embed_time: float = 0.0
    summarize_count: int = 0
    embed_count: int = 0

    # Per-tier tracking for accurate ETA
    total_time_by_tier: dict[int, float] = field(default_factory=dict)
    count_by_tier: dict[int, int] = field(default_factory=dict)
    totals_by_tier: dict[int, int] = field(default_factory=dict)

    # Throughput tracker for throttling (5 min window like other processors)
    throughput_tracker: "ThroughputTracker" = field(default_factory=lambda: ThroughputTracker(window_seconds=300.0))

    def set_totals_by_tier(self, totals: dict[int, int]) -> None:
        """Set total counts by tier for ETA calculation."""
        with self._lock:
            self.totals_by_tier = dict(totals)

    def record(self, summarize_time: float, embed_time: float, tier: int = 0) -> None:
        """Record timing for a completed feature."""
        with self._lock:
            self.total_summarize_time += summarize_time
            self.total_embed_time += embed_time
            self.summarize_count += 1
            if embed_time > 0:
                self.embed_count += 1

            # Track per-tier timing
            if tier > 0:
                total_time = summarize_time + embed_time
                self.total_time_by_tier[tier] = self.total_time_by_tier.get(tier, 0.0) + total_time
                self.count_by_tier[tier] = self.count_by_tier.get(tier, 0) + 1

    def record_task_runtime(self, runtime: float, concurrent_workers: int = 1) -> None:
        """Record task wall-clock runtime for throttling with concurrency context."""
        self.throughput_tracker.record_completion(runtime, concurrent_workers)

    def get_throughput_stats(self) -> tuple[float, float, int]:
        """Get throughput statistics for throttling."""
        return self.throughput_tracker.get_stats()

    def get_throughput_stats_with_concurrency(self) -> tuple[float, float, int, float, float]:
        """Get throughput statistics with concurrency context."""
        return self.throughput_tracker.get_stats_with_concurrency()

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

    def _get_avg_for_tier(self, tier: int, global_avg: float) -> float:
        """Get average time for a tier with fallback."""
        # Exact tier match
        if tier in self.count_by_tier and self.count_by_tier[tier] > 0:
            return self.total_time_by_tier[tier] / self.count_by_tier[tier]

        # Find closest tier with data
        tiers_with_data = [t for t in self.count_by_tier if self.count_by_tier[t] > 0]
        if tiers_with_data:
            closest = min(tiers_with_data, key=lambda t: abs(t - tier))
            base_avg = self.total_time_by_tier[closest] / self.count_by_tier[closest]
            # Scale by tier ratio - use empirically-derived exponent for sub-linear scaling
            tier_ratio = (tier / closest) ** TIER_SCALING_EXPONENT if closest > 0 else 1.0
            return base_avg * tier_ratio

        return global_avg

    def eta_seconds(self, completed: int, total: int, num_workers: int = 1) -> float | None:
        """Calculate ETA based on per-tier average times."""
        with self._lock:
            if self.summarize_count == 0:
                return None

            global_avg = (self.total_summarize_time + self.total_embed_time) / self.summarize_count

            # Use tier-aware calculation if available
            if self.totals_by_tier:
                total_work_time = 0.0
                for tier, tot in self.totals_by_tier.items():
                    done = self.count_by_tier.get(tier, 0)
                    remaining = tot - done
                    if remaining > 0:
                        avg = self._get_avg_for_tier(tier, global_avg)
                        total_work_time += remaining * avg
                if total_work_time > 0:
                    return total_work_time / max(num_workers, 1)

            # Fallback to simple calculation
            remaining = total - completed
            if remaining <= 0:
                return 0.0
            return (remaining * global_avg) / max(num_workers, 1)

    def get_eta_breakdown_with_avg(self, num_workers: int = 1) -> list[tuple[str, int, float, bool, int, int]]:
        """Get average time per tier for display.

        Returns:
            List of (type, tier, avg_seconds, is_fallback, done, total) tuples,
            matching the format used by processor.py TimingStats.
            Type is always "feature" for this class.
        """
        with self._lock:
            if not self.totals_by_tier:
                return []

            global_avg = (self.total_summarize_time + self.total_embed_time) / self.summarize_count if self.summarize_count > 0 else 0.0

            breakdown = []
            for tier, total in self.totals_by_tier.items():
                done = self.count_by_tier.get(tier, 0)
                avg = self._get_avg_for_tier(tier, global_avg)
                # is_fallback: True if we don't have actual data for this tier
                is_fallback = tier not in self.count_by_tier or self.count_by_tier[tier] == 0
                breakdown.append(("feature", tier, avg, is_fallback, done, total))

            # Sort by tier descending (largest first)
            breakdown.sort(key=lambda x: -x[1])
            return breakdown


@dataclass
class FeatureWorkerStatus:
    """Track what each worker is doing."""

    _lock: threading.Lock = field(default_factory=threading.Lock)
    _status: dict[int, tuple[str, str, str, str, float]] = field(default_factory=dict)  # worker_id -> (feature_name, stage, model, ctx_size, start_time)

    def set(self, worker_id: int, feature_name: str, stage: str, model: str = "", ctx_size: str = "", start_time: float = 0.0) -> None:
        with self._lock:
            self._status[worker_id] = (feature_name, stage, model, ctx_size, start_time)

    def clear(self, worker_id: int) -> None:
        with self._lock:
            self._status.pop(worker_id, None)

    def get_all(self) -> dict[int, tuple[str, str, str, str, float]]:
        with self._lock:
            return dict(self._status)

    def get_max_active_runtime(self) -> float:
        """Get maximum runtime of currently active workers."""
        with self._lock:
            if not self._status:
                return 0.0
            now = time.time()
            # start_time is index 4 in the tuple
            return max(now - v[4] for v in self._status.values() if v[4] > 0)


@dataclass
class FeatureProgressState:
    """Combined state for display updates."""

    total: int
    completed: int
    failed: int
    timing: FeatureTimingStats
    workers: FeatureWorkerStatus
    num_workers: int = 1
    allowed_workers: int = 0  # Current throttle-allowed workers (0 = use num_workers)
    current_max: float = 0.0  # Max runtime of active workers (for throttle display)
    avg_base_time: float = 0.0  # Historical base time per worker
    completion_count: int = 0  # Number of completions used for avg_base_time


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
# FEATURE SUMMARY PROMPTS (Optimized for Prefix Caching)
# =============================================================================
# System message is STATIC and gets cached by Ollama's KV cache.
# User message contains VARIABLE content (feature name, member summaries).

FEATURE_SYSTEM_PROMPT = """You are analyzing a code feature containing related functions/methods.
Based on the member function summaries provided, describe what this feature does.

Write a 6-8 sentence summary describing:
1. The overall purpose of this feature
2. The key operations it provides
3. How the member functions work together

Write ONLY the summary. No reasoning, explanations, or bullet points.
Start directly with what it provides - never start with "This feature...", "The X feature...", or similar."""

FEATURE_USER_PROMPT = """Feature name: {label}
Number of members: {member_count}

Member function summaries:
{member_summaries}"""

# Legacy single-prompt template (kept for backwards compatibility)
FEATURE_SUMMARY_PROMPT = """You are analyzing a code feature containing related functions/methods.
Based on the member function summaries below, describe what this feature does.

Feature name: {label}
Number of members: {member_count}

Member function summaries:
{member_summaries}

Write a 6-8 sentence summary describing:
1. The overall purpose of this feature
2. The key operations it provides
3. How the member functions work together

Start directly with what it provides - never start with "This feature...", "The X feature...", or similar.

Summary:"""


# =============================================================================
# FEATURE PROCESSING
# =============================================================================


def _build_feature_prompt(
    cluster: ClusterResult,
    member_summaries: dict[str, str],
    config: FeatureProcessingConfig,
) -> tuple[list[dict[str, str]], int]:
    """Build prompt messages and compute context size for feature summary.

    Args:
        cluster: Cluster result with member info.
        member_summaries: Dict mapping element_id to summary.
        config: Processing configuration.

    Returns:
        Tuple of (messages list, num_ctx).
    """
    from shared.ai.context_size import compute_aggregation_num_ctx

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

    # Build messages optimized for prefix caching
    user_content = FEATURE_USER_PROMPT.format(
        label=cluster.label or f"cluster_{cluster.cluster_id}",
        member_count=cluster.size,
        member_summaries="\n".join(summaries_text),
    )

    messages = [
        {"role": "system", "content": FEATURE_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    # Compute dynamic context size based on prompt length
    prompt_chars = len(FEATURE_SYSTEM_PROMPT) + len(user_content)
    num_ctx = compute_aggregation_num_ctx(prompt_chars, task_type="feature")

    return messages, num_ctx


def _generate_feature_summary(
    cluster: ClusterResult,
    member_summaries: dict[str, str],
    llm_client: SummarizationLLMClient,
    config: FeatureProcessingConfig,
    messages: list[dict[str, str]] | None = None,
    num_ctx: int | None = None,
) -> tuple[str, int]:
    """Generate summary for a feature based on member summaries.

    Uses message-based format (system + user) optimized for Ollama's KV cache
    prefix caching. The system message contains static instructions that get
    cached, while the user message has variable content.

    Args:
        cluster: Cluster result with member info.
        member_summaries: Dict mapping element_id to summary.
        llm_client: LLM client for text generation.
        config: Processing configuration.
        messages: Pre-built messages (optional, will be built if not provided).
        num_ctx: Pre-computed context size (optional, will be computed if not provided).

    Returns:
        Tuple of (generated feature summary, num_ctx used).
    """
    # Build prompt if not provided
    if messages is None or num_ctx is None:
        messages, num_ctx = _build_feature_prompt(cluster, member_summaries, config)

    raw_summary = llm_client.generate_from_messages(
        messages=messages,
        temperature=config.summarize_temperature,
        top_p=config.summarize_top_p,
        max_tokens=config.summarize_max_tokens,
        timeout=config.summarize_timeout,
        num_ctx=num_ctx,
    )

    # Clean summary
    summary = raw_summary.strip()
    if summary.lower().startswith("summary:"):
        summary = summary[8:].strip()
    if not summary.endswith("."):
        summary += "."

    return summary, num_ctx


def _embed_feature(
    summary: str,
    embed_client: CodeEmbeddingClient,
    config: FeatureProcessingConfig,
) -> list[float]:
    """Generate embedding for feature summary.

    Args:
        summary: Feature summary text.
        embed_client: Embedding client.
        config: Processing configuration.

    Returns:
        Embedding vector.
    """
    embedding = embed_client.embed_single(summary, timeout=config.embed_timeout)

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
    repo: Repository,
    llm_client: SummarizationLLMClient,
    embed_client: CodeEmbeddingClient,
    config: FeatureProcessingConfig,
    worker_id: int,
    worker_status: FeatureWorkerStatus,
    on_status_change: Callable[[], None] | None = None,
    cluster_id_to_label: dict[int, str] | None = None,
) -> ProcessedFeature:
    """Process a single feature: summarize -> embed -> index.

    Args:
        cluster: Cluster to process.
        member_summaries: Summaries of member elements.
        scope: Repository scope.
        repository: Repository name.
        username: Username/branch.
        repo: Search repository.
        llm_client: LLM client for summarization.
        embed_client: Embedding client.
        config: Processing configuration.
        worker_id: Worker thread ID.
        worker_status: Status tracker for workers.
        on_status_change: Optional callback when worker status changes.
        cluster_id_to_label: Mapping of cluster IDs to their labels (for connected features).

    Returns:
        ProcessedFeature with timing info and success/error status.
    """
    summarize_time = 0.0
    embed_time = 0.0

    label = cluster.label or f"cluster_{cluster.cluster_id}"
    display_name = f"{label} ({cluster.size} members)"

    # Track stage start time for elapsed display
    stage_start_time = 0.0

    def update_status(stage: str, model: str = "", ctx_size: str = "") -> None:
        nonlocal stage_start_time
        stage_start_time = time.time()
        worker_status.set(worker_id, display_name, stage, model, ctx_size, stage_start_time)
        if on_status_change:
            on_status_change()

    def format_ctx(num_ctx: int) -> str:
        return f"{num_ctx // 1024}K" if num_ctx >= 1024 else str(num_ctx)

    try:
        # Step 1: Generate feature summary
        # Build prompt and compute context size FIRST so we can display it during LLM call
        messages, num_ctx = _build_feature_prompt(cluster, member_summaries, config)
        # Display tiered model name for Ollama (e.g., "qwen3:4b-instruct-4k")
        model_display = _get_model_display_name(config.summarize_model, config.provider, num_ctx)
        update_status("summarizing", model_display, format_ctx(num_ctx))
        api_start = time.time()
        summary, _ = _generate_feature_summary(cluster, member_summaries, llm_client, config, messages, num_ctx)
        summarize_time = time.time() - api_start

        # Step 2: Embed feature summary
        update_status("embedding")
        api_start = time.time()
        embedding = _embed_feature(summary, embed_client, config)
        embed_time = time.time() - api_start

        # Step 3: Index feature document
        update_status("indexing")
        feature_id = f"{scope}:{repository}:{username}:feature:{label}:{cluster.cluster_id}"

        # Build connected_features from cluster.connected_clusters (from soft clustering)
        connected_features: list[dict[str, Any]] | None = None
        if cluster.connected_clusters and cluster_id_to_label:
            connected_features = []
            for conn in cluster.connected_clusters:
                conn_label = cluster_id_to_label.get(conn.cluster_id)
                if conn_label:
                    conn_feature_id = f"{scope}:{repository}:{username}:feature:{conn_label}:{conn.cluster_id}"
                    connected_features.append({
                        "feature_id": conn_feature_id,
                        "label": conn_label,
                        "affinity": conn.affinity,
                    })

        repo.index_feature(
            feature_id=feature_id,
            scope=scope,
            repository=repository,
            username=username,
            cluster_id=str(cluster.cluster_id),
            label=label,
            summary=summary,
            embedding=embedding,
            member_ids=cluster.element_ids,
            connected_features=connected_features,
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
    repo: Repository,
    config: FeatureProcessingConfig | None = None,
    on_progress: Callable[[FeatureProgressState], None] | None = None,
    on_status_change: Callable[[], None] | None = None,
    worker_status: FeatureWorkerStatus | None = None,
    timing_stats: FeatureTimingStats | None = None,
    magaldi_config: "MagaldiConfig | None" = None,
    on_tier_distribution: Callable[[dict[int, int]], None] | None = None,
) -> dict[str, Any]:
    """Process features: summarize -> embed -> index (parallel).

    Args:
        clustering_result: Result from HDBSCAN clustering.
        scope: Repository scope.
        repository: Repository name.
        username: Username/branch.
        repo: Search repository.
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
    repo.delete_features(scope, repository, username)

    # Fetch all member summaries in batch
    all_member_ids = []
    for cluster in clusters:
        all_member_ids.extend(cluster.element_ids)

    member_summaries = repo.get_summaries_batch(all_member_ids)

    # Initialize LLM clients
    llm_client = SummarizationLLMClient(
        url=config.api_base,
        model=config.summarize_model,
        provider=config.provider,
        api_key=config.api_key,
    )
    embed_client = CodeEmbeddingClient(
        url=config.api_base,
        model=config.embed_model,
        provider=config.provider,
        api_key=config.api_key,
    )

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

    # Build cluster_id -> label mapping for connected features (soft clustering)
    cluster_id_to_label: dict[int, str] = {
        c.cluster_id: c.label or f"cluster_{c.cluster_id}"
        for c in clusters
    }

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
                # Insert in sorted position to keep lower IDs at front
                bisect.insort(available_worker_ids, wid)

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
                repo=repo,
                llm_client=llm_client,
                embed_client=embed_client,
                config=config,
                worker_id=wid,
                worker_status=worker_status,
                on_status_change=on_status_change,
                cluster_id_to_label=cluster_id_to_label,
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

    # Pre-compute context tier for each cluster based on prompt size
    from shared.ai.context_size import TIER_MAX_WORKERS, compute_aggregation_num_ctx, iter_by_tier

    def estimate_cluster_tier(cluster: ClusterResult) -> int:
        """Estimate context tier for a cluster based on member summaries."""
        summaries_text = []
        for i, element_id in enumerate(cluster.element_ids[:config.max_member_summaries]):
            summary = member_summaries.get(element_id, "")
            name = cluster.element_names[i] if i < len(cluster.element_names) else "unknown"
            if summary:
                summaries_text.append(f"- {name}(): {summary}")
        if not summaries_text:
            summaries_text = [f"- {name}()" for name in cluster.element_names[:10]]

        user_content = FEATURE_USER_PROMPT.format(
            label=cluster.label or f"cluster_{cluster.cluster_id}",
            member_count=cluster.size,
            member_summaries="\n".join(summaries_text),
        )
        prompt_chars = len(FEATURE_SYSTEM_PROMPT) + len(user_content)
        return compute_aggregation_num_ctx(prompt_chars, task_type="feature")

    # Group clusters by tier and process each tier with appropriate max_workers
    tier_groups = iter_by_tier(clusters, estimate_cluster_tier)

    # Build cluster_id -> tier mapping for tracking
    cluster_to_tier: dict[int, int] = {}
    tier_counts: dict[int, int] = {}
    for tier, _, tier_clusters in tier_groups:
        tier_counts[tier] = len(tier_clusters)
        for cluster in tier_clusters:
            cluster_to_tier[cluster.cluster_id] = tier

    # Set tier totals for ETA calculation
    timing_stats.set_totals_by_tier(tier_counts)

    # Report tier distribution if callback provided
    if on_tier_distribution:
        on_tier_distribution(tier_counts)

    # Handle workers=0 (auto) - will use tier-specific limits
    max_pool_workers = config.num_workers if config.num_workers > 0 else max(TIER_MAX_WORKERS.values())

    # Mutable state for callbacks
    state = {"completed": 0, "failed": 0, "current_workers": max_pool_workers}

    def on_complete(cluster: ClusterResult, processed: ProcessedFeature, avg_workers: float) -> None:
        """Handle completed feature processing."""
        nonlocal processed_features, errors
        tier = cluster_to_tier.get(processed.cluster_id, 0)

        # Record for throttling
        wall_time = processed.summarize_time + processed.embed_time
        timing_stats.record_task_runtime(wall_time, avg_workers)
        timing_stats.record(processed.summarize_time, processed.embed_time, tier=tier)

        if processed.success:
            state["completed"] += 1
            processed_features[processed.cluster_id] = (processed.label, processed.summary)
        else:
            state["failed"] += 1
            errors.append(f"Feature {cluster.label}: {processed.error}")

    def on_tick(throttle_info: "ThrottleDisplayInfo") -> None:
        """Update progress display."""
        if on_progress:
            progress_state = FeatureProgressState(
                total=total,
                completed=state["completed"],
                failed=state["failed"],
                timing=timing_stats,
                workers=worker_status,
                num_workers=state["current_workers"],
                allowed_workers=throttle_info.allowed_workers,
                current_max=throttle_info.current_max,
                avg_base_time=throttle_info.avg_base_time,
                completion_count=throttle_info.completion_count,
            )
            on_progress(progress_state)

    # Create throttle context
    throttle_ctx = ThrottleContext(
        tier_timeout=180,
        base_workers=max_pool_workers,
        throughput_tracker=timing_stats.throughput_tracker,
    )

    try:
        for _tier, _tier_max_workers, tier_clusters in tier_groups:
            # Use full max workers like Phase 4 - time-based throttling handles scaling
            state["current_workers"] = max_pool_workers

            # Reset worker ID pool for this tier
            available_worker_ids.clear()
            available_worker_ids.extend(range(max_pool_workers))

            run_throttled_tier(
                items=list(tier_clusters),
                tier=_tier,
                effective_workers=max_pool_workers,
                process_fn=process_wrapper,
                throttle_ctx=throttle_ctx,
                get_max_runtime=worker_status.get_max_active_runtime,
                on_complete=on_complete,
                on_tick=on_tick,
            )

        # Update final counts
        completed_count = state["completed"]
        failed_count = state["failed"]

    except KeyboardInterrupt:
        if 'executor' in dir():
            executor.shutdown(wait=False, cancel_futures=True)
        for wid in range(max_pool_workers):
            worker_status.clear(wid)
        raise

    return {
        "processed": completed_count,
        "failed": failed_count,
        "errors": errors,
        "elapsed": timing_stats.elapsed,
        "avg_summarize": timing_stats.avg_summarize_time,
        "avg_embed": timing_stats.avg_embed_time,
        "processed_features": processed_features,  # cluster_id -> (label, summary)
    }
