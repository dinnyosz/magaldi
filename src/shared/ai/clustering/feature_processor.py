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
from shared.ai.context_size import TIER_TIMEOUTS
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
from shared.db.elasticsearch import ElasticsearchRepository


def _get_model_display_name(model_name: str, provider: str, num_ctx: int) -> str:
    """Get the display name for a model, including tier suffix for Ollama models.

    Args:
        model_name: Base model name.
        provider: LLM provider (ollama, openai, etc.).
        num_ctx: Context size being used.

    Returns:
        Model name with tier suffix for Ollama models (e.g., "qwen3:4b-instruct-4k"),
        or the original name for other providers.
    """
    if provider == "ollama":
        from shared.ai.ollama_models import get_tiered_model_name

        return get_tiered_model_name(model_name, num_ctx)
    return model_name


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class FeatureProcessingConfig:
    """Configuration for feature processing."""

    summarize_model: str = "qwen3:4b-instruct"
    embed_model: str = "snowflake-arctic-embed2"
    api_base: str = "http://localhost:11434"  # API base URL (for Ollama or custom endpoints)
    provider: str = "ollama"  # LLM provider: ollama, openai, anthropic, etc.
    api_key: str | None = None  # API key for cloud providers

    # Summarization settings (based on arxiv.org/html/2507.03160v2)
    summarize_temperature: float = 0.2
    summarize_top_p: float = 0.95
    summarize_max_tokens: int = 512  # Longer for feature summaries
    summarize_timeout: int = 90

    # Embedding settings
    embed_dimensions: int = 1024
    embed_timeout: int = 30

    # Parallel processing
    num_workers: int = 4

    # Feature summary settings
    max_member_summaries: int = 20  # Max member summaries to include in prompt

    # Context size for aggregation tasks (feature/subfeature summaries)
    aggregation_context_size: int = 16384  # Fixed large context for aggregation tasks


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

    # Per-tier tracking for accurate ETA
    total_time_by_tier: dict[int, float] = field(default_factory=dict)
    count_by_tier: dict[int, int] = field(default_factory=dict)
    totals_by_tier: dict[int, int] = field(default_factory=dict)

    # Throughput tracker for throttling
    throughput_tracker: "ThroughputTracker" = field(default_factory=lambda: ThroughputTracker())

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
            # Scale by tier ratio
            tier_ratio = tier / closest if closest > 0 else 1.0
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


@dataclass
class SubfeatureTimingStats:
    """Thread-safe timing statistics for subfeature processing."""

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

    # Throughput tracker for throttling
    throughput_tracker: ThroughputTracker = field(default_factory=ThroughputTracker)

    def set_totals_by_tier(self, totals: dict[int, int]) -> None:
        """Set total counts by tier for ETA calculation."""
        with self._lock:
            self.totals_by_tier = dict(totals)

    def record(self, summarize_time: float, embed_time: float, tier: int = 0) -> None:
        """Record timing for a completed subfeature."""
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
        with self._lock:
            return self.total_summarize_time / self.summarize_count if self.summarize_count > 0 else 0.0

    @property
    def avg_embed_time(self) -> float:
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
            # Scale by tier ratio
            tier_ratio = tier / closest if closest > 0 else 1.0
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
            Type is always "subfeature" for this class.
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
                breakdown.append(("subfeature", tier, avg, is_fallback, done, total))

            # Sort by tier descending (largest first)
            breakdown.sort(key=lambda x: -x[1])
            return breakdown


@dataclass
class SubfeatureWorkerStatus:
    """Track what each worker is doing for subfeature processing."""

    _lock: threading.Lock = field(default_factory=threading.Lock)
    _status: dict[int, tuple[str, str, str, str, str, float]] = field(default_factory=dict)  # worker_id -> (parent_feature, stage, model, subfeature, ctx_size, start_time)

    def set(self, worker_id: int, parent_feature: str, stage: str, model: str = "", subfeature: str = "", ctx_size: str = "", start_time: float = 0.0) -> None:
        with self._lock:
            self._status[worker_id] = (parent_feature, stage, model, subfeature, ctx_size, start_time)

    def clear(self, worker_id: int) -> None:
        with self._lock:
            self._status.pop(worker_id, None)

    def get_all(self) -> dict[int, tuple[str, str, str, str, str, float]]:
        with self._lock:
            return dict(self._status)

    def get_max_active_runtime(self) -> float:
        """Get maximum runtime of currently active workers."""
        with self._lock:
            if not self._status:
                return 0.0
            now = time.time()
            # start_time is index 5 in the tuple
            return max(now - v[5] for v in self._status.values() if v[5] > 0)


@dataclass
class SubfeatureProgressState:
    """Progress state for subfeature processing."""

    total: int  # Total subfeatures to process
    completed: int
    failed: int
    timing: SubfeatureTimingStats
    workers: SubfeatureWorkerStatus
    num_workers: int = 1
    allowed_workers: int = 0  # Current throttle-allowed workers (0 = use num_workers)
    current_max: float = 0.0  # Max runtime of active workers (for throttle display)
    avg_base_time: float = 0.0  # Historical base time per worker


@dataclass
class SubfeatureLabelingState:
    """Progress state for subfeature labeling during discovery."""

    total_features: int  # Total large features to process
    features_processed: int  # Large features processed so far
    current_feature: str  # Current feature being labeled
    subclusters_labeled: int  # Total subclusters labeled so far
    model: str = ""  # Model used for labeling
    phase_start: float = 0.0  # Start time for elapsed calculation


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


@dataclass
class SubfeatureWorkItem:
    """Work item for queue-based subfeature processing."""

    parent_label: str
    parent_summary: str
    sub_cluster: Any  # ClusterResult
    member_summaries: dict[str, str]  # element_id -> summary


@dataclass
class ProcessedSubfeature:
    """Result from processing a single subfeature."""

    parent_label: str
    sub_label: str
    success: bool
    summarize_time: float
    embed_time: float
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

Write ONLY the summary. No reasoning, explanations, or bullet points."""

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
    es_repo: ElasticsearchRepository,
    llm_client: SummarizationLLMClient,
    embed_client: CodeEmbeddingClient,
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
        llm_client: LLM client for summarization.
        embed_client: Embedding client.
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
    magaldi_config: MagaldiConfig | None = None,
    on_tier_distribution: Callable[[dict[int, int]], None] | None = None,
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
                llm_client=llm_client,
                embed_client=embed_client,
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
            )
            on_progress(progress_state)

    # Create throttle context
    throttle_ctx = ThrottleContext(
        tier_timeout=180,
        base_workers=max_pool_workers,
        throughput_tracker=timing_stats.throughput_tracker,
    )

    try:
        for _tier, tier_max_workers, tier_clusters in tier_groups:
            effective_workers = min(max_pool_workers, tier_max_workers)
            state["current_workers"] = effective_workers

            # Reset worker ID pool for this tier
            available_worker_ids.clear()
            available_worker_ids.extend(range(effective_workers))

            run_throttled_tier(
                items=list(tier_clusters),
                tier=_tier,
                effective_workers=effective_workers,
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


# =============================================================================
# SUBFEATURE SUMMARY PROMPTS (Optimized for Prefix Caching)
# =============================================================================
# System message is STATIC and gets cached by Ollama's KV cache.
# User message contains VARIABLE content with parent context at TOP for cache reuse.

SUBFEATURE_SYSTEM_PROMPT = """You are analyzing a sub-group of related functions/methods within a larger feature.

Write a 6-8 sentence summary describing:
1. What this specific sub-group does within the context of the parent feature
2. How the member functions work together
3. The key operations this sub-group provides

Write ONLY the summary. No reasoning, explanations, or bullet points."""

SUBFEATURE_USER_PROMPT = """Parent feature: {parent_label}
Parent feature description: {parent_summary}

This sub-group contains {member_count} related functions within the parent feature.

Member function summaries:
{member_summaries}"""

# Legacy single-prompt template (kept for backwards compatibility)
SUBFEATURE_SUMMARY_PROMPT = """You are analyzing a sub-group of related functions/methods within a larger feature.

Parent feature: {parent_label}
Parent feature description: {parent_summary}

This sub-group contains {member_count} related functions within the parent feature.

Member function summaries:
{member_summaries}

Write a 6-8 sentence summary describing:
1. What this specific sub-group does within the context of the parent feature
2. How the member functions work together
3. The key operations this sub-group provides

Summary:"""


# =============================================================================
# SUBFEATURE PROCESSING
# =============================================================================


def _build_subfeature_prompt(
    cluster: ClusterResult,
    member_summaries: dict[str, str],
    parent_label: str,
    parent_summary: str,
    config: FeatureProcessingConfig,
) -> tuple[list[dict[str, str]], int]:
    """Build prompt messages and compute context size for subfeature summary.

    Args:
        cluster: Cluster result with member info.
        member_summaries: Dict mapping element_id to summary.
        parent_label: Label of the parent feature.
        parent_summary: Summary of the parent feature.
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
    # Parent context at top allows cache reuse for subfeatures of same parent
    user_content = SUBFEATURE_USER_PROMPT.format(
        parent_label=parent_label,
        parent_summary=parent_summary,
        member_count=cluster.size,
        member_summaries="\n".join(summaries_text),
    )

    messages = [
        {"role": "system", "content": SUBFEATURE_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    # Compute dynamic context size based on prompt length
    prompt_chars = len(SUBFEATURE_SYSTEM_PROMPT) + len(user_content)
    num_ctx = compute_aggregation_num_ctx(prompt_chars, task_type="subfeature")

    return messages, num_ctx


def _generate_subfeature_summary(
    cluster: ClusterResult,
    member_summaries: dict[str, str],
    parent_label: str,
    parent_summary: str,
    llm_client: SummarizationLLMClient,
    config: FeatureProcessingConfig,
    messages: list[dict[str, str]] | None = None,
    num_ctx: int | None = None,
) -> tuple[str, int]:
    """Generate summary for a subfeature based on member summaries.

    Uses message-based format (system + user) optimized for Ollama's KV cache
    prefix caching. The system message contains static instructions that get
    cached, while the user message has parent context at the top for cache
    reuse across subfeatures of the same parent.

    Args:
        cluster: Cluster result with member info.
        member_summaries: Dict mapping element_id to summary.
        parent_label: Label of the parent feature.
        parent_summary: Summary of the parent feature.
        llm_client: LLM client for text generation.
        config: Processing configuration.
        messages: Pre-built messages (optional, will be built if not provided).
        num_ctx: Pre-computed context size (optional, will be computed if not provided).

    Returns:
        Tuple of (generated subfeature summary, num_ctx used).
    """
    # Build prompt if not provided
    if messages is None or num_ctx is None:
        messages, num_ctx = _build_subfeature_prompt(
            cluster, member_summaries, parent_label, parent_summary, config
        )

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


def _process_single_subfeature(
    work_item: SubfeatureWorkItem,
    scope: str,
    repository: str,
    username: str,
    es_repo: ElasticsearchRepository,
    llm_client: SummarizationLLMClient,
    embed_client: CodeEmbeddingClient,
    config: FeatureProcessingConfig,
    worker_id: int,
    worker_status: SubfeatureWorkerStatus,
    on_status_change: Callable[[], None] | None = None,
) -> ProcessedSubfeature:
    """Process a single subfeature: summarize -> embed -> index."""
    sub_label = work_item.sub_cluster.label or f"subcluster_{work_item.sub_cluster.cluster_id}"
    summarize_time = 0.0
    embed_time = 0.0

    def format_ctx(num_ctx: int) -> str:
        return f"{num_ctx // 1024}K" if num_ctx >= 1024 else str(num_ctx)

    try:
        # Build prompt and compute context size FIRST so we can display it during LLM call
        messages, num_ctx = _build_subfeature_prompt(
            work_item.sub_cluster, work_item.member_summaries,
            work_item.parent_label, work_item.parent_summary, config,
        )

        # Update status: summarizing (with context size and start time)
        # Display tiered model name for Ollama (e.g., "qwen3:4b-instruct-4k")
        model_display = _get_model_display_name(config.summarize_model, config.provider, num_ctx)
        stage_start = time.time()
        worker_status.set(worker_id, work_item.parent_label, "summarize", model_display, sub_label, format_ctx(num_ctx), stage_start)
        if on_status_change:
            on_status_change()

        # Generate subfeature summary
        summ_start = time.time()
        summary, _ = _generate_subfeature_summary(
            work_item.sub_cluster, work_item.member_summaries,
            work_item.parent_label, work_item.parent_summary,
            llm_client, config, messages, num_ctx,
        )
        summarize_time = time.time() - summ_start

        # Update status: embedding
        stage_start = time.time()
        worker_status.set(worker_id, work_item.parent_label, "embed", config.embed_model, sub_label, "", stage_start)
        if on_status_change:
            on_status_change()

        # Embed subfeature
        embed_start = time.time()
        embedding = embed_client.embed_single(summary, timeout=config.embed_timeout)
        if validate_vector(embedding, config.embed_dimensions):
            embedding = normalize_vector(embedding)
        else:
            embedding = None
        embed_time = time.time() - embed_start

        # Update status: indexing
        stage_start = time.time()
        worker_status.set(worker_id, work_item.parent_label, "index", "", sub_label, "", stage_start)
        if on_status_change:
            on_status_change()

        # Build subfeature ID
        subfeature_id = f"{scope}:{repository}:{username}:subfeature:{work_item.parent_label}:{sub_label}:{work_item.sub_cluster.cluster_id}"

        # Index subfeature
        es_repo.index_subfeature(
            subfeature_id=subfeature_id,
            scope=scope,
            repository=repository,
            username=username,
            cluster_id=str(work_item.sub_cluster.cluster_id),
            label=sub_label,
            summary=summary,
            embedding=embedding,
            member_ids=work_item.sub_cluster.element_ids,
            parent_feature_label=work_item.parent_label,
            parent_feature_summary=work_item.parent_summary,
        )

        worker_status.clear(worker_id)
        if on_status_change:
            on_status_change()

        return ProcessedSubfeature(
            parent_label=work_item.parent_label,
            sub_label=sub_label,
            success=True,
            summarize_time=summarize_time,
            embed_time=embed_time,
        )

    except Exception as e:
        worker_status.clear(worker_id)
        if on_status_change:
            on_status_change()

        return ProcessedSubfeature(
            parent_label=work_item.parent_label,
            sub_label=sub_label,
            success=False,
            summarize_time=summarize_time,
            embed_time=embed_time,
            error=str(e),
        )


def process_subfeatures(
    clustering_result: ClusteringResult,
    processed_features: dict[int, tuple[str, str]],
    scope: str,
    repository: str,
    username: str,
    es_repo: ElasticsearchRepository,
    config: FeatureProcessingConfig | None = None,
    subcluster_config: SubClusterConfig | None = None,
    on_progress: Callable[[SubfeatureProgressState], None] | None = None,
    on_labeling_progress: Callable[[SubfeatureLabelingState], None] | None = None,
    magaldi_config: MagaldiConfig | None = None,
    timing_stats: SubfeatureTimingStats | None = None,
    worker_status: SubfeatureWorkerStatus | None = None,
    on_tier_distribution: Callable[[dict[int, int]], None] | None = None,
) -> dict[str, Any]:
    """Process subfeatures for large features (>20 members).

    Uses queue-based parallel processing with workers.

    Args:
        clustering_result: Result from main clustering.
        processed_features: Dict mapping cluster_id to (label, summary).
        scope: Repository scope.
        repository: Repository name.
        username: Username/branch.
        es_repo: Elasticsearch repository.
        config: Feature processing configuration.
        subcluster_config: Sub-clustering configuration.
        on_progress: Optional callback for processing progress updates.
        on_labeling_progress: Optional callback for labeling progress updates.
        magaldi_config: Optional config for Redis job tracking.
        timing_stats: Optional timing stats (created if not provided).
        worker_status: Optional worker status (created if not provided).

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
        return {"subfeatures_created": 0, "parent_features_processed": 0, "errors": []}

    # Delete existing subfeatures
    es_repo.delete_subfeatures(scope, repository, username)

    # Initialize Redis job tracking if config provided
    redis_repo = None
    labeling_redis_repo = None
    if magaldi_config:
        from shared.db.redis import (
            RedisSubfeatureJobRepository,
            RedisSubfeatureLabelingJobRepository,
        )
        redis_repo = RedisSubfeatureJobRepository(magaldi_config)
        labeling_redis_repo = RedisSubfeatureLabelingJobRepository(magaldi_config)
        # Clear stale subfeature queue data
        client = redis_repo._get_client()
        for key_type in ["jobs", "running", "queue"]:
            client.delete(f"magaldi:subfeature:{key_type}:{scope}:{repository}:{username}")
            client.delete(f"magaldi:subfeature_labeling:{key_type}:{scope}:{repository}:{username}")

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

    # Sub-clustering config
    cluster_config = ClusterConfig(
        min_cluster_size=subcluster_config.min_cluster_size,
        min_samples=subcluster_config.min_samples,
        api_base=config.api_base,
        labeling_model=config.summarize_model,
        provider=config.provider,
    )
    sub_clusterer = FeatureClusterer(cluster_config)

    # Initialize timing and worker tracking
    if timing_stats is None:
        timing_stats = SubfeatureTimingStats()
    timing_stats.phase_start = time.time()
    if worker_status is None:
        worker_status = SubfeatureWorkerStatus()

    # Phase 1: Discovery - identify all subfeatures to process
    work_queue: list[SubfeatureWorkItem] = []
    parents_with_subfeatures: set[str] = set()
    features_processed = 0
    subclusters_labeled = 0
    total_large_features = len(large_clusters)
    labeling_start = time.time()  # Track labeling phase start

    for cluster in large_clusters:
        parent_label, parent_summary = processed_features.get(
            cluster.cluster_id, (cluster.label or f"cluster_{cluster.cluster_id}", "")
        )

        # Add labeling job to Redis queue
        if labeling_redis_repo:
            labeling_redis_repo.add_job(parent_label, 0, scope, repository, username)
            labeling_redis_repo.mark_running(parent_label, scope, repository, username)

        # Update labeling progress - starting this feature
        if on_labeling_progress:
            on_labeling_progress(SubfeatureLabelingState(
                total_features=total_large_features,
                features_processed=features_processed,
                current_feature=parent_label,
                subclusters_labeled=subclusters_labeled,
                model=config.summarize_model,
                phase_start=labeling_start,
            ))

        # Fetch embeddings for cluster members
        member_docs = es_repo.get_documents_batch(cluster.element_ids)
        elements_with_embeddings = []

        for element_id in cluster.element_ids:
            doc = member_docs.get(element_id)
            # Use summary_embedding (the primary embedding field)
            emb = doc.get("summary_embedding") if doc else None
            if emb:
                idx = cluster.element_ids.index(element_id)
                name = cluster.element_names[idx] if idx < len(cluster.element_names) else ""
                elements_with_embeddings.append({
                    "element_id": element_id,
                    "embedding": emb,
                    "name": name,
                    "element_type": doc.get("element_type", "function"),
                })

        if len(elements_with_embeddings) < subcluster_config.min_cluster_size:
            if labeling_redis_repo:
                labeling_redis_repo.mark_completed(parent_label, scope, repository, username)
            features_processed += 1
            continue

        try:
            # Run sub-clustering
            sub_result = sub_clusterer.cluster(elements_with_embeddings)

            if not sub_result.clusters:
                if labeling_redis_repo:
                    labeling_redis_repo.mark_completed(parent_label, scope, repository, username)
                features_processed += 1
                continue

            # Label sub-clusters
            sub_result = sub_clusterer.label_clusters(sub_result)
            subclusters_labeled += len(sub_result.clusters)

            # Mark labeling as completed
            if labeling_redis_repo:
                labeling_redis_repo.mark_completed(parent_label, scope, repository, username)

            # Update labeling progress - after labeling this feature's subclusters
            if on_labeling_progress:
                on_labeling_progress(SubfeatureLabelingState(
                    total_features=total_large_features,
                    features_processed=features_processed,
                    current_feature=parent_label,
                    subclusters_labeled=subclusters_labeled,
                    model=config.summarize_model,
                    phase_start=labeling_start,
                ))

            # Fetch member summaries for all sub-cluster members
            all_sub_member_ids = []
            for sub_cluster in sub_result.clusters:
                all_sub_member_ids.extend(sub_cluster.element_ids)
            member_summaries = es_repo.get_summaries_batch(all_sub_member_ids)

            # Add each sub-cluster to the work queue
            for sub_cluster in sub_result.clusters:
                sub_label = sub_cluster.label or f"subcluster_{sub_cluster.cluster_id}"
                work_queue.append(SubfeatureWorkItem(
                    parent_label=parent_label,
                    parent_summary=parent_summary,
                    sub_cluster=sub_cluster,
                    member_summaries=member_summaries,
                ))
                parents_with_subfeatures.add(parent_label)

                # Add to Redis queue
                if redis_repo:
                    subfeature_id = f"{scope}:{repository}:{username}:subfeature:{parent_label}:{sub_label}:{sub_cluster.cluster_id}"
                    redis_repo.add_job(subfeature_id, scope, repository, username, sub_label, parent_label)

            features_processed += 1

        except Exception as e:
            # Skip this cluster if sub-clustering fails
            if labeling_redis_repo:
                labeling_redis_repo.mark_failed(parent_label, scope, repository, username, str(e))
            features_processed += 1
            continue

    if not work_queue:
        return {"subfeatures_created": 0, "parent_features_processed": 0, "errors": []}

    # Phase 2: Process subfeatures in parallel with workers
    total = len(work_queue)
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

    # Mutable state for throttle info
    subfeature_state: dict = {
        "allowed_workers": config.num_workers,
        "current_max": 0.0,
        "avg_base_time": 0.0,
    }

    def on_status_change(throttle_info: "ThrottleDisplayInfo | None" = None) -> None:
        if throttle_info is not None:
            subfeature_state["allowed_workers"] = throttle_info.allowed_workers
            subfeature_state["current_max"] = throttle_info.current_max
            subfeature_state["avg_base_time"] = throttle_info.avg_base_time
        if on_progress:
            on_progress(SubfeatureProgressState(
                total=total,
                completed=completed_count,
                failed=failed_count,
                timing=timing_stats,
                workers=worker_status,
                num_workers=config.num_workers,
                allowed_workers=subfeature_state["allowed_workers"],
                current_max=subfeature_state["current_max"],
                avg_base_time=subfeature_state["avg_base_time"],
            ))

    def process_wrapper(work_item: SubfeatureWorkItem) -> ProcessedSubfeature:
        wid = acquire_worker_id()
        sub_label = work_item.sub_cluster.label or f"subcluster_{work_item.sub_cluster.cluster_id}"
        subfeature_id = f"{scope}:{repository}:{username}:subfeature:{work_item.parent_label}:{sub_label}:{work_item.sub_cluster.cluster_id}"

        # Mark as running in Redis
        if redis_repo:
            redis_repo.mark_running(subfeature_id, scope, repository, username)

        try:
            result = _process_single_subfeature(
                work_item=work_item,
                scope=scope,
                repository=repository,
                username=username,
                es_repo=es_repo,
                llm_client=llm_client,
                embed_client=embed_client,
                config=config,
                worker_id=wid,
                worker_status=worker_status,
                on_status_change=on_status_change,
            )
            # Mark as completed or failed in Redis based on result
            if redis_repo:
                if result.success:
                    redis_repo.mark_completed(subfeature_id, scope, repository, username)
                else:
                    redis_repo.mark_failed(subfeature_id, scope, repository, username, result.error or "Unknown error")
            return result
        finally:
            release_worker_id(wid)

    # Pre-compute context tier for each work item based on prompt size
    from shared.ai.context_size import TIER_MAX_WORKERS, compute_aggregation_num_ctx, iter_by_tier

    def estimate_subfeature_tier(work_item: SubfeatureWorkItem) -> int:
        """Estimate context tier for a subfeature work item."""
        summaries_text = []
        cluster = work_item.sub_cluster
        for i, element_id in enumerate(cluster.element_ids[:config.max_member_summaries]):
            summary = work_item.member_summaries.get(element_id, "")
            name = cluster.element_names[i] if i < len(cluster.element_names) else "unknown"
            if summary:
                summaries_text.append(f"- {name}(): {summary}")
        if not summaries_text:
            summaries_text = [f"- {name}()" for name in cluster.element_names[:10]]

        user_content = SUBFEATURE_USER_PROMPT.format(
            parent_label=work_item.parent_label,
            parent_summary=work_item.parent_summary,
            member_count=cluster.size,
            member_summaries="\n".join(summaries_text),
        )
        prompt_chars = len(SUBFEATURE_SYSTEM_PROMPT) + len(user_content)
        return compute_aggregation_num_ctx(prompt_chars, task_type="subfeature")

    # Group work items by tier and process each tier with appropriate max_workers
    tier_groups = iter_by_tier(work_queue, estimate_subfeature_tier)

    # Build work_item index -> tier mapping for tracking
    # Use (parent_label, sub_cluster.cluster_id) as key
    work_item_to_tier: dict[tuple[str, int], int] = {}
    tier_counts: dict[int, int] = {}
    for tier, _, tier_items in tier_groups:
        tier_counts[tier] = len(tier_items)
        for item in tier_items:
            work_item_to_tier[(item.parent_label, item.sub_cluster.cluster_id)] = tier

    # Set tier totals for ETA calculation
    timing_stats.set_totals_by_tier(tier_counts)

    # Report tier distribution if callback provided
    if on_tier_distribution:
        on_tier_distribution(tier_counts)

    # Handle workers=0 (auto) - will use tier-specific limits
    max_pool_workers = config.num_workers if config.num_workers > 0 else max(TIER_MAX_WORKERS.values())

    # Mutable state for callbacks
    state = {"completed": 0, "failed": 0}

    def on_complete(work_item: SubfeatureWorkItem, processed: ProcessedSubfeature, avg_workers: float) -> None:
        """Handle completed subfeature processing."""
        nonlocal errors
        tier = work_item_to_tier.get(
            (work_item.parent_label, work_item.sub_cluster.cluster_id), 0
        )

        # Record for throttling
        wall_time = processed.summarize_time + processed.embed_time
        timing_stats.record_task_runtime(wall_time, avg_workers)
        timing_stats.record(processed.summarize_time, processed.embed_time, tier=tier)

        if processed.success:
            state["completed"] += 1
        else:
            state["failed"] += 1
            if processed.error:
                errors.append(f"Subfeature {processed.sub_label} of {processed.parent_label}: {processed.error}")

    # Create throttle context
    throttle_ctx = ThrottleContext(
        tier_timeout=180,
        base_workers=max_pool_workers,
        throughput_tracker=timing_stats.throughput_tracker,
    )

    try:
        for _tier, tier_max_workers, tier_items in tier_groups:
            effective_workers = min(max_pool_workers, tier_max_workers)

            # Reset worker ID pool for this tier
            available_worker_ids.clear()
            available_worker_ids.extend(range(effective_workers))

            run_throttled_tier(
                items=list(tier_items),
                tier=_tier,
                effective_workers=effective_workers,
                process_fn=process_wrapper,
                throttle_ctx=throttle_ctx,
                get_max_runtime=worker_status.get_max_active_runtime,
                on_complete=on_complete,
                on_tick=on_status_change,
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
        "subfeatures_created": completed_count,
        "parent_features_processed": len(parents_with_subfeatures),
        "errors": errors,
    }
