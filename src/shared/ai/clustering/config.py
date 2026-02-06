"""Shared configuration for feature and subfeature processing."""

from __future__ import annotations

from dataclasses import dataclass


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


@dataclass
class FeatureProcessingConfig:
    """Configuration for feature processing."""

    summarize_model: str = "qwen3:4b-instruct"
    embed_model: str = "qwen3-embedding:0.6b"
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
