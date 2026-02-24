"""Data models for element processing.

Contains configuration and result dataclasses used throughout the processor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from shared.config import ModelConfig


def _get_model_display_name(model_config: "ModelConfig", num_ctx: int) -> str:
    """Get the display name for a model, including tier suffix for Ollama models.

    Args:
        model_config: Model configuration.
        num_ctx: Context size being used.

    Returns:
        Model name with tier suffix for Ollama models (e.g., "qwen3:4b-instruct-4k"),
        or the original name for other providers.
    """
    if model_config.provider == "ollama":
        from shared.ai.ollama_models import get_tiered_model_name

        return get_tiered_model_name(model_config.name, num_ctx)
    return model_config.name


# =============================================================================
# DATA CLASSES
# =============================================================================


def _default_summarize_model() -> "ModelConfig":
    """Create default summarize model config."""
    from shared.config import ModelConfig
    return ModelConfig(
        name="qwen3:4b-instruct", provider="llamacpp", url="http://localhost:8080"
    )


def _default_summarize_model_small() -> "ModelConfig":
    """Create default small summarize model config."""
    from shared.config import ModelConfig
    return ModelConfig(
        name="qwen3:1.7b", provider="llamacpp", url="http://localhost:8081"
    )


def _default_embed_model() -> "ModelConfig":
    """Create default embed model config."""
    from shared.config import ModelConfig
    return ModelConfig(
        name="qwen3-embedding:0.6b", provider="ollama", url="http://localhost:11434", dimensions=1024
    )


@dataclass
class ProcessingConfig:
    """Configuration for unified processing.

    Uses ModelConfig objects that encapsulate name, provider, url, api_key.
    """

    # Model configurations (encapsulate name + provider + url + api_key)
    # Note: Each llamacpp model needs its own server instance on a different port
    summarize_model: "ModelConfig" = field(default_factory=_default_summarize_model)
    summarize_model_small: "ModelConfig" = field(default_factory=_default_summarize_model_small)
    embed_model: "ModelConfig" = field(default_factory=_default_embed_model)

    skip_ai: bool = False

    # Summarization settings (based on arxiv.org/html/2507.03160v2)
    summarize_temperature: float = 0.2
    summarize_max_tokens: int = 512
    summarize_timeout: int = 180  # 3 minutes to handle queue wait with many workers
    max_code_tokens: int = 4000

    # Handcrafted summary threshold: functions/methods with <= this many
    # non-empty lines get a handcrafted summary instead of LLM summarization.
    # Set to 0 to disable. Default 5 (~24% of functions in typical codebases).
    handcrafted_max_lines: int = 5

    # Embedding settings
    embed_dimensions: int = 1024
    embed_max_context: int = 8192
    embed_timeout: int = 120  # 2 minutes for embedding batches

    # Parallel processing (0 = use default of 8 workers)
    num_workers: int = 0

    # Context sizes per element type (for LLM num_ctx parameter)
    context_sizes: dict[str, int] = field(default_factory=dict)

    def get_model_for_element_type(self, element_type: str, num_ctx: int = 0) -> "ModelConfig":
        """Get the appropriate model config for an element type.

        Uses small model for functions, methods, variables, constants.
        Uses main model for files, classes — unless element fits in the
        small-model tier threshold, where the small model is sufficient
        for trivially small elements.
        """
        from .timing import _ALWAYS_SMALL_TYPES, _SMALL_MODEL_TIER_THRESHOLD

        if element_type in _ALWAYS_SMALL_TYPES:
            return self.summarize_model_small
        if num_ctx > 0 and num_ctx <= _SMALL_MODEL_TIER_THRESHOLD:
            return self.summarize_model_small
        return self.summarize_model


@dataclass
class ProcessingResult:
    """Result of unified processing."""

    scope: str
    repository: str
    username: str

    # Counts
    elements_processed: int = 0
    elements_skipped: int = 0  # Content unchanged and already fully processed
    elements_deleted: int = 0  # Stale elements removed (in ES but not in code)
    elements_failed: int = 0

    # By type
    summarized: int = 0
    embedded: int = 0
    indexed: int = 0

    # Errors
    errors: list[str] = field(default_factory=list)

    # Failed elements with errors
    failed_elements: list[tuple[str, str]] = field(default_factory=list)  # (element_id, error)


@dataclass
class ProcessedElement:
    """Result from processing a single element."""

    element_id: str
    success: bool
    wall_time: float
    summarize_time: float
    embed_time: float  # Total embed time (summary + code)
    summary_embed_time: float = 0.0  # Time for summary embedding
    code_embed_time: float = 0.0  # Time for code embedding
    prompt_tokens: int = 0  # Estimated tokens in full prompt
    response_tokens: int = 0  # Estimated tokens in LLM response
    assigned_tier: int = 0  # Context tier assigned to this element
    error: str | None = None
