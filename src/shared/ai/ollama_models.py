"""Ollama model alias management for context tier optimization.

This module manages tiered model aliases for Ollama to avoid model reload
overhead when context size changes between requests.

Instead of passing num_ctx per request (which causes Ollama to unload/reload
the model), we create aliases with fixed context sizes:
  - qwen3:4b-instruct-2k  (num_ctx=2048)
  - qwen3:4b-instruct-4k  (num_ctx=4096)
  - qwen3:4b-instruct-8k  (num_ctx=8192)
  - etc.

The model weights are shared via mmap, so only the KV cache memory differs.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import httpx

from shared.ai.context_size import CONTEXT_TIERS, TIER_SUFFIXES

if TYPE_CHECKING:
    from shared.config import ModelConfig

logger = logging.getLogger(__name__)


@dataclass
class OllamaModel:
    """Represents an Ollama model from the API."""

    name: str
    size: int = 0
    modified_at: str = ""


def list_ollama_models(base_url: str = "http://localhost:11434") -> list[OllamaModel]:
    """List all models available in Ollama.

    Args:
        base_url: Ollama API base URL.

    Returns:
        List of OllamaModel instances.

    Raises:
        httpx.HTTPError: If the API request fails.
    """
    url = f"{base_url.rstrip('/')}/api/tags"
    try:
        response = httpx.get(url, timeout=30.0)
        response.raise_for_status()
        data = response.json()

        models = []
        for model_data in data.get("models", []):
            models.append(OllamaModel(
                name=model_data.get("name", ""),
                size=model_data.get("size", 0),
                modified_at=model_data.get("modified_at", ""),
            ))
        return models
    except httpx.HTTPError as e:
        logger.warning(f"Failed to list Ollama models: {e}")
        raise


def get_tiered_model_name(base_model: str, num_ctx: int) -> str:
    """Get the tiered model alias name for a given context size.

    Args:
        base_model: Base model name (e.g., "qwen3:4b-instruct").
        num_ctx: Context size (will be rounded up to nearest tier).

    Returns:
        Tiered model name (e.g., "qwen3:4b-instruct-4k").
    """
    # Find the appropriate tier
    tier = CONTEXT_TIERS[-1]  # Default to largest
    for t in CONTEXT_TIERS:
        if num_ctx <= t:
            tier = t
            break

    suffix = TIER_SUFFIXES.get(tier, f"-{tier // 1024}k")
    return f"{base_model}{suffix}"


def is_tiered_model(model_name: str) -> bool:
    """Check if a model name is already a tiered alias.

    Args:
        model_name: Model name to check.

    Returns:
        True if the model name has a tier suffix.
    """
    pattern = r"-\d+k$"
    return bool(re.search(pattern, model_name))


def get_base_model_name(tiered_model: str) -> str:
    """Extract the base model name from a tiered alias.

    Args:
        tiered_model: Tiered model name (e.g., "qwen3:4b-instruct-4k").

    Returns:
        Base model name (e.g., "qwen3:4b-instruct").
    """
    pattern = r"-\d+k$"
    return re.sub(pattern, "", tiered_model)


def create_tiered_alias(
    base_model: str,
    num_ctx: int,
    base_url: str = "http://localhost:11434",
) -> bool:
    """Create a single tiered model alias in Ollama.

    Args:
        base_model: Base model name to create alias from.
        num_ctx: Context size for this alias.
        base_url: Ollama API base URL.

    Returns:
        True if alias was created successfully.
    """
    alias_name = get_tiered_model_name(base_model, num_ctx)

    url = f"{base_url.rstrip('/')}/api/create"
    payload = {
        "model": alias_name,
        "from": base_model,
        "parameters": {"num_ctx": num_ctx},
        "stream": False,
    }

    try:
        # Use longer timeout for model creation
        response = httpx.post(url, json=payload, timeout=120.0)
        response.raise_for_status()
        logger.info(f"Created Ollama alias: {alias_name} (num_ctx={num_ctx})")
        return True
    except httpx.HTTPError as e:
        logger.error(f"Failed to create Ollama alias {alias_name}: {e}")
        return False


def model_exists_in_ollama(model_name: str, base_url: str = "http://localhost:11434") -> bool:
    """Check if a model exists in Ollama.

    Args:
        model_name: Model name to check.
        base_url: Ollama API base URL.

    Returns:
        True if model exists.
    """
    try:
        existing_models = {m.name for m in list_ollama_models(base_url)}
        # Check exact match or with :latest suffix
        if model_name in existing_models:
            return True
        if f"{model_name}:latest" in existing_models:
            return True
        return False
    except httpx.HTTPError:
        return False


def ensure_tiered_model(
    base_model: str,
    num_ctx: int,
    base_url: str = "http://localhost:11434",
) -> str:
    """Ensure a specific tiered model alias exists.

    Checks Ollama directly - creates the alias if it doesn't exist.

    Args:
        base_model: Base model name (e.g., "qwen3:4b-instruct").
        num_ctx: Context size (will be rounded up to nearest tier).
        base_url: Ollama API base URL.

    Returns:
        Tiered model name (e.g., "qwen3:4b-instruct-4k").
        Falls back to base_model if creation fails.
    """
    # Skip if base model is already a tiered alias
    if is_tiered_model(base_model):
        return base_model

    # Get the tiered model name for this context size
    tiered_name = get_tiered_model_name(base_model, num_ctx)

    # Check if it already exists in Ollama
    if model_exists_in_ollama(tiered_name, base_url):
        logger.debug(f"Tiered model {tiered_name} already exists")
        return tiered_name

    # Check if base model exists before trying to create alias
    if not model_exists_in_ollama(base_model, base_url):
        logger.warning(f"Base model {base_model} not found in Ollama")
        return base_model

    # Create the tiered alias
    # Find the actual tier for this num_ctx
    tier = CONTEXT_TIERS[-1]
    for t in CONTEXT_TIERS:
        if num_ctx <= t:
            tier = t
            break

    if create_tiered_alias(base_model, tier, base_url):
        return tiered_name
    else:
        logger.warning(f"Failed to create {tiered_name}, using base model")
        return base_model


@dataclass
class TierCreationResult:
    """Result of creating tiered model aliases."""

    tier_map: dict[int, str]  # tier size -> model alias name
    created: list[int]  # tiers that were created
    existed: list[int]  # tiers that already existed
    failed: list[int]  # tiers that failed to create


def ensure_all_tiers(
    base_model: str,
    base_url: str = "http://localhost:11434",
    tiers: list[int] | None = None,
) -> dict[int, str]:
    """Ensure all tiered model aliases exist for a base model.

    Creates any missing tiered aliases. Checks Ollama directly for each.

    Args:
        base_model: Base model name (e.g., "qwen3:4b-instruct").
        base_url: Ollama API base URL.
        tiers: List of context tiers to create. Defaults to CONTEXT_TIERS.

    Returns:
        Dict mapping tier size to model alias name.
    """
    result = ensure_all_tiers_verbose(base_model, base_url, tiers)
    return result.tier_map


def ensure_all_tiers_verbose(
    base_model: str,
    base_url: str = "http://localhost:11434",
    tiers: list[int] | None = None,
) -> TierCreationResult:
    """Ensure all tiered model aliases exist for a base model with detailed results.

    Creates any missing tiered aliases. Checks Ollama directly for each.

    Args:
        base_model: Base model name (e.g., "qwen3:4b-instruct").
        base_url: Ollama API base URL.
        tiers: List of context tiers to create. Defaults to CONTEXT_TIERS.

    Returns:
        TierCreationResult with tier_map and lists of created/existed/failed tiers.
    """
    if tiers is None:
        tiers = CONTEXT_TIERS

    tier_map: dict[int, str] = {}
    created: list[int] = []
    existed: list[int] = []
    failed: list[int] = []

    # Skip if base model is already a tiered alias
    if is_tiered_model(base_model):
        logger.debug(f"Model {base_model} is already a tiered alias, skipping")
        return TierCreationResult(
            tier_map={t: base_model for t in tiers},
            created=[],
            existed=list(tiers),
            failed=[],
        )

    # Check if base model exists
    if not model_exists_in_ollama(base_model, base_url):
        logger.warning(f"Base model {base_model} not found in Ollama, skipping tier creation")
        return TierCreationResult(
            tier_map={t: base_model for t in tiers},
            created=[],
            existed=[],
            failed=list(tiers),
        )

    # Get existing models once for efficiency
    try:
        existing_models = {m.name for m in list_ollama_models(base_url)}
    except httpx.HTTPError:
        logger.warning(f"Cannot list Ollama models, skipping tier creation for {base_model}")
        return TierCreationResult(
            tier_map={t: base_model for t in tiers},
            created=[],
            existed=[],
            failed=list(tiers),
        )

    for tier in tiers:
        alias_name = get_tiered_model_name(base_model, tier)

        if alias_name in existing_models:
            logger.debug(f"Tiered alias {alias_name} already exists")
            tier_map[tier] = alias_name
            existed.append(tier)
        else:
            if create_tiered_alias(base_model, tier, base_url):
                tier_map[tier] = alias_name
                created.append(tier)
            else:
                # Fall back to base model if creation fails
                tier_map[tier] = base_model
                failed.append(tier)
                logger.warning(f"Using base model {base_model} for tier {tier}")

    return TierCreationResult(tier_map=tier_map, created=created, existed=existed, failed=failed)


def resolve_ollama_model(
    litellm_model: str,
    api_base: str | None,
    num_ctx: int | None,
) -> tuple[str, int | None]:
    """Resolve the appropriate Ollama model for a given context size.

    For Ollama models, returns a tiered alias (e.g., "ollama/qwen3:4b-instruct-4k")
    with the num_ctx still passed (harmless redundancy, acts as fallback).

    Checks Ollama directly - no caching. Creates the alias if it doesn't exist.

    For non-Ollama models, returns the original model and num_ctx unchanged.

    Args:
        litellm_model: LiteLLM model string (e.g., "ollama/qwen3:4b-instruct").
        api_base: API base URL.
        num_ctx: Desired context size.

    Returns:
        Tuple of (model_string, num_ctx_param).
        For Ollama: ("ollama/qwen3:4b-instruct-4k", num_ctx) - tiered model with fallback param
        For others: (original_model, original_num_ctx)
    """
    # Only handle Ollama models
    if not litellm_model.startswith("ollama/"):
        return litellm_model, num_ctx

    # If no num_ctx specified, use base model
    if num_ctx is None:
        return litellm_model, None

    # Extract base model name (remove "ollama/" prefix)
    base_model = litellm_model[7:]  # Remove "ollama/"

    # Skip if already a tiered model
    if is_tiered_model(base_model):
        return litellm_model, num_ctx

    # Ensure the tiered model exists (checks Ollama directly, creates if needed)
    base_url = api_base or "http://localhost:11434"
    tiered_model = ensure_tiered_model(base_model, num_ctx, base_url)

    # Still pass num_ctx as harmless fallback
    return f"ollama/{tiered_model}", num_ctx


def warmup_ollama_tiers(
    litellm_model: str,
    api_base: str | None = None,
    tiers: list[int] | None = None,
) -> dict[int, str]:
    """Pre-create tiered model aliases for an Ollama model.

    Call this before processing to ensure all tier aliases exist.
    Checks Ollama directly - no caching.
    Does nothing for non-Ollama models.

    Args:
        litellm_model: LiteLLM model string (e.g., "ollama/qwen3:4b-instruct").
        api_base: API base URL.
        tiers: List of context tiers. Defaults to CONTEXT_TIERS.

    Returns:
        Dict mapping tier size to full LiteLLM model string.
        Empty dict for non-Ollama models.
    """
    result = warmup_ollama_tiers_verbose(litellm_model, api_base, tiers)
    return result.tier_map if result else {}


def warmup_ollama_tiers_verbose(
    litellm_model: str,
    api_base: str | None = None,
    tiers: list[int] | None = None,
) -> TierCreationResult | None:
    """Pre-create tiered model aliases for an Ollama model with detailed results.

    Call this before processing to ensure all tier aliases exist.
    Checks Ollama directly - no caching.
    Does nothing for non-Ollama models.

    Args:
        litellm_model: LiteLLM model string (e.g., "ollama/qwen3:4b-instruct").
        api_base: API base URL.
        tiers: List of context tiers. Defaults to CONTEXT_TIERS.

    Returns:
        TierCreationResult with detailed info, or None for non-Ollama models.
    """
    if not litellm_model.startswith("ollama/"):
        return None

    base_model = litellm_model[7:]  # Remove "ollama/"

    if is_tiered_model(base_model):
        logger.debug(f"Model {base_model} is already tiered, skipping warmup")
        return None

    base_url = api_base or "http://localhost:11434"

    result = ensure_all_tiers_verbose(
        base_model=base_model,
        base_url=base_url,
        tiers=tiers or CONTEXT_TIERS,
    )

    # Convert tier_map to include "ollama/" prefix
    result.tier_map = {tier: f"ollama/{name}" for tier, name in result.tier_map.items()}

    return result
