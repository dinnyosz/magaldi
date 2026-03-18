"""llama.cpp tiered model alias management via GGUF symlinks.

This module manages tiered model aliases for llama.cpp's router mode by
creating symlinked GGUF files with context tier suffixes. Each symlink
gets its own preset section with a matching ctx-size.

Instead of using one large context window for all requests (wasting KV cache
memory), we create symlinks for each tier:
  - Qwen3-4B-Instruct-2507-Q4_K_M-2k.gguf → Qwen3-4B-Instruct-2507-Q4_K_M.gguf  (ctx-size=2048)
  - Qwen3-4B-Instruct-2507-Q4_K_M-4k.gguf → Qwen3-4B-Instruct-2507-Q4_K_M.gguf  (ctx-size=4096)
  - Qwen3-4B-Instruct-2507-Q4_K_M-8k.gguf → Qwen3-4B-Instruct-2507-Q4_K_M.gguf  (ctx-size=8192)
  - etc.

The GGUF weights are shared via mmap — symlinks don't duplicate memory.
Each alias gets its own KV cache allocation based on its preset ctx-size.

Mirrors the Ollama tiered alias pattern (see ollama_models.py) but uses
filesystem symlinks + presets instead of Ollama's model creation API.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from shared.ai.context_size import CONTEXT_TIERS, TIER_SUFFIXES

logger = logging.getLogger(__name__)


def get_tiered_model_name(base_model: str, num_ctx: int) -> str:
    """Get the tiered model alias name for a given context size.

    Args:
        base_model: Base model name (GGUF stem, e.g., "Qwen3-4B-Instruct-2507-Q4_K_M").
        num_ctx: Context size (will be rounded up to nearest tier).

    Returns:
        Tiered model name (e.g., "Qwen3-4B-Instruct-2507-Q4_K_M-4k").
    """
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
        model_name: Model name to check (GGUF stem or full name).

    Returns:
        True if the model name has a tier suffix.
    """
    return bool(re.search(r"-\d+k$", model_name))


def get_base_model_name(tiered_model: str) -> str:
    """Extract the base model name from a tiered alias.

    Args:
        tiered_model: Tiered model name (e.g., "Qwen3-4B-Instruct-2507-Q4_K_M-4k").

    Returns:
        Base model name (e.g., "Qwen3-4B-Instruct-2507-Q4_K_M").
    """
    return re.sub(r"-\d+k$", "", tiered_model)


def is_tier_symlink(path: Path) -> bool:
    """Check if a GGUF path is a tier symlink (symlink with tier suffix).

    Args:
        path: Path to check.

    Returns:
        True if the path is a symlink AND has a tier suffix in its stem.
    """
    return path.is_symlink() and is_tiered_model(path.stem)


# ─────────────────────────────────────────────────────────────────────────────
# SYMLINK CREATION
# ─────────────────────────────────────────────────────────────────────────────


def create_tiered_symlink(
    base_gguf: Path,
    tier: int,
    models_dir: Path | None = None,
) -> Path | None:
    """Create a single tiered symlink for a GGUF file.

    Creates a symlink like Qwen3-4B-Instruct-2507-Q4_K_M-4k.gguf → Qwen3-4B-Instruct-2507-Q4_K_M.gguf
    in the same directory as the base GGUF file.

    Args:
        base_gguf: Path to the base GGUF file.
        tier: Context tier (e.g., 2048, 4096).
        models_dir: Directory for symlinks (defaults to base_gguf's parent).

    Returns:
        Path to the created symlink, or None if creation failed.
    """
    target_dir = models_dir or base_gguf.parent
    tiered_name = get_tiered_model_name(base_gguf.stem, tier)
    symlink_path = target_dir / f"{tiered_name}.gguf"

    if symlink_path.exists():
        if symlink_path.is_symlink():
            logger.debug("Tier symlink already exists: %s", symlink_path.name)
            return symlink_path
        else:
            logger.warning("Non-symlink file exists at %s, skipping", symlink_path.name)
            return None

    try:
        # Use relative symlink so models_dir can be moved
        symlink_path.symlink_to(base_gguf.name)
        logger.info("Created tier symlink: %s → %s", symlink_path.name, base_gguf.name)
        return symlink_path
    except OSError as e:
        logger.error("Failed to create tier symlink %s: %s", symlink_path.name, e)
        return None


@dataclass
class TierSymlinkResult:
    """Result of creating tiered symlinks for a model."""

    base_model: str
    created: list[int] = field(default_factory=list)
    existed: list[int] = field(default_factory=list)
    failed: list[int] = field(default_factory=list)
    skipped: int = 0

    @property
    def total_created(self) -> int:
        return len(self.created)


def ensure_all_tier_symlinks(
    base_gguf: Path,
    models_dir: Path | None = None,
    tiers: list[int] | None = None,
) -> TierSymlinkResult:
    """Create tier symlinks for all context tiers.

    Args:
        base_gguf: Path to the base GGUF file.
        models_dir: Directory for symlinks (defaults to base_gguf's parent).
        tiers: List of context tiers. Defaults to CONTEXT_TIERS.

    Returns:
        TierSymlinkResult with details of what was created.
    """
    if tiers is None:
        tiers = CONTEXT_TIERS

    result = TierSymlinkResult(base_model=base_gguf.stem)

    if not base_gguf.exists():
        logger.warning("Base GGUF not found: %s", base_gguf)
        result.failed = list(tiers)
        return result

    # Skip if base model is already a tiered alias
    if is_tiered_model(base_gguf.stem):
        logger.debug("Model %s is already a tiered alias, skipping", base_gguf.stem)
        result.skipped = len(tiers)
        return result

    target_dir = models_dir or base_gguf.parent

    for tier in tiers:
        tiered_name = get_tiered_model_name(base_gguf.stem, tier)
        symlink_path = target_dir / f"{tiered_name}.gguf"

        if symlink_path.exists():
            if symlink_path.is_symlink():
                result.existed.append(tier)
            else:
                logger.warning("Non-symlink file at %s, skipping tier %d", symlink_path.name, tier)
                result.failed.append(tier)
            continue

        symlink = create_tiered_symlink(base_gguf, tier, target_dir)
        if symlink is not None:
            result.created.append(tier)
        else:
            result.failed.append(tier)

    return result


def cleanup_tier_symlinks(
    models_dir: Path,
    base_model: str | None = None,
) -> int:
    """Remove tier symlinks from models directory.

    Args:
        models_dir: Directory containing GGUF files and symlinks.
        base_model: If set, only clean symlinks for this base model.

    Returns:
        Number of symlinks removed.
    """
    removed = 0
    for path in models_dir.glob("*.gguf"):
        if not is_tier_symlink(path):
            continue
        if base_model and get_base_model_name(path.stem) != base_model:
            continue
        try:
            path.unlink()
            logger.info("Removed tier symlink: %s", path.name)
            removed += 1
        except OSError as e:
            logger.error("Failed to remove %s: %s", path.name, e)
    return removed


# ─────────────────────────────────────────────────────────────────────────────
# MODEL RESOLUTION (for API calls)
# ─────────────────────────────────────────────────────────────────────────────


def resolve_llamacpp_model(
    model_name: str,
    num_ctx: int | None,
) -> str:
    """Resolve a llamacpp model name to its tiered variant.

    Appends the appropriate tier suffix based on num_ctx.
    No network calls needed — symlinks and presets handle the rest.

    Args:
        model_name: Base model name (e.g., "Qwen3-4B-Instruct-2507-Q4_K_M").
        num_ctx: Desired context size.

    Returns:
        Tiered model name (e.g., "Qwen3-4B-Instruct-2507-Q4_K_M-4k").
        Returns base model if no num_ctx or already tiered.
    """
    if num_ctx is None:
        return model_name

    if is_tiered_model(model_name):
        return model_name

    return get_tiered_model_name(model_name, num_ctx)
