"""Context size computation utilities for KV cache optimization.

This module computes optimal context sizes (num_ctx) for LLM inference.
Using appropriate context sizes improves KV cache efficiency for local
LLM providers like Ollama.

Two approaches are supported:
1. Per-element: Each element gets its own optimal tier based on its size
2. Per-type (legacy): One tier per element type based on max observed size
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable

T = TypeVar("T")

# Context size tiers (powers of 2 for memory alignment)
CONTEXT_TIERS = [1024, 2048, 4096, 8192, 16384, 32768]

# Sentinel tier for handcrafted summaries (no LLM context needed).
# Elements assigned this tier skip LLM summarization and use code/docstring directly.
# Use is_handcrafted_tier() to check — sub-tiers exist for display categorization.
HANDCRAFTED_TIER = 0

# Craft sub-tiers: split handcrafted elements by reason for display.
# All share the same processing characteristics (no LLM, max parallelism).
# Values must be < min(CONTEXT_TIERS) and >= 0 to pass tier tracking checks.
CRAFT_TEST = 1
CRAFT_IMPORT = 2
CRAFT_SMALL = 3
CRAFT_DOCSTRING = 4

# All handcrafted tier values (for iteration and membership checks)
HANDCRAFTED_TIERS = frozenset({HANDCRAFTED_TIER, CRAFT_TEST, CRAFT_IMPORT, CRAFT_SMALL, CRAFT_DOCSTRING})

# Map craft_reason string to tier value
CRAFT_REASON_TO_TIER = {
    "test": CRAFT_TEST,
    "import": CRAFT_IMPORT,
    "small": CRAFT_SMALL,
    "docstring": CRAFT_DOCSTRING,
}


def is_handcrafted_tier(tier: int) -> bool:
    """Check if a tier value is any handcrafted sub-tier."""
    return tier in HANDCRAFTED_TIERS


# Human-readable abbreviations for each tier (single source of truth)
TIER_ABBREV = {tier: f"{tier // 1024}k" for tier in CONTEXT_TIERS}
TIER_ABBREV[HANDCRAFTED_TIER] = "craft"
TIER_ABBREV[CRAFT_TEST] = "test"
TIER_ABBREV[CRAFT_IMPORT] = "import"
TIER_ABBREV[CRAFT_SMALL] = "small"
TIER_ABBREV[CRAFT_DOCSTRING] = "doc"

# Suffixes for Ollama tiered model aliases (e.g., "qwen3:4b-instruct-2k")
TIER_SUFFIXES = {tier: f"-{tier // 1024}k" for tier in CONTEXT_TIERS}

# Rich colors per tier for CLI display (red=biggest → yellow → green=smallest)
TIER_COLORS = {
    32768: "red",
    16384: "bright_red",
    8192: "yellow",
    4096: "bright_yellow",
    2048: "bright_green",
    1024: "green",
    HANDCRAFTED_TIER: "dim green",
    CRAFT_TEST: "dim green",
    CRAFT_IMPORT: "dim green",
    CRAFT_SMALL: "dim green",
    CRAFT_DOCSTRING: "dim green",
}

# Tier scaling exponent for ETA estimation fallback.
# Empirically derived from timing data: time_ratio = tier_ratio^TIER_SCALING_EXPONENT
# Higher tiers take longer, but sub-linearly (larger context doesn't mean proportionally longer output).
# Analysis of real data showed exponent ~0.65, compared to sqrt (0.5).
TIER_SCALING_EXPONENT = 0.65

# Max concurrent workers per tier (upper bound for exploration budget)
# The exploration orchestrator dynamically discovers the optimal concurrency
# within [1, max_workers] — these values set the search space ceiling.
# All LLM tiers use the same ceiling; the throttler self-limits based on
# measured throughput, emergency scale-down, and GSS probing.
DEFAULT_MAX_WORKERS = 16
TIER_MAX_WORKERS = {
    **dict.fromkeys(HANDCRAFTED_TIERS, DEFAULT_MAX_WORKERS),  # No LLM - maximum parallelism
    1024: DEFAULT_MAX_WORKERS,
    2048: DEFAULT_MAX_WORKERS,
    4096: DEFAULT_MAX_WORKERS,
    8192: DEFAULT_MAX_WORKERS,
    16384: DEFAULT_MAX_WORKERS,
    32768: DEFAULT_MAX_WORKERS,
}

# Parallel slots per tier for llama.cpp presets.
# With a fixed 40k context per model alias, parallel determines how many
# concurrent requests the server handles per tier alias. Lower tiers need
# less KV cache per slot → can fit more parallel slots.
# Used as effective max_workers ceiling for llamacpp in DependencyTracker.
TIER_PARALLEL = {
    32768: 1,
    16384: 2,
    8192: 4,
    4096: 8,
    2048: 16,
    1024: 16,
}

# Base timeout per tier in seconds (for 1 worker).
# Actual timeout scales with sqrt(base_workers) via get_tier_timeout().
# Derived from previous fixed values: old_timeout / sqrt(TIER_MAX_WORKERS[tier]).
# NOTE: These are used for throttle calculations; Ollama may not enforce actual timeouts.
TIER_BASE_TIMEOUTS = {
    **dict.fromkeys(HANDCRAFTED_TIERS, 15),  # No LLM - very fast even solo
    1024: 90,    # 1.5 minutes solo
    2048: 104,   # ~1.7 minutes solo
    4096: 127,   # ~2.1 minutes solo
    8192: 240,   # 4 minutes solo
    16384: 425,  # ~7 minutes solo
    32768: 600,  # 10 minutes solo (max workers=1, no scaling)
}

DEFAULT_BASE_TIMEOUT = 104  # Fallback (2048-tier equivalent)


def get_tier_timeout(tier: int, base_workers: int) -> float:
    """Get timeout for a tier scaled by worker count.

    Formula: base_timeout * sqrt(base_workers)

    With 1 worker, you get the base timeout (tight budget — you're the
    entire pipeline). With 16 workers, you get 4x the base (parallelism
    absorbs individual slow tasks).

    Uses base_workers (configured maximum), NOT dynamic active count,
    to avoid circular dependency with the throttle formula that computes
    recommended workers FROM the timeout.

    Args:
        tier: Context tier (e.g., 2048, 4096).
        base_workers: Configured maximum worker count.

    Returns:
        Timeout in seconds.
    """
    base = TIER_BASE_TIMEOUTS.get(tier, DEFAULT_BASE_TIMEOUT)
    return base * math.sqrt(max(1, base_workers))

# Estimated prompt overhead per element type (tokens), EXCLUDING raw code.
# Includes: system prompt + user template + parent context + output budget.
#
# System prompts (fixed text, measured):
#   file: 189, class: 189, interface: 153, trait: 149, enum: 128,
#   type_alias: 133, function: 177, method: 169, constant: 126,
#   variable: 129, import: 114
#
# User template (min static text, measured):
#   file: 200, class: 212, interface: 166, trait: 161, enum: 126,
#   type_alias: 132, function: 199, method: 206, constant: 136,
#   variable: 134, import: 114
#
# Parent context injected into prompts (bounded by OUTPUT_TOKEN_BUDGETS):
#   file_summary: up to 500 tokens (P95: ~350), used by all child types
#   class_context: up to 450 tokens (P95: ~300), used by method/function/variable/constant
#   function_context: up to 500 tokens (P95: ~300), used by variable/constant
#   sig/docstring duplication: ~50-100 tokens (in template AND code)
#
# Parent context estimates (P95 realistic, not worst-case):
#   file: ~50 (imports section only, no parent context)
#   class: ~400 (file_summary 350 + base_classes/attrs ~50)
#   interface/trait: ~400 (file_summary 350 + base_classes ~50)
#   enum: ~350 (file_summary 350)
#   type_alias: ~350 (file_summary 350)
#   function: ~750 (file_summary 350 + class_context 300 + sig/docstring ~100)
#   method: ~750 (file_summary 350 + class_context 300 + sig/docstring ~100)
#   constant: ~700 (file_summary 350 + function_context 300 + usage_examples ~50)
#   variable: ~1050 (file_summary 350 + class_context 300 + function_context 300 + usages ~100)
#
# Output budget (empirically measured from ~2300 elements, set to max * 1.25):
#   file/function: max ~400 tokens → 500 budget
#   class: max ~340 tokens → 450 budget
#   method: max ~305 tokens → 400 budget
#   interface/trait: max ~230 tokens → 300 budget
#   enum: max ~200 tokens → 250 budget
#   type_alias: max ~150 tokens → 200 budget
#   constant: max ~160 tokens → 200 budget
#   variable: max ~360 tokens → 450 budget
#   import: handcrafted (no LLM) → 100 budget
PROMPT_OVERHEAD = {
    "file": 950,        # 189 + 200 + 50 + 500 = 939 → 950
    "class": 1250,      # 189 + 212 + 400 + 450 = 1251 → 1250
    "interface": 950,   # 153 + 166 + 400 + 300 = 1019 → 950 (smaller output)
    "trait": 950,       # 149 + 161 + 400 + 300 = 1010 → 950
    "enum": 850,        # 128 + 126 + 350 + 250 = 854 → 850
    "type_alias": 800,  # 133 + 132 + 350 + 200 = 815 → 800
    "function": 1650,   # 177 + 199 + 750 + 500 = 1626 → 1650
    "method": 1550,     # 169 + 206 + 750 + 400 = 1525 → 1550
    "constant": 1150,   # 126 + 136 + 700 + 200 = 1162 → 1150
    "variable": 1800,   # 129 + 134 + 1050 + 450 = 1763 → 1800
    "import": 400,      # 114 + 114 + 0 + 100 = 328 → 400 (no parent context)
}

DEFAULT_OVERHEAD = 800


def compute_element_num_ctx(element_type: str, char_count: int) -> int:
    """Compute optimal context size for a specific element.

    This is the preferred approach - each element gets assigned to the
    smallest context tier that fits its actual size, maximizing KV cache
    efficiency.

    The total includes prompt overhead (which includes output budget) + code tokens:
        - 200 char function (50 code + 1100 overhead = 1150) → 2048 tier
        - 4000 char function (1000 code + 1100 overhead = 2100) → 4096 tier
        - 72000 char file (18000 code + 950 overhead) → 32768 tier

    Args:
        element_type: Type of code element (file, class, function, etc.)
        char_count: Character count of this element's raw code.

    Returns:
        Optimal num_ctx value from CONTEXT_TIERS.
    """
    # Estimate tokens: ~4 chars per token for code
    estimated_tokens = char_count // 4
    overhead = PROMPT_OVERHEAD.get(element_type, DEFAULT_OVERHEAD)
    total_tokens = estimated_tokens + overhead

    # Find smallest tier that fits
    for tier in CONTEXT_TIERS:
        if total_tokens < tier:
            return tier

    # Fallback to largest tier
    return CONTEXT_TIERS[-1]


def compute_prompt_num_ctx(prompt_chars: int, max_tokens: int) -> int:
    """Compute optimal context tier from an already-built prompt.

    Unlike compute_element_num_ctx (which estimates from raw_code + static
    overhead), this function uses the actual materialized prompt length.
    Use this after build_prompt() for accurate tier selection that accounts
    for variable-length parent context.

    Args:
        prompt_chars: Character count of the full prompt string.
        max_tokens: Output token budget (max_tokens for generation).

    Returns:
        Optimal num_ctx value from CONTEXT_TIERS.
    """
    # Estimate tokens: ~4 chars per token for code/mixed content
    prompt_tokens = prompt_chars // 4
    total_tokens = prompt_tokens + max_tokens

    # Find smallest tier that fits
    for tier in CONTEXT_TIERS:
        if total_tokens < tier:
            return tier

    # Fallback to largest tier
    return CONTEXT_TIERS[-1]


def compute_num_ctx(element_type: str, max_chars: int) -> int:
    """Compute optimal context size for an element type (legacy).

    This function computes a single context size for all elements of a type
    based on the maximum observed size. Consider using compute_element_num_ctx()
    for per-element sizing which is more efficient.

    Args:
        element_type: Type of code element (file, class, function, etc.)
        max_chars: Maximum character count observed for this type.

    Returns:
        Optimal num_ctx value from CONTEXT_TIERS.
    """
    return compute_element_num_ctx(element_type, max_chars)


def compute_context_sizes(max_chars_by_type: dict[str, int]) -> dict[str, int]:
    """Compute context sizes for all element types (legacy).

    This function computes one context size per element type based on max
    observed sizes. Consider using compute_element_num_ctx() for per-element
    sizing which is more efficient.

    Args:
        max_chars_by_type: Dict mapping element_type to max char count.

    Returns:
        Dict mapping element_type to optimal num_ctx.
    """
    return {
        element_type: compute_element_num_ctx(element_type, max_chars)
        for element_type, max_chars in max_chars_by_type.items()
    }


# Prompt overhead estimates for aggregation tasks (in tokens)
# These include system prompts, templates, and typical output sizes
AGGREGATION_OVERHEAD = {
    "labeling": 150,        # System (~100) + output (~50)
    "feature": 200,         # System (~120) + template (~30) + output (~50)
    "subfeature": 250,      # System (~130) + parent context (~70) + output (~50)
    "glossary_extract": 250,  # System (~200) + output (~50)
    "glossary_summary": 350,  # System (~250) + output (~100)
}


def compute_aggregation_num_ctx(
    prompt_chars: int,
    task_type: str = "feature",
    safety_multiplier: float = 2.0,
) -> int:
    """Compute optimal context size for aggregation tasks (features, glossary).

    Uses a safety multiplier (default 2x) to ensure sufficient headroom for
    variable content lengths and model output.

    Args:
        prompt_chars: Total character count of the prompt content (excluding
            system prompt overhead which is added automatically).
        task_type: Type of aggregation task for overhead calculation.
            One of: labeling, feature, subfeature, glossary_extract, glossary_summary.
        safety_multiplier: Multiplier for safety margin (default 2.0 = double).

    Returns:
        Optimal num_ctx value from CONTEXT_TIERS.

    Example:
        - 2000 char feature summaries → ~570 tokens + 200 overhead = 770
        - With 2x multiplier → 1540 → 2048 tier
    """
    # Estimate tokens: ~3.5 chars per token for natural language summaries
    estimated_tokens = int(prompt_chars / 3.5)
    overhead = AGGREGATION_OVERHEAD.get(task_type, 200)
    total_tokens = int((estimated_tokens + overhead) * safety_multiplier)

    # Find smallest tier that fits
    for tier in CONTEXT_TIERS:
        if total_tokens < tier:
            return tier

    # Fallback to largest tier
    return CONTEXT_TIERS[-1]


# =============================================================================
# TIER-BASED BATCHING UTILITIES
# =============================================================================


def get_max_workers_for_tier(tier: int) -> int:
    """Get max workers for a context tier.

    Args:
        tier: Context tier (e.g., 2048, 4096).

    Returns:
        Max workers for this tier.
    """
    return TIER_MAX_WORKERS.get(tier, 1)


def get_tier_effective_workers(tier: int, max_workers: int, provider: str | None = None) -> int:
    """Get effective worker count for a tier, respecting provider-specific limits.

    For llamacpp, the effective ceiling is the tier's parallel slot count from
    TIER_PARALLEL (since llama-server can only handle that many concurrent
    requests per tier alias). The result is capped at max_workers.

    For other providers, returns min(max_workers, TIER_MAX_WORKERS[tier]).

    Args:
        tier: Context tier (e.g., 2048, 4096).
        max_workers: CLI-configured maximum worker count.
        provider: LLM provider (ollama, llamacpp, openai, etc.).

    Returns:
        Effective worker count for this tier.
    """
    if provider == "llamacpp":
        parallel_limit = TIER_PARALLEL.get(tier)
        if parallel_limit is not None:
            return min(max_workers, parallel_limit)
    return min(max_workers, get_max_workers_for_tier(tier))


def group_by_tier(items: list[T], tier_fn: Callable[[T], int]) -> dict[int, list[T]]:
    """Group items by their context tier.

    Args:
        items: List of items to group.
        tier_fn: Function that takes an item and returns its context tier.

    Returns:
        Dict mapping tier to list of items in that tier.
    """
    groups: dict[int, list[T]] = defaultdict(list)
    for item in items:
        tier = tier_fn(item)
        groups[tier].append(item)
    return dict(groups)


def iter_by_tier(
    items: list[T],
    tier_fn: Callable[[T], int],
) -> list[tuple[int, int, list[T]]]:
    """Iterate items grouped by tier with max_workers.

    Yields tiers in order from largest to smallest context (finish big ones first
    to avoid long tail at the end).

    Args:
        items: List of items to process.
        tier_fn: Function that takes an item and returns its context tier.

    Returns:
        List of (tier, max_workers, items) tuples sorted by tier descending.
    """
    groups = group_by_tier(items, tier_fn)

    # Sort by tier descending (largest first = finish big ones first)
    result = []
    for tier in sorted(groups.keys(), reverse=True):
        max_workers = get_max_workers_for_tier(tier)
        result.append((tier, max_workers, groups[tier]))

    return result
