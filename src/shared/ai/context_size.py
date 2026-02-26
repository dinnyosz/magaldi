"""Context size computation utilities for KV cache optimization.

This module computes optimal context sizes (num_ctx) for LLM inference.
Using appropriate context sizes improves KV cache efficiency for local
LLM providers like Ollama.

Two approaches are supported:
1. Per-element: Each element gets its own optimal tier based on its size
2. Per-type (legacy): One tier per element type based on max observed size
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable

T = TypeVar("T")

# Context size tiers (powers of 2 for memory alignment)
CONTEXT_TIERS = [1024, 2048, 4096, 8192, 16384, 32768]

# Sentinel tier for handcrafted summaries (no LLM context needed).
# Elements assigned this tier skip LLM summarization and use code/docstring directly.
HANDCRAFTED_TIER = 0

# Human-readable abbreviations for each tier (single source of truth)
TIER_ABBREV = {tier: f"{tier // 1024}k" for tier in CONTEXT_TIERS}
TIER_ABBREV[HANDCRAFTED_TIER] = "craft"

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
}

# Tier scaling exponent for ETA estimation fallback.
# Empirically derived from timing data: time_ratio = tier_ratio^TIER_SCALING_EXPONENT
# Higher tiers take longer, but sub-linearly (larger context doesn't mean proportionally longer output).
# Analysis of real data showed exponent ~0.65, compared to sqrt (0.5).
TIER_SCALING_EXPONENT = 0.65

# Max concurrent workers per tier (inversely proportional to context size)
# Smaller contexts = more parallelism, larger contexts = less to avoid GPU saturation
# Tuned for M4 Pro 48GB - adjust for different hardware
TIER_MAX_WORKERS = {
    HANDCRAFTED_TIER: 16,  # No LLM - maximum parallelism
    1024: 16,  # Tiny context - maximum parallelism
    2048: 12,  # Small context - high parallelism
    4096: 8,   # Medium-small
    8192: 4,   # Medium
    16384: 2,  # Large - limited parallelism
    32768: 1,  # Very large - sequential to avoid OOM
}

# Timeout per tier in seconds (scales with context size)
# Larger contexts take proportionally longer to process
# NOTE: These are used for throttle calculations; Ollama may not enforce actual timeouts
TIER_TIMEOUTS = {
    HANDCRAFTED_TIER: 30,  # 30 seconds - no LLM, very fast
    1024: 60,    # 1 minute - tiny elements
    2048: 120,   # 2 minutes
    4096: 120,   # 2 minutes
    8192: 180,   # 3 minutes
    16384: 240,  # 4 minutes
    32768: 240,  # 4 minutes
}

# Estimated prompt overhead per element type (tokens), EXCLUDING raw code.
# Includes: system prompt + user template + typical variable content + output budget.
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
# Variable content (estimated typical, e.g. file_summary, class_context,
#   signature/docstring duplication from raw_code):
#   file: ~50, class: ~130, function: ~200 (sig+docstring in template AND code),
#   method: ~240, variable/constant: ~160, interface/trait: ~70, enum/type_alias: ~60
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
    "class": 1000,      # 189 + 212 + 130 + 450 = 981 → 1000
    "interface": 700,   # 153 + 166 + 70 + 300 = 689 → 700
    "trait": 700,       # 149 + 161 + 70 + 300 = 680 → 700
    "enum": 600,        # 128 + 126 + 60 + 250 = 564 → 600
    "type_alias": 550,  # 133 + 132 + 60 + 200 = 525 → 550
    "function": 1100,   # 177 + 199 + 200 + 500 = 1076 → 1100
    "method": 1050,     # 169 + 206 + 240 + 400 = 1015 → 1050
    "constant": 650,    # 126 + 136 + 160 + 200 = 622 → 650
    "variable": 900,    # 129 + 134 + 160 + 450 = 873 → 900
    "import": 400,      # 114 + 114 + 60 + 100 = 388 → 400
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
