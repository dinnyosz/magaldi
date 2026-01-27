"""Context size computation utilities for KV cache optimization.

This module computes optimal context sizes (num_ctx) for LLM inference.
Using appropriate context sizes improves KV cache efficiency for local
LLM providers like Ollama.

Two approaches are supported:
1. Per-element: Each element gets its own optimal tier based on its size
2. Per-type (legacy): One tier per element type based on max observed size
"""

from __future__ import annotations

# Context size tiers (powers of 2 for memory alignment)
CONTEXT_TIERS = [2048, 4096, 8192, 16384, 32768]

# Max concurrent workers per tier (inversely proportional to context size)
# Smaller contexts = more parallelism, larger contexts = less to avoid GPU saturation
# Tuned for M4 Pro 48GB - adjust for different hardware
TIER_MAX_WORKERS = {
    2048: 12,  # Small context - max parallelism
    4096: 8,   # Medium-small
    8192: 4,   # Medium
    16384: 2,  # Large - limited parallelism
    32768: 1,  # Very large - sequential to avoid OOM
}

# Estimated prompt overhead per element type (tokens)
# Accounts for system prompt, user template, and parent context
# - file: system prompt (~162) + template (~50) + imports (~50) = ~262
# - class: system prompt (~150) + file_summary (~200) + attrs (~100) = ~450
# - function: system prompt (~137) + file_summary (~200) + class_context (~200) + sig/docstring (~100) = ~637
# - method: system prompt (~137) + file_summary (~200) + class_summary (~200) + sig/state (~100) = ~637
# - variable/constant: system prompt (~80) + file_summary (~200) + class_context (~200) + function_context (~200) = ~680
PROMPT_OVERHEAD = {
    "file": 300,
    "class": 500,
    "function": 700,
    "method": 700,
    "variable": 700,
    "constant": 700,
}

DEFAULT_OVERHEAD = 500


def compute_element_num_ctx(element_type: str, char_count: int) -> int:
    """Compute optimal context size for a specific element.

    This is the preferred approach - each element gets assigned to the
    smallest context tier that fits its actual size, maximizing KV cache
    efficiency.

    Example:
        - 200 char function (50 tokens + 700 overhead) → 2048 tier
        - 72000 char file (18000 tokens + 300 overhead) → 32768 tier

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
