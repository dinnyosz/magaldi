"""Context size computation utilities for KV cache optimization.

This module computes optimal context sizes (num_ctx) per element type
based on observed maximum code sizes during parsing. Using type-specific
context sizes improves KV cache efficiency for local LLM inference.
"""

from __future__ import annotations

# Context size tiers (powers of 2 for memory alignment)
CONTEXT_TIERS = [2048, 4096, 8192, 16384, 32768]

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


def compute_num_ctx(element_type: str, max_chars: int) -> int:
    """Compute optimal context size for an element type.

    Args:
        element_type: Type of code element (file, class, function, etc.)
        max_chars: Maximum character count observed for this type.

    Returns:
        Optimal num_ctx value from CONTEXT_TIERS.
    """
    # Estimate tokens: ~4 chars per token for code
    estimated_tokens = max_chars // 4
    overhead = PROMPT_OVERHEAD.get(element_type, DEFAULT_OVERHEAD)
    total_tokens = estimated_tokens + overhead

    # Find smallest tier that fits
    for tier in CONTEXT_TIERS:
        if total_tokens < tier:
            return tier

    # Fallback to largest tier
    return CONTEXT_TIERS[-1]


def compute_context_sizes(max_chars_by_type: dict[str, int]) -> dict[str, int]:
    """Compute context sizes for all element types.

    Args:
        max_chars_by_type: Dict mapping element_type to max char count.

    Returns:
        Dict mapping element_type to optimal num_ctx.
    """
    return {
        element_type: compute_num_ctx(element_type, max_chars)
        for element_type, max_chars in max_chars_by_type.items()
    }
