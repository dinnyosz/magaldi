"""Prompt templates for LLM-based variable scoring."""

from __future__ import annotations

SYSTEM_PROMPT = """\
You are scoring variables for a coding agent's search index.
For each variable, decide: KEEP or DROP.

Output ONLY numbered lines. Format: N. KEEP or N. DROP
Never echo the variable code.

KEEP: module-level constants, config values, string templates/prompts, \
framework instances (Flask, FastAPI app), DB connections/engines, loggers, \
type aliases, enums, compiled regex, __all__ exports, sentinels, \
configuration dicts/lists, query strings, decorators

DROP: loop counters, temp variables, function/method call results, \
short names (i, j, x, tmp, res, val, err, _), intermediate computations, \
local assignments inside functions, generic assignments like \
result = func(), data = obj.method(), response = requests.get()

Most variables should be DROP. Only ~30% are worth keeping.

Example input:
1. [src/app.py] MAX_RETRIES = 3
2. [src/app.py] result = process_items(data)

Correct output:
1. KEEP
2. DROP\
"""


def build_user_prompt(
    variables: list[tuple[int, str, str, str]],
    token_budget: int = 2048,
) -> str:
    """Build the user prompt listing variables to score.

    Variables should already be batched to fit within the token budget
    (by ``_build_batches`` in production or ``build_training_batches`` in
    training). This function only truncates individual variables whose
    raw code alone exceeds ``max_code_chars``.

    Args:
        variables: List of (index, file_path, name, raw_code) tuples.
        token_budget: Total context window size. Per-variable truncation
            limit is derived as the content portion of this budget:
            ``(token_budget - overhead) * 4 - safety_margin``.
            The overall batch size is controlled upstream.

    Returns:
        Formatted user prompt.
    """
    # Per-variable truncation limit: prevent a single huge variable from
    # consuming the entire budget. Subtract fixed overhead (~270 tokens for
    # system prompt + header + output base) before converting to chars.
    max_code_chars = max(200, (token_budget - 270) * 4 - 200)

    # /no_think disables Qwen3's thinking mode via soft-switch.
    # This is a fallback for servers that don't honor chat_template_kwargs.
    lines = ["/no_think\nClassify these variables:"]
    for idx, file_path, _name, raw_code in variables:
        code = raw_code.replace("\n", " ").strip()
        if len(code) > max_code_chars:
            code = code[: max_code_chars - 3] + "..."
        lines.append(f"{idx}. [{file_path}] {code}")

    return "\n".join(lines)
