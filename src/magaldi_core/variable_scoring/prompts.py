"""Prompt templates for LLM-based variable scoring."""

from __future__ import annotations

from typing import Any

SYSTEM_PROMPT = """\
Return JSON: {"1":{"cv":N,"ar":N,"dd":N,"gu":N,"vc":N,"nq":N,"ss":N},...}
Score 1-10: cv=config ar=architecture dd=data_def gu=usefulness vc=complexity nq=naming ss=scope
LOW(1-3): loop vars, temp, generic results, single-letter names. HIGH(7-10): constants, framework, DB, loggers, types
KEEP: {"cv":9,"ar":1,"dd":1,"gu":8,"vc":1,"nq":8,"ss":9} DROP: {"cv":1,"ar":1,"dd":1,"gu":1,"vc":1,"nq":1,"ss":1}\
"""

# JSON schema for structured output — enforces format at the inference level.
# Each variable gets 7 integer scores keyed by its 1-based index.
SCORING_JSON_SCHEMA: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "variable_scores",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": {
                "type": "object",
                "properties": {
                    "cv": {"type": "integer"},
                    "ar": {"type": "integer"},
                    "dd": {"type": "integer"},
                    "gu": {"type": "integer"},
                    "vc": {"type": "integer"},
                    "nq": {"type": "integer"},
                    "ss": {"type": "integer"},
                },
                "required": ["cv", "ar", "dd", "gu", "vc", "nq", "ss"],
            },
        },
    },
}

# Compact key → VariableScore field name mapping
_SCORE_KEY_MAP: dict[str, str] = {
    "cv": "config_value",
    "ar": "architectural_role",
    "dd": "data_definition",
    "gu": "general_usefulness",
    "vc": "value_complexity",
    "nq": "naming_quality",
    "ss": "scope_significance",
}


def build_user_prompt(
    variables: list[tuple[int, str, str, str]],
    token_budget: int = 1200,
) -> str:
    """Build the user prompt listing variables to score.

    Variables should already be batched to fit within the token budget
    (by ``_build_batches`` in production or ``build_training_batches`` in
    training). This function only truncates individual variables whose
    raw code alone exceeds ``max_code_chars``.

    Args:
        variables: List of (index, file_path, name, raw_code) tuples.
        token_budget: Maximum code chars per individual variable
            (``token_budget * 4 - 200``). Variables exceeding this are
            truncated. The overall batch size is controlled upstream.

    Returns:
        Formatted user prompt.
    """
    # Per-variable truncation limit: prevent a single huge variable from
    # consuming the entire budget. Batch-level packing is done upstream.
    max_code_chars = token_budget * 4 - 200  # ~4 chars/token, minus overhead

    lines = ["Score these variables:"]
    for idx, file_path, _name, raw_code in variables:
        code = raw_code.replace("\n", " ").strip()
        if len(code) > max_code_chars:
            code = code[: max_code_chars - 3] + "..."
        lines.append(f"{idx}. [{file_path}] {code}")

    return "\n".join(lines)
