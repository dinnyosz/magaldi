"""Prompt templates for LLM-based variable scoring."""

from __future__ import annotations

from typing import Any

SYSTEM_PROMPT = """\
Score variables for a code search index. For each variable, score seven dimensions (1-10).

Ask: "Would a coding agent benefit from finding this variable?" If no, score all 1s.

Dimensions:
- config_value: Config, feature flag, URL, path, prompt template, SQL, __all__
- architectural_role: DB connection, router, logger, app instance, middleware, sentinel
- data_definition: Schema, type alias, enum, named tuple, mapping, protocol
- general_usefulness: Would a coding agent benefit from finding this?
- value_complexity: Simple literal=1, function call=3, collection/config=7, complex=9
- naming_quality: Single letter=1, abbreviation=3, descriptive=7, domain-specific=9
- scope_significance: Module-level=9, class attribute=6, function local=2, temp=1

Most variables should score LOW (~30% are worth keeping).
LOW: loop counters, temp vars, generic call results (result=func(), data=obj.method())
HIGH: module constants, framework instances, DB connections, loggers, type aliases, \
config dicts, query strings, export lists\
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
