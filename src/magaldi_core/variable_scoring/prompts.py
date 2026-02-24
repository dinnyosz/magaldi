"""Prompt templates for LLM-based variable scoring."""

from __future__ import annotations

SYSTEM_PROMPT = """\
Score variables for code discovery usefulness. For each variable, return four \
scores (1-10) as comma-separated integers on one line:
config_value,architectural_role,data_definition,general_usefulness

Scoring dimensions:
- config_value: Configuration constant, feature flag, tuning parameter, URL, path
- architectural_role: Infrastructure (DB connection, router, logger, singleton, app instance, middleware)
- data_definition: Data structure, schema, type alias, enum value, named tuple
- general_usefulness: Would a developer searching the codebase benefit from finding this?

Rules:
- One line per variable, prefixed with its number
- Scores only, no explanations
- Loop counters and temporaries score 1 on all dimensions
- Module-level constants typically score high on config_value
- Framework instances (Flask app, Express router) score high on architectural_role\
"""


def build_user_prompt(variables: list[tuple[int, str, str, str]]) -> str:
    """Build the user prompt listing variables to score.

    Args:
        variables: List of (index, file_path, name, raw_code) tuples.

    Returns:
        Formatted user prompt.
    """
    lines = ["Score these variables:"]
    for idx, file_path, name, raw_code in variables:
        # Truncate long code to keep prompt compact
        code = raw_code.replace("\n", " ").strip()
        if len(code) > 120:
            code = code[:117] + "..."
        lines.append(f"{idx}. [{file_path}] {code}")

    return "\n".join(lines)
