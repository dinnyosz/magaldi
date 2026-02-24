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
- Most variables should score LOW. Only ~30% of variables are worth keeping.
- Score 1,1,1,1 for: loop counters, temp variables, function/method call results, \
short names (i, j, x, tmp, res, val, err, _)
- Score 1,1,1,1 for: generic assignments like result = func(), data = obj.method(), \
response = requests.get(), items = process()
- Score HIGH only for: module-level constants (MAX_RETRIES=3), framework instances \
(app=Flask(__name__)), DB connections, loggers, type aliases, enums, compiled patterns

Examples (HIGH):
1. [src/config.py] MAX_RETRIES = 3 → 1. 9,1,1,8
2. [src/app.py] app = Flask(__name__) → 2. 1,10,1,9
3. [src/models.py] Status = Enum("Status", "ACTIVE INACTIVE") → 3. 1,1,9,8
4. [src/db.py] engine = create_engine(DATABASE_URL) → 4. 1,9,1,9

Examples (LOW - score 1,1,1,1):
5. [src/utils.py] result = process_items(data) → 5. 1,1,1,1
6. [src/handler.py] response = requests.get(url) → 6. 1,1,1,1
7. [src/loop.py] i = 0 → 7. 1,1,1,1
8. [src/parser.py] tmp = [] → 8. 1,1,1,1\
"""


def build_user_prompt(variables: list[tuple[int, str, str, str]]) -> str:
    """Build the user prompt listing variables to score.

    Args:
        variables: List of (index, file_path, name, raw_code) tuples.

    Returns:
        Formatted user prompt.
    """
    lines = ["Score these variables:"]
    for idx, file_path, _name, raw_code in variables:
        # Truncate long code to keep prompt compact
        code = raw_code.replace("\n", " ").strip()
        if len(code) > 120:
            code = code[:117] + "..."
        lines.append(f"{idx}. [{file_path}] {code}")

    return "\n".join(lines)
