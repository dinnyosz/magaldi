"""Prompt templates for LLM-based variable scoring."""

from __future__ import annotations

SYSTEM_PROMPT = """\
You are scoring variables for a coding agent's search index. The agent uses this \
index to find relevant code when navigating a codebase. Score each variable on \
four dimensions (1-10) as comma-separated integers:
config_value,architectural_role,data_definition,general_usefulness

Ask: "If a coding agent searched for how this system works, would finding this \
variable help?" If no, score 1,1,1,1.

Scoring dimensions:
- config_value: Configuration, feature flag, tuning parameter, URL, path, prompt \
template, SQL/query string, default value, __all__ export list
- architectural_role: Infrastructure (DB connection, router, logger, app instance, \
middleware, compiled regex, decorator, sentinel)
- data_definition: Data structure, schema, type alias, enum, named tuple, mapping, \
protocol, TypeVar
- general_usefulness: Would a coding agent benefit from finding this while working?

Rules:
- Output ONLY the number and four scores per line. Never echo the variable code.
- Format: N. score,score,score,score
- Most variables should score LOW. Only ~30% of variables are worth keeping.
- Score 1,1,1,1 for: loop counters, temp variables, function/method call results, \
short names (i, j, x, tmp, res, val, err, _), intermediate computations, \
local assignments inside functions
- Score 1,1,1,1 for: generic assignments like result = func(), data = obj.method(), \
response = requests.get(), items = process(), client = Client()
- Score HIGH only for: module-level constants, string templates/prompts, framework \
instances, DB connections, loggers, type aliases, enums, compiled patterns, \
configuration dicts/lists, query strings, export lists, sentinels

Example input:
1. [src/app.py] MAX_RETRIES = 3
2. [src/app.py] app = Flask(__name__)
3. [src/app.py] result = process_items(data)
4. [src/db.py] engine = create_engine(DATABASE_URL)
5. [src/db.py] logger = logging.getLogger(__name__)
6. [src/db.py] data = json.loads(payload)

Correct output (scores only, no code):
1. 9,1,1,8
2. 1,10,1,9
3. 1,1,1,1
4. 1,9,1,9
5. 1,9,1,7
6. 1,1,1,1\
"""


def build_user_prompt(variables: list[tuple[int, str, str, str]]) -> str:
    """Build the user prompt listing variables to score.

    Args:
        variables: List of (index, file_path, name, raw_code) tuples.

    Returns:
        Formatted user prompt.
    """
    # /no_think disables Qwen3's thinking mode via soft-switch.
    # This is a fallback for servers that don't honor chat_template_kwargs.
    lines = ["/no_think\nScore these variables:"]
    for idx, file_path, _name, raw_code in variables:
        # Truncate long code to keep prompt compact
        code = raw_code.replace("\n", " ").strip()
        if len(code) > 300:
            code = code[:297] + "..."
        lines.append(f"{idx}. [{file_path}] {code}")

    return "\n".join(lines)
