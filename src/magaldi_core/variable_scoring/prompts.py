"""Prompt templates for LLM-based variable scoring."""

from __future__ import annotations

SYSTEM_PROMPT = """\
You are scoring variables for a coding agent's search index. The agent uses this \
index to find relevant code when navigating a codebase. Score each variable on \
seven dimensions (1-10) as comma-separated integers:
config_value,architectural_role,data_definition,general_usefulness,value_complexity,naming_quality,scope_significance

Ask: "If a coding agent searched for how this system works, would finding this \
variable help?" If no, score 1,1,1,1,1,1,1.

Scoring dimensions:
- config_value: Configuration, feature flag, tuning parameter, URL, path, prompt \
template, SQL/query string, default value, __all__ export list
- architectural_role: Infrastructure (DB connection, router, logger, app instance, \
middleware, compiled regex, decorator, sentinel)
- data_definition: Data structure, schema, type alias, enum, named tuple, mapping, \
protocol, TypeVar
- general_usefulness: Would a coding agent benefit from finding this while working?
- value_complexity: How complex/interesting is the assigned value? Simple literal=1, \
function call=3, multi-element collection/dict/config=7, complex expression/template=9
- naming_quality: How descriptive/self-documenting is the name? Single letter=1, \
abbreviation=3, clear descriptive name=7, fully qualified domain name=9
- scope_significance: Module-level constant=9, class attribute=6, function local=2, \
loop/temp variable=1

Rules:
- Output ONLY the number and seven scores per line. Never echo the variable code.
- Format: N. score,score,score,score,score,score,score
- Most variables should score LOW. Only ~30% of variables are worth keeping.
- Score 1,1,1,1,1,1,1 for: loop counters, temp variables, function/method call results, \
short names (i, j, x, tmp, res, val, err, _), intermediate computations, \
local assignments inside functions
- Score 1,1,1,1,1,1,1 for: generic assignments like result = func(), data = obj.method(), \
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
1. 9,1,1,8,2,8,9
2. 1,10,1,9,3,7,9
3. 1,1,1,1,2,2,2
4. 1,9,1,9,4,7,9
5. 1,9,1,7,3,7,9
6. 1,1,1,1,2,1,2\
"""


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

    # /no_think disables Qwen3's thinking mode via soft-switch.
    # This is a fallback for servers that don't honor chat_template_kwargs.
    lines = ["/no_think\nScore these variables:"]
    for idx, file_path, _name, raw_code in variables:
        code = raw_code.replace("\n", " ").strip()
        if len(code) > max_code_chars:
            code = code[: max_code_chars - 3] + "..."
        lines.append(f"{idx}. [{file_path}] {code}")

    return "\n".join(lines)
