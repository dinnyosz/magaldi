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
- One line per variable, prefixed with its number
- Scores only, no explanations
- Most variables should score LOW. Only ~30% of variables are worth keeping.
- Score 1,1,1,1 for: loop counters, temp variables, function/method call results, \
short names (i, j, x, tmp, res, val, err, _), intermediate computations, \
local assignments inside functions
- Score 1,1,1,1 for: generic assignments like result = func(), data = obj.method(), \
response = requests.get(), items = process(), client = Client()
- Score HIGH only for: module-level constants, string templates/prompts, framework \
instances, DB connections, loggers, type aliases, enums, compiled patterns, \
configuration dicts/lists, query strings, export lists, sentinels

Examples (HIGH - keep):
1. MAX_RETRIES = 3 → 1. 9,1,1,8
2. app = Flask(__name__) → 2. 1,10,1,9
3. Status = Enum("Status", "ACTIVE INACTIVE") → 3. 1,1,9,8
4. engine = create_engine(DATABASE_URL) → 4. 1,9,1,9
5. SYSTEM_PROMPT = "You are a helpful assistant..." → 5. 9,1,1,9
6. PATTERN = re.compile(r"\\d+\\.\\s*") → 6. 1,8,1,7
7. ENDPOINTS = {"/api/users": handle_users, ...} → 7. 9,1,8,9
8. logger = logging.getLogger(__name__) → 8. 1,9,1,7
9. __all__ = ["parse", "discover", "index"] → 9. 8,1,1,8
10. QUERY = "(function_definition name: (identifier) @name)" → 10. 9,1,1,9
11. MISSING = object() → 11. 1,7,1,6
12. UserSchema = z.object({ name: z.string(), ... }) → 12. 1,1,10,9

Examples (LOW - drop, score 1,1,1,1):
13. result = process_items(data) → 13. 1,1,1,1
14. response = requests.get(url) → 14. 1,1,1,1
15. i = 0 → 15. 1,1,1,1
16. tmp = [] → 16. 1,1,1,1
17. client = HttpClient(base_url) → 17. 1,1,1,1
18. data = json.loads(payload) → 18. 1,1,1,1
19. count = len(items) → 19. 1,1,1,1
20. key = f"user:{user_id}" → 20. 1,1,1,1\
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
