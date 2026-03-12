#!/usr/bin/env python3
"""Compare variable scoring output between qwen3:4b-instruct and qwen3:0.6b.

Sends the same batch of variables to both models and compares scores.
Requires Ollama running with both models available.
"""

import sys
import time

import litellm

# Suppress litellm noise
litellm.suppress_debug_info = True

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

# Test variables: mix of obvious keeps, obvious drops, and ambiguous ones
TEST_VARIABLES = """/no_think
Score these variables:
1. [src/config.py] DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://localhost/mydb")
2. [src/config.py] MAX_RETRIES = 3
3. [src/config.py] TIMEOUT_SECONDS = 30
4. [src/app.py] app = FastAPI(title="My API", version="1.0")
5. [src/app.py] router = APIRouter(prefix="/api/v1")
6. [src/app.py] logger = logging.getLogger(__name__)
7. [src/models.py] UserRole = Enum("UserRole", ["ADMIN", "USER", "GUEST"])
8. [src/models.py] EmailPattern = re.compile(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\\.[a-zA-Z0-9-.]+$")
9. [src/utils.py] result = process_data(input_file)
10. [src/utils.py] items = [x for x in range(10)]
11. [src/utils.py] temp = get_connection()
12. [src/db.py] engine = create_engine(DATABASE_URL, pool_size=5)
13. [src/db.py] SessionLocal = sessionmaker(autocommit=False, bind=engine)
14. [src/auth.py] SECRET_KEY = os.getenv("JWT_SECRET", "dev-secret")
15. [src/auth.py] TOKEN_EXPIRY = timedelta(hours=24)
16. [src/handlers.py] response = requests.get(url)
17. [src/handlers.py] parsed = json.loads(response.text)
18. [src/schema.py] UserSchema = TypedDict("UserSchema", {"name": str, "email": str, "role": UserRole})
19. [src/middleware.py] CORS_ORIGINS = ["http://localhost:3000", "https://myapp.com"]
20. [src/middleware.py] rate_limiter = RateLimiter(max_requests=100, window=60)"""

# Expected: items 1-8, 12-15, 18-20 should score HIGH; 9-11, 16-17 should score LOW
EXPECTED_KEEP = {1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14, 15, 18, 19, 20}
EXPECTED_DROP = {9, 10, 11, 16, 17}

MODELS = [
    ("qwen3:4b-instruct", "ollama_chat/qwen3:4b-instruct"),
    ("ministral-3:3b", "ollama_chat/ministral-3:3b"),
    ("qwen3.5:2b", "ollama_chat/qwen3.5:2b"),
    ("granite4:3b", "ollama_chat/granite4:3b"),
    ("granite3.3:2b", "ollama_chat/granite3.3:2b"),
    ("phi4-mini", "ollama_chat/phi4-mini"),
    ("gemma3:4b", "ollama_chat/gemma3:4b"),
    ("llama3.2:3b", "ollama_chat/llama3.2:3b"),
    ("qwen2.5-coder:3b", "ollama_chat/qwen2.5-coder:3b"),
    # --- Previously untested installed models ---
    ("smollm3:3b", "ollama_chat/alibayram/smollm3"),
    ("falcon-h1:3b", "ollama_chat/falcon-h1-3b"),
    ("qwen3.5:0.8b", "ollama_chat/qwen3.5:0.8b"),
    # --- Granite MoE models ---
    ("granite3.1-moe:1b", "ollama_chat/granite3.1-moe:1b"),
    ("granite3.1-moe:3b", "ollama_chat/granite3.1-moe:3b"),
]

API_BASE = "http://localhost:11434"


def run_model(display_name: str, litellm_model: str) -> tuple[str, float]:
    """Run scoring on a model, return (output, elapsed_seconds)."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": TEST_VARIABLES},
    ]

    start = time.time()
    response = litellm.completion(
        model=litellm_model,
        messages=messages,
        temperature=0.1,
        max_tokens=500,
        api_base=API_BASE,
        timeout=120,
        # Disable thinking for Ollama models — prevents models from burning
        # all tokens on reasoning with empty content
        think=False,
    )
    elapsed = time.time() - start

    content = response.choices[0].message.content or ""

    # Fallback: if content is empty, check reasoning_content (some thinking
    # models put everything there when thinking suppression is ignored)
    if not content.strip():
        reasoning = getattr(response.choices[0].message, "reasoning_content", None)
        if reasoning:
            content = reasoning

    # Strip <think> tags if present
    import re
    output = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

    return output, elapsed


def parse_scores(output: str) -> dict[int, tuple[int, int, int, int]]:
    """Parse output into {index: (c, a, d, g)} scores."""
    import re
    scores = {}
    for match in re.finditer(r"(\d+)\.\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)", output):
        idx = int(match.group(1))
        scores[idx] = (
            int(match.group(2)),
            int(match.group(3)),
            int(match.group(4)),
            int(match.group(5)),
        )
    return scores


def main():
    threshold = 5
    print(f"Threshold: max score >= {threshold} → KEEP\n")
    print(f"{'Variable':<6} {'Expected':<10}", end="")
    for name, _ in MODELS:
        print(f" {name:<30}", end="")
    print()
    print("-" * (16 + 30 * len(MODELS)))

    all_results = {}
    for display_name, litellm_model in MODELS:
        print(f"\nRunning {display_name}...", end="", flush=True)
        try:
            output, elapsed = run_model(display_name, litellm_model)
            scores = parse_scores(output)
            all_results[display_name] = (scores, elapsed, output)
            print(f" done ({elapsed:.1f}s, {len(scores)}/20 parsed)")
        except Exception as e:
            print(f" FAILED: {e}")
            all_results[display_name] = ({}, 0, str(e))

    # Print comparison table
    print(f"\n{'='*80}")
    print(f"{'Var':<5} {'Expected':<9}", end="")
    for name, _ in MODELS:
        print(f" {'Scores':<14} {'Decision':<8}", end="")
    print()
    print("-" * (14 + 22 * len(MODELS)))

    for i in range(1, 21):
        expected = "KEEP" if i in EXPECTED_KEEP else "DROP"
        print(f"{i:<5} {expected:<9}", end="")

        for display_name, _ in MODELS:
            scores, _, _ = all_results.get(display_name, ({}, 0, ""))
            if i in scores:
                c, a, d, g = scores[i]
                max_s = max(c, a, d, g)
                decision = "KEEP" if max_s >= threshold else "DROP"
                match = "✓" if (decision == expected) else "✗"
                print(f" {c},{a},{d},{g:<10} {decision} {match}  ", end="")
            else:
                print(f" {'MISSING':<14} {'?':<8}", end="")
        print()

    # Summary
    print(f"\n{'='*80}")
    print("Summary:")
    for display_name, _ in MODELS:
        scores, elapsed, output = all_results.get(display_name, ({}, 0, ""))
        if not scores:
            print(f"  {display_name}: FAILED")
            continue

        correct = 0
        total = 0
        for i in range(1, 21):
            if i in scores:
                total += 1
                c, a, d, g = scores[i]
                max_s = max(c, a, d, g)
                decision = "KEEP" if max_s >= threshold else "DROP"
                expected = "KEEP" if i in EXPECTED_KEEP else "DROP"
                if decision == expected:
                    correct += 1

        parsed_pct = len(scores) / 20 * 100
        accuracy = correct / total * 100 if total > 0 else 0
        print(f"  {display_name}: {correct}/{total} correct ({accuracy:.0f}%), "
              f"{len(scores)}/20 parsed ({parsed_pct:.0f}%), {elapsed:.1f}s")

    # Print raw outputs
    print(f"\n{'='*80}")
    print("Raw outputs:")
    for display_name, _ in MODELS:
        _, _, output = all_results.get(display_name, ({}, 0, ""))
        print(f"\n--- {display_name} ---")
        print(output)


if __name__ == "__main__":
    main()
