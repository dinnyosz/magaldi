#!/usr/bin/env python3
"""Deep test of gemma3:4b for binary keep/drop variable scoring.

Uses the same approach as production variable scoring:
- Batched variables in a single message
- Text-based output: "N. KEEP" or "N. DROP"
- Regex parsing, no format forcing
- /no_think soft-switch, <think> tag stripping

Tests:
1. Basic 8-variable batch
2. All 32 variables in one batch
3. Multiple prompt variations
4. Parallel batch throughput
5. Multi-model comparison with best prompt
"""

import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

API_BASE = "http://localhost:11434"
MODEL = "gemma3:4b"

# --- System prompts to test ---

SYSTEM_PROMPTS = {
    "production-style": (
        "You are scoring variables for a coding agent's search index.\n"
        "For each variable, decide: KEEP or DROP.\n\n"
        "Output ONLY numbered lines. Format: N. KEEP or N. DROP\n"
        "Never echo the variable code.\n\n"
        "KEEP: module-level constants, config values, string templates/prompts, "
        "framework instances, DB connections, loggers, type aliases, enums, "
        "compiled patterns, configuration dicts/lists, query strings, export lists, "
        "sentinels, middleware\n\n"
        "DROP: loop counters, temp variables, function/method call results, "
        "short names (i, j, x, tmp, res, val, err, _), intermediate computations, "
        "local assignments inside functions, generic assignments like "
        "result = func(), data = obj.method(), response = requests.get()\n\n"
        "Most variables should be DROP. Only ~30% are worth keeping.\n\n"
        "Example input:\n"
        "1. [src/app.py] MAX_RETRIES = 3\n"
        "2. [src/app.py] result = process_items(data)\n\n"
        "Correct output:\n"
        "1. KEEP\n"
        "2. DROP"
    ),
    "concise": (
        "For each variable, output: N. KEEP or N. DROP\n"
        "KEEP: constants, config, DB, loggers, types, enums, framework instances, "
        "middleware, prompts, SQL, compiled regex\n"
        "DROP: loop vars, temp, result=func(), single-letter, intermediate, locals\n"
        "~30% keep, ~70% drop. Output ONLY numbered decisions."
    ),
    "question": (
        'For each variable ask: "Would a coding agent benefit from finding this '
        'in a search index?"\n'
        "Output: N. KEEP or N. DROP\n"
        "KEEP: constants, config, infrastructure, types, schemas, framework\n"
        "DROP: loop vars, temp, result=func(), single-letter, intermediate\n"
        "Most should be DROP. Output ONLY the numbered decisions."
    ),
}

# --- Test data ---

BASIC_CASES = [
    ("src/config.py", "MAX_RETRIES = 3", True),
    ("src/app.py", "app = Flask(__name__)", True),
    ("src/app.py", "result = process_items(data)", False),
    ("src/db.py", "engine = create_engine(DATABASE_URL)", True),
    ("src/db.py", "logger = logging.getLogger(__name__)", True),
    ("src/db.py", "data = json.loads(payload)", False),
    ("src/models.py", 'UserSchema = TypedDict("UserSchema", {"name": str, "email": str})', True),
    ("src/utils.py", "i = 0", False),
]

EDGE_CASES = [
    # Clear keeps
    ("src/config.py", 'DATABASE_URL = "postgresql://localhost/mydb"', True),
    ("src/config.py", "CORS_ORIGINS = ['http://localhost:3000', 'https://myapp.com']", True),
    ("src/middleware.py", "cors = CORS(app, origins=ALLOWED_ORIGINS)", True),
    ("src/tasks.py", "celery = Celery('tasks', broker=REDIS_URL)", True),
    ("src/models.py", "Base = declarative_base()", True),
    ("src/config.py", "LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')", True),
    ("src/auth.py", "JWT_SECRET = os.environ['JWT_SECRET']", True),
    ("src/db.py", "SessionLocal = sessionmaker(bind=engine)", True),
    # Clear drops
    ("src/utils.py", "x = get_value()", False),
    ("src/utils.py", "tmp = []", False),
    ("src/tasks.py", "_ = do_something()", False),
    ("src/utils.py", "for item in items: total += item.price", False),
    ("src/app.py", "res = requests.get(url)", False),
    ("src/utils.py", "val = int(s)", False),
    ("src/utils.py", "buf = io.BytesIO()", False),
    ("src/utils.py", "n = len(items)", False),
    # Borderline / tricky
    ("src/app.py", "router = APIRouter(prefix='/api/v1')", True),
    ("src/db.py", "conn = engine.connect()", False),
    ("src/utils.py", "pattern = re.compile(r'^[a-z]+$')", True),
    ("src/app.py", "response = make_response(data)", False),
    ("src/config.py", "TIMEOUT = 30", True),
    ("src/models.py", "Status = Enum('Status', ['ACTIVE', 'INACTIVE', 'PENDING'])", True),
    ("src/utils.py", "items = list(filter(None, raw_items))", False),
    ("src/app.py", "db = SQLAlchemy(app)", True),
]

ALL_CASES = BASIC_CASES + EDGE_CASES

LABELS = {i + 1: code[:35] for i, (_, code, _) in enumerate(ALL_CASES)}

MULTI_MODELS = [
    # qwen3.5 family (0.8b, 2b, 4b available locally)
    "qwen3.5:0.8b",
    "qwen3.5:2b",
    "qwen3.5:4b",
    # Other small models for comparison
    "qwen3:1.7b",
    "qwen3:4b",
    "llama3.2:3b",
    "gemma3:4b",
]


# --- Core functions (matching production approach) ---


def _strip_think_tags(output: str) -> str:
    """Strip <think>...</think> blocks from LLM output."""
    return re.sub(r"<think>.*?</think>", "", output, flags=re.DOTALL).strip()


def _build_user_prompt(cases: list[tuple[str, str, bool]]) -> str:
    """Build user prompt matching production format."""
    lines = ["/no_think\nScore these variables:"]
    for i, (file_path, code, _) in enumerate(cases, 1):
        lines.append(f"{i}. [{file_path}] {code}")
    return "\n".join(lines)


def _parse_batch_scores(
    output: str, batch_size: int
) -> list[bool | None]:
    """Parse batched keep/drop output using regex.

    Matches production-style patterns:
    - "1. KEEP"
    - "1: DROP"
    - "1) KEEP"
    """
    output = _strip_think_tags(output)

    scores: list[bool | None] = [None] * batch_size
    pattern = re.compile(
        r"(\d+)\s*[.):]\s*(keep|drop)",
        re.IGNORECASE,
    )
    for match in pattern.finditer(output):
        idx = int(match.group(1))
        if 1 <= idx <= batch_size:
            scores[idx - 1] = match.group(2).upper() == "KEEP"

    return scores


def call_ollama(
    model: str,
    system_prompt: str,
    user_prompt: str,
    num_predict: int = 200,
) -> tuple[str, float]:
    """Call Ollama API, no format forcing."""
    start = time.time()
    resp = requests.post(
        f"{API_BASE}/api/chat",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "think": False,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": num_predict,
                "num_ctx": 2048,
            },
        },
        timeout=60,
    )
    elapsed = time.time() - start
    data = resp.json()
    content = data.get("message", {}).get("content", "")
    return content.strip(), elapsed


# --- Test runners ---


def run_batch_test(
    cases: list[tuple[str, str, bool]],
    label: str,
    system_prompt: str,
    model: str = MODEL,
) -> dict:
    """Test batched approach matching production flow."""
    user_prompt = _build_user_prompt(cases)
    num_predict = len(cases) * 10 + 30

    try:
        output, elapsed = call_ollama(model, system_prompt, user_prompt, num_predict)
    except Exception as e:
        return {
            "label": label,
            "correct": 0,
            "total": len(cases),
            "parse_fails": len(cases),
            "errors": [f"  ERROR: {e}"],
            "total_time": 0.0,
            "raw": "",
        }

    scores = _parse_batch_scores(output, len(cases))

    correct = 0
    total = len(cases)
    errors = []
    parse_fails = 0

    for i, (file_path, code, expected) in enumerate(cases):
        decision = scores[i]
        if decision is None:
            parse_fails += 1
            errors.append(f"  {i+1:2d}. {code[:35]:35s} MISSING")
        elif decision == expected:
            correct += 1
        else:
            exp = "KEEP" if expected else "DROP"
            act = "KEEP" if decision else "DROP"
            errors.append(f"  {i+1:2d}. {code[:35]:35s} expected {exp}, got {act}")

    return {
        "label": label,
        "correct": correct,
        "total": total,
        "parse_fails": parse_fails,
        "errors": errors,
        "total_time": elapsed,
        "raw": output[:400],
    }


def run_parallel_batches(
    all_cases: list[tuple[str, str, bool]],
    system_prompt: str,
    batch_size: int = 8,
    workers: int = 4,
    model: str = MODEL,
) -> dict:
    """Split cases into batches and process in parallel."""
    # Split into batches
    batches = []
    for i in range(0, len(all_cases), batch_size):
        batches.append(all_cases[i : i + batch_size])

    correct = 0
    total = len(all_cases)
    errors = []
    parse_fails = 0

    start = time.time()

    def process_batch(batch_with_idx):
        batch_idx, batch = batch_with_idx
        return batch_idx, run_batch_test(
            batch, f"batch-{batch_idx}", system_prompt, model=model
        )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(process_batch, (i, b)): i
            for i, b in enumerate(batches)
        }
        for f in as_completed(futures):
            batch_idx, result = f.result()
            correct += result["correct"]
            parse_fails += result["parse_fails"]
            errors.extend(result["errors"])

    wall_time = time.time() - start

    return {
        "label": f"parallel-{workers}w-{batch_size}b",
        "correct": correct,
        "total": total,
        "parse_fails": parse_fails,
        "errors": errors,
        "total_time": wall_time,
        "num_batches": len(batches),
    }


def main():
    print("=" * 70)
    print(f"Deep test: {MODEL} for binary keep/drop scoring")
    print("(production approach: batched, text output, regex parsing)")
    print("=" * 70)

    all_results = []

    # --- Test 1: Different prompts, basic 8 cases ---
    print(f"\n{'─' * 70}")
    print("Test 1: Prompt comparison (basic 8 cases)")
    print(f"{'─' * 70}")

    for name, prompt in SYSTEM_PROMPTS.items():
        r = run_batch_test(BASIC_CASES, f"basic-{name}", prompt)
        all_results.append(r)
        pf = f" ({r['parse_fails']} missing)" if r["parse_fails"] > 0 else ""
        status = "PERFECT" if r["correct"] == r["total"] else f"{r['correct']}/{r['total']}"
        print(f"  [{name}] {status}{pf} | {r['total_time']:.1f}s")
        for e in r["errors"]:
            print(e)
        # Show first few lines of raw output
        raw_lines = r["raw"].split("\n")[:4]
        print(f"    raw: {' | '.join(l.strip() for l in raw_lines if l.strip())}")

    # --- Test 2: Best prompt, all 32 cases ---
    print(f"\n{'─' * 70}")
    print(f"Test 2: All {len(ALL_CASES)} cases with each prompt")
    print(f"{'─' * 70}")

    best_prompt_name = None
    best_prompt_score = -1

    for name, prompt in SYSTEM_PROMPTS.items():
        r = run_batch_test(ALL_CASES, f"all-{name}", prompt)
        all_results.append(r)
        pf = f" ({r['parse_fails']} missing)" if r["parse_fails"] > 0 else ""
        status = "PERFECT" if r["correct"] == r["total"] else f"{r['correct']}/{r['total']}"
        print(f"  [{name}] {status}{pf} | {r['total_time']:.1f}s")
        for e in r["errors"]:
            print(e)

        if r["correct"] > best_prompt_score:
            best_prompt_score = r["correct"]
            best_prompt_name = name

    best_prompt = SYSTEM_PROMPTS[best_prompt_name]
    print(f"\n  Best prompt: {best_prompt_name} ({best_prompt_score}/{len(ALL_CASES)})")

    # --- Test 3: Parallel batched throughput ---
    print(f"\n{'─' * 70}")
    print(f"Test 3: Parallel batch throughput (best prompt: {best_prompt_name})")
    print(f"{'─' * 70}")

    for batch_size in [8, 16]:
        for workers in [2, 4]:
            r = run_parallel_batches(
                ALL_CASES, best_prompt, batch_size=batch_size, workers=workers
            )
            all_results.append(r)
            pf = f" ({r['parse_fails']} missing)" if r["parse_fails"] > 0 else ""
            status = "PERFECT" if r["correct"] == r["total"] else f"{r['correct']}/{r['total']}"
            print(
                f"  batch_size={batch_size} workers={workers}: "
                f"{status}{pf} | {r['total_time']:.1f}s wall | "
                f"{r['num_batches']} batches"
            )
            if r["errors"]:
                for e in r["errors"][:5]:
                    print(e)
                if len(r["errors"]) > 5:
                    print(f"    ... and {len(r['errors']) - 5} more")

    # --- Test 4: Multi-model comparison with best prompt ---
    print(f"\n{'─' * 70}")
    print(f"Test 4: Multi-model comparison (best prompt: {best_prompt_name})")
    print(f"{'─' * 70}")

    model_results = []
    for model in MULTI_MODELS:
        r = run_batch_test(ALL_CASES, f"model-{model}", best_prompt, model=model)
        model_results.append(r)
        pf = f" ({r['parse_fails']} missing)" if r["parse_fails"] > 0 else ""
        status = "PERFECT" if r["correct"] == r["total"] else f"{r['correct']}/{r['total']}"
        print(f"  {model:25s} {status}{pf} | {r['total_time']:.1f}s")
        for e in r["errors"]:
            print(e)

    # --- Summary ---
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Test':35s} | {'Score':8s} | {'Fails':6s} | {'Time':8s} | {'Per var':8s}")
    print(f"{'-' * 35}-+-{'-' * 8}-+-{'-' * 6}-+-{'-' * 8}-+-{'-' * 8}")
    for r in all_results:
        score = f"{r['correct']}/{r['total']}"
        if r["correct"] == r["total"]:
            score += " *"
        pf = str(r["parse_fails"]) if r["parse_fails"] > 0 else "-"
        t = f"{r['total_time']:.1f}s"
        per_var = f"{r['total_time'] / r['total'] * 1000:.0f}ms"
        print(f"{r['label']:35s} | {score:8s} | {pf:6s} | {t:8s} | {per_var:8s}")

    print(f"\n  Multi-model comparison:")
    print(f"  {'Model':25s} | {'Score':8s} | {'Fails':6s} | {'Time':8s}")
    print(f"  {'-' * 25}-+-{'-' * 8}-+-{'-' * 6}-+-{'-' * 8}")
    for r in model_results:
        model_name = r["label"].replace("model-", "")
        score = f"{r['correct']}/{r['total']}"
        if r["correct"] == r["total"]:
            score += " *"
        pf = str(r["parse_fails"]) if r["parse_fails"] > 0 else "-"
        t = f"{r['total_time']:.1f}s"
        print(f"  {model_name:25s} | {score:8s} | {pf:6s} | {t:8s}")

    print(f"\n* = perfect score")


if __name__ == "__main__":
    main()
