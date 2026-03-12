#!/usr/bin/env python3
"""Head-to-head: binary KEEP/DROP vs 7-dimension scoring.

Both approaches ultimately produce the same decision (keep or drop).
This compares:
- Accuracy: which approach makes better keep/drop decisions?
- Speed: tokens in/out, time per batch
- Reliability: parse success rate
- Consistency: same result across runs

Uses production-style prompts for both, no format forcing.
"""

import re
import time

import requests

API_BASE = "http://localhost:11434"

# ═══════════════════════════════════════════════════════════════
# System prompts
# ═══════════════════════════════════════════════════════════════

BINARY_PROMPT = (
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
)

# This is the actual production prompt from prompts.py
DIMENSION_PROMPT = """\
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

# ═══════════════════════════════════════════════════════════════
# Test data
# ═══════════════════════════════════════════════════════════════

# (file_path, code, expected_keep)
TEST_CASES = [
    # KEEP: config
    ("src/config.py", "MAX_RETRIES = 3", True),
    ("src/config.py", 'DATABASE_URL = "postgresql://localhost/mydb"', True),
    ("src/config.py", "CORS_ORIGINS = ['http://localhost:3000', 'https://myapp.com']", True),
    ("src/config.py", "TIMEOUT = 30", True),
    ("src/config.py", "LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')", True),
    ("src/config.py", "MAX_UPLOAD_SIZE = 10 * 1024 * 1024", True),
    ("src/auth.py", "JWT_SECRET = os.environ['JWT_SECRET']", True),
    ("src/config.py", "API_VERSION = 'v2'", True),
    # KEEP: infra
    ("src/app.py", "app = Flask(__name__)", True),
    ("src/db.py", "engine = create_engine(DATABASE_URL)", True),
    ("src/db.py", "logger = logging.getLogger(__name__)", True),
    ("src/db.py", "SessionLocal = sessionmaker(bind=engine)", True),
    ("src/middleware.py", "cors = CORS(app, origins=ALLOWED_ORIGINS)", True),
    ("src/tasks.py", "celery = Celery('tasks', broker=REDIS_URL)", True),
    ("src/app.py", "router = APIRouter(prefix='/api/v1')", True),
    ("src/app.py", "db = SQLAlchemy(app)", True),
    # KEEP: types
    ("src/models.py", 'UserSchema = TypedDict("UserSchema", {"name": str, "email": str})', True),
    ("src/models.py", "Base = declarative_base()", True),
    ("src/models.py", "Status = Enum('Status', ['ACTIVE', 'INACTIVE', 'PENDING'])", True),
    # KEEP: patterns
    ("src/utils.py", "pattern = re.compile(r'^[a-z]+$')", True),
    ("src/app.py", "__all__ = ['app', 'create_app', 'db']", True),
    # DROP: loop/temp
    ("src/utils.py", "i = 0", False),
    ("src/utils.py", "x = get_value()", False),
    ("src/utils.py", "tmp = []", False),
    ("src/tasks.py", "_ = do_something()", False),
    ("src/utils.py", "n = len(items)", False),
    ("src/utils.py", "val = int(s)", False),
    ("src/utils.py", "j = i + 1", False),
    # DROP: generic call results
    ("src/app.py", "result = process_items(data)", False),
    ("src/db.py", "data = json.loads(payload)", False),
    ("src/app.py", "res = requests.get(url)", False),
    ("src/app.py", "response = make_response(data)", False),
    ("src/utils.py", "items = list(filter(None, raw_items))", False),
    ("src/utils.py", "buf = io.BytesIO()", False),
    ("src/utils.py", "output = func(input_data)", False),
    # DROP: intermediate
    ("src/utils.py", "for item in items: total += item.price", False),
    ("src/utils.py", "lines = content.split('\\n')", False),
    ("src/utils.py", "cleaned = text.strip().lower()", False),
    # Borderline
    ("src/db.py", "conn = engine.connect()", False),
    ("src/utils.py", "lock = threading.Lock()", True),
    ("src/config.py", "DEBUG = True", True),
]

MODELS = [
    "qwen3.5:2b",
    "qwen3.5:4b",
    "gemma3:4b",
]


# ═══════════════════════════════════════════════════════════════
# Parsers
# ═══════════════════════════════════════════════════════════════


def _strip_think(output: str) -> str:
    return re.sub(r"<think>.*?</think>", "", output, flags=re.DOTALL).strip()


def parse_binary(output: str, batch_size: int) -> list[bool | None]:
    """Parse 'N. KEEP' / 'N. DROP' output."""
    output = _strip_think(output)
    scores: list[bool | None] = [None] * batch_size
    pattern = re.compile(r"(\d+)\s*[.):]\s*(keep|drop)", re.IGNORECASE)
    for match in pattern.finditer(output):
        idx = int(match.group(1))
        if 1 <= idx <= batch_size:
            scores[idx - 1] = match.group(2).upper() == "KEEP"
    return scores


def parse_dimensions(output: str, batch_size: int, threshold: int = 5) -> list[bool | None]:
    """Parse 'N. s,s,s,s,s,s,s' output and apply max_score >= threshold."""
    output = _strip_think(output)
    scores: list[bool | None] = [None] * batch_size

    # 7-dimension pattern
    pat7 = re.compile(
        r"(\d+)\.\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)"
    )
    matched = set()
    for match in pat7.finditer(output):
        idx = int(match.group(1))
        if 1 <= idx <= batch_size:
            dims = [min(10, max(1, int(match.group(g)))) for g in range(2, 9)]
            scores[idx - 1] = max(dims) >= threshold
            matched.add(idx)

    # 4-dimension fallback
    pat4 = re.compile(
        r"(\d+)\.\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)(?!\s*,\s*\d)"
    )
    for match in pat4.finditer(output):
        idx = int(match.group(1))
        if 1 <= idx <= batch_size and idx not in matched:
            dims = [min(10, max(1, int(match.group(g)))) for g in range(2, 6)]
            scores[idx - 1] = max(dims) >= threshold

    return scores


# ═══════════════════════════════════════════════════════════════
# Test runner
# ═══════════════════════════════════════════════════════════════


def call_ollama(model: str, system_prompt: str, user_prompt: str, num_predict: int) -> tuple[str, float]:
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
            "options": {"temperature": 0.1, "num_predict": num_predict, "num_ctx": 2048},
        },
        timeout=120,
    )
    elapsed = time.time() - start
    data = resp.json()
    content = data.get("message", {}).get("content", "")
    return content.strip(), elapsed


def build_user_prompt(cases: list) -> str:
    lines = ["/no_think\nScore these variables:"]
    for i, (fp, code, _) in enumerate(cases, 1):
        lines.append(f"{i}. [{fp}] {code}")
    return "\n".join(lines)


def run_test(
    cases: list,
    model: str,
    approach: str,  # "binary" or "dimensions"
    batch_size: int = 8,
) -> dict:
    """Run a full test with given approach, splitting into batches."""
    if approach == "binary":
        system_prompt = BINARY_PROMPT
        parser = parse_binary
        tokens_per_var = 10  # "N. KEEP\n" ~= 5-8 tokens
    else:
        system_prompt = DIMENSION_PROMPT
        parser = parse_dimensions
        tokens_per_var = 25  # "N. 9,2,1,8,3,7,9\n" ~= 20-25 tokens

    total_correct = 0
    total_total = 0
    total_fp = 0
    total_fn = 0
    total_missing = 0
    total_time = 0.0
    total_output_chars = 0
    errors = []

    # Process in batches
    for start in range(0, len(cases), batch_size):
        batch = cases[start : start + batch_size]
        user_prompt = build_user_prompt(batch)
        num_predict = len(batch) * tokens_per_var + 30

        try:
            output, elapsed = call_ollama(model, system_prompt, user_prompt, num_predict)
        except Exception as e:
            errors.append(f"  BATCH ERROR: {e}")
            total_missing += len(batch)
            total_total += len(batch)
            continue

        total_time += elapsed
        total_output_chars += len(output)

        scores = parser(output, len(batch))
        for j, (fp, code, expected) in enumerate(batch):
            total_total += 1
            decision = scores[j]

            if decision is None:
                total_missing += 1
                effective = True  # default to keep
            else:
                effective = decision

            if effective == expected:
                total_correct += 1
            else:
                exp = "KEEP" if expected else "DROP"
                act = "KEEP" if effective else "DROP"
                dflt = " (def)" if decision is None else ""
                errors.append(f"  {code[:40]:40s} {exp}->{act}{dflt}")
                if not expected and effective:
                    total_fp += 1
                elif expected and not effective:
                    total_fn += 1

    return {
        "model": model,
        "approach": approach,
        "correct": total_correct,
        "total": total_total,
        "accuracy": total_correct / total_total if total_total else 0,
        "false_pos": total_fp,
        "false_neg": total_fn,
        "missing": total_missing,
        "time": total_time,
        "output_chars": total_output_chars,
        "prompt_chars": len(system_prompt),
        "errors": errors,
    }


def main():
    n = len(TEST_CASES)
    keeps = sum(1 for _, _, e in TEST_CASES if e)
    drops = n - keeps

    print("=" * 70)
    print("Head-to-head: Binary KEEP/DROP vs 7-Dimension Scoring")
    print("=" * 70)
    print(f"\nTest cases: {n} ({keeps} keep, {drops} drop)")
    print(f"Binary prompt:    {len(BINARY_PROMPT):4d} chars")
    print(f"Dimension prompt: {len(DIMENSION_PROMPT):4d} chars")
    print(f"Batch size: 8, default-to-KEEP for missing scores")

    all_results = []

    for model in MODELS:
        print(f"\n{'═' * 70}")
        print(f"Model: {model}")
        print(f"{'═' * 70}")

        for approach in ["binary", "dimensions"]:
            # Run 3 times for stability
            runs = []
            for run_num in range(3):
                r = run_test(TEST_CASES, model, approach)
                runs.append(r)

            # Average
            avg_acc = sum(r["accuracy"] for r in runs) / len(runs)
            avg_fp = sum(r["false_pos"] for r in runs) / len(runs)
            avg_fn = sum(r["false_neg"] for r in runs) / len(runs)
            avg_miss = sum(r["missing"] for r in runs) / len(runs)
            avg_time = sum(r["time"] for r in runs) / len(runs)
            avg_output = sum(r["output_chars"] for r in runs) / len(runs)

            best = max(runs, key=lambda r: r["accuracy"])
            worst = min(runs, key=lambda r: r["accuracy"])

            label = f"{model} / {approach}"
            print(f"\n  [{approach}]")
            print(f"    Accuracy:  {avg_acc*100:.1f}% avg  (best {best['accuracy']*100:.0f}%, worst {worst['accuracy']*100:.0f}%)")
            print(f"    FP (kept wrong):  {avg_fp:.1f}  |  FN (dropped wrong): {avg_fn:.1f}  |  Missing: {avg_miss:.1f}")
            print(f"    Time:      {avg_time:.1f}s avg  |  Output: {avg_output:.0f} chars avg")

            if best["errors"]:
                print(f"    Errors (best run):")
                for e in best["errors"][:8]:
                    print(f"    {e}")
                if len(best["errors"]) > 8:
                    print(f"      ... +{len(best['errors']) - 8} more")

            all_results.append({
                "model": model,
                "approach": approach,
                "avg_acc": avg_acc,
                "best_acc": best["accuracy"],
                "worst_acc": worst["accuracy"],
                "avg_fp": avg_fp,
                "avg_fn": avg_fn,
                "avg_miss": avg_miss,
                "avg_time": avg_time,
                "avg_output": avg_output,
                "prompt_chars": runs[0]["prompt_chars"],
            })

    # ═══════════════════════════════════════════════════════════════
    # Summary table
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'═' * 70}")
    print("SUMMARY (3-run average, default-to-KEEP)")
    print(f"{'═' * 70}")
    print(
        f"  {'Model + Approach':30s} | {'Acc':6s} | {'Best':5s} | "
        f"{'FP':4s} | {'FN':4s} | {'Miss':5s} | {'Time':6s} | {'Out':6s} | {'Prompt':6s}"
    )
    print(
        f"  {'-'*30}-+-{'-'*6}-+-{'-'*5}-+-"
        f"{'-'*4}-+-{'-'*4}-+-{'-'*5}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}"
    )
    for r in all_results:
        label = f"{r['model']} / {r['approach']}"
        print(
            f"  {label:30s} | {r['avg_acc']*100:4.1f}% | {r['best_acc']*100:3.0f}% | "
            f"{r['avg_fp']:4.1f} | {r['avg_fn']:4.1f} | {r['avg_miss']:5.1f} | "
            f"{r['avg_time']:5.1f}s | {r['avg_output']:5.0f} | {r['prompt_chars']:5d}"
        )

    # Winner per model
    print(f"\n  Per-model winner:")
    for model in MODELS:
        model_results = [r for r in all_results if r["model"] == model]
        binary = next(r for r in model_results if r["approach"] == "binary")
        dims = next(r for r in model_results if r["approach"] == "dimensions")
        if binary["avg_acc"] > dims["avg_acc"]:
            delta = (binary["avg_acc"] - dims["avg_acc"]) * 100
            print(f"    {model}: binary wins by {delta:.1f}pp")
        elif dims["avg_acc"] > binary["avg_acc"]:
            delta = (dims["avg_acc"] - binary["avg_acc"]) * 100
            print(f"    {model}: dimensions wins by {delta:.1f}pp")
        else:
            print(f"    {model}: tie")
        speed = binary["avg_time"] / dims["avg_time"] if dims["avg_time"] > 0 else 0
        print(f"      speed: binary {binary['avg_time']:.1f}s vs dims {dims['avg_time']:.1f}s ({speed:.1f}x)")


if __name__ == "__main__":
    main()
