#!/usr/bin/env python3
"""Fine-tune the binary keep/drop prompt to improve accuracy.

The dimension prompt works better for 2b because it gives the model
structured guidance (7 axes to think about). Can we get the same
benefit with a binary prompt that teaches the model what to look for?

Tests many prompt variations to find the best one.
"""

import re
import time

import requests

API_BASE = "http://localhost:11434"
MODELS = ["qwen3.5:2b", "qwen3.5:4b"]

# ═══════════════════════════════════════════════════════════════
# Prompt variations
# ═══════════════════════════════════════════════════════════════

PROMPTS = {
    # Baseline: the binary prompt from previous tests
    "baseline": (
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

    # Guided: teach the model to check multiple aspects (like dimensions but binary)
    "checklist": (
        "For each variable, check these questions and output KEEP or DROP:\n"
        "- Is it a configuration value, constant, URL, path, or flag? -> KEEP\n"
        "- Is it infrastructure (DB, logger, app instance, middleware, cache)? -> KEEP\n"
        "- Is it a type definition, schema, enum, or protocol? -> KEEP\n"
        "- Is it a compiled pattern, query string, or template? -> KEEP\n"
        "- Is it module-level with a descriptive name? -> likely KEEP\n"
        "- Is it a loop variable, temp, or single letter (i,j,x,_)? -> DROP\n"
        "- Is it result = func() or data = something()? -> DROP\n"
        "- Is it a short-lived local inside a function? -> DROP\n\n"
        "If ANY keep-question is yes, output KEEP. Otherwise DROP.\n"
        "Format: N. KEEP or N. DROP. Output ONLY numbered lines.\n"
        "Most variables should be DROP (~70%)."
    ),

    # Name-aware: focus on what the name and value reveal
    "name-value": (
        "For each variable decide KEEP or DROP based on its name and value.\n\n"
        "KEEP if name is:\n"
        "- ALL_CAPS (constants, config): MAX_RETRIES, DATABASE_URL, CORS_ORIGINS\n"
        "- Known infrastructure: app, engine, logger, db, router, celery, redis, cors\n"
        "- Type/schema names: *Schema, *Type, Base, Status, *Enum\n"
        "- Descriptive compound names: session_factory, email_pattern, welcome_template\n\n"
        "DROP if name is:\n"
        "- Single letter or very short: i, j, x, n, k, tmp, val, res, buf, cnt, idx\n"
        "- Generic: result, data, response, output, items, text, lines, parts, cleaned\n"
        "- Underscore: _\n\n"
        "KEEP if value is:\n"
        "- A well-known framework call: Flask(), SQLAlchemy(), Celery(), CORS(), APIRouter()\n"
        "- create_engine(), getLogger(), sessionmaker(), compile()\n"
        "- A string literal (URL, path, SQL, template)\n"
        "- A collection of config values\n\n"
        "DROP if value is:\n"
        "- A generic function call: func(), get(), loads(), split(), strip()\n"
        "- A simple computation: len(), int(), i+1, 0, []\n\n"
        "Format: N. KEEP or N. DROP. ONLY numbered lines. ~30% keep."
    ),

    # Pattern matching: concrete examples
    "examples-heavy": (
        "Decide KEEP or DROP for each variable. Output: N. KEEP or N. DROP\n\n"
        "KEEP examples:\n"
        "- MAX_RETRIES = 3 (config constant)\n"
        "- app = Flask(__name__) (framework)\n"
        "- engine = create_engine(URL) (DB)\n"
        "- logger = logging.getLogger(__name__) (logging)\n"
        "- db = SQLAlchemy(app) (ORM)\n"
        "- router = APIRouter(prefix='/api') (routing)\n"
        "- celery = Celery('tasks', broker=URL) (task queue)\n"
        "- UserSchema = TypedDict(...) (type)\n"
        "- Status = Enum('Status', [...]) (enum)\n"
        "- DATABASE_URL = 'postgresql://...' (config)\n"
        "- CORS_ORIGINS = [...] (config list)\n"
        "- pattern = re.compile(...) (compiled regex)\n"
        "- __all__ = [...] (export list)\n\n"
        "DROP examples:\n"
        "- i = 0 (loop counter)\n"
        "- result = process_items(data) (call result)\n"
        "- data = json.loads(payload) (call result)\n"
        "- res = requests.get(url) (call result)\n"
        "- tmp = [] (temp)\n"
        "- x = get_value() (generic)\n"
        "- _ = do_something() (throwaway)\n"
        "- n = len(items) (computation)\n"
        "- val = int(s) (conversion)\n"
        "- response = make_response(data) (intermediate)\n"
        "- lines = content.split('\\n') (intermediate)\n"
        "- conn = engine.connect() (short-lived)\n\n"
        "~30% keep. ONLY numbered lines."
    ),

    # Hybrid: short checklist + examples
    "hybrid": (
        "Decide KEEP or DROP for each variable.\n"
        "Format: N. KEEP or N. DROP. ONLY numbered lines.\n\n"
        "KEEP if it matches ANY of these:\n"
        "1. ALL_CAPS name (constant/config): MAX_RETRIES, DATABASE_URL\n"
        "2. Well-known infrastructure: Flask(), SQLAlchemy(), Celery(), "
        "create_engine(), getLogger(), APIRouter(), CORS(), Redis()\n"
        "3. Type/schema/enum: TypedDict, Enum, Literal, TypeVar, declarative_base()\n"
        "4. Compiled pattern or template: re.compile(), SQL string, format string\n"
        "5. Export list: __all__ = [...]\n\n"
        "DROP everything else. Especially:\n"
        "- Single letter / short names: i, j, x, n, tmp, val, res, buf, cnt, _\n"
        "- Generic calls: result=func(), data=loads(), response=get()\n"
        "- Intermediate: split(), strip(), filter(), len(), int()\n"
        "- Short-lived locals: conn=connect(), ctx=context(), token=get()\n\n"
        "~30% keep, ~70% drop."
    ),

    # Decision tree: step by step
    "decision-tree": (
        "For each variable, follow this decision tree:\n\n"
        "1. Is the name ALL_CAPS? -> KEEP (it's a constant)\n"
        "2. Is the name single-letter or <=3 chars (i,j,x,n,tmp,val,res,buf)? -> DROP\n"
        "3. Is the value a known framework: Flask, SQLAlchemy, Celery, CORS, "
        "APIRouter, Redis, Limiter, create_engine, getLogger, sessionmaker? -> KEEP\n"
        "4. Is the value TypedDict, Enum, Literal, TypeVar, declarative_base, "
        "re.compile, __all__? -> KEEP\n"
        "5. Is it name = generic_func(args)? -> DROP\n"
        "6. Is it a string/list constant at module level? -> KEEP\n"
        "7. Everything else -> DROP\n\n"
        "Format: N. KEEP or N. DROP. ONLY numbered lines. ~30% keep."
    ),
}

# ═══════════════════════════════════════════════════════════════
# Test data
# ═══════════════════════════════════════════════════════════════

TEST_CASES = [
    # KEEP: config (8)
    ("src/config.py", "MAX_RETRIES = 3", True),
    ("src/config.py", 'DATABASE_URL = "postgresql://localhost/mydb"', True),
    ("src/config.py", "CORS_ORIGINS = ['http://localhost:3000', 'https://myapp.com']", True),
    ("src/config.py", "TIMEOUT = 30", True),
    ("src/config.py", "LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')", True),
    ("src/config.py", "MAX_UPLOAD_SIZE = 10 * 1024 * 1024", True),
    ("src/auth.py", "JWT_SECRET = os.environ['JWT_SECRET']", True),
    ("src/config.py", "API_VERSION = 'v2'", True),
    # KEEP: infra (8)
    ("src/app.py", "app = Flask(__name__)", True),
    ("src/db.py", "engine = create_engine(DATABASE_URL)", True),
    ("src/db.py", "logger = logging.getLogger(__name__)", True),
    ("src/db.py", "SessionLocal = sessionmaker(bind=engine)", True),
    ("src/middleware.py", "cors = CORS(app, origins=ALLOWED_ORIGINS)", True),
    ("src/tasks.py", "celery = Celery('tasks', broker=REDIS_URL)", True),
    ("src/app.py", "router = APIRouter(prefix='/api/v1')", True),
    ("src/app.py", "db = SQLAlchemy(app)", True),
    # KEEP: types + patterns (5)
    ("src/models.py", 'UserSchema = TypedDict("UserSchema", {"name": str, "email": str})', True),
    ("src/models.py", "Base = declarative_base()", True),
    ("src/models.py", "Status = Enum('Status', ['ACTIVE', 'INACTIVE', 'PENDING'])", True),
    ("src/utils.py", "pattern = re.compile(r'^[a-z]+$')", True),
    ("src/app.py", "__all__ = ['app', 'create_app', 'db']", True),
    # DROP: loop/temp (7)
    ("src/utils.py", "i = 0", False),
    ("src/utils.py", "x = get_value()", False),
    ("src/utils.py", "tmp = []", False),
    ("src/tasks.py", "_ = do_something()", False),
    ("src/utils.py", "n = len(items)", False),
    ("src/utils.py", "val = int(s)", False),
    ("src/utils.py", "j = i + 1", False),
    # DROP: generic call results (7)
    ("src/app.py", "result = process_items(data)", False),
    ("src/db.py", "data = json.loads(payload)", False),
    ("src/app.py", "res = requests.get(url)", False),
    ("src/app.py", "response = make_response(data)", False),
    ("src/utils.py", "items = list(filter(None, raw_items))", False),
    ("src/utils.py", "buf = io.BytesIO()", False),
    ("src/utils.py", "output = func(input_data)", False),
    # DROP: intermediate (3)
    ("src/utils.py", "for item in items: total += item.price", False),
    ("src/utils.py", "lines = content.split('\\n')", False),
    ("src/utils.py", "cleaned = text.strip().lower()", False),
    # Borderline (3)
    ("src/db.py", "conn = engine.connect()", False),
    ("src/utils.py", "lock = threading.Lock()", True),
    ("src/config.py", "DEBUG = True", True),
]


def _strip_think(output: str) -> str:
    return re.sub(r"<think>.*?</think>", "", output, flags=re.DOTALL).strip()


def parse_binary(output: str, batch_size: int) -> list[bool | None]:
    output = _strip_think(output)
    scores: list[bool | None] = [None] * batch_size
    pattern = re.compile(r"(\d+)\s*[.):]\s*(keep|drop)", re.IGNORECASE)
    for match in pattern.finditer(output):
        idx = int(match.group(1))
        if 1 <= idx <= batch_size:
            scores[idx - 1] = match.group(2).upper() == "KEEP"
    return scores


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


def run_test(cases: list, model: str, system_prompt: str, batch_size: int = 8) -> dict:
    total_correct = 0
    total_total = 0
    total_fp = 0
    total_fn = 0
    total_missing = 0
    total_time = 0.0
    errors = []

    for start in range(0, len(cases), batch_size):
        batch = cases[start: start + batch_size]
        user_prompt = build_user_prompt(batch)
        num_predict = len(batch) * 10 + 30

        try:
            output, elapsed = call_ollama(model, system_prompt, user_prompt, num_predict)
        except Exception as e:
            errors.append(f"  BATCH ERROR: {e}")
            total_missing += len(batch)
            total_total += len(batch)
            continue

        total_time += elapsed
        scores = parse_binary(output, len(batch))

        for j, (fp, code, expected) in enumerate(batch):
            total_total += 1
            decision = scores[j]
            effective = decision if decision is not None else True  # default keep

            if decision is None:
                total_missing += 1

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
        "correct": total_correct,
        "total": total_total,
        "accuracy": total_correct / total_total if total_total else 0,
        "false_pos": total_fp,
        "false_neg": total_fn,
        "missing": total_missing,
        "time": total_time,
        "errors": errors,
    }


def main():
    n = len(TEST_CASES)
    keeps = sum(1 for _, _, e in TEST_CASES if e)
    drops = n - keeps

    print("=" * 70)
    print("Binary prompt fine-tuning: find the best guidance approach")
    print("=" * 70)
    print(f"\nTest cases: {n} ({keeps} keep, {drops} drop)")
    print(f"Batch size: 8, temp: 0.1, default-to-KEEP for missing")
    print(f"\nPrompts to test: {len(PROMPTS)}")
    for name, prompt in PROMPTS.items():
        print(f"  {name:20s}: {len(prompt):4d} chars")

    for model in MODELS:
        print(f"\n{'═' * 70}")
        print(f"Model: {model}")
        print(f"{'═' * 70}")

        results = []
        for prompt_name, system_prompt in PROMPTS.items():
            # Run 3 times for stability
            runs = []
            for _ in range(3):
                r = run_test(TEST_CASES, model, system_prompt)
                runs.append(r)

            avg_acc = sum(r["accuracy"] for r in runs) / len(runs)
            avg_fp = sum(r["false_pos"] for r in runs) / len(runs)
            avg_fn = sum(r["false_neg"] for r in runs) / len(runs)
            avg_miss = sum(r["missing"] for r in runs) / len(runs)
            avg_time = sum(r["time"] for r in runs) / len(runs)
            best = max(runs, key=lambda r: r["accuracy"])

            results.append({
                "name": prompt_name,
                "avg_acc": avg_acc,
                "best_acc": best["accuracy"],
                "avg_fp": avg_fp,
                "avg_fn": avg_fn,
                "avg_miss": avg_miss,
                "avg_time": avg_time,
                "best_errors": best["errors"],
                "chars": len(system_prompt),
            })

            pct = f"{avg_acc*100:.1f}%"
            best_pct = f"{best['accuracy']*100:.0f}%"
            print(
                f"  {prompt_name:20s}: {pct:6s} avg ({best_pct} best) | "
                f"FP={avg_fp:.1f} FN={avg_fn:.1f} miss={avg_miss:.1f} | "
                f"{avg_time:.1f}s"
            )

        # Rank by accuracy
        ranked = sorted(results, key=lambda r: r["avg_acc"], reverse=True)
        print(f"\n  Ranking:")
        for i, r in enumerate(ranked):
            marker = " <-- BEST" if i == 0 else ""
            print(
                f"    {i+1}. {r['name']:20s} {r['avg_acc']*100:5.1f}% "
                f"(FP={r['avg_fp']:.1f} FN={r['avg_fn']:.1f} miss={r['avg_miss']:.1f})"
                f"{marker}"
            )

        # Show errors for best prompt
        best_prompt = ranked[0]
        if best_prompt["best_errors"]:
            print(f"\n  Errors from best prompt ({best_prompt['name']}, best run):")
            for e in best_prompt["best_errors"]:
                print(f"  {e}")

    # Cross-model summary
    print(f"\n{'═' * 70}")
    print("CROSS-MODEL SUMMARY")
    print(f"{'═' * 70}")

    # Collect all results keyed by (model, prompt)
    all_data: dict[tuple[str, str], dict] = {}
    for model in MODELS:
        for prompt_name, system_prompt in PROMPTS.items():
            runs = []
            for _ in range(3):
                r = run_test(TEST_CASES, model, system_prompt)
                runs.append(r)
            avg_acc = sum(r["accuracy"] for r in runs) / len(runs)
            avg_fp = sum(r["false_pos"] for r in runs) / len(runs)
            avg_fn = sum(r["false_neg"] for r in runs) / len(runs)
            avg_time = sum(r["time"] for r in runs) / len(runs)
            all_data[(model, prompt_name)] = {
                "acc": avg_acc, "fp": avg_fp, "fn": avg_fn, "time": avg_time,
            }

    # Print cross-model table
    col_w = 14
    print(f"\n  {'Prompt':20s}", end="")
    for model in MODELS:
        print(f" | {model:^{col_w}s}", end="")
    print()
    print(f"  {'-'*20}", end="")
    for _ in MODELS:
        print(f"-+-{'-'*col_w}", end="")
    print()

    for prompt_name in PROMPTS:
        print(f"  {prompt_name:20s}", end="")
        for model in MODELS:
            d = all_data.get((model, prompt_name))
            if d:
                print(f" | {d['acc']*100:5.1f}% {d['time']:4.1f}s", end="")
                pad = col_w - 12
                print(" " * pad, end="")
            else:
                print(f" | {'N/A':^{col_w}s}", end="")
        print()

    # Best per model
    print(f"\n  Best prompt per model:")
    for model in MODELS:
        best_name = max(
            PROMPTS.keys(),
            key=lambda p: all_data.get((model, p), {}).get("acc", 0),
        )
        d = all_data[(model, best_name)]
        print(
            f"    {model:15s}: {best_name:20s} ({d['acc']*100:.1f}%, "
            f"FP={d['fp']:.1f}, FN={d['fn']:.1f})"
        )

    # Best overall
    best_key = max(all_data.keys(), key=lambda k: all_data[k]["acc"])
    d = all_data[best_key]
    print(
        f"\n  Overall best: {best_key[0]} + {best_key[1]} "
        f"({d['acc']*100:.1f}%)"
    )


if __name__ == "__main__":
    main()
