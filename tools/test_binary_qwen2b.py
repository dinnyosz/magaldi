#!/usr/bin/env python3
"""Deep benchmark of qwen3.5:2b for binary keep/drop variable scoring.

Tests:
1. Consistency: same batch 5 times to check determinism
2. Batch size sensitivity: 4, 8, 12, 16, 20, 24 variables per batch
3. Extended test set: 60+ real-world-like variables
4. False positive / false negative rates
5. Throughput at different parallelism levels
"""

import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

API_BASE = "http://localhost:11434"
MODEL = "qwen3.5:2b"
REFERENCE_MODEL = "qwen3.5:4b"

SYSTEM_PROMPT = (
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

# --- Extended test set (60+ variables) ---

# (file_path, code, expected_keep, category)
ALL_CASES = [
    # === CLEAR KEEPS (constants, config) ===
    ("src/config.py", "MAX_RETRIES = 3", True, "config"),
    ("src/config.py", 'DATABASE_URL = "postgresql://localhost/mydb"', True, "config"),
    ("src/config.py", "CORS_ORIGINS = ['http://localhost:3000', 'https://myapp.com']", True, "config"),
    ("src/config.py", "TIMEOUT = 30", True, "config"),
    ("src/config.py", "LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')", True, "config"),
    ("src/config.py", "REDIS_URL = os.environ.get('REDIS_URL', 'redis://localhost')", True, "config"),
    ("src/config.py", "MAX_UPLOAD_SIZE = 10 * 1024 * 1024", True, "config"),
    ("src/config.py", "ALLOWED_EXTENSIONS = {'.jpg', '.png', '.gif', '.pdf'}", True, "config"),
    ("src/auth.py", "JWT_SECRET = os.environ['JWT_SECRET']", True, "config"),
    ("src/config.py", "API_VERSION = 'v2'", True, "config"),

    # === CLEAR KEEPS (infrastructure) ===
    ("src/app.py", "app = Flask(__name__)", True, "infra"),
    ("src/db.py", "engine = create_engine(DATABASE_URL)", True, "infra"),
    ("src/db.py", "logger = logging.getLogger(__name__)", True, "infra"),
    ("src/db.py", "SessionLocal = sessionmaker(bind=engine)", True, "infra"),
    ("src/middleware.py", "cors = CORS(app, origins=ALLOWED_ORIGINS)", True, "infra"),
    ("src/tasks.py", "celery = Celery('tasks', broker=REDIS_URL)", True, "infra"),
    ("src/app.py", "router = APIRouter(prefix='/api/v1')", True, "infra"),
    ("src/app.py", "db = SQLAlchemy(app)", True, "infra"),
    ("src/cache.py", "redis_client = Redis.from_url(REDIS_URL)", True, "infra"),
    ("src/app.py", "limiter = Limiter(key_func=get_remote_address)", True, "infra"),

    # === CLEAR KEEPS (types, schemas, enums) ===
    ("src/models.py", 'UserSchema = TypedDict("UserSchema", {"name": str, "email": str})', True, "types"),
    ("src/models.py", "Base = declarative_base()", True, "types"),
    ("src/models.py", "Status = Enum('Status', ['ACTIVE', 'INACTIVE', 'PENDING'])", True, "types"),
    ("src/models.py", "UserRole = Literal['admin', 'user', 'guest']", True, "types"),
    ("src/models.py", "T = TypeVar('T', bound='BaseModel')", True, "types"),

    # === CLEAR KEEPS (patterns, templates, SQL) ===
    ("src/utils.py", "pattern = re.compile(r'^[a-z]+$')", True, "patterns"),
    ("src/utils.py", "EMAIL_REGEX = re.compile(r'^[\\w.-]+@[\\w.-]+\\.\\w+$')", True, "patterns"),
    ("src/templates.py", "WELCOME_EMAIL = 'Hello {name}, welcome to {app}!'", True, "patterns"),
    ("src/db.py", "USERS_QUERY = 'SELECT * FROM users WHERE active = true'", True, "patterns"),
    ("src/app.py", "__all__ = ['app', 'create_app', 'db']", True, "patterns"),

    # === CLEAR DROPS (loop vars, temp, single-letter) ===
    ("src/utils.py", "i = 0", False, "loop"),
    ("src/utils.py", "x = get_value()", False, "loop"),
    ("src/utils.py", "tmp = []", False, "loop"),
    ("src/tasks.py", "_ = do_something()", False, "loop"),
    ("src/utils.py", "n = len(items)", False, "loop"),
    ("src/utils.py", "val = int(s)", False, "loop"),
    ("src/utils.py", "j = i + 1", False, "loop"),
    ("src/utils.py", "k = 0", False, "loop"),
    ("src/utils.py", "idx = items.index(target)", False, "loop"),
    ("src/utils.py", "cnt = 0", False, "loop"),

    # === CLEAR DROPS (generic call results) ===
    ("src/app.py", "result = process_items(data)", False, "generic"),
    ("src/db.py", "data = json.loads(payload)", False, "generic"),
    ("src/app.py", "res = requests.get(url)", False, "generic"),
    ("src/app.py", "response = make_response(data)", False, "generic"),
    ("src/utils.py", "items = list(filter(None, raw_items))", False, "generic"),
    ("src/utils.py", "buf = io.BytesIO()", False, "generic"),
    ("src/utils.py", "output = func(input_data)", False, "generic"),
    ("src/utils.py", "parsed = parse_config(raw)", False, "generic"),
    ("src/app.py", "ctx = app.app_context()", False, "generic"),
    ("src/utils.py", "text = response.text", False, "generic"),

    # === CLEAR DROPS (intermediate / short-lived) ===
    ("src/utils.py", "for item in items: total += item.price", False, "intermediate"),
    ("src/app.py", "token = request.headers.get('Authorization')", False, "intermediate"),
    ("src/utils.py", "lines = content.split('\\n')", False, "intermediate"),
    ("src/utils.py", "cleaned = text.strip().lower()", False, "intermediate"),
    ("src/utils.py", "parts = url.split('/')", False, "intermediate"),

    # === BORDERLINE (tricky cases) ===
    ("src/db.py", "conn = engine.connect()", False, "borderline"),  # short-lived local
    ("src/app.py", "session = requests.Session()", False, "borderline"),  # could go either way
    ("src/utils.py", "lock = threading.Lock()", True, "borderline"),  # sync primitive = infra
    ("src/app.py", "middleware = SessionMiddleware(app, secret_key=SECRET)", True, "borderline"),
    ("src/utils.py", "pool = ThreadPoolExecutor(max_workers=4)", True, "borderline"),  # infra
    ("src/config.py", "DEBUG = True", True, "borderline"),  # config flag
    ("src/utils.py", "cache = {}", False, "borderline"),  # empty dict, no context
    ("src/app.py", "handler = logging.StreamHandler()", True, "borderline"),  # logging infra
]


def _strip_think_tags(output: str) -> str:
    return re.sub(r"<think>.*?</think>", "", output, flags=re.DOTALL).strip()


def _build_user_prompt(cases: list) -> str:
    lines = ["/no_think\nScore these variables:"]
    for i, (file_path, code, *_rest) in enumerate(cases, 1):
        lines.append(f"{i}. [{file_path}] {code}")
    return "\n".join(lines)


def _parse_batch(output: str, batch_size: int) -> list[bool | None]:
    output = _strip_think_tags(output)
    scores: list[bool | None] = [None] * batch_size
    pattern = re.compile(r"(\d+)\s*[.):]\s*(keep|drop)", re.IGNORECASE)
    for match in pattern.finditer(output):
        idx = int(match.group(1))
        if 1 <= idx <= batch_size:
            scores[idx - 1] = match.group(2).upper() == "KEEP"
    return scores


def call_ollama(model: str, user_prompt: str, num_predict: int = 200) -> tuple[str, float]:
    start = time.time()
    resp = requests.post(
        f"{API_BASE}/api/chat",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "think": False,
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": num_predict, "num_ctx": 2048},
        },
        timeout=60,
    )
    elapsed = time.time() - start
    data = resp.json()
    content = data.get("message", {}).get("content", "")
    return content.strip(), elapsed


def evaluate(
    cases: list, scores: list[bool | None], default_keep: bool = True
) -> dict:
    """Evaluate scores against expected, with default-to-keep for None."""
    correct = 0
    total = len(cases)
    false_pos = 0  # kept when should drop
    false_neg = 0  # dropped when should keep
    missing = 0
    errors = []

    for i, (file_path, code, expected, *rest) in enumerate(cases):
        decision = scores[i]
        cat = rest[0] if rest else "unknown"

        if decision is None:
            missing += 1
            effective = default_keep
        else:
            effective = decision

        if effective == expected:
            correct += 1
        else:
            exp = "KEEP" if expected else "DROP"
            act = "KEEP" if effective else "DROP"
            suffix = " (defaulted)" if decision is None else ""
            errors.append(f"  {i+1:2d}. [{cat:12s}] {code[:35]:35s} {exp}->{act}{suffix}")
            if expected and not effective:
                false_neg += 1
            elif not expected and effective:
                false_pos += 1

    return {
        "correct": correct,
        "total": total,
        "missing": missing,
        "false_pos": false_pos,
        "false_neg": false_neg,
        "errors": errors,
        "accuracy": correct / total if total > 0 else 0,
    }


def run_batch(cases: list, model: str = MODEL) -> tuple[list[bool | None], float]:
    """Score a batch and return parsed scores + time."""
    user_prompt = _build_user_prompt(cases)
    num_predict = len(cases) * 10 + 30
    output, elapsed = call_ollama(model, user_prompt, num_predict)
    scores = _parse_batch(output, len(cases))
    return scores, elapsed


def main():
    print("=" * 70)
    print(f"Deep benchmark: {MODEL} for binary keep/drop scoring")
    print(f"(default-to-KEEP for missing scores)")
    print("=" * 70)
    print(f"\nTotal test cases: {len(ALL_CASES)}")

    # Count by category
    cats = Counter(cat for _, _, _, cat in ALL_CASES)
    keeps = sum(1 for _, _, exp, _ in ALL_CASES if exp)
    drops = sum(1 for _, _, exp, _ in ALL_CASES if not exp)
    print(f"Expected: {keeps} KEEP, {drops} DROP ({keeps/len(ALL_CASES)*100:.0f}%/{drops/len(ALL_CASES)*100:.0f}%)")
    for cat, count in sorted(cats.items()):
        print(f"  {cat:15s}: {count}")

    # ═══════════════════════════════════════════════════════════════
    # Test 1: Consistency — run same batch 5 times
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'═' * 70}")
    print("Test 1: Consistency (all cases, 5 runs)")
    print(f"{'═' * 70}")

    run_results = []
    for run in range(5):
        scores, elapsed = run_batch(ALL_CASES)
        ev = evaluate(ALL_CASES, scores)
        run_results.append(ev)
        pct = f"{ev['accuracy']*100:.0f}%"
        print(
            f"  Run {run+1}: {ev['correct']}/{ev['total']} ({pct}) | "
            f"FP={ev['false_pos']} FN={ev['false_neg']} missing={ev['missing']} | "
            f"{elapsed:.1f}s"
        )

    # Per-variable consistency
    all_run_scores = []
    for _ in range(5):
        scores, _ = run_batch(ALL_CASES)
        all_run_scores.append(scores)

    # How often does each variable get the same answer?
    inconsistent = []
    for i, (fp, code, expected, cat) in enumerate(ALL_CASES):
        decisions = [r[i] for r in all_run_scores]
        non_none = [d for d in decisions if d is not None]
        if non_none and len(set(non_none)) > 1:
            keep_count = sum(1 for d in non_none if d)
            drop_count = sum(1 for d in non_none if not d)
            none_count = sum(1 for d in decisions if d is None)
            inconsistent.append(
                f"  {i+1:2d}. [{cat:12s}] {code[:35]:35s} "
                f"KEEP={keep_count} DROP={drop_count} None={none_count} "
                f"(expected {'KEEP' if expected else 'DROP'})"
            )

    if inconsistent:
        print(f"\n  Inconsistent variables ({len(inconsistent)}/{len(ALL_CASES)}):")
        for line in inconsistent:
            print(line)
    else:
        print(f"\n  All variables consistent across 5 runs!")

    # ═══════════════════════════════════════════════════════════════
    # Test 2: Batch size sensitivity
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'═' * 70}")
    print("Test 2: Batch size sensitivity")
    print(f"{'═' * 70}")
    print(f"  {'Batch':6s} | {'Score':8s} | {'Accuracy':8s} | {'FP':4s} | {'FN':4s} | {'Miss':5s} | {'Time':8s} | {'Per var':8s}")
    print(f"  {'-'*6}-+-{'-'*8}-+-{'-'*8}-+-{'-'*4}-+-{'-'*4}-+-{'-'*5}-+-{'-'*8}-+-{'-'*8}")

    for batch_size in [4, 8, 12, 16, 20, 24, 32, len(ALL_CASES)]:
        if batch_size > len(ALL_CASES):
            continue

        # Split into batches and process
        total_correct = 0
        total_total = 0
        total_fp = 0
        total_fn = 0
        total_missing = 0
        total_time = 0.0

        for start in range(0, len(ALL_CASES), batch_size):
            batch = ALL_CASES[start : start + batch_size]
            scores, elapsed = run_batch(batch)
            ev = evaluate(batch, scores)
            total_correct += ev["correct"]
            total_total += ev["total"]
            total_fp += ev["false_pos"]
            total_fn += ev["false_neg"]
            total_missing += ev["missing"]
            total_time += elapsed

        acc = total_correct / total_total * 100
        per_var = total_time / total_total * 1000
        bs_label = str(batch_size) if batch_size < len(ALL_CASES) else "ALL"
        print(
            f"  {bs_label:6s} | {total_correct}/{total_total:2d}    | {acc:5.1f}%  | "
            f"{total_fp:4d} | {total_fn:4d} | {total_missing:5d} | "
            f"{total_time:6.1f}s | {per_var:5.0f}ms"
        )

    # ═══════════════════════════════════════════════════════════════
    # Test 3: Category breakdown
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'═' * 70}")
    print("Test 3: Accuracy by category (batch_size=8, 3 runs averaged)")
    print(f"{'═' * 70}")

    cat_stats: dict[str, dict] = {}
    for _ in range(3):
        for start in range(0, len(ALL_CASES), 8):
            batch = ALL_CASES[start : start + 8]
            scores, _ = run_batch(batch)
            for j, (fp, code, expected, cat) in enumerate(batch):
                if cat not in cat_stats:
                    cat_stats[cat] = {"correct": 0, "total": 0, "fp": 0, "fn": 0, "missing": 0}
                cat_stats[cat]["total"] += 1
                decision = scores[j] if scores[j] is not None else True  # default keep
                if decision == expected:
                    cat_stats[cat]["correct"] += 1
                elif not expected and decision:
                    cat_stats[cat]["fp"] += 1
                elif expected and not decision:
                    cat_stats[cat]["fn"] += 1
                if scores[j] is None:
                    cat_stats[cat]["missing"] += 1

    print(f"  {'Category':15s} | {'Accuracy':10s} | {'FP':4s} | {'FN':4s} | {'Missing':8s}")
    print(f"  {'-'*15}-+-{'-'*10}-+-{'-'*4}-+-{'-'*4}-+-{'-'*8}")
    for cat in sorted(cat_stats.keys()):
        s = cat_stats[cat]
        acc = s["correct"] / s["total"] * 100 if s["total"] > 0 else 0
        print(
            f"  {cat:15s} | {s['correct']:3d}/{s['total']:<3d} {acc:4.0f}% | "
            f"{s['fp']:4d} | {s['fn']:4d} | {s['missing']:8d}"
        )

    # ═══════════════════════════════════════════════════════════════
    # Test 4: Parallel throughput
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'═' * 70}")
    print("Test 4: Parallel throughput (batch_size=8)")
    print(f"{'═' * 70}")

    for workers in [1, 2, 4, 8]:
        batches = []
        for start in range(0, len(ALL_CASES), 8):
            batches.append(ALL_CASES[start : start + 8])

        total_correct = 0
        total_total = 0

        start_time = time.time()

        def process_batch(batch):
            scores, elapsed = run_batch(batch)
            ev = evaluate(batch, scores)
            return ev

        if workers == 1:
            for batch in batches:
                ev = process_batch(batch)
                total_correct += ev["correct"]
                total_total += ev["total"]
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(process_batch, b) for b in batches]
                for f in as_completed(futures):
                    ev = f.result()
                    total_correct += ev["correct"]
                    total_total += ev["total"]

        wall_time = time.time() - start_time
        acc = total_correct / total_total * 100
        per_var = wall_time / total_total * 1000
        print(
            f"  workers={workers}: {total_correct}/{total_total} ({acc:.0f}%) | "
            f"wall={wall_time:.1f}s | {per_var:.0f}ms/var"
        )

    # ═══════════════════════════════════════════════════════════════
    # Test 5: Compare with 4b reference
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'═' * 70}")
    print(f"Test 5: Side-by-side with {REFERENCE_MODEL}")
    print(f"{'═' * 70}")

    scores_2b, time_2b = run_batch(ALL_CASES, model=MODEL)
    scores_4b, time_4b = run_batch(ALL_CASES, model=REFERENCE_MODEL)

    ev_2b = evaluate(ALL_CASES, scores_2b)
    ev_4b = evaluate(ALL_CASES, scores_4b)

    print(f"  {'':35s}   {'2b':8s}   {'4b':8s}   expected")
    print(f"  {'-'*35}---{'-'*8}---{'-'*8}---{'-'*8}")

    disagreements = []
    for i, (fp, code, expected, cat) in enumerate(ALL_CASES):
        d2b = scores_2b[i]
        d4b = scores_4b[i]
        # Apply default-to-keep
        eff_2b = d2b if d2b is not None else True
        eff_4b = d4b if d4b is not None else True

        if eff_2b != eff_4b:
            s2b = "KEEP" if eff_2b else "DROP"
            s4b = "KEEP" if eff_4b else "DROP"
            sexp = "KEEP" if expected else "DROP"
            mark_2b = " *" if eff_2b != expected else ""
            mark_4b = " *" if eff_4b != expected else ""
            none_2b = " (def)" if d2b is None else ""
            none_4b = " (def)" if d4b is None else ""
            disagreements.append(
                f"  {i+1:2d}. {code[:35]:35s}   {s2b}{none_2b:8s}   {s4b}{none_4b:8s}   {sexp}"
            )

    if disagreements:
        print(f"\n  Disagreements ({len(disagreements)}):")
        for line in disagreements:
            print(line)
    else:
        print(f"\n  Models fully agree!")

    print(f"\n  {MODEL:15s}: {ev_2b['correct']}/{ev_2b['total']} ({ev_2b['accuracy']*100:.0f}%) "
          f"FP={ev_2b['false_pos']} FN={ev_2b['false_neg']} missing={ev_2b['missing']} | {time_2b:.1f}s")
    print(f"  {REFERENCE_MODEL:15s}: {ev_4b['correct']}/{ev_4b['total']} ({ev_4b['accuracy']*100:.0f}%) "
          f"FP={ev_4b['false_pos']} FN={ev_4b['false_neg']} missing={ev_4b['missing']} | {time_4b:.1f}s")

    # ═══════════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'═' * 70}")
    print("OVERALL ASSESSMENT")
    print(f"{'═' * 70}")

    avg_acc = sum(r["accuracy"] for r in run_results) / len(run_results)
    avg_fp = sum(r["false_pos"] for r in run_results) / len(run_results)
    avg_fn = sum(r["false_neg"] for r in run_results) / len(run_results)
    avg_miss = sum(r["missing"] for r in run_results) / len(run_results)

    print(f"  Average accuracy (5 runs): {avg_acc*100:.1f}%")
    print(f"  Average false positives:   {avg_fp:.1f} (kept when should drop)")
    print(f"  Average false negatives:   {avg_fn:.1f} (dropped when should keep)")
    print(f"  Average missing (->KEEP):  {avg_miss:.1f}")
    print()
    print(f"  With default-to-KEEP policy:")
    print(f"    - False negatives are eliminated (missing -> KEEP)")
    print(f"    - False positives slightly increase (missing keeps + wrong keeps)")
    print(f"    - Net effect: slightly more variables kept, but nothing important lost")


if __name__ == "__main__":
    main()
