#!/usr/bin/env python3
"""Test binary keep/drop with ONE variable per message.

Hypothesis: small models struggle with batched output formatting but may
handle a single keep/drop decision trivially.

Trade-off: more API calls, but each call is tiny and fast.
"""

import time

import requests

SYSTEM_PROMPT = (
    "You are scoring variables for a code search index.\n"
    "Reply with exactly one word: KEEP or DROP.\n"
    "KEEP: constants, config, DB connections, loggers, types, enums, "
    "framework instances, middleware, prompts, SQL.\n"
    "DROP: loop vars, temp, generic results, single-letter names, "
    "result=func(), intermediate computations.\n"
    "Most variables should be DROP."
)

# Test cases: (file_path, code, expected_keep)
TEST_CASES = [
    ("src/config.py", "MAX_RETRIES = 3", True),
    ("src/app.py", "app = Flask(__name__)", True),
    ("src/app.py", "result = process_items(data)", False),
    ("src/db.py", "engine = create_engine(DATABASE_URL)", True),
    ("src/db.py", "logger = logging.getLogger(__name__)", True),
    ("src/db.py", "data = json.loads(payload)", False),
    ("src/models.py", 'UserSchema = TypedDict("UserSchema", {"name": str, "email": str})', True),
    ("src/utils.py", "i = 0", False),
    # Extra edge cases
    ("src/utils.py", "x = get_value()", False),
    ("src/config.py", 'DATABASE_URL = "postgresql://localhost/mydb"', True),
    ("src/app.py", "tmp = []", False),
    ("src/middleware.py", "cors = CORS(app, origins=ALLOWED_ORIGINS)", True),
    ("src/tasks.py", "_ = do_something()", False),
    ("src/models.py", "class UserModel(Base): pass", True),
    ("src/utils.py", "for item in items: total += item.price", False),
    ("src/config.py", "CORS_ORIGINS = ['http://localhost:3000', 'https://myapp.com']", True),
]

MODELS = [
    "qwen3.5:0.8b",
    "qwen3:1.7b",
    "granite3.3:2b",
    "llama3.2:1b",
    "qwen3.5:2b",
    "qwen2.5-coder:3b",
    "llama3.2:3b",
    "phi4-mini",
    "qwen3:4b",
    "qwen3.5:4b",
    "gemma3:4b",
]

API_BASE = "http://localhost:11434"


def call_ollama(model: str, system_prompt: str, user_prompt: str) -> tuple[str, float]:
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
            "options": {"temperature": 0.1, "num_predict": 10, "num_ctx": 1024},
        },
        timeout=30,
    )
    elapsed = time.time() - start
    data = resp.json()
    content = data.get("message", {}).get("content", "")
    return content.strip(), elapsed


def parse_single_decision(output: str) -> bool | None:
    """Parse a single KEEP/DROP response."""
    lower = output.lower().strip()
    # Direct match
    if lower.startswith("keep"):
        return True
    if lower.startswith("drop"):
        return False
    # Check if it contains keep/drop anywhere
    if "keep" in lower and "drop" not in lower:
        return True
    if "drop" in lower and "keep" not in lower:
        return False
    return None


def test_model(model: str) -> dict:
    """Test a model on all test cases, one at a time."""
    correct = 0
    total = len(TEST_CASES)
    errors = []
    parse_fails = 0
    total_time = 0.0

    for file_path, code, expected_keep in TEST_CASES:
        user_msg = f"[{file_path}] {code}"
        try:
            output, elapsed = call_ollama(model, SYSTEM_PROMPT, user_msg)
            total_time += elapsed
        except Exception as e:
            errors.append(f"  {code[:30]:30s} ERROR: {e}")
            continue

        decision = parse_single_decision(output)
        if decision is None:
            parse_fails += 1
            errors.append(f"  {code[:30]:30s} PARSE FAIL: {output[:50]}")
        elif decision == expected_keep:
            correct += 1
        else:
            exp = "KEEP" if expected_keep else "DROP"
            act = "KEEP" if decision else "DROP"
            errors.append(f"  {code[:30]:30s} expected {exp}, got {act} (raw: {output[:30]})")

    return {
        "model": model,
        "correct": correct,
        "total": total,
        "parse_fails": parse_fails,
        "errors": errors,
        "total_time": total_time,
        "avg_time": total_time / total if total > 0 else 0,
    }


def main():
    print("=" * 70)
    print("Binary Keep/Drop: ONE variable per message (no format forcing)")
    print("=" * 70)
    print(f"\nSystem prompt ({len(SYSTEM_PROMPT)} chars):")
    print(f"  {SYSTEM_PROMPT[:100]}...")
    print(f"\nTest cases: {len(TEST_CASES)}")
    print(f"num_predict: 10 tokens (just need KEEP or DROP)")

    results = []

    for model in MODELS:
        print(f"\n{'=' * 70}")
        print(f"Model: {model}")
        print(f"{'=' * 70}")

        result = test_model(model)
        results.append(result)

        acc = f"{result['correct']}/{result['total']}"
        pf = f" ({result['parse_fails']} parse fails)" if result["parse_fails"] > 0 else ""
        status = "PERFECT" if result["correct"] == result["total"] else acc
        print(f"  {status}{pf} | total: {result['total_time']:.1f}s | avg: {result['avg_time']*1000:.0f}ms/var")

        if result["errors"]:
            for err in result["errors"]:
                print(f"  {err}")

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Model':25s} | {'Score':8s} | {'Fails':6s} | {'Total':8s} | {'Avg/var':10s}")
    print(f"{'-' * 25}-+-{'-' * 8}-+-{'-' * 6}-+-{'-' * 8}-+-{'-' * 10}")

    for r in results:
        score = f"{r['correct']}/{r['total']}"
        if r["correct"] == r["total"]:
            score += " *"
        pf = str(r["parse_fails"]) if r["parse_fails"] > 0 else "-"
        total = f"{r['total_time']:.1f}s"
        avg = f"{r['avg_time']*1000:.0f}ms"
        print(f"{r['model']:25s} | {score:8s} | {pf:6s} | {total:8s} | {avg:10s}")

    print(f"\n* = perfect score")


if __name__ == "__main__":
    main()
