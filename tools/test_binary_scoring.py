#!/usr/bin/env python3
"""Test binary keep/drop variable scoring with small models.

Instead of asking LLMs to score 7 dimensions (1-10), just ask:
"Should this variable be kept in the search index or dropped?"

No format forcing — models output freely, we parse whatever they give.
"""

import json
import re
import time

import requests

# --- Prompts to test ---

PROMPTS = {
    "numbered": (
        "For each variable, output KEEP or DROP on a line:\n"
        "N. KEEP or N. DROP\n"
        "KEEP: constants, config, DB, loggers, types, enums, framework instances\n"
        "DROP: loop vars, temp, generic results, single-letter, x=func()\n"
        "Most variables should be DROP (~70%). Output ONLY the numbered lines."
    ),
    "with-examples": (
        "For each variable, output KEEP or DROP:\n"
        "Format: N. KEEP or N. DROP\n"
        "KEEP: constants, config values, DB connections, loggers, type defs, "
        "app instances, middleware, compiled regex, prompts, SQL\n"
        "DROP: loop counters, temp vars, result=func(), i/j/x/tmp/res/val, "
        "intermediate computations, short-lived locals\n"
        "~30% keep, ~70% drop. Output ONLY the numbered decisions."
    ),
    "question-based": (
        'For each variable ask: "Would a developer searching the codebase '
        'benefit from finding this?"\n'
        "Output: N. KEEP or N. DROP\n"
        "DROP: loop vars, temp, result=func(), single-letter names.\n"
        "KEEP: constants, config, infrastructure, types, schemas.\n"
        "Output ONLY the numbered decisions."
    ),
}

# --- Test variables (mix of keep and drop) ---

TEST_VARIABLES = """Score these variables:
1. [src/config.py] MAX_RETRIES = 3
2. [src/app.py] app = Flask(__name__)
3. [src/app.py] result = process_items(data)
4. [src/db.py] engine = create_engine(DATABASE_URL)
5. [src/db.py] logger = logging.getLogger(__name__)
6. [src/db.py] data = json.loads(payload)
7. [src/models.py] UserSchema = TypedDict("UserSchema", {"name": str, "email": str})
8. [src/utils.py] i = 0"""

# Expected: 1=keep, 2=keep, 3=drop, 4=keep, 5=keep, 6=drop, 7=keep, 8=drop
EXPECTED = {
    "1": True,   # MAX_RETRIES - config constant
    "2": True,   # app - framework instance
    "3": False,  # result - generic call result
    "4": True,   # engine - DB infrastructure
    "5": True,   # logger - infrastructure
    "6": False,  # data - generic local
    "7": True,   # UserSchema - type definition
    "8": False,  # i - loop counter
}
LABELS = {
    "1": "MAX_RETRIES",
    "2": "app",
    "3": "result",
    "4": "engine",
    "5": "logger",
    "6": "data",
    "7": "UserSchema",
    "8": "i",
}

MODELS = [
    # Sub-1B
    "qwen3.5:0.8b",
    # 1-2B
    "qwen3:1.7b",
    "granite3.3:2b",
    "llama3.2:1b",
    "qwen3.5:2b",
    # 2-4B
    "qwen2.5-coder:3b",
    "llama3.2:3b",
    "phi4-mini",
    "qwen3:4b",
    "qwen3.5:4b",
    "gemma3:4b",
]

API_BASE = "http://localhost:11434"


def call_ollama(model: str, system_prompt: str, user_prompt: str) -> tuple[str, float]:
    """Call Ollama API without any format forcing."""
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
            "options": {"temperature": 0.1, "num_predict": 300, "num_ctx": 2048},
        },
        timeout=120,
    )
    elapsed = time.time() - start
    data = resp.json()
    content = data.get("message", {}).get("content", "")
    return content, elapsed


def parse_binary_scores(output: str, batch_size: int = 8) -> dict[str, bool] | None:
    """Parse keep/drop decisions from free-form model output.

    Handles multiple formats:
    - "1. KEEP"  / "1. DROP"
    - "1. keep"  / "1. drop"
    - "1: KEEP"  / "1: DROP"
    - JSON: {"1": true, "2": false}
    - "1. KEEP - MAX_RETRIES is a config constant"  (with explanations)
    """
    cleaned = output.strip()

    # Try JSON first (in case model outputs JSON anyway)
    json_result = _try_parse_json(cleaned)
    if json_result:
        return json_result

    # Try text-based parsing: look for "N. KEEP" or "N. DROP" patterns
    text_result = _try_parse_text(cleaned, batch_size)
    if text_result:
        return text_result

    return None


def _try_parse_json(text: str) -> dict[str, bool] | None:
    """Try to parse JSON output."""
    cleaned = text
    # Strip code fences
    if cleaned.startswith("```"):
        first_nl = cleaned.find("\n")
        if first_nl != -1:
            cleaned = cleaned[first_nl + 1:]
        if cleaned.rstrip().endswith("```"):
            cleaned = cleaned.rstrip()[:-3].rstrip()

    # Find JSON object in text
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1:
        return None

    try:
        data = json.loads(cleaned[start:end + 1])
    except json.JSONDecodeError:
        return None

    if not isinstance(data, dict):
        return None

    result = {}
    for key, value in data.items():
        if isinstance(value, bool):
            result[key] = value
        elif isinstance(value, int):
            result[key] = value > 0
        elif isinstance(value, str):
            result[key] = value.lower() in ("true", "keep", "yes", "1")
        elif isinstance(value, dict):
            return None  # Nested scores, not binary
    return result if result else None


def _try_parse_text(text: str, batch_size: int) -> dict[str, bool] | None:
    """Parse text-based keep/drop output."""
    # Match patterns like: "1. KEEP", "1: DROP", "1 KEEP", "1. KEEP - reason"
    pattern = re.compile(
        r"(\d+)\s*[.):]\s*(keep|drop|KEEP|DROP)",
        re.IGNORECASE,
    )
    matches = pattern.findall(text)
    if not matches:
        return None

    result = {}
    for idx_str, decision in matches:
        idx = idx_str
        if 1 <= int(idx) <= batch_size:
            result[idx] = decision.upper() == "KEEP"

    return result if result else None


def evaluate(scores: dict[str, bool]) -> tuple[int, int, list[str]]:
    """Compare scores to expected. Returns (correct, total, error_details)."""
    correct = 0
    total = 0
    errors = []
    for key, expected_val in EXPECTED.items():
        total += 1
        actual = scores.get(key)
        if actual is None:
            errors.append(f"  {key}. {LABELS[key]}: MISSING")
        elif actual == expected_val:
            correct += 1
        else:
            exp = "KEEP" if expected_val else "DROP"
            act = "KEEP" if actual else "DROP"
            errors.append(f"  {key}. {LABELS[key]}: expected {exp}, got {act}")
    return correct, total, errors


def test_model_prompt(model: str, prompt_name: str, system_prompt: str) -> dict:
    """Test a single model with a single prompt. Returns result dict."""
    try:
        output, elapsed = call_ollama(model, system_prompt, TEST_VARIABLES)
    except Exception as e:
        return {"model": model, "prompt": prompt_name, "error": str(e)}

    scores = parse_binary_scores(output)
    if scores is None:
        return {
            "model": model,
            "prompt": prompt_name,
            "parsed": False,
            "raw": output[:200],
            "elapsed": elapsed,
        }

    correct, total, errors = evaluate(scores)
    return {
        "model": model,
        "prompt": prompt_name,
        "parsed": True,
        "correct": correct,
        "total": total,
        "accuracy": f"{correct}/{total}",
        "scores": scores,
        "errors": errors,
        "elapsed": elapsed,
        "raw": output[:300],
    }


def main():
    print("=" * 70)
    print("Binary Keep/Drop Variable Scoring Test (no format forcing)")
    print("=" * 70)

    print(f"\nExpected decisions:")
    for key in sorted(EXPECTED.keys()):
        verdict = "KEEP" if EXPECTED[key] else "DROP"
        print(f"  {key}. {LABELS[key]:15s} -> {verdict}")

    print(f"\nPrompts to test: {len(PROMPTS)}")
    for name, prompt in PROMPTS.items():
        print(f"\n  [{name}] ({len(prompt)} chars):")
        print(f"    {prompt[:100]}...")

    # Results matrix
    results = []

    for model in MODELS:
        print(f"\n{'=' * 70}")
        print(f"Model: {model}")
        print(f"{'=' * 70}")

        for prompt_name, system_prompt in PROMPTS.items():
            result = test_model_prompt(model, prompt_name, system_prompt)
            results.append(result)

            if "error" in result:
                print(f"  [{prompt_name}] ERROR: {result['error']}")
            elif not result["parsed"]:
                print(f"  [{prompt_name}] PARSE FAILED:")
                print(f"    {result['raw'][:120]}")
            else:
                status = "PERFECT" if result["correct"] == result["total"] else f"{result['accuracy']}"
                print(f"  [{prompt_name}] {status} ({result['elapsed']:.1f}s)")
                if result["errors"]:
                    for err in result["errors"]:
                        print(f"    {err}")
                # Show raw output for debugging
                raw_first_line = result["raw"].split("\n")[0][:80]
                print(f"    output: {raw_first_line}...")

    # Summary table
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Model':25s} | {'numbered':10s} | {'with-examples':15s} | {'question':10s}")
    print(f"{'-' * 25}-+-{'-' * 10}-+-{'-' * 15}-+-{'-' * 10}")

    for model in MODELS:
        row = f"{model:25s}"
        for prompt_name in PROMPTS:
            matching = [r for r in results if r["model"] == model and r["prompt"] == prompt_name]
            if matching:
                r = matching[0]
                if "error" in r:
                    cell = "ERROR"
                elif not r["parsed"]:
                    cell = "FAIL"
                else:
                    cell = f"{r['accuracy']}"
                    if r["correct"] == r["total"]:
                        cell += " *"
            else:
                cell = "N/A"
            width = 15 if prompt_name == "with-examples" else 10
            row += f" | {cell:{width}s}"
        print(row)

    print(f"\n* = perfect score")
    print(f"\nDone!")


if __name__ == "__main__":
    main()
