#!/usr/bin/env python3
"""Quick test: JSON structured output scoring with small models.

Tests whether small models can produce valid JSON scores when constrained
by Ollama's structured output grammar.
"""

import json
import sys
import time

sys.path.insert(0, "src")

from magaldi_core.variable_scoring.prompts import SCORING_JSON_SCHEMA, SYSTEM_PROMPT
from magaldi_core.variable_scoring import _parse_scores
from shared.ai.llm_client import LLMClient

MODELS = [
    # Sub-1B models
    "ollama_chat/qwen3.5:0.8b",
    # 1-2B models
    "ollama_chat/qwen3:1.7b",
    "ollama_chat/granite3.3:2b",
    "ollama_chat/llama3.2:1b",
    "ollama_chat/qwen3.5:2b",
    # 2-4B models
    "ollama_chat/qwen2.5-coder:3b",
    "ollama_chat/llama3.2:3b",
    "ollama_chat/phi4-mini",
    "ollama_chat/qwen3:4b",
    "ollama_chat/qwen3.5:4b",
    "ollama_chat/gemma3:4b",
]

TEST_VARIABLES = """Score these variables:
1. [src/config.py] MAX_RETRIES = 3
2. [src/app.py] app = Flask(__name__)
3. [src/app.py] result = process_items(data)
4. [src/db.py] engine = create_engine(DATABASE_URL)
5. [src/db.py] logger = logging.getLogger(__name__)
6. [src/db.py] data = json.loads(payload)
7. [src/models.py] UserSchema = TypedDict("UserSchema", {"name": str, "email": str})
8. [src/utils.py] i = 0"""

API_BASE = "http://localhost:11434"


def test_model(model: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"Model: {model}")
    print(f"{'=' * 70}")

    client = LLMClient(model=model, api_base=API_BASE)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": TEST_VARIABLES},
    ]

    start = time.time()
    try:
        output = client.generate_from_messages(
            messages=messages,
            temperature=0.1,
            max_tokens=400,
            timeout=120,
            num_ctx=2048,
            response_format=SCORING_JSON_SCHEMA,
        )
    except Exception as e:
        print(f"  ERROR: {e}")
        return

    elapsed = time.time() - start
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Raw output ({len(output)} chars):")
    print(f"  {output[:500]}")

    # Try parsing
    scores = _parse_scores(output, batch_size=8)
    parsed_count = sum(1 for s in scores if s is not None)
    print(f"\n  Parsed: {parsed_count}/8 variables")

    for i, score in enumerate(scores):
        if score is not None:
            label = ["MAX_RETRIES", "app", "result", "engine", "logger", "data", "UserSchema", "i"][i]
            t = score.as_tuple()
            max_s = score.max_score
            verdict = "KEEP" if score.passes_threshold() else "DROP"
            print(f"    {i+1}. {label:15s} → {t}  max={max_s:2d}  {verdict}")
        else:
            print(f"    {i+1}. PARSE FAILED")

    # Quality check
    print(f"\n  Quality check:")
    if scores[0] and scores[0].passes_threshold():
        print(f"    ✓ MAX_RETRIES kept (config constant)")
    else:
        print(f"    ✗ MAX_RETRIES dropped (should be kept)")

    if scores[1] and scores[1].passes_threshold():
        print(f"    ✓ app kept (framework instance)")
    else:
        print(f"    ✗ app dropped (should be kept)")

    if scores[2] and not scores[2].passes_threshold():
        print(f"    ✓ result dropped (generic call result)")
    else:
        print(f"    ✗ result kept (should be dropped)")

    if scores[4] and scores[4].passes_threshold():
        print(f"    ✓ logger kept (infrastructure)")
    else:
        print(f"    ✗ logger dropped (should be kept)")

    if scores[5] and not scores[5].passes_threshold():
        print(f"    ✓ data dropped (generic local)")
    else:
        print(f"    ✗ data kept (should be dropped)")

    if scores[7] and not scores[7].passes_threshold():
        print(f"    ✓ i dropped (loop counter)")
    else:
        print(f"    ✗ i kept (should be dropped)")


if __name__ == "__main__":
    print("Testing JSON structured output scoring with small models")
    print(f"System prompt: {len(SYSTEM_PROMPT)} chars")
    print(f"JSON schema: {json.dumps(SCORING_JSON_SCHEMA, indent=2)[:200]}...")

    for model in MODELS:
        test_model(model)

    print(f"\n{'=' * 70}")
    print("Done!")
