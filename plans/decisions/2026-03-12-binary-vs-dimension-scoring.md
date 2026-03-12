# Decision: Binary Keep/Drop vs 7-Dimension Variable Scoring

**Date:** 2026-03-12
**Status:** Research complete, no production change yet

## Context

Variable scoring currently uses a 7-dimension system (config_value, architectural_role, data_definition, general_usefulness, value_complexity, naming_quality, scope_significance) scored 1-10 each. The LLM outputs `N. 9,2,1,8,3,7,9` and we apply `max(scores) >= 5` to decide keep/drop.

The individual dimension scores are **never used downstream** — only the binary keep/drop decision matters. No MCP tools, web UI, summarization, or queries use the individual numbers. They're computed, stored, and ignored.

Question: can we simplify to just asking the LLM "KEEP or DROP"?

## Research Findings

### Approach Comparison (41 test cases, 3-run average, temp=0.1)

| Model | Binary (baseline) | 7-Dimensions | Binary speed advantage |
|---|---|---|---|
| qwen3.5:4b | **82.9%** | 71.5% | 2.2x faster |
| qwen3.5:2b | 63.4% | **74.8%** | 2.8x faster |
| gemma3:4b | 68.3% | **82.1%** | 3.5x faster |

### Error Patterns

- **Binary weakness:** False negatives (drops things it should keep) — misses `DATABASE_URL`, `engine`, `celery` as keepable
- **Dimension weakness:** False positives (keeps things it should drop) — `result=func()`, `data=json.loads()` score high enough on max_score

### Prompt Variations Tested (6 binary prompts)

| Prompt style | qwen3.5:2b | qwen3.5:4b |
|---|---|---|
| baseline (KEEP/DROP lists) | 65.9% | **82.9%** |
| name-value (name+value guidance) | **68.3%** | 66.7% |
| examples-heavy (many examples) | 61.0% | 56.1% |
| checklist (question-based) | 56.1% | 62.6% |
| hybrid (checklist + examples) | 53.7% | 74.0% |
| decision-tree (step-by-step) | 56.9% | 74.0% |

Longer/more structured prompts cause massive parse failures on 2b (25-41 missing scores).

### Model Comparison (binary baseline prompt, 32 cases)

| Model | Score | Parse fails | Time |
|---|---|---|---|
| qwen3.5:4b | **32/32** | 0 | 4.9s |
| qwen3.5:2b | 30/32 | 0 | 2.9s |
| gemma3:4b | 29/32 | 0 | 2.9s |
| llama3.2:3b | 25/32 | 0 | 3.4s |
| qwen3:4b | 0/32 | 32 (outputs reasoning) | 5.2s |
| qwen3.5:0.8b | 0/32 | 32 | 1.0s |

### Single-Variable-Per-Message Approach

Tested sending one variable per API call instead of batching:
- gemma3:4b achieved 16/16 perfect with single calls
- But most models had 50%+ parse failure rate
- Throughput: 69ms/var with 8 parallel workers
- Not practical for large repos (1000 vars = 69 seconds)

## Conclusions

1. **qwen3.5:4b + binary = best accuracy** (83%), 2x faster than dimensions
2. **qwen3.5:2b needs dimensional structure** — the 7-axis format acts as a reasoning crutch, improving accuracy by 11pp over binary
3. **No prompt engineering closes the gap for 2b** — best binary prompt (name-value, 68%) still trails dimensions (75%)
4. **Binary is always faster** — fewer output tokens (10/var vs 25/var), shorter prompts
5. **False positives (keeping junk) are safer than false negatives (losing good stuff)** — dimensions produce more FP, binary more FN

## Decision

**No production change yet.** The current dimension-based approach with qwen3.5:4b works. If we switch:

- **If using qwen3.5:4b:** Switch to binary (simpler, faster, more accurate)
- **If using qwen3.5:2b:** Keep dimensions (the structure helps the smaller model)
- **If fine-tuning a model:** Train on binary labels derived from teacher's dimension scores

## Test Scripts

All in `tools/`:
- `test_binary_scoring.py` — Multi-model binary test, no format forcing
- `test_binary_single.py` — One-variable-per-message approach
- `test_binary_gemma.py` — Deep gemma3:4b + multi-model comparison
- `test_binary_qwen2b.py` — Deep qwen3.5:2b benchmark (consistency, batch size, categories)
- `test_binary_vs_dimensions.py` — Head-to-head binary vs 7-dimension comparison
- `test_binary_prompts.py` — 6 prompt variations across models
