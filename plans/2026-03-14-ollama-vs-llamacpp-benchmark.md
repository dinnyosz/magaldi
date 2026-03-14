# Ollama vs llama-server Benchmark Results (2026-03-14)

## Setup

- **Repo**: magaldi/magaldi (1,191 files, 489K lines, ~20K elements)
- **Hardware**: Same machine, same models (Qwen3 4B + 1.7B)
- **Workers**: 32

## Phase Timing Comparison

| Phase | Ollama instruct | llama-server Qwen3 | Ratio |
|-------|----------------|-------------------|-------|
| Phase 1: Discovery | 0.5s | 0.5s | 1.0x |
| Phase 2: Change Detection | 0.3s | 0.3s | 1.0x |
| Phase 3: Parsing | 25s | 25s | 1.0x |
| Phase 4: Variable Scoring | 29min | 24min | 1.2x |
| Phase 5: Processing | **6h 14min** | **3h 44min** | **1.7x** |
| Phase 6: Analysis | 6.8min | 7min | 1.0x |
| Phase 7: Feature Extraction | 30min | 24min | 1.2x |
| Phase 8: Glossary Extraction | 15min | 0.3s | N/A (new) |
| **TOTAL** | **7h 36min** | **4h 41min** | **1.6x** |

## Models Used

**Ollama run:**
- `qwen3:4b-instruct` (+ tiered: -1k, -2k, -4k, -8k, -16k, -32k)
- `qwen3:1.7b-instruct` (+ tiered: -1k, -2k, -4k, -8k, -16k, -32k)
- `qwen3-embedding:0.6b`

**llama-server run:**
- `Qwen3-4B-Q4_K_M` (GGUF, with `--parallel 32`)
- `Qwen3-1.7B-Q4_K_M` (GGUF)
- Embeddings via Ollama (same)

## Processing Stats

| Metric | Ollama | llama-server |
|--------|--------|-------------|
| Elements processed | 20,113 | 20,564 |
| Failed | **222 (1.1%)** | **0** |
| Avg wall time | 1.117s | 0.655s |
| Avg summarize time | 5.399s | 4.931s |
| Avg embed time | 2.115s | 1.348s |
| Total tokens | 7.3M | 6.4M |

## Variable Scoring

| Metric | Ollama | llama-server |
|--------|--------|-------------|
| Kept | 5,140 | 5,429 |
| Dropped | 9,003 | 8,681 |
| Keep rate | 36% | 38% |
| Errors | 0 | 0 |

## Root Cause of Slowdown

Ollama defaults to `OLLAMA_NUM_PARALLEL=1`, processing requests via FIFO queue (essentially serial). llama-server uses `--parallel 32` with continuous batching, processing multiple requests concurrently on GPU with shared KV cache.

Community benchmarks confirm: under concurrent load, llama.cpp achieves 2-3.3x higher throughput than Ollama.

## 222 Failures

Not investigated yet. Did not occur on llama-server. Likely Ollama timeout/connection issues under sustained load with 32 concurrent workers.

## Recommendations

1. **Parse pipeline**: Use llama-server for batch processing (Phase 4-5-7). 1.6x faster, 0 failures.
2. **MCP/interactive**: Ollama is fine for single-request workloads (MCP tools, chat).
3. **Hybrid setup**: Could run both — Ollama for MCP/embedding, llama-server for parse.
4. **If staying on Ollama**: Set `OLLAMA_NUM_PARALLEL=4+` and investigate failures.
