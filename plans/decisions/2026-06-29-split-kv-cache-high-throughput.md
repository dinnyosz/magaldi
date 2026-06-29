## Decision: Switch from unified to split KV cache (high-throughput mode)

**Original plan:** Use `--kv-unified` flag to share a single KV cache pool across all parallel slots per model. This was set up in March 2026 (decision: `2026-03-18-unified-kv-cache-ctx-size.md`) to reduce memory waste with per-tier context sizing.

**Deviation:** Remove `--kv-unified` and enable split KV cache mode by setting `LLAMA_SET_ROWS=1` environment variable on the llama-server process.

**Why:** llama.cpp PR #14363 ("add high-throughput mode", merged July 2025) introduced split KV cache where each parallel slot gets its own KV buffer. This eliminates wasteful cross-sequence attention compute that the unified mode performs. Benchmarks show ~3× token generation throughput improvement with 32 parallel sequences (154 → 486 t/s on Qwen 2.5 3B Q8_0, M2 Ultra).

**Options considered:**
1. Keep `--kv-unified` — no code changes, but wastes significant compute on cross-sequence attention that gets discarded. Magaldi sends thousands of independent requests with zero shared prefix.
2. Enable split KV cache (remove `--kv-unified`, add `LLAMA_SET_ROWS=1`) — each slot gets isolated KV buffers, attention only computes within-sequence. Ideal for magaldi's embarrassingly parallel workload. Slightly higher memory (per-slot buffers vs shared pool), but the throughput gain far outweighs it.
3. Wait for `LLAMA_SET_ROWS` to become default — the PR notes this will eventually be default once all backends have `ggml_set_rows()` coverage. But Metal already supports it, and magaldi runs exclusively on Apple Silicon.

**Final decision:** Option 2 — enable split KV cache now. Magaldi's workload (many independent summarization/scoring requests, no shared prefix, cache-reuse already disabled) is the textbook use case for this optimization. The `LLAMA_SET_ROWS=1` env var is the only requirement for Metal backend, which is already our target platform.
