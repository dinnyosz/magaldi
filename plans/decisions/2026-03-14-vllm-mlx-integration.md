## Decision: Use vllm-mlx (open-source) instead of vMLX (closed-source)

**Original plan:** User wanted to investigate vMLX from vmlx.net for integration similar to llamacpp.

**Deviation:** Chose vllm-mlx (github.com/waybarrios/vllm-mlx) instead of vMLX.

**Why:** vMLX is a closed-source macOS GUI app with no CLI, no Python API, and no programmatic control. vllm-mlx is open-source (Apache 2.0), has a CLI (`vllm-mlx serve`), exposes an OpenAI-compatible API, and supports continuous batching — making it suitable for automated server management.

**Options considered:**
1. vMLX (vmlx.net) - Closed-source, GUI-only, has OpenAI-compatible API at localhost:8000 but no way to start/stop/configure programmatically
2. vllm-mlx (waybarrios/vllm-mlx) - Open-source, CLI-based, OpenAI-compatible API, continuous batching, 21-87% faster than llama.cpp in benchmarks
3. vLLM Metal (vllm-project/vllm-metal) - Community hardware plugin for vLLM on Apple Silicon, still experimental

**Final decision:** vllm-mlx — best balance of performance, open-source accessibility, and CLI-based management that fits magaldi's architecture.

---

## Decision: Restructure CLI into backend-specific command groups

**Original plan:** All LLM server management was under `magaldi llm *`, which only managed llama.cpp but the name implied a generic LLM interface.

**Deviation:** Split into three backend-specific groups: `magaldi llamacpp *`, `magaldi ollama *`, `magaldi vllm-mlx *`.

**Why:** The generic `llm` name was misleading — it only managed llama.cpp. Ollama had backend-specific code but no CLI commands. Adding vllm-mlx as a third backend made the restructuring necessary. Each backend has different concepts (GGUF models vs HuggingFace repos vs Ollama tags) that warrant separate command groups.

**Options considered:**
1. Keep `magaldi llm *` and add subcommands per backend (e.g., `magaldi llm llamacpp serve`) - Too verbose, nested groups
2. Rename to `magaldi llamacpp *` and add sibling groups - Clean separation, honest naming, easy to extend
3. Keep generic `magaldi llm *` that auto-detects backend - Too magical, hard to debug

**Final decision:** Option 2 — separate groups with a hidden deprecated `magaldi llm *` alias for backward compatibility.

---

## Decision: No separate pull/download command for vllm-mlx

**Original plan:** llamacpp has `magaldi llamacpp pull` to download GGUF models from HuggingFace.

**Deviation:** vllm-mlx has no equivalent pull command.

**Why:** The upstream `vllm-mlx serve` auto-downloads MLX models from HuggingFace on first use via `mlx-lm`. MLX models use safetensors format in HF repos (not GGUF files), so the download mechanism is fundamentally different. Adding a separate pull step would duplicate what the upstream tool already handles seamlessly.

**Final decision:** Let `vllm-mlx serve` handle model downloads automatically. The model source is the `name` field in `magaldi.yaml` (a HuggingFace repo ID like `mlx-community/Qwen3-4B-4bit`).
