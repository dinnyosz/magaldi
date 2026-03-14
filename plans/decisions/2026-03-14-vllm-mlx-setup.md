## Decision: vllm-mlx configuration workarounds

**Original plan:** Use vllm-mlx with --continuous-batching, --chunked-prefill-tokens 512, --enable-prefix-cache, and max-num-seqs matching worker count.

**Deviation:** Disabled chunked-prefill-tokens and prefix cache. Also fixed thinking model detection to exclude instruct variants.

**Why:** Two vllm-mlx 0.2.6 bugs:
1. `--chunked-prefill-tokens` causes `'list' object has no attribute 'shape'` crash in batch generation step
2. Prefix cache triggers the same crash when batch_generator recreates with cached entries on sampler param changes

A third issue: LLMClient's `_build_kwargs` sent `chat_template_kwargs.enable_thinking=False` for ALL local models (including non-thinking instruct models), which also crashed vllm-mlx.

**Options considered:**
1. Wait for vllm-mlx upstream fix — unknown timeline, blocks testing
2. Disable chunked-prefill + prefix cache — loses some throughput optimization but works
3. Downgrade vllm-mlx — no known working version with these features

**Final decision:** Option 2. Also fixed three code issues:
- `_check_is_thinking_model()`: Added `"instruct" in base` check so instruct models skip thinking suppression entirely
- `_build_kwargs()`: Changed `elif _is_local:` to `elif self._is_thinking_model and _is_local:` so non-thinking models don't get `chat_template_kwargs`
- `summarization.py` + `embedding.py`: Auto-append `/v1` to api_base for llamacpp/vllm-mlx providers (was missing, would cause 404s)
