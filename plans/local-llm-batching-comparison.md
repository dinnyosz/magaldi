# Local LLM Inference: Batching vs Parallel Requests

## Key Concepts

### True Batching
Multiple prompts are stacked into a single tensor and processed in **one forward pass**. This shares matrix multiplications across sequences, resulting in dramatically better GPU utilization.

```
10 requests batched: ~1-2× forward pass time (shared compute)
```

### Parallel Slots / Concurrency
Multiple independent inference contexts run **separate forward passes**. The model weights are loaded once, but each request gets its own KV cache and compute.

```
10 requests with num_parallel=10: ~10× forward pass time (just concurrent)
```

## Tool Comparison (macOS Focus)

| Tool | True Batching | Notes |
|------|---------------|-------|
| **Ollama** | ❌ No | `OLLAMA_NUM_PARALLEL` creates N separate contexts with N separate forward passes. Context size is divided by N. |
| **LM Studio** | ❌ No | Queues requests sequentially. Workaround: load multiple model instances + load balancer. |
| **llama.cpp server** | ✅ Yes | `-cb` flag enables continuous batching (now default). Best GGUF option on Mac. |
| **MLX (mlx-lm)** | ⚠️ Partial | `batch_generate()` exists but server doesn't expose it yet. Open feature request. |
| **mlx_parallm** | ✅ Yes | Third-party library with batched KV cache for MLX. |
| **vLLM** | ✅ Yes | Best-in-class with PagedAttention, but CUDA-only (no macOS GPU support). |

## Ollama's Implementation

Ollama uses llama.cpp under the hood, which **does** support continuous batching. However, Ollama's `num_parallel` setting:

- Allocates N independent token-buffer contexts
- Runs N separate forward passes
- Divides context size by N (evidence of buffer duplication, not true batching)

This is concurrency, not batching.

## Recommendations

### For Batch Workloads on macOS

1. **Best option:** Run `llama-server` directly with continuous batching:
   ```bash
   llama-server -m model.gguf -c 4096 --parallel 4 -cb
   ```

2. **If using LM Studio:** Load multiple instances of the same model and use a load balancer to distribute requests.

3. **Experimental:** Use `mlx_parallm` for native Apple Silicon batching with MLX models.

### For Cloud/Linux with NVIDIA GPU

Use **vLLM** for production batch workloads:
```bash
vllm serve meta-llama/Llama-3-8B --max-num-seqs 64
```

## Client-Side Best Practices

Match your client concurrency to the server's capacity:

```python
import asyncio

CONCURRENCY = 4  # match num_parallel or batch slots

async def process_all(prompts):
    semaphore = asyncio.Semaphore(CONCURRENCY)
    
    async def bounded_query(prompt):
        async with semaphore:
            return await query(prompt)
    
    return await asyncio.gather(*[bounded_query(p) for p in prompts])
```

## Docker Limitations on macOS

Docker on macOS runs in a Linux VM with **no GPU passthrough** to Apple Silicon. This means:

- llama.cpp in Docker = CPU only
- vLLM in Docker = CPU only (defeats the purpose)
- Must run natively for Metal/MPS acceleration

## References

- [Ollama GitHub Issue #10699](https://github.com/ollama/ollama/issues/10699) - Discussion on batching vs concurrency
- [llama.cpp Discussion #4130](https://github.com/ggml-org/llama.cpp/discussions/4130) - Parallelization/batching explanation
- [mlx_parallm](https://github.com/willccbb/mlx_parallm) - Batched inference for MLX
- [mlx-lm Issue #178](https://github.com/ml-explore/mlx-lm/issues/178) - Request for batch processing
