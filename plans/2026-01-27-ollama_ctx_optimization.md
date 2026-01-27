This guide outlines how to optimize Ollama for high-throughput parallel inference by using Context Tiering via model aliases.
🚀 The Core Concept: Model Aliasing
To save memory without sacrificing speed, you should not use one "catch-all" large context size. Instead, create multiple aliases of the same model, each locked to a specific context tier. Because Ollama uses Memory Mapping (mmap), the model weights are shared across all aliases—you only pay the VRAM "tax" for the weights once.
1. The Strategy: Power-of-2 Tiers
Group your requests into specific tiers to minimize configuration changes and maximize VRAM efficiency.
| Alias Name | num_ctx | Ideal For... |
|---|---|---|
| model-2k | 2048 | Simple chat, classification, short summaries. |
| model-4k | 4096 | RAG results, standard document analysis. |
| model-8k | 8192 | Long-form content generation. |
| model-16k | 16384 | Large codebases or multi-document analysis. |
| model-32k | 32768 | Deep research and massive context windows. |
🛠️ Implementation Steps
Step 1: Create the Aliases
Run this bash script to quickly generate your tiered models. Replace llama3 with your preferred model name.
for size in 2048 4096 8192 16384 32768; do
  echo "FROM llama3\nPARAMETER num_ctx $size" | ollama create llama3-${size}
done

Step 2: Configure the Ollama Server
You must allow Ollama to keep these variants "warm" in memory simultaneously. Set these environment variables before starting the Ollama service.
 * OLLAMA_MAX_LOADED_MODELS=5: Allows all tiers to stay in VRAM.
 * OLLAMA_NUM_PARALLEL=8: Allows your 8 workers to process requests concurrently within those tiers.
 * OLLAMA_KEEP_ALIVE=-1: Keeps the tiers loaded indefinitely (prevents unloading after 5 mins).
Step 3: Logic for your Application
In your code, calculate the token count of your prompt and route it to the smallest possible alias:
def get_model_alias(token_count):
    if token_count <= 2048: return "llama3-2048"
    if token_count <= 4096: return "llama3-4096"
    if token_count <= 8192: return "llama3-8192"
    if token_count <= 16384: return "llama3-16384"
    return "llama3-32768"

📊 Why This Wins
1. Memory Efficiency
Without this, 8 workers at 32k context would require massive VRAM (e.g., ~35GB+). With tiering, if 6 of your 8 workers are doing short tasks (2k), you save nearly 20GB of VRAM, which can be used to run larger, more intelligent models.
2. Zero Reload Latency
If you don't use aliases, changing num_ctx forces Ollama to:
 * Stop the current runner.
 * Purge VRAM.
 * Reload weights and re-allocate the buffer.
   This takes 2-10 seconds per change. Aliases keep the different "lanes" open so switching is instant.
3. Shared Weights
The llama3 weights (the 5GB-40GB of parameters) are not duplicated. Ollama is smart enough to point all aliases to the same physical memory block.
> Note: Ensure your hardware has enough VRAM to hold the Weights + The sum of the KV Caches for the active workers.
> 
Would you like me to help you calculate the exact VRAM footprint for your specific GPU and model size?
