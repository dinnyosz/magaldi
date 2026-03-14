## Decision: Use no-think Ollama Modelfile variants for Qwen3

**Original plan:** Use default Ollama Qwen3 models (qwen3:4b, qwen3:1.7b) with `think=false` API parameter to suppress thinking mode.

**Deviation:** Switched to user-created Modelfile variants (qwen3:4b-instruct-2k, qwen3:1.7b-2k) that have the `<think>` primer removed from the chat template.

**Why:** Ollama's Go-based chat templates for Qwen3 hardcode `<think>` as the assistant turn primer. The `think=false` API parameter is ineffective because the template already starts the assistant response in thinking mode. This caused variable scoring to exhaust its entire token budget on reasoning, return empty content, trigger LLMError, and default all variables to KEEP (100% keep rate vs expected ~40-60%). Previous llama-server runs using Jinja templates had 60-80% drop rates.

**Options considered:**
1. `/no_think` system message directive — works with llama-server's Jinja templates but Ollama's Go templates treat it as literal text, not a directive. Model still reasons inline.
2. `think=false` API parameter — Ollama passes this to the template but the Go template ignores it and still prepends `<think>` to assistant turn.
3. Custom Modelfiles with `<think>` primer removed — user-created `-instruct-2k` and `-2k` variants end the template with `<|im_start|>assistant\n` instead of `<|im_start|>assistant\n<think>\n`. This completely prevents thinking mode.
4. Three-layer defense (API param + system message + strip tags) — implemented as fallback but doesn't prevent the core issue of wasted token budget on reasoning.

**Final decision:** Use custom Modelfile variants (option 3). Testing confirmed qwen3:4b-instruct-2k produces perfect variable scoring output: 47% drop rate, 32 tokens, clean `1. KEEP / 2. DROP` format. The three-layer defense (option 4) was kept as additional safety but the Modelfile change is the primary fix.
