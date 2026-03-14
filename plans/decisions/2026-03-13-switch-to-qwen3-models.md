# Decision: Switch parse models from Qwen3.5 to Qwen3

**Original plan:** Use Qwen3.5-4B and Qwen3.5-2B for summarization and variable scoring in the parse pipeline.

**Deviation:** Switching to Qwen3-4B and Qwen3-1.7B as the default parse models.

**Why:** User requested adding Qwen3 models and switching the parse pipeline to use them. Qwen3 models offer improved performance over Qwen3.5 with better reasoning capabilities and efficiency.

**Options considered:**
1. Keep Qwen3.5 as defaults, add Qwen3 as optional alternatives - simpler but doesn't test the new models
2. Switch defaults to Qwen3 - directly uses the new models for all parse operations

**Final decision:** Switch all parse model assignments to Qwen3 variants (qwen3-4b for summarize_model and variable_scoring_model, qwen3-1.7b for summarize_model_small). Qwen3.5 models remain configured and available for manual selection.

**Impact:**
- All new parse operations will use Qwen3 models by default
- Variable scoring and code summarization will benefit from improved reasoning
- Faster inference for small model tasks (1.7B vs 2B)
- Backward compatibility maintained - users can still specify Qwen3.5 models if needed

**Date:** 2026-03-13
