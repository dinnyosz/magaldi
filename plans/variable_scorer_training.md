# Variable Scorer Model Training — Full Context & Status

## Overview

Training a small LLM (Qwen2.5-1.5B-Instruct) to replace the large teacher model (qwen3-coder:30b) for Phase 4 variable scoring in Magaldi's parse pipeline. The model scores code variables on 7 dimensions (1-9 scale) to decide which are worth indexing.

## The 7 Scoring Dimensions

1. `config_value` — Is this a configuration constant/setting?
2. `architectural_role` — Does it define architecture (routes, middleware, DI)?
3. `data_definition` — Does it define data structures (schemas, models)?
4. `general_usefulness` — Is it generally useful for understanding the codebase?
5. `value_complexity` — How complex/interesting is the assigned value?
6. `naming_quality` — How descriptive is the variable name?
7. `scope_significance` — Module-level vs local scope importance?

Variables with any dimension ≥ threshold (default 7) are kept; rest are dropped.

## Architecture

```
Teacher model (qwen3-coder:30b via Ollama)
  → generates scored training data (11,873 raw samples)
  → assembled into ChatML conversations (2,538 train / 282 val)

Student model (Qwen2.5-1.5B-Instruct)
  → fine-tuned with LoRA via mlx-lm on Apple Silicon
  → fused → converted to GGUF → served via Ollama
  → replaces teacher for production scoring
```

## Key Files

| File | Purpose |
|------|---------|
| `tools/training/generate_scoring_data.py` | Generates training data using teacher model. Scores variables from test repos. |
| `tools/training/train_variable_scorer.py` | Fine-tunes model using mlx-lm (Apple Silicon) or Unsloth (NVIDIA). |
| `tools/training/evaluate_scorer.py` | Evaluates trained model via Ollama against validation set. |
| `tools/training/configs/variable_scorer.yaml` | Training hyperparameters config. |
| `tools/training/configs/variable_scorer_0.5b.yaml` | Alternative config for 0.5B model (not yet tested). |
| `tools/training/exports/Modelfile` | Ollama Modelfile for the trained GGUF model. |
| `tools/training/data/variable_scorer/raw/` | 11,873 individual scored variable JSON files. |
| `tools/training/data/variable_scorer/train.jsonl` | 2,538 assembled training conversations. |
| `tools/training/data/variable_scorer/validation.jsonl` | 282 assembled validation conversations. |

## Training Data Pipeline

### Step 1: Generate raw scored data
```bash
.venv/bin/python tools/training/generate_scoring_data.py \
  --repos-dir test_repos --sample-size 10000 --cache --seed 123
```
- Uses 50 test repos in `test_repos/`
- Teacher model: `qwen3-coder:30b` via Ollama
- Speed: ~2.1 variables/sec
- Output: individual JSON files per variable in `data/variable_scorer/raw/`

### Step 2: Assemble into conversations
The `build_training_batches()` function in `generate_scoring_data.py` groups variables into conversations with mixed batch sizes (1-30 variables per conversation), adds the system prompt, and creates ChatML format.

**Important**: The `format_training_example()` function previously injected `<think>\n\n</think>\n\n` tags for Qwen3. This was REMOVED for Qwen2.5 (non-thinking model). Any new assembly must NOT include think tags.

Current assembly: 3 random seeds for augmentation, deduplication by assistant content → 2,538 train / 282 val.

### Step 3: Train
```bash
.venv/bin/python tools/training/train_variable_scorer.py \
  --train-data tools/training/data/variable_scorer/train.jsonl \
  --val-data tools/training/data/variable_scorer/validation.jsonl \
  --output-dir tools/training/models/variable-scorer-qwen2.5-v2 \
  --config tools/training/configs/variable_scorer.yaml \
  --backend mlx -v
```

### Step 4: Export to GGUF
```bash
# Convert fused model to GGUF
python llama.cpp/convert_hf_to_gguf.py \
  tools/training/models/variable-scorer-qwen2.5-v2/merged \
  --outfile tools/training/exports/magaldi-variable-scorer-qwen2.5-1.5b-f16.gguf \
  --outtype f16

# Quantize
llama.cpp/build/bin/llama-quantize \
  tools/training/exports/magaldi-variable-scorer-qwen2.5-1.5b-f16.gguf \
  tools/training/exports/magaldi-variable-scorer-qwen2.5-1.5b-q8_0.gguf Q8_0

# Register with Ollama
ollama create magaldi-scorer -f tools/training/exports/Modelfile
```

### Step 5: Evaluate
```bash
.venv/bin/python tools/training/evaluate_scorer.py \
  --model magaldi-scorer --limit 50
```

## Current Training Config

```yaml
training:
  backend: mlx
  base_model: Qwen/Qwen2.5-1.5B-Instruct
  lora_rank: 16
  lora_alpha: 32
  lora_dropout: 0.1
  lora_target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
  learning_rate: 5.0e-5
  weight_decay: 0.01
  warmup_ratio: 0.1
  num_layers: 28           # All transformer layers get LoRA adapters
  epochs: 10
  batch_size: 2
  gradient_accumulation_steps: 2
  max_seq_len: 2048
  mask_prompt: true         # Loss only on assistant responses
  seed: 42
```

## Training Progress (v2 — current run)

| Checkpoint | Val Loss | Notes |
|-----------|----------|-------|
| Iter 1    | 0.467    | Initial baseline |
| Iter 1270 | 0.109    | −77% — massive improvement |
| Iter 2540 | 0.084    | −23% — still improving |
| Iter 3810 | 0.068    | −19% — steady decline |
| Iter 5080 | ???      | Not yet reached |
| Iter 6350 | ???      | Final (end of 10 epochs) |

- Peak memory: 9.1 GB (very reasonable)
- Training speed: ~0.15-0.18 it/sec
- Total iterations: 6,350
- Currently at: ~iter 4,470 (70% done)
- ETA: ~3-4 more hours from current point

## Previous Attempts & Lessons Learned

### Attempt 1: Qwen3-0.6B (prior session)
- Trained successfully via MLX
- Output correct via MLX adapter inference
- **FAILED via Ollama/GGUF**: garbage output
- Root cause: Qwen3's `<think>` tags handled differently in MLX vs llama.cpp

### Attempt 2: Qwen3-1.7B (prior session)
- Larger model to try to fix Qwen3 issues
- Hit NaN loss and mode collapse
- Same think tag incompatibility between MLX and GGUF

### Attempt 3: Qwen2.5-1.5B v1 (this session)
- Switched to non-thinking model (user's suggestion)
- Only 226 training examples
- `num_layers` bug: was set to `lora_rank` (16) instead of 28
- **Result**: Mode collapse — model outputs all `1,1,1,1,1,1,1` for most inputs
- Val loss reached 0.183 (decent but model didn't generalize)

### Attempt 4: Qwen2.5-1.5B v2 (current)
- 2,538 training examples (11x more)
- Fixed `num_layers` to 28 (all layers)
- LR 5e-5, 10 epochs, all 7 target modules
- Val loss already at 0.068 and still dropping
- **Status: IN PROGRESS** — training running

## Bugs Fixed

1. **`num_layers = lora_rank` bug** in `train_variable_scorer.py`
   - Was: `"num_layers": lora_rank` (set to 16)
   - Fixed: `"num_layers": num_layers` (reads from config, set to 28)
   - Impact: Only 16/28 layers had LoRA adapters, severely limiting model capacity

2. **Think tag injection** in `generate_scoring_data.py`
   - Was: `format_training_example()` added `<think>\n\n</think>\n\n` prefix to assistant content
   - Fixed: Removed those 4 lines
   - Impact: Qwen2.5 doesn't use think tags, so this caused format mismatch

3. **Insufficient training data**
   - Was: 226 training conversations assembled from 2,872 raw samples
   - Fixed: Generated 10K+ more samples (11,873 total raw), assembled with 3 random seeds → 2,538 unique conversations

## Data Distribution Analysis

- At the score-line level: ~49% are all-1s (trivial variables), ~51% have non-1 scores
- At the individual score level: ~70% of all scores are 1
- This class imbalance was the primary cause of mode collapse with small data
- With 2,538 examples (vs 226), the model sees enough varied examples to learn the non-trivial patterns

## Remaining Steps After Training Completes

1. **Fuse LoRA adapters** — `train_variable_scorer.py` does this automatically after training
2. **Convert to GGUF** — Use `convert_hf_to_gguf.py` and `llama-quantize`
3. **Register with Ollama** — `ollama create magaldi-scorer -f Modelfile`
4. **Evaluate** — Run `evaluate_scorer.py` to check:
   - Format accuracy (model produces parseable scores)
   - Line-level exact match rate
   - Per-score accuracy
   - Mean absolute error per dimension
   - Keep/drop decision accuracy
5. **Integration test** — Run Phase 4 with the new model on a real repo

## Deep Research: Improvement Recommendations

_Research conducted 2026-03-12 from industry best practices, Unsloth docs, Sebastian Raschka's experiments, Predibase distillation playbook, mlx-lm docs, and community benchmarks._

---

### 🔴 Critical: Overfitting Risk (10 Epochs is Too Many)

The current config trains for **10 epochs**, which contradicts every major recommendation:

- **Unsloth guide**: "1-3 epochs; training >3 epochs offers diminishing returns and increases memorization risk"
- **Sebastian Raschka**: "Multi-epoch training on static datasets often deteriorates results due to overfitting. Single-epoch training is recommended for instruction fine-tuning."
- **Predibase playbook**: Training longer helps, but only with robust evaluation — without early stopping, you risk fitting to noise

**The val loss dropping from 0.467 → 0.068 looks great, but could mask overfitting.** Val loss on a small set (282 examples) can decrease while generalization worsens — the model memorizes the scoring patterns of your specific repos rather than learning general scoring rules.

**Recommendation**:
- Reduce to **3 epochs** for v3
- Add **early stopping** with patience=3 on validation loss
- Keep a separate **held-out test set** (10% of raw data, never used in training/validation) to check true generalization
- If you must run more epochs, use the best checkpoint by val loss, not the last one

---

### 🔴 Critical: Instruction Masking May Hurt

Your config uses `mask_prompt: true` (loss only on assistant responses). Research from Raschka's analysis of "Instruction Tuning With Loss Over Instructions" (2024) found:

> "Not masking instructions actually performs better than the conventional practice of masking them. Including instruction tokens in the loss helps reduce overfitting tendency."

This is especially relevant for **small datasets** and **short responses** (your scoring outputs are very short — just `N. X,X,X,X,X,X,X` lines).

**Recommendation**: Try `mask_prompt: false` in v3. The system prompt teaches the model the scoring rubric — including it in the loss may help the model internalize the scoring criteria better, not just memorize output patterns.

---

### 🟡 High Priority: Add Cosine Decay LR Schedule

Your current config uses a constant learning rate with warmup. mlx-lm natively supports cosine decay scheduling via YAML config:

```yaml
lr_schedule:
  name: cosine_decay
  warmup: 200
  warmup_init: 1e-6
  arguments: [5e-5, 6350, 1e-6]  # [initial_lr, total_steps, min_lr]
```

**Why**: Cosine decay smoothly reduces LR toward the end of training, preventing the model from "overshooting" the loss landscape in later epochs. This is standard practice for LoRA fine-tuning.

**Known bug**: There's a documented issue (ml-explore/mlx#2617) where cosine decay starts prematurely during warmup. Test by logging LR values to verify correct behavior.

**Recommendation**: Add `lr_schedule` config to `variable_scorer.yaml`. Set `arguments` to `[5e-5, <total_iters>, 1e-6]`.

---

### 🟡 High Priority: Use QLoRA (4-bit Base) to Cut Memory ~60%

Current peak memory is 9.1 GB with full-precision base model + LoRA. mlx-lm supports QLoRA automatically — if you point `base_model` to a 4-bit quantized model, it trains QLoRA instead of LoRA.

- **Full precision 1.5B**: ~9 GB peak
- **4-bit QLoRA 1.5B**: estimated ~3-4 GB peak
- **Quality impact**: Negligible for structured output tasks like scoring (Raschka found QLoRA had "negligible performance impact" vs LoRA)
- **Speed tradeoff**: ~39% slower training per iteration, but enables larger batch sizes which may offset

**Recommendation**: For the 1.5B model, current 9.1 GB is manageable. But if you try the **0.5B model**, QLoRA would let you run batch_size=8+ for much faster training. More relevant if you ever scale to larger models.

---

### 🟡 High Priority: LoRA Rank 8 May Be Sufficient

Current rank=16 with alpha=32 (2x ratio). Research consensus:

- **Raschka**: "There is very little statistical difference between ranks 8 and 256 when LoRA is applied across all layers"
- **Unsloth**: "Choose 16 or 32" as starting point
- **Key insight**: What matters most is applying LoRA to ALL linear layers (which you already do — all 7 target modules)

For your task (structured numeric output, not creative generation), rank 8 likely suffices:
- ~50% fewer trainable parameters → faster training
- Better regularization → less overfitting risk
- Lower memory usage

**Recommendation**: Test rank=8 with alpha=16 in v3. Compare eval metrics against rank=16 — if similar, keep the smaller adapter.

---

### 🟡 High Priority: Improve Data Diversity Strategy

Current: 11,873 raw samples from 50 test repos, ~49% all-1s (trivial variables).

The **Predibase distillation playbook** recommends:

1. **Ablation studies**: Test 25%, 50%, 75%, 100% of data to find where marginal utility diminishes. If performance plateaus at 50%, focus on quality not quantity.

2. **Rebalance the dataset**: 49% all-1s creates a strong bias. Options:
   - **Undersample all-1s to 30%** (matching the "~30% worth keeping" rule from the system prompt)
   - **Oversample non-trivial examples 2x** (with slight noise: ±1 on borderline scores)
   - Don't duplicate — **re-score with different temperature** (0.3, 0.5) for natural variation

3. **Harder examples matter most**: Add adversarial cases that are tricky:
   - `result = db.execute(complex_query)` → should be low (generic assignment) despite complex-looking value
   - `x = CONFIG` → short name but high scope significance
   - `TIMEOUT = 30` → trivial value but high config_value

4. **Cross-validation**: Instead of a single 90/10 split, use **k-fold** (k=5) to get more reliable metrics and catch overfitting across different data slices.

---

### 🟢 Medium Priority: Try the 0.5B Model

The `variable_scorer_0.5b.yaml` exists but hasn't been tested. For structured scoring output (essentially a classification/regression task, not generation):

- **0.5B may be sufficient** — you're teaching pattern matching, not creative reasoning
- **Inference speed**: ~3x faster than 1.5B → directly improves Phase 4 throughput
- **Training speed**: ~3x faster → faster iteration cycles
- **Memory**: Could enable batch_size=8-16 → more stable gradients

**Recommendation**: Once v2 evaluation completes, train a 0.5B v1 with the optimized hyperparameters from v3 learnings. Compare eval metrics — if within 5% of 1.5B, use 0.5B for production.

---

### 🟢 Medium Priority: GGUF Export Strategy

Your config says `quantization: Q4_K_M` but the v1 export used Q8_0. Research findings:

- **Q4_K_M**: Best quality-to-size ratio for 1.5B models. Uses K-quant two-level scheme (piecewise-affine) for better accuracy per bit than Q8_0's naive INT8.
- **Q8_0**: Near-lossless but 2x the file size. Overkill for a scoring model.
- **For Qwen2.5 specifically**: "Performance under quantization remained remarkably stable, with Q4_K_M within tolerable margins of BF16"

**Recommendation**: Export both Q4_K_M and Q8_0, evaluate both via `evaluate_scorer.py`. If Q4_K_M passes all success criteria, use it — smaller file = faster model loading in Ollama.

---

### 🟡 High Priority: Token-Budget Batch Packing

Current `build_training_batches()` picks random batch sizes (1-30 vars) without checking whether they fit in `max_seq_len`. And `build_user_prompt()` only truncates individual mega-long variables — it doesn't enforce a total budget. This means large batches can exceed `max_seq_len` and get silently truncated during training, losing data.

**Fix**: Rewrite batch assembly to use greedy bin-packing against the token budget:

```
budget = max_seq_len (e.g. 1024 tokens)
budget -= system_prompt_tokens (~400)
budget -= chatml_overhead (~20)

for each variable in shuffled results:
    cost = estimate_tokens(user_line) + estimate_tokens(score_line)
    if cost <= remaining_budget:
        add to current batch, deduct cost
    else:
        flush current batch as a training example
        start new batch with this variable
```

**Benefits**:
- Every training example is *guaranteed* to fit in `max_seq_len` — no truncation, no wasted padding
- Batch sizes become dynamic: short variables → bigger batches, long variables → smaller batches
- Eliminates the arbitrary `[(1,3), (4,10), (11,20), (21,30)]` distribution
- Matches how production scoring works (it should also pack to budget)

**Where to implement**: `build_training_batches()` in `generate_scoring_data.py`. Also update production `build_user_prompt()` in `prompts.py` to use the same packing logic.

---

### 🟢 Medium Priority: Faster Training Speed

Current: 0.15-0.18 it/sec (~10 hours total for 6,350 iterations). Options to speed up:

1. **Fewer epochs** (3 instead of 10) → 3x faster wall clock

2. **Larger effective batch size**: If memory allows, try `batch_size: 4, gradient_accumulation_steps: 2` for effective batch 8. Larger batches = fewer iterations needed.

3. **Reduce eval frequency**: Current `steps_per_eval = iters // 5` (every 20% of training). Evaluation on 282 examples at each eval is expensive. Try `steps_per_eval = iters // 3` (eval only 3 times).

4. **Token-budget packing** (see above): eliminates wasted padding → faster per-iteration.

---

### 🟢 Low Priority: Handling Class Imbalance Without Custom Loss

Since mlx-lm doesn't support custom loss functions (focal loss, weighted CE), handle imbalance through **data engineering**:

1. **Undersample**: Cap all-1s examples at 30% of dataset
2. **Augment non-trivial examples**: For variables with high scores, create 2-3 variants:
   - Same variable, different batch context (different surrounding variables)
   - Slight score perturbation (±1 on non-extreme dimensions) to teach confidence boundaries
3. **Curriculum batching**: Order training examples so early batches have more diverse (non-trivial) examples, later batches are more random. This isn't true curriculum learning but gives the model early exposure to the harder cases.

---

### Concrete v3 Config Recommendation

Based on all research, here's the optimized config:

```yaml
training:
  backend: mlx
  base_model: Qwen/Qwen2.5-1.5B-Instruct
  lora_rank: 8           # Down from 16 — sufficient for structured output
  lora_alpha: 16          # Maintain 2x ratio
  lora_dropout: 0.05      # Reduced — "not that useful" per Unsloth
  lora_target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
  learning_rate: 2.0e-4   # Higher LR — standard for LoRA per Unsloth guide
  weight_decay: 0.01
  warmup_ratio: 0.05      # 5% warmup (was 10%)
  num_layers: 28           # Keep all layers
  epochs: 3               # Down from 10 — prevent overfitting
  batch_size: 4            # Up from 2 (test memory first)
  gradient_accumulation_steps: 2  # Effective batch = 8
  max_seq_len: 1024        # Down from 2048 (verify data fits first)
  mask_prompt: false       # CHANGED — include instruction in loss
  seed: 42
  lr_schedule:
    name: cosine_decay
    warmup: 100
    warmup_init: 1e-6
    arguments: [2e-4, 950, 1e-6]  # [lr, total_iters, min_lr]
```

Key changes from v2:
- Epochs: 10 → 3 (prevent overfitting)
- LR: 5e-5 → 2e-4 (standard LoRA LR, compensated by fewer epochs)
- Rank: 16 → 8 (sufficient for structured output)
- mask_prompt: true → false (research shows unmasked is better for short outputs)
- batch_size: 2 → 4 (faster iteration)
- max_seq_len: 2048 → 1024 (verify data fits)
- Added cosine decay LR schedule

---

### Research Sources

- [Unsloth LoRA Hyperparameters Guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide)
- [Sebastian Raschka: Practical Tips for Finetuning LLMs Using LoRA](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms)
- [Raschka: Instruction Masking Research Insights](https://magazine.sebastianraschka.com/p/llm-research-insights-instruction)
- [Predibase LLM Distillation Playbook](https://github.com/predibase/llm_distillation_playbook)
- [Databricks: Efficient Fine-Tuning with LoRA Guide](https://www.databricks.com/blog/efficient-fine-tuning-lora-guide-llms)
- [mlx-lm LoRA Documentation](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md)
- [mlx-lm LR Scheduler Bug](https://github.com/ml-explore/mlx/issues/2617)
- [Latitude: Fine-Tuning LLMs on Imbalanced Data](https://latitude-blog.ghost.io/blog/fine-tuning-llms-on-imbalanced-data-best-practices/)
- [LoRA Dropout as Sparsity Regularizer](https://arxiv.org/html/2404.09610v1)
- [GGUF Quantization Comparison (Q4_K_M vs Q8_0)](https://medium.com/@paul.ilvez/demystifying-llm-quantization-suffixes-what-q4-k-m-q8-0-and-q6-k-really-mean-0ec2770f17d3)
- [Qwen GGUF Docs](https://qwen.readthedocs.io/en/latest/quantization/llama.cpp.html)
- [DZone: Fine-Tuning LLMs Locally Using MLX LM](https://dzone.com/articles/fine-tuning-llms-locally-using-mlx-lm-guide)
- [MLX Cosine Decay Docs](https://ml-explore.github.io/mlx/build/html/python/optimizers/_autosummary/mlx.optimizers.cosine_decay.html)

## Model Artifacts Location

```
tools/training/models/
├── variable-scorer-qwen2.5/          # v1 (mode collapsed, deprecated)
│   ├── adapters/
│   └── merged/
└── variable-scorer-qwen2.5-v2/       # v2 (current, in training)
    ├── adapters/
    │   ├── adapters.safetensors
    │   ├── 0000500_adapters.safetensors
    │   └── ...
    └── merged/                        # Created after training completes

tools/training/exports/
├── Modelfile                          # Ollama config for Qwen2.5 (ChatML template)
├── magaldi-variable-scorer-qwen2.5-1.5b-f16.gguf   # v1 (deprecated)
└── magaldi-variable-scorer-qwen2.5-1.5b-q8_0.gguf  # v1 (deprecated)
```

## Dependencies

- `mlx-lm >= 0.22.0` — MLX training/inference
- `gguf` — GGUF conversion helper
- `llama.cpp` — GGUF conversion and quantization tools
- `ollama` — Model serving
- `pyyaml` — Config loading
- `requests` — Ollama API calls in evaluation script
