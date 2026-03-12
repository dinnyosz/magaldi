---
name: train-variable-scorer
description: >
  Orchestrate the full variable scorer training pipeline: data generation,
  LoRA fine-tuning, GGUF export, Ollama registration, and evaluation.
  Each step runs in a background subagent with adaptive progress reporting.
---

# Train Variable Scorer Pipeline

End-to-end orchestrator for training/retraining the variable scorer model.
Runs each pipeline step sequentially in background subagents, monitors progress,
and reports back to the user with ETAs.

## When to Use

Invoked by `/train-variable-scorer` or when the user asks to "train the scorer",
"retrain the variable scorer", "run the training pipeline", etc.

## Prerequisites

- Ollama running locally (`http://localhost:11434`)
- Teacher model pulled (default: `qwen3-coder:30b`)
- Test repos cloned in `test_repos/` (with `magaldi.yaml` in each)
- `mlx-lm >= 0.22.0` installed in venv
- `llama.cpp` built locally (for GGUF conversion)

## Pipeline Steps

| Step | Script | Expected Duration | Report Frequency |
|------|--------|-------------------|------------------|
| 1. Generate data | `generate_scoring_data.py` | 30-90 min (depends on sample size) | Every 5 min |
| 2. Train model | `train_variable_scorer.py` | 30-120 min (depends on epochs/data) | Every 10 min |
| 3. Export to GGUF + register | `export_to_ollama.py` | 2-5 min | Once when done |
| 4. Evaluate | `evaluate_scorer.py` | 2-10 min | Once when done |

## Orchestrator Workflow

### 1. Auto-detect configs and ask user

Before starting, discover available training configs and let the user choose:

**Step 1a: Scan for configs**

List all YAML files matching `tools/training/configs/variable_scorer*.yaml`.
For each config, read the file and extract a summary:
- `training.base_model` — the model being fine-tuned
- `training.epochs` — number of epochs
- `training.lora_rank` — LoRA rank
- `export.ollama_model_name` — what it'll be registered as

**Step 1b: Ask the user which config to use**

Use `AskUserQuestion` to present the available configs. Format each option with
the key details so the user can make an informed choice. Example:

```
AskUserQuestion({
  questions: [{
    question: "Which training config should I use?",
    header: "Config",
    options: [
      {
        label: "variable_scorer.yaml (Recommended)",
        description: "Qwen2.5-1.5B, 3 epochs, rank 8, Q4_K_M → magaldi-scorer-1.7b"
      },
      {
        label: "variable_scorer_0.5b.yaml",
        description: "Qwen2.5-0.5B, 3 epochs, rank 8, Q4_K_M → magaldi-scorer-0.5b"
      }
    ],
    multiSelect: false
  }]
})
```

If only one config exists, skip the question and use it directly (but still
inform the user which config was selected).

**Step 1c: Set parameters**

After the user picks a config, read it and populate all parameters:

```
repos_dir        = "test_repos"
config_path      = (user's chosen config path)
output_model_dir = "tools/training/models/variable-scorer-v3"
ollama_name      = (from config: export.ollama_model_name)
quantization     = (from config: export.quantization)
llama_cpp_path   = "tools/llama.cpp"
eval_limit       = 50
# These are read from config, but can be overridden by CLI flags:
# sample_size    = (from config: data_generation.sample_size, default 10000)
# teacher_model  = (from config: data_generation.teacher_model)
# Cache is ON by default (no flag needed). Use --no-cache to disable.
```

If the user specified overrides in their message (e.g. "train with 5000 samples"),
apply those on top of the config values.

### 2. Initialize tracking

```
TodoWrite([
  {content: "Step 1: Generate training data", status: "pending"},
  {content: "Step 2: Train model with LoRA", status: "pending"},
  {content: "Step 3: Export to GGUF + register with Ollama", status: "pending"},
  {content: "Step 4: Evaluate model", status: "pending"},
])
```

### 3. Execute steps sequentially

For each step:

1. Mark step `in_progress` in TodoWrite
2. Post a short Slack message: what's starting, expected duration
3. Launch the command in a **background Bash** task
4. Monitor with adaptive polling (see Reporting Strategy below)
5. When done: post result summary, mark `completed`
6. If failed: post error, stop pipeline, ask user how to proceed

### 4. Reporting Strategy — Adaptive Frequency

The key insight: don't spam the user with updates for long-running steps, but don't leave them in the dark either.

**Algorithm:**

```
estimated_duration = step's expected duration (from table above)

if estimated_duration <= 5 min:
    report_interval = 60s       # every minute
    check_interval  = 15s       # poll output every 15s
elif estimated_duration <= 30 min:
    report_interval = 5 min     # every 5 minutes
    check_interval  = 30s
else:
    report_interval = 10 min    # every 10 minutes
    check_interval  = 60s
```

**What to report:**

Each progress update should be ONE short Slack message (2-3 lines max):
- Current status (e.g., "Scored 1,247/3,000 variables")
- Rate (e.g., "2.1 vars/sec")
- ETA (e.g., "~14 min remaining")

Parse progress from the command's stdout/stderr:
- Step 1: Look for `[N/TOTAL]` progress lines, extract scored count
- Step 2: Look for `Iter N:` lines from mlx-lm, extract iteration and val loss
- Step 3: Look for conversion/registration log lines
- Step 4: Look for `[N/TOTAL]` evaluation progress

**ETA calculation:**

```
elapsed = time.now() - step_start
items_done = parsed from output
items_total = parsed from output or known from config
rate = items_done / elapsed
remaining = (items_total - items_done) / rate
```

### 5. Final summary

After all steps complete, post a summary:

```
Training pipeline complete!

*Data*: 10,000 variables scored, 9,000 train / 1,000 val examples
*Training*: 3 epochs, final val loss: 0.072
*Model*: magaldi-scorer registered in Ollama (Q4_K_M, 892 MB)
*Evaluation*:
  Format accuracy: 99.2%
  Keep/drop accuracy: 93.1%
  Per-score accuracy: 61.4%
  MAE: 1.12

Total time: 1h 47m
```

---

## Step Commands

### Step 1: Generate Training Data

```bash
.venv/bin/python tools/training/generate_scoring_data.py \
  --repos-dir {repos_dir} \
  --config {config_path} \
  --output-dir tools/training/data/variable_scorer \
  --seed 42 \
  -v 2>&1
```

Note: `--config` loads `sample_size`, `teacher_model`, `max_trivial_ratio`, etc. from the
config YAML's `data_generation` section. CLI flags override config values.

Cache is ON by default — previously scored results in `raw/` are reused automatically.
If raw/ already has >= sample_size results, repo parsing and teacher scoring are skipped
entirely (fast path). Use `--force-rescore` to re-parse repos and score new variables,
or `--no-cache` to start completely fresh.

**Progress parsing**: Each scored variable outputs a progress line like:
`[  123/10000] KEEP 9,1,1,8,2,8,9  MAX_RETRIES ...`

Extract current/total from `[N/TOTAL]` pattern. If the early-exit path triggers,
look for `"Raw data sufficient"` message instead — scoring was skipped.

**Completion**: Look for the `Output` section with `Train:` and `Validation:` lines.

### Step 2: Train Model

```bash
.venv/bin/python tools/training/train_variable_scorer.py \
  --train-data tools/training/data/variable_scorer/train.jsonl \
  --val-data tools/training/data/variable_scorer/validation.jsonl \
  --output-dir {output_model_dir} \
  --config {config_path} \
  --backend mlx \
  -v 2>&1
```

**Progress parsing**: mlx-lm outputs lines like:
`Iter 100: Train loss 0.234, Learning Rate 1.5e-04, It/sec 0.18, Tokens/sec 180`
`Iter 200: Val loss 0.156, Val took 12.3s`

Extract iteration number, total iterations (from config: epochs * steps_per_epoch),
and val loss.

**Completion**: Look for "Merged model saved to" or "LoRA adapters saved to".

### Step 3: Export to GGUF + Register

```bash
.venv/bin/python tools/training/export_to_ollama.py \
  --model-dir {output_model_dir}/merged \
  --output-dir tools/training/exports/ \
  --quantization {quantization} \
  --ollama-name {ollama_name} \
  --llama-cpp-path {llama_cpp_path} \
  --test \
  -v 2>&1
```

**Progress parsing**: This is short — just report when each sub-step completes
(converting, quantizing, registering, smoke test).

**Completion**: Look for "Export complete!".

### Step 4: Evaluate

```bash
.venv/bin/python tools/training/evaluate_scorer.py \
  --model {ollama_name} \
  --val-data tools/training/data/variable_scorer/validation.jsonl \
  --limit {eval_limit} \
  -v 2>&1
```

**Progress parsing**: Look for `[N/TOTAL]` evaluation lines.

**Completion**: Look for the results table with "Variable Scorer Evaluation" header.
Parse the key metrics (format accuracy, keep/drop accuracy, per-score accuracy, MAE)
from the output.

---

## Error Handling

- If any step fails (non-zero exit code), stop the pipeline immediately
- Post the last 20 lines of output to help diagnose
- Ask the user: "Step N failed. Want me to show the full output, retry, or skip?"
- For Step 1 with `--cache`: suggest re-running (cache reuses already-scored variables)
- For Step 2: suggest checking GPU/memory and retrying with smaller batch size

## Configuration Reference

The skill auto-detects all `tools/training/configs/variable_scorer*.yaml` files
and asks the user to pick one before starting. Key sections in each config:

- `training.base_model` — HuggingFace model name
- `training.epochs` — Number of training epochs
- `training.max_seq_len` — Token budget for batch packing
- `data_generation.teacher_model` — Ollama model for scoring
- `data_generation.sample_size` — Number of variables to sample (default 10000)
- `data_generation.max_trivial_ratio` — Rebalancing threshold
- `export.quantization` — GGUF quantization type
- `export.ollama_model_name` — Name to register in Ollama

## Subagent Guidance

This skill does NOT use Task subagents — it uses **background Bash** commands directly.
The orchestrator (the main Claude session) monitors the background tasks using
`Read` on the task output file or `tail` via Bash, and posts progress updates to Slack.

This is simpler and more reliable than nested subagents because:
1. The scripts already produce structured progress output
2. Background Bash tasks can run for hours without timeout
3. The orchestrator maintains full context of all steps

## Notes

- Always use `.venv/bin/python` (not `python`) — the venv has all dependencies
- The project root is `/Users/dinnyosz/code/magaldi`
- llama.cpp is at `tools/llama.cpp` — verify with `ls tools/llama.cpp/convert_hf_to_gguf.py`
- Training on Apple Silicon (M-series) via MLX — no GPU needed
- The `--cache` flag on Step 1 is safe to always use — it reuses already-scored variables from `raw/`
