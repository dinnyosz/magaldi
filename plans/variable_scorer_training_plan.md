# Phase 4 Variable Scorer — Custom Model Training Pipeline

## Context

Replace the general-purpose `qwen3.5:4b` with a tiny fine-tuned `Qwen3-0.6B` LoRA model for Magaldi's Phase 4 (variable scoring). The training data is generated locally by a teacher model (`qwen3-coder:30b-a3b` via Ollama). This is the PoC — if it works, the same pipeline is reused for Phase 5 (summarizer), Phase 7 (features), and Phase 8 (glossary).

**Hardware**: M4 Pro 48GB — all training and inference happens locally.
**Training method**: LoRA via MLX (primary) or QLoRA via Unsloth (GPU alternative). Base model weights stay frozen, only adapter matrices are trained (~50MB), then merged back into base for export.

## File Structure to Create

```
tools/training/
├── README.md                              # Setup guide + end-to-end walkthrough
├── requirements.txt                       # Training-only deps (mlx-lm, transformers, etc.)
├── configs/
│   └── variable_scorer.yaml               # Hyperparams + data gen config
├── generate_scoring_data.py               # Script 1: Teacher → JSONL
├── train_variable_scorer.py               # Script 2: LoRA fine-tune
├── export_to_ollama.py                    # Script 3: GGUF + ollama create
├── evaluate_scorer.py                     # Script 4: Benchmark comparison
└── data/                                  # gitignored
    └── variable_scorer/
        ├── raw/                           # Per-variable teacher results (for resume)
        ├── train.jsonl                    # Training set (ChatML format)
        └── validation.jsonl               # Held-out validation (ChatML format)
```

Also add to `.gitignore`: `tools/training/data/`, `tools/training/models/`, `tools/training/exports/`

---

## Script 1: `generate_scoring_data.py`

**Purpose**: Extract variables from parsed repos, score them with the teacher model, quality-filter, output ChatML JSONL.

**CLI**:
```bash
python tools/training/generate_scoring_data.py \
  --source opensearch \
  --scope test-repo \
  --repos click requests zod \
  --teacher-model qwen3-coder:30b-a3b \
  --ollama-url http://localhost:11434 \
  --mode individual \
  --output-dir tools/training/data/variable_scorer \
  --validation-split 0.1 \
  --cache \
  --seed 42
```

**Data sources** (two options):
1. **OpenSearch** (primary): Fetch variables from already-parsed repos using `Repository.get_all_elements(scope, repo, "main", element_types=["variable", "constant"])`
2. **Direct parse** (fallback, no DB needed): Run Phases 1-3 programmatically on a repo path using `InMemoryFileStateRepository`

**Scoring modes**:
- `individual`: 1 variable per teacher call (higher quality, ~1s each)
- `batch`: Multiple variables per call using `_build_batches()` (faster, ~5-10x)

**Quality filtering** (all gates must pass):
1. Format: `_parse_scores()` returns non-None
2. Range: all 4 dimensions in [1, 10]
3. Heuristic agreement: `should_drop_variable()` drops should have max_score < 5
4. Variance: reject batches where all variables got identical scores

**Cache**: Each result saved to `raw/{hash}.json`. On `--cache`, skip already-scored element_ids. Final JSONL assembled from raw/ with filters re-applied (allows tuning thresholds without re-scoring).

**Output format** (ChatML, compatible with both Unsloth and mlx-lm):
```json
{"conversations": [
  {"role": "system", "content": "<SYSTEM_PROMPT from prompts.py>"},
  {"role": "user", "content": "Score these variables:\n1. [src/config.py] MAX_RETRIES = 3"},
  {"role": "assistant", "content": "1. 9,1,1,8"}
]}
```

**Key reuse**: Import `SYSTEM_PROMPT`, `build_user_prompt` from `magaldi_core.variable_scoring.prompts`, `_parse_scores` from `magaldi_core.variable_scoring`, `should_drop_variable` from `magaldi_core.variable_scoring.heuristic_filter`, `LLMClient.from_ollama()` from `shared.ai.llm_client`.

**Expected volume**: 20-50K variables from test repos → ~5K clean examples after filtering. At ~1s/var individual mode, generation takes a few hours.

---

## Script 2: `train_variable_scorer.py`

**Purpose**: LoRA fine-tune Qwen3-0.6B on the generated data.

**CLI**:
```bash
python tools/training/train_variable_scorer.py \
  --train-data tools/training/data/variable_scorer/train.jsonl \
  --val-data tools/training/data/variable_scorer/validation.jsonl \
  --base-model Qwen/Qwen3-0.6B-Instruct \
  --output-dir tools/training/models/variable-scorer-v1 \
  --config tools/training/configs/variable_scorer.yaml \
  --backend mlx \
  --epochs 3
```

**Backends**:
- `mlx` (primary): Uses `mlx-lm` library, native on M4 Pro. ~30-60 min for 5K examples × 3 epochs.
- `unsloth` (alternative): For NVIDIA GPU users. Uses QLoRA (4-bit base + LoRA adapters). Documented but not the default path.

**Hyperparameters** (in `configs/variable_scorer.yaml`):
- LoRA rank: 16, alpha: 32, dropout: 0.05
- Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- LR: 2e-4, cosine schedule, warmup: 3%
- Batch size: 4, grad accum: 4, max_seq_len: 2048

**Output**: Merged model in HuggingFace Safetensors format at `models/variable-scorer-v1/`

---

## Script 3: `export_to_ollama.py`

**Purpose**: Convert merged model → GGUF → quantize → register with Ollama.

**CLI**:
```bash
python tools/training/export_to_ollama.py \
  --model-dir tools/training/models/variable-scorer-v1 \
  --output-dir tools/training/exports/ \
  --quantization Q4_K_M \
  --ollama-name magaldi-scorer \
  --llama-cpp-path ~/llama.cpp \
  --test
```

**Steps**:
1. Convert HF → GGUF (via llama.cpp `convert_hf_to_gguf.py` or mlx-lm native GGUF export)
2. Quantize to Q4_K_M (~400MB final size)
3. Generate Modelfile (temperature=0.1, num_ctx=2048, stop=`<|im_end|>`)
4. `ollama create magaldi-scorer -f Modelfile`
5. Verification test: score the same 20 test variables from `compare_scoring_models.py`, check format + keep/drop accuracy

---

## Script 4: `evaluate_scorer.py`

**Purpose**: Compare fine-tuned model vs current production model using teacher labels as ground truth.

**CLI**:
```bash
python tools/training/evaluate_scorer.py \
  --val-data tools/training/data/variable_scorer/validation.jsonl \
  --models magaldi-scorer qwen3.5:4b \
  --ollama-url http://localhost:11434 \
  --threshold 5
```

**Metrics** (priority order):
1. **False drop rate** (critical): Model drops what teacher keeps → permanent data loss. Target: ≤2%
2. **Format accuracy**: Output parses as `N. c,a,d,g`. Target: ≥98%
3. **Keep/drop agreement**: Binary decision matches teacher. Target: ≥90%
4. **Score MAE**: Mean absolute error across 4 dimensions
5. **Throughput**: tok/s — expect 5-10x faster than qwen3.5:4b

**Reuses**: Pattern from `tools/compare_scoring_models.py` (multi-model comparison table), `_parse_scores()` for parsing, `SYSTEM_PROMPT` for prompts.

---

## Config: `configs/variable_scorer.yaml`

```yaml
training:
  backend: mlx
  base_model: Qwen/Qwen3-0.6B-Instruct
  lora_rank: 16
  lora_alpha: 32
  lora_dropout: 0.05
  lora_target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
  learning_rate: 2.0e-4
  weight_decay: 0.01
  warmup_ratio: 0.03
  epochs: 3
  batch_size: 4
  gradient_accumulation_steps: 4
  max_seq_len: 2048
  seed: 42

data_generation:
  teacher_model: qwen3-coder:30b-a3b
  ollama_url: http://localhost:11434
  mode: individual
  validation_split: 0.1

export:
  quantization: Q4_K_M
  ollama_model_name: magaldi-scorer
```

## `requirements.txt`

```
# Training-only deps (install in dedicated venv or alongside magaldi)
mlx>=0.22.0
mlx-lm>=0.22.0
transformers>=4.46.0
datasets>=3.0.0
tokenizers>=0.20.0
safetensors>=0.4.0
pyyaml>=6.0
rich>=13.0.0
litellm>=1.0.0
# Unsloth (NVIDIA GPU alternative — see README)
```

---

## Integration with Magaldi (after training)

One config change in `magaldi.yaml`:
```yaml
models:
  variable_scoring:
    name: magaldi-scorer
    provider: ollama
    url: http://localhost:11434
```

Zero code changes to Magaldi. The model runs through the existing `LLMClient.generate_from_messages()` path.

---

## Success Criteria

1. Format accuracy ≥ 98%
2. Keep/drop agreement ≥ 90% vs teacher
3. False drop rate ≤ 2%
4. Throughput ≥ 3x faster than qwen3.5:4b
5. Model size ≤ 0.5GB (GGUF Q4_K_M)
6. Integrates with `magaldi parse` via config change only

---

## Implementation Order

1. Directory structure + config + requirements + .gitignore entries
2. `generate_scoring_data.py` (most complex — implement first, test with 1 small repo)
3. `train_variable_scorer.py` (MLX backend primary)
4. `export_to_ollama.py` (GGUF conversion + registration)
5. `evaluate_scorer.py` (benchmark comparison)
6. `README.md` (end-to-end guide)
7. End-to-end test: generate → train → export → evaluate → `magaldi parse`

---

## Key Files to Reuse

- `src/magaldi_core/variable_scoring/prompts.py` — SYSTEM_PROMPT, build_user_prompt()
- `src/magaldi_core/variable_scoring/__init__.py` — _parse_scores(), _build_batches()
- `src/magaldi_core/variable_scoring/heuristic_filter.py` — should_drop_variable()
- `src/magaldi_core/variable_scoring/models.py` — VariableScore, VariableScoringConfig
- `src/shared/ai/llm_client.py` — LLMClient.from_ollama()
- `tools/compare_scoring_models.py` — TEST_VARIABLES, evaluation pattern
