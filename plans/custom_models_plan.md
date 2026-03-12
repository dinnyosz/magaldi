# Custom Purpose-Built Models for Magaldi

> **Goal**: Replace general-purpose LLMs (currently `qwen3.5:4b`) with tiny, purpose-built fine-tuned models for each Magaldi phase, trained on synthetically generated data.

## Big Picture: Which Models We Need

| Model | Phase | Task Type | Base Model | Training Examples |
|-------|-------|-----------|------------|-------------------|
| `magaldi-variable-scorer` | 4 | Classification (4 scores) | Qwen3-0.6B | 5,000 |
| `magaldi-summarizer` | 5 | Text generation (code→summary) | Qwen3-1.7B | 15,000 |
| `magaldi-feature-namer` | 7 | Text generation (summaries→feature summary+label) | Qwen3-0.6B | 3,000 |
| `magaldi-glossary-extractor` | 8a | Extraction (text→JSON terms) | Qwen3-0.6B | 2,000 |
| `magaldi-glossary-writer` | 8b | Text generation (term+context→definition) | Qwen3-0.6B | 2,000 |

**Phase 6 (Call Resolution)**: No model needed — uses pre-computed embeddings only.
**Embedding model**: Keep `qwen3-embedding:0.6b` — contrastive training is different, current model works well.

We build each model separately, starting with Phase 4 as the proof-of-concept.

---

# Phase 4: `magaldi-variable-scorer` — Detailed Plan

## 1. What This Model Does

Score variables/constants on 4 dimensions (1-10 each) to decide if they belong in the search index. Only ~30% of variables should survive. This is essentially a *structured classification* task — the most predictable and easiest to validate of all our phases.

### Current Flow
```
Variables → Heuristic pre-filter (drops obvious junk: single-letter, throwaway names)
         → LLM scoring (batched, 4 scores per variable)
         → Threshold check (max score ≥ 5 → keep)
```

### Current Prompt Structure

**System message** (~660 tokens, static):
- Defines 4 scoring dimensions: `config_value`, `architectural_role`, `data_definition`, `general_usefulness`
- Includes examples of high/low scoring variables
- Rules: "most variables should score LOW", "only ~30% worth keeping"

**User message** (variable, per batch):
```
/no_think
Score these variables:
1. [src/config.py] MAX_RETRIES = 3
2. [src/utils.py] _temp = []
3. [src/db.py] engine = create_engine(DATABASE_URL)
```

**Expected output**:
```
1. 9,1,1,8
2. 1,1,1,1
3. 1,9,1,9
```

### Current Config
- Model: `qwen3.5:4b` (general-purpose, way overpowered for this task)
- Temperature: 0.1
- Token budget per batch: ~1200 content tokens
- Output budget: `batch_size * 20 + 50` tokens
- Parallelism: up to 12 workers with runtime-aware throttling

---

## 2. What Will Run the Model

### Training

| Option | Framework | Hardware | Time (0.6B) | Cost |
|--------|-----------|----------|-------------|------|
| **Recommended** | [Unsloth](https://unsloth.ai) + QLoRA | Any GPU ≥8GB VRAM | ~15-30 min | $0 |
| Mac option | [MLX](https://mlx-framework.org) + LoRA | Apple Silicon M1+ | ~30-60 min | $0 |
| Cloud option | Unsloth on [Google Colab](https://colab.research.google.com) (free T4) | T4 16GB | ~20-40 min | $0 |

**QLoRA** (4-bit quantized LoRA): Freezes the base model weights, trains a small set of low-rank adapter matrices in 16-bit on top of the 4-bit quantized base. This means:
- 0.6B model uses ~0.5GB VRAM for the base + ~50MB for LoRA adapters
- Fits easily on any modern GPU or Apple Silicon Mac

**Training hyperparameters** (starting point):
```yaml
base_model: Qwen/Qwen3-0.6B
method: qlora
lora_rank: 16
lora_alpha: 32
lora_target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
learning_rate: 2e-4
epochs: 3
batch_size: 4
gradient_accumulation: 4
max_seq_length: 2048
optimizer: adamw_8bit
warmup_ratio: 0.05
weight_decay: 0.01
```

### Inference (same as today — zero changes to Magaldi)

```
Fine-tuned model (HF Safetensors)
    → merge LoRA adapters into base
    → convert to GGUF via llama.cpp (convert_hf_to_gguf.py)
    → quantize to Q4_K_M (best size/quality trade-off)
    → ollama create magaldi-variable-scorer -f Modelfile
    → change model name in magaldi config
```

The model runs through the exact same `LLMClient.generate_from_messages()` path. Only the model name changes from `qwen3.5:4b` to `magaldi-variable-scorer`.

**GGUF size**: ~0.4GB (Q4_K_M quantized 0.6B model)
**VRAM usage**: ~0.5GB at runtime
**Speedup expected**: 5-8x faster than qwen3.5:4b (smaller model = faster inference)

---

## 3. How Much Training Data

### Target: 5,000 examples

Research backing:
- ~1,000 high-quality examples are sufficient for narrow instruction tuning
- Qwen3-0.6B shows the *largest gains* from fine-tuning among small models
- 5,000 gives us a 4,500/500 train/validation split with margin for quality filtering

### Distribution strategy

We want the training data to reflect what the model will see in production:

| Category | Count | Description |
|----------|-------|-------------|
| Clear DROP (score 1,1,1,1) | 2,000 (40%) | Throwaway locals, loop vars, generic results |
| Clear KEEP (max score ≥ 7) | 1,500 (30%) | Constants, configs, framework instances, schemas |
| Borderline (max score 4-6) | 1,000 (20%) | Variables where the scoring is nuanced |
| Edge cases | 500 (10%) | Very long code, unusual patterns, multi-language quirks |

This 40/30/20/10 split mirrors real-world distribution where most variables are junk.

### Language distribution

| Language | % of examples | Repos |
|----------|---------------|-------|
| Python | 25% | FastAPI, Django, httpie, pandas |
| JavaScript | 20% | Express, lodash, Next.js |
| TypeScript | 20% | Angular, Prisma, tRPC |
| Rust | 12% | ripgrep, tokio, actix-web |
| Java | 12% | Spring Boot, Guava |
| PHP | 8% | Laravel, Symfony |
| Bash | 3% | Various scripts |

---

## 4. Synthetic Data Generation Pipeline

### Step 1: Collect Variables from Real Codebases

Use Magaldi's existing parser (Phases 1-3) to parse 20-30 repos and extract all variables:

```bash
# Parse repos (already works today)
for repo in repos/*.yaml; do
    magaldi parse /path/to/repo --user training --phase 3
done
```

This gives us raw tuples of `(element_id, file_path, name, raw_code)` for every variable/constant in those repos.

**Expected yield**: 20 repos × ~2,500 variables each = ~50,000 raw variables

### Step 2: Sample & Stratify

From the 50,000 raw variables:

1. Run through `heuristic_filter.py` — this gives us 3 buckets:
   - **Heuristic-dropped** (~40%): Known junk. Sample 500 for training (model should confirm these are 1,1,1,1)
   - **Heuristic-kept** (~60%): Need LLM scoring. This is the main training source.

2. Stratify the heuristic-kept variables by:
   - Language (proportional to target distribution)
   - Code pattern (assignments, function calls, data structures, decorators, etc.)
   - Name pattern (UPPER_CASE, camelCase, snake_case, short, long)

3. Select ~6,000 candidates (buffer for quality filtering → keep 5,000)

### Step 3: Generate Teacher Labels with Claude

Each variable scored individually (not batched) by Claude Sonnet for maximum quality:

```python
import anthropic

client = anthropic.Anthropic()

# For each variable, build the exact prompt the student model will see
for batch in training_batches:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},  # Same as production
        {"role": "user", "content": build_user_prompt(batch)},  # Same format
    ]

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        temperature=0.2,
        messages=[
            {"role": "user", "content": f"{SYSTEM_PROMPT}\n\n{build_user_prompt(batch)}"}
        ]
    )

    # Parse and validate
    scores = parse_scores(response.content[0].text, len(batch))
    # ... quality checks ...
```

**BUT**: For better quality, we also score variables *individually* (1 per call) with more context:

```python
ENHANCED_SCORING_PROMPT = """
Score this single variable on four dimensions (1-10).
Consider the file context and what a coding agent would gain from finding it.

Variable: [{file_path}] {raw_code}

Think about:
- Is this a configuration value, URL, path, prompt template? (config_value)
- Is this infrastructure? DB, router, logger, app instance? (architectural_role)
- Is this a data structure, schema, type alias, enum? (data_definition)
- Would an AI agent benefit from finding this? (general_usefulness)

Output format: config_value,architectural_role,data_definition,general_usefulness
Output ONLY the four numbers.
"""
```

Then we reformat these individual scores into the batched format for training.

### Step 4: Quality Filtering

Each example must pass ALL gates:

```python
def validate_training_example(batch_input, batch_output):
    """All gates must pass."""

    # Gate 1: Format — output matches N. score,score,score,score exactly
    scores = parse_scores(batch_output, batch_size)
    if any(s is None for s in scores):
        return False, "format_error"

    # Gate 2: Range — all scores 1-10
    for score in scores:
        if not all(1 <= v <= 10 for v in score.as_tuple()):
            return False, "out_of_range"

    # Gate 3: Distribution — not all same scores (lazy model)
    unique_tuples = set(s.as_tuple() for s in scores)
    if len(unique_tuples) == 1 and len(scores) > 3:
        return False, "uniform_scores"

    # Gate 4: Heuristic agreement — heuristic-dropped vars should score low
    for i, (_, _, name, raw_code) in enumerate(batch_input):
        should_drop, _ = heuristic_filter.should_drop_variable(name, raw_code)
        if should_drop and scores[i].max_score >= 7:
            return False, f"heuristic_disagreement_{name}"

    # Gate 5: UPPER_CASE constants should generally score high
    for i, (_, _, name, _) in enumerate(batch_input):
        if name.isupper() and len(name) > 3 and scores[i].max_score < 4:
            return False, f"constant_scored_low_{name}"

    return True, "ok"
```

### Step 5: Format for Training

Convert to Unsloth's expected ChatML / conversation format:

```jsonl
{"conversations": [{"role": "system", "content": "You are scoring variables for a coding agent's search index..."}, {"role": "user", "content": "Score these variables:\n1. [src/config.py] MAX_RETRIES = 3\n2. [src/utils.py] _temp = []\n3. [src/db.py] engine = create_engine(DATABASE_URL)"}, {"role": "assistant", "content": "1. 9,1,1,8\n2. 1,1,1,1\n3. 1,9,1,9"}]}
```

**Key**: The system prompt is IDENTICAL to production. The user prompt is built with `build_user_prompt()`. The assistant content is the teacher's validated output. No `/no_think` in training data (that's a runtime workaround for Qwen3 thinking mode).

### Batch size variation in training data

Mix batch sizes to teach the model flexibility:
- 10% of examples: 1-3 variables (small batch)
- 40% of examples: 4-10 variables (medium batch)
- 40% of examples: 11-20 variables (large batch)
- 10% of examples: 21-30 variables (max batch)

---

## 5. Training Procedure

### Environment Setup

```bash
# Option A: Colab (recommended for first attempt)
# Just open the notebook — Unsloth provides official templates

# Option B: Local GPU
pip install unsloth
pip install --upgrade torch

# Option C: Mac (MLX)
pip install mlx-lm
```

### Unsloth Training Script (Pseudocode)

```python
from unsloth import FastLanguageModel

# 1. Load base model with 4-bit quantization
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen3-0.6B",
    max_seq_length=2048,
    load_in_4bit=True,
)

# 2. Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # LoRA rank
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
)

# 3. Load training data
from datasets import load_dataset
dataset = load_dataset("json", data_files="data/variable_scorer/train.jsonl")

# 4. Format for ChatML
def format_prompt(example):
    return tokenizer.apply_chat_template(
        example["conversations"],
        tokenize=False,
        add_generation_prompt=False,
    )

# 5. Train
from trl import SFTTrainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    max_seq_length=2048,
    args=TrainingArguments(
        output_dir="outputs/variable_scorer",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_ratio=0.05,
        weight_decay=0.01,
        logging_steps=10,
        save_strategy="epoch",
        fp16=True,
    ),
)
trainer.train()

# 6. Merge LoRA and save
model.save_pretrained_merged("models/variable_scorer_merged", tokenizer)
```

### Export to GGUF + Ollama

```bash
# Convert to GGUF
python llama.cpp/convert_hf_to_gguf.py models/variable_scorer_merged \
    --outtype q4_k_m \
    --outfile models/magaldi-variable-scorer-q4_k_m.gguf

# Create Ollama model
cat > Modelfile <<EOF
FROM models/magaldi-variable-scorer-q4_k_m.gguf

PARAMETER temperature 0.1
PARAMETER top_p 0.95
PARAMETER num_ctx 2048
PARAMETER stop "<|im_end|>"
EOF

ollama create magaldi-variable-scorer -f Modelfile

# Test
ollama run magaldi-variable-scorer "Score these variables:
1. [src/config.py] MAX_RETRIES = 3
2. [src/utils.py] _temp = []"
```

---

## 6. Evaluation & Benchmarking

### Validation Set

Hold out 500 examples (10%) never seen during training. Run both models on the same inputs.

### Metrics

| Metric | What it measures | Target |
|--------|-----------------|--------|
| **Format accuracy** | % of outputs that parse correctly as `N. s,s,s,s` | ≥ 98% |
| **Score agreement** | % of scores within ±1 of teacher labels | ≥ 85% |
| **Keep/drop agreement** | % of variables where keep/drop decision matches teacher | ≥ 90% |
| **Throughput** | Variables scored per second | ≥ 5x current |
| **False drop rate** | % of teacher-KEEPs that model DROPs | ≤ 5% |
| **False keep rate** | % of teacher-DROPs that model KEEPs | ≤ 15% (acceptable, LLM catches) |

### Benchmark Procedure

```python
# Run both models on same 500-example validation set
for example in validation_set:
    current_output = qwen35_4b.score(example.input)
    finetuned_output = magaldi_scorer.score(example.input)
    teacher_output = example.expected_output  # Claude's labels

    # Compare all three
    compare(current_output, finetuned_output, teacher_output)
```

### End-to-End Integration Test

Parse 2 held-out repos with both models, compare:
1. Total variables kept/dropped
2. Quality spot-check: are the right variables surviving?
3. Wall-clock time for Phase 4

---

## 7. Integration with Magaldi

### Config Change (the only production code change needed)

In `magaldi.yaml` or model config:
```yaml
models:
  variable_scoring:
    provider: ollama
    name: magaldi-variable-scorer
    # Everything else stays the same
```

The model routes through the existing `LLMClient` → `generate_from_messages()` path. The system prompt, user prompt format, output parsing — all unchanged.

### Potential optimization: Remove `/no_think` workaround

The fine-tuned model won't have thinking mode. We can remove the `/no_think` prefix from user prompts when using `magaldi-variable-scorer`, simplifying the prompt slightly. But this is optional — the model will learn to ignore it if present.

### Potential optimization: Smaller context window

Since the fine-tuned model is trained specifically on this task, it may not need the full 2048 context. We could test with 1024 — halving context = faster inference.

---

## 8. Cost & Timeline

### Data Generation Cost

- ~6,000 Claude Sonnet calls (5,000 kept after filtering)
- Per call: ~800 input tokens + ~100 output tokens (single variable) or ~2,000 + ~400 (batch)
- Estimated total: **~$5-10** in API costs

### Timeline

| Step | Duration | Effort |
|------|----------|--------|
| Set up repo list + parse 20 repos | 1 day | Automated (existing parser) |
| Generate + filter training data | 1-2 days | Script + validation |
| Train model (Unsloth/MLX) | 30 min | Automated |
| Export to GGUF + Ollama | 15 min | Scripted |
| Evaluate + benchmark | 1 day | Automated + manual spot-check |
| Integration test | 0.5 days | Run magaldi parse with new model |
| **Total** | **~4-5 days** | |

---

## 9. File Structure

```
tools/training/
├── README.md                          # Setup and usage guide
├── requirements.txt                   # unsloth, transformers, datasets, anthropic
├── repos.yaml                         # List of training repos + metadata
├── generate_variable_scorer_data.py   # Data generation for Phase 4
├── quality_filters.py                 # Validation gates
├── train_variable_scorer.py           # Unsloth QLoRA training
├── export_to_ollama.py                # GGUF conversion + ollama create
├── evaluate_variable_scorer.py        # Benchmark against current model
├── data/
│   └── variable_scorer/
│       ├── train.jsonl                # Training examples (4,500)
│       ├── validation.jsonl           # Held-out validation (500)
│       └── raw/                       # Pre-filtering intermediate data
└── configs/
    └── variable_scorer.yaml           # Training hyperparameters
```

---

## 10. Success Criteria

The PoC is successful if:

1. ✅ Format accuracy ≥ 98% (outputs parse correctly)
2. ✅ Keep/drop agreement ≥ 90% vs teacher labels
3. ✅ Throughput ≥ 3x faster than qwen3.5:4b
4. ✅ False drop rate ≤ 5% (we don't lose valuable variables)
5. ✅ Integrates with `magaldi parse` with only a config change
6. ✅ Model size ≤ 0.5GB (GGUF Q4_K_M)

If these criteria are met, we proceed to the summarizer (Phase 5) using the same pipeline.

---

## Research Sources

- [Unsloth: Fine-tuning Qwen3](https://unsloth.ai/docs/models/qwen3-how-to-run-and-fine-tune)
- [MLX: Local Fine-tuning on Apple Silicon](https://mlx-framework.org)
- [Fine-tuning LLMs with Limited Data (arxiv)](https://arxiv.org/abs/2411.09539)
- [Synthetic Data for LLMs (ACL 2025)](https://synth-data-acl.github.io/)
- [GGUF Import into Ollama](https://docs.ollama.com/import)
- [Unsloth → Ollama Export](https://unsloth.ai/docs/basics/inference-and-deployment/saving-to-ollama)
- [12 SLMs Benchmarked for Fine-tuning](https://www.distillabs.ai/blog/we-benchmarked-12-small-language-models-across-8-tasks-to-find-the-best-base-model-for-fine-tuning)
