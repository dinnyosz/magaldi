# Variable Scorer Training Pipeline

Fine-tune a tiny Qwen3-0.6B model to replace the general-purpose `qwen3.5:4b` for Magaldi's Phase 4 variable scoring. The fine-tuned model runs ~4x faster at ~1/6th the size while matching quality on this narrow task.

## Overview

```
generate_scoring_data.py   Parse repos → score variables with teacher model → ChatML JSONL
train_variable_scorer.py   Fine-tune Qwen3-0.6B with LoRA (MLX or Unsloth)
export_to_ollama.py        Convert to GGUF → register with Ollama
evaluate_scorer.py         Benchmark fine-tuned vs production model
```

## Quick Start

### 1. Generate Training Data

Requires: repos with `magaldi.yaml`, teacher model running in Ollama.

```bash
# Pull teacher model
ollama pull qwen3-coder:30b

# Generate scored training data from repos
python tools/training/generate_scoring_data.py \
  --repos /path/to/repo1 /path/to/repo2 \
  --teacher-model qwen3-coder:30b \
  --output-dir tools/training/data/variable_scorer \
  --cache
```

The script:
- Parses repos directly (Phases 1-3) to extract variables
- Scores each variable with the teacher model on 7 dimensions
- Applies heuristic pre-filtering (obvious junk gets all-1s without calling the teacher)
- Quality-filters results (range check, heuristic agreement, constant sanity)
- Outputs ChatML JSONL with varied batch sizes for training

Use `--cache` to reuse previously scored results from `raw/` (skips re-scoring known variables).

### 2. Train

```bash
# Install training dependencies
pip install -r tools/training/requirements.txt

# Train with MLX (Apple Silicon)
python tools/training/train_variable_scorer.py \
  --train-data tools/training/data/variable_scorer/train.jsonl \
  --val-data tools/training/data/variable_scorer/validation.jsonl \
  --output-dir tools/training/models/variable-scorer-v1 \
  --backend mlx

# Train with Unsloth (NVIDIA GPU)
python tools/training/train_variable_scorer.py \
  --train-data tools/training/data/variable_scorer/train.jsonl \
  --val-data tools/training/data/variable_scorer/validation.jsonl \
  --output-dir tools/training/models/variable-scorer-v1 \
  --backend unsloth
```

### 3. Export to Ollama

```bash
python tools/training/export_to_ollama.py \
  --model-dir tools/training/models/variable-scorer-v1/merged \
  --llama-cpp-path ~/llama.cpp \
  --ollama-name magaldi-scorer \
  --test
```

### 4. Evaluate

```bash
python tools/training/evaluate_scorer.py \
  --val-data tools/training/data/variable_scorer/validation.jsonl \
  --models magaldi-scorer qwen3.5:4b
```

Success criteria:
- Format accuracy >= 98%
- Keep/drop accuracy >= 90%
- False drop rate <= 2%

## Scoring Dimensions

The training data uses 7 scoring dimensions (expanded from production's 4):

| Dimension | What it measures |
|-----------|-----------------|
| `config_value` | Configuration, feature flags, URLs, prompts, SQL strings |
| `architectural_role` | Infrastructure: DB connections, routers, loggers, middleware |
| `data_definition` | Schemas, type aliases, enums, named tuples, protocols |
| `general_usefulness` | Would a coding agent benefit from finding this? |
| `value_complexity` | Simple literal (1) → complex expression/template (9) |
| `naming_quality` | Single letter (1) → fully descriptive name (9) |
| `scope_significance` | Loop/temp (1) → function local (2) → class attr (6) → module-level (9) |

## Directory Structure

```
tools/training/
├── configs/
│   └── variable_scorer.yaml    # Hyperparameters and config
├── data/
│   └── variable_scorer/        # Generated training data (gitignored)
│       ├── raw/                # Per-variable JSON (resume cache)
│       ├── train.jsonl         # Training examples
│       └── validation.jsonl    # Held-out validation set
├── models/                     # Trained models (gitignored)
├── exports/                    # GGUF files (gitignored)
├── generate_scoring_data.py
├── train_variable_scorer.py
├── export_to_ollama.py
├── evaluate_scorer.py
├── requirements.txt
└── README.md
```

## Config

Edit `configs/variable_scorer.yaml` to adjust:

- **training**: base model, LoRA rank/alpha, learning rate, epochs, batch size
- **data_generation**: teacher model, Ollama URL, temperature
- **export**: quantization type, Ollama model name
