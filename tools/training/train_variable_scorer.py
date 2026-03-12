#!/usr/bin/env python3
"""Fine-tune Qwen3-0.6B for variable scoring using LoRA.

Supports two backends:
- mlx: Native on Apple Silicon (M1+), uses mlx-lm library
- unsloth: For NVIDIA GPUs, uses QLoRA (4-bit base + LoRA adapters)

Usage:
    python tools/training/train_variable_scorer.py \
      --train-data tools/training/data/variable_scorer/train.jsonl \
      --val-data tools/training/data/variable_scorer/validation.jsonl \
      --output-dir tools/training/models/variable-scorer-v1 \
      --config tools/training/configs/variable_scorer.yaml \
      --backend mlx
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load training config from YAML."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def convert_jsonl_to_chatml(jsonl_path: Path, output_path: Path) -> int:
    """Convert our JSONL format to the format expected by mlx-lm.

    mlx-lm expects JSONL with a "messages" key (not "conversations").

    Returns number of examples converted.
    """
    count = 0
    with open(jsonl_path) as fin, open(output_path, "w") as fout:
        for line in fin:
            data = json.loads(line)
            # Rename "conversations" to "messages" for mlx-lm
            messages = data.get("conversations", data.get("messages", []))
            fout.write(json.dumps({"messages": messages}) + "\n")
            count += 1
    return count


def train_mlx(
    train_data: str,
    val_data: str,
    output_dir: str,
    config: dict,
) -> None:
    """Train using mlx-lm LoRA on Apple Silicon."""
    try:
        import mlx_lm  # noqa: F401
    except ImportError:
        logger.error(
            "mlx-lm not installed. Install with: pip install mlx-lm>=0.22.0"
        )
        sys.exit(1)

    training_cfg = config.get("training", {})
    base_model = training_cfg.get("base_model", "Qwen/Qwen3-0.6B-Instruct")
    epochs = training_cfg.get("epochs", 3)
    batch_size = training_cfg.get("batch_size", 4)
    grad_accum = training_cfg.get("gradient_accumulation_steps", 4)
    lr = training_cfg.get("learning_rate", 2e-4)
    lora_rank = training_cfg.get("lora_rank", 16)
    num_layers = training_cfg.get("num_layers", 16)
    seed = training_cfg.get("seed", 42)
    max_seq_len = training_cfg.get("max_seq_len", 2048)

    # Convert JSONL to mlx-lm format in a temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        train_mlx_path = tmpdir_path / "train.jsonl"
        val_mlx_path = tmpdir_path / "valid.jsonl"

        train_count = convert_jsonl_to_chatml(Path(train_data), train_mlx_path)
        val_count = convert_jsonl_to_chatml(Path(val_data), val_mlx_path)
        logger.info("Converted %d train, %d val examples", train_count, val_count)

        # Build mlx-lm config YAML (LoRA params + training params in one file)
        lora_alpha = training_cfg.get("lora_alpha", 32)
        iters = epochs * (train_count // (batch_size * grad_accum) + 1)
        steps_per_eval = max(1, iters // 5)  # eval ~5 times during training

        mask_prompt = training_cfg.get("mask_prompt", True)

        warmup_ratio = training_cfg.get("warmup_ratio", 0.03)
        warmup_iters = max(1, int(iters * warmup_ratio))

        mlx_config = {
            "model": base_model,
            "data": tmpdir,
            "train": True,
            "fine_tune_type": "lora",
            "optimizer": "adamw",
            "num_layers": num_layers,
            "batch_size": batch_size,
            "iters": iters,
            "val_batches": -1,  # use entire validation set
            "learning_rate": lr,
            "steps_per_report": 1,
            "steps_per_eval": steps_per_eval,
            "grad_accumulation_steps": grad_accum,
            "max_seq_length": max_seq_len,
            "mask_prompt": mask_prompt,
            "grad_checkpoint": True,
            "seed": seed,
            "lora_parameters": {
                "rank": lora_rank,
                "alpha": lora_alpha,
                "dropout": training_cfg.get("lora_dropout", 0.05),
                "scale": lora_alpha / lora_rank,
            },
        }

        # Add LR schedule if configured (e.g. cosine_decay)
        lr_schedule_cfg = training_cfg.get("lr_schedule")
        if lr_schedule_cfg:
            schedule = {
                "name": lr_schedule_cfg["name"],
                "warmup": warmup_iters,
                "arguments": [lr, iters, lr_schedule_cfg.get("min_lr", 1e-6)],
            }
            if "warmup_init" in lr_schedule_cfg:
                schedule["warmup_init"] = lr_schedule_cfg["warmup_init"]
            mlx_config["lr_schedule"] = schedule
            logger.info(
                "LR schedule: %s, warmup=%d iters, lr=%g -> %g",
                schedule["name"], warmup_iters, lr, schedule["arguments"][2],
            )

        if mask_prompt:
            logger.info("mask_prompt=True: loss computed only on assistant responses")
        else:
            logger.info("mask_prompt=False: loss includes instruction tokens")
        mlx_config_path = tmpdir_path / "mlx_config.yaml"
        with open(mlx_config_path, "w") as f:
            yaml.dump(mlx_config, f)

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Build mlx-lm lora command
        cmd = [
            sys.executable, "-m", "mlx_lm.lora",
            "--config", str(mlx_config_path),
            "--adapter-path", str(output_path / "adapters"),
        ]

        logger.info("Running mlx-lm LoRA training: %s", " ".join(cmd))
        result = subprocess.run(cmd, check=False)

        if result.returncode != 0:
            logger.error("mlx-lm training failed with exit code %d", result.returncode)
            sys.exit(1)

        logger.info("LoRA adapters saved to %s", output_path / "adapters")

        # Fuse adapters into base model
        fuse_cmd = [
            sys.executable, "-m", "mlx_lm.fuse",
            "--model", base_model,
            "--adapter-path", str(output_path / "adapters"),
            "--save-path", str(output_path / "merged"),
        ]

        logger.info("Fusing adapters: %s", " ".join(fuse_cmd))
        result = subprocess.run(fuse_cmd, check=False)

        if result.returncode != 0:
            logger.error("mlx-lm fuse failed with exit code %d", result.returncode)
            sys.exit(1)

        logger.info("Merged model saved to %s", output_path / "merged")


def train_unsloth(
    train_data: str,
    val_data: str,
    output_dir: str,
    config: dict,
) -> None:
    """Train using Unsloth QLoRA on NVIDIA GPU."""
    try:
        from unsloth import FastLanguageModel  # noqa: F401
    except ImportError:
        logger.error(
            "Unsloth not installed. Install with: pip install unsloth"
        )
        sys.exit(1)

    from datasets import load_dataset
    from trl import SFTTrainer
    from transformers import TrainingArguments

    training_cfg = config.get("training", {})
    base_model = training_cfg.get("base_model", "Qwen/Qwen3-0.6B-Instruct")
    epochs = training_cfg.get("epochs", 3)
    batch_size = training_cfg.get("batch_size", 4)
    grad_accum = training_cfg.get("gradient_accumulation_steps", 4)
    lr = training_cfg.get("learning_rate", 2e-4)
    lora_rank = training_cfg.get("lora_rank", 16)
    lora_alpha = training_cfg.get("lora_alpha", 32)
    lora_dropout = training_cfg.get("lora_dropout", 0.05)
    max_seq_len = training_cfg.get("max_seq_len", 2048)
    seed = training_cfg.get("seed", 42)
    target_modules = training_cfg.get("lora_target_modules", [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load base model with 4-bit quantization
    logger.info("Loading base model: %s", base_model)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=max_seq_len,
        load_in_4bit=True,
    )

    # Add LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
    )

    # Load training data
    dataset = load_dataset("json", data_files={
        "train": train_data,
        "validation": val_data,
    })

    # Format for ChatML
    def format_prompt(example: dict) -> dict:
        conversations = example.get("conversations", example.get("messages", []))
        text = tokenizer.apply_chat_template(
            conversations,
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    dataset = dataset.map(format_prompt)

    # Train
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        max_seq_length=max_seq_len,
        dataset_text_field="text",
        args=TrainingArguments(
            output_dir=str(output_path / "checkpoints"),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            learning_rate=lr,
            warmup_ratio=training_cfg.get("warmup_ratio", 0.03),
            weight_decay=training_cfg.get("weight_decay", 0.01),
            logging_steps=10,
            save_strategy="epoch",
            eval_strategy="epoch",
            fp16=True,
            seed=seed,
        ),
    )
    trainer.train()

    # Merge LoRA and save
    logger.info("Merging LoRA adapters into base model...")
    model.save_pretrained_merged(str(output_path / "merged"), tokenizer)
    logger.info("Merged model saved to %s", output_path / "merged")


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune Qwen3-0.6B for variable scoring."
    )

    parser.add_argument(
        "--train-data",
        required=True,
        help="Path to training JSONL file",
    )
    parser.add_argument(
        "--val-data",
        required=True,
        help="Path to validation JSONL file",
    )
    parser.add_argument(
        "--output-dir",
        default="tools/training/models/variable-scorer-v1",
        help="Output directory for trained model",
    )
    parser.add_argument(
        "--config",
        default="tools/training/configs/variable_scorer.yaml",
        help="Path to training config YAML",
    )
    parser.add_argument(
        "--backend",
        choices=["mlx", "unsloth"],
        default="mlx",
        help="Training backend (default: mlx)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Override number of training epochs",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    config = load_config(args.config)

    # CLI overrides
    if args.epochs:
        config.setdefault("training", {})["epochs"] = args.epochs

    if args.backend == "mlx":
        train_mlx(args.train_data, args.val_data, args.output_dir, config)
    elif args.backend == "unsloth":
        train_unsloth(args.train_data, args.val_data, args.output_dir, config)

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
