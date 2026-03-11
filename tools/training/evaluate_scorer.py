#!/usr/bin/env python3
"""Evaluate fine-tuned variable scorer against teacher labels.

Compares the fine-tuned model and the current production model using
the held-out validation set as ground truth.

Usage:
    python tools/training/evaluate_scorer.py \
      --val-data tools/training/data/variable_scorer/validation.jsonl \
      --models magaldi-scorer qwen3.5:4b \
      --ollama-url http://localhost:11434 \
      --threshold 5
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import litellm

# Add project root to path
_project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_project_root / "src"))

from magaldi_core.variable_scoring import _parse_scores
from magaldi_core.variable_scoring.prompts import SYSTEM_PROMPT

litellm.suppress_debug_info = True
logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass
class ModelMetrics:
    """Evaluation metrics for a single model."""

    model_name: str
    total_examples: int = 0
    total_variables: int = 0
    format_correct: int = 0  # Scores that parsed correctly
    format_total: int = 0  # Total expected scores
    score_agreement: int = 0  # Scores within ±1 of teacher
    score_total: int = 0  # Total score comparisons (4 per variable)
    keep_drop_correct: int = 0  # Binary decision matches teacher
    keep_drop_total: int = 0
    false_drops: int = 0  # Teacher KEEP, model DROP (worst error)
    false_keeps: int = 0  # Teacher DROP, model KEEP (acceptable)
    teacher_keeps: int = 0  # Total teacher KEEPs
    teacher_drops: int = 0  # Total teacher DROPs
    score_diffs: list[int] = field(default_factory=list)  # Absolute diffs per dimension
    elapsed: float = 0.0

    @property
    def format_accuracy(self) -> float:
        return self.format_correct / self.format_total * 100 if self.format_total else 0

    @property
    def score_agreement_pct(self) -> float:
        return self.score_agreement / self.score_total * 100 if self.score_total else 0

    @property
    def keep_drop_accuracy(self) -> float:
        return self.keep_drop_correct / self.keep_drop_total * 100 if self.keep_drop_total else 0

    @property
    def false_drop_rate(self) -> float:
        return self.false_drops / self.teacher_keeps * 100 if self.teacher_keeps else 0

    @property
    def false_keep_rate(self) -> float:
        return self.false_keeps / self.teacher_drops * 100 if self.teacher_drops else 0

    @property
    def mean_absolute_error(self) -> float:
        return sum(self.score_diffs) / len(self.score_diffs) if self.score_diffs else 0

    @property
    def throughput(self) -> float:
        return self.total_variables / self.elapsed if self.elapsed else 0


# =============================================================================
# EVALUATION
# =============================================================================


def load_validation_examples(val_path: str) -> list[dict]:
    """Load validation examples from JSONL.

    Each example has "conversations" with system, user, assistant messages.
    Returns list of dicts with "user_prompt" and "teacher_scores" keys.
    """
    examples = []
    with open(val_path) as f:
        for line in f:
            data = json.loads(line)
            convos = data.get("conversations", data.get("messages", []))

            user_prompt = None
            teacher_output = None
            for msg in convos:
                if msg["role"] == "user":
                    user_prompt = msg["content"]
                elif msg["role"] == "assistant":
                    teacher_output = msg["content"]

            if user_prompt and teacher_output:
                # Count variables in user prompt
                var_count = len(re.findall(r"^\d+\.", user_prompt, re.MULTILINE))

                # Parse teacher scores
                teacher_scores = _parse_scores(teacher_output, var_count)

                examples.append({
                    "user_prompt": user_prompt,
                    "teacher_output": teacher_output,
                    "teacher_scores": teacher_scores,
                    "var_count": var_count,
                })

    return examples


def evaluate_model(
    model_name: str,
    examples: list[dict],
    ollama_url: str,
    threshold: int = 5,
) -> ModelMetrics:
    """Run all validation examples through a model and compute metrics."""
    metrics = ModelMetrics(model_name=model_name)

    start_time = time.time()

    for example in examples:
        user_prompt = example["user_prompt"]
        teacher_scores = example["teacher_scores"]
        var_count = example["var_count"]

        metrics.total_examples += 1
        metrics.total_variables += var_count

        # Call model
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        try:
            output_budget = var_count * 20 + 50
            response = litellm.completion(
                model=f"ollama_chat/{model_name}",
                messages=messages,
                temperature=0.1,
                max_tokens=output_budget,
                api_base=ollama_url,
                timeout=120,
                think=False,
            )

            content = response.choices[0].message.content or ""
            if not content.strip():
                reasoning = getattr(response.choices[0].message, "reasoning_content", None)
                if reasoning:
                    content = reasoning

            # Strip think tags
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

        except Exception:
            logger.warning("Model %s failed on example", model_name, exc_info=True)
            metrics.format_total += var_count
            continue

        # Parse model scores
        model_scores = _parse_scores(content, var_count)

        # Compare with teacher
        for i in range(var_count):
            metrics.format_total += 1

            teacher = teacher_scores[i]
            model = model_scores[i]

            if model is not None:
                metrics.format_correct += 1
            else:
                continue

            if teacher is None:
                continue

            # Score agreement (within ±1 for each dimension)
            t_tuple = teacher.as_tuple()
            m_tuple = model.as_tuple()

            for t_val, m_val in zip(t_tuple, m_tuple):
                metrics.score_total += 1
                diff = abs(t_val - m_val)
                metrics.score_diffs.append(diff)
                if diff <= 1:
                    metrics.score_agreement += 1

            # Keep/drop decision
            teacher_keep = teacher.passes_threshold(threshold)
            model_keep = model.passes_threshold(threshold)

            metrics.keep_drop_total += 1
            if teacher_keep:
                metrics.teacher_keeps += 1
            else:
                metrics.teacher_drops += 1

            if teacher_keep == model_keep:
                metrics.keep_drop_correct += 1
            elif teacher_keep and not model_keep:
                metrics.false_drops += 1
            elif not teacher_keep and model_keep:
                metrics.false_keeps += 1

    metrics.elapsed = time.time() - start_time
    return metrics


def print_results(all_metrics: list[ModelMetrics], threshold: int) -> None:
    """Print comparison table."""
    print(f"\n{'='*80}")
    print(f"Variable Scorer Evaluation (threshold={threshold})")
    print(f"{'='*80}\n")

    # Header
    header = f"{'Metric':<30}"
    for m in all_metrics:
        header += f" {m.model_name:<20}"
    print(header)
    print("-" * (30 + 20 * len(all_metrics)))

    # Metrics rows
    rows = [
        ("Examples", lambda m: f"{m.total_examples}"),
        ("Variables", lambda m: f"{m.total_variables}"),
        ("Format accuracy", lambda m: f"{m.format_accuracy:.1f}%"),
        ("Score agreement (±1)", lambda m: f"{m.score_agreement_pct:.1f}%"),
        ("Keep/drop accuracy", lambda m: f"{m.keep_drop_accuracy:.1f}%"),
        ("False drop rate", lambda m: f"{m.false_drop_rate:.1f}%"),
        ("False keep rate", lambda m: f"{m.false_keep_rate:.1f}%"),
        ("Mean abs error", lambda m: f"{m.mean_absolute_error:.2f}"),
        ("Throughput (var/s)", lambda m: f"{m.throughput:.1f}"),
        ("Total time (s)", lambda m: f"{m.elapsed:.1f}"),
    ]

    for label, fn in rows:
        row = f"{label:<30}"
        for m in all_metrics:
            row += f" {fn(m):<20}"
        print(row)

    # Pass/fail summary
    print(f"\n{'='*80}")
    print("Success Criteria:")

    targets = [
        ("Format accuracy >= 98%", lambda m: m.format_accuracy >= 98),
        ("Keep/drop accuracy >= 90%", lambda m: m.keep_drop_accuracy >= 90),
        ("False drop rate <= 2%", lambda m: m.false_drop_rate <= 2),
    ]

    for label, check in targets:
        row = f"  {label:<40}"
        for m in all_metrics:
            passed = check(m)
            row += f" {'PASS' if passed else 'FAIL':<20}"
        print(row)


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned variable scorer."
    )

    parser.add_argument(
        "--val-data",
        required=True,
        help="Path to validation JSONL file",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Ollama model names to evaluate",
    )
    parser.add_argument(
        "--ollama-url",
        default="http://localhost:11434",
        help="Ollama API URL",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=5,
        help="Keep/drop threshold (default: 5)",
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

    # Load validation data
    examples = load_validation_examples(args.val_data)
    logger.info("Loaded %d validation examples", len(examples))

    if not examples:
        logger.error("No validation examples found in %s", args.val_data)
        sys.exit(1)

    # Evaluate each model
    all_metrics = []
    for model_name in args.models:
        logger.info("Evaluating model: %s", model_name)
        metrics = evaluate_model(model_name, examples, args.ollama_url, args.threshold)
        all_metrics.append(metrics)

    # Print results
    print_results(all_metrics, args.threshold)


if __name__ == "__main__":
    main()
