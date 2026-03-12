#!/Users/dinnyosz/code/magaldi/.venv/bin/python
"""Evaluate variable scoring model against teacher labels via Ollama.

Standalone CLI script that sends validation examples to an Ollama model
and compares predicted scores to the expected scores from training data.

The validation JSONL uses ChatML-style messages with 7-dimension scores:
  config_value, architectural_role, data_definition, general_usefulness,
  value_complexity, naming_quality, scope_significance

Usage:
    python tools/training/evaluate_scorer.py \
      --val-data tools/training/data/variable_scorer/validation.jsonl \
      --model magaldi-scorer \
      --limit 20

    # Evaluate multiple models side by side:
    python tools/training/evaluate_scorer.py \
      --val-data tools/training/data/variable_scorer/validation.jsonl \
      --model magaldi-scorer qwen3.5:4b \
      --limit 50
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

try:
    import requests
except ImportError:
    print("ERROR: 'requests' package required. Install with: pip install requests", file=sys.stderr)
    sys.exit(1)


# ============================================================================
# Constants
# ============================================================================

DIMENSION_NAMES = [
    "config_value",
    "architectural_role",
    "data_definition",
    "general_usefulness",
    "value_complexity",
    "naming_quality",
    "scope_significance",
]
NUM_DIMENSIONS = len(DIMENSION_NAMES)

# Score line pattern: "N. X,X,X,X,X,X,X" where N is 1-based index
# Tolerates optional spaces around commas and after the period
SCORE_LINE_RE = re.compile(
    r"(\d+)\.\s*"
    r"(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)"
)

# Default validation data path relative to project root
DEFAULT_VAL_DATA = "tools/training/data/variable_scorer/validation.jsonl"


# ============================================================================
# Data structures
# ============================================================================


@dataclass
class ScoreLine:
    """A single variable's 7-dimension score vector."""

    scores: tuple[int, ...]  # Length 7

    @property
    def is_all_ones(self) -> bool:
        return all(s == 1 for s in self.scores)

    @property
    def has_high_score(self) -> bool:
        """At least one dimension >= 5."""
        return any(s >= 5 for s in self.scores)

    @property
    def max_score(self) -> int:
        return max(self.scores)


@dataclass
class ValidationExample:
    """A single validation example with system/user/assistant messages."""

    system_prompt: str
    user_prompt: str
    expected_output: str
    expected_scores: list[ScoreLine | None]  # One per variable
    var_count: int


@dataclass
class ModelMetrics:
    """Accumulated evaluation metrics for a single model."""

    model_name: str
    total_examples: int = 0
    total_variables: int = 0

    # Format correctness: did the model produce parseable score lines?
    format_correct: int = 0
    format_total: int = 0

    # Line-level exact match: entire 7-score line matches
    exact_match: int = 0
    exact_match_total: int = 0

    # Per-score accuracy: each individual score matches exactly
    per_score_match: int = 0
    per_score_total: int = 0

    # Per-dimension absolute error accumulation
    dimension_abs_errors: list[float] = field(
        default_factory=lambda: [0.0] * NUM_DIMENSIONS
    )
    dimension_counts: list[int] = field(
        default_factory=lambda: [0] * NUM_DIMENSIONS
    )

    # All absolute diffs (across all dimensions)
    all_abs_diffs: list[int] = field(default_factory=list)

    # Keep/drop confusion (threshold-based)
    keep_drop_correct: int = 0
    keep_drop_total: int = 0
    false_drops: int = 0  # Teacher KEEP, model DROP
    false_keeps: int = 0  # Teacher DROP, model KEEP
    teacher_keeps: int = 0
    teacher_drops: int = 0

    # All-1s confusion matrix
    both_all_ones: int = 0      # Both teacher and model say all 1s
    both_has_high: int = 0      # Both say has high scores
    teacher_all_ones_model_high: int = 0  # Teacher all 1s, model has high
    teacher_high_model_all_ones: int = 0  # Teacher has high, model all 1s
    confusion_total: int = 0

    # API errors
    api_errors: int = 0
    elapsed: float = 0.0

    # ---- Computed properties ----

    @property
    def format_accuracy(self) -> float:
        return self.format_correct / self.format_total * 100 if self.format_total else 0

    @property
    def exact_match_rate(self) -> float:
        return self.exact_match / self.exact_match_total * 100 if self.exact_match_total else 0

    @property
    def per_score_accuracy(self) -> float:
        return self.per_score_match / self.per_score_total * 100 if self.per_score_total else 0

    @property
    def mean_absolute_error(self) -> float:
        return sum(self.all_abs_diffs) / len(self.all_abs_diffs) if self.all_abs_diffs else 0

    def dimension_mae(self, dim_idx: int) -> float:
        count = self.dimension_counts[dim_idx]
        return self.dimension_abs_errors[dim_idx] / count if count else 0

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
    def throughput(self) -> float:
        return self.total_variables / self.elapsed if self.elapsed else 0


# ============================================================================
# Score parsing
# ============================================================================


def parse_score_lines(output: str, expected_count: int) -> list[ScoreLine | None]:
    """Parse model output into score lines.

    Expected format per line:
        N. X,X,X,X,X,X,X

    Returns a list of length expected_count with ScoreLine or None for
    lines that could not be parsed.
    """
    # Strip <think>...</think> blocks (Qwen3 thinking models)
    output = re.sub(r"<think>.*?</think>", "", output, flags=re.DOTALL).strip()

    results: list[ScoreLine | None] = [None] * expected_count

    for match in SCORE_LINE_RE.finditer(output):
        idx = int(match.group(1))
        if 1 <= idx <= expected_count:
            scores = tuple(
                min(10, max(1, int(match.group(i + 2))))
                for i in range(NUM_DIMENSIONS)
            )
            results[idx - 1] = ScoreLine(scores=scores)

    return results


# ============================================================================
# Data loading
# ============================================================================


def load_validation_examples(val_path: str, limit: int = 0) -> list[ValidationExample]:
    """Load validation examples from JSONL.

    Each line has a "conversations" (or "messages") key with ChatML messages:
    system, user, assistant.

    Args:
        val_path: Path to the validation JSONL file.
        limit: Maximum number of examples to load. 0 = all.

    Returns:
        List of ValidationExample objects.
    """
    examples: list[ValidationExample] = []
    path = Path(val_path)

    if not path.exists():
        print(f"ERROR: Validation file not found: {val_path}", file=sys.stderr)
        sys.exit(1)

    with open(path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"WARNING: Skipping line {line_num}: {e}", file=sys.stderr)
                continue

            convos = data.get("conversations", data.get("messages", []))

            system_prompt = ""
            user_prompt = ""
            assistant_output = ""

            for msg in convos:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "system":
                    system_prompt = content
                elif role == "user":
                    user_prompt = content
                elif role == "assistant":
                    assistant_output = content

            if not user_prompt or not assistant_output:
                continue

            # Count variables in user prompt (lines starting with "N.")
            var_count = len(re.findall(r"^\d+\.", user_prompt, re.MULTILINE))
            if var_count == 0:
                continue

            # Parse expected scores
            expected_scores = parse_score_lines(assistant_output, var_count)

            examples.append(ValidationExample(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                expected_output=assistant_output,
                expected_scores=expected_scores,
                var_count=var_count,
            ))

            if limit and len(examples) >= limit:
                break

    return examples


# ============================================================================
# Ollama API
# ============================================================================


def call_ollama(
    model: str,
    system_prompt: str,
    user_prompt: str,
    ollama_url: str,
    var_count: int,
) -> str | None:
    """Send a chat completion request to Ollama and return the response content.

    Uses temperature=0 for deterministic output.

    Returns:
        The assistant's response text, or None on error.
    """
    url = f"{ollama_url}/api/chat"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    # Output budget: ~25 tokens per variable for 7 scores + overhead
    num_predict = var_count * 25 + 50

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0,
            "num_predict": num_predict,
        },
    }

    try:
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        content = data.get("message", {}).get("content", "")
        return content
    except requests.exceptions.ConnectionError:
        print(
            f"ERROR: Cannot connect to Ollama at {ollama_url}. "
            "Is Ollama running?",
            file=sys.stderr,
        )
        return None
    except requests.exceptions.Timeout:
        print(f"WARNING: Ollama request timed out for model {model}", file=sys.stderr)
        return None
    except requests.exceptions.HTTPError as e:
        print(f"WARNING: Ollama HTTP error: {e}", file=sys.stderr)
        return None
    except (json.JSONDecodeError, KeyError) as e:
        print(f"WARNING: Failed to parse Ollama response: {e}", file=sys.stderr)
        return None


# ============================================================================
# Evaluation
# ============================================================================


def evaluate_model(
    model_name: str,
    examples: list[ValidationExample],
    ollama_url: str,
    threshold: int = 5,
    verbose: bool = False,
) -> ModelMetrics:
    """Run all validation examples through a model and compute metrics."""
    metrics = ModelMetrics(model_name=model_name)
    start_time = time.time()

    for ex_idx, example in enumerate(examples, 1):
        metrics.total_examples += 1
        metrics.total_variables += example.var_count

        if verbose:
            print(f"  [{ex_idx}/{len(examples)}] {example.var_count} variables...", end=" ", flush=True)

        # Call Ollama
        content = call_ollama(
            model=model_name,
            system_prompt=example.system_prompt,
            user_prompt=example.user_prompt,
            ollama_url=ollama_url,
            var_count=example.var_count,
        )

        if content is None:
            metrics.api_errors += 1
            metrics.format_total += example.var_count
            if verbose:
                print("ERROR")
            continue

        # Strip thinking tags
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

        # Parse predicted scores
        predicted = parse_score_lines(content, example.var_count)

        if verbose:
            parsed_count = sum(1 for s in predicted if s is not None)
            print(f"parsed {parsed_count}/{example.var_count}")

        # Compare predicted vs expected
        for i in range(example.var_count):
            metrics.format_total += 1
            teacher = example.expected_scores[i]
            model_score = predicted[i]

            # Format correctness
            if model_score is not None:
                metrics.format_correct += 1
            else:
                continue

            if teacher is None:
                continue

            # Line-level exact match
            metrics.exact_match_total += 1
            if teacher.scores == model_score.scores:
                metrics.exact_match += 1

            # Per-score accuracy and per-dimension MAE
            for dim_idx in range(NUM_DIMENSIONS):
                t_val = teacher.scores[dim_idx]
                m_val = model_score.scores[dim_idx]

                metrics.per_score_total += 1
                if t_val == m_val:
                    metrics.per_score_match += 1

                diff = abs(t_val - m_val)
                metrics.all_abs_diffs.append(diff)
                metrics.dimension_abs_errors[dim_idx] += diff
                metrics.dimension_counts[dim_idx] += 1

            # Keep/drop decision (any dimension >= threshold)
            teacher_keep = teacher.has_high_score if threshold == 5 else teacher.max_score >= threshold
            model_keep = model_score.has_high_score if threshold == 5 else model_score.max_score >= threshold

            metrics.keep_drop_total += 1
            if teacher_keep:
                metrics.teacher_keeps += 1
            else:
                metrics.teacher_drops += 1

            if teacher_keep == model_keep:
                metrics.keep_drop_correct += 1
            elif teacher_keep and not model_keep:
                metrics.false_drops += 1
            else:
                metrics.false_keeps += 1

            # All-1s vs has-high confusion
            metrics.confusion_total += 1
            t_all_ones = teacher.is_all_ones
            m_all_ones = model_score.is_all_ones

            if t_all_ones and m_all_ones:
                metrics.both_all_ones += 1
            elif not t_all_ones and not m_all_ones:
                metrics.both_has_high += 1
            elif t_all_ones and not m_all_ones:
                metrics.teacher_all_ones_model_high += 1
            else:
                metrics.teacher_high_model_all_ones += 1

    metrics.elapsed = time.time() - start_time
    return metrics


# ============================================================================
# Reporting
# ============================================================================


def print_results(all_metrics: list[ModelMetrics], threshold: int) -> None:
    """Print a comparison summary table."""
    col_width = max(22, max(len(m.model_name) + 2 for m in all_metrics))
    label_width = 35

    print(f"\n{'=' * 80}")
    print(f"Variable Scorer Evaluation (threshold={threshold})")
    print(f"{'=' * 80}\n")

    # Header
    header = f"{'Metric':<{label_width}}"
    for m in all_metrics:
        header += f" {m.model_name:<{col_width}}"
    print(header)
    print("-" * (label_width + (col_width + 1) * len(all_metrics)))

    # General stats
    rows: list[tuple[str, ...]] = [
        ("Examples",) + tuple(f"{m.total_examples}" for m in all_metrics),
        ("Variables",) + tuple(f"{m.total_variables}" for m in all_metrics),
        ("API errors",) + tuple(f"{m.api_errors}" for m in all_metrics),
        ("",),  # blank separator
        ("Format accuracy",) + tuple(f"{m.format_accuracy:.1f}%" for m in all_metrics),
        ("Exact match rate (line)",) + tuple(f"{m.exact_match_rate:.1f}%" for m in all_metrics),
        ("Per-score accuracy",) + tuple(f"{m.per_score_accuracy:.1f}%" for m in all_metrics),
        ("Mean absolute error",) + tuple(f"{m.mean_absolute_error:.2f}" for m in all_metrics),
        ("",),
        ("Keep/drop accuracy",) + tuple(f"{m.keep_drop_accuracy:.1f}%" for m in all_metrics),
        ("False drop rate",) + tuple(f"{m.false_drop_rate:.1f}%" for m in all_metrics),
        ("False keep rate",) + tuple(f"{m.false_keep_rate:.1f}%" for m in all_metrics),
        ("",),
        ("Throughput (var/s)",) + tuple(f"{m.throughput:.1f}" for m in all_metrics),
        ("Total time (s)",) + tuple(f"{m.elapsed:.1f}" for m in all_metrics),
    ]

    for row in rows:
        if len(row) == 1 and row[0] == "":
            print()
            continue
        line = f"{row[0]:<{label_width}}"
        for val in row[1:]:
            line += f" {val:<{col_width}}"
        print(line)

    # Per-dimension MAE
    print(f"\n{'=' * 80}")
    print("Mean Absolute Error by Dimension")
    print("-" * (label_width + (col_width + 1) * len(all_metrics)))

    for dim_idx, dim_name in enumerate(DIMENSION_NAMES):
        line = f"  {dim_name:<{label_width - 2}}"
        for m in all_metrics:
            mae = m.dimension_mae(dim_idx)
            line += f" {mae:<{col_width}.3f}"
        print(line)

    # All-1s confusion matrix
    print(f"\n{'=' * 80}")
    print("All-1s vs Has-High Confusion Matrix")
    print("-" * (label_width + (col_width + 1) * len(all_metrics)))

    confusion_rows = [
        ("Both all-1s (agree low)",) + tuple(
            f"{m.both_all_ones} ({m.both_all_ones / m.confusion_total * 100:.1f}%)"
            if m.confusion_total else "0"
            for m in all_metrics
        ),
        ("Both has-high (agree high)",) + tuple(
            f"{m.both_has_high} ({m.both_has_high / m.confusion_total * 100:.1f}%)"
            if m.confusion_total else "0"
            for m in all_metrics
        ),
        ("Teacher=all-1s, Model=high",) + tuple(
            f"{m.teacher_all_ones_model_high} ({m.teacher_all_ones_model_high / m.confusion_total * 100:.1f}%)"
            if m.confusion_total else "0"
            for m in all_metrics
        ),
        ("Teacher=high, Model=all-1s",) + tuple(
            f"{m.teacher_high_model_all_ones} ({m.teacher_high_model_all_ones / m.confusion_total * 100:.1f}%)"
            if m.confusion_total else "0"
            for m in all_metrics
        ),
    ]

    for row in confusion_rows:
        line = f"  {row[0]:<{label_width - 2}}"
        for val in row[1:]:
            line += f" {val:<{col_width}}"
        print(line)

    # Pass/fail summary
    print(f"\n{'=' * 80}")
    print("Success Criteria:")

    targets: list[tuple[str, ...]] = [
        ("Format accuracy >= 98%",) + tuple(
            "PASS" if m.format_accuracy >= 98 else "FAIL"
            for m in all_metrics
        ),
        ("Keep/drop accuracy >= 90%",) + tuple(
            "PASS" if m.keep_drop_accuracy >= 90 else "FAIL"
            for m in all_metrics
        ),
        ("False drop rate <= 2%",) + tuple(
            "PASS" if m.false_drop_rate <= 2 else "FAIL"
            for m in all_metrics
        ),
        ("Per-score accuracy >= 50%",) + tuple(
            "PASS" if m.per_score_accuracy >= 50 else "FAIL"
            for m in all_metrics
        ),
        ("MAE <= 1.5",) + tuple(
            "PASS" if m.mean_absolute_error <= 1.5 else "FAIL"
            for m in all_metrics
        ),
    ]

    for row in targets:
        line = f"  {row[0]:<{label_width - 2}}"
        for val in row[1:]:
            line += f" {val:<{col_width}}"
        print(line)

    print()


# ============================================================================
# CLI
# ============================================================================


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parents[1]
    default_val = str(project_root / DEFAULT_VAL_DATA)

    parser = argparse.ArgumentParser(
        description="Evaluate variable scoring model against teacher labels via Ollama.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  %(prog)s --model magaldi-scorer
  %(prog)s --model magaldi-scorer qwen3.5:4b --limit 50
  %(prog)s --val-data path/to/validation.jsonl --model my-model
""",
    )

    parser.add_argument(
        "--model",
        nargs="+",
        default=["magaldi-scorer"],
        help="Ollama model name(s) to evaluate (default: magaldi-scorer)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum number of validation examples to evaluate (default: 20, 0=all)",
    )
    parser.add_argument(
        "--val-data",
        default=default_val,
        help=f"Path to validation JSONL file (default: {DEFAULT_VAL_DATA})",
    )
    parser.add_argument(
        "--ollama-url",
        default="http://localhost:11434",
        help="Ollama API base URL (default: http://localhost:11434)",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=5,
        help="Keep/drop threshold for any dimension (default: 5)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show per-example progress",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load validation data
    examples = load_validation_examples(args.val_data, limit=args.limit)
    print(f"Loaded {len(examples)} validation examples from {args.val_data}")

    if not examples:
        print(f"ERROR: No validation examples found in {args.val_data}", file=sys.stderr)
        sys.exit(1)

    total_vars = sum(ex.var_count for ex in examples)
    print(f"Total variables to score: {total_vars}")
    print(f"Models to evaluate: {', '.join(args.model)}")
    print()

    # Check Ollama connectivity
    try:
        resp = requests.get(f"{args.ollama_url}/api/tags", timeout=5)
        resp.raise_for_status()
        available_models = [m["name"] for m in resp.json().get("models", [])]
        for model_name in args.model:
            # Check if model is available (Ollama may use full name with :latest)
            found = any(
                model_name == m or model_name == m.split(":")[0]
                for m in available_models
            )
            if not found:
                print(f"WARNING: Model '{model_name}' not found in Ollama. Available: {', '.join(available_models)}", file=sys.stderr)
    except Exception:
        print(f"WARNING: Cannot reach Ollama at {args.ollama_url}. Proceeding anyway...", file=sys.stderr)

    # Evaluate each model
    all_metrics: list[ModelMetrics] = []
    for model_name in args.model:
        print(f"Evaluating: {model_name}")
        metrics = evaluate_model(
            model_name=model_name,
            examples=examples,
            ollama_url=args.ollama_url,
            threshold=args.threshold,
            verbose=args.verbose,
        )
        all_metrics.append(metrics)
        print(f"  Done in {metrics.elapsed:.1f}s ({metrics.throughput:.1f} var/s)")

    # Print results
    print_results(all_metrics, args.threshold)


if __name__ == "__main__":
    main()
