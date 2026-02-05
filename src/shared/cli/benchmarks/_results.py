"""Benchmark results output and markdown generation.

This module contains functions for displaying benchmark results and
saving them to markdown files.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from rich.table import Table

if TYPE_CHECKING:
    from shared.ai.ollama_benchmark import BenchmarkResult, EvaluationResult
    from shared.config import ModelConfig


def save_benchmark_markdown(
    repo_path: str,
    models_tested: list[str],
    eval_models: list[str],
    elements: list,
    results: dict,
    evaluation_results: dict,
    output_dir: str = "plans/benchmarks/data",
) -> str:
    """Save benchmark results to markdown file.

    Args:
        repo_path: Repository path.
        models_tested: List of models tested.
        eval_models: List of models used for evaluation (LLM-as-judge).
        elements: List of CodeElements.
        results: Dict mapping model -> list of BenchmarkResult.
        evaluation_results: Dict mapping element_index -> {eval_model -> EvaluationResult}.
        output_dir: Directory to save markdown file.

    Returns:
        Path to saved markdown file.
    """
    from datetime import datetime as dt
    from pathlib import Path

    from shared.ai.ollama_benchmark import EVALUATION_CRITERIA
    from shared.ai.summarization import clean_summary

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate filename with timestamp
    timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
    repo_name = Path(repo_path).name
    filename = f"{timestamp}_{repo_name}_benchmark.md"
    filepath = output_path / filename

    lines: list[str] = []

    # Header
    lines.append(f"# Benchmark Results: {repo_name}")
    lines.append("")
    lines.append(f"**Date:** {dt.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Models Tested:** {', '.join(models_tested)}")
    lines.append(f"**Evaluation Models:** {', '.join(eval_models)}")
    lines.append(f"**Elements:** {len(elements)}")
    lines.append("")

    # Helper to get average score from evaluation results
    def get_average(elem_idx: int, model: str, eval_model: str) -> float | None:
        eval_res = evaluation_results[elem_idx].get(eval_model)
        if eval_res and model in eval_res.evaluations:
            return eval_res.evaluations[model].average
        return None

    def get_notes(elem_idx: int, model: str, eval_model: str) -> str:
        eval_res = evaluation_results[elem_idx].get(eval_model)
        if eval_res and model in eval_res.evaluations:
            return eval_res.evaluations[model].notes or ""
        return ""

    def get_criteria_scores(elem_idx: int, model: str, eval_model: str) -> dict[str, int]:
        eval_res = evaluation_results[elem_idx].get(eval_model)
        if eval_res and model in eval_res.evaluations:
            return eval_res.evaluations[model].scores
        return {}

    # Build real success indices
    real_success_indices_by_model: dict[str, list[int]] = {}
    real_successes_by_model: dict[str, list] = {}
    for model in models_tested:
        model_results = results[model]
        real_successes = []
        real_success_indices = []
        for i, r in enumerate(model_results):
            if r.success and r.response.strip():
                cleaned = clean_summary(r.response)
                if cleaned.strip():
                    real_successes.append(r)
                    real_success_indices.append(i)
        real_success_indices_by_model[model] = real_success_indices
        real_successes_by_model[model] = real_successes

    # Summary per evaluator
    for eval_model in eval_models:
        lines.append(f"## Summary ({eval_model})")
        lines.append("")
        lines.append("| Model | Score | Success | Avg Time | tok/s |")
        lines.append("|-------|-------|---------|----------|-------|")

        for model in models_tested:
            real_successes = real_successes_by_model[model]
            real_success_indices = real_success_indices_by_model[model]

            # Calculate average scaled score for real successes
            model_scores = [
                get_average(i, model, eval_model)
                for i in real_success_indices
            ]
            valid_scores = [s for s in model_scores if s is not None]
            avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0

            if real_successes:
                avg_wall = sum(r.total_time for r in real_successes) / len(real_successes)
                avg_tps = sum(r.tokens_per_second for r in real_successes) / len(real_successes)
                lines.append(
                    f"| {model} | {avg_score:.1f}/10 | {len(real_successes)}/{len(results[model])} | {avg_wall:.2f}s | {avg_tps:.0f} |"
                )
            else:
                lines.append(f"| {model} | - | 0/{len(results[model])} | - | - |")

        lines.append("")

    # Criteria breakdown by element type
    elements_by_type: dict[str, list[int]] = defaultdict(list)
    for i, elem in enumerate(elements):
        elements_by_type[elem.element_type].append(i)

    lines.append("## Criteria Scores by Element Type")
    lines.append("")

    for elem_type in sorted(elements_by_type.keys()):
        indices = elements_by_type[elem_type]
        criteria = EVALUATION_CRITERIA.get(elem_type, {})

        lines.append(f"### {elem_type} ({len(indices)} elements)")
        lines.append("")

        # Header
        header = "| Criterion |"
        separator = "|-----------|"
        for model in models_tested:
            header += f" {model} |"
            separator += "-" * (len(model) + 2) + "|"
        lines.append(header)
        lines.append(separator)

        for criterion in criteria.keys():
            row = f"| {criterion} |"
            for model in models_tested:
                criterion_scores = []
                for i in indices:
                    r = results[model][i]
                    if r.success and r.response.strip():
                        cleaned = clean_summary(r.response)
                        if cleaned.strip():
                            for em in eval_models:
                                scores = get_criteria_scores(i, model, em)
                                if criterion in scores:
                                    criterion_scores.append(scores[criterion])
                if criterion_scores:
                    avg = sum(criterion_scores) / len(criterion_scores)
                    row += f" {avg:.1f} |"
                else:
                    row += " - |"
            lines.append(row)

        # Overall row
        overall_row = "| **Overall** |"
        for model in models_tested:
            model_scores_data = []
            for i in indices:
                r = results[model][i]
                if r.success and r.response.strip():
                    cleaned = clean_summary(r.response)
                    if cleaned.strip():
                        for em in eval_models:
                            score = get_average(i, model, em)
                            if score is not None:
                                model_scores_data.append(score)
            if model_scores_data:
                avg = sum(model_scores_data) / len(model_scores_data)
                overall_row += f" **{avg:.1f}/10** |"
            else:
                overall_row += " - |"
        lines.append(overall_row)

        lines.append("")

    # Averaged summary across all evaluators (if multiple)
    if len(eval_models) > 1:
        lines.append("## Summary (Averaged Across Evaluators)")
        lines.append("")
        lines.append("| Model | Avg Score | Success | Avg Time | tok/s |")
        lines.append("|-------|-----------|---------|----------|-------|")

        for model in models_tested:
            real_successes = real_successes_by_model[model]
            real_success_indices = real_success_indices_by_model[model]

            all_scores = []
            for i in real_success_indices:
                for em in eval_models:
                    score = get_average(i, model, em)
                    if score is not None:
                        all_scores.append(score)
            avg_score = sum(all_scores) / len(all_scores) if all_scores else 0

            if real_successes:
                avg_wall = sum(r.total_time for r in real_successes) / len(real_successes)
                avg_tps = sum(r.tokens_per_second for r in real_successes) / len(real_successes)
                lines.append(
                    f"| {model} | {avg_score:.1f}/10 | {len(real_successes)}/{len(results[model])} | {avg_wall:.2f}s | {avg_tps:.0f} |"
                )
            else:
                lines.append(f"| {model} | - | 0/{len(results[model])} | - | - |")

        lines.append("")

    # Detailed results per element
    lines.append("## Detailed Results")
    lines.append("")

    for i, elem in enumerate(elements):
        elem_name = f"{elem.element_type}:{elem.name}"
        lines.append(f"### {elem_name}")
        lines.append("")
        lines.append(f"**File:** `{elem.relative_path}`")
        lines.append(f"**Lines:** {elem.line_start}-{elem.line_end}")
        lines.append("")

        # Source code snippet
        if elem.raw_code:
            lines.append("<details>")
            lines.append("<summary>Source Code</summary>")
            lines.append("")
            lines.append("```")
            # Truncate if too long
            code = elem.raw_code
            if len(code) > 2000:
                code = code[:2000] + "\n... (truncated)"
            lines.append(code)
            lines.append("```")
            lines.append("</details>")
            lines.append("")

        # Summary comparison table
        lines.append("| Model | Score | Summary |")
        lines.append("|-------|-------|---------|")

        for model in models_tested:
            model_result = results[model][i]

            summary = ""
            generation_ok = False
            if model_result.success and model_result.response.strip():
                summary = clean_summary(model_result.response)
                generation_ok = bool(summary.strip())

            if generation_ok:
                avg_score = sum(
                    get_average(i, model, em) or 0
                    for em in eval_models
                ) / len(eval_models)
                # Escape pipe characters in summary
                summary_escaped = summary.replace("|", "\\|").replace("\n", " ")
                if len(summary_escaped) > 200:
                    summary_escaped = summary_escaped[:200] + "..."
                lines.append(f"| {model} | {avg_score:.1f}/10 | {summary_escaped} |")
            else:
                error_msg = model_result.error if model_result.error else "empty after cleaning"
                lines.append(f"| {model} | N/A | *(failed: {error_msg[:30]})* |")

        lines.append("")

        # Full summaries with criteria breakdown (expandable)
        lines.append("<details>")
        lines.append("<summary>Full Summaries & Criteria</summary>")
        lines.append("")

        for model in models_tested:
            model_result = results[model][i]

            summary = ""
            generation_ok = False
            if model_result.success and model_result.response.strip():
                summary = clean_summary(model_result.response)
                generation_ok = bool(summary.strip())

            if generation_ok:
                # Get scores and notes from evaluators
                score_parts = []
                for eval_name in eval_models:
                    score = get_average(i, model, eval_name)
                    notes = get_notes(i, model, eval_name)
                    eval_short = eval_name.split('/')[0] if '/' in eval_name else eval_name
                    if score is not None:
                        if notes:
                            score_parts.append(f"{eval_short}: {score:.1f}/10 - {notes}")
                        else:
                            score_parts.append(f"{eval_short}: {score:.1f}/10")
                    else:
                        score_parts.append(f"{eval_short}: ?")

                lines.append(f"**{model}**")
                lines.append("")
                lines.append(f"Scores: {' | '.join(score_parts)}")
                lines.append("")

                # Show criteria breakdown
                criteria = EVALUATION_CRITERIA.get(elem.element_type, {})
                if criteria:
                    lines.append("Criteria:")
                    for eval_name in eval_models:
                        scores = get_criteria_scores(i, model, eval_name)
                        if scores:
                            eval_short = eval_name.split('/')[0] if '/' in eval_name else eval_name
                            criteria_str = ", ".join(f"{k}={v}" for k, v in scores.items())
                            lines.append(f"- {eval_short}: {criteria_str}")
                    lines.append("")

                lines.append(f"> {summary}")
            else:
                error_msg = model_result.error if model_result.error else "empty after cleaning"
                lines.append(f"**{model}** (N/A - failed: {error_msg[:40]})")
                lines.append("")
                lines.append("> *(no valid output)*")
            lines.append("")

        lines.append("</details>")
        lines.append("")
        lines.append("---")
        lines.append("")

    # Write file
    with open(filepath, "w") as f:
        f.write("\n".join(lines))

    return str(filepath)


def display_benchmark_summary(
    models_to_test: list["ModelConfig"],
    elements: list,
    results: dict[str, list["BenchmarkResult"]],
    evaluation_results: dict[int, dict[str, "EvaluationResult"]],
    eval_display_names: list[str],
    console,
) -> None:
    """Display benchmark summary tables in the console.

    Args:
        models_to_test: List of model configurations tested.
        elements: List of code elements.
        results: Dict mapping model display name to benchmark results.
        evaluation_results: Dict mapping element index to evaluation results.
        eval_display_names: List of evaluator display names.
        console: Rich console for output.
    """
    from shared.ai.ollama_benchmark import EVALUATION_CRITERIA
    from shared.ai.summarization import clean_summary

    tested_model_names = [f"{mc.provider}/{mc.name}" for mc in models_to_test]
    prompts = [(elem, None) for elem in elements]

    # Build real_success_indices once
    real_success_indices_by_model: dict[str, list[int]] = {}
    real_successes_by_model: dict[str, list] = {}
    for model_name in tested_model_names:
        model_results_list = results[model_name]
        real_successes = []
        real_success_indices = []
        for i, r in enumerate(model_results_list):
            if r.success and r.response.strip():
                cleaned = clean_summary(r.response)
                if cleaned.strip():
                    real_successes.append(r)
                    real_success_indices.append(i)
        real_success_indices_by_model[model_name] = real_success_indices
        real_successes_by_model[model_name] = real_successes

    # Helper to get scaled score (1-10) from evaluation results
    def get_average(elem_idx: int, model_name: str, eval_name: str) -> float | None:
        eval_res = evaluation_results[elem_idx].get(eval_name)
        if eval_res and model_name in eval_res.evaluations:
            return eval_res.evaluations[model_name].average
        return None

    # Helper to get criteria scores
    def get_criteria_scores(elem_idx: int, model_name: str, eval_name: str) -> dict[str, int]:
        eval_res = evaluation_results[elem_idx].get(eval_name)
        if eval_res and model_name in eval_res.evaluations:
            return eval_res.evaluations[model_name].scores
        return {}

    # Show summary table per evaluator
    for eval_name in eval_display_names:
        console.print(f"\n[bold]Evaluator: {eval_name}[/]")

        summary_table = Table(show_header=True, header_style="bold cyan")
        summary_table.add_column("Model", style="cyan", no_wrap=True)
        summary_table.add_column("Score", justify="center")
        summary_table.add_column("Success", justify="center")
        summary_table.add_column("Avg Time", justify="right")
        summary_table.add_column("tok/s", justify="right")

        for model_name in tested_model_names:
            real_successes = real_successes_by_model[model_name]
            real_success_indices = real_success_indices_by_model[model_name]

            # Calculate average scaled score for real successes
            model_scores = [
                get_average(i, model_name, eval_name)
                for i in real_success_indices
            ]
            valid_scores = [s for s in model_scores if s is not None]
            avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0

            if real_successes:
                avg_wall = sum(r.total_time for r in real_successes) / len(real_successes)
                avg_tps = sum(r.tokens_per_second for r in real_successes) / len(real_successes)

                # Color-code score (1-10 scale)
                if avg_score >= 8:
                    score_style = "bold green"
                elif avg_score >= 6:
                    score_style = "yellow"
                else:
                    score_style = "red"

                summary_table.add_row(
                    model_name,
                    f"[{score_style}]{avg_score:.1f}/10[/]" if valid_scores else "-",
                    f"{len(real_successes)}/{len(results[model_name])}",
                    f"{avg_wall:.2f}s",
                    f"{avg_tps:.0f}",
                )
            else:
                summary_table.add_row(
                    model_name,
                    "-",
                    f"0/{len(results[model_name])}",
                    "-",
                    "-",
                )

        console.print(summary_table)

    # Criteria breakdown by element type
    console.print("\n[bold]Criteria Scores by Element Type[/]")

    # Group elements by type
    elements_by_type: dict[str, list[int]] = defaultdict(list)
    for i, (elem, _) in enumerate(prompts):
        elements_by_type[elem.element_type].append(i)

    # Show criteria breakdown for each element type
    for elem_type in sorted(elements_by_type.keys()):
        indices = elements_by_type[elem_type]
        criteria = EVALUATION_CRITERIA.get(elem_type, {})

        console.print(f"\n[cyan]{elem_type}[/] ({len(indices)} elements)")

        # Build table with criteria as rows, models as columns
        criteria_table = Table(show_header=True, header_style="bold", expand=False)
        criteria_table.add_column("Criterion", style="dim", no_wrap=True, min_width=20)
        for model_name in tested_model_names:
            criteria_table.add_column(model_name.replace("/", "\n"), justify="center")

        for criterion in criteria.keys():
            row = [criterion]
            for model_name in tested_model_names:
                # Average this criterion across all elements of this type and all evaluators
                criterion_scores = []
                for i in indices:
                    r = results[model_name][i]
                    if r.success and r.response.strip():
                        cleaned = clean_summary(r.response)
                        if cleaned.strip():
                            for e_name in eval_display_names:
                                scores = get_criteria_scores(i, model_name, e_name)
                                if criterion in scores:
                                    criterion_scores.append(scores[criterion])
                if criterion_scores:
                    avg = sum(criterion_scores) / len(criterion_scores)
                    # Color based on 1-10 scale
                    if avg >= 8:
                        row.append(f"[bold green]{avg:.1f}[/]")
                    elif avg >= 6:
                        row.append(f"[yellow]{avg:.1f}[/]")
                    else:
                        row.append(f"[red]{avg:.1f}[/]")
                else:
                    row.append("-")
            criteria_table.add_row(*row)

        # Add overall row
        overall_row = ["[bold]Overall[/]"]
        for model_name in tested_model_names:
            model_scores = []
            for i in indices:
                r = results[model_name][i]
                if r.success and r.response.strip():
                    cleaned = clean_summary(r.response)
                    if cleaned.strip():
                        for e_name in eval_display_names:
                            score = get_average(i, model_name, e_name)
                            if score is not None:
                                model_scores.append(score)
            if model_scores:
                avg = sum(model_scores) / len(model_scores)
                if avg >= 8:
                    overall_row.append(f"[bold green]{avg:.1f}/10[/]")
                elif avg >= 6:
                    overall_row.append(f"[yellow]{avg:.1f}/10[/]")
                else:
                    overall_row.append(f"[red]{avg:.1f}/10[/]")
            else:
                overall_row.append("-")
        criteria_table.add_row(*overall_row)

        console.print(criteria_table)

    # Token stats
    console.print("\n[bold]Token Statistics[/]")
    for model_name in tested_model_names:
        successful = [r for r in results[model_name] if r.success]
        if successful:
            total_prompt = sum(r.prompt_tokens for r in successful)
            total_output = sum(r.output_tokens for r in successful)
            console.print(f"  {model_name}: {total_prompt:,} prompt tokens, {total_output:,} output tokens")
