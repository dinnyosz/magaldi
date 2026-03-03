"""CLI command for RALPH prompt optimization.

Provides `magaldi prompt-improver` command that iteratively improves
summarization prompts for a specific code element.
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
import yaml
from rich.table import Table

from shared.cli._shared import console, main
from shared.config import load_config


@main.command("prompt-improver")
@click.argument("element_id")
@click.option(
    "--max-iterations", "-n",
    default=5,
    show_default=True,
    help="Maximum number of RALPH iterations (including baseline).",
)
@click.option(
    "--target-score", "-t",
    default=8.0,
    show_default=True,
    help="Stop when average score >= this value (1-10 scale).",
)
@click.option(
    "--model", "-m",
    default=None,
    help="Override summary model (e.g., 'ollama/qwen3.5:4b', 'qwen3.5:4b').",
)
@click.option(
    "--eval-model", "-e",
    default=None,
    help="Override evaluation model (e.g., 'qwen3.5:4b').",
)
@click.option(
    "--improver-model", "-i",
    default=None,
    help="Override prompt improver model. Defaults to eval model.",
)
@click.option(
    "--save-best", "-s",
    default=None,
    type=click.Path(),
    help="Save best prompt to YAML file.",
)
def prompt_improver(
    element_id: str,
    max_iterations: int,
    target_score: float,
    model: str | None,
    eval_model: str | None,
    improver_model: str | None,
    save_best: str | None,
) -> None:
    """Optimize summarization prompts using RALPH feedback loop.

    Fetches an indexed element by ELEMENT_ID (element ID or hash), generates
    summaries, evaluates them, and iteratively improves the prompt until
    convergence.

    ELEMENT_ID is an element ID (scope:repo:user:path:type:name:line)
    or a hash prefix (hex string, e.g., 'abc123def').
    """
    config = load_config(skip_validation=True)

    # Resolve model overrides
    summary_mc = None
    eval_mc = None
    improver_mc = None

    if model:
        from shared.cli.benchmarks._helpers import parse_model_spec_to_config
        summary_mc = parse_model_spec_to_config(model)

    if eval_model:
        from shared.cli.benchmarks._helpers import parse_model_spec_to_config
        eval_mc = parse_model_spec_to_config(eval_model)

    if improver_model:
        from shared.cli.benchmarks._helpers import parse_model_spec_to_config
        improver_mc = parse_model_spec_to_config(improver_model)

    # Print header
    console.print("[bold blue]RALPH Prompt Optimization[/]")
    console.print(f"  Element: [cyan]{element_id}[/]")
    console.print(f"  Max iterations: {max_iterations}")
    console.print(f"  Target score: {target_score}")

    # Show model info
    from shared.ai.prompt_improver import PromptImprover

    improver = PromptImprover(
        config=config,
        summary_model=summary_mc,
        eval_model=eval_mc,
        improver_model=improver_mc,
    )

    console.print(f"  Summary model: [cyan]{improver._summary_mc.provider}/{improver._summary_mc.name}[/]")
    console.print(f"  Eval model: [cyan]{improver._eval_mc.provider}/{improver._eval_mc.name}[/]")
    console.print(f"  Improver model: [cyan]{improver._improver_mc.provider}/{improver._improver_mc.name}[/]")
    console.print()

    try:
        # Run with live progress
        def on_iteration(iter_result: object) -> None:
            """Print progress for each iteration."""
            from shared.ai.prompt_improver import IterationResult
            assert isinstance(iter_result, IterationResult)

            i = iter_result.iteration
            score = iter_result.avg_score

            if i == 0:
                label = "Baseline (production prompt)"
            else:
                label = f"Improved prompt"
                if iter_result.prompt.reasoning:
                    label += f" — {iter_result.prompt.reasoning[:80]}"

            # Score delta from previous
            delta_str = ""
            if i > 0:
                # We don't have access to previous here, but the main result will show it
                pass

            score_color = "green" if score >= target_score else "yellow" if score >= 6.0 else "red"
            target_marker = " [green]✓ TARGET REACHED[/]" if score >= target_score else ""

            console.print(
                f"  Iteration {i}: [{score_color}]{score:.1f}/10[/] "
                f"({iter_result.summary_time:.1f}s gen + {iter_result.eval_time:.1f}s eval) "
                f"[dim]{label}[/]{target_marker}"
            )

            # Show the summary
            if iter_result.summary:
                summary_preview = iter_result.summary[:200]
                if len(iter_result.summary) > 200:
                    summary_preview += "..."
                console.print(f"    [dim]Summary: {summary_preview}[/]")

        result = improver.run(
            element_id=element_id,
            max_iterations=max_iterations,
            target_score=target_score,
            on_iteration=on_iteration,
        )

        # Results header
        console.print()
        if result.converged:
            console.print(f"[bold green]Converged![/] {result.stop_reason}")
        else:
            console.print(f"[bold yellow]Stopped:[/] {result.stop_reason}")

        console.print(f"  Element: [cyan]{result.element_type}:{result.element_name}[/] ({result.element_id})")
        console.print(f"  Best score: [bold]{result.best_score:.1f}/10[/] (iteration {result.best_iteration})")

        # Build detailed score table
        if result.iterations:
            _print_score_table(result)

        # Show best prompt
        best = result.best_result
        if best:
            console.print(f"\n[bold]Best Prompt (iteration {best.iteration}):[/]")
            if best.prompt.reasoning:
                console.print(f"  [dim]Reasoning: {best.prompt.reasoning}[/]")
            console.print(f"\n  [bold]System prompt:[/]")
            console.print(f"  {best.prompt.system_prompt[:500]}{'...' if len(best.prompt.system_prompt) > 500 else ''}")
            console.print(f"\n  [bold]Summary:[/]")
            console.print(f"  {best.summary}")

        # Save best prompt to YAML
        if save_best and best:
            _save_best_prompt(save_best, result)

    except ValueError as e:
        console.print(f"\n[red]Error:[/] {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted[/]")
        sys.exit(130)
    except Exception as e:
        console.print(f"\n[red]Error:[/] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def _print_score_table(result: object) -> None:
    """Print a Rich table with per-iteration, per-criterion scores."""
    from shared.ai.prompt_improver import RALPHResult
    assert isinstance(result, RALPHResult)

    if not result.iterations:
        return

    # Get criteria from first iteration that has scores
    criteria_names: list[str] = []
    for it in result.iterations:
        if it.scores.scores:
            criteria_names = list(it.scores.scores.keys())
            break

    if not criteria_names:
        return

    # Build table
    table = Table(title="Iteration Scores", show_lines=True)
    table.add_column("Iter", justify="center", style="bold", width=6)
    table.add_column("Avg", justify="center", style="bold", width=6)

    for criterion in criteria_names:
        table.add_column(criterion, justify="center", width=max(len(criterion), 5))

    table.add_column("Time", justify="right", width=8)

    for it in result.iterations:
        is_best = it.iteration == result.best_iteration
        iter_label = f"{it.iteration} {'★' if is_best else ' '}"

        avg_str = f"{it.avg_score:.1f}"
        if it.avg_score >= 8.0:
            avg_str = f"[green]{avg_str}[/]"
        elif it.avg_score >= 6.0:
            avg_str = f"[yellow]{avg_str}[/]"
        else:
            avg_str = f"[red]{avg_str}[/]"

        criterion_values = []
        for c in criteria_names:
            score = it.scores.scores.get(c, 0)
            if score >= 8:
                criterion_values.append(f"[green]{score}[/]")
            elif score >= 6:
                criterion_values.append(f"[yellow]{score}[/]")
            else:
                criterion_values.append(f"[red]{score}[/]")

        time_str = f"{it.summary_time + it.eval_time:.1f}s"

        table.add_row(iter_label, avg_str, *criterion_values, time_str)

    console.print(table)


def _save_best_prompt(path: str, result: object) -> None:
    """Save the best prompt to a YAML file."""
    from shared.ai.prompt_improver import RALPHResult
    assert isinstance(result, RALPHResult)

    best = result.best_result
    if not best:
        return

    output = {
        "element_type": result.element_type,
        "element_name": result.element_name,
        "element_id": result.element_id,
        "best_iteration": result.best_iteration,
        "best_score": round(result.best_score, 2),
        "converged": result.converged,
        "stop_reason": result.stop_reason,
        "prompt": best.prompt.to_dict(),
        "summary": best.summary,
        "scores": {
            "average": round(best.avg_score, 2),
            "criteria": dict(best.scores.scores),
            "notes": best.scores.notes,
        },
        "all_iterations": [
            {
                "iteration": it.iteration,
                "avg_score": round(it.avg_score, 2),
                "scores": dict(it.scores.scores),
                "summary": it.summary,
                "reasoning": it.prompt.reasoning,
            }
            for it in result.iterations
        ],
    }

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.dump(output, default_flow_style=False, sort_keys=False, allow_unicode=True))
    console.print(f"\n[bold green]Best prompt saved to:[/] {out_path}")
