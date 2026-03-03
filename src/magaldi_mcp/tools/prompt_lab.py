"""Prompt Lab tools for RALPH prompt optimization.

Provides MCP tool for iteratively improving summarization prompts.
"""

from __future__ import annotations

from typing import Any


def prompt_lab_improve(
    repo: Any,  # noqa: ARG001 — required by MCP dispatch pattern
    element_id: str,
    max_iterations: int = 5,
    target_score: float = 8.0,
) -> str:
    """Run RALPH prompt optimization loop for an element.

    Args:
        repo: Repository instance (passed by MCP dispatch, not used directly —
              PromptImprover creates its own repo for element retrieval).
        element_id: Element ID or hash prefix.
        max_iterations: Maximum RALPH iterations.
        target_score: Stop when avg score >= this.

    Returns:
        Formatted text report with iteration scores, best prompt, and reasoning.
    """
    from shared.ai.prompt_improver import PromptImprover
    from shared.config import get_config

    config = get_config()
    improver = PromptImprover(config=config)

    result = improver.run(
        element_id=element_id,
        max_iterations=max_iterations,
        target_score=target_score,
    )

    return _format_result(result)


def _format_result(result: Any) -> str:
    """Format RALPHResult as a text report for MCP output."""
    from shared.ai.prompt_improver import RALPHResult
    assert isinstance(result, RALPHResult)

    lines: list[str] = []

    # Header
    lines.append(f"RALPH Prompt Optimization: {result.element_type}:{result.element_name}")
    lines.append(f"Element: {result.element_id}")
    lines.append("")

    if result.converged:
        lines.append(f"✓ Converged: {result.stop_reason}")
    else:
        lines.append(f"⚠ Stopped: {result.stop_reason}")

    lines.append(f"Best score: {result.best_score:.1f}/10 (iteration {result.best_iteration})")
    lines.append("")

    # Iteration details
    lines.append("=" * 60)
    lines.append("ITERATIONS")
    lines.append("=" * 60)

    prev_avg: float | None = None
    prev_scores: dict[str, int] = {}

    for it in result.iterations:
        # Header
        delta_str = ""
        if prev_avg is not None:
            delta = it.avg_score - prev_avg
            delta_str = f" ({'+' if delta >= 0 else ''}{delta:.1f})"
        is_best = it.iteration == result.best_iteration
        star = " ★" if is_best else ""

        lines.append("")
        lines.append(f"--- Iteration {it.iteration}{star} ---")
        lines.append(f"Score: {it.avg_score:.1f}/10{delta_str}")

        # Per-criterion scores with deltas
        if it.scores.scores:
            score_parts = []
            for criterion, s in it.scores.scores.items():
                c_delta = ""
                if criterion in prev_scores:
                    d = s - prev_scores[criterion]
                    if d > 0:
                        c_delta = f"(↑{d})"
                    elif d < 0:
                        c_delta = f"(↓{abs(d)})"
                score_parts.append(f"  {criterion}: {s}/10 {c_delta}")
            lines.append("Criteria:")
            lines.extend(score_parts)

        # Evaluator notes
        if it.scores.notes:
            lines.append(f"Notes: {it.scores.notes}")

        # Improvement reasoning
        if it.iteration > 0 and it.prompt.reasoning:
            lines.append(f"Improvement: {it.prompt.reasoning}")

        # Summary
        if it.summary:
            lines.append(f"Summary: {it.summary}")

        # Timing
        lines.append(f"Time: {it.summary_time:.1f}s gen + {it.eval_time:.1f}s eval")

        prev_avg = it.avg_score
        prev_scores = dict(it.scores.scores)

    # Best prompt
    best = result.best_result
    if best:
        lines.append("")
        lines.append("=" * 60)
        lines.append(f"BEST PROMPT (iteration {best.iteration})")
        lines.append("=" * 60)
        if best.prompt.reasoning and best.iteration > 0:
            lines.append(f"Reasoning: {best.prompt.reasoning}")
        lines.append("")
        lines.append("System prompt:")
        lines.append(best.prompt.system_prompt)
        lines.append("")
        lines.append("User template:")
        lines.append(best.prompt.user_template)
        lines.append("")
        lines.append("Best summary:")
        lines.append(best.summary)

    return "\n".join(lines)
