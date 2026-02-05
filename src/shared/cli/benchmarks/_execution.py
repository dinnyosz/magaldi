"""Benchmark execution and evaluation.

This module contains functions for running benchmarks with hierarchical context
and evaluating summaries using LLM-as-judge.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from shared.ai.ollama_benchmark import BenchmarkResult, EvaluationResult
    from shared.config import BenchmarkConfig, ModelConfig


def build_summarization_prompt(
    element,
    parent_summaries: dict[str, str] | None = None,
) -> str:
    """Build a summarization prompt for an element using the same prompts as parse CLI.

    Args:
        element: CodeElement from parsing.
        parent_summaries: Dict with 'file' and/or 'class' summaries for context.

    Returns:
        Prompt string for summarization.
    """
    from shared.ai.summarization import build_prompt

    if parent_summaries is None:
        parent_summaries = {}

    return build_prompt(element, parent_summaries, max_code_tokens=4000)


def run_hierarchical_benchmarks(
    models_to_test: list["ModelConfig"],
    elements: list,
    repo_path: str,
    benchmark_config: "BenchmarkConfig",
    console,
) -> dict[str, list["BenchmarkResult"]]:
    """Run benchmarks with hierarchical context (file -> class -> method).

    Args:
        models_to_test: List of model configurations to test.
        elements: List of code elements to benchmark.
        repo_path: Path to the repository.
        benchmark_config: Benchmark configuration.
        console: Rich console for output.

    Returns:
        Dict mapping model display name to list of benchmark results.
    """
    from pathlib import Path

    from shared.ai.ollama_benchmark import BenchmarkClient
    from shared.ai.summarization import clean_summary

    from ._helpers import get_model_api_config

    # Sort elements by hierarchy level
    level_order = {"file": 0, "class": 1, "function": 2, "method": 2, "variable": 3, "constant": 3}
    sorted_elements = sorted(elements, key=lambda e: (e.relative_path, level_order.get(e.element_type, 99), e.line_start))

    # Build element index mapping
    element_indices = {id(elem): i for i, elem in enumerate(elements)}

    # Initialize results dict
    results: dict[str, list["BenchmarkResult"]] = {f"{mc.provider}/{mc.name}": [] for mc in models_to_test}

    # Test each model with hierarchical summarization
    for mc in models_to_test:
        display_name = f"{mc.provider}/{mc.name}"
        console.print(f"\n  [cyan]{display_name}[/]")

        # Track summaries for hierarchical context (per model)
        file_summaries: dict[str, str] = {}
        class_summaries: dict[tuple[str, str], str] = {}
        function_summaries: dict[str, str] = {}

        # Process elements in hierarchical order
        model_results: dict[int, "BenchmarkResult"] = {}

        for elem in sorted_elements:
            elem_name = f"{elem.element_type}:{elem.name}"
            elem_idx = element_indices[id(elem)]

            # For file elements, load raw_code from disk if not stored
            if elem.element_type == "file" and not elem.raw_code:
                file_full_path = Path(repo_path) / elem.relative_path
                if file_full_path.exists():
                    try:
                        elem.raw_code = file_full_path.read_text(encoding="utf-8")
                    except Exception:
                        elem.raw_code = ""

            # Build parent summaries based on element type
            parent_summaries: dict[str, str] = {}
            if elem.element_type != "file":
                if elem.relative_path in file_summaries:
                    parent_summaries["file"] = file_summaries[elem.relative_path]
            if elem.element_type in ("method", "variable", "constant"):
                if elem.parent_id:
                    for (fp, cn), summ in class_summaries.items():
                        if fp == elem.relative_path:
                            parent_summaries["class"] = summ
                            break
            if elem.element_type in ("variable", "constant"):
                if elem.parent_id and elem.parent_id in function_summaries:
                    parent_summaries["function"] = function_summaries[elem.parent_id]

            # Build prompt with parent context
            prompt = build_summarization_prompt(elem, parent_summaries)

            # For thinking models, add directive to skip thinking
            if any(mc.name.startswith(tm) for tm in BenchmarkClient.THINKING_MODELS):
                prompt = prompt + "\n\n/no_think"

            with console.status(f"    [{len(model_results)+1}/{len(elements)}] {elem_name}..."):
                model_params = benchmark_config.get_model_params(mc.name)
                max_tokens = model_params.max_tokens or benchmark_config.max_tokens
                api_config = get_model_api_config(mc)
                api_model = mc.get_litellm_model()
                bench_client = BenchmarkClient(api_base=mc.url)
                result = bench_client.generate(
                    model=api_model,
                    prompt=prompt,
                    temperature=model_params.temperature,
                    top_p=model_params.top_p,
                    top_k=model_params.top_k,
                    min_p=model_params.min_p,
                    repetition_penalty=model_params.repetition_penalty,
                    presence_penalty=model_params.presence_penalty,
                    max_tokens=max_tokens,
                    timeout=benchmark_config.timeout,
                    **api_config,
                )
                model_results[elem_idx] = result

            # Store summary for child elements
            if result.success and result.response.strip():
                cleaned = clean_summary(result.response)
                if elem.element_type == "file":
                    file_summaries[elem.relative_path] = cleaned
                elif elem.element_type == "class":
                    class_summaries[(elem.relative_path, elem.name)] = cleaned
                elif elem.element_type in ("function", "method"):
                    function_summaries[elem.element_id] = cleaned

            if result.success:
                context_str = ""
                if parent_summaries:
                    context_str = f" [dim](+{'+'.join(parent_summaries.keys())} context)[/]"
                console.print(
                    f"    [green]✓[/] {elem_name:<40} | "
                    f"[bold]{result.total_time:.2f}s[/] | "
                    f"{result.prompt_chars}→{result.output_chars} chr | "
                    f"{result.prompt_tokens}→{result.output_tokens} tok | "
                    f"{result.tokens_per_second:.0f} t/s{context_str}"
                )
            else:
                console.print(f"    [red]✗[/] {elem_name:<40} | {result.error}")

        # Store results in original element order
        results[display_name] = [model_results[i] for i in range(len(elements))]

    return results


def evaluate_summaries(
    models_to_test: list["ModelConfig"],
    elements: list,
    results: dict[str, list["BenchmarkResult"]],
    available_models_by_provider: dict[str, list[str]],
    benchmark_config: "BenchmarkConfig",
    console,
) -> dict[int, dict[str, "EvaluationResult"]]:
    """Evaluate summaries using LLM judges.

    Args:
        models_to_test: List of model configurations that were tested.
        elements: List of code elements.
        results: Dict mapping model display name to benchmark results.
        available_models_by_provider: Dict of available models per provider.
        benchmark_config: Benchmark configuration.
        console: Rich console for output.

    Returns:
        Dict mapping element index to dict of evaluator name to evaluation result.
    """
    from shared.ai.ollama_benchmark import (
        EVALUATION_CRITERIA,
        BenchmarkClient,
        EvaluationResult,
        build_evaluation_prompt,
        parse_evaluation_response,
    )
    from shared.ai.summarization import clean_summary

    from ._helpers import get_model_api_config

    # Get eval model configs from benchmark config
    eval_model_configs: list["ModelConfig"] = []
    for eval_ref in benchmark_config.eval_models:
        try:
            eval_mc = benchmark_config.get_model(eval_ref)
            available = available_models_by_provider.get(eval_mc.provider, [])
            if available == ["*"] or eval_mc.name in available or f"{eval_mc.name}:latest" in available:
                eval_model_configs.append(eval_mc)
            else:
                base = eval_mc.name.rsplit(":", 1)[0] if ":" in eval_mc.name else eval_mc.name
                if base in available or f"{base}:latest" in available:
                    eval_model_configs.append(eval_mc)
                else:
                    console.print(f"  [yellow]Eval model {eval_ref} not available, skipping[/]")
        except KeyError:
            console.print(f"  [yellow]Eval model {eval_ref} not configured, skipping[/]")

    if not eval_model_configs:
        console.print(f"  [yellow]No eval models available, using last tested model[/]")
        eval_model_configs = [models_to_test[-1]]

    eval_display_names = [f"{mc.provider}/{mc.name}" for mc in eval_model_configs]
    console.print(f"  Using {len(eval_model_configs)} evaluator(s): [cyan]{', '.join(eval_display_names)}[/]")

    # Show criteria for each element type
    prompts = [(elem, None) for elem in elements]
    element_types_in_test = set(elem.element_type for elem, _ in prompts)
    for elem_type in sorted(element_types_in_test):
        criteria = EVALUATION_CRITERIA.get(elem_type, {})
        console.print(f"  [dim]{elem_type} criteria: {', '.join(criteria.keys())}[/]")

    # Initialize results structure
    evaluation_results: dict[int, dict[str, EvaluationResult]] = {
        i: {} for i in range(len(prompts))
    }
    total_elements = len(prompts)
    tested_model_names = [f"{mc.provider}/{mc.name}" for mc in models_to_test]

    for i, (elem, _) in enumerate(prompts):
        elem_name = f"{elem.element_type}:{elem.name}"
        progress = f"[{i+1}/{total_elements}]"

        # Build summaries dict for evaluation
        summaries: dict[str, str] = {}
        for mc in models_to_test:
            model_key = f"{mc.provider}/{mc.name}"
            result = results[model_key][i]
            if result.success and result.response.strip():
                summaries[model_key] = clean_summary(result.response)
            else:
                summaries[model_key] = ""

        # Build evaluation prompt
        eval_prompt = build_evaluation_prompt(
            element_type=elem.element_type,
            element_name=elem.name,
            source_code=elem.raw_code or "",
            summaries=summaries,
        )

        # Evaluate with each eval model
        for eval_mc in eval_model_configs:
            eval_display_name = f"{eval_mc.provider}/{eval_mc.name}"
            eval_api_config = get_model_api_config(eval_mc)
            eval_api_model = eval_mc.get_litellm_model()

            max_retries = 3
            eval_result_obj = EvaluationResult(
                element_type=elem.element_type,
                element_name=elem.name,
            )

            for attempt in range(max_retries):
                with console.status(f"  {progress} [{eval_display_name}] Evaluating {elem_name}..." + (f" (retry {attempt})" if attempt > 0 else "")):
                    eval_client = BenchmarkClient(api_base=eval_mc.url)
                    eval_result = eval_client.generate(
                        model=eval_api_model,
                        prompt=eval_prompt,
                        temperature=0.1 + (attempt * 0.1),
                        max_tokens=1024,
                        timeout=benchmark_config.timeout,
                        **eval_api_config,
                    )

                if eval_result.success:
                    eval_result_obj.raw_response = eval_result.response
                    evaluations, error = parse_evaluation_response(
                        eval_result.response,
                        elem.element_type,
                        tested_model_names,
                    )
                    eval_result_obj.evaluations = evaluations
                    eval_result_obj.parse_error = error

                    if not error:
                        break
                    elif attempt < max_retries - 1:
                        continue
                else:
                    eval_result_obj.parse_error = eval_result.error
                    if attempt < max_retries - 1:
                        continue

            evaluation_results[i][eval_display_name] = eval_result_obj

        # Display scores
        rating_parts = []
        for model_name in tested_model_names:
            model_scores = []
            for eval_name in eval_display_names:
                eval_res = evaluation_results[i].get(eval_name)
                if eval_res and model_name in eval_res.evaluations:
                    model_scores.append(eval_res.evaluations[model_name].average)
            if model_scores:
                avg = sum(model_scores) / len(model_scores)
                rating_parts.append(f"{model_name}: {avg:.1f}")
            else:
                rating_parts.append(f"{model_name}: ?")
        rating_str = " | ".join(rating_parts)

        errors = [
            evaluation_results[i].get(eval_name, EvaluationResult("", "")).parse_error
            for eval_name in eval_display_names
        ]
        has_errors = any(e for e in errors)

        if has_errors:
            console.print(f"  {progress} [yellow]~[/] {elem_name} | {rating_str} [dim](parse issues)[/]")
        else:
            console.print(f"  {progress} [green]✓[/] {elem_name} | {rating_str}")

    return evaluation_results
