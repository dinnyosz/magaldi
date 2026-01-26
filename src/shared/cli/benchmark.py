"""Benchmark command for the Magaldi CLI.

This module contains the benchmark-models command for comparing LLM performance.
"""

from __future__ import annotations

# Import from sibling modules within the package
from shared.cli._shared import console, format_duration, main
from shared.cli._runners import run_discovery, run_parsing
from shared.cli._printers import print_discovery_result, print_parsing_result

import sys
from collections import defaultdict
from typing import TYPE_CHECKING

import click
from rich.table import Table

from shared.config import load_config

if TYPE_CHECKING:
    from magaldi_core.change_detection import ChangeManifest
    from magaldi_core.code_parser import ParsingResult
    from magaldi_core.discovery import DiscoveryResult
    from shared.config import BenchmarkConfig, MagaldiConfig, ModelConfig
    from shared.ai.ollama_benchmark import BenchmarkResult, EvaluationResult


# =============================================================================
# BENCHMARK HELPER FUNCTIONS
# =============================================================================


def _parse_model_spec_to_config(spec: str, ollama_url: str | None = None) -> "ModelConfig":
    """Parse CLI model specification into a ModelConfig.

    Formats:
        "qwen3:4b" → Ollama model (default)
        "lmstudio:xxx" → LM Studio model
        "openai:gpt-4" → OpenAI model
    """
    from shared.config import ModelConfig

    # Check for explicit provider prefix
    for prefix in ["lmstudio:", "openai:", "anthropic:", "vllm:", "llamacpp:"]:
        if spec.startswith(prefix):
            provider = prefix[:-1]
            model_name = spec[len(prefix):]
            if provider == "lmstudio":
                return ModelConfig(
                    name=model_name,
                    provider="lmstudio",
                    url="http://localhost:1234",
                    api_key="lm-studio",
                )
            elif provider == "openai":
                return ModelConfig(name=model_name, provider="openai")
            else:
                return ModelConfig(name=model_name, provider=provider)

    # Default to Ollama with default URL
    return ModelConfig(
        name=spec,
        provider="ollama",
        url=ollama_url or "http://localhost:11434",
    )


def _get_model_api_config(model_config: "ModelConfig") -> dict:
    """Get api_base and api_key for a ModelConfig.

    Returns dict with 'api_base' and 'api_key' keys for LiteLLM.
    """
    result = {}
    api_base = model_config.get_api_base()
    if api_base:
        result["api_base"] = api_base
    if model_config.api_key:
        result["api_key"] = model_config.api_key
    return result


def _check_backend_connections(
    model_configs: list["ModelConfig"],
) -> tuple[dict[str, list[str]], list["ModelConfig"], list["ModelConfig"]]:
    """Check backend connections and determine which models are available.

    Args:
        model_configs: List of model configurations to check.

    Returns:
        Tuple of (available_models_by_provider, models_to_test, missing_models).
    """
    from shared.ai.ollama_benchmark import BenchmarkClient

    # Group models by provider to check each backend once
    models_by_provider: dict[str, list["ModelConfig"]] = defaultdict(list)
    for mc in model_configs:
        models_by_provider[mc.provider].append(mc)

    # Check each provider's connection
    available_models_by_provider: dict[str, list[str]] = {}

    for provider, provider_models in models_by_provider.items():
        url = provider_models[0].url

        if provider == "ollama":
            client = BenchmarkClient(api_base=url)
            if client.check_connection():
                available = client.list_models()
                available_models_by_provider[provider] = available
                console.print(f"  [green]✓[/] Ollama ({url}) | {len(available)} models")
            else:
                console.print(f"[red]Cannot connect to Ollama at {url}[/]")
                available_models_by_provider[provider] = []

        elif provider in ("lmstudio", "llamacpp"):
            try:
                import requests
                api_url = f"{url.rstrip('/')}/v1/models"
                resp = requests.get(api_url, timeout=5)
                if resp.status_code == 200:
                    data = resp.json()
                    available = [m["id"] for m in data.get("data", [])]
                    available_models_by_provider[provider] = available
                    console.print(f"  [green]✓[/] {provider} ({url}) | {len(available)} models")
                else:
                    console.print(f"  [yellow]![/] {provider} ({url}) | Cannot list models (status {resp.status_code})")
                    available_models_by_provider[provider] = []
            except Exception as e:
                console.print(f"  [yellow]![/] {provider} ({url}) | Connection failed: {e}")
                available_models_by_provider[provider] = []

        else:
            # OpenAI, Anthropic, etc. - trust they're available
            available_models_by_provider[provider] = ["*"]
            console.print(f"  [green]✓[/] {provider} (cloud API)")

    # Check availability of each configured model
    missing_models: list["ModelConfig"] = []
    models_to_test: list["ModelConfig"] = []

    for mc in model_configs:
        available = available_models_by_provider.get(mc.provider, [])

        # Cloud providers - trust availability
        if available == ["*"]:
            models_to_test.append(mc)
            continue

        # Check if model is in available list
        if mc.name in available or f"{mc.name}:latest" in available:
            models_to_test.append(mc)
        else:
            # Try without tag
            base = mc.name.rsplit(":", 1)[0] if ":" in mc.name else mc.name
            if base in available or f"{base}:latest" in available:
                models_to_test.append(mc)
            else:
                missing_models.append(mc)

    return available_models_by_provider, models_to_test, missing_models


def _warmup_benchmark_models(
    models_to_test: list["ModelConfig"],
    benchmark_config: "BenchmarkConfig",
) -> list["ModelConfig"]:
    """Warm up models and return the list of models that succeeded.

    Args:
        models_to_test: List of model configurations to warm up.
        benchmark_config: Benchmark configuration.

    Returns:
        List of models that successfully warmed up.
    """
    from shared.ai.ollama_benchmark import BenchmarkClient

    models_failed_warmup: list["ModelConfig"] = []
    for mc in models_to_test:
        display_name = f"{mc.provider}/{mc.name}"
        api_config = _get_model_api_config(mc)
        api_model = mc.get_litellm_model()
        with console.status(f"[bold blue]Warming up {display_name}...[/]"):
            warmup_client = BenchmarkClient(api_base=mc.url)
            success, warmup_time, error = warmup_client.warmup(
                api_model,
                timeout=120,
                **api_config,
            )
        if success:
            console.print(f"  [green]✓[/] {display_name} ({warmup_time:.1f}s)")
        else:
            console.print(f"  [red]✗[/] {display_name}: {error}")
            models_failed_warmup.append(mc)

    # Remove failed models
    return [mc for mc in models_to_test if mc not in models_failed_warmup]


def _run_hierarchical_benchmarks(
    models_to_test: list["ModelConfig"],
    elements: list,
    repo_path: str,
    benchmark_config: "BenchmarkConfig",
) -> dict[str, list["BenchmarkResult"]]:
    """Run benchmarks with hierarchical context (file → class → method).

    Args:
        models_to_test: List of model configurations to test.
        elements: List of code elements to benchmark.
        repo_path: Path to the repository.
        benchmark_config: Benchmark configuration.

    Returns:
        Dict mapping model display name to list of benchmark results.
    """
    from pathlib import Path

    from shared.ai.ollama_benchmark import BenchmarkClient
    from shared.ai.summarization import clean_summary

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
            prompt = _build_summarization_prompt(elem, parent_summaries)

            # For thinking models, add directive to skip thinking
            if any(mc.name.startswith(tm) for tm in BenchmarkClient.THINKING_MODELS):
                prompt = prompt + "\n\n/no_think"

            with console.status(f"    [{len(model_results)+1}/{len(elements)}] {elem_name}..."):
                model_params = benchmark_config.get_model_params(mc.name)
                max_tokens = model_params.max_tokens or benchmark_config.max_tokens
                api_config = _get_model_api_config(mc)
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


def _evaluate_summaries(
    models_to_test: list["ModelConfig"],
    elements: list,
    results: dict[str, list["BenchmarkResult"]],
    available_models_by_provider: dict[str, list[str]],
    benchmark_config: "BenchmarkConfig",
) -> dict[int, dict[str, "EvaluationResult"]]:
    """Evaluate summaries using LLM judges.

    Args:
        models_to_test: List of model configurations that were tested.
        elements: List of code elements.
        results: Dict mapping model display name to benchmark results.
        available_models_by_provider: Dict of available models per provider.
        benchmark_config: Benchmark configuration.

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
            eval_api_config = _get_model_api_config(eval_mc)
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


def _create_full_manifest(discovery_result: "DiscoveryResult") -> "ChangeManifest":
    """Create a ChangeManifest with all discovered files as 'new'.

    This skips change detection and treats all files as new.
    """
    import hashlib
    import os
    from datetime import datetime as dt
    from pathlib import Path

    from magaldi_core.change_detection import ChangeManifest, FileInfo
    from magaldi_core.discovery import SUPPORTED_EXTENSIONS, _is_excluded_dir, _is_excluded_file

    new_files = []
    repo_path = discovery_result.repo_path

    for root, dirs, files in os.walk(repo_path):
        # Filter directories
        dirs[:] = [
            d for d in dirs
            if not _is_excluded_dir(
                d,
                discovery_result.exclude_directories + ["node_modules", ".git", "__pycache__", ".venv", "venv"]
            )
        ]

        for file_name in files:
            abs_path = Path(root) / file_name
            rel_path = str(abs_path.relative_to(repo_path))

            # Check extension
            ext = abs_path.suffix.lower()
            if ext not in SUPPORTED_EXTENSIONS:
                continue

            # Check exclusions
            if _is_excluded_file(file_name, discovery_result.exclude_files):
                continue

            # Compute hash
            try:
                with open(abs_path, "rb") as f:
                    file_hash = hashlib.sha256(f.read()).hexdigest()
            except Exception:
                continue

            new_files.append(FileInfo(
                relative_path=rel_path,
                absolute_path=abs_path,
                language=SUPPORTED_EXTENSIONS[ext],
                hash=file_hash,
            ))

    return ChangeManifest(
        scope=discovery_result.scope,
        repository=discovery_result.repository,
        username=discovery_result.username,
        timestamp=dt.now(),
        total_files_scanned=len(new_files),
        new_files=new_files,
        modified_files=[],
        deleted_files=[],
        unchanged_count=0,
        skipped_count=0,
    )


def _select_benchmark_files(
    parsing_result: "ParsingResult",
    forced_path: str | None,
    num_files: int = 5,
    max_per_type: int = 10,
) -> list[dict] | None:
    """Select files for benchmarking.

    Args:
        parsing_result: Result from parsing phase.
        forced_path: Specific file path to use, or None for random selection.
        num_files: Number of files to select (default 5).
        max_per_type: Maximum elements per type per file (default 10).

    Returns:
        List of dicts with 'path' and 'elements' keys, or None if no valid files found.
    """
    import random

    def cap_elements_by_type(elements: list, max_per_type: int) -> list:
        """Cap elements to max_per_type per element_type per file."""
        by_type: dict[str, list] = defaultdict(list)
        for elem in elements:
            by_type[elem.element_type].append(elem)

        result = []
        for elem_type, type_elements in by_type.items():
            if len(type_elements) > max_per_type:
                # Randomly sample to get variety
                result.extend(random.sample(type_elements, max_per_type))
            else:
                result.extend(type_elements)

        # Sort by line number to maintain order
        result.sort(key=lambda e: e.line_start)
        return result

    parsed_files = parsing_result.parsed_files

    if forced_path:
        # Find the specific file (single file mode)
        for pf in parsed_files:
            path = pf.file_info.relative_path
            if path == forced_path or path.endswith(forced_path):
                elements = cap_elements_by_type(list(pf.elements), max_per_type)
                return [{
                    "path": path,
                    "elements": elements,
                }]
        return None

    # Random selection: prefer files with 3+ elements
    candidates = [
        pf for pf in parsed_files
        if len(pf.elements) >= 3
    ]

    if not candidates:
        # Fall back to any file with elements
        candidates = [pf for pf in parsed_files if len(pf.elements) > 0]

    if not candidates:
        return None

    # Select up to num_files randomly
    selected_files = random.sample(candidates, min(num_files, len(candidates)))

    result = []
    for pf in selected_files:
        elements = cap_elements_by_type(list(pf.elements), max_per_type)
        result.append({
            "path": pf.file_info.relative_path,
            "elements": elements,
        })

    return result


def _save_benchmark_markdown(
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
    import os
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


def _build_summarization_prompt(
    element: "CodeElement",
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


# =============================================================================
# BENCHMARK-MODELS COMMAND
# =============================================================================


@main.command("benchmark-models")
@click.argument("repo_path", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option("--file", "-f", "file_path", default=None, help="Specific file to benchmark (relative path)")
@click.option("--num-files", "-n", default=5, help="Number of random files to select (default: 5)")
@click.option("--max-per-type", default=10, help="Max elements per type per file (default: 10)")
@click.option("--models", "-m", default=None, help="Comma-separated list of Ollama models (default: from config)")
@click.option("--ollama-url", default=None, help="Ollama API URL (default: from config or http://localhost:11434)")
@click.option("--user", "-u", default="benchmark", help="Username for parsing (default: benchmark)")
def benchmark_models(
    repo_path: str,
    file_path: str | None,
    num_files: int,
    max_per_type: int,
    models: str | None,
    ollama_url: str | None,
    user: str,
) -> None:
    """Benchmark Ollama models on code summarization.

    Parses a repository, selects random files (or a specific one), and runs
    summarization benchmarks with detailed timing and LLM-as-judge evaluation.

    By default, selects 5 random files with max 10 elements per type per file.

    REPO_PATH is the path to the repository to benchmark.
    """
    from collections import Counter

    from shared.ai.ollama_benchmark import EVALUATION_CRITERIA
    from shared.ai.summarization import clean_summary

    # Load config for defaults
    config = load_config(skip_validation=True)
    benchmark_config = config.benchmark

    # Build list of ModelConfig objects
    if models:
        # CLI override: parse model specs
        model_configs = [_parse_model_spec_to_config(m.strip(), ollama_url) for m in models.split(",")]
    else:
        # Use models from benchmark config
        model_configs = benchmark_config.get_benchmark_models()

    # For display purposes, create a list of model names
    model_names = [mc.name for mc in model_configs]

    console.print("[bold blue]Magaldi Model Benchmark[/]")
    console.print(f"  Repository: {repo_path}")
    console.print(f"  Models: {', '.join(model_names)}")
    console.print()

    try:
        # Phase 1: Discovery
        console.print("[bold blue]Phase 1:[/] Discovery")
        discovery_result = run_discovery(repo_path, user)
        print_discovery_result(discovery_result)

        # Create manifest with ALL files (skip change detection)
        console.print("\n[bold blue]Phase 2:[/] Creating file manifest (all files)")
        manifest = _create_full_manifest(discovery_result)
        console.print(f"  {manifest.files_to_parse} files to parse")

        if manifest.files_to_parse == 0:
            console.print("\n[red]No supported files found in repository.[/]")
            sys.exit(1)

        # Phase 3: Parsing
        console.print("\n[bold blue]Phase 3:[/] Parsing")
        parsing_result = run_parsing(manifest)
        print_parsing_result(parsing_result)

        # Select files to benchmark
        console.print("\n[bold blue]Phase 4:[/] File Selection")
        selected_files = _select_benchmark_files(parsing_result, file_path, num_files=num_files, max_per_type=max_per_type)
        if not selected_files:
            console.print("\n[red]No valid files found for benchmarking.[/]")
            sys.exit(1)

        # Combine all elements from selected files
        elements = []
        for sf in selected_files:
            elements.extend(sf["elements"])

        console.print(f"  Selected {len(selected_files)} files:")
        for sf in selected_files:
            file_types = Counter(e.element_type for e in sf["elements"])
            type_str = ", ".join(f"{t}: {c}" for t, c in sorted(file_types.items()))
            console.print(f"    [cyan]{sf['path']}[/] ({len(sf['elements'])} elements: {type_str})")

        # Show total element type breakdown
        total_type_counts = Counter(e.element_type for e in elements)
        total_type_summary = ", ".join(f"{t}: {c}" for t, c in sorted(total_type_counts.items()))
        console.print(f"  Total: {len(elements)} elements ({total_type_summary})")

        # Phase 5: Check backend connections
        console.print("\n[bold blue]Phase 5:[/] Backend Connection")
        available_models_by_provider, models_to_test, missing_models = _check_backend_connections(model_configs)

        if missing_models:
            console.print(f"  [yellow]Missing models (skipped):[/]")
            for mc in missing_models:
                console.print(f"    [dim]✗ {mc.provider}/{mc.name}[/]")

        if not models_to_test:
            console.print("\n[red]No models available to test.[/]")
            console.print("[yellow]Pull models with:[/]")
            for mc in missing_models:
                if mc.provider == "ollama":
                    console.print(f"  ollama pull {mc.name}")
            sys.exit(1)

        test_model_names = [f"{mc.provider}/{mc.name}" for mc in models_to_test]
        console.print(f"  Testing models: {', '.join(test_model_names)}")

        # Phase 6: Warmup models
        console.print("\n[bold blue]Phase 6:[/] Model Warmup")
        models_to_test = _warmup_benchmark_models(models_to_test, benchmark_config)

        if not models_to_test:
            console.print("[red]No models available after warmup.[/]")
            sys.exit(1)

        # Phase 7: Run benchmarks with hierarchical context (file → class → method)
        console.print("\n[bold blue]Phase 7:[/] Benchmarking (with hierarchical context)")
        results = _run_hierarchical_benchmarks(models_to_test, elements, repo_path, benchmark_config)

        # For prompts tracking in Phase 8 and summary
        prompts = [(elem, None) for elem in elements]

        # Phase 8: LLM Evaluation of summaries (multi-criteria JSON-based)
        console.print("\n[bold blue]Phase 8:[/] LLM Evaluation (multi-criteria)")
        evaluation_results = _evaluate_summaries(
            models_to_test, elements, results, available_models_by_provider, benchmark_config
        )

        # Get display names for results processing
        tested_model_names = [f"{mc.provider}/{mc.name}" for mc in models_to_test]

        # Get eval models for reporting (same logic as in _evaluate_summaries)
        eval_model_configs = []
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
            except KeyError:
                pass
        if not eval_model_configs:
            eval_model_configs = [models_to_test[-1]]
        eval_display_names = [f"{mc.provider}/{mc.name}" for mc in eval_model_configs]

        # Benchmark summary table (detailed summaries are saved to markdown file only)
        console.print("\n" + "=" * 70)
        console.print("[bold]Benchmark Summary (Multi-Criteria Evaluation)[/]")
        console.print("=" * 70)

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

        # Save results to markdown
        markdown_path = _save_benchmark_markdown(
            repo_path=repo_path,
            models_tested=tested_model_names,
            eval_models=eval_display_names,
            elements=elements,
            results=results,
            evaluation_results=evaluation_results,
        )
        console.print(f"\n[bold green]Results saved to:[/] {markdown_path}")
        console.print("\n[green]Benchmark complete.[/]")

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted[/]")
        sys.exit(130)
    except Exception as e:
        console.print(f"\n[red]Error:[/] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


# =============================================================================
# ENTRY POINT
# =============================================================================


if __name__ == "__main__":
    main()
