"""Benchmark commands for the Magaldi CLI.

This package contains the benchmark-models command for comparing LLM performance
on code summarization tasks.

Modules:
- _helpers: Model configuration and warmup utilities
- _files: File selection and manifest creation
- _execution: Benchmark execution and evaluation
- _results: Results output and markdown generation
"""

from __future__ import annotations

import sys
from collections import Counter
from typing import TYPE_CHECKING

import click

from shared.cli._printers import print_discovery_result, print_parsing_result
from shared.cli._runners import run_discovery, run_parsing
from shared.cli._shared import console, main
from shared.config import load_config

from ._execution import evaluate_summaries, run_hierarchical_benchmarks
from ._files import create_full_manifest, select_benchmark_files
from ._helpers import (
    check_backend_connections,
    parse_model_spec_to_config,
    warmup_benchmark_models,
)
from ._results import display_benchmark_summary, save_benchmark_markdown

if TYPE_CHECKING:
    from shared.config import ModelConfig

__all__ = [
    "benchmark_models",
    # Helpers (for potential reuse)
    "parse_model_spec_to_config",
    "check_backend_connections",
    "warmup_benchmark_models",
    # Files
    "create_full_manifest",
    "select_benchmark_files",
    # Execution
    "run_hierarchical_benchmarks",
    "evaluate_summaries",
    # Results
    "save_benchmark_markdown",
    "display_benchmark_summary",
]


@main.command("benchmark-models")
@click.argument("repo_path", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option("--file", "-f", "file_path", default=None, help="Specific file to benchmark (relative path)")
@click.option("--num-files", "-n", default=5, help="Number of random files to select (default: 5)")
@click.option("--max-per-type", default=10, help="Max elements per type per file (default: 10)")
@click.option("--models", "-m", default=None, help="Comma-separated list of Ollama models (default: from config)")
@click.option("--ollama-url", default=None, help="Ollama API URL (default: from config or http://localhost:11434)")
@click.option("--user", "-u", default="benchmark", help="Username for parsing (default: benchmark)")
@click.option("--skip-warmup", is_flag=True, default=False, help="Skip model warmup phase")
def benchmark_models(
    repo_path: str,
    file_path: str | None,
    num_files: int,
    max_per_type: int,
    models: str | None,
    ollama_url: str | None,
    user: str,
    skip_warmup: bool,
) -> None:
    """Benchmark Ollama models on code summarization.

    Parses a repository, selects random files (or a specific one), and runs
    summarization benchmarks with detailed timing and LLM-as-judge evaluation.

    By default, selects 5 random files with max 10 elements per type per file.

    REPO_PATH is the path to the repository to benchmark.
    """
    # Load config for defaults
    config = load_config(skip_validation=True)
    benchmark_config = config.benchmark

    # Build list of ModelConfig objects
    if models:
        # CLI override: parse model specs
        model_configs = [parse_model_spec_to_config(m.strip(), ollama_url) for m in models.split(",")]
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
        manifest = create_full_manifest(discovery_result)
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
        selected_files = select_benchmark_files(parsing_result, file_path, num_files=num_files, max_per_type=max_per_type)
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
        available_models_by_provider, models_to_test, missing_models = check_backend_connections(model_configs, console)

        if missing_models:
            console.print("  [yellow]Missing models (skipped):[/]")
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

        # Phase 6: Warmup models (optional)
        if skip_warmup:
            console.print("\n[bold blue]Phase 6:[/] Model Warmup [dim](skipped)[/]")
        else:
            console.print("\n[bold blue]Phase 6:[/] Model Warmup")
            models_to_test = warmup_benchmark_models(models_to_test, benchmark_config, console)

            if not models_to_test:
                console.print("[red]No models available after warmup.[/]")
                sys.exit(1)

        # Phase 7: Run benchmarks with hierarchical context (file -> class -> method)
        console.print("\n[bold blue]Phase 7:[/] Benchmarking (with hierarchical context)")
        results = run_hierarchical_benchmarks(models_to_test, elements, repo_path, benchmark_config, console)

        # Phase 8: LLM Evaluation of summaries (multi-criteria JSON-based)
        console.print("\n[bold blue]Phase 8:[/] LLM Evaluation (multi-criteria)")
        evaluation_results = evaluate_summaries(
            models_to_test, elements, results, available_models_by_provider, benchmark_config, console
        )

        # Get display names for results processing
        tested_model_names = [f"{mc.provider}/{mc.name}" for mc in models_to_test]

        # Get eval models for reporting (same logic as in evaluate_summaries)
        eval_model_configs: list[ModelConfig] = []
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

        # Display summary
        console.print("\n" + "=" * 70)
        console.print("[bold]Benchmark Summary (Multi-Criteria Evaluation)[/]")
        console.print("=" * 70)

        display_benchmark_summary(
            models_to_test,
            elements,
            results,
            evaluation_results,
            eval_display_names,
            console,
        )

        # Save results to markdown
        markdown_path = save_benchmark_markdown(
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
