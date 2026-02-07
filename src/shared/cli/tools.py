"""Magaldi Tools CLI - Standalone utility commands.

Entry point: magaldi-tools <command> [options]

Commands:
- passport-benchmark: Benchmark semantic passport embedding formats
"""

from __future__ import annotations

import click


@click.group()
def tools_cli() -> None:
    """Magaldi development and benchmarking tools."""


@tools_cli.command("passport-benchmark")
@click.option("--scope", default=None, help="Repository scope (auto-detected from magaldi.yaml)")
@click.option(
    "--repository", default=None, help="Repository name (auto-detected from magaldi.yaml)"
)
@click.option("--n-callers", default=50, type=int, help="Number of callers to sample")
@click.option("--n-negatives", default=200, type=int, help="Number of negative distractors")
@click.option("--seed", default=42, type=int, help="Random seed")
@click.option("--batch-size", default=20, type=int, help="Embedding batch size")
@click.option("--variants", default=None, help="Comma-separated variant names (default: all)")
@click.option(
    "--output-dir", default="plans/benchmarks/data", help="Output directory for markdown report"
)
def passport_benchmark(
    scope: str | None,
    repository: str | None,
    n_callers: int,
    n_negatives: int,
    seed: int,
    batch_size: int,
    variants: str | None,
    output_dir: str,
) -> None:
    """Benchmark semantic passport embedding formats for call resolution.

    Compares different "passport" text formats for code element embeddings,
    measuring how well each format's cosine similarity predicts actual
    caller-callee relationships.
    """
    from shared.cli.benchmark_passport import _auto_detect_repo, run_benchmark

    # Auto-detect scope/repository if not provided
    if not scope or not repository:
        auto_scope, auto_repo = _auto_detect_repo()
        scope = scope or auto_scope
        repository = repository or auto_repo

    if not scope or not repository:
        click.echo(
            "Error: Could not determine scope/repository. "
            "Provide --scope and --repository, or run from a directory with magaldi.yaml",
            err=True,
        )
        raise SystemExit(1)

    variant_list = None
    if variants:
        variant_list = [v.strip() for v in variants.split(",")]

    run_benchmark(
        scope=scope,
        repository=repository,
        n_callers=n_callers,
        n_negatives=n_negatives,
        seed=seed,
        batch_size=batch_size,
        variants=variant_list,
        output_dir=output_dir,
    )


def main() -> None:
    """Entry point for magaldi-tools."""
    tools_cli()
