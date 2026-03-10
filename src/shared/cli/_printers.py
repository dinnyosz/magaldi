"""Output formatting and printing functions for CLI.

This module contains all the print_* functions that format
and display results from various phases.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.table import Table

from shared.cli._shared import console, format_duration

if TYPE_CHECKING:
    from magaldi_core.change_detection import ChangeManifest
    from magaldi_core.code_parser import ParsingResult
    from magaldi_core.discovery import DiscoveryResult
    from magaldi_core.processor import TimingStats
    from magaldi_core.variable_scoring.models import ScoringResult
    from shared.cli.parse_logger import ParseRunLogger


def print_discovery_result(result: DiscoveryResult) -> None:
    """Print discovery phase results."""
    langs = ", ".join(f"{lang}({s.files})" for lang, s in sorted(result.languages.items(), key=lambda x: -x[1].files))
    console.print(f"  {result.scope}/{result.repository} @{result.username} | {result.total_files} files, {result.total_lines:,} lines | {langs}")


def print_change_manifest(manifest: ChangeManifest) -> None:
    """Print change detection results."""
    parts = [f"scanned {manifest.total_files_scanned}"]
    if len(manifest.new_files):
        parts.append(f"[green]+{len(manifest.new_files)} new[/]")
    if len(manifest.modified_files):
        parts.append(f"[yellow]{len(manifest.modified_files)} modified[/]")
    if len(manifest.deleted_files):
        parts.append(f"[red]-{len(manifest.deleted_files)} del[/]")
    if manifest.unchanged_count:
        parts.append(f"{manifest.unchanged_count} unchanged")
    if manifest.skipped_count:
        parts.append(f"skipped {manifest.skipped_count}")
    console.print(f"  {' | '.join(parts)}")


def print_parsing_result(result: ParsingResult, *, skip_ai: bool = False) -> None:
    """Print parsing results including context size analysis."""
    # Existing summary line
    types = ", ".join(f"{t}: [green]{c}[/]" for t, c in sorted(result.elements_by_type.items()))
    failed = f" | [red]{len(result.failed_files)} failed[/]" if result.failed_files else ""
    console.print(f"  [green]{len(result.parsed_files)}[/] files → [green]{result.total_elements}[/] elements ({types}){failed}")

    # Per-tier context size summary (compact one-liner)
    # Skip when --skip-ai: tiers are for LLM KV cache optimization, irrelevant without AI
    if not skip_ai:
        tiers = result.elements_by_tier
        non_empty_tiers = {tier: stats for tier, stats in tiers.items() if stats["count"] > 0}

        if non_empty_tiers:
            tier_parts = []
            for tier in sorted(non_empty_tiers.keys()):
                count = non_empty_tiers[tier]["count"]
                tier_parts.append(f"[cyan]{tier // 1024}k[/]:{count}")
            console.print(f"  [dim]Context tiers:[/] {' [dim]|[/] '.join(tier_parts)}")


def print_feature_result(result: dict) -> None:
    """Print feature extraction results."""
    if not result:
        return

    coverage = result.get("elements_covered", 0)
    total = result.get("total_elements", 0)
    pct = result.get("coverage_pct", 0)
    failed = result.get("failed", 0)
    elapsed = result.get("elapsed", 0)
    subfeatures = result.get("subfeatures_created", 0)

    # Summary line
    parts = [
        f"[green]{result['cluster_count']} features[/]",
    ]
    if subfeatures > 0:
        parts.append(f"[yellow]{subfeatures} sub-features[/]")
    parts.append(f"covering {coverage}/{total} ({pct:.0f}%)")
    if failed > 0:
        parts.append(f"[red]{failed} failed[/]")
    if elapsed > 0:
        parts.append(f"{format_duration(elapsed)} elapsed")
    console.print(f"  {' | '.join(parts)}")

    # Show top features with their labels (sorted by element count)
    if result.get("clusters"):
        console.print("  [dim]Top features (by element count):[/]")
        sorted_clusters = sorted(result["clusters"], key=lambda c: c.get("size", 0), reverse=True)
        for cluster in sorted_clusters[:5]:
            label = cluster.get("label") or f"feature_{cluster['cluster_id']}"
            names = ", ".join(cluster.get("sample_names", [])[:3])
            console.print(f"    [cyan]{label}[/] ({cluster['size']} elements): {names}...")


def print_scoring_result(result: ScoringResult) -> None:
    """Print variable scoring results."""
    if result.total_variables == 0:
        console.print("  [dim]No variables to score[/]")
        return

    parts = [
        f"[green]{result.kept} kept[/]",
        f"[red]{result.dropped} dropped[/]",
        f"of {result.total_variables} variables",
        f"in {result.batch_count} batches",
    ]
    if result.elapsed > 0:
        parts.append(f"{format_duration(result.elapsed)} elapsed")
    if result.errors > 0:
        parts.append(f"[yellow]{result.errors} errors[/]")

    console.print(f"  {' | '.join(parts)}")


def print_processing_result(
    processed: int, skipped: int, indexed: int, skip_ai: bool,
    avg_wall: float = 0.0, avg_summ: float = 0.0, avg_embed: float = 0.0, elapsed: float = 0.0,
    _timing_stats: TimingStats | None = None, _num_workers: int = 4,
    deleted: int = 0
) -> None:
    """Print processing results."""
    parts = []
    if deleted:
        parts.append(f"[red]{deleted} deleted[/]")
    if processed:
        parts.append(f"[green]{processed} processed[/]")
    if skipped:
        parts.append(f"{skipped} unchanged")
    if indexed:
        parts.append(f"{indexed} indexed")
    if skip_ai:
        parts.append("[yellow]AI skipped[/]")
    if elapsed > 0:
        parts.append(f"{format_duration(elapsed)} elapsed")
        # Effective wall time = elapsed / items that made API calls (not counting skipped)
        api_processed = processed - skipped
        if api_processed > 0:
            effective = elapsed / api_processed
            parts.append(f"{effective:.2f}s/item effective")
    if avg_wall > 0:
        parts.append(f"avg: {avg_wall:.1f}s total, {avg_summ:.1f}s summ, {avg_embed:.1f}s embed")
    console.print(f"  {' | '.join(parts)}")


def print_summary(
    discovery: DiscoveryResult,
    manifest: ChangeManifest,
    processed: int,
    indexed: int,
    skip_ai: bool,
) -> None:
    """Print final summary."""
    console.print("\n" + "=" * 60)
    console.print("[bold green]Parse Complete[/]")
    console.print("=" * 60)

    table = Table(show_header=False, box=None)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Repository", f"{discovery.scope}/{discovery.repository}")
    table.add_row("User", discovery.username)
    table.add_row("Files parsed", str(manifest.files_to_parse))
    table.add_row("Elements processed", str(processed))
    table.add_row("Elements indexed", str(indexed))

    if skip_ai:
        table.add_row("AI processing", "Skipped")

    console.print(table)
    console.print()


def print_phase_timings(run_logger: ParseRunLogger) -> None:
    """Print phase timing breakdown table at end of parse run."""
    timings = run_logger.get_phase_timings()
    if not timings:
        return

    total_elapsed = run_logger.get_total_elapsed()
    token_totals = run_logger.get_token_totals()
    has_tokens = token_totals["count"] > 0

    console.print()
    table = Table(show_header=True, box=None, padding=(0, 2))
    table.add_column("Phase", style="cyan", min_width=30)
    table.add_column("Duration", style="green", justify="right", min_width=10)
    table.add_column("%", style="dim", justify="right", min_width=6)
    if has_tokens:
        table.add_column("In tokens", style="yellow", justify="right", min_width=10)
        table.add_column("Out tokens", style="magenta", justify="right", min_width=10)

    for name, duration in timings:
        pct = (duration / total_elapsed * 100) if total_elapsed > 0 else 0
        if has_tokens:
            # Look up token usage for this phase
            phase_usage = run_logger._token_usage.get(name, {})
            phase_totals = phase_usage.get("totals", {})
            in_tok = phase_totals.get("input", 0)
            out_tok = phase_totals.get("output", 0)
            in_str = f"{in_tok:,}" if in_tok > 0 else "[dim]—[/]"
            out_str = f"{out_tok:,}" if out_tok > 0 else "[dim]—[/]"
            table.add_row(name, format_duration(duration), f"{pct:.0f}%", in_str, out_str)
        else:
            table.add_row(name, format_duration(duration), f"{pct:.0f}%")

    # Total row
    if has_tokens:
        table.add_row("─" * 30, "─" * 10, "─" * 6, "─" * 10, "─" * 10)
        table.add_row(
            "[bold]Total[/]",
            f"[bold]{format_duration(total_elapsed)}[/]",
            "",
            f"[bold]{token_totals['input']:,}[/]",
            f"[bold]{token_totals['output']:,}[/]",
        )
    else:
        table.add_row("─" * 30, "─" * 10, "─" * 6)
        table.add_row("[bold]Total[/]", f"[bold]{format_duration(total_elapsed)}[/]", "")

    console.print(table)

    # Per-model token usage table (one row per model + totals)
    model_totals = run_logger.get_model_totals()
    if model_totals:
        console.print()
        model_table = Table(show_header=True, box=None, padding=(0, 2))
        model_table.add_column("Model", style="cyan", min_width=30)
        model_table.add_column("Elements", style="dim", justify="right", min_width=10)
        model_table.add_column("In tokens", style="yellow", justify="right", min_width=12)
        model_table.add_column("Out tokens", style="magenta", justify="right", min_width=12)

        grand_input = 0
        grand_output = 0
        grand_count = 0
        for model, data in sorted(model_totals.items()):
            model_table.add_row(
                model,
                f"{data['count']:,}",
                f"{data['input']:,}",
                f"{data['output']:,}",
            )
            grand_input += data["input"]
            grand_output += data["output"]
            grand_count += data["count"]

        model_table.add_row("─" * 30, "─" * 10, "─" * 12, "─" * 12)
        model_table.add_row(
            "[bold]Total[/]",
            f"[bold]{grand_count:,}[/]",
            f"[bold]{grand_input:,}[/]",
            f"[bold]{grand_output:,}[/]",
        )
        console.print(model_table)
