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


def print_discovery_result(result: "DiscoveryResult") -> None:
    """Print discovery phase results."""
    langs = ", ".join(f"{lang}({s.files})" for lang, s in sorted(result.languages.items(), key=lambda x: -x[1].files))
    console.print(f"  {result.scope}/{result.repository} @{result.username} | {result.total_files} files, {result.total_lines:,} lines | {langs}")


def print_change_manifest(manifest: "ChangeManifest") -> None:
    """Print change detection results."""
    parts = [f"scanned {manifest.total_files_scanned}"]
    if len(manifest.new_files): parts.append(f"[green]+{len(manifest.new_files)} new[/]")
    if len(manifest.modified_files): parts.append(f"[yellow]{len(manifest.modified_files)} modified[/]")
    if len(manifest.deleted_files): parts.append(f"[red]-{len(manifest.deleted_files)} del[/]")
    if manifest.unchanged_count: parts.append(f"{manifest.unchanged_count} unchanged")
    if manifest.skipped_count: parts.append(f"skipped {manifest.skipped_count}")
    console.print(f"  {' | '.join(parts)}")


def print_parsing_result(result: "ParsingResult") -> None:
    """Print parsing results including context size analysis."""
    # Existing summary line
    types = ", ".join(f"{t}: [green]{c}[/]" for t, c in sorted(result.elements_by_type.items()))
    failed = f" | [red]{len(result.failed_files)} failed[/]" if result.failed_files else ""
    console.print(f"  [green]{len(result.parsed_files)}[/] files → [green]{result.total_elements}[/] elements ({types}){failed}")

    # Per-tier context size analysis
    tiers = result.elements_by_tier
    non_empty_tiers = {tier: stats for tier, stats in tiers.items() if stats["count"] > 0}

    if non_empty_tiers:
        total_elements = sum(s["count"] for s in non_empty_tiers.values())

        console.print()
        console.print("  [dim]Context tiers (per-element KV cache optimization):[/]")

        # Main tier table
        tier_table = Table(show_header=True, header_style="dim", box=None, padding=(0, 2))
        tier_table.add_column("Tier", justify="right", style="cyan")
        tier_table.add_column("Count", justify="right", style="green")
        tier_table.add_column("%", justify="right", style="yellow")
        tier_table.add_column("Max Tokens", justify="right", style="dim")
        # Add columns for each element type
        all_types = ["file", "class", "function", "method", "variable", "constant"]
        for etype in all_types:
            tier_table.add_column(etype, justify="right", style="dim")

        # Sort by tier size ascending
        for tier in sorted(non_empty_tiers.keys()):
            stats = non_empty_tiers[tier]
            count = stats["count"]
            pct = count / total_elements * 100 if total_elements > 0 else 0
            max_tokens = stats["max_tokens"]
            by_type = stats["by_type"]

            # Build row
            row = [
                f"{tier // 1024}k",
                f"{count:,}",
                f"{pct:.1f}%",
                f"{max_tokens:,}",
            ]
            # Add count for each type
            for etype in all_types:
                type_count = by_type.get(etype, 0)
                row.append(str(type_count) if type_count > 0 else "-")

            tier_table.add_row(*row)

        console.print(tier_table)

        # Largest elements per tier (separate section)
        console.print()
        console.print("  [dim]Largest elements per tier:[/]")
        for tier in sorted(non_empty_tiers.keys()):
            stats = non_empty_tiers[tier]
            largest = stats["largest"]
            if largest:
                name, path, chars, etype = largest
                tier_label = f"{tier // 1024}k"
                # Show full path, truncate from start if too long
                display_path = path if len(path) <= 60 else "..." + path[-57:]
                console.print(f"    [cyan]{tier_label:>3}[/] [yellow]{display_path}[/]")
                console.print(f"        [green]{name}[/] [dim]({etype}, {chars:,} chars)[/]")


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


def print_processing_result(
    processed: int, skipped: int, indexed: int, skip_ai: bool,
    avg_wall: float = 0.0, avg_summ: float = 0.0, avg_embed: float = 0.0, elapsed: float = 0.0,
    timing_stats: "TimingStats | None" = None, num_workers: int = 4,
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
    discovery: "DiscoveryResult",
    manifest: "ChangeManifest",
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
