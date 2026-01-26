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

    # Context size analysis table
    max_chars = result.max_chars_by_type
    context_sizes = result.context_sizes
    largest_elements = result.largest_elements_by_type

    if max_chars:
        console.print()
        console.print("  [dim]Context size analysis (for KV cache optimization):[/]")
        console.print("  [dim]  Element Type   Max Chars   Est. Tokens   Context Size   Largest Element[/]")
        console.print("  [dim]  ────────────────────────────────────────────────────────────────────────────────────────────────[/]")

        # Sort by context size descending for readability
        sorted_types = sorted(max_chars.keys(), key=lambda t: context_sizes.get(t, 0), reverse=True)

        from shared.ai.context_size import PROMPT_OVERHEAD, DEFAULT_OVERHEAD

        for element_type in sorted_types:
            chars = max_chars[element_type]
            overhead = PROMPT_OVERHEAD.get(element_type, DEFAULT_OVERHEAD)
            tokens = chars // 4 + overhead
            ctx_size = context_sizes.get(element_type, 0)
            # Get largest element info
            largest = largest_elements.get(element_type)
            if largest:
                name, path, _ = largest
                # Truncate path if too long
                if len(path) > 30:
                    path = "..." + path[-27:]
                largest_info = f"{path}:{name}"
            else:
                largest_info = "-"
            console.print(f"  [dim]  {element_type:<14} {chars:>9,}   {tokens:>11,}   {ctx_size:>12,}   {largest_info}[/]")


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
