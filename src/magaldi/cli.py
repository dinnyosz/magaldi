"""Magaldi CLI - Code discovery engine for AI agents and developers.

Commands:
    magaldi parse /path/to/repo --user <username>
"""

from __future__ import annotations

import sys

import click
from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn, ProgressBar
from rich.table import Table
from rich.text import Text

from magaldi.config import MagaldiConfig, load_config
from magaldi.parser.change_detection import (
    ChangeManifest,
    InMemoryFileStateRepository,
    detect_changes,
)
from magaldi.parser.code_parser import ParsingResult, parse_files
from magaldi.parser.discovery import DiscoveryError, DiscoveryResult, discover
from magaldi.processing.processor import (
    ProcessingConfig,
    ProcessingResult,
    ProgressState,
    TimingStats,
    WorkerStatus,
    process_elements,
)

console = Console()


def format_duration(seconds: float) -> str:
    """Format duration as hh:mm:ss or mm:ss."""
    total_secs = int(seconds)
    hours, remainder = divmod(total_secs, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


# =============================================================================
# MAIN CLI GROUP
# =============================================================================


@click.group()
@click.version_option(version="0.1.0", prog_name="magaldi")
def main() -> None:
    """Magaldi - Code discovery engine for AI agents and developers."""
    pass


# =============================================================================
# PARSE COMMAND
# =============================================================================


@main.command()
@click.argument("repo_path", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option("--user", "-u", required=True, help="Username/branch (use 'main' for primary parse)")
@click.option("--skip-ai", is_flag=True, help="Skip AI processing (summarization and embedding)")
@click.option("--dry-run", is_flag=True, help="Use in-memory storage (no database required)")
@click.option("--ollama-url", default=None, help="Ollama API URL (default: from config)")
@click.option("--workers", "-w", default=4, type=int, help="Number of parallel workers (default: 4)")
@click.option("--force-clean", is_flag=True, help="Delete all indexed data for this repo/user before parsing")
def parse(
    repo_path: str, user: str, skip_ai: bool, dry_run: bool,
    ollama_url: str | None, workers: int, force_clean: bool
) -> None:
    """Parse a repository and index its code elements.

    REPO_PATH is the path to the repository to parse.
    """
    # Load configuration (skip validation in dry-run mode)
    config = load_config(skip_validation=dry_run)
    if ollama_url:
        config.ollama.url = ollama_url

    if dry_run:
        console.print("[yellow]Dry run mode:[/] Using in-memory storage\n")

    try:
        # Phase 1: Discovery
        console.print("[bold blue]Phase 1:[/] Discovery")
        discovery_result = run_discovery(repo_path, user)
        print_discovery_result(discovery_result)

        # Force clean: Delete existing index data before change detection
        if force_clean and not dry_run:
            console.print("[yellow]Force clean:[/] Deleting existing index data...")
            from magaldi.db.elasticsearch import ElasticsearchRepository
            es_repo = ElasticsearchRepository(config)
            deleted = es_repo.delete_by_repository(
                scope=discovery_result.scope,
                repository=discovery_result.repository,
                username=user,
            )
            console.print(f"  Deleted {deleted} documents")

        # Phase 2: Change Detection
        console.print("\n[bold blue]Phase 2:[/] Change Detection")
        manifest = run_change_detection(discovery_result, config, dry_run)
        print_change_manifest(manifest)

        if manifest.files_to_parse == 0:
            console.print("\n[green]No files to parse. Repository is up to date.[/]")
            return

        # Phase 3: Parsing
        console.print("\n[bold blue]Phase 3:[/] Parsing")
        parsing_result = run_parsing(manifest)
        print_parsing_result(parsing_result)

        # Phase 4: Processing (summarize -> embed -> index)
        console.print("\n[bold blue]Phase 4:[/] Processing")
        processed, skipped, indexed, avg_wall, avg_summ, avg_embed, elapsed, timing_stats = run_processing(
            parsing_result, manifest, config, dry_run, skip_ai, workers
        )
        print_processing_result(processed, skipped, indexed, skip_ai, avg_wall, avg_summ, avg_embed, elapsed, timing_stats, workers)

        print_summary(discovery_result, manifest, processed, indexed, skip_ai)

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted, cancelling workers...[/]")
        # Force immediate exit - os._exit skips Python cleanup (no threading errors)
        import os
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(130)
    except DiscoveryError as e:
        console.print(f"\n[red]Discovery error:[/] {e}")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[red]Error:[/] {e}")
        if "--dry-run" not in sys.argv:
            console.print("[dim]Hint: Use --dry-run to test without database[/]")
        sys.exit(1)


# =============================================================================
# PHASE RUNNERS
# =============================================================================


def run_discovery(repo_path: str, username: str) -> DiscoveryResult:
    """Run Phase 1: Discovery."""
    with console.status("[bold blue]Discovering repository...[/]"):
        return discover(repo_path, username)


def run_change_detection(
    discovery_result: DiscoveryResult,
    config: MagaldiConfig,
    dry_run: bool,
) -> ChangeManifest:
    """Run Phase 2: Change Detection."""
    if dry_run:
        file_state_repo = InMemoryFileStateRepository()
    else:
        from magaldi.db.elasticsearch import ElasticsearchFileStateRepository
        file_state_repo = ElasticsearchFileStateRepository(config)

    total_files = discovery_result.total_files

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]Hashing files[/]"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("({task.completed}/{task.total})"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("hashing", total=total_files)

        def on_progress(completed: int, total: int) -> None:
            progress.update(task, completed=completed, total=total)

        return detect_changes(discovery_result, file_state_repo, on_progress)


def run_parsing(manifest: ChangeManifest) -> ParsingResult:
    """Run Phase 3: Parsing."""
    total_files = manifest.files_to_parse

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]Parsing files[/]"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("({task.completed}/{task.total})"),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task("parsing", total=total_files)

        def on_progress(completed: int, total: int) -> None:
            progress.update(task, completed=completed, total=total)

        return parse_files(manifest, on_progress)


def run_processing(
    parsing_result: ParsingResult,
    manifest: ChangeManifest,
    config: MagaldiConfig,
    dry_run: bool,
    skip_ai: bool,
    workers: int,
) -> tuple[int, int, int, float, float, float, float, TimingStats | None]:
    """Run unified processing: summarize -> embed -> index.

    Returns:
        Tuple of (processed, skipped, indexed, avg_wall, avg_summ, avg_embed, elapsed, timing_stats).
    """
    if dry_run:
        total = parsing_result.total_elements
        console.print(f"  [dim]Dry run: would process {total} elements[/]")
        return (0, 0, 0, 0.0, 0.0, 0.0, 0.0, None)

    from magaldi.db.elasticsearch import ElasticsearchRepository

    es_repo = ElasticsearchRepository(config)

    proc_config = ProcessingConfig(
        summarize_model=config.ollama.summarize_model,
        embed_model=config.ollama.embed_model,
        ollama_url=config.ollama.url,
        skip_ai=skip_ai,
        num_workers=workers,
    )

    # Build file hashes dict from manifest
    file_hashes: dict[str, str] = {}
    for fi in manifest.new_files:
        file_hashes[fi.relative_path] = fi.hash
    for fi in manifest.modified_files:
        file_hashes[fi.relative_path] = fi.hash

    def build_display(state: ProgressState, num_workers: int) -> RenderableType:
        """Build Rich display from progress state."""
        # Progress info
        pct = (state.completed / state.total * 100) if state.total > 0 else 0
        eta = state.timing.eta_seconds(state.completed, state.total)
        eta_str = f" | ~{format_duration(eta)} ETA" if eta else ""
        elapsed_str = format_duration(state.timing.elapsed)

        # Build visual progress bar
        bar_width = 30
        filled = int(bar_width * pct / 100)
        bar = "━" * filled + "╺" + "─" * (bar_width - filled - 1) if filled < bar_width else "━" * bar_width
        bar_text = Text()
        bar_text.append("  ")
        bar_text.append(bar[:filled], style="green")
        if filled < bar_width:
            bar_text.append(bar[filled:], style="dim")
        bar_text.append(" ")
        bar_text.append(f"{state.completed}", style="green")
        bar_text.append("/", style="dim")
        bar_text.append(f"{state.total}", style="cyan")
        bar_text.append(f" ({pct:.0f}%)", style="green")
        bar_text.append(" | ", style="dim")
        bar_text.append(elapsed_str, style="cyan")
        bar_text.append(" elapsed", style="dim")
        if eta:
            bar_text.append(" | ~", style="dim")
            bar_text.append(format_duration(eta), style="yellow")
            bar_text.append(" ETA", style="dim")

        # Worker table
        worker_table = Table(show_header=False, box=None, padding=(0, 1))
        worker_table.add_column("ID", style="dim", width=4)
        worker_table.add_column("Stage", style="cyan", width=12)
        worker_table.add_column("Element")

        workers_data = state.workers.get_all()
        for wid in range(num_workers):
            if wid in workers_data:
                elem, stage = workers_data[wid]
                worker_table.add_row(f"[{wid}]", stage, elem)
            else:
                worker_table.add_row(f"[{wid}]", "[dim]idle[/]", "")

        # Per-type stats - show remaining count and API time per element
        type_stats = state.timing.get_type_stats()
        type_parts = []
        for t in ["file", "class", "function", "method", "constant", "variable"]:
            if t in type_stats:
                done, tot, avg_wall, avg_summ, avg_embed = type_stats[t]
                remaining = tot - done
                if remaining > 0:
                    api_time = avg_summ + avg_embed
                    type_parts.append(f"[cyan]{t}[/]:[green]{remaining}/{tot}[/] [dim]({api_time:.1f}s)[/]")
        type_line = f"  [dim]Remaining:[/] {' [dim]|[/] '.join(type_parts)}" if type_parts else ""

        # Stats line - effective wall time (elapsed/done) = actual throughput with parallelism
        effective_wall = state.timing.elapsed / state.completed if state.completed > 0 else 0.0
        total_api = state.timing.avg_summarize_time + state.timing.avg_embed_time
        stats = f"  [dim]Throughput:[/] [green]{effective_wall:.2f}s[/]/elem [dim]|[/] [dim]API:[/] [green]{total_api:.1f}s[/]/elem [dim]([/][green]{state.timing.avg_summarize_time:.1f}s[/] summ + [green]{state.timing.avg_embed_time:.1f}s[/] embed[dim])[/]"

        parts: list[RenderableType] = [bar_text, worker_table]
        if type_line:
            parts.append(type_line)
        parts.append(stats)

        return Group(*parts)

    # Create shared state objects that will be updated by workers
    timing_stats = TimingStats()
    worker_status = WorkerStatus()
    total = parsing_result.total_elements

    # Initialize state so display works from the start
    current_state = ProgressState(
        total=total,
        completed=0,
        skipped=0,
        failed=0,
        timing=timing_stats,
        workers=worker_status,
    )

    class LiveDisplay:
        """Wrapper that Rich can call to get current display."""
        def __rich__(self) -> RenderableType:
            return build_display(current_state, workers)

    with Live(LiveDisplay(), console=console, refresh_per_second=10) as live:
        def on_progress(state: ProgressState) -> None:
            nonlocal current_state
            current_state = state
            live.refresh()  # Force refresh on progress

        def on_status_change() -> None:
            """Called when worker status changes."""
            live.refresh()  # Force refresh on status change

        result = process_elements(
            parsing_result.parsed_files,
            manifest.scope,
            manifest.repository,
            manifest.username,
            es_repo,
            proc_config,
            on_progress,
            file_hashes,
            on_status_change,
            worker_status,
            timing_stats,
        )

    # Get timing stats from current state
    avg_summ = current_state.timing.avg_summarize_time
    avg_embed = current_state.timing.avg_embed_time
    elapsed = current_state.timing.elapsed
    # Effective wall time = elapsed / processed (shows throughput with parallelism)
    avg_wall = elapsed / result.elements_processed if result.elements_processed > 0 else 0.0

    return (result.elements_processed, result.elements_skipped, result.indexed, avg_wall, avg_summ, avg_embed, elapsed, timing_stats)


# =============================================================================
# OUTPUT FORMATTING
# =============================================================================


def print_discovery_result(result: DiscoveryResult) -> None:
    """Print discovery phase results."""
    langs = ", ".join(f"{lang}({s.files})" for lang, s in sorted(result.languages.items(), key=lambda x: -x[1].files))
    console.print(f"  {result.scope}/{result.repository} @{result.username} | {result.total_files} files, {result.total_lines:,} lines | {langs}")


def print_change_manifest(manifest: ChangeManifest) -> None:
    """Print change detection results."""
    parts = [f"scanned {manifest.total_files_scanned}"]
    if len(manifest.new_files): parts.append(f"[green]+{len(manifest.new_files)} new[/]")
    if len(manifest.modified_files): parts.append(f"[yellow]~{len(manifest.modified_files)} mod[/]")
    if len(manifest.deleted_files): parts.append(f"[red]-{len(manifest.deleted_files)} del[/]")
    if manifest.unchanged_count: parts.append(f"={manifest.unchanged_count} unchanged")
    if manifest.skipped_count: parts.append(f"skipped {manifest.skipped_count}")
    console.print(f"  {' | '.join(parts)}")


def print_parsing_result(result: ParsingResult) -> None:
    """Print parsing results."""
    types = ", ".join(f"{t}:[green]{c}[/]" for t, c in sorted(result.elements_by_type.items()))
    failed = f" | [red]{len(result.failed_files)} failed[/]" if result.failed_files else ""
    console.print(f"  [green]{len(result.parsed_files)}[/] files → [green]{result.total_elements}[/] elements ({types}){failed}")


def print_processing_result(
    processed: int, skipped: int, indexed: int, skip_ai: bool,
    avg_wall: float = 0.0, avg_summ: float = 0.0, avg_embed: float = 0.0, elapsed: float = 0.0,
    timing_stats: TimingStats | None = None, num_workers: int = 4
) -> None:
    """Print processing results."""
    parts = []
    if processed:
        parts.append(f"[green]{processed} processed[/]")
    if skipped:
        parts.append(f"[dim]{skipped} skipped (already in ES)[/]")
    if indexed:
        parts.append(f"{indexed} indexed")
    if skip_ai:
        parts.append("[yellow]AI skipped[/]")
    if elapsed > 0:
        parts.append(f"{format_duration(elapsed)} elapsed")
        # Effective wall time (elapsed / processed) shows actual throughput
        if processed > 0:
            effective = elapsed / processed
            parts.append(f"{effective:.2f}s/elem effective")
    if avg_wall > 0:
        parts.append(f"avg: {avg_wall:.1f}s total, {avg_summ:.1f}s summ, {avg_embed:.1f}s embed")
    console.print(f"  {' | '.join(parts)}")

    # Show per-type stats
    if timing_stats:
        type_stats = timing_stats.get_type_stats()
        type_parts = []
        for t in ["file", "class", "function", "method", "constant", "variable"]:
            if t in type_stats:
                done, tot, avg_wall, avg_summ, avg_embed = type_stats[t]
                if done > 0:
                    api_time = avg_summ + avg_embed
                    type_parts.append(f"[cyan]{t}[/]: {done} @ {avg_wall:.1f}s wall, {api_time:.1f}s api")
        if type_parts:
            console.print(f"  [dim]Per-type:[/] {' | '.join(type_parts)}")


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


# =============================================================================
# ENTRY POINT
# =============================================================================


if __name__ == "__main__":
    main()
