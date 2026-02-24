"""Phase runner functions for CLI commands.

This module contains the run_* functions that execute each phase
of the parsing and processing pipeline.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from rich.console import Group, RenderableType
from rich.live import Live
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text

from shared.cli._shared import console, format_duration, get_model_column_width
from shared.cli.extract import build_eta_table

if TYPE_CHECKING:
    from magaldi_core.change_detection import ChangeManifest
    from magaldi_core.code_parser import ParsingResult
    from magaldi_core.discovery import DiscoveryResult
    from magaldi_core.processor import ProgressState, TimingStats
    from magaldi_core.variable_scoring.models import ScoringProgressState, ScoringResult
    from shared.config import MagaldiConfig
    from shared.db.store import Repository


def run_discovery(repo_path: str, username: str, skip_tests: bool = False) -> "DiscoveryResult":
    """Run Phase 1: Discovery."""
    from magaldi_core.discovery import discover

    with console.status("[bold blue]Discovering repository...[/]"):
        return discover(repo_path, username, skip_tests=skip_tests)


def run_change_detection(
    discovery_result: "DiscoveryResult",
    config: "MagaldiConfig",
    dry_run: bool,
) -> "ChangeManifest":
    """Run Phase 2: Change Detection."""
    from magaldi_core.change_detection import (
        InMemoryFileStateRepository,
        detect_changes,
    )

    if dry_run:
        file_state_repo = InMemoryFileStateRepository()
    else:
        from shared.db.store import FileStateRepository
        file_state_repo = FileStateRepository(config)

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


def run_parsing(manifest: "ChangeManifest") -> "ParsingResult":
    """Run Phase 3: Parsing."""
    from magaldi_core.code_parser import parse_files

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


def _build_scoring_display(state: "ScoringProgressState", num_workers: int) -> RenderableType:
    """Build Rich display for variable scoring progress."""
    # Progress bar
    pct = (state.completed_batches / state.total_batches * 100) if state.total_batches > 0 else 0
    eta = state.eta_seconds()
    elapsed = state.elapsed
    elapsed_str = format_duration(elapsed)

    bar_width = 30
    filled = int(bar_width * pct / 100)
    bar = "━" * filled + "╺" + "─" * (bar_width - filled - 1) if filled < bar_width else "━" * bar_width
    bar_text = Text()
    bar_text.append("  ")
    bar_text.append(bar[:filled], style="green")
    if filled < bar_width:
        bar_text.append(bar[filled:], style="dim")
    bar_text.append(" ")
    bar_text.append(f"{state.completed_batches}", style="green")
    bar_text.append("/", style="dim")
    bar_text.append(f"{state.total_batches}", style="cyan")
    bar_text.append(" batches", style="dim")
    bar_text.append(f" ({pct:.0f}%)", style="green")
    bar_text.append(" | ", style="dim")
    bar_text.append(f"{state.completed_variables}", style="green")
    bar_text.append("/", style="dim")
    bar_text.append(f"{state.total_variables}", style="cyan")
    bar_text.append(" vars", style="dim")
    if state.errors > 0:
        bar_text.append(" | ", style="dim")
        bar_text.append(f"{state.errors} errors", style="red")
    bar_text.append(" | ", style="dim")
    bar_text.append(elapsed_str, style="cyan")
    bar_text.append(" elapsed", style="dim")
    if eta is not None and eta > 0:
        bar_text.append(" | ~", style="dim")
        bar_text.append(format_duration(eta), style="yellow")
        bar_text.append(" ETA", style="dim")

    # Score distribution line
    score_text = Text()
    if state.completed_variables > 0:
        keep_pct = (state.kept / state.completed_variables * 100) if state.completed_variables > 0 else 0
        score_text.append("  ")
        score_text.append(f"{state.kept} kept", style="green")
        score_text.append(" | ", style="dim")
        score_text.append(f"{state.dropped} dropped", style="red")
        score_text.append(f" ({keep_pct:.0f}% keep rate)", style="dim")

    # Worker table
    import time as time_mod
    worker_table = Table(show_header=False, box=None, padding=0)
    worker_table.add_column("ID", style="dim", width=4)
    worker_table.add_column("Status", style="cyan", width=10)
    worker_table.add_column("Batch", style="yellow", width=10)
    worker_table.add_column("Size", style="magenta", width=10)
    worker_table.add_column("Time", style="green", width=8)

    workers_data = state.workers.get_all()
    now = time_mod.time()

    for wid in range(num_workers):
        if wid in workers_data:
            batch_num, batch_size, start_time = workers_data[wid]
            worker_elapsed = now - start_time if start_time > 0 else 0
            worker_table.add_row(
                f"[{wid}]",
                "scoring",
                f"#{batch_num}",
                f"{batch_size} vars",
                f"{worker_elapsed:.1f}s",
            )
        else:
            worker_table.add_row(f"[{wid}]", "[dim]idle[/]", "", "", "")

    # Throughput stats
    stats_text = Text()
    if state.completed_batches > 0:
        stats_text.append("  ")
        stats_text.append("Throughput:", style="dim")
        stats_text.append(f" {state.avg_batch_time:.2f}s", style="green")
        stats_text.append("/batch", style="dim")
        stats_text.append(" | ", style="dim")
        stats_text.append(f"{state.avg_variable_time:.2f}s", style="green")
        stats_text.append("/var", style="dim")
        # Vars per second
        vps = state.completed_variables / elapsed if elapsed > 0 else 0
        stats_text.append(" | ", style="dim")
        stats_text.append(f"{vps:.1f}", style="green")
        stats_text.append(" vars/s", style="dim")
        stats_text.append(" | ", style="dim")
        stats_text.append("Workers:", style="dim")
        stats_text.append(f" {state.workers.active_count()}", style="green")
        stats_text.append("/", style="dim")
        stats_text.append(f"{num_workers}", style="cyan")

    parts: list[RenderableType] = [bar_text]
    if score_text.plain:
        parts.append(score_text)
    parts.append(worker_table)
    if stats_text.plain:
        parts.append(stats_text)

    return Group(*parts)


def run_variable_scoring(
    parsing_result: "ParsingResult",
    config: "MagaldiConfig",
    workers: int = 0,
) -> "ScoringResult":
    """Run Phase 4: Variable Scoring.

    Scores all variable/constant elements using the LLM to determine
    which are useful for code discovery. Variables scoring below threshold
    are removed from parsing_result in place.

    Args:
        parsing_result: Result from Phase 3 (modified in place).
        config: Magaldi configuration.
        workers: Max parallel workers (0=auto).

    Returns:
        ScoringResult with statistics.
    """
    from magaldi_core.variable_scoring import score_variables
    from magaldi_core.variable_scoring.models import (
        ScoringProgressState,
        ScoringResult,
        ScoringWorkerStatus,
        VariableScoringConfig,
    )
    from shared.ai.summarization import SummarizationLLMClient

    # Collect all variable/constant elements
    variables: list[tuple[str, str, str, str]] = []
    for pf in parsing_result.parsed_files:
        for elem in pf.elements:
            if elem.element_type in ("variable", "constant"):
                variables.append((
                    elem.element_id,
                    pf.file_info.relative_path,
                    elem.name,
                    elem.raw_code or "",
                ))

    if not variables:
        return ScoringResult()

    # Create LLM client using the small model (scoring is simple)
    model_config = config.llm.get_summarize_model_small()
    llm_client = SummarizationLLMClient(
        url=model_config.url,
        model=model_config.name,
        provider=model_config.provider,
        api_key=model_config.api_key,
    )

    scoring_config = VariableScoringConfig()
    effective_workers = workers if workers > 0 else 12

    # Create shared state for live display
    worker_status = ScoringWorkerStatus()
    progress_state = ScoringProgressState(
        total_variables=len(variables),
        num_workers=effective_workers,
        workers=worker_status,
    )

    class LiveScoringDisplay:
        def __rich__(self) -> RenderableType:
            return _build_scoring_display(current_state, display_workers)

    current_state = progress_state
    display_workers = effective_workers

    def on_progress(state: "ScoringProgressState") -> None:
        nonlocal current_state
        current_state = state

    with Live(LiveScoringDisplay(), console=console, refresh_per_second=4):
        result = score_variables(
            variables=variables,
            llm_client=llm_client._client,
            config=scoring_config,
            max_workers=effective_workers,
            on_progress=on_progress,
            progress_state=progress_state,
            worker_status=worker_status,
        )

    # Remove variables that scored below threshold from parsing_result
    dropped_ids = {
        eid for eid, score in result.scores.items()
        if not score.passes_threshold(scoring_config.threshold)
    }

    if dropped_ids:
        for pf in parsing_result.parsed_files:
            pf.elements = [
                elem for elem in pf.elements
                if elem.element_id not in dropped_ids
            ]

    return result


def run_processing(
    parsing_result: "ParsingResult",
    manifest: "ChangeManifest",
    config: "MagaldiConfig",
    dry_run: bool,
    skip_ai: bool,
    workers: int,
    compact: bool = False,
) -> tuple[int, int, int, float, float, float, float, "TimingStats | None", list[tuple[str, str]], int]:
    """Run unified processing: summarize -> embed -> index.

    Args:
        compact: If True, hide worker table in display (for watch mode).

    Returns:
        Tuple of (processed, skipped, indexed, avg_wall, avg_summ, avg_embed, elapsed, timing_stats, failed_elements, deleted).
    """
    from magaldi_core.processor import (
        ProcessingConfig,
        ProgressState,
        TimingStats,
        WorkerStatus,
        process_elements,
    )

    if dry_run:
        total = parsing_result.total_elements
        console.print(f"  [dim]Dry run: would process {total} elements[/]")
        return (0, 0, 0, 0.0, 0.0, 0.0, 0.0, None, [], 0)

    from shared.db.store import Repository

    repo = Repository(config)

    # Handle deleted files: remove elements for files that no longer exist on disk
    # Note: Modified files are handled by process_elements' smart delete (compares
    # existing indexed elements with newly parsed elements, deletes only stale ones)
    deleted_from_files = 0
    if manifest.deleted_files:
        for file_info in manifest.deleted_files:
            count = repo.delete_by_file(
                manifest.scope, manifest.repository, manifest.username, file_info.relative_path
            )
            deleted_from_files += count

    # Default worker count (matches DependencyTracker.DEFAULT_WORKERS)
    DEFAULT_WORKERS = 8

    # Smart worker count: cap at total elements (no point having more workers than items)
    total_elements = parsing_result.total_elements
    effective_workers = workers if workers > 0 else DEFAULT_WORKERS
    if total_elements > 0 and total_elements < effective_workers:
        effective_workers = total_elements
        if workers > 0:
            console.print(f"  [dim]Reducing workers from {workers} to {total_elements} (matching element count)[/]")

    proc_config = ProcessingConfig(
        summarize_model=config.llm.get_summarize_model(),
        summarize_model_small=config.llm.get_summarize_model_small(),
        embed_model=config.llm.get_embed_model(),
        skip_ai=skip_ai,
        num_workers=effective_workers,
    )

    # Use effective workers for display
    display_workers = effective_workers

    # Pass context sizes from parsing to processing (for KV cache optimization)
    proc_config.context_sizes = parsing_result.context_sizes

    # Calculate model column width for display
    model_col_width = get_model_column_width(config)

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
        # Use actual allowed workers for ETA (accounts for throttling)
        parallelism = state.parallelism
        if parallelism and parallelism.tier_changing:
            effective_workers = 1
        elif parallelism and parallelism.throttle_decision:
            effective_workers = parallelism.throttle_decision.recommended_workers
        else:
            effective_workers = state.num_workers
        eta = state.timing.eta_seconds(state.completed, state.total, effective_workers)
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
        # Show breakdown: unchanged | non-AI | failed | total found
        if state.skipped > 0:
            bar_text.append(" | ", style="dim")
            bar_text.append(f"{state.skipped} unchanged", style="dim")
        if state.non_ai_skipped > 0:
            bar_text.append(" | ", style="dim")
            bar_text.append(f"{state.non_ai_skipped} imports", style="dim")
        if state.failed > 0:
            bar_text.append(" | ", style="dim")
            bar_text.append(f"{state.failed} failed", style="red")
        if state.total_found > 0:
            bar_text.append(" | ", style="dim")
            bar_text.append(f"{state.total_found} total", style="dim")
        bar_text.append(" | ", style="dim")
        bar_text.append(elapsed_str, style="cyan")
        bar_text.append(" elapsed", style="dim")
        if eta:
            bar_text.append(" | ~", style="dim")
            bar_text.append(format_duration(eta), style="yellow")
            bar_text.append(" ETA", style="dim")

        # Worker table
        import time as time_mod
        worker_table = Table(show_header=False, box=None, padding=0)
        worker_table.add_column("ID", style="dim", width=4)
        worker_table.add_column("Stage", style="cyan", width=12)
        worker_table.add_column("Model", style="yellow", width=model_col_width)
        worker_table.add_column("Ctx", style="magenta", width=4)
        worker_table.add_column("Time", style="green", width=6)
        worker_table.add_column("Element")

        workers_data = state.workers.get_all()
        now = time_mod.time()

        # Determine allowed workers for idle vs throttled distinction
        parallelism = state.parallelism
        if parallelism and parallelism.tier_changing:
            allowed_workers = 1
        elif parallelism and parallelism.throttle_decision:
            allowed_workers = parallelism.throttle_decision.recommended_workers
        else:
            allowed_workers = num_workers

        for wid in range(num_workers):
            if wid in workers_data:
                elem, stage, model, ctx_size, start_time = workers_data[wid]
                elapsed = now - start_time if start_time > 0 else 0
                elapsed_str = f"{elapsed:.1f}s" if elapsed > 0 else ""
                worker_table.add_row(f"[{wid}]", stage, model, ctx_size, elapsed_str, elem)
            elif wid < allowed_workers:
                # Worker could run but no tasks available
                worker_table.add_row(f"[{wid}]", "[dim]idle[/]", "", "", "", "")
            else:
                # Worker is throttled - not allowed to run
                worker_table.add_row(f"[{wid}]", "[dim yellow]throttled[/]", "", "", "", "")

        # Per-type-per-tier ETA breakdown table
        type_colors = {
            "file": "cyan",
            "class": "magenta",
            "interface": "magenta",
            "type_alias": "magenta",
            "function": "blue",
            "method": "green",
            "constant": "yellow",
            "variable": "red",
            "import": "bright_black",
        }
        type_order = ["file", "class", "interface", "type_alias", "function", "method", "constant", "variable"]
        eta_breakdown = state.timing.get_eta_breakdown_with_avg(effective_workers)
        eta_table = build_eta_table(eta_breakdown, type_order, type_colors)

        # Fallback: per-type single-line if no tier data yet
        type_line = ""
        if not eta_table:
            type_stats = state.timing.get_type_stats()
            type_parts = []
            for t in type_order:
                if t in type_stats:
                    done, tot, avg_wall, avg_summ, avg_embed = type_stats[t]
                    color = type_colors.get(t, "white")
                    if done >= tot:
                        type_parts.append(f"[{color}]{t}[/]: [green]{done}/{tot}[/] [dim]({avg_wall:.1f}s)[/]")
                    else:
                        type_parts.append(f"[{color}]{t}[/]: [yellow]{done}/{tot}[/] [dim]({avg_wall:.1f}s)[/]")
            type_line = f"  [dim]Progress:[/] {' [dim]|[/] '.join(type_parts)}" if type_parts else ""

        # Stats line
        # state.completed already excludes unchanged/skipped elements
        effective_wall = state.timing.elapsed / state.completed if state.completed > 0 else 0.0
        total_api = state.timing.avg_summarize_time + state.timing.avg_embed_time
        summ_emb = state.timing.avg_summary_embed_time
        code_emb = state.timing.avg_code_embed_time
        stats = f"  [dim]Throughput:[/] [green]{effective_wall:.2f}s[/]/item [dim]|[/] [dim]API:[/] [green]{total_api:.1f}s[/]/item [dim]([/][green]{state.timing.avg_summarize_time:.1f}s[/] summ + [cyan]{summ_emb:.1f}s[/] summ_emb + [cyan]{code_emb:.1f}s[/] code_emb[dim])[/]"

        # Parallelism stats: running/allowed/baseline (info)
        # - running: current non-idle threads
        # - allowed: current max (may be reduced by throttling or warmup)
        # - baseline: configured max workers
        running_count = len(workers_data)
        parallelism = state.parallelism
        baseline = num_workers

        # Determine allowed workers and status info
        if parallelism and parallelism.tier_changing:
            # Tier change in progress
            allowed = 1
            if parallelism.running > 1:
                # Old tier tasks still draining
                info = "[yellow]tier draining...[/]"
            else:
                # Warmup task loading model
                info = "[yellow]model warmup[/]"
        elif parallelism and parallelism.throttle_decision:
            # Always use recommended_workers - covers both throttling AND ramp-up
            allowed = parallelism.throttle_decision.recommended_workers
            # No verbose reason text - numbers speak for themselves
            info = None
        else:
            # No throttle decision available
            allowed = baseline
            info = None

        # Format: running/allowed/baseline (info) - or just running/baseline if normal
        if allowed < baseline:
            stats += f" [dim]|[/] [dim]Workers:[/] [green]{running_count}[/]/[yellow]{allowed}[/]/[cyan]{baseline}[/]"
            if info:
                stats += f" [dim]([/]{info}[dim])[/]"
        else:
            stats += f" [dim]|[/] [dim]Workers:[/] [green]{running_count}[/]/[cyan]{baseline}[/]"

        # Show runtime info for throttling decisions
        parallelism = state.parallelism
        if parallelism and parallelism.throttle_decision:
            td = parallelism.throttle_decision
            # Show max (raw) and per-worker comparison (normalized vs historical)
            if td.current_max > 0 or td.completed_avg > 0:
                effective_max = max(td.current_max, td.historical_max)
                # Normalize max by running workers for comparison with historical base
                normalized_max = effective_max / max(running_count, 1)
                stats += f" [dim]|[/] [dim]Max:[/] [yellow]{effective_max:.1f}s[/]"
                stats += f" [dim]|[/] [dim]Per Worker:[/] [yellow]{normalized_max:.1f}s[/] [dim]vs[/] [cyan]{td.completed_avg:.1f}s[/] [dim](last {td.completion_count})[/]"

        parts: list[RenderableType] = [bar_text]
        if eta_table:
            parts.append(eta_table)
        elif type_line:
            parts.append(type_line)
        if not compact:
            parts.append(worker_table)
        parts.append(stats)

        # Show recent errors if any
        if state.recent_errors:
            error_text = Text()
            error_text.append("  Errors:\n", style="red bold")
            for i, (elem_name, error) in enumerate(state.recent_errors):
                error_text.append("    ", style="dim")
                error_text.append(f"{elem_name}", style="yellow")
                error_text.append(":\n", style="dim")
                short_error = error[:250] + "..." if len(error) > 250 else error
                error_text.append(f"      {short_error}\n", style="red")
            parts.append(error_text)

        return Group(*parts)

    # Create shared state objects
    timing_stats = TimingStats()
    timing_stats.phase_start = time.time()  # Initialize now so elapsed display is correct
    worker_status = WorkerStatus()
    total = parsing_result.total_elements

    # Initialize state (will be replaced by processor's on_progress callback)
    current_state = ProgressState(
        total=total,
        completed=0,
        skipped=0,
        failed=0,
        timing=timing_stats,
        workers=worker_status,
        num_workers=display_workers,
        total_found=total,
        non_ai_skipped=0,
    )

    class LiveDisplay:
        def __rich__(self) -> RenderableType:
            return build_display(current_state, display_workers)

    with Live(LiveDisplay(), console=console, refresh_per_second=4) as live:
        def on_progress(state: ProgressState) -> None:
            nonlocal current_state
            current_state = state
            # Let Rich handle refresh at configured rate (4/sec)

        def on_status_change() -> None:
            # Let Rich handle refresh at configured rate
            pass

        result = process_elements(
            parsing_result.parsed_files,
            manifest.scope,
            manifest.repository,
            manifest.username,
            repo,
            proc_config,
            on_progress,
            file_hashes,
            on_status_change,
            worker_status,
            timing_stats,
            magaldi_config=config if not dry_run else None,
        )

    # Check for processing errors (including stalls)
    if result.errors:
        for error in result.errors:
            if "Processing stalled" in error:
                console.print(f"\n  [red]Warning:[/] {error}")

    # Display tier accuracy summary (only when issues detected)
    tier_summary = timing_stats.get_tier_accuracy_summary()
    if tier_summary.get("has_issues"):
        input_rows = tier_summary.get("input", [])
        output_rows = tier_summary.get("output", [])

        if input_rows:
            tier_table = Table(title="Context Tier Accuracy", box=None, padding=(0, 2), expand=False)
            tier_table.add_column("type", style="cyan")
            tier_table.add_column("tier", style="magenta", justify="right")
            tier_table.add_column("count", justify="right")
            tier_table.add_column("overflows", style="red", justify="right")
            tier_table.add_column("avg headroom", justify="right")
            tier_table.add_column("worst", justify="right")
            for elem_type, tier, count, overflows, avg_pct, worst_pct in input_rows:
                overflow_style = "red bold" if overflows > 0 else "dim"
                worst_style = "red" if worst_pct < 0 else ("yellow" if worst_pct < 10 else "green")
                tier_table.add_row(
                    elem_type, str(tier), str(count),
                    f"[{overflow_style}]{overflows}[/]",
                    f"{avg_pct:.0f}%",
                    f"[{worst_style}]{worst_pct:.0f}%[/]",
                )
            console.print()
            console.print(tier_table)

        if output_rows:
            out_table = Table(title="Output Token Usage", box=None, padding=(0, 2), expand=False)
            out_table.add_column("type", style="cyan")
            out_table.add_column("avg", justify="right")
            out_table.add_column("max", justify="right")
            out_table.add_column("budget", justify="right")
            for elem_type, avg_tokens, max_tokens, budget in output_rows:
                exceeded = max_tokens > budget
                max_style = "red bold" if exceeded else "green"
                suffix = " [red]exceeded[/]" if exceeded else ""
                out_table.add_row(
                    elem_type, str(avg_tokens),
                    f"[{max_style}]{max_tokens}[/]",
                    f"{budget}{suffix}",
                )
            console.print()
            console.print(out_table)

    # Get timing stats from current state
    avg_summ = current_state.timing.avg_summarize_time
    avg_embed = current_state.timing.avg_embed_time
    elapsed = current_state.timing.elapsed
    api_processed = result.elements_processed - result.elements_skipped
    avg_wall = elapsed / api_processed if api_processed > 0 else 0.0

    # Total deleted = from deleted files + stale elements from modified files
    total_deleted = deleted_from_files + result.elements_deleted
    return (result.elements_processed, result.elements_skipped, result.indexed, avg_wall, avg_summ, avg_embed, elapsed, timing_stats, result.failed_elements, total_deleted)


def run_hierarchy_extraction(
    scope: str,
    repository: str,
    username: str,
    repo: "Repository",
    cli_entry_point: str | None = None,
    api_prefix: str = "/api/v1",
) -> tuple[int, int]:
    """Run hierarchy extraction: CLI commands and HTTP routes.

    Args:
        scope: Repository scope
        repository: Repository name
        username: Username/branch
        repo: Search repository instance
        cli_entry_point: CLI entry point name (e.g., "magaldi")
        api_prefix: API URL prefix for route hierarchy (e.g., "/api/v1")

    Returns:
        Tuple of (relationships_indexed, external_refs_indexed)
    """
    from magaldi_core.analysis.hierarchy_extractors import (
        CliHierarchyExtractor,
        RouteHierarchyExtractor,
        elements_to_element_info,
        elements_to_route_info,
    )
    from shared.db.repositories.relationships import RelationshipsRepository

    client = repo._get_client()
    all_relationships = []
    all_external_refs = []

    # =========================================================================
    # CLI Hierarchy Extraction
    # =========================================================================

    # Query elements with decorators (for CLI commands)
    decorator_result = client.search(
        index="magaldi-code-elements",
        body={
            "size": 10000,
            "query": {
                "bool": {
                    "filter": [
                        {"term": {"scope": scope}},
                        {"term": {"repository": repository}},
                        {"term": {"username": username}},
                        {"exists": {"field": "decorator_details"}},
                    ],
                },
            },
            "_source": [
                "hash_id",
                "element_id",
                "name",
                "relative_path",
                "line_start",
                "decorators",
                "decorator_details",
                "summary",
            ],
        },
    )

    decorator_docs = [hit["_source"] for hit in decorator_result.get("hits", {}).get("hits", [])]

    if decorator_docs:
        elements = elements_to_element_info(decorator_docs)
        if elements:
            cli_extractor = CliHierarchyExtractor(
                scope=scope,
                repository=repository,
                username=username,
                cli_entry_point=cli_entry_point,
            )
            cli_rels, cli_refs = cli_extractor.extract(elements)
            all_relationships.extend(cli_rels)
            all_external_refs.extend(cli_refs)

    # =========================================================================
    # HTTP Route Hierarchy Extraction
    # =========================================================================

    # Query elements with http_routes
    route_result = client.search(
        index="magaldi-code-elements",
        body={
            "size": 10000,
            "query": {
                "bool": {
                    "filter": [
                        {"term": {"scope": scope}},
                        {"term": {"repository": repository}},
                        {"term": {"username": username}},
                        {"exists": {"field": "http_routes"}},
                    ],
                },
            },
            "_source": [
                "hash_id",
                "element_id",
                "name",
                "relative_path",
                "line_start",
                "http_routes",
                "summary",
            ],
        },
    )

    route_docs = [hit["_source"] for hit in route_result.get("hits", {}).get("hits", [])]

    if route_docs:
        route_elements = elements_to_route_info(route_docs)
        if route_elements:
            route_extractor = RouteHierarchyExtractor(
                scope=scope,
                repository=repository,
                username=username,
                api_prefix=api_prefix,
            )
            route_rels, route_refs = route_extractor.extract(route_elements)
            all_relationships.extend(route_rels)
            all_external_refs.extend(route_refs)

    # =========================================================================
    # Store Results
    # =========================================================================

    if not all_relationships and not all_external_refs:
        return (0, 0)

    # Store in relationships index (use same config as repo)
    rel_repo = RelationshipsRepository(repo.config)

    # Delete existing relationships for this user before re-indexing
    rel_repo.delete_relationships_for_user(scope, repository, username)
    rel_repo.delete_external_refs_for_user(scope, repository, username)

    # Index new relationships and external refs
    rel_result = rel_repo.bulk_index_relationships(all_relationships)
    ref_result = rel_repo.bulk_index_external_refs(all_external_refs)

    return (rel_result["indexed"], ref_result["indexed"])


def run_call_resolution(
    repo: "Repository",
    scope: str,
    repository: str,
    username: str,
    skip_resolve: bool = False,
    console: "Console | None" = None,
) -> None:
    """Run Phase 6: Call Resolution (static + embedding + semantic relationships).

    Args:
        repo: Search repository instance.
        scope: Repository scope.
        repository: Repository name.
        username: Username/branch.
        skip_resolve: If True, skip call resolution (but still compute semantic relationships).
        console: Rich console for output.
    """
    from rich.console import Console

    if console is None:
        console = Console()

    if not skip_resolve:
        from magaldi_core.call_resolution import resolve_all_calls

        console.print("\n  [bold]Static Call Resolution[/]")
        try:
            total_calls, import_resolved, type_resolved = resolve_all_calls(
                repo, scope, repository, username,
            )
            total_resolved = import_resolved + type_resolved
            console.print(f"  Full pass: {total_resolved}/{total_calls} resolved")
            console.print(f"    via imports: {import_resolved}, via type annotations: {type_resolved}")
        except Exception as e:
            console.print(f"  [yellow]Warning: Static call resolution failed: {e}[/]")

        # Embedding-based resolution for remaining untyped calls
        from magaldi_core.call_resolution import resolve_calls_by_embedding

        console.print("\n  [bold]Embedding Call Resolution[/]")
        try:
            total_processed, single_resolved, embedding_resolved = resolve_calls_by_embedding(
                repo, scope, repository, username,
            )
            total_resolved = single_resolved + embedding_resolved
            console.print(f"  Resolved: {total_resolved}/{total_processed} untyped calls")
            console.print(f"    single match: {single_resolved}, via similarity: {embedding_resolved}")
        except Exception as e:
            console.print(f"  [yellow]Warning: Embedding call resolution failed: {e}[/]")
    else:
        console.print("\n  [dim]Call resolution skipped (--skip-resolve)[/]")

    # Semantic relationships always run (independent of call resolution)
    from magaldi_core.call_resolution import compute_semantic_relationships

    console.print("\n  [bold]Semantic Relationships[/]")
    try:
        elements_processed, total_relationships = compute_semantic_relationships(
            repo, scope, repository, username,
        )
        console.print(f"  Processed {elements_processed} elements, stored {total_relationships} relationships")
    except Exception as e:
        console.print(f"  [yellow]Warning: Semantic relationships failed: {e}[/]")
