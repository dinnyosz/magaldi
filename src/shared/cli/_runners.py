"""Phase runner functions for CLI commands.

This module contains the run_* functions that execute each phase
of the parsing and processing pipeline.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from rich.console import Group, RenderableType
from rich.live import Live
from rich.markup import escape as rich_escape
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
    from rich.console import Console

    from magaldi_core.change_detection import ChangeManifest
    from magaldi_core.code_parser import ParsingResult
    from magaldi_core.discovery import DiscoveryResult
    from magaldi_core.processor import ProgressState, TimingStats
    from magaldi_core.variable_scoring.models import ScoringProgressState, ScoringResult
    from shared.config import MagaldiConfig
    from shared.db.store import Repository


def run_discovery(repo_path: str, username: str, skip_tests: bool = False) -> DiscoveryResult:
    """Run Phase 1: Discovery."""
    from magaldi_core.discovery import discover

    with console.status("[bold blue]Discovering repository...[/]"):
        return discover(repo_path, username, skip_tests=skip_tests)


def run_change_detection(
    discovery_result: DiscoveryResult,
    config: MagaldiConfig,
    dry_run: bool,
) -> ChangeManifest:
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

    try:
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
    finally:
        if hasattr(file_state_repo, "close"):
            file_state_repo.close()


def run_parsing(manifest: ChangeManifest) -> ParsingResult:
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


def _build_scoring_display(state: ScoringProgressState, num_workers: int) -> RenderableType:
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

    allowed = state.allowed_workers if state.allowed_workers is not None else num_workers

    # Determine exploration budget cap — workers beyond this are permanently
    # disabled for this tier, so collapse them into a single summary line.
    td = state.throttle_decision
    explore_cap = getattr(td, 'explore_cap', None) if td is not None else None
    budget_disabled = 0
    display_total = num_workers
    if explore_cap is not None and explore_cap < num_workers:
        budget_disabled = num_workers - explore_cap
        display_total = explore_cap

    active_count = len(workers_data)
    idle_slots = max(0, allowed - active_count)
    throttled_slots = max(0, display_total - allowed)

    # Show active workers first (renumbered 1..N for consistent display)
    for display_id, wid in enumerate(sorted(workers_data.keys()), start=1):
        batch_num, batch_size, start_time = workers_data[wid]
        worker_elapsed = now - start_time if start_time > 0 else 0
        worker_table.add_row(
            f"[{display_id}]",
            "scoring",
            f"batch#{batch_num}",
            f"{batch_size} vars",
            f"{worker_elapsed:.1f}s",
        )
    # Continue numbering for idle and throttled slots
    next_id = active_count + 1
    # Then idle slots (allowed but not active)
    for i in range(idle_slots):
        worker_table.add_row(f"[{next_id + i}]", "[dim]idle[/]", "", "", "")
    next_id += idle_slots
    # Then throttled slots (beyond allowed limit but within budget)
    for i in range(throttled_slots):
        worker_table.add_row(f"[{next_id + i}]", "[dim yellow]throttled[/]", "", "", "")
    # Budget summary text (rendered outside the table so it's not constrained by column widths)
    budget_text = None
    if budget_disabled > 0:
        budget_text = Text()
        budget_text.append(f"  {budget_disabled} workers disabled (exploration budget)", style="dim")

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

        # Workers: running/allowed/baseline when throttled, running/baseline otherwise
        running_count = state.workers.active_count()
        allowed = state.allowed_workers if state.allowed_workers is not None else num_workers
        stats_text.append("Workers:", style="dim")
        if allowed < num_workers:
            stats_text.append(f" {running_count}", style="green")
            stats_text.append("/", style="dim")
            stats_text.append(f"{allowed}", style="yellow")
            stats_text.append("/", style="dim")
            stats_text.append(f"{num_workers}", style="cyan")
        else:
            stats_text.append(f" {running_count}", style="green")
            stats_text.append("/", style="dim")
            stats_text.append(f"{num_workers}", style="cyan")

        # Show runtime info when throttle decision available
        td = state.throttle_decision
        if td is not None and (td.current_max > 0 or td.completed_avg > 0):
            effective_max = max(td.current_max, td.historical_max)
            normalized_max = effective_max / max(running_count, 1)
            stats_text.append(" | ", style="dim")
            stats_text.append("Max:", style="dim")
            stats_text.append(f" {effective_max:.1f}s", style="yellow")
            stats_text.append(" | ", style="dim")
            stats_text.append("Per Worker:", style="dim")
            stats_text.append(f" {normalized_max:.1f}s", style="yellow")
            stats_text.append(" vs ", style="dim")
            stats_text.append(f"{td.completed_avg:.1f}s", style="cyan")
            if td.completion_count > 0:
                stats_text.append(f" (last {td.completion_count})", style="dim")
            if td.peak_concurrency is not None:
                stats_text.append(" | ", style="dim")
                stats_text.append(f"Peak@{td.peak_concurrency}", style="magenta")
            # Hold / Emergency indicators
            is_hold = getattr(td, 'is_hold', False)
            is_emergency = getattr(td, 'is_emergency', False)
            if is_emergency:
                stats_text.append(" | ", style="dim")
                stats_text.append("⛔ EMERGENCY", style="bold bright_red")
            elif is_hold:
                stats_text.append(" | ", style="dim")
                stats_text.append("⏸ HOLD", style="bold bright_yellow")

            exploration_status = getattr(td, 'exploration_status', None)
            if exploration_status:
                stats_text.append(" | ", style="dim")
                stats_text.append(exploration_status, style="bright_cyan")
            explore_cap = getattr(td, 'explore_cap', None)
            if explore_cap is not None:
                stats_text.append(" | ", style="dim")
                stats_text.append("Range: ", style="dim")
                stats_text.append(f"1‥{explore_cap}", style="bright_cyan")
                stats_text.append(f"/{num_workers}", style="dim")

    # Build per-level throughput line (separate from stats)
    levels_text = None
    td = state.throttle_decision
    if td is not None and hasattr(td, 'all_levels') and td.all_levels:
        from shared.throttling import build_throughput_levels_text
        explore = getattr(td, 'exploration_target', None)
        prob_data = getattr(td, 'prob_map_data', None)
        ecap = getattr(td, 'explore_cap', None)
        levels_text = build_throughput_levels_text(td.all_levels, td.peak_concurrency, max_workers=num_workers, exploration_target=explore, prob_map_data=prob_data, explore_cap=ecap)

    parts: list[RenderableType] = [bar_text]
    if score_text.plain:
        parts.append(score_text)
    parts.append(worker_table)
    if budget_text is not None:
        parts.append(budget_text)
    if stats_text.plain:
        parts.append(stats_text)
    if levels_text is not None:
        parts.append(levels_text)

    return Group(*parts)


def run_variable_scoring(
    parsing_result: ParsingResult,
    config: MagaldiConfig,
    workers: int = 0,
) -> ScoringResult:
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

    # Remove test variables/constants from parsed files entirely —
    # they are self-documenting noise that wastes LLM scoring, embedding,
    # and indexing resources. Purge them here so they never reach Phase 5.
    for pf in parsing_result.parsed_files:
        pf.elements = [
            elem for elem in pf.elements
            if not (elem.is_test and elem.element_type in ("variable", "constant"))
        ]

    # Collect remaining variable/constant elements for scoring
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

    # Heuristic pre-filter: drop obvious low-value variables before LLM
    from magaldi_core.variable_scoring.heuristic_filter import apply_heuristic_filter

    total_before_heuristic = len(variables)
    variables, heuristic_drops = apply_heuristic_filter(variables)
    heuristic_count = total_before_heuristic - len(variables)

    # Remove heuristic-dropped variables from parsing_result immediately
    if heuristic_drops:
        for pf in parsing_result.parsed_files:
            pf.elements = [
                elem for elem in pf.elements
                if elem.element_id not in heuristic_drops
            ]

    if not variables:
        return ScoringResult(
            total_variables=total_before_heuristic,
            heuristic_dropped=heuristic_count,
            dropped=heuristic_count,
        )

    # Create LLM client using the main model (small model scores everything 1,1,1,1)
    model_config = config.llm.get_summarize_model()
    model_name = model_config.name

    scoring_config = VariableScoringConfig()

    # Score cache: reuse scores from previous runs for unchanged variables
    from magaldi_core.variable_scoring.models import VariableScore
    from shared.db.repositories import Repository

    cached_scores: dict[str, VariableScore] = {}
    cache_count = 0
    score_cache_repo: Repository | None = None
    try:
        score_cache_repo = Repository(config)
        # Build element_id -> content_hash mapping for validation
        elem_hashes: dict[str, str | None] = {}
        for pf in parsing_result.parsed_files:
            for elem in pf.elements:
                if elem.element_type in ("variable", "constant"):
                    elem_hashes[elem.element_id] = elem.content_hash

        all_ids = [eid for eid, _, _, _ in variables]
        cached_state = score_cache_repo.get_variable_scoring_state(all_ids)

        for eid, state in cached_state.items():
            if (
                state.get("score_model") == model_name
                and state.get("content_hash") == elem_hashes.get(eid)
                and state.get("variable_score")
            ):
                score_data = state["variable_score"]
                cached_scores[eid] = VariableScore(
                    config_value=score_data.get("config_value", 1),
                    architectural_role=score_data.get("architectural_role", 1),
                    data_definition=score_data.get("data_definition", 1),
                    general_usefulness=score_data.get("general_usefulness", 1),
                )

        cache_count = len(cached_scores)
        # Filter out cached variables — only send uncached to LLM
        if cached_scores:
            variables = [
                (eid, fp, name, code)
                for eid, fp, name, code in variables
                if eid not in cached_scores
            ]
    except Exception:  # noqa: BLE001
        # Cache lookup is best-effort — continue without cache on any error
        pass
    finally:
        if score_cache_repo is not None:
            score_cache_repo.close()

    # If all variables are cached, skip LLM entirely
    if not variables and cached_scores:
        # Apply threshold to cached scores
        result = ScoringResult(total_variables=total_before_heuristic)
        all_scores = dict(cached_scores)
        for _eid, score in all_scores.items():
            if score.passes_threshold(scoring_config.threshold):
                result.kept += 1
            else:
                result.dropped += 1
        result.scores = all_scores
        result.heuristic_dropped = heuristic_count
        result.cached_scores = cache_count
        result.llm_scored = 0
        result.dropped += heuristic_count
        result.model_name = model_name

        # Still need to remove low-scoring and attach scores
        _apply_scores_to_elements(parsing_result, all_scores, scoring_config.threshold)
        return result

    llm_client = SummarizationLLMClient.from_model_config(model_config)
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

    def on_progress(state: ScoringProgressState) -> None:
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

    # Merge cached scores into LLM result
    if cached_scores:
        result.scores.update(cached_scores)
        # Recount kept/dropped with merged scores
        result.kept = 0
        result.dropped = 0
        for _eid, score in result.scores.items():
            if score.passes_threshold(scoring_config.threshold):
                result.kept += 1
            else:
                result.dropped += 1

    # Print batch samples: random variables with scores across batches
    if result.batch_samples:
        console.print()
        console.print(
            f"  [bold dim]─── Variable scoring samples "
            f"(model={model_config.name}, {len(result.batch_samples)} batches sampled) ───[/]"
        )
        for batch_idx, samples in enumerate(result.batch_samples):
            console.print(f"  [bold dim]Batch {batch_idx + 1}:[/]")
            for file_path, name, raw_code, score in samples:
                verdict = "[green]KEEP[/]" if score.passes_threshold() else "[red]DROP[/]"
                scores_str = ",".join(str(s) for s in score.as_tuple())
                console.print(f"    {verdict} [{scores_str}] [cyan]{name}[/] [dim]({file_path})[/]")
                # Show raw_code preserving newlines, indented and truncated
                code_lines = (raw_code or "").split("\n")
                for line in code_lines[:6]:
                    truncated = line[:100] + "…" if len(line) > 100 else line
                    console.print(f"    [dim]│[/] {truncated}")
                if len(code_lines) > 6:
                    console.print(f"    [dim]│ ... ({len(code_lines) - 6} more lines)[/]")
            console.print()
        console.print("  [bold dim]───────────────────────────────────────[/]")

    # Apply threshold + attach scores to elements for Phase 5 storage
    _apply_scores_to_elements(parsing_result, result.scores, scoring_config.threshold)

    # Update result with optimization stats
    result.heuristic_dropped = heuristic_count
    result.cached_scores = cache_count
    result.llm_scored = len(variables)
    result.total_variables = total_before_heuristic
    result.dropped += heuristic_count  # Add heuristic drops to total
    result.model_name = model_name

    return result


def _apply_scores_to_elements(
    parsing_result: ParsingResult,
    scores: dict,
    threshold: int,
) -> None:
    """Remove below-threshold variables and attach scores to kept elements.

    Modifies *parsing_result* in place:
    - Removes elements whose score is below *threshold*
    - Sets ``variable_score`` dict on kept variable/constant elements for
      Phase 5 to persist into OpenSearch.
    """
    dropped_ids = {
        eid for eid, score in scores.items()
        if not score.passes_threshold(threshold)
    }

    for pf in parsing_result.parsed_files:
        pf.elements = [
            elem for elem in pf.elements
            if elem.element_id not in dropped_ids
        ]
        # Attach scores to surviving variable/constant elements
        for elem in pf.elements:
            if elem.element_id in scores:
                score = scores[elem.element_id]
                elem.variable_score = {
                    "config_value": score.config_value,
                    "architectural_role": score.architectural_role,
                    "data_definition": score.data_definition,
                    "general_usefulness": score.general_usefulness,
                }


def run_processing(
    parsing_result: ParsingResult,
    manifest: ChangeManifest,
    config: MagaldiConfig,
    dry_run: bool,
    skip_ai: bool,
    workers: int,
    compact: bool = False,
    use_docstrings: bool = True,
) -> tuple[int, int, int, float, float, float, float, TimingStats | None, list[tuple[str, str]], int]:
    """Run unified processing: summarize -> embed -> index.

    Args:
        compact: If True, hide worker table in display (for watch mode).
        use_docstrings: If True, use docstring descriptions as summaries
            instead of LLM for elements with meaningful docstrings.

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
        use_docstrings=use_docstrings,
        num_workers=effective_workers,
        summarize_temperature=config.llm.summarize_temperature,
        summarize_top_p=config.llm.summarize_top_p,
        summarize_top_k=config.llm.summarize_top_k,
        summarize_min_p=config.llm.summarize_min_p,
        summarize_presence_penalty=config.llm.summarize_presence_penalty,
        summarize_repetition_penalty=config.llm.summarize_repetition_penalty,
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

        # Determine exploration budget cap — workers beyond this are permanently
        # disabled for this tier, so collapse them into a single summary line
        # instead of showing individual rows.
        explore_cap = None
        if parallelism and parallelism.throttle_decision:
            explore_cap = getattr(parallelism.throttle_decision, 'explore_cap', None)
        budget_disabled = 0
        display_total = num_workers
        if explore_cap is not None and explore_cap < num_workers:
            budget_disabled = num_workers - explore_cap
            display_total = explore_cap

        active_count = len(workers_data)
        idle_slots = max(0, allowed_workers - active_count)
        throttled_slots = max(0, display_total - allowed_workers)

        # Show active workers first (renumbered 1..N for consistent display)
        for display_id, wid in enumerate(sorted(workers_data.keys()), start=1):
            elem, stage, model, ctx_size, start_time = workers_data[wid]
            elapsed = now - start_time if start_time > 0 else 0
            elapsed_str = f"{elapsed:.1f}s" if elapsed > 0 else ""
            worker_table.add_row(f"[{display_id}]", stage, model, ctx_size, elapsed_str, elem)
        # Continue numbering for idle and throttled slots
        next_id = active_count + 1
        # Then idle slots (allowed but not active)
        for i in range(idle_slots):
            worker_table.add_row(f"[{next_id + i}]", "[dim]idle[/]", "", "", "", "")
        next_id += idle_slots
        # Then throttled slots (beyond allowed limit but within budget)
        for i in range(throttled_slots):
            worker_table.add_row(f"[{next_id + i}]", "[dim yellow]throttled[/]", "", "", "", "")
        # Budget summary text (rendered outside the table so it's not constrained by column widths)
        budget_text = None
        if budget_disabled > 0:
            budget_text = Text()
            budget_text.append(f"  {budget_disabled} workers disabled (exploration budget)", style="dim")

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
        type_order = ["file", "class", "interface", "type_alias", "function", "method", "constant", "variable", "import"]
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
                if td.peak_concurrency is not None:
                    stats += f" [dim]|[/] [magenta]Peak@{td.peak_concurrency}[/]"
                # Hold / Emergency indicators
                is_hold = getattr(td, 'is_hold', False)
                is_emergency = getattr(td, 'is_emergency', False)
                if is_emergency:
                    stats += " [dim]|[/] [bold bright_red]⛔ EMERGENCY[/]"
                elif is_hold:
                    stats += " [dim]|[/] [bold bright_yellow]⏸ HOLD[/]"

                exploration_status = getattr(td, 'exploration_status', None)
                if exploration_status:
                    stats += f" [dim]|[/] [bright_cyan]{exploration_status}[/]"
                explore_cap = getattr(td, 'explore_cap', None)
                if explore_cap is not None:
                    stats += f" [dim]|[/] [dim]Range:[/] [bright_cyan]1‥{explore_cap}[/][dim]/{num_workers}[/]"

        # Build per-level throughput line (separate from stats)
        levels_line = None
        if parallelism and parallelism.throttle_decision:
            td = parallelism.throttle_decision
            if hasattr(td, 'all_levels') and td.all_levels:
                from shared.throttling import format_throughput_levels
                explore = getattr(td, 'exploration_target', None)
                prob_data = getattr(td, 'prob_map_data', None)
                ecap = getattr(td, 'explore_cap', None)
                levels_line = format_throughput_levels(td.all_levels, td.peak_concurrency, max_workers=num_workers, exploration_target=explore, prob_map_data=prob_data, explore_cap=ecap)

        parts: list[RenderableType] = [bar_text]
        if eta_table:
            parts.append(eta_table)
        elif type_line:
            parts.append(type_line)
        if not compact:
            parts.append(worker_table)
        if budget_text is not None:
            parts.append(budget_text)
        parts.append(stats)
        if levels_line:
            parts.append(levels_line)

        # Show recent errors if any
        if state.recent_errors:
            error_text = Text()
            error_text.append("  Errors:\n", style="red bold")
            for _i, (elem_name, error) in enumerate(state.recent_errors):
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

    with Live(LiveDisplay(), console=console, refresh_per_second=4):
        def on_progress(state: ProgressState) -> None:
            nonlocal current_state
            current_state = state
            # Let Rich handle refresh at configured rate (4/sec)

        def on_status_change() -> None:
            # Let Rich handle refresh at configured rate
            pass

        # Derive score_model for caching variable scores in Phase 5
        _score_model = None
        if not skip_ai:
            import contextlib
            with contextlib.suppress(Exception):
                _score_model = config.llm.get_summarize_model().name

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
            score_model=_score_model,
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

    # Close the repository client to release connections
    repo.close()

    # Total deleted = from deleted files + stale elements from modified files
    total_deleted = deleted_from_files + result.elements_deleted
    return (result.elements_processed, result.elements_skipped, result.indexed, avg_wall, avg_summ, avg_embed, elapsed, timing_stats, result.failed_elements, total_deleted)


def run_hierarchy_extraction(
    scope: str,
    repository: str,
    username: str,
    repo: Repository,
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

    # Close the relationships repository client
    rel_repo.close()

    return (rel_result["indexed"], ref_result["indexed"])


def run_call_resolution(
    repo: Repository,
    scope: str,
    repository: str,
    username: str,
    skip_resolve: bool = False,
    console: Console | None = None,
    max_workers: int = 1,
) -> None:
    """Run Phase 6: Call Resolution (static + embedding + semantic relationships).

    Args:
        repo: Search repository instance.
        scope: Repository scope.
        repository: Repository name.
        username: Username/branch.
        skip_resolve: If True, skip call resolution (but still compute semantic relationships).
        console: Rich console for output.
        max_workers: Number of threads for parallel call resolution (1 = sequential).
    """
    from rich.console import Console

    if console is None:
        console = Console()

    # Resolve effective worker count for call resolution.
    # 0 = auto: default to 1 (sequential) since call resolution is
    # I/O-bound on OpenSearch and parallelism needs explicit opt-in.
    effective_workers = max_workers if max_workers > 0 else 1

    with repo.bulk_buffer():

        if not skip_resolve:
            from magaldi_core.call_resolution import resolve_all_calls

            console.print("\n  [bold]Static Call Resolution[/]")
            if effective_workers > 1:
                console.print(f"  [dim]Using {effective_workers} worker threads[/]")
            try:
                (
                    total_calls,
                    import_resolved,
                    type_resolved,
                    constructor_resolved,
                    scope_resolved,
                    super_resolved,
                ) = resolve_all_calls(
                    repo, scope, repository, username,
                    max_workers=effective_workers,
                    on_step=lambda msg: console.print(f"    [dim]{msg}[/]"),
                )
                total_resolved = (
                    import_resolved + type_resolved + constructor_resolved
                    + scope_resolved + super_resolved
                )
                console.print(f"  Full pass: {total_resolved}/{total_calls} resolved")
                console.print(f"    via imports: {import_resolved}")
                console.print(f"    via types: {type_resolved}")
                console.print(f"    via constructors: {constructor_resolved}")
                console.print(f"    via scope: {scope_resolved}")
                if super_resolved:
                    console.print(f"    via super: {super_resolved}")
            except Exception as e:
                console.print(f"  [yellow]Warning: Static call resolution failed: {rich_escape(str(e))}[/]")

            # Flush + refresh so static resolution writes are visible to embedding resolution
            repo.flush()
            repo.refresh()

            # Embedding-based resolution for remaining untyped calls
            from magaldi_core.call_resolution import resolve_calls_by_embedding

            console.print("\n  [bold]Embedding Call Resolution[/]")
            try:
                total_processed, single_resolved, embedding_resolved = resolve_calls_by_embedding(
                    repo, scope, repository, username,
                )
                total_resolved = single_resolved + embedding_resolved
                console.print(f"  Resolved: {total_resolved}/{total_processed} untyped calls")
                console.print(f"    single match: {single_resolved}")
                console.print(f"    via RRF similarity: {embedding_resolved}")
            except Exception as e:
                console.print(f"  [yellow]Warning: Embedding call resolution failed: {rich_escape(str(e))}[/]")
        else:
            console.print("\n  [dim]Call resolution skipped (--skip-resolve)[/]")

        # Flush + refresh so embedding resolution writes are visible to semantic relationships
        repo.flush()
        repo.refresh()

        # Semantic relationships always run (independent of call resolution)
        from magaldi_core.call_resolution import compute_semantic_relationships

        console.print("\n  [bold]Semantic Relationships[/]")
        try:
            elements_processed, total_relationships = compute_semantic_relationships(
                repo, scope, repository, username,
            )
            console.print(f"  Processed {elements_processed} elements, stored {total_relationships} relationships")
        except Exception as e:
            console.print(f"  [yellow]Warning: Semantic relationships failed: {rich_escape(str(e))}[/]")
