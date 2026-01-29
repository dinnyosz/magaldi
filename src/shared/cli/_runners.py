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

from shared.ai.context_size import TIER_MAX_WORKERS
from shared.cli._shared import console, format_duration, get_model_column_width

if TYPE_CHECKING:
    from magaldi_core.change_detection import ChangeManifest
    from magaldi_core.code_parser import ParsingResult
    from magaldi_core.discovery import DiscoveryResult
    from magaldi_core.processor import ProgressState, TimingStats
    from shared.config import MagaldiConfig
    from shared.db.elasticsearch import ElasticsearchRepository


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
        from shared.db.elasticsearch import ElasticsearchFileStateRepository
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


def run_processing(
    parsing_result: "ParsingResult",
    manifest: "ChangeManifest",
    config: "MagaldiConfig",
    dry_run: bool,
    skip_ai: bool,
    workers: int,
) -> tuple[int, int, int, float, float, float, float, "TimingStats | None", list[tuple[str, str]], int]:
    """Run unified processing: summarize -> embed -> index.

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

    from shared.db.elasticsearch import ElasticsearchRepository

    es_repo = ElasticsearchRepository(config)

    # Handle deleted files: remove elements for files that no longer exist on disk
    # Note: Modified files are handled by process_elements' smart delete (compares
    # existing ES elements with newly parsed elements, deletes only stale ones)
    deleted_from_files = 0
    if manifest.deleted_files:
        for file_info in manifest.deleted_files:
            count = es_repo.delete_by_file(
                manifest.scope, manifest.repository, manifest.username, file_info.relative_path
            )
            deleted_from_files += count

    from shared.ai.context_size import TIER_MAX_WORKERS

    proc_config = ProcessingConfig(
        summarize_model=config.llm.get_summarize_model(),
        summarize_model_small=config.llm.get_summarize_model_small(),
        embed_model=config.llm.get_embed_model(),
        skip_ai=skip_ai,
        num_workers=workers,
    )

    # Calculate actual max workers for display (0 = auto from tier defaults)
    display_workers = workers if workers > 0 else max(TIER_MAX_WORKERS.values())

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
        eta = state.timing.eta_seconds(state.completed, state.total, state.num_workers)
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
        if state.skipped > 0:
            bar_text.append(" | ", style="dim")
            bar_text.append(f"{state.skipped} unchanged", style="dim")
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
        for wid in range(num_workers):
            if wid in workers_data:
                elem, stage, model, ctx_size, start_time = workers_data[wid]
                elapsed = now - start_time if start_time > 0 else 0
                elapsed_str = f"{elapsed:.1f}s" if elapsed > 0 else ""
                worker_table.add_row(f"[{wid}]", stage, model, ctx_size, elapsed_str, elem)
            else:
                worker_table.add_row(f"[{wid}]", "[dim]idle[/]", "", "", "", "")

        # Per-type stats
        type_colors = {
            "file": "cyan",
            "class": "magenta",
            "function": "blue",
            "method": "green",
            "constant": "yellow",
            "variable": "red",
        }
        type_stats = state.timing.get_type_stats()
        type_parts = []
        for t in ["file", "class", "function", "method", "constant", "variable"]:
            if t in type_stats:
                done, tot, avg_wall, avg_summ, avg_embed = type_stats[t]
                api_time = avg_summ + avg_embed
                color = type_colors.get(t, "white")
                if done >= tot:
                    type_parts.append(f"[{color}]{t}[/]: [green]{done}/{tot}[/] [dim]({api_time:.1f}s)[/]")
                else:
                    type_parts.append(f"[{color}]{t}[/]: [yellow]{done}/{tot}[/] [dim]({api_time:.1f}s)[/]")
        type_line = f"  [dim]Progress:[/] {' [dim]|[/] '.join(type_parts)}" if type_parts else ""

        # Stats line
        api_processed = state.completed - state.skipped
        effective_wall = state.timing.elapsed / api_processed if api_processed > 0 else 0.0
        total_api = state.timing.avg_summarize_time + state.timing.avg_embed_time
        summ_emb = state.timing.avg_summary_embed_time
        code_emb = state.timing.avg_code_embed_time
        stats = f"  [dim]Throughput:[/] [green]{effective_wall:.2f}s[/]/item [dim]|[/] [dim]API:[/] [green]{total_api:.1f}s[/]/item [dim]([/][green]{state.timing.avg_summarize_time:.1f}s[/] summ + [cyan]{summ_emb:.1f}s[/] summ_emb + [cyan]{code_emb:.1f}s[/] code_emb[dim])[/]"

        # Parallelism stats (compact, next to throughput)
        # Use fresh running count from workers_data (already fetched above)
        running_count = len(workers_data)

        # Check if all workers are on the same context tier
        # Filter out "-" (used during embed stage) and empty strings
        ctx_sizes = {data[3] for data in workers_data.values() if data[3] and data[3] != "-" and data[3].endswith("K")}
        if len(ctx_sizes) == 1:
            # All on same tier - show tier-specific stats with throttle info
            ctx_str = ctx_sizes.pop()  # e.g., "2K", "4K"
            try:
                ctx_int = int(ctx_str.rstrip("K")) * 1024  # "4K" -> 4096
                tier_limit = TIER_MAX_WORKERS.get(ctx_int, num_workers)

                # Check for runtime-based throttling
                parallelism = state.parallelism
                if parallelism and parallelism.throttle_decision and parallelism.throttle_decision.should_throttle:
                    # Show throttled state with effective limit
                    effective = parallelism.throttle_decision.recommended_workers
                    stats += f" [dim]|[/] [dim]Workers:[/] [green]{running_count}[/]/[red]{effective}[/]"
                    stats += f" [dim]([/][red]throttled: {parallelism.throttle_decision.reason}[/][dim])[/]"
                elif parallelism and parallelism.tier_changing:
                    # Waiting for tier change
                    stats += f" [dim]|[/] [dim]Workers:[/] [green]{running_count}[/]/[yellow]0[/]"
                    stats += f" [dim]([/][yellow]tier changing...[/][dim])[/]"
                else:
                    throttled = num_workers - tier_limit
                    stats += f" [dim]|[/] [dim]Workers:[/] [green]{running_count}[/]/[cyan]{tier_limit}[/]"
                    if throttled > 0:
                        stats += f" [dim]([/][yellow]{throttled} throttled[/][dim])[/]"
            except ValueError:
                # Fallback if parsing fails
                stats += f" [dim]|[/] [dim]Workers:[/] [green]{running_count}[/]/[cyan]{num_workers}[/]"
        else:
            # Mixed tiers or no workers - just show running/max
            parallelism = state.parallelism
            if parallelism and parallelism.tier_changing:
                stats += f" [dim]|[/] [dim]Workers:[/] [green]{running_count}[/]/[yellow]0[/] [dim]([/][yellow]tier changing...[/][dim])[/]"
            else:
                stats += f" [dim]|[/] [dim]Workers:[/] [green]{running_count}[/]/[cyan]{num_workers}[/]"

        # Show max runtime info if available
        parallelism = state.parallelism
        if parallelism and parallelism.throttle_decision:
            td = parallelism.throttle_decision
            if td.current_max > 0 or td.historical_max > 0:
                stats += f" [dim]|[/] [dim]Max:[/] [cyan]{td.current_max:.1f}s[/]"
                if td.historical_max > 0:
                    stats += f" [dim](hist:[/] [yellow]{td.historical_max:.1f}s[/][dim])[/]"

        parts: list[RenderableType] = [bar_text, worker_table]
        if type_line:
            parts.append(type_line)
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

    # Initialize state
    current_state = ProgressState(
        total=total,
        completed=0,
        skipped=0,
        failed=0,
        timing=timing_stats,
        workers=worker_status,
        num_workers=display_workers,
    )

    class LiveDisplay:
        def __rich__(self) -> RenderableType:
            return build_display(current_state, display_workers)

    with Live(LiveDisplay(), console=console, refresh_per_second=10) as live:
        def on_progress(state: ProgressState) -> None:
            nonlocal current_state
            current_state = state
            live.refresh()

        def on_status_change() -> None:
            live.refresh()

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
            magaldi_config=config if not dry_run else None,
        )

    # Get timing stats from current state
    avg_summ = current_state.timing.avg_summarize_time
    avg_embed = current_state.timing.avg_embed_time
    elapsed = current_state.timing.elapsed
    api_processed = result.elements_processed - result.elements_skipped
    avg_wall = elapsed / api_processed if api_processed > 0 else 0.0

    # Phase 2 call resolution: resolve cross-file calls using imports and type annotations
    if result.indexed > 0:
        from magaldi_core.call_resolution import resolve_cross_file_calls

        console.print("\n  [bold]Call Resolution[/]")
        try:
            total_calls, import_resolved, type_resolved = resolve_cross_file_calls(
                es_repo,
                manifest.scope,
                manifest.repository,
                manifest.username,
            )
            total_resolved = import_resolved + type_resolved
            console.print(f"  Cross-file: {total_resolved}/{total_calls} resolved")
            console.print(f"    via imports: {import_resolved}, via type annotations: {type_resolved}")
        except Exception as e:
            console.print(f"  [yellow]Warning: Call resolution failed: {e}[/]")

    # Total deleted = from deleted files + stale elements from modified files
    total_deleted = deleted_from_files + result.elements_deleted
    return (result.elements_processed, result.elements_skipped, result.indexed, avg_wall, avg_summ, avg_embed, elapsed, timing_stats, result.failed_elements, total_deleted)


def run_hierarchy_extraction(
    scope: str,
    repository: str,
    username: str,
    es_repo: "ElasticsearchRepository",
    cli_entry_point: str | None = None,
    api_prefix: str = "/api/v1",
) -> tuple[int, int]:
    """Run hierarchy extraction: CLI commands and HTTP routes.

    Args:
        scope: Repository scope
        repository: Repository name
        username: Username/branch
        es_repo: Elasticsearch repository instance
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

    client = es_repo._get_client()
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

    # Store in relationships index (use same config as es_repo)
    rel_repo = RelationshipsRepository(es_repo.config)

    # Delete existing relationships for this user before re-indexing
    rel_repo.delete_relationships_for_user(scope, repository, username)
    rel_repo.delete_external_refs_for_user(scope, repository, username)

    # Index new relationships and external refs
    rel_result = rel_repo.bulk_index_relationships(all_relationships)
    ref_result = rel_repo.bulk_index_external_refs(all_external_refs)

    return (rel_result["indexed"], ref_result["indexed"])
