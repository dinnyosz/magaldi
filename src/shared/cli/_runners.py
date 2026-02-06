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

        # ETA breakdown per (type, tier) - show avg time per item in a grid table
        eta_breakdown = state.timing.get_eta_breakdown_with_avg(state.num_workers)
        eta_table = None
        if eta_breakdown:
            tier_abbrev = {2048: "2k", 4096: "4k", 8192: "8k", 16384: "16k", 32768: "32k"}
            type_abbrev = {"function": "fn", "method": "mth", "class": "cls", "file": "file", "variable": "var", "constant": "const"}
            tiers = [32768, 16384, 8192, 4096, 2048]
            type_order = ["file", "class", "function", "method", "variable", "constant", "import"]

            # Build lookup from breakdown data
            eta_data: dict[tuple[str, int], tuple[float, bool, int, int]] = {}
            for elem_type, tier, avg_time, is_fallback, done, total in eta_breakdown:
                eta_data[(elem_type, tier)] = (avg_time, is_fallback, done, total)

            # Create grid table: rows=types, columns=tiers
            eta_table = Table(show_header=True, box=None, padding=(0, 2), expand=False)
            eta_table.add_column("", style="dim", width=10)  # type column
            tier_colors = {32768: "magenta", 16384: "blue", 8192: "cyan", 4096: "green", 2048: "yellow"}
            for tier in tiers:
                color = tier_colors.get(tier, "white")
                eta_table.add_column(f"[{color}]{tier_abbrev.get(tier, f'{tier//1024}k')}[/]", justify="center")

            # Type colors for row labels
            type_colors = {"file": "cyan", "class": "magenta", "function": "blue", "method": "green", "variable": "yellow", "constant": "red", "import": "bright_black"}

            # Add rows for each element type that has data
            for elem_type in type_order:
                has_data = any((elem_type, t) in eta_data for t in tiers)
                if not has_data:
                    continue

                type_color = type_colors.get(elem_type, "white")
                row = [f"[{type_color}]{elem_type}[/]"]
                for tier in tiers:
                    if (elem_type, tier) in eta_data:
                        avg_time, is_fallback, done, total = eta_data[(elem_type, tier)]
                        # Color progress: green if done, yellow if in progress
                        if done >= total:
                            count_str = f"[green]{done}/{total}[/]"
                        elif done > 0:
                            count_str = f"[yellow]{done}[/][dim]/{total}[/]"
                        else:
                            count_str = f"[dim]{done}/{total}[/]"
                        # Time styling (time first, then count)
                        if avg_time > 0:
                            time_style = "dim cyan" if is_fallback else "cyan"
                            time_str = f"[{time_style}]{avg_time:.1f}s[/]"
                            if is_fallback:
                                time_str = f"~{time_str}"
                            cell = f"{time_str} {count_str}"
                        else:
                            cell = f"[dim]-[/] {count_str}"
                    else:
                        cell = ""
                    row.append(cell)
                eta_table.add_row(*row)

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

        # Per-type stats
        type_colors = {
            "file": "cyan",
            "class": "magenta",
            "function": "blue",
            "method": "green",
            "constant": "yellow",
            "variable": "red",
            "import": "bright_black",
        }
        type_stats = state.timing.get_type_stats()
        type_parts = []
        for t in ["file", "class", "function", "method", "constant", "variable", "import"]:
            if t in type_stats:
                done, tot, avg_wall, avg_summ, avg_embed = type_stats[t]
                color = type_colors.get(t, "white")
                if done >= tot:
                    type_parts.append(f"[{color}]{t}[/]: [green]{done}/{tot}[/] [dim]({avg_wall:.1f}s)[/]")
                else:
                    type_parts.append(f"[{color}]{t}[/]: [yellow]{done}/{tot}[/] [dim]({avg_wall:.1f}s)[/]")
        type_line = f"  [dim]Progress:[/] {' [dim]|[/] '.join(type_parts)}" if type_parts else ""

        # Stats line
        # Only count actually processed elements for throughput (exclude unchanged/skipped)
        processed_count = state.completed - state.skipped
        effective_wall = state.timing.elapsed / processed_count if processed_count > 0 else 0.0
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
        if not compact:
            parts.append(worker_table)
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
            es_repo,
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


def run_call_resolution(
    es_repo: "ElasticsearchRepository",
    scope: str,
    repository: str,
    username: str,
    skip_resolve: bool = False,
    console: "Console | None" = None,
) -> None:
    """Run Phase 5: Call Resolution (static + embedding + semantic relationships).

    Args:
        es_repo: Elasticsearch repository instance.
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
                es_repo, scope, repository, username,
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
                es_repo, scope, repository, username,
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
            es_repo, scope, repository, username,
        )
        console.print(f"  Processed {elements_processed} elements, stored {total_relationships} relationships")
    except Exception as e:
        console.print(f"  [yellow]Warning: Semantic relationships failed: {e}[/]")
