"""Glossary extraction CLI commands for the Magaldi CLI.

This module contains the extract-glossary command and the glossary extraction runner.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click
from rich.console import Group, RenderableType
from rich.live import Live
from rich.table import Table
from rich.text import Text

from shared.cli._shared import console, format_duration, get_model_column_width, main
from shared.cli.extract import (
    display_tier_distribution,
)
from shared.config import load_config

if TYPE_CHECKING:
    from shared.config import MagaldiConfig
    from shared.db.store import Repository


# =============================================================================
# GLOSSARY EXTRACTION RUNNER
# =============================================================================


def run_glossary_extraction(
    scope: str,
    repository: str,
    username: str,
    config: MagaldiConfig,
    repo: Repository | None = None,
    workers: int = 8,
    compact: bool = False,
) -> dict | None:
    """Run Glossary Extraction using AI-powered extraction from feature summaries.

    Args:
        scope: Repository scope.
        repository: Repository name.
        username: Username/branch.
        config: Magaldi configuration.
        repo: Optional ES repository (creates one if not provided).
        workers: Number of concurrent workers.
        compact: If True, hide worker table in display (for watch mode).

    Returns:
        Dict with glossary extraction results or None if no features.
    """
    # Handle workers=0 (auto) - use tier-based default like Phase 4
    if workers <= 0:
        from shared.ai.context_size import TIER_MAX_WORKERS
        workers = max(TIER_MAX_WORKERS.values())

    from shared.ai.glossary.ai_extractor import (
        GlossaryProgressState,
        GlossaryTimingStats,
        GlossaryWorkerStatus,
        extract_glossary_from_features_concurrent,
    )
    from shared.db.store import Repository

    own_repo = repo is None
    if repo is None:
        repo = Repository(config)

    try:
        # Fetch features and subfeatures
        with console.status("[bold blue]Fetching features...[/]"):
            features = repo.get_features(scope, repository, username)
            subfeatures = repo.get_subfeatures(scope, repository, username)

        all_features = features + subfeatures

        if not all_features:
            console.print("  [dim]No features found[/]")
            return None

        console.print(f"  Found {len(features)} features, {len(subfeatures)} subfeatures")

        # Delete existing glossary entries BEFORE extraction (fresh start)
        with console.status("[bold blue]Clearing existing glossary...[/]"):
            deleted = repo.delete_glossary(scope, repository, username)
            if deleted > 0:
                console.print(f"  Deleted {deleted} existing entries")

        # Get model name for display
        model_name = config.llm.get_summarize_model().name

        # Track current phase
        current_phase = {"name": "Extracting terms from features"}
        model_col_width = get_model_column_width(config)

        def build_glossary_display(state: GlossaryProgressState, num_workers: int, phase: str) -> RenderableType:
            """Build Rich display for glossary extraction progress."""
            # Phase header
            phase_text = Text()
            phase_text.append("  ")
            phase_text.append(phase, style="bold magenta")

            # Progress info
            pct = (state.completed / state.total * 100) if state.total > 0 else 0
            # Use actual allowed workers for ETA (accounts for throttling)
            effective_workers = state.allowed_workers if state.allowed_workers > 0 else state.num_workers
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
            bar_text.append(" | ", style="dim")
            bar_text.append(elapsed_str, style="cyan")
            bar_text.append(" elapsed", style="dim")
            if eta > 0:
                bar_text.append(" | ~", style="dim")
                bar_text.append(format_duration(eta), style="yellow")
                bar_text.append(" ETA", style="dim")

            # ETA breakdown per tier - show as table like Phase 4
            eta_breakdown = state.timing.get_eta_breakdown_with_avg(state.num_workers)
            eta_table = None
            if eta_breakdown:
                tier_abbrev = {2048: "2k", 4096: "4k", 8192: "8k", 16384: "16k", 32768: "32k"}
                tiers = [32768, 16384, 8192, 4096, 2048]
                type_order = ["glossary"]

                # Build lookup from breakdown data
                eta_data: dict[tuple[str, int], tuple[float, bool, int, int]] = {}
                for elem_type, tier, avg_time, is_fallback, done, total in eta_breakdown:
                    eta_data[(elem_type, tier)] = (avg_time, is_fallback, done, total)

                # Create grid table: rows=types, columns=tiers
                eta_table = Table(show_header=True, box=None, padding=(0, 2), expand=False)
                eta_table.add_column("", style="dim", width=10)
                tier_colors = {32768: "magenta", 16384: "blue", 8192: "cyan", 4096: "green", 2048: "yellow"}
                for tier in tiers:
                    if any((t, tier) in eta_data for t in type_order):
                        color = tier_colors.get(tier, "white")
                        eta_table.add_column(f"[{color}]{tier_abbrev.get(tier, f'{tier//1024}k')}[/]", justify="center")

                type_colors = {"glossary": "magenta"}
                for elem_type in type_order:
                    has_data = any((elem_type, t) in eta_data for t in tiers)
                    if not has_data:
                        continue

                    type_color = type_colors.get(elem_type, "white")
                    row = [f"[{type_color}]{elem_type}[/]"]
                    for tier in tiers:
                        if not any((t, tier) in eta_data for t in type_order):
                            continue
                        if (elem_type, tier) in eta_data:
                            avg_time, is_fallback, done, total = eta_data[(elem_type, tier)]
                            if done >= total:
                                count_str = f"[green]{done}/{total}[/]"
                            elif done > 0:
                                count_str = f"[yellow]{done}[/][dim]/{total}[/]"
                            else:
                                count_str = f"[dim]{done}/{total}[/]"
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
            worker_table = Table(show_header=False, box=None, padding=0)
            worker_table.add_column("ID", style="dim", width=4)
            worker_table.add_column("Stage", style="cyan", width=14)
            worker_table.add_column("Model", style="yellow", width=model_col_width)
            worker_table.add_column("Ctx", style="magenta", width=4)
            worker_table.add_column("Item")

            workers_data = state.workers.get_all()

            # Use allowed_workers from throttle decision (0 = use num_workers)
            allowed_workers = state.allowed_workers if state.allowed_workers > 0 else num_workers

            for wid in range(num_workers):
                if wid in workers_data:
                    item_label, model, ctx_size = workers_data[wid]
                    stage = "summarizing" if "summar" in phase.lower() else "extracting"
                    worker_table.add_row(f"[{wid}]", stage, model, ctx_size, item_label)
                elif wid < allowed_workers:
                    worker_table.add_row(f"[{wid}]", "[dim]idle[/]", "", "", "")
                else:
                    worker_table.add_row(f"[{wid}]", "[dim yellow]throttled[/]", "", "", "")

            # Stats line - different labels for each phase
            avg_api = state.timing.avg_api_time
            wall_time = state.timing.elapsed / state.completed if state.completed > 0 else 0
            is_phase2 = "summary" in phase.lower()
            stats = Text()
            stats.append("  ")
            stats.append("Wall: ", style="dim")
            stats.append(f"{wall_time:.2f}s", style="green")
            if is_phase2:
                stats.append("/term | ", style="dim")
            else:
                stats.append("/feature | ", style="dim")
            stats.append("API: ", style="dim")
            stats.append(f"{avg_api:.1f}s", style="green")
            if is_phase2:
                stats.append("/term", style="dim")
            else:
                stats.append("/feature | ", style="dim")
                stats.append("Terms: ", style="dim")
                stats.append(f"{state.terms_extracted}", style="cyan")
            if state.failed > 0:
                stats.append(" | ", style="dim")
                stats.append(f"{state.failed} failed", style="red")

            # Worker count with throttle info
            running_count = len(workers_data)
            stats.append(" | ", style="dim")
            stats.append("Workers: ", style="dim")
            stats.append(f"{running_count}", style="green")
            stats.append("/", style="dim")
            if allowed_workers < num_workers:
                stats.append(f"{allowed_workers}", style="yellow")
                stats.append("/", style="dim")
            stats.append(f"{num_workers}", style="cyan")

            # Show throttle info
            if state.current_max > 0 or state.avg_base_time > 0:
                normalized_max = state.current_max / max(running_count, 1)
                stats.append(" | ", style="dim")
                stats.append("Max: ", style="dim")
                stats.append(f"{state.current_max:.1f}s", style="yellow")
                stats.append(" | ", style="dim")
                stats.append("Per Worker: ", style="dim")
                stats.append(f"{normalized_max:.1f}s", style="yellow")
                stats.append(" vs ", style="dim")
                stats.append(f"{state.avg_base_time:.1f}s", style="cyan")
                stats.append(f" (last {state.completion_count})", style="dim")

            # Build group with optional eta_table
            elements = [phase_text, bar_text]
            if eta_table:
                elements.append(eta_table)
            if not compact:
                elements.append(worker_table)
            elements.append(stats)
            return Group(*elements)

        # Create shared state objects
        timing_stats = GlossaryTimingStats()
        worker_status = GlossaryWorkerStatus()
        total = len(all_features)

        # Initialize state
        current_state = GlossaryProgressState(
            total=total,
            completed=0,
            failed=0,
            terms_extracted=0,
            timing=timing_stats,
            workers=worker_status,
            num_workers=workers,
        )

        class LiveGlossaryDisplay:
            """Wrapper that Rich can call to get current display."""
            def __rich__(self) -> RenderableType:
                return build_glossary_display(current_state, workers, current_phase["name"])

        # Display tier distribution for glossary extraction (Phase 1)
        from shared.ai.context_size import compute_aggregation_num_ctx
        from shared.ai.glossary.ai_extractor import (
            GLOSSARY_EXTRACTION_SYSTEM_PROMPT,
            GLOSSARY_EXTRACTION_USER_PROMPT,
        )

        def estimate_glossary_tier(feature: dict) -> int:
            label = feature.get("label", "")[:40]
            summary = feature.get("summary", "")
            user_content = GLOSSARY_EXTRACTION_USER_PROMPT.format(label=label, summary=summary)
            prompt_chars = len(GLOSSARY_EXTRACTION_SYSTEM_PROMPT) + len(user_content)
            return compute_aggregation_num_ctx(prompt_chars, task_type="glossary_extract")

        display_tier_distribution(all_features, estimate_glossary_tier)

        # Print header before Live display
        console.print(f"  Processing {total} features with {workers} workers...")
        console.print("  Phase 1: Extract terms | Phase 2: Generate summaries")
        console.print()

        # Track Phase 1 final stats for display after completion
        phase1_stats: dict[str, Any] = {}

        with Live(LiveGlossaryDisplay(), console=console, refresh_per_second=10) as live:
            def on_progress(state: GlossaryProgressState) -> None:
                nonlocal current_state
                current_state = state
                live.refresh()

            def on_status_change() -> None:
                live.refresh()

            def on_phase_change(phase_name: str) -> None:
                # If switching to Phase 2, print Phase 1 summary first
                if "summary" in phase_name.lower() and current_state.completed > 0:
                    # Capture Phase 1 stats before they get reset
                    phase1_stats["completed"] = current_state.completed
                    phase1_stats["elapsed"] = current_state.timing.elapsed
                    phase1_stats["avg_api"] = current_state.timing.avg_api_time
                    phase1_stats["terms"] = current_state.terms_extracted
                    phase1_stats["failed"] = current_state.failed

                    # Print Phase 1 completion summary above the live display
                    wall_time = phase1_stats["elapsed"] / phase1_stats["completed"] if phase1_stats["completed"] > 0 else 0
                    live.console.print()
                    live.console.print("  [bold green]✓ Phase 1 complete[/]")
                    live.console.print(f"    [green]{phase1_stats['completed']}[/] features processed in [cyan]{format_duration(phase1_stats['elapsed'])}[/]")
                    live.console.print(f"    [cyan]{phase1_stats['terms']}[/] terms extracted | Wall: [green]{wall_time:.2f}s[/]/feature | API: [green]{phase1_stats['avg_api']:.1f}s[/]/feature")
                    if phase1_stats["failed"] > 0:
                        live.console.print(f"    [red]{phase1_stats['failed']} failed[/]")
                    live.console.print()

                current_phase["name"] = phase_name
                live.refresh()

            glossary_items = extract_glossary_from_features_concurrent(
                all_features,
                config,
                num_workers=workers,
                on_progress=on_progress,
                on_status_change=on_status_change,
                worker_status=worker_status,
                timing_stats=timing_stats,
                on_phase_change=on_phase_change,
                # Incremental indexing - items are indexed as they complete in Phase 2
                repo=repo,
                scope=scope,
                repository=repository,
                username=username,
            )

        if not glossary_items:
            console.print("  [dim]No glossary items extracted[/]")
            return {"terms_count": 0}

        # Print summary stats - show both wall time and API time
        avg_api = timing_stats.avg_api_time
        elapsed = timing_stats.elapsed
        features_processed = timing_stats.features_processed
        wall_time = elapsed / features_processed if features_processed > 0 else 0
        console.print()
        console.print(f"  [green]{len(glossary_items)}[/] unique terms extracted and indexed in [cyan]{format_duration(elapsed)}[/]")
        console.print(f"  Wall: [green]{wall_time:.2f}s[/]/feature | API: [green]{avg_api:.1f}s[/]/feature | Model: [yellow]{model_name}[/]")

        return {
            "terms_count": len(glossary_items),
            "terms": [item.name for item in glossary_items],
        }

    finally:
        if own_repo:
            repo.close()


# =============================================================================
# EXTRACT-GLOSSARY COMMAND
# =============================================================================


@main.command("extract-glossary")
@click.argument("repo_path", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option("--user", "-u", required=True, help="Username/branch to extract glossary from")
@click.option("--workers", "-w", default=0, type=int, help="Max concurrent workers (0=auto based on context tier)")
def extract_glossary(
    repo_path: str,
    user: str,
    workers: int,
) -> None:
    """Extract glossary terms from feature summaries using AI.

    Analyzes feature and subfeature summaries to extract domain-specific
    terminology and stores them for enhanced code discovery.

    REPO_PATH is the path to the repository (used to load magaldi.yaml).
    """
    from magaldi_core.discovery import load_repo_config

    config = load_config(skip_validation=False)

    # Load repo config to get scope/repository
    repo_path_obj = Path(repo_path)
    repo_config_path = repo_path_obj / "magaldi.yaml"
    if not repo_config_path.exists():
        console.print(f"[red]Error:[/] magaldi.yaml not found in {repo_path}")
        sys.exit(1)

    repo_config = load_repo_config(repo_path_obj)
    scope = repo_config.scope
    repository = repo_config.name

    console.print("[bold blue]Glossary Extraction[/]")
    console.print(f"  Repository: {scope}/{repository} @{user}")
    console.print()

    try:
        result = run_glossary_extraction(
            scope=scope,
            repository=repository,
            username=user,
            config=config,
            workers=workers,
        )

        if result:
            terms_count = result.get("terms_count", 0)
            console.print(f"\n[green]Done![/] Extracted {terms_count} terms")
        else:
            console.print("[yellow]No features found to extract glossary from.[/]")

    except Exception as e:
        console.print(f"\n[red]Error:[/] {e}")
        sys.exit(1)
