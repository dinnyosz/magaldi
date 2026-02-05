"""Extract command utilities for the Magaldi CLI.

This module contains shared utilities for feature and glossary extraction:
- Tier distribution display
- Common display utilities (progress bars, ETA tables, worker tables)
- Phase helpers for clustering and labeling

The actual CLI commands and runner functions are in:
- feature_commands.py: extract-features command and run_feature_extraction
- glossary_commands.py: extract-glossary command and run_glossary_extraction

For backward compatibility, run_feature_extraction and run_glossary_extraction
are re-exported from this module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rich.console import Group, RenderableType
from rich.live import Live
from rich.table import Table
from rich.text import Text

from shared.cli._shared import console, format_duration

if TYPE_CHECKING:
    from shared.config import MagaldiConfig


# =============================================================================
# TIER DISTRIBUTION DISPLAY
# =============================================================================


def display_tier_distribution(
    items: list[Any],
    tier_fn: callable,
) -> None:
    """Display tier distribution before processing.

    Args:
        items: List of items to process.
        tier_fn: Function that takes an item and returns its context tier.
    """
    from collections import Counter

    # Count items per tier
    tier_counts: Counter[int] = Counter()
    for item in items:
        tier = tier_fn(item)
        tier_counts[tier] += 1

    _display_tier_counts(dict(tier_counts))


def _display_tier_counts(tier_counts: dict[int, int]) -> None:
    """Display tier counts dict.

    Args:
        tier_counts: Dict mapping tier (e.g., 2048, 4096) to count.
    """
    # Build display string - just counts, no worker limits (time-based throttling handles scaling)
    parts = []
    for tier in sorted(tier_counts.keys()):
        count = tier_counts[tier]
        tier_str = f"{tier // 1024}K"
        parts.append(f"[cyan]{tier_str}[/]: {count}")

    distribution = " | ".join(parts)
    console.print(f"  [dim]Context tiers:[/] {distribution}")


# =============================================================================
# COMMON DISPLAY UTILITIES
# =============================================================================


def build_progress_bar(
    completed: int,
    total: int,
    elapsed: float,
    eta: float | None = None,
    bar_width: int = 30,
) -> Text:
    """Build a common progress bar with elapsed and ETA.

    Args:
        completed: Number of completed items.
        total: Total number of items.
        elapsed: Elapsed time in seconds.
        eta: Estimated time remaining in seconds.
        bar_width: Width of the progress bar.

    Returns:
        Rich Text object with the progress bar.
    """
    pct = (completed / total * 100) if total > 0 else 0
    filled = int(bar_width * pct / 100)
    bar = "━" * filled + "╺" + "─" * (bar_width - filled - 1) if filled < bar_width else "━" * bar_width

    bar_text = Text()
    bar_text.append("  ")
    bar_text.append(bar[:filled], style="green")
    if filled < bar_width:
        bar_text.append(bar[filled:], style="dim")
    bar_text.append(" ")
    bar_text.append(f"{completed}", style="green")
    bar_text.append("/", style="dim")
    bar_text.append(f"{total}", style="cyan")
    bar_text.append(f" ({pct:.0f}%)", style="green")
    bar_text.append(" | ", style="dim")
    bar_text.append(format_duration(elapsed), style="cyan")
    bar_text.append(" elapsed", style="dim")
    if eta is not None and eta > 0:
        bar_text.append(" | ~", style="dim")
        bar_text.append(format_duration(eta), style="yellow")
        bar_text.append(" ETA", style="dim")

    return bar_text


def build_eta_table(
    eta_breakdown: list[tuple[str, int, float, bool, int, int]],
    type_order: list[str],
    type_colors: dict[str, str] | None = None,
) -> Table | None:
    """Build ETA breakdown table per tier.

    Args:
        eta_breakdown: List of (elem_type, tier, avg_time, is_fallback, done, total).
        type_order: Order of element types to display.
        type_colors: Optional mapping of elem_type to color.

    Returns:
        Rich Table or None if no breakdown data.
    """
    if not eta_breakdown:
        return None

    tier_abbrev = {2048: "2k", 4096: "4k", 8192: "8k", 16384: "16k", 32768: "32k"}
    tiers = [32768, 16384, 8192, 4096, 2048]
    type_colors = type_colors or {}

    # Build lookup from breakdown data
    eta_data: dict[tuple[str, int], tuple[float, bool, int, int]] = {}
    for elem_type, tier, avg_time, is_fallback, done, total in eta_breakdown:
        eta_data[(elem_type, tier)] = (avg_time, is_fallback, done, total)

    # Create grid table: rows=types, columns=tiers
    eta_table = Table(show_header=True, box=None, padding=(0, 2), expand=False)
    eta_table.add_column("", style="dim", width=10)
    tier_colors_map = {32768: "magenta", 16384: "blue", 8192: "cyan", 4096: "green", 2048: "yellow"}

    for tier in tiers:
        if any((t, tier) in eta_data for t in type_order):
            color = tier_colors_map.get(tier, "white")
            eta_table.add_column(f"[{color}]{tier_abbrev.get(tier, f'{tier//1024}k')}[/]", justify="center")

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

    return eta_table


def build_worker_table(
    workers_data: dict[int, tuple],
    num_workers: int,
    allowed_workers: int,
    model_col_width: int,
    columns: list[tuple[str, str, int]],
    row_builder: callable,
) -> Table:
    """Build worker status table.

    Args:
        workers_data: Dict of worker_id -> tuple of worker data.
        num_workers: Total number of workers.
        allowed_workers: Number of allowed workers (after throttling).
        model_col_width: Width of model column.
        columns: List of (name, style, width) for columns.
        row_builder: Function(wid, workers_data) -> list of row values.

    Returns:
        Rich Table with worker status.
    """
    import time as time_mod

    worker_table = Table(show_header=False, box=None, padding=0)
    worker_table.add_column("ID", style="dim", width=4)
    for name, style, width in columns:
        worker_table.add_column(name, style=style, width=width)

    now = time_mod.time()

    for wid in range(num_workers):
        if wid in workers_data:
            row = row_builder(wid, workers_data, now)
            worker_table.add_row(f"[{wid}]", *row)
        elif wid < allowed_workers:
            idle_row = ["[dim]idle[/]"] + [""] * (len(columns) - 1)
            worker_table.add_row(f"[{wid}]", *idle_row)
        else:
            throttled_row = ["[dim yellow]throttled[/]"] + [""] * (len(columns) - 1)
            worker_table.add_row(f"[{wid}]", *throttled_row)

    return worker_table


# =============================================================================
# PHASE HELPERS
# =============================================================================


def _run_clustering_phase(
    elements: list[dict],
    clusterer: Any,
) -> Any:
    """Run HDBSCAN clustering with Live progress display.

    Args:
        elements: List of elements with embeddings.
        clusterer: FeatureClusterer instance.

    Returns:
        ClusteringResult from the clusterer.
    """
    from shared.ai.clustering.clusterer import ClusteringProgressState

    # Track clustering progress state
    clustering_state: dict[str, Any] = {
        "phase": "starting",
        "phase_description": "Initializing",
        "current_step": 0,
        "total_steps": 0,
        "n_elements": len(elements),
        "n_features": 0,
        "elapsed_seconds": 0,
        "eta_seconds": None,
    }

    def build_clustering_display() -> RenderableType:
        """Build Rich display for clustering progress."""
        phase = clustering_state["phase"]
        current = clustering_state["current_step"]
        total = clustering_state["total_steps"]
        elapsed = clustering_state.get("elapsed_seconds", 0)
        eta = clustering_state.get("eta_seconds")

        # Use common progress bar builder
        bar_text = build_progress_bar(current, total, elapsed, eta)

        # Phase description
        status_text = Text()
        status_text.append("  ")
        if phase == "hdbscan":
            status_text.append("🔬 ", style="yellow")
            status_text.append("HDBSCAN soft clustering", style="cyan")
            status_text.append(f" | {clustering_state['n_elements']} elements", style="dim")
        elif phase == "complete":
            status_text.append("✓ ", style="green")
            status_text.append("Clustering complete", style="green")
            status_text.append(f" | {clustering_state['n_features']} features", style="dim")
        else:
            status_text.append("⏳ ", style="yellow")
            status_text.append("Initializing...", style="cyan")

        return Group(bar_text, status_text)

    def on_clustering_progress(state: ClusteringProgressState) -> None:
        """Update clustering state from callback."""
        clustering_state["phase"] = state.phase
        clustering_state["phase_description"] = state.phase_description
        clustering_state["current_step"] = state.current_step
        clustering_state["total_steps"] = state.total_steps
        clustering_state["n_elements"] = state.n_elements
        clustering_state["n_features"] = state.n_features
        clustering_state["cooccurrence_density"] = state.cooccurrence_density
        clustering_state["elapsed_seconds"] = state.elapsed_seconds
        clustering_state["eta_seconds"] = state.eta_seconds

    with Live(build_clustering_display(), console=console, refresh_per_second=10) as live:
        def update_display(state: ClusteringProgressState) -> None:
            on_clustering_progress(state)
            live.update(build_clustering_display())

        return clusterer.cluster(elements, on_progress=update_display)


def _run_labeling_phase(
    clusterer: Any,
    clustering_result: Any,
    scope: str,
    repository: str,
    username: str,
    config: MagaldiConfig,
) -> Any:
    """Run feature labeling with Live progress display.

    Args:
        clusterer: FeatureClusterer instance.
        clustering_result: Result from clustering phase.
        scope: Repository scope.
        repository: Repository name.
        username: Username/branch.
        config: Magaldi configuration.

    Returns:
        Updated ClusteringResult with labels.
    """
    from shared.ai.clustering.clusterer import LabelingProgressState, LabelingTimingStats

    console.print(f"  Labeling {clustering_result.cluster_count} features...")

    labeling_timing = LabelingTimingStats()

    def build_labeling_display(state: LabelingProgressState) -> RenderableType:
        """Build Rich display for labeling progress."""
        eta = state.timing.eta_seconds(state.completed, state.total)
        bar_text = build_progress_bar(state.completed, state.total, state.timing.elapsed, eta)

        # Current cluster line with model and context size
        current_text = Text()
        if state.current_cluster:
            current_text.append("  labeling  ", style="cyan")
            if state.model:
                current_text.append(f"{state.model}  ", style="yellow")
            if state.ctx_size:
                current_text.append(f"{state.ctx_size}  ", style="magenta")
            current_text.append(state.current_cluster, style="white")
        else:
            current_text.append("  ", style="dim")

        # Stats line
        stats_text = Text()
        if state.timing.label_count > 0:
            stats_text.append("  ")
            stats_text.append("Avg: ", style="dim")
            stats_text.append(f"{state.timing.avg_label_time:.1f}s", style="green")
            stats_text.append("/label", style="dim")
            if state.skipped > 0:
                stats_text.append(" | ", style="dim")
                stats_text.append(f"{state.skipped} skipped", style="yellow")
            if state.failed > 0:
                stats_text.append(" | ", style="dim")
                stats_text.append(f"{state.failed} failed", style="red")

        return Group(bar_text, current_text, stats_text)

    labeling_state = LabelingProgressState(
        total=clustering_result.cluster_count,
        completed=0,
        skipped=0,
        failed=0,
        timing=labeling_timing,
        current_cluster="",
    )

    class LiveLabelingDisplay:
        def __rich__(self) -> RenderableType:
            return build_labeling_display(labeling_state)

    with Live(LiveLabelingDisplay(), console=console, refresh_per_second=10) as live:
        def on_labeling_progress(state: LabelingProgressState) -> None:
            nonlocal labeling_state
            labeling_state = state
            live.refresh()

        return clusterer.label_clusters(
            clustering_result,
            on_progress=on_labeling_progress,
            timing_stats=labeling_timing,
            scope=scope,
            repository=repository,
            username=username,
            magaldi_config=config,
        )


# =============================================================================
# BACKWARD COMPATIBILITY RE-EXPORTS
# =============================================================================
# Import and re-export run_feature_extraction and run_glossary_extraction
# from their new modules to maintain backward compatibility with existing code
# that imports from shared.cli.extract


def run_feature_extraction(
    scope: str,
    repository: str,
    username: str,
    config: MagaldiConfig,
    skip_labeling: bool = False,
    workers: int = 4,
    compact: bool = False,
) -> dict | None:
    """Run Phase 5: Feature Extraction.

    This is a re-export for backward compatibility.
    The actual implementation is in feature_commands.py.
    """
    from shared.cli.feature_commands import run_feature_extraction as _run_feature_extraction
    return _run_feature_extraction(
        scope=scope,
        repository=repository,
        username=username,
        config=config,
        skip_labeling=skip_labeling,
        workers=workers,
        compact=compact,
    )


def run_glossary_extraction(
    scope: str,
    repository: str,
    username: str,
    config: MagaldiConfig,
    es_repo: Any = None,
    workers: int = 8,
    compact: bool = False,
) -> dict | None:
    """Run Glossary Extraction using AI-powered extraction from feature summaries.

    This is a re-export for backward compatibility.
    The actual implementation is in glossary_commands.py.
    """
    from shared.cli.glossary_commands import run_glossary_extraction as _run_glossary_extraction
    return _run_glossary_extraction(
        scope=scope,
        repository=repository,
        username=username,
        config=config,
        es_repo=es_repo,
        workers=workers,
        compact=compact,
    )
