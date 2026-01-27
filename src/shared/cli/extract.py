"""Extract commands for the Magaldi CLI.

This module contains the extract-features and extract-glossary commands,
along with their runner functions.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click
from rich.console import Group, RenderableType
from rich.live import Live
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)
from rich.table import Table
from rich.text import Text

from shared.cli._printers import print_feature_result
from shared.cli._shared import console, format_duration, get_model_column_width, main
from shared.config import load_config

if TYPE_CHECKING:
    from shared.config import MagaldiConfig
    from shared.db.elasticsearch import ElasticsearchRepository


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
    from shared.ai.context_size import TIER_MAX_WORKERS

    # Build display string
    parts = []
    for tier in sorted(tier_counts.keys()):
        count = tier_counts[tier]
        tier_str = f"{tier // 1024}K"
        max_workers = TIER_MAX_WORKERS.get(tier, 1)
        parts.append(f"[cyan]{tier_str}[/]: {count} ({max_workers}w)")

    distribution = " | ".join(parts)
    console.print(f"  [dim]Context tiers:[/] {distribution}")


# =============================================================================
# FEATURE EXTRACTION RUNNER
# =============================================================================


def run_feature_extraction(
    scope: str,
    repository: str,
    username: str,
    config: MagaldiConfig,
    min_cluster_size: int = 5,
    min_samples: int = 3,
    skip_labeling: bool = False,
    workers: int = 4,
) -> dict | None:
    """Run Phase 5: Feature Extraction.

    Returns:
        Dict with feature extraction results or None if no elements to extract.
    """
    # Handle workers=0 (auto) - use tier-based default like Phase 4
    if workers <= 0:
        from shared.ai.context_size import TIER_MAX_WORKERS
        workers = max(TIER_MAX_WORKERS.values())

    from shared.ai.clustering.clusterer import (
        ClusterConfig,
        FeatureClusterer,
        LabelingProgressState,
        LabelingTimingStats,
    )
    from shared.ai.clustering.feature_processor import (
        FeatureProcessingConfig,
        FeatureProgressState,
        FeatureTimingStats,
        FeatureWorkerStatus,
        SubClusterConfig,
        SubfeatureLabelingState,
        SubfeatureProgressState,
        SubfeatureTimingStats,
        SubfeatureWorkerStatus,
        process_features,
        process_subfeatures,
    )
    from shared.db.elasticsearch import ElasticsearchRepository

    es_repo = ElasticsearchRepository(config)

    try:
        # Fetch embeddings for functions/methods only
        with console.status("[bold blue]Fetching embeddings...[/]"):
            elements = es_repo.get_all_embeddings(
                scope=scope,
                repository=repository,
                username=username,
                element_types=["function", "method"],
            )

        if not elements:
            console.print("  [dim]No functions/methods with embeddings found[/]")
            return None

        console.print(f"  Found {len(elements)} functions/methods with embeddings")

        # Run HDBSCAN clustering
        summarize_model = config.llm.get_summarize_model()
        cluster_config = ClusterConfig(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            api_base=summarize_model.get_api_base(),
            labeling_model=summarize_model.name,
            provider=summarize_model.provider,
        )

        clusterer = FeatureClusterer(cluster_config)

        # Rename summary_embedding to embedding for clusterer compatibility
        for elem in elements:
            if "summary_embedding" in elem:
                elem["embedding"] = elem.pop("summary_embedding")

        with console.status("[bold blue]Clustering...[/]"):
            clustering_result = clusterer.cluster(elements)

        # Calculate coverage stats
        elements_in_features = sum(c.size for c in clustering_result.clusters)
        coverage_pct = (elements_in_features / clustering_result.total_elements * 100) if clustering_result.total_elements > 0 else 0

        console.print(
            f"  Found [green]{clustering_result.cluster_count}[/] features "
            f"covering [green]{elements_in_features}[/]/{clustering_result.total_elements} "
            f"functions/methods ([green]{coverage_pct:.0f}%[/]) | "
            f"{clustering_result.outlier_count} unclustered"
        )

        if not clustering_result.clusters:
            return {
                "cluster_count": 0,
                "outlier_count": clustering_result.outlier_count,
                "total_elements": clustering_result.total_elements,
                "elements_covered": 0,
                "coverage_pct": 0,
                "clusters": [],
            }

        # Label features with Ollama (quick labels)
        if not skip_labeling:
            console.print(f"  Labeling {clustering_result.cluster_count} features...")

            labeling_timing = LabelingTimingStats()

            def build_labeling_display(state: LabelingProgressState) -> RenderableType:
                """Build Rich display for labeling progress."""
                pct = (state.completed / state.total * 100) if state.total > 0 else 0
                eta = state.timing.eta_seconds(state.completed, state.total)
                elapsed_str = format_duration(state.timing.elapsed)

                # Progress bar
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

                clustering_result = clusterer.label_clusters(
                    clustering_result,
                    on_progress=on_labeling_progress,
                    timing_stats=labeling_timing,
                    scope=scope,
                    repository=repository,
                    username=username,
                    magaldi_config=config,
                )

        # Clear existing feature assignments
        es_repo.clear_cluster_assignments(scope, repository, username)

        # Build assignments for ES update
        assignments = []
        for cluster in clustering_result.clusters:
            for element_id in cluster.element_ids:
                assignments.append({
                    "element_id": element_id,
                    "cluster_id": str(cluster.cluster_id),
                    "cluster_label": cluster.label,
                })

        # Update ES with feature assignments
        if assignments:
            with console.status("[bold blue]Saving cluster assignments...[/]"):
                es_repo.update_cluster_assignments(assignments)

        # Process features: summarize -> embed -> index (with progress)
        # Fetch member summaries for tier estimation
        all_member_ids = []
        for cluster in clustering_result.clusters:
            all_member_ids.extend(cluster.element_ids)
        member_summaries = es_repo.get_summaries_batch(all_member_ids)

        # Display tier distribution before processing
        from shared.ai.clustering.feature_processor import (
            FEATURE_SYSTEM_PROMPT,
            FEATURE_USER_PROMPT,
        )
        from shared.ai.context_size import compute_aggregation_num_ctx

        def estimate_feature_tier(cluster: Any) -> int:
            summaries_text = []
            for i, element_id in enumerate(cluster.element_ids[:30]):
                summary = member_summaries.get(element_id, "")
                name = cluster.element_names[i] if i < len(cluster.element_names) else "unknown"
                if summary:
                    summaries_text.append(f"- {name}(): {summary}")
            if not summaries_text:
                summaries_text = [f"- {name}()" for name in cluster.element_names[:10]]
            user_content = FEATURE_USER_PROMPT.format(
                label=cluster.label or f"cluster_{cluster.cluster_id}",
                member_count=cluster.size,
                member_summaries="\n".join(summaries_text),
            )
            prompt_chars = len(FEATURE_SYSTEM_PROMPT) + len(user_content)
            return compute_aggregation_num_ctx(prompt_chars, task_type="feature")

        display_tier_distribution(clustering_result.clusters, estimate_feature_tier)
        console.print(f"  Processing {clustering_result.cluster_count} features with {workers} workers...")

        summarize_model = config.llm.get_summarize_model()
        embed_model = config.llm.get_embed_model()
        proc_config = FeatureProcessingConfig(
            summarize_model=summarize_model.name,
            embed_model=embed_model.name,
            api_base=summarize_model.get_api_base() or "",
            provider=summarize_model.provider,
            api_key=summarize_model.api_key,
            num_workers=workers,
        )

        timing_stats = FeatureTimingStats()
        worker_status = FeatureWorkerStatus()
        model_col_width = get_model_column_width(config)

        def build_feature_display(state: FeatureProgressState, num_workers: int) -> RenderableType:
            """Build Rich display for feature processing progress."""
            pct = (state.completed / state.total * 100) if state.total > 0 else 0
            eta = state.timing.eta_seconds(state.completed, state.total, state.num_workers)
            elapsed_str = format_duration(state.timing.elapsed)

            # Progress bar
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
            import time as time_mod
            worker_table = Table(show_header=False, box=None, padding=0)
            worker_table.add_column("ID", style="dim", width=4)
            worker_table.add_column("Stage", style="cyan", width=12)
            worker_table.add_column("Model", style="yellow", width=model_col_width)
            worker_table.add_column("Ctx", style="magenta", width=4)
            worker_table.add_column("Time", style="green", width=6)
            worker_table.add_column("Feature")

            workers_data = state.workers.get_all()
            now = time_mod.time()
            for wid in range(num_workers):
                if wid in workers_data:
                    feature_name, stage, model, ctx_size, start_time = workers_data[wid]
                    elapsed = now - start_time if start_time > 0 else 0
                    elapsed_str = f"{elapsed:.1f}s" if elapsed > 0 else ""
                    worker_table.add_row(f"[{wid}]", stage, model, ctx_size, elapsed_str, feature_name)
                else:
                    worker_table.add_row(f"[{wid}]", "[dim]idle[/]", "", "", "", "")

            # Stats line - show both wall time and API time
            avg_api = state.timing.avg_summarize_time + state.timing.avg_embed_time
            wall_time = state.timing.elapsed / state.completed if state.completed > 0 else 0
            stats = f"  [dim]Wall:[/] [green]{wall_time:.2f}s[/]/feature [dim]|[/] [dim]API:[/] [green]{avg_api:.1f}s[/]/feature [dim]([/][green]{state.timing.avg_summarize_time:.1f}s[/] summ + [green]{state.timing.avg_embed_time:.1f}s[/] embed[dim])[/]"

            return Group(bar_text, worker_table, stats)

        current_state = FeatureProgressState(
            total=clustering_result.cluster_count,
            completed=0,
            failed=0,
            timing=timing_stats,
            workers=worker_status,
            num_workers=workers,
        )

        class LiveFeatureDisplay:
            def __rich__(self) -> RenderableType:
                return build_feature_display(current_state, workers)

        with Live(LiveFeatureDisplay(), console=console, refresh_per_second=10) as live:
            def on_progress(state: FeatureProgressState) -> None:
                nonlocal current_state
                current_state = state
                live.refresh()

            def on_status_change() -> None:
                live.refresh()

            proc_result = process_features(
                clustering_result=clustering_result,
                scope=scope,
                repository=repository,
                username=username,
                es_repo=es_repo,
                config=proc_config,
                on_progress=on_progress,
                on_status_change=on_status_change,
                worker_status=worker_status,
                timing_stats=timing_stats,
                magaldi_config=config,
            )

        # Process sub-features for large clusters (>20 members)
        processed_features = proc_result.get("processed_features", {})
        large_cluster_count = sum(
            1 for c in clustering_result.clusters
            if c.size > 20
        )

        subfeature_result = {"subfeatures_created": 0, "parent_features_processed": 0}
        if large_cluster_count > 0:
            console.print(f"\n[bold cyan]Phase 6: Sub-feature Processing[/] ({large_cluster_count} large features)")

            sub_timing_stats = SubfeatureTimingStats()
            sub_worker_status = SubfeatureWorkerStatus()

            # Labeling state for discovery phase
            import time as time_mod
            labeling_phase_start = time_mod.time()
            current_labeling_state = SubfeatureLabelingState(
                total_features=large_cluster_count,
                features_processed=0,
                current_feature="",
                subclusters_labeled=0,
                model="",
                phase_start=labeling_phase_start,
            )

            def build_sub_labeling_display(state: SubfeatureLabelingState) -> RenderableType:
                """Build Rich display for subfeature labeling progress (matches Phase 5 style)."""
                import time as time_mod
                pct = (state.features_processed / state.total_features * 100) if state.total_features > 0 else 0
                elapsed = time_mod.time() - state.phase_start if state.phase_start > 0 else 0
                elapsed_str = format_duration(elapsed)

                # Progress bar (no header - header is printed before Live display)
                bar_width = 30
                filled = int(bar_width * pct / 100)
                bar = "━" * filled + "╺" + "─" * (bar_width - filled - 1) if filled < bar_width else "━" * bar_width
                bar_text = Text()
                bar_text.append("  ")
                bar_text.append(bar[:filled], style="green")
                if filled < bar_width:
                    bar_text.append(bar[filled:], style="dim")
                bar_text.append(" ")
                bar_text.append(f"{state.features_processed}", style="green")
                bar_text.append("/", style="dim")
                bar_text.append(f"{state.total_features}", style="cyan")
                bar_text.append(f" ({pct:.0f}%)", style="green")
                bar_text.append(" | ", style="dim")
                bar_text.append(elapsed_str, style="cyan")
                bar_text.append(" elapsed", style="dim")

                # Current feature and model
                current_text = Text()
                if state.current_feature:
                    current_text.append("  labeling  ", style="cyan")
                    if state.model:
                        current_text.append(f"{state.model}  ", style="yellow")
                    current_text.append(state.current_feature, style="white")
                else:
                    current_text.append("  ", style="dim")

                # Stats line (matches Phase 5 format)
                stats_text = Text()
                if state.subclusters_labeled > 0:
                    stats_text.append("  ")
                    stats_text.append(f"{state.subclusters_labeled} sub-clusters labeled", style="green")

                return Group(bar_text, current_text, stats_text)

            # Combined live display will be set later
            combined_live = None

            def on_labeling_progress(state: SubfeatureLabelingState) -> None:
                nonlocal current_labeling_state
                current_labeling_state = state
                if combined_live:
                    combined_live.refresh()

            def build_subfeature_display(state: SubfeatureProgressState, num_workers: int) -> RenderableType:
                """Build Rich display for subfeature processing progress (matches Phase 5)."""
                pct = (state.completed / state.total * 100) if state.total > 0 else 0
                eta = state.timing.eta_seconds(state.completed, state.total, state.num_workers)
                elapsed_str = format_duration(state.timing.elapsed)

                # Header (plain text like Phase 5)
                header_text = Text()
                header_text.append(f"  Summarizing {state.total} sub-features with {num_workers} workers...")

                # Progress bar
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
                import time as time_mod
                worker_table = Table(show_header=False, box=None, padding=0)
                worker_table.add_column("ID", style="dim", width=4)
                worker_table.add_column("Stage", style="cyan", width=10)
                worker_table.add_column("Model", style="yellow", width=model_col_width)
                worker_table.add_column("Ctx", style="blue", width=4)
                worker_table.add_column("Time", style="green", width=6)
                worker_table.add_column("Parent", style="magenta", width=20)
                worker_table.add_column("Subfeature")

                workers_data = state.workers.get_all()
                now = time_mod.time()
                for wid in range(num_workers):
                    if wid in workers_data:
                        parent_feature, stage, model, subfeature, ctx_size, start_time = workers_data[wid]
                        elapsed = now - start_time if start_time > 0 else 0
                        elapsed_str = f"{elapsed:.1f}s" if elapsed > 0 else ""
                        display_parent = parent_feature[:17] + "..." if len(parent_feature) > 20 else parent_feature
                        display_sub = subfeature[:28] + "..." if len(subfeature) > 31 else subfeature
                        worker_table.add_row(f"[{wid}]", stage, model, ctx_size, elapsed_str, display_parent, display_sub)
                    else:
                        worker_table.add_row(f"[{wid}]", "[dim]idle[/]", "", "", "", "", "")

                # Stats line - show both wall time and API time
                avg_api = state.timing.avg_summarize_time + state.timing.avg_embed_time
                wall_time = state.timing.elapsed / state.completed if state.completed > 0 else 0
                stats_text = Text()
                if state.timing.summarize_count > 0:
                    stats_text.append("  ")
                    stats_text.append("Wall: ", style="dim")
                    stats_text.append(f"{wall_time:.2f}s", style="green")
                    stats_text.append("/subfeature | ", style="dim")
                    stats_text.append("API: ", style="dim")
                    stats_text.append(f"{avg_api:.1f}s", style="green")
                    stats_text.append("/subfeature (", style="dim")
                    stats_text.append(f"{state.timing.avg_summarize_time:.1f}s", style="cyan")
                    stats_text.append(" summ + ", style="dim")
                    stats_text.append(f"{state.timing.avg_embed_time:.1f}s", style="cyan")
                    stats_text.append(" embed)", style="dim")
                    if state.failed > 0:
                        stats_text.append(" | ", style="dim")
                        stats_text.append(f"{state.failed} failed", style="red")

                return Group(header_text, bar_text, worker_table, stats_text)

            current_sub_state = SubfeatureProgressState(
                total=0,
                completed=0,
                failed=0,
                timing=sub_timing_stats,
                workers=sub_worker_status,
                num_workers=workers,
            )

            # Track which phase we're in
            in_processing_phase = False
            labeling_elapsed = 0.0

            def build_labeling_summary(state: SubfeatureLabelingState) -> RenderableType:
                """Build summary of completed labeling phase (matches Phase 5 'Avg' line style)."""
                # Calculate avg time per feature
                avg_time = labeling_elapsed / state.total_features if state.total_features > 0 else 0

                # Empty line + stats line (like Phase 5)
                empty_line = Text("")
                stats_line = Text()
                stats_line.append("  ")
                stats_line.append("Avg: ", style="dim")
                stats_line.append(f"{avg_time:.1f}s", style="green")
                stats_line.append("/feature | ", style="dim")
                stats_line.append(f"{state.subclusters_labeled}", style="cyan")
                stats_line.append(" sub-clusters labeled", style="dim")

                return Group(empty_line, stats_line)

            class LiveSubfeatureDisplay:
                def __rich__(self) -> RenderableType:
                    if in_processing_phase:
                        # Show labeling summary + processing display
                        return Group(
                            build_labeling_summary(current_labeling_state),
                            build_subfeature_display(current_sub_state, workers)
                        )
                    else:
                        return build_sub_labeling_display(current_labeling_state)

            # Print header before Live display (matches Phase 5 style)
            console.print(f"  Finding and labeling sub-clusters in {large_cluster_count} large features...")

            with Live(LiveSubfeatureDisplay(), console=console, refresh_per_second=10) as live:
                combined_live = live  # Make accessible to on_labeling_progress

                def on_sub_progress(state: SubfeatureProgressState) -> None:
                    nonlocal current_sub_state, in_processing_phase, labeling_elapsed
                    if not in_processing_phase:
                        in_processing_phase = True
                        # Capture final labeling elapsed time
                        labeling_elapsed = time_mod.time() - current_labeling_state.phase_start
                    current_sub_state = state
                    live.refresh()

                def on_subfeature_tier_distribution(tier_counts: dict[int, int]) -> None:
                    """Display subfeature tier distribution before processing."""
                    live.console.print()
                    _display_tier_counts(tier_counts)

                subfeature_result = process_subfeatures(
                    clustering_result=clustering_result,
                    processed_features=processed_features,
                    scope=scope,
                    repository=repository,
                    username=username,
                    es_repo=es_repo,
                    config=proc_config,
                    subcluster_config=SubClusterConfig(),
                    on_progress=on_sub_progress,
                    on_labeling_progress=on_labeling_progress,
                    magaldi_config=config,
                    timing_stats=sub_timing_stats,
                    worker_status=sub_worker_status,
                    on_tier_distribution=on_subfeature_tier_distribution,
                )

            if subfeature_result.get("subfeatures_created", 0) > 0:
                console.print(
                    f"  [green]Created {subfeature_result['subfeatures_created']} sub-features "
                    f"from {subfeature_result['parent_features_processed']} large features[/]"
                )

        return {
            "cluster_count": clustering_result.cluster_count,
            "outlier_count": clustering_result.outlier_count,
            "total_elements": clustering_result.total_elements,
            "elements_covered": elements_in_features,
            "coverage_pct": coverage_pct,
            "processed": proc_result.get("processed", 0),
            "failed": proc_result.get("failed", 0),
            "elapsed": proc_result.get("elapsed", 0),
            "subfeatures_created": subfeature_result.get("subfeatures_created", 0),
            "parent_features_with_subfeatures": subfeature_result.get("parent_features_processed", 0),
            "clusters": [
                {
                    "cluster_id": c.cluster_id,
                    "label": c.label,
                    "size": c.size,
                    "sample_names": c.element_names[:5],
                }
                for c in clustering_result.clusters
            ],
        }

    finally:
        es_repo.close()


# =============================================================================
# GLOSSARY EXTRACTION RUNNER
# =============================================================================


def run_glossary_extraction(
    scope: str,
    repository: str,
    username: str,
    config: MagaldiConfig,
    es_repo: ElasticsearchRepository | None = None,
    workers: int = 8,
) -> dict | None:
    """Run Glossary Extraction using AI-powered extraction from feature summaries.

    Args:
        scope: Repository scope.
        repository: Repository name.
        username: Username/branch.
        config: Magaldi configuration.
        es_repo: Optional ES repository (creates one if not provided).
        workers: Number of concurrent workers.

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
    from shared.db.elasticsearch import ElasticsearchRepository

    own_es_repo = es_repo is None
    if es_repo is None:
        es_repo = ElasticsearchRepository(config)

    try:
        # Fetch features and subfeatures
        with console.status("[bold blue]Fetching features...[/]"):
            features = es_repo.get_features(scope, repository, username)
            subfeatures = es_repo.get_subfeatures(scope, repository, username)

        all_features = features + subfeatures

        if not all_features:
            console.print("  [dim]No features found[/]")
            return None

        console.print(f"  Found {len(features)} features, {len(subfeatures)} subfeatures")

        # Delete existing glossary entries BEFORE extraction (fresh start)
        with console.status("[bold blue]Clearing existing glossary...[/]"):
            deleted = es_repo.delete_glossary(scope, repository, username)
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
            bar_text.append(" | ", style="dim")
            bar_text.append(elapsed_str, style="cyan")
            bar_text.append(" elapsed", style="dim")
            if eta > 0:
                bar_text.append(" | ~", style="dim")
                bar_text.append(format_duration(eta), style="yellow")
                bar_text.append(" ETA", style="dim")

            # Worker table
            worker_table = Table(show_header=False, box=None, padding=0)
            worker_table.add_column("ID", style="dim", width=4)
            worker_table.add_column("Stage", style="cyan", width=14)
            worker_table.add_column("Model", style="yellow", width=model_col_width)
            worker_table.add_column("Ctx", style="magenta", width=4)
            worker_table.add_column("Item")

            workers_data = state.workers.get_all()
            for wid in range(num_workers):
                if wid in workers_data:
                    item_label, model, ctx_size = workers_data[wid]
                    stage = "summarizing" if "summar" in phase.lower() else "extracting"
                    worker_table.add_row(f"[{wid}]", stage, model, ctx_size, item_label)
                else:
                    worker_table.add_row(f"[{wid}]", "[dim]idle[/]", "", "", "")

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

            return Group(phase_text, bar_text, worker_table, stats)

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
        console.print(f"  [green]{len(glossary_items)}[/] unique terms extracted in [cyan]{format_duration(elapsed)}[/]")
        console.print(f"  Wall: [green]{wall_time:.2f}s[/]/feature | API: [green]{avg_api:.1f}s[/]/feature | Model: [yellow]{model_name}[/]")

        # Build feature lookup for feature associations
        feature_lookup: dict[str, dict] = {}
        for feature in all_features:
            fid = feature.get("feature_id") or feature.get("subfeature_id", "")
            if fid:
                feature_lookup[fid] = {
                    "label": feature.get("label", ""),
                    "member_count": feature.get("member_count", 0),
                }

        # Index new glossary entries with progress
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]Indexing glossary[/]"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("({task.completed}/{task.total})"),
            console=console,
        ) as progress:
            task = progress.add_task("Indexing", total=len(glossary_items))

            for item in glossary_items:
                glossary_id = f"{scope}:{repository}:{username}:glossary:{item.name}"

                # Build feature associations (source features this term was extracted from)
                feature_associations = []
                for fid in item.source_feature_ids:
                    if fid in feature_lookup:
                        feature_data = feature_lookup[fid]
                        feature_associations.append({
                            "feature_id": fid,
                            "feature_label": feature_data["label"],
                            # Legacy fields kept for API compatibility
                            "frequency": 1,
                            "total_members": 0,
                            "percentage": 0.0,
                        })

                es_repo.index_glossary(
                    glossary_id=glossary_id,
                    scope=scope,
                    repository=repository,
                    username=username,
                    term=item.name,
                    total_count=len(item.source_feature_ids),
                    element_ids=item.source_feature_ids,
                    file_paths=[],
                    description=item.description,
                    feature_associations=feature_associations,
                )
                progress.update(task, advance=1)

        console.print(f"  Indexed [green]{len(glossary_items)}[/] glossary entries")

        return {
            "terms_count": len(glossary_items),
            "terms": [item.name for item in glossary_items],
        }

    finally:
        if own_es_repo:
            es_repo.close()


# =============================================================================
# EXTRACT-FEATURES COMMAND
# =============================================================================


@main.command("extract-features")
@click.argument("repo_path", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option("--user", "-u", required=True, help="Username/branch to extract features from")
@click.option("--min-cluster-size", default=5, type=int, help="Minimum elements per feature (default: 5)")
@click.option("--min-samples", default=3, type=int, help="HDBSCAN min_samples parameter (default: 3)")
@click.option("--skip-labeling", is_flag=True, help="Skip Ollama feature labeling")
@click.option("--workers", "-w", default=0, type=int, help="Max parallel workers (0=auto based on context tier)")
def extract_features(
    repo_path: str,
    user: str,
    min_cluster_size: int,
    min_samples: int,
    skip_labeling: bool,
    workers: int,
) -> None:
    """Extract features from indexed code elements.

    Groups semantically similar functions/methods into features
    using HDBSCAN clustering on vector embeddings.

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

    console.print("[bold blue]Feature Extraction[/]")
    console.print(f"  Repository: {scope}/{repository} @{user}")
    console.print()

    try:
        result = run_feature_extraction(
            scope=scope,
            repository=repository,
            username=user,
            config=config,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            skip_labeling=skip_labeling,
            workers=workers,
        )

        if result:
            print_feature_result(result)
            console.print("\n[green]Feature extraction complete.[/]")
        else:
            console.print("[yellow]No elements to extract features from.[/]")

    except Exception as e:
        console.print(f"\n[red]Error:[/] {e}")
        sys.exit(1)


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
