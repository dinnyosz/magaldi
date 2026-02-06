"""Feature extraction CLI commands for the Magaldi CLI.

This module contains the extract-features command and the feature extraction runner.
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

from shared.cli._printers import print_feature_result
from shared.cli._shared import console, format_duration, get_model_column_width, main
from shared.cli.extract import (
    _run_clustering_phase,
    _run_labeling_phase,
    build_eta_table,
    build_progress_bar,
    display_tier_distribution,
    _display_tier_counts,
)
from shared.config import load_config

if TYPE_CHECKING:
    from shared.config import MagaldiConfig


# =============================================================================
# FEATURE EXTRACTION RUNNER
# =============================================================================


def run_feature_extraction(
    scope: str,
    repository: str,
    username: str,
    config: MagaldiConfig,
    skip_labeling: bool = False,
    workers: int = 4,
    compact: bool = False,
) -> dict | None:
    """Run Phase 6: Feature Extraction.

    Args:
        compact: If True, hide worker table in display (for watch mode).

    Returns:
        Dict with feature extraction results or None if no elements to extract.
    """
    # Handle workers=0 (auto) - use tier-based default like Phase 4
    if workers <= 0:
        from shared.ai.context_size import TIER_MAX_WORKERS
        workers = max(TIER_MAX_WORKERS.values())

    from shared.ai.clustering.clusterer import ClusterConfig, FeatureClusterer
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

        # Run HDBSCAN clustering - config from magaldi.yaml
        summarize_model = config.llm.get_summarize_model()
        cluster_config = ClusterConfig.from_magaldi_config(
            config=config,
            api_base=summarize_model.get_api_base(),
            labeling_model=summarize_model.name,
            provider=summarize_model.provider,
        )

        clusterer = FeatureClusterer(cluster_config)

        # Rename summary_embedding to embedding for clusterer compatibility
        for elem in elements:
            if "summary_embedding" in elem:
                elem["embedding"] = elem.pop("summary_embedding")

        # Run clustering with progress display
        clustering_result = _run_clustering_phase(elements, clusterer)

        # Calculate coverage stats
        elements_in_features = sum(c.size for c in clustering_result.clusters)
        coverage_pct = (elements_in_features / clustering_result.total_elements * 100) if clustering_result.total_elements > 0 else 0

        console.print(
            f"  Found [green]{clustering_result.cluster_count}[/] features "
            f"covering [green]{elements_in_features}[/]/{clustering_result.total_elements} "
            f"functions/methods ([green]{coverage_pct:.0f}%[/]) | "
            f"{clustering_result.outlier_count} unclustered"
        )

        # Show feature affinity stats if soft clustering was used
        if clustering_result.is_soft_clustering and clustering_result.clusters:
            # Count features with connections
            features_with_connections = sum(
                1 for c in clustering_result.clusters if c.connected_clusters
            )
            # Total connections
            total_connections = sum(
                len(c.connected_clusters) for c in clustering_result.clusters
            )
            # Average connections per feature
            avg_connections = (
                total_connections / len(clustering_result.clusters)
                if clustering_result.clusters
                else 0
            )
            # Elements with cross-membership (>1 membership above threshold)
            elements_with_overlap = sum(
                1 for memberships in clustering_result.element_memberships.values()
                if len(memberships) > 1
            )
            overlap_pct = (
                elements_with_overlap / elements_in_features * 100
                if elements_in_features > 0
                else 0
            )

            console.print(
                f"  Feature affinity: [cyan]{features_with_connections}[/]/{clustering_result.cluster_count} "
                f"features connected (avg [cyan]{avg_connections:.1f}[/] connections) | "
                f"[cyan]{elements_with_overlap}[/] elements overlap ([cyan]{overlap_pct:.0f}%[/])"
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
            clustering_result = _run_labeling_phase(
                clusterer, clustering_result, scope, repository, username, config
            )

        # Clear existing feature assignments
        es_repo.clear_cluster_assignments(scope, repository, username)

        # Build assignments for ES update
        assignments = []
        for cluster in clustering_result.clusters:
            for element_id in cluster.element_ids:
                assignment: dict[str, Any] = {
                    "element_id": element_id,
                    "cluster_id": str(cluster.cluster_id),
                    "cluster_label": cluster.label,
                }

                # Add soft memberships if available (from soft clustering)
                if clustering_result.is_soft_clustering and element_id in clustering_result.element_memberships:
                    memberships = clustering_result.element_memberships[element_id]
                    assignment["feature_memberships"] = [
                        {
                            "feature_id": f"feature_{m.cluster_id}",
                            "label": None,  # Will be filled after labeling
                            "score": m.score,
                            "is_primary": m.is_primary,
                        }
                        for m in memberships
                    ]

                assignments.append(assignment)

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
            import time as time_mod

            # Use actual allowed workers for ETA (accounts for throttling)
            effective_workers = state.allowed_workers if state.allowed_workers > 0 else state.num_workers
            eta = state.timing.eta_seconds(state.completed, state.total, effective_workers)

            # Use common progress bar builder
            bar_text = build_progress_bar(state.completed, state.total, state.timing.elapsed, eta)

            # Use common ETA table builder
            eta_breakdown = state.timing.get_eta_breakdown_with_avg(state.num_workers)
            eta_table = build_eta_table(eta_breakdown, ["feature"], {"feature": "blue"})

            # Worker table
            workers_data = state.workers.get_all()
            now = time_mod.time()
            allowed_workers = state.allowed_workers if state.allowed_workers > 0 else num_workers

            worker_table = Table(show_header=False, box=None, padding=0)
            worker_table.add_column("ID", style="dim", width=4)
            worker_table.add_column("Stage", style="cyan", width=12)
            worker_table.add_column("Model", style="yellow", width=model_col_width)
            worker_table.add_column("Ctx", style="magenta", width=4)
            worker_table.add_column("Time", style="green", width=6)
            worker_table.add_column("Feature")

            for wid in range(num_workers):
                if wid in workers_data:
                    feature_name, stage, model, ctx_size, start_time = workers_data[wid]
                    elapsed = now - start_time if start_time > 0 else 0
                    elapsed_str = f"{elapsed:.1f}s" if elapsed > 0 else ""
                    worker_table.add_row(f"[{wid}]", stage, model, ctx_size, elapsed_str, feature_name)
                elif wid < allowed_workers:
                    worker_table.add_row(f"[{wid}]", "[dim]idle[/]", "", "", "", "")
                else:
                    worker_table.add_row(f"[{wid}]", "[dim yellow]throttled[/]", "", "", "", "")

            # Stats line
            avg_api = state.timing.avg_summarize_time + state.timing.avg_embed_time
            wall_time = state.timing.elapsed / state.completed if state.completed > 0 else 0
            running_count = len(workers_data)

            stats = f"  [dim]Wall:[/] [green]{wall_time:.2f}s[/]/feature [dim]|[/] [dim]API:[/] [green]{avg_api:.1f}s[/]/feature [dim]([/][green]{state.timing.avg_summarize_time:.1f}s[/] summ + [green]{state.timing.avg_embed_time:.1f}s[/] embed[dim])[/]"
            if allowed_workers < num_workers:
                stats += f" [dim]|[/] [dim]Workers:[/] [green]{running_count}[/]/[yellow]{allowed_workers}[/]/[cyan]{num_workers}[/]"
            else:
                stats += f" [dim]|[/] [dim]Workers:[/] [green]{running_count}[/]/[cyan]{num_workers}[/]"

            if state.current_max > 0 or state.avg_base_time > 0:
                normalized_max = state.current_max / max(running_count, 1)
                stats += f" [dim]|[/] [dim]Max:[/] [yellow]{state.current_max:.1f}s[/]"
                stats += f" [dim]|[/] [dim]Per Worker:[/] [yellow]{normalized_max:.1f}s[/] [dim]vs[/] [cyan]{state.avg_base_time:.1f}s[/] [dim](last {state.completion_count})[/]"

            elements = [bar_text]
            if eta_table:
                elements.append(eta_table)
            if not compact:
                elements.append(worker_table)
            elements.append(stats)
            return Group(*elements)

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

        # Update element feature_memberships with proper feature_ids and labels
        # (they were saved with placeholders before feature summarization)
        cluster_id_to_feature: dict[str, tuple[str, str]] = {}
        for cluster in clustering_result.clusters:
            cluster_id = str(cluster.cluster_id)
            if cluster_id in processed_features:
                pf = processed_features[cluster_id]
                label = pf.get("label", cluster.label or f"cluster_{cluster_id}")
                feature_id = f"{scope}:{repository}:{username}:feature:{label}:{cluster_id}"
                cluster_id_to_feature[cluster_id] = (feature_id, label)

        if cluster_id_to_feature:
            with console.status("[bold blue]Updating element feature memberships...[/]"):
                updated = es_repo.update_element_feature_memberships(
                    scope, repository, username, cluster_id_to_feature
                )
            console.print(f"  Updated [green]{updated}[/] elements with feature labels")

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
                """Build Rich display for subfeature labeling progress."""
                import time as time_mod
                elapsed = time_mod.time() - state.phase_start if state.phase_start > 0 else 0
                bar_text = build_progress_bar(state.features_processed, state.total_features, elapsed)

                # Current feature and model
                current_text = Text()
                if state.current_feature:
                    current_text.append("  labeling  ", style="cyan")
                    if state.model:
                        current_text.append(f"{state.model}  ", style="yellow")
                    current_text.append(state.current_feature, style="white")
                else:
                    current_text.append("  ", style="dim")

                # Stats line
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
                """Build Rich display for subfeature processing progress."""
                import time as time_mod

                effective_workers = state.allowed_workers if state.allowed_workers > 0 else state.num_workers
                eta = state.timing.eta_seconds(state.completed, state.total, effective_workers)

                # Header
                header_text = Text()
                header_text.append(f"  Summarizing {state.total} sub-features with {num_workers} workers...")

                # Use common progress bar and ETA table builders
                bar_text = build_progress_bar(state.completed, state.total, state.timing.elapsed, eta)
                eta_breakdown = state.timing.get_eta_breakdown_with_avg(state.num_workers)
                eta_table = build_eta_table(eta_breakdown, ["subfeature"], {"subfeature": "green"})

                # Worker table
                workers_data = state.workers.get_all()
                now = time_mod.time()
                allowed_workers = state.allowed_workers if state.allowed_workers > 0 else num_workers

                worker_table = Table(show_header=False, box=None, padding=(0, 1))
                worker_table.add_column("ID", style="dim", width=4)
                worker_table.add_column("Stage", style="cyan", width=10)
                worker_table.add_column("Model", style="yellow", width=model_col_width)
                worker_table.add_column("Ctx", style="blue", width=4)
                worker_table.add_column("Time", style="green", width=6)
                worker_table.add_column("Parent", style="magenta", width=28)
                worker_table.add_column("Subfeature")

                for wid in range(num_workers):
                    if wid in workers_data:
                        parent_feature, stage, model, subfeature, ctx_size, start_time = workers_data[wid]
                        elapsed = now - start_time if start_time > 0 else 0
                        elapsed_str = f"{elapsed:.1f}s" if elapsed > 0 else ""
                        display_parent = parent_feature[:25] + "..." if len(parent_feature) > 28 else parent_feature
                        display_sub = subfeature[:35] + "..." if len(subfeature) > 38 else subfeature
                        worker_table.add_row(f"[{wid}]", stage, model, ctx_size, elapsed_str, display_parent, display_sub)
                    elif wid < allowed_workers:
                        worker_table.add_row(f"[{wid}]", "[dim]idle[/]", "", "", "", "", "")
                    else:
                        worker_table.add_row(f"[{wid}]", "[dim yellow]throttled[/]", "", "", "", "", "")

                # Stats line
                avg_api = state.timing.avg_summarize_time + state.timing.avg_embed_time
                wall_time = state.timing.elapsed / state.completed if state.completed > 0 else 0
                running_count = len(workers_data)
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
                    stats_text.append(" | ", style="dim")
                    stats_text.append("Workers: ", style="dim")
                    stats_text.append(f"{running_count}", style="green")
                    stats_text.append("/", style="dim")
                    if allowed_workers < num_workers:
                        stats_text.append(f"{allowed_workers}", style="yellow")
                        stats_text.append("/", style="dim")
                    stats_text.append(f"{num_workers}", style="cyan")

                    if state.current_max > 0 or state.avg_base_time > 0:
                        normalized_max = state.current_max / max(running_count, 1)
                        stats_text.append(" | ", style="dim")
                        stats_text.append("Max: ", style="dim")
                        stats_text.append(f"{state.current_max:.1f}s", style="yellow")
                        stats_text.append(" | ", style="dim")
                        stats_text.append("Per Worker: ", style="dim")
                        stats_text.append(f"{normalized_max:.1f}s", style="yellow")
                        stats_text.append(" vs ", style="dim")
                        stats_text.append(f"{state.avg_base_time:.1f}s", style="cyan")
                        stats_text.append(f" (last {state.completion_count})", style="dim")

                elements = [header_text, bar_text]
                if eta_table:
                    elements.append(eta_table)
                if not compact:
                    elements.append(worker_table)
                elements.append(stats_text)
                return Group(*elements)

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
# EXTRACT-FEATURES COMMAND
# =============================================================================


@main.command("extract-features")
@click.argument("repo_path", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option("--user", "-u", required=True, help="Username/branch to extract features from")
@click.option("--skip-labeling", is_flag=True, help="Skip Ollama feature labeling")
@click.option("--workers", "-w", default=0, type=int, help="Max parallel workers (0=auto based on context tier)")
def extract_features(
    repo_path: str,
    user: str,
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
