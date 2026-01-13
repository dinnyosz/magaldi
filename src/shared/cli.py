"""Magaldi CLI - Code discovery engine for AI agents and developers.

Commands:
    magaldi parse /path/to/repo --user <username>
"""

from __future__ import annotations

# Suppress warnings from LiteLLM/aiohttp before any imports
import warnings
warnings.filterwarnings("ignore", category=ResourceWarning, message=".*[Uu]nclosed.*")
warnings.filterwarnings("ignore", message="Pydantic serializer warnings")

# Patch aiohttp to not emit unclosed session warnings
# This is necessary because aiohttp uses both warnings.warn AND loop.call_exception_handler
import aiohttp.client
import aiohttp.connector

_original_client_del = aiohttp.client.ClientSession.__del__
_original_connector_del = aiohttp.connector.BaseConnector.__del__

def _quiet_client_del(self, _warnings=None):
    """Suppress unclosed session warning."""
    pass  # Do nothing - session will be GC'd anyway

def _quiet_connector_del(self, _warnings=None):
    """Suppress unclosed connector warning."""
    pass  # Do nothing - connector will be GC'd anyway

aiohttp.client.ClientSession.__del__ = _quiet_client_del
aiohttp.connector.BaseConnector.__del__ = _quiet_connector_del

import sys

import click
from rich.console import Console, Group, RenderableType
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

from shared.config import MagaldiConfig, load_config
from magaldi_core.change_detection import (
    ChangeManifest,
    InMemoryFileStateRepository,
    detect_changes,
)
from magaldi_core.code_parser import ParsingResult, parse_files
from magaldi_core.discovery import DiscoveryError, DiscoveryResult, discover
from magaldi_core.processor import (
    ProcessingConfig,
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


def check_ollama_models(config: MagaldiConfig, skip_ai: bool) -> list[str]:
    """Check if required Ollama models are available.

    Returns:
        List of error messages (empty if all models are available).
    """
    if skip_ai:
        return []

    import requests

    errors = []
    url = config.llm.url.rstrip("/")

    # Check Ollama server is running
    try:
        response = requests.get(f"{url}/api/tags", timeout=5)
        response.raise_for_status()
        available_models = {m.get("name") for m in response.json().get("models", [])}
    except requests.exceptions.ConnectionError:
        return [f"Cannot connect to Ollama at {url}. Is it running?"]
    except Exception as e:
        return [f"Error connecting to Ollama: {e}"]

    def model_available(model: str) -> bool:
        """Check if model is available (handles :latest tag)."""
        if model in available_models:
            return True
        # Try with :latest suffix
        if f"{model}:latest" in available_models:
            return True
        # Try without tag if model has one
        if ":" in model:
            base = model.rsplit(":", 1)[0]
            if base in available_models or f"{base}:latest" in available_models:
                return True
        return False

    # Check required models
    required_models = [
        config.llm.summarize_model,
        config.llm.summarize_model_small,
        config.llm.embed_model,
    ]

    for model in required_models:
        if not model_available(model):
            errors.append(f"Model '{model}' not found. Run: ollama pull {model}")

    return errors


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
@click.option("--skip-features", is_flag=True, help="Skip feature extraction after processing")
@click.option("--dry-run", is_flag=True, help="Use in-memory storage (no database required)")
@click.option("--llm-url", default=None, help="LLM API URL (default: from config)")
@click.option("--workers", "-w", default=4, type=int, help="Number of parallel workers (default: 4)")
@click.option("--force-clean", is_flag=True, help="Delete all indexed data for this repo/user before parsing")
def parse(
    repo_path: str, user: str, skip_ai: bool, skip_features: bool, dry_run: bool,
    llm_url: str | None, workers: int, force_clean: bool
) -> None:
    """Parse a repository and index its code elements.

    REPO_PATH is the path to the repository to parse.
    """
    # Load configuration (skip validation in dry-run mode)
    config = load_config(skip_validation=dry_run)
    if llm_url:
        config.llm.url = llm_url

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
            from shared.db.elasticsearch import ElasticsearchRepository
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

        # Pre-flight check: Verify Ollama models are available
        if not dry_run and not skip_ai:
            model_errors = check_ollama_models(config, skip_ai)
            if model_errors:
                console.print("\n[red]Pre-flight check failed:[/]")
                for err in model_errors:
                    console.print(f"  [red]✗[/] {err}")
                sys.exit(1)

        # Phase 4: Processing (summarize -> embed -> index)
        console.print("\n[bold blue]Phase 4:[/] Processing")
        processed, skipped, indexed, avg_wall, avg_summ, avg_embed, elapsed, timing_stats, failed_elements = run_processing(
            parsing_result, manifest, config, dry_run, skip_ai, workers
        )
        print_processing_result(processed, skipped, indexed, skip_ai, avg_wall, avg_summ, avg_embed, elapsed, timing_stats, workers)
        # Display errors if any
        if failed_elements:
            console.print(f"\n  [red]Errors ({len(failed_elements)}):[/]")
            for element_id, error in failed_elements[:10]:  # Show first 10
                # Shorten element_id for display
                short_id = element_id.split(":")[-3:]  # type:name:line
                console.print(f"    [dim]{':'.join(short_id)}[/] → [red]{error}[/]")
            if len(failed_elements) > 10:
                console.print(f"    [dim]... and {len(failed_elements) - 10} more errors[/]")

        # Phase 5: Feature Extraction (optional)
        if not skip_features and not skip_ai and not dry_run and processed > 0:
            console.print("\n[bold blue]Phase 5:[/] Feature Extraction")
            feature_result = run_feature_extraction(
                discovery_result.scope,
                discovery_result.repository,
                user,
                config,
                workers=workers,
            )
            if feature_result:
                print_feature_result(feature_result)

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
# EXTRACT-FEATURES COMMAND
# =============================================================================


@main.command("extract-features")
@click.argument("repo_path", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option("--user", "-u", required=True, help="Username/branch to extract features from")
@click.option("--min-cluster-size", default=5, type=int, help="Minimum elements per feature (default: 5)")
@click.option("--min-samples", default=3, type=int, help="HDBSCAN min_samples parameter (default: 3)")
@click.option("--skip-labeling", is_flag=True, help="Skip Ollama feature labeling")
@click.option("--workers", "-w", default=4, type=int, help="Number of parallel workers (default: 4)")
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
    from pathlib import Path

    from magaldi_core.discovery import load_repo_config

    config = load_config(skip_validation=False)

    # Load repo config to get scope/repository
    repo_config_path = Path(repo_path) / "magaldi.yaml"
    if not repo_config_path.exists():
        console.print(f"[red]Error:[/] magaldi.yaml not found in {repo_path}")
        sys.exit(1)

    repo_config = load_repo_config(repo_config_path)
    scope = repo_config["scope"]
    repository = Path(repo_path).name

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
        cluster_config = ClusterConfig(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            api_base=config.llm.url,
            labeling_model=config.llm.summarize_model,
        )

        clusterer = FeatureClusterer(cluster_config)

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

                # Current cluster line with model
                current_text = Text()
                if state.current_cluster:
                    current_text.append("  [labeling] ", style="cyan")
                    if state.model:
                        current_text.append(f"[{state.model}] ", style="yellow")
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
        console.print(f"  Processing {clustering_result.cluster_count} features with {workers} workers...")

        proc_config = FeatureProcessingConfig(
            summarize_model=config.llm.summarize_model,
            embed_model=config.llm.embed_model,
            api_base=config.llm.url,
            provider=config.llm.provider,
            api_key=config.llm.api_key,
            num_workers=workers,
        )

        timing_stats = FeatureTimingStats()
        worker_status = FeatureWorkerStatus()

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
            worker_table = Table(show_header=False, box=None, padding=(0, 1))
            worker_table.add_column("ID", style="dim", width=4)
            worker_table.add_column("Stage", style="cyan", width=12)
            worker_table.add_column("Model", style="yellow")
            worker_table.add_column("Feature")

            workers_data = state.workers.get_all()
            for wid in range(num_workers):
                if wid in workers_data:
                    feature_name, stage, model = workers_data[wid]
                    worker_table.add_row(f"[{wid}]", stage, model, feature_name)
                else:
                    worker_table.add_row(f"[{wid}]", "[dim]idle[/]", "", "")

            # Stats line
            avg_api = state.timing.avg_summarize_time + state.timing.avg_embed_time
            stats = f"  [dim]Avg:[/] [green]{avg_api:.1f}s[/]/feature [dim]([/][green]{state.timing.avg_summarize_time:.1f}s[/] summ + [green]{state.timing.avg_embed_time:.1f}s[/] embed[dim])[/]"

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
            current_labeling_state = SubfeatureLabelingState(
                total_features=large_cluster_count,
                features_processed=0,
                current_feature="",
                subclusters_labeled=0,
                model="",
            )

            def build_labeling_display(state: SubfeatureLabelingState) -> RenderableType:
                """Build Rich display for subfeature labeling progress."""
                pct = (state.features_processed / state.total_features * 100) if state.total_features > 0 else 0

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
                bar_text.append(f"{state.features_processed}", style="green")
                bar_text.append("/", style="dim")
                bar_text.append(f"{state.total_features}", style="cyan")
                bar_text.append(f" ({pct:.0f}%)", style="green")
                bar_text.append(" features", style="dim")

                # Current feature and model
                current_text = Text()
                if state.current_feature:
                    current_text.append("  [labeling] ", style="cyan")
                    if state.model:
                        current_text.append(f"[{state.model}] ", style="yellow")
                    current_text.append(state.current_feature, style="white")
                else:
                    current_text.append("  ", style="dim")

                # Stats line
                stats_text = Text()
                if state.subclusters_labeled > 0:
                    stats_text.append("  ")
                    stats_text.append(f"{state.subclusters_labeled} subclusters labeled", style="green")

                return Group(bar_text, current_text, stats_text)

            # Combined live display will be set later
            combined_live = None

            def on_labeling_progress(state: SubfeatureLabelingState) -> None:
                nonlocal current_labeling_state
                current_labeling_state = state
                if combined_live:
                    combined_live.refresh()

            console.print("  Labeling and processing sub-features with {workers} workers...".format(workers=workers))

            def build_subfeature_display(state: SubfeatureProgressState, num_workers: int) -> RenderableType:
                """Build Rich display for subfeature processing progress (matches Phase 5)."""
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
                worker_table = Table(show_header=False, box=None, padding=(0, 1))
                worker_table.add_column("ID", style="dim", width=4)
                worker_table.add_column("Stage", style="cyan", width=12)
                worker_table.add_column("Model", style="yellow", width=22)
                worker_table.add_column("Parent", style="magenta", width=20)
                worker_table.add_column("Subfeature")

                workers_data = state.workers.get_all()
                for wid in range(num_workers):
                    if wid in workers_data:
                        parent_feature, stage, model, subfeature = workers_data[wid]
                        display_model = model[:19] + "..." if len(model) > 22 else model
                        display_parent = parent_feature[:17] + "..." if len(parent_feature) > 20 else parent_feature
                        display_sub = subfeature[:28] + "..." if len(subfeature) > 31 else subfeature
                        worker_table.add_row(f"[{wid}]", stage, display_model, display_parent, display_sub)
                    else:
                        worker_table.add_row(f"[{wid}]", "[dim]idle[/]", "", "", "")

                # Stats line
                avg_api = state.timing.avg_summarize_time + state.timing.avg_embed_time
                stats_text = Text()
                if state.timing.summarize_count > 0:
                    stats_text.append("  ")
                    stats_text.append("Avg: ", style="dim")
                    stats_text.append(f"{avg_api:.1f}s", style="green")
                    stats_text.append("/subfeature (", style="dim")
                    stats_text.append(f"{state.timing.avg_summarize_time:.1f}s", style="cyan")
                    stats_text.append(" summ + ", style="dim")
                    stats_text.append(f"{state.timing.avg_embed_time:.1f}s", style="cyan")
                    stats_text.append(" embed)", style="dim")
                    if state.failed > 0:
                        stats_text.append(" | ", style="dim")
                        stats_text.append(f"{state.failed} failed", style="red")

                return Group(bar_text, worker_table, stats_text)

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

            class LiveSubfeatureDisplay:
                def __rich__(self) -> RenderableType:
                    if in_processing_phase:
                        return build_subfeature_display(current_sub_state, workers)
                    else:
                        return build_labeling_display(current_labeling_state)

            with Live(LiveSubfeatureDisplay(), console=console, refresh_per_second=10) as live:
                combined_live = live  # Make accessible to on_labeling_progress
                def on_sub_progress(state: SubfeatureProgressState) -> None:
                    nonlocal current_sub_state, in_processing_phase
                    if not in_processing_phase:
                        in_processing_phase = True
                    current_sub_state = state
                    live.refresh()

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
) -> tuple[int, int, int, float, float, float, float, TimingStats | None, list[tuple[str, str]]]:
    """Run unified processing: summarize -> embed -> index.

    Returns:
        Tuple of (processed, skipped, indexed, avg_wall, avg_summ, avg_embed, elapsed, timing_stats, failed_elements).
    """
    if dry_run:
        total = parsing_result.total_elements
        console.print(f"  [dim]Dry run: would process {total} elements[/]")
        return (0, 0, 0, 0.0, 0.0, 0.0, 0.0, None, [])

    from shared.db.elasticsearch import ElasticsearchRepository

    es_repo = ElasticsearchRepository(config)

    proc_config = ProcessingConfig(
        summarize_model=config.llm.summarize_model,
        summarize_model_small=config.llm.summarize_model_small,
        embed_model=config.llm.embed_model,
        api_base=config.llm.url,
        provider=config.llm.provider,
        api_key=config.llm.api_key,
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
        eta = state.timing.eta_seconds(state.completed, state.total, state.num_workers)
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
        worker_table.add_column("Model", style="yellow")
        worker_table.add_column("Element")

        workers_data = state.workers.get_all()
        for wid in range(num_workers):
            if wid in workers_data:
                elem, stage, model = workers_data[wid]
                worker_table.add_row(f"[{wid}]", stage, model, elem)
            else:
                worker_table.add_row(f"[{wid}]", "[dim]idle[/]", "", "")

        # Per-type stats - show progress count and API time per element
        # Each element type has a distinct color for visual differentiation
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
                    # Completed - green for counts
                    type_parts.append(f"[{color}]{t}[/]: [green]{done}/{tot}[/] [dim]({api_time:.1f}s)[/]")
                else:
                    # In progress - yellow for counts
                    type_parts.append(f"[{color}]{t}[/]: [yellow]{done}/{tot}[/] [dim]({api_time:.1f}s)[/]")
        type_line = f"  [dim]Progress:[/] {' [dim]|[/] '.join(type_parts)}" if type_parts else ""

        # Stats line - effective wall time (elapsed/done) = actual throughput with parallelism
        effective_wall = state.timing.elapsed / state.completed if state.completed > 0 else 0.0
        total_api = state.timing.avg_summarize_time + state.timing.avg_embed_time
        stats = f"  [dim]Throughput:[/] [green]{effective_wall:.2f}s[/]/elem [dim]|[/] [dim]API:[/] [green]{total_api:.1f}s[/]/elem [dim]([/][green]{state.timing.avg_summarize_time:.1f}s[/] summ + [green]{state.timing.avg_embed_time:.1f}s[/] embed[dim])[/]"

        parts: list[RenderableType] = [bar_text, worker_table]
        if type_line:
            parts.append(type_line)
        parts.append(stats)

        # Show recent errors if any
        if state.recent_errors:
            error_text = Text()
            error_text.append("  Errors: ", style="red bold")
            for i, (elem_name, error) in enumerate(state.recent_errors):
                if i > 0:
                    error_text.append(" | ", style="dim")
                error_text.append(f"{elem_name}", style="dim")
                error_text.append(": ", style="dim")
                # Truncate long error messages
                short_error = error[:50] + "..." if len(error) > 50 else error
                error_text.append(short_error, style="red")
            parts.append(error_text)

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
        num_workers=workers,
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
            magaldi_config=config if not dry_run else None,
        )

    # Get timing stats from current state
    avg_summ = current_state.timing.avg_summarize_time
    avg_embed = current_state.timing.avg_embed_time
    elapsed = current_state.timing.elapsed
    # Effective wall time = elapsed / processed (shows throughput with parallelism)
    avg_wall = elapsed / result.elements_processed if result.elements_processed > 0 else 0.0

    return (result.elements_processed, result.elements_skipped, result.indexed, avg_wall, avg_summ, avg_embed, elapsed, timing_stats, result.failed_elements)


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
# WEB COMMANDS
# =============================================================================


@main.group()
def web() -> None:
    """Web UI commands."""
    pass


@web.command("serve")
@click.option("--host", "-h", default=None, help="Host to bind to (default: from config)")
@click.option("--port", "-p", default=None, type=int, help="Port to bind to (default: from config)")
@click.option("--reload", "-r", is_flag=True, help="Enable auto-reload for development")
def web_serve(host: str | None, port: int | None, reload: bool) -> None:
    """Start the web server."""
    from magaldi_web.app import run_server

    config = load_config()
    host = host or config.web.host
    port = port or config.web.port

    console.print(f"[bold blue]Starting Magaldi Web UI[/]")
    console.print(f"  URL: http://{host}:{port}")
    console.print(f"  Auto-reload: {'enabled' if reload else 'disabled'}")
    console.print()

    run_server(host=host, port=port, reload=reload)


# =============================================================================
# ENTRY POINT
# =============================================================================


if __name__ == "__main__":
    main()
