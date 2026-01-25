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
                short_id = element_id.split(":")[-3:]  # format: type:name:line
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

        # Phase 6: Glossary Extraction (after features so we can extract from summaries)
        if not skip_ai and not dry_run and processed > 0:
            console.print("\n[bold blue]Phase 6:[/] Glossary Extraction")
            run_glossary_extraction(
                scope=discovery_result.scope,
                repository=discovery_result.repository,
                username=user,
                config=config,
                workers=workers,
            )

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
@click.option("--workers", "-w", default=8, type=int, help="Number of concurrent workers (default: 8)")
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
    from pathlib import Path
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

                # Current cluster line with model
                current_text = Text()
                if state.current_cluster:
                    current_text.append("  labeling  ", style="cyan")
                    if state.model:
                        current_text.append(f"{state.model}  ", style="yellow")
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
            worker_table = Table(show_header=False, box=None, padding=0)
            worker_table.add_column("ID", style="dim", width=4)
            worker_table.add_column("Stage", style="cyan", width=14)
            worker_table.add_column("Model", style="yellow", width=26)
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

            def build_labeling_display(state: SubfeatureLabelingState) -> RenderableType:
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
                worker_table = Table(show_header=False, box=None, padding=0)
                worker_table.add_column("ID", style="dim", width=4)
                worker_table.add_column("Stage", style="cyan", width=14)
                worker_table.add_column("Model", style="yellow", width=26)
                worker_table.add_column("Parent", style="magenta", width=20)
                worker_table.add_column("Subfeature")

                workers_data = state.workers.get_all()
                for wid in range(num_workers):
                    if wid in workers_data:
                        parent_feature, stage, model, subfeature = workers_data[wid]
                        display_parent = parent_feature[:17] + "..." if len(parent_feature) > 20 else parent_feature
                        display_sub = subfeature[:25] + "..." if len(subfeature) > 28 else subfeature
                        worker_table.add_row(f"[{wid}]", stage, model, display_parent, display_sub)
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
                        return build_labeling_display(current_labeling_state)

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


def run_glossary_extraction(
    scope: str,
    repository: str,
    username: str,
    config: MagaldiConfig,
    es_repo: "ElasticsearchRepository | None" = None,
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
        model_name = config.llm.summarize_model

        # Track current phase
        current_phase = {"name": "Extracting terms from features"}

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
            worker_table.add_column("Model", style="yellow", width=26)
            worker_table.add_column("Item")

            workers_data = state.workers.get_all()
            for wid in range(num_workers):
                if wid in workers_data:
                    item_label, model = workers_data[wid]
                    stage = "summarizing" if "summary" in phase.lower() else "extracting"
                    worker_table.add_row(f"[{wid}]", stage, model, item_label)
                else:
                    worker_table.add_row(f"[{wid}]", "[dim]idle[/]", "", "")

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

        # Print header before Live display
        console.print(f"  Processing {total} features with {workers} workers...")
        console.print(f"  Phase 1: Extract terms | Phase 2: Generate summaries")
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
                    live.console.print(f"  [bold green]✓ Phase 1 complete[/]")
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

        # Print summary stats
        avg_api = timing_stats.avg_api_time
        elapsed = timing_stats.elapsed
        console.print()
        console.print(f"  [green]{len(glossary_items)}[/] unique terms extracted in [cyan]{format_duration(elapsed)}[/]")
        console.print(f"  Avg: [green]{avg_api:.1f}s[/]/feature | Model: [yellow]{model_name}[/]")

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
        worker_table = Table(show_header=False, box=None, padding=0)
        worker_table.add_column("ID", style="dim", width=4)
        worker_table.add_column("Stage", style="cyan", width=14)
        worker_table.add_column("Model", style="yellow", width=26)
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

        # Stats line - effective wall time = elapsed / items that actually made API calls
        # Don't include skipped (unchanged) items in throughput calculation
        api_processed = state.completed - state.skipped
        effective_wall = state.timing.elapsed / api_processed if api_processed > 0 else 0.0
        total_api = state.timing.avg_summarize_time + state.timing.avg_embed_time
        # Show separate summary_embed and code_embed times
        summ_emb = state.timing.avg_summary_embed_time
        code_emb = state.timing.avg_code_embed_time
        stats = f"  [dim]Throughput:[/] [green]{effective_wall:.2f}s[/]/item [dim]|[/] [dim]API:[/] [green]{total_api:.1f}s[/]/item [dim]([/][green]{state.timing.avg_summarize_time:.1f}s[/] summ + [cyan]{summ_emb:.1f}s[/] summ_emb + [cyan]{code_emb:.1f}s[/] code_emb[dim])[/]"

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
    # Effective wall time = elapsed / items that made API calls (not counting skipped/unchanged)
    api_processed = result.elements_processed - result.elements_skipped
    avg_wall = elapsed / api_processed if api_processed > 0 else 0.0

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
    if len(manifest.modified_files): parts.append(f"[yellow]{len(manifest.modified_files)} modified[/]")
    if len(manifest.deleted_files): parts.append(f"[red]-{len(manifest.deleted_files)} del[/]")
    if manifest.unchanged_count: parts.append(f"{manifest.unchanged_count} unchanged")
    if manifest.skipped_count: parts.append(f"skipped {manifest.skipped_count}")
    console.print(f"  {' | '.join(parts)}")


def print_parsing_result(result: ParsingResult) -> None:
    """Print parsing results."""
    types = ", ".join(f"{t}: [green]{c}[/]" for t, c in sorted(result.elements_by_type.items()))
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
        parts.append(f"{skipped} unchanged")
    if indexed:
        parts.append(f"{indexed} indexed")
    if skip_ai:
        parts.append("[yellow]AI skipped[/]")
    if elapsed > 0:
        parts.append(f"{format_duration(elapsed)} elapsed")
        # Effective wall time = elapsed / items that made API calls (not counting skipped)
        api_processed = processed - skipped
        if api_processed > 0:
            effective = elapsed / api_processed
            parts.append(f"{effective:.2f}s/item effective")
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
# BENCHMARK-MODELS COMMAND
# =============================================================================


@main.command("benchmark-models")
@click.argument("repo_path", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option("--file", "-f", "file_path", default=None, help="Specific file to benchmark (relative path)")
@click.option("--num-files", "-n", default=5, help="Number of random files to select (default: 5)")
@click.option("--max-per-type", default=10, help="Max elements per type per file (default: 10)")
@click.option("--models", "-m", default=None, help="Comma-separated list of Ollama models (default: from config)")
@click.option("--ollama-url", default=None, help="Ollama API URL (default: from config or http://localhost:11434)")
@click.option("--user", "-u", default="benchmark", help="Username for parsing (default: benchmark)")
def benchmark_models(
    repo_path: str,
    file_path: str | None,
    num_files: int,
    max_per_type: int,
    models: str | None,
    ollama_url: str | None,
    user: str,
) -> None:
    """Benchmark Ollama models on code summarization.

    Parses a repository, selects random files (or a specific one), and runs
    summarization benchmarks with detailed timing and LLM-as-judge evaluation.

    By default, selects 5 random files with max 10 elements per type per file.

    REPO_PATH is the path to the repository to benchmark.
    """
    import random
    from datetime import datetime as dt
    from pathlib import Path

    from shared.ai.ollama_benchmark import BenchmarkClient, BenchmarkResult
    from magaldi_core.change_detection import ChangeManifest, FileInfo

    # Load config for defaults
    config = load_config(skip_validation=True)
    benchmark_config = config.benchmark

    # Parse models list (CLI overrides config)
    if models:
        model_list = [m.strip() for m in models.split(",")]
    else:
        model_list = benchmark_config.models

    # Ollama URL (CLI overrides config, config overrides default)
    if ollama_url is None:
        ollama_url = benchmark_config.ollama_url or config.llm.url

    # LM Studio URL
    lmstudio_url = benchmark_config.lmstudio_url

    def is_lmstudio_model(model_spec: str) -> bool:
        """Check if the model is an LM Studio model."""
        return model_spec.startswith("lmstudio:")

    def get_model_api_config(model_spec: str) -> dict:
        """Get api_base and api_key for a model spec.

        Returns dict with 'api_base' and 'api_key' keys.
        """
        if is_lmstudio_model(model_spec):
            return {"api_base": lmstudio_url, "api_key": "lm-studio"}
        # For Ollama and others, use defaults (handled by client)
        return {}

    def to_litellm_model(model_spec: str) -> str:
        """Convert model specification to LiteLLM format for display/tracking.

        Examples:
            "qwen3:4b" -> "ollama/qwen3:4b"
            "openai:gpt-4" -> "openai/gpt-4"
            "lmstudio:ibm/granite" -> "lmstudio/ibm/granite" (keeps lmstudio prefix)
            "ollama/qwen3:4b" -> "ollama/qwen3:4b" (already in format)
        """
        # Check for explicit prefixes FIRST (before checking for /)
        # This handles cases like "lmstudio:ibm/granite-4-h-tiny"
        for prefix in ["openai:", "anthropic:", "vllm:", "lmstudio:"]:
            if model_spec.startswith(prefix):
                provider = prefix[:-1]
                model_name = model_spec[len(prefix):]
                return f"{provider}/{model_name}"
        # Already in LiteLLM format (provider/model)
        if "/" in model_spec and not model_spec.startswith("hf.co/"):
            return model_spec
        # Default to Ollama
        return f"ollama/{model_spec}"

    def to_litellm_api_model(model_spec: str) -> str:
        """Convert model specification to the actual LiteLLM API format.

        LM Studio uses OpenAI-compatible API, so lmstudio/ becomes openai/.
        """
        litellm_model = to_litellm_model(model_spec)
        if litellm_model.startswith("lmstudio/"):
            return "openai/" + litellm_model[9:]
        return litellm_model

    console.print("[bold blue]Magaldi Model Benchmark[/]")
    console.print(f"  Repository: {repo_path}")
    console.print(f"  Models: {', '.join(model_list)}")
    console.print()

    try:
        # Phase 1: Discovery
        console.print("[bold blue]Phase 1:[/] Discovery")
        discovery_result = run_discovery(repo_path, user)
        print_discovery_result(discovery_result)

        # Create manifest with ALL files (skip change detection)
        console.print("\n[bold blue]Phase 2:[/] Creating file manifest (all files)")
        manifest = _create_full_manifest(discovery_result)
        console.print(f"  {manifest.files_to_parse} files to parse")

        if manifest.files_to_parse == 0:
            console.print("\n[red]No supported files found in repository.[/]")
            sys.exit(1)

        # Phase 3: Parsing
        console.print("\n[bold blue]Phase 3:[/] Parsing")
        parsing_result = run_parsing(manifest)
        print_parsing_result(parsing_result)

        # Select files to benchmark
        console.print("\n[bold blue]Phase 4:[/] File Selection")
        selected_files = _select_benchmark_files(parsing_result, file_path, num_files=num_files, max_per_type=max_per_type)
        if not selected_files:
            console.print("\n[red]No valid files found for benchmarking.[/]")
            sys.exit(1)

        # Combine all elements from selected files
        elements = []
        for sf in selected_files:
            elements.extend(sf["elements"])

        console.print(f"  Selected {len(selected_files)} files:")
        from collections import Counter
        for sf in selected_files:
            file_types = Counter(e.element_type for e in sf["elements"])
            type_str = ", ".join(f"{t}: {c}" for t, c in sorted(file_types.items()))
            console.print(f"    [cyan]{sf['path']}[/] ({len(sf['elements'])} elements: {type_str})")

        # Show total element type breakdown
        total_type_counts = Counter(e.element_type for e in elements)
        total_type_summary = ", ".join(f"{t}: {c}" for t, c in sorted(total_type_counts.items()))
        console.print(f"  Total: {len(elements)} elements ({total_type_summary})")

        # Check backend connections
        console.print("\n[bold blue]Phase 5:[/] Backend Connection")

        # Create unified client (uses LiteLLM)
        client = BenchmarkClient(api_base=ollama_url)
        if not client.check_connection():
            console.print(f"[red]Cannot connect to Ollama at {ollama_url}[/]")
            sys.exit(1)

        available_models = client.list_models()
        console.print(f"  [green]✓[/] Ollama ({ollama_url}) | {len(available_models)} models")

        # Check LM Studio connection if any lmstudio models are configured
        lmstudio_models_configured = any(m.startswith("lmstudio:") for m in model_list)
        available_lmstudio_models: list[str] = []
        if lmstudio_models_configured:
            try:
                import requests
                resp = requests.get(f"{lmstudio_url}/models", timeout=5)
                if resp.status_code == 200:
                    data = resp.json()
                    available_lmstudio_models = [m["id"] for m in data.get("data", [])]
                    console.print(f"  [green]✓[/] LM Studio ({lmstudio_url}) | {len(available_lmstudio_models)} models")
                else:
                    console.print(f"  [yellow]![/] LM Studio ({lmstudio_url}) | Cannot list models (status {resp.status_code})")
            except Exception as e:
                console.print(f"  [yellow]![/] LM Studio ({lmstudio_url}) | Connection failed: {e}")

        # Convert model specs to LiteLLM format and check availability
        # Store mapping from litellm format to original spec for api_config lookup
        missing_models = []
        models_to_test = []
        model_specs: dict[str, str] = {}  # litellm_model -> original_spec
        for model_spec in model_list:
            litellm_model = to_litellm_model(model_spec)

            # For Ollama models, check availability
            if litellm_model.startswith("ollama/"):
                model_name = litellm_model[7:]  # Remove "ollama/" prefix
                if model_name in available_models or f"{model_name}:latest" in available_models:
                    models_to_test.append(litellm_model)
                    model_specs[litellm_model] = model_spec
                else:
                    # Try without tag
                    base = model_name.rsplit(":", 1)[0] if ":" in model_name else model_name
                    if base in available_models or f"{base}:latest" in available_models:
                        models_to_test.append(litellm_model)
                        model_specs[litellm_model] = model_spec
                    else:
                        missing_models.append(model_spec)
            # For LM Studio models, check availability
            elif litellm_model.startswith("lmstudio/"):
                model_name = litellm_model[9:]  # Remove "lmstudio/" prefix
                if available_lmstudio_models:
                    if model_name in available_lmstudio_models:
                        models_to_test.append(litellm_model)
                        model_specs[litellm_model] = model_spec
                    else:
                        missing_models.append(model_spec)
                else:
                    # LM Studio not reachable, skip these models
                    missing_models.append(model_spec)
            else:
                # Other models - trust they're available
                models_to_test.append(litellm_model)
                model_specs[litellm_model] = model_spec

        if missing_models:
            console.print(f"  [yellow]Missing models (skipped):[/]")
            for m in missing_models:
                console.print(f"    [dim]✗ {m}[/]")

        if not models_to_test:
            console.print("\n[red]No models available to test.[/]")
            console.print("[yellow]Pull models with:[/]")
            for m in missing_models:
                console.print(f"  ollama pull {m}")
            sys.exit(1)

        console.print(f"  Testing models: {', '.join(models_to_test)}")

        # Warmup models
        console.print("\n[bold blue]Phase 6:[/] Model Warmup")
        models_failed_warmup = []
        for litellm_model in models_to_test:
            # Get api config for this model (LM Studio needs different api_base)
            original_spec = model_specs.get(litellm_model, litellm_model)
            api_config = get_model_api_config(original_spec)
            # Use API model format (lmstudio/ -> openai/ for LiteLLM)
            api_model = to_litellm_api_model(original_spec)
            with console.status(f"[bold blue]Warming up {litellm_model}...[/]"):
                success, warmup_time, error = client.warmup(
                    api_model,
                    timeout=120,
                    **api_config,
                )
            if success:
                console.print(f"  [green]✓[/] {litellm_model} ({warmup_time:.1f}s)")
            else:
                console.print(f"  [red]✗[/] {litellm_model}: {error}")
                models_failed_warmup.append(litellm_model)

        # Remove failed models
        for m in models_failed_warmup:
            models_to_test.remove(m)

        if not models_to_test:
            console.print("[red]No models available after warmup.[/]")
            sys.exit(1)

        # Run benchmarks with hierarchical context (file → class → method)
        console.print("\n[bold blue]Phase 7:[/] Benchmarking (with hierarchical context)")
        results: dict[str, list[BenchmarkResult]] = {m: [] for m in models_to_test}

        # Sort elements by hierarchy level: file (0) → class (1) → function/method (2) → variable (3)
        level_order = {"file": 0, "class": 1, "function": 2, "method": 2, "variable": 3, "constant": 3}
        sorted_elements = sorted(elements, key=lambda e: (e.relative_path, level_order.get(e.element_type, 99), e.line_start))

        # Build list of (element, index) to track original order for results
        element_indices = {id(elem): i for i, elem in enumerate(elements)}
        prompts = [(elem, None) for elem in elements]  # Placeholder, prompts built per-model

        # Test each model with hierarchical summarization
        for litellm_model in models_to_test:
            original_spec = model_specs.get(litellm_model, litellm_model)
            console.print(f"\n  [cyan]{litellm_model}[/]")

            # Track summaries for hierarchical context (per model)
            # file_summaries[relative_path] = summary
            # class_summaries[(relative_path, class_name)] = summary
            # function_summaries[element_id] = summary
            file_summaries: dict[str, str] = {}
            class_summaries: dict[tuple[str, str], str] = {}
            function_summaries: dict[str, str] = {}  # element_id -> summary

            # Process elements in hierarchical order
            model_results: dict[int, BenchmarkResult] = {}  # index -> result

            for elem in sorted_elements:
                elem_name = f"{elem.element_type}:{elem.name}"
                elem_idx = element_indices[id(elem)]

                # For file elements, load raw_code from disk (not stored in parsed elements)
                if elem.element_type == "file" and not elem.raw_code:
                    file_full_path = Path(repo_path) / elem.relative_path
                    if file_full_path.exists():
                        try:
                            elem.raw_code = file_full_path.read_text(encoding="utf-8")
                        except Exception:
                            elem.raw_code = ""

                # Build parent summaries based on element type
                parent_summaries: dict[str, str] = {}
                if elem.element_type != "file":
                    # Add file summary if available
                    if elem.relative_path in file_summaries:
                        parent_summaries["file"] = file_summaries[elem.relative_path]
                if elem.element_type in ("method", "variable", "constant"):
                    # Add class summary if available (find parent class)
                    if elem.parent_id:
                        # Try to find class summary by parent
                        for (fp, cn), summ in class_summaries.items():
                            if fp == elem.relative_path:
                                parent_summaries["class"] = summ
                                break
                if elem.element_type in ("variable", "constant"):
                    # Add function/method summary if parent is a function/method
                    if elem.parent_id and elem.parent_id in function_summaries:
                        parent_summaries["function"] = function_summaries[elem.parent_id]

                # Build prompt with parent context
                prompt = _build_summarization_prompt(elem, parent_summaries)

                # For thinking models, add directive to skip thinking
                from shared.ai.ollama_benchmark import BenchmarkClient
                model_name = litellm_model.split("/")[-1] if "/" in litellm_model else litellm_model
                if any(model_name.startswith(tm) for tm in BenchmarkClient.THINKING_MODELS):
                    prompt = prompt + "\n\n/no_think"

                with console.status(f"    [{len(model_results)+1}/{len(elements)}] {elem_name}..."):
                    # Get per-model parameters from config (use original spec for pattern matching)
                    model_params = benchmark_config.get_model_params(original_spec)
                    # Use per-model max_tokens if set, otherwise default
                    max_tokens = model_params.max_tokens or benchmark_config.max_tokens
                    # Get api config for this model (LM Studio needs different api_base)
                    api_config = get_model_api_config(original_spec)
                    # Use API model format (lmstudio/ -> openai/ for LiteLLM)
                    api_model = to_litellm_api_model(original_spec)
                    result = client.generate(
                        model=api_model,
                        prompt=prompt,
                        temperature=model_params.temperature,
                        top_p=model_params.top_p,
                        top_k=model_params.top_k,
                        min_p=model_params.min_p,
                        repetition_penalty=model_params.repetition_penalty,
                        presence_penalty=model_params.presence_penalty,
                        max_tokens=max_tokens,
                        timeout=benchmark_config.timeout,
                        **api_config,
                    )
                    model_results[elem_idx] = result

                # Store summary for child elements
                if result.success and result.response.strip():
                    from shared.ai.summarization import clean_summary
                    cleaned = clean_summary(result.response)
                    if elem.element_type == "file":
                        file_summaries[elem.relative_path] = cleaned
                    elif elem.element_type == "class":
                        class_summaries[(elem.relative_path, elem.name)] = cleaned
                    elif elem.element_type in ("function", "method"):
                        function_summaries[elem.element_id] = cleaned

                if result.success:
                    context_str = ""
                    if parent_summaries:
                        context_str = f" [dim](+{'+'.join(parent_summaries.keys())} context)[/]"
                    console.print(
                        f"    [green]✓[/] {elem_name:<40} | "
                        f"[bold]{result.total_time:.2f}s[/] | "
                        f"{result.prompt_chars}→{result.output_chars} chr | "
                        f"{result.prompt_tokens}→{result.output_tokens} tok | "
                        f"{result.tokens_per_second:.0f} t/s{context_str}"
                    )
                else:
                    console.print(f"    [red]✗[/] {elem_name:<40} | {result.error}")

            # Store results in original element order
            results[litellm_model] = [model_results[i] for i in range(len(elements))]

        # Phase 8: LLM Evaluation of summaries (multi-criteria JSON-based)
        console.print("\n[bold blue]Phase 8:[/] LLM Evaluation (multi-criteria)")

        from shared.ai.ollama_benchmark import (
            EVALUATION_CRITERIA,
            CriteriaScores,
            EvaluationResult,
            build_evaluation_prompt,
            parse_evaluation_response,
        )

        # Validate configured eval models, filter to available ones
        # Store as (original_spec, litellm_model) tuples for api_config lookup
        eval_model_pairs: list[tuple[str, str]] = []
        for em in benchmark_config.eval_models:
            litellm_model = to_litellm_model(em)
            # For Ollama models, check availability
            if litellm_model.startswith("ollama/"):
                model_name = litellm_model[7:]  # Remove "ollama/" prefix
                if model_name in available_models or f"{model_name}:latest" in available_models:
                    eval_model_pairs.append((em, litellm_model))
                else:
                    base = model_name.rsplit(":", 1)[0] if ":" in model_name else model_name
                    if base in available_models or f"{base}:latest" in available_models:
                        eval_model_pairs.append((em, litellm_model))
                    else:
                        console.print(f"  [yellow]Eval model {em} not available, skipping[/]")
            else:
                # Non-Ollama models - trust they're available
                eval_model_pairs.append((em, litellm_model))

        if not eval_model_pairs:
            console.print(f"  [yellow]No eval models available, using {models_to_test[-1]}[/]")
            # Use last tested model as fallback
            last_model = model_list[-1]
            eval_model_pairs = [(last_model, to_litellm_model(last_model))]

        eval_models = [litellm for _, litellm in eval_model_pairs]
        eval_model_specs = {litellm: orig for orig, litellm in eval_model_pairs}
        console.print(f"  Using {len(eval_models)} evaluator(s): [cyan]{', '.join(eval_models)}[/]")

        # Show criteria for each element type
        element_types_in_test = set(elem.element_type for elem, _ in prompts)
        for elem_type in sorted(element_types_in_test):
            criteria = EVALUATION_CRITERIA.get(elem_type, {})
            console.print(f"  [dim]{elem_type} criteria: {', '.join(criteria.keys())}[/]")

        # evaluation_results[element_index][eval_model] = EvaluationResult
        evaluation_results: dict[int, dict[str, EvaluationResult]] = {
            i: {} for i in range(len(prompts))
        }
        total_elements = len(prompts)

        for i, (elem, _) in enumerate(prompts):
            elem_name = f"{elem.element_type}:{elem.name}"
            progress = f"[{i+1}/{total_elements}]"

            # Build summaries dict for evaluation
            summaries: dict[str, str] = {}
            for model in models_to_test:
                result = results[model][i]
                if result.success and result.response.strip():
                    from shared.ai.summarization import clean_summary
                    summaries[model] = clean_summary(result.response)
                else:
                    summaries[model] = ""

            # Build evaluation prompt with element-specific criteria
            eval_prompt = build_evaluation_prompt(
                element_type=elem.element_type,
                element_name=elem.name,
                source_code=elem.raw_code or "",
                summaries=summaries,
            )

            # Evaluate with each eval model
            for eval_model in eval_models:
                # Get api config for this eval model (LM Studio needs different api_base)
                eval_original_spec = eval_model_specs.get(eval_model, eval_model)
                eval_api_config = get_model_api_config(eval_original_spec)
                # Use API model format (lmstudio/ -> openai/ for LiteLLM)
                eval_api_model = to_litellm_api_model(eval_original_spec)

                # Retry up to 3 times if parsing fails
                max_retries = 3
                eval_result_obj = EvaluationResult(
                    element_type=elem.element_type,
                    element_name=elem.name,
                )

                for attempt in range(max_retries):
                    with console.status(f"  {progress} [{eval_model}] Evaluating {elem_name}..." + (f" (retry {attempt})" if attempt > 0 else "")):
                        eval_result = client.generate(
                            model=eval_api_model,
                            prompt=eval_prompt,
                            temperature=0.1 + (attempt * 0.1),  # Slightly increase temp on retry
                            max_tokens=1024,  # More tokens for JSON output
                            timeout=benchmark_config.timeout,
                            **eval_api_config,
                        )

                    if eval_result.success:
                        eval_result_obj.raw_response = eval_result.response
                        evaluations, error = parse_evaluation_response(
                            eval_result.response,
                            elem.element_type,
                            models_to_test,
                        )
                        eval_result_obj.evaluations = evaluations
                        eval_result_obj.parse_error = error

                        # Check if all models got evaluated
                        if not error:
                            break  # All ratings present, we're done
                        elif attempt < max_retries - 1:
                            continue  # Retry
                    else:
                        eval_result_obj.parse_error = eval_result.error
                        if attempt < max_retries - 1:
                            continue  # Retry on failure

                evaluation_results[i][eval_model] = eval_result_obj

            # Display scores (show average scaled score for each model)
            rating_parts = []
            for m in models_to_test:
                model_scores = []
                for em in eval_models:
                    eval_res = evaluation_results[i].get(em)
                    if eval_res and m in eval_res.evaluations:
                        model_scores.append(eval_res.evaluations[m].average)
                if model_scores:
                    avg = sum(model_scores) / len(model_scores)
                    rating_parts.append(f"{m}: {avg:.1f}")
                else:
                    rating_parts.append(f"{m}: ?")
            rating_str = " | ".join(rating_parts)

            # Check for errors
            errors = [
                evaluation_results[i].get(em, EvaluationResult("", "")).parse_error
                for em in eval_models
            ]
            has_errors = any(e for e in errors)

            if has_errors:
                console.print(f"  {progress} [yellow]~[/] {elem_name} | {rating_str} [dim](parse issues)[/]")
            else:
                console.print(f"  {progress} [green]✓[/] {elem_name} | {rating_str}")

        # Benchmark summary table (detailed summaries are saved to markdown file only)
        console.print("\n" + "=" * 70)
        console.print("[bold]Benchmark Summary (Multi-Criteria Evaluation)[/]")
        console.print("=" * 70)

        from shared.ai.summarization import clean_summary

        # Build real_success_indices once
        real_success_indices_by_model: dict[str, list[int]] = {}
        real_successes_by_model: dict[str, list] = {}
        for model in models_to_test:
            model_results = results[model]
            real_successes = []
            real_success_indices = []
            for i, r in enumerate(model_results):
                if r.success and r.response.strip():
                    cleaned = clean_summary(r.response)
                    if cleaned.strip():
                        real_successes.append(r)
                        real_success_indices.append(i)
            real_success_indices_by_model[model] = real_success_indices
            real_successes_by_model[model] = real_successes

        # Helper to get scaled score (1-10) from evaluation results
        def get_average(elem_idx: int, model: str, eval_model: str) -> float | None:
            eval_res = evaluation_results[elem_idx].get(eval_model)
            if eval_res and model in eval_res.evaluations:
                return eval_res.evaluations[model].average
            return None

        # Helper to get criteria scores
        def get_criteria_scores(elem_idx: int, model: str, eval_model: str) -> dict[str, int]:
            eval_res = evaluation_results[elem_idx].get(eval_model)
            if eval_res and model in eval_res.evaluations:
                return eval_res.evaluations[model].scores
            return {}

        # Show summary table per evaluator
        for eval_model in eval_models:
            console.print(f"\n[bold]Evaluator: {eval_model}[/]")

            summary_table = Table(show_header=True, header_style="bold cyan")
            summary_table.add_column("Model", style="cyan", no_wrap=True)
            summary_table.add_column("Score", justify="center")
            summary_table.add_column("Success", justify="center")
            summary_table.add_column("Avg Time", justify="right")
            summary_table.add_column("tok/s", justify="right")

            for model in models_to_test:
                real_successes = real_successes_by_model[model]
                real_success_indices = real_success_indices_by_model[model]

                # Calculate average scaled score for real successes
                model_scores = [
                    get_average(i, model, eval_model)
                    for i in real_success_indices
                ]
                valid_scores = [s for s in model_scores if s is not None]
                avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0

                if real_successes:
                    avg_wall = sum(r.total_time for r in real_successes) / len(real_successes)
                    avg_tps = sum(r.tokens_per_second for r in real_successes) / len(real_successes)

                    # Color-code score (1-10 scale)
                    if avg_score >= 8:
                        score_style = "bold green"
                    elif avg_score >= 6:
                        score_style = "yellow"
                    else:
                        score_style = "red"

                    summary_table.add_row(
                        model,
                        f"[{score_style}]{avg_score:.1f}/10[/]" if valid_scores else "-",
                        f"{len(real_successes)}/{len(results[model])}",
                        f"{avg_wall:.2f}s",
                        f"{avg_tps:.0f}",
                    )
                else:
                    summary_table.add_row(
                        model,
                        "-",
                        f"0/{len(results[model])}",
                        "-",
                        "-",
                    )

            console.print(summary_table)

        # Criteria breakdown by element type
        console.print("\n[bold]Criteria Scores by Element Type[/]")

        # Group elements by type
        from collections import defaultdict
        elements_by_type: dict[str, list[int]] = defaultdict(list)
        for i, (elem, _) in enumerate(prompts):
            elements_by_type[elem.element_type].append(i)

        # Show criteria breakdown for each element type
        for elem_type in sorted(elements_by_type.keys()):
            indices = elements_by_type[elem_type]
            criteria = EVALUATION_CRITERIA.get(elem_type, {})

            console.print(f"\n[cyan]{elem_type}[/] ({len(indices)} elements)")

            # Build table with criteria as rows, models as columns
            criteria_table = Table(show_header=True, header_style="bold", expand=False)
            criteria_table.add_column("Criterion", style="dim", no_wrap=True, min_width=20)
            for model in models_to_test:
                criteria_table.add_column(model.replace("/", "\n"), justify="center")

            for criterion in criteria.keys():
                row = [criterion]
                for model in models_to_test:
                    # Average this criterion across all elements of this type and all evaluators
                    criterion_scores = []
                    for i in indices:
                        r = results[model][i]
                        if r.success and r.response.strip():
                            cleaned = clean_summary(r.response)
                            if cleaned.strip():
                                for em in eval_models:
                                    scores = get_criteria_scores(i, model, em)
                                    if criterion in scores:
                                        criterion_scores.append(scores[criterion])
                    if criterion_scores:
                        avg = sum(criterion_scores) / len(criterion_scores)
                        # Color based on 1-10 scale
                        if avg >= 8:
                            row.append(f"[bold green]{avg:.1f}[/]")
                        elif avg >= 6:
                            row.append(f"[yellow]{avg:.1f}[/]")
                        else:
                            row.append(f"[red]{avg:.1f}[/]")
                    else:
                        row.append("-")
                criteria_table.add_row(*row)

            # Add overall row
            overall_row = ["[bold]Overall[/]"]
            for model in models_to_test:
                model_scores = []
                for i in indices:
                    r = results[model][i]
                    if r.success and r.response.strip():
                        cleaned = clean_summary(r.response)
                        if cleaned.strip():
                            for em in eval_models:
                                score = get_average(i, model, em)
                                if score is not None:
                                    model_scores.append(score)
                if model_scores:
                    avg = sum(model_scores) / len(model_scores)
                    if avg >= 8:
                        overall_row.append(f"[bold green]{avg:.1f}/10[/]")
                    elif avg >= 6:
                        overall_row.append(f"[yellow]{avg:.1f}/10[/]")
                    else:
                        overall_row.append(f"[red]{avg:.1f}/10[/]")
                else:
                    overall_row.append("-")
            criteria_table.add_row(*overall_row)

            console.print(criteria_table)

        # Averaged summary across all evaluators
        if len(eval_models) > 1:
            console.print(f"\n[bold]Averaged Across {len(eval_models)} Evaluators[/]")

            avg_summary_table = Table(show_header=True, header_style="bold cyan")
            avg_summary_table.add_column("Model", style="cyan", no_wrap=True)
            avg_summary_table.add_column("Avg Score", justify="center")
            avg_summary_table.add_column("Success", justify="center")
            avg_summary_table.add_column("Avg Time", justify="right")
            avg_summary_table.add_column("tok/s", justify="right")

            for model in models_to_test:
                real_successes = real_successes_by_model[model]
                real_success_indices = real_success_indices_by_model[model]

                # Average across all evaluators
                all_scores = []
                for i in real_success_indices:
                    for em in eval_models:
                        score = get_average(i, model, em)
                        if score is not None:
                            all_scores.append(score)
                avg_score = sum(all_scores) / len(all_scores) if all_scores else 0

                if real_successes:
                    avg_wall = sum(r.total_time for r in real_successes) / len(real_successes)
                    avg_tps = sum(r.tokens_per_second for r in real_successes) / len(real_successes)

                    if avg_score >= 8:
                        score_style = "bold green"
                    elif avg_score >= 6:
                        score_style = "yellow"
                    else:
                        score_style = "red"

                    avg_summary_table.add_row(
                        model,
                        f"[{score_style}]{avg_score:.1f}/10[/]" if all_scores else "-",
                        f"{len(real_successes)}/{len(results[model])}",
                        f"{avg_wall:.2f}s",
                        f"{avg_tps:.0f}",
                    )
                else:
                    avg_summary_table.add_row(model, "-", f"0/{len(results[model])}", "-", "-")

            console.print(avg_summary_table)

        # Scores by Character Length
        console.print("\n[bold]Scores by Input Character Length[/]")

        # Define character length ranges
        char_ranges = [
            (0, 500, "0-500"),
            (500, 1000, "500-1K"),
            (1000, 2000, "1K-2K"),
            (2000, 4000, "2K-4K"),
            (4000, 8000, "4K-8K"),
            (8000, float("inf"), "8K+"),
        ]

        def get_char_range_label(chars: int) -> str:
            for min_c, max_c, label in char_ranges:
                if min_c <= chars < max_c:
                    return label
            return "8K+"

        # Group elements by character length range
        elements_by_char_range: dict[str, list[int]] = defaultdict(list)
        for i, (elem, _) in enumerate(prompts):
            # Get prompt char count from any successful result
            prompt_chars = 0
            for model in models_to_test:
                if results[model][i].success and results[model][i].prompt_chars > 0:
                    prompt_chars = results[model][i].prompt_chars
                    break
            if prompt_chars == 0:
                # Fallback: estimate from raw_code length
                prompt_chars = len(elem.raw_code or "") + 200  # ~200 chars for prompt template
            range_label = get_char_range_label(prompt_chars)
            elements_by_char_range[range_label].append(i)

        # Show averaged char range table
        console.print(f"\n[dim]Scores by Character Length (averaged across evaluators)[/]")
        char_table = Table(show_header=True, header_style="bold cyan", expand=False)
        char_table.add_column("Char Range", style="cyan", no_wrap=True, min_width=10)
        char_table.add_column("Count", justify="right", no_wrap=True, min_width=5)
        for model in models_to_test:
            char_table.add_column(model.replace("/", "\n"), justify="center")

        # Sort by range order
        range_order = {label: i for i, (_, _, label) in enumerate(char_ranges)}
        for range_label in sorted(elements_by_char_range.keys(), key=lambda x: range_order.get(x, 99)):
            indices = elements_by_char_range[range_label]
            row = [range_label, str(len(indices))]
            for model in models_to_test:
                range_scores = []
                for i in indices:
                    r = results[model][i]
                    if r.success and r.response.strip():
                        cleaned = clean_summary(r.response)
                        if cleaned.strip():
                            for em in eval_models:
                                score = get_average(i, model, em)
                                if score is not None:
                                    range_scores.append(score)
                if range_scores:
                    avg = sum(range_scores) / len(range_scores)
                    if avg >= 8:
                        row.append(f"[bold green]{avg:.1f}[/]")
                    elif avg >= 6:
                        row.append(f"[yellow]{avg:.1f}[/]")
                    else:
                        row.append(f"[red]{avg:.1f}[/]")
                else:
                    row.append("-")
            char_table.add_row(*row)

        console.print(char_table)

        # Average scores by element type AND character range (single table)
        console.print("\n[bold]Average Scores by Element Type & Character Range[/]")

        # Build: element_type -> char_range -> model -> list of scores
        type_range_scores: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )

        for i, (elem, _) in enumerate(prompts):
            prompt_chars = 0
            for model in models_to_test:
                if results[model][i].success and results[model][i].prompt_chars > 0:
                    prompt_chars = results[model][i].prompt_chars
                    break
            if prompt_chars == 0:
                prompt_chars = len(elem.raw_code or "") + 200

            range_label = get_char_range_label(prompt_chars)
            elem_type = elem.element_type

            for model in models_to_test:
                r = results[model][i]
                if r.success and r.response.strip():
                    cleaned = clean_summary(r.response)
                    if cleaned.strip():
                        for em in eval_models:
                            score = get_average(i, model, em)
                            if score is not None:
                                type_range_scores[elem_type][range_label][model].append(score)

        # Single combined table (element type first, then char range)
        combined_table = Table(show_header=True, header_style="bold cyan", show_lines=False, expand=False)
        combined_table.add_column("Element Type", style="cyan", no_wrap=True, min_width=12)
        combined_table.add_column("Char Range", style="dim", no_wrap=True, min_width=10)
        combined_table.add_column("Count", justify="right", no_wrap=True, min_width=5)
        for model in models_to_test:
            combined_table.add_column(model.replace("/", "\n"), justify="center")

        for elem_type in sorted(type_range_scores.keys()):
            range_data = type_range_scores[elem_type]
            for range_label in sorted(range_data.keys(), key=lambda x: range_order.get(x, 99)):
                model_scores_data = range_data[range_label]
                count = max(len(s) for s in model_scores_data.values()) if model_scores_data else 0
                count = count // len(eval_models) if eval_models else count
                row = [elem_type, range_label, str(count)]
                for model in models_to_test:
                    if model in model_scores_data and model_scores_data[model]:
                        avg = sum(model_scores_data[model]) / len(model_scores_data[model])
                        if avg >= 8:
                            row.append(f"[bold green]{avg:.1f}[/]")
                        elif avg >= 6:
                            row.append(f"[yellow]{avg:.1f}[/]")
                        else:
                            row.append(f"[red]{avg:.1f}[/]")
                    else:
                        row.append("-")
                combined_table.add_row(*row)

        console.print(combined_table)

        # =================================================================
        # FULL METRICS TABLE with hierarchical totals
        # =================================================================
        console.print("\n[bold]Full Metrics Breakdown[/]")

        # Build: element_type -> char_range -> criterion -> model -> list of scores
        full_metrics: dict[str, dict[str, dict[str, dict[str, list[int]]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        )
        # Also track counts: element_type -> char_range -> count
        element_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

        for i, (elem, _) in enumerate(prompts):
            # Get prompt chars
            prompt_chars = 0
            for model in models_to_test:
                if results[model][i].success and results[model][i].prompt_chars > 0:
                    prompt_chars = results[model][i].prompt_chars
                    break
            if prompt_chars == 0:
                prompt_chars = len(elem.raw_code or "") + 200

            range_label = get_char_range_label(prompt_chars)
            elem_type = elem.element_type
            criteria = EVALUATION_CRITERIA.get(elem_type, {})

            # Check if any model has a valid response for this element
            has_valid_response = False
            for model in models_to_test:
                r = results[model][i]
                if r.success and r.response.strip():
                    cleaned = clean_summary(r.response)
                    if cleaned.strip():
                        has_valid_response = True
                        break

            if has_valid_response:
                element_counts[elem_type][range_label] += 1

            # Collect per-criterion scores
            for model in models_to_test:
                r = results[model][i]
                if r.success and r.response.strip():
                    cleaned = clean_summary(r.response)
                    if cleaned.strip():
                        for em in eval_models:
                            scores = get_criteria_scores(i, model, em)
                            for criterion in criteria.keys():
                                if criterion in scores:
                                    full_metrics[elem_type][range_label][criterion][model].append(scores[criterion])

        # Create full metrics table
        metrics_table = Table(show_header=True, header_style="bold magenta", show_lines=True, expand=False)
        metrics_table.add_column("Element Type", style="cyan", no_wrap=True, min_width=12)
        metrics_table.add_column("Char Range", style="dim", no_wrap=True, min_width=10)
        metrics_table.add_column("Criterion", style="dim", no_wrap=True, min_width=18)
        metrics_table.add_column("Count", justify="right", no_wrap=True, min_width=5)
        for model in models_to_test:
            metrics_table.add_column(model.replace("/", "\n"), justify="center")

        def format_score(avg: float) -> str:
            if avg >= 8:
                return f"[bold green]{avg:.1f}[/]"
            elif avg >= 6:
                return f"[yellow]{avg:.1f}[/]"
            else:
                return f"[red]{avg:.1f}[/]"

        def calc_avg(scores: list) -> float | None:
            return sum(scores) / len(scores) if scores else None

        grand_totals: dict[str, list[float]] = defaultdict(list)

        for elem_type in sorted(full_metrics.keys()):
            type_totals: dict[str, list[float]] = defaultdict(list)
            range_data = full_metrics[elem_type]

            for range_label in sorted(range_data.keys(), key=lambda x: range_order.get(x, 99)):
                criterion_data = range_data[range_label]
                range_totals: dict[str, list[float]] = defaultdict(list)
                count = element_counts[elem_type][range_label]
                criteria = EVALUATION_CRITERIA.get(elem_type, {})

                # Add row for each criterion
                for criterion in criteria.keys():
                    if criterion in criterion_data:
                        row = [elem_type, range_label, criterion, str(count)]
                        for model in models_to_test:
                            if model in criterion_data[criterion] and criterion_data[criterion][model]:
                                scores = criterion_data[criterion][model]
                                avg = calc_avg(scores)
                                if avg is not None:
                                    row.append(format_score(avg))
                                    range_totals[model].append(avg)
                                    type_totals[model].append(avg)
                                    grand_totals[model].append(avg)
                                else:
                                    row.append("-")
                            else:
                                row.append("-")
                        metrics_table.add_row(*row)

                # Subtotal for element_type + char_range
                subtotal_row = ["", "", "[bold]Subtotal[/]", f"[bold]{count}[/]"]
                for model in models_to_test:
                    if range_totals[model]:
                        avg = calc_avg(range_totals[model])
                        subtotal_row.append(f"[bold]{format_score(avg)}[/]" if avg else "-")
                    else:
                        subtotal_row.append("-")
                metrics_table.add_row(*subtotal_row)

            # Total for element_type (across all char ranges)
            type_count = sum(element_counts[elem_type].values())
            type_total_row = [f"[bold cyan]{elem_type}[/]", "[bold]TOTAL[/]", "", f"[bold]{type_count}[/]"]
            for model in models_to_test:
                if type_totals[model]:
                    avg = calc_avg(type_totals[model])
                    type_total_row.append(f"[bold]{format_score(avg)}[/]" if avg else "-")
                else:
                    type_total_row.append("-")
            metrics_table.add_row(*type_total_row)

        # Grand total
        total_count = sum(sum(rc.values()) for rc in element_counts.values())
        grand_total_row = ["[bold magenta]GRAND TOTAL[/]", "", "", f"[bold]{total_count}[/]"]
        for model in models_to_test:
            if grand_totals[model]:
                avg = calc_avg(grand_totals[model])
                grand_total_row.append(f"[bold]{format_score(avg)}[/]" if avg else "-")
            else:
                grand_total_row.append("-")
        metrics_table.add_row(*grand_total_row)

        console.print(metrics_table)

        # Token stats
        console.print("\n[bold]Token Statistics[/]")
        for model in models_to_test:
            successful = [r for r in results[model] if r.success]
            if successful:
                total_prompt = sum(r.prompt_tokens for r in successful)
                total_output = sum(r.output_tokens for r in successful)
                console.print(f"  {model}: {total_prompt:,} prompt tokens, {total_output:,} output tokens")

        # Save results to markdown
        markdown_path = _save_benchmark_markdown(
            repo_path=repo_path,
            models_tested=models_to_test,
            eval_models=eval_models,
            elements=elements,
            results=results,
            evaluation_results=evaluation_results,
        )
        console.print(f"\n[bold green]Results saved to:[/] {markdown_path}")

        # Close client
        client.close()
        console.print("\n[green]Benchmark complete.[/]")

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted[/]")
        sys.exit(130)
    except Exception as e:
        console.print(f"\n[red]Error:[/] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def _create_full_manifest(discovery_result: DiscoveryResult) -> ChangeManifest:
    """Create a ChangeManifest with all discovered files as 'new'.

    This skips change detection and treats all files as new.
    """
    import hashlib
    import os
    from datetime import datetime as dt
    from pathlib import Path

    from magaldi_core.change_detection import ChangeManifest, FileInfo
    from magaldi_core.discovery import SUPPORTED_EXTENSIONS, _is_excluded_dir, _is_excluded_file

    new_files = []
    repo_path = discovery_result.repo_path

    for root, dirs, files in os.walk(repo_path):
        # Filter directories
        dirs[:] = [
            d for d in dirs
            if not _is_excluded_dir(
                d,
                discovery_result.exclude_directories + ["node_modules", ".git", "__pycache__", ".venv", "venv"]
            )
        ]

        for file_name in files:
            abs_path = Path(root) / file_name
            rel_path = str(abs_path.relative_to(repo_path))

            # Check extension
            ext = abs_path.suffix.lower()
            if ext not in SUPPORTED_EXTENSIONS:
                continue

            # Check exclusions
            if _is_excluded_file(file_name, discovery_result.exclude_files):
                continue

            # Compute hash
            try:
                with open(abs_path, "rb") as f:
                    file_hash = hashlib.sha256(f.read()).hexdigest()
            except Exception:
                continue

            new_files.append(FileInfo(
                relative_path=rel_path,
                absolute_path=abs_path,
                language=SUPPORTED_EXTENSIONS[ext],
                hash=file_hash,
            ))

    return ChangeManifest(
        scope=discovery_result.scope,
        repository=discovery_result.repository,
        username=discovery_result.username,
        timestamp=dt.now(),
        total_files_scanned=len(new_files),
        new_files=new_files,
        modified_files=[],
        deleted_files=[],
        unchanged_count=0,
        skipped_count=0,
    )


def _select_benchmark_files(
    parsing_result: ParsingResult,
    forced_path: str | None,
    num_files: int = 5,
    max_per_type: int = 10,
) -> list[dict] | None:
    """Select files for benchmarking.

    Args:
        parsing_result: Result from parsing phase.
        forced_path: Specific file path to use, or None for random selection.
        num_files: Number of files to select (default 5).
        max_per_type: Maximum elements per type per file (default 10).

    Returns:
        List of dicts with 'path' and 'elements' keys, or None if no valid files found.
    """
    import random
    from collections import defaultdict

    def cap_elements_by_type(elements: list, max_per_type: int) -> list:
        """Cap elements to max_per_type per element_type per file."""
        by_type: dict[str, list] = defaultdict(list)
        for elem in elements:
            by_type[elem.element_type].append(elem)

        result = []
        for elem_type, type_elements in by_type.items():
            if len(type_elements) > max_per_type:
                # Randomly sample to get variety
                result.extend(random.sample(type_elements, max_per_type))
            else:
                result.extend(type_elements)

        # Sort by line number to maintain order
        result.sort(key=lambda e: e.line_start)
        return result

    parsed_files = parsing_result.parsed_files

    if forced_path:
        # Find the specific file (single file mode)
        for pf in parsed_files:
            path = pf.file_info.relative_path
            if path == forced_path or path.endswith(forced_path):
                elements = cap_elements_by_type(list(pf.elements), max_per_type)
                return [{
                    "path": path,
                    "elements": elements,
                }]
        return None

    # Random selection: prefer files with 3+ elements
    candidates = [
        pf for pf in parsed_files
        if len(pf.elements) >= 3
    ]

    if not candidates:
        # Fall back to any file with elements
        candidates = [pf for pf in parsed_files if len(pf.elements) > 0]

    if not candidates:
        return None

    # Select up to num_files randomly
    selected_files = random.sample(candidates, min(num_files, len(candidates)))

    result = []
    for pf in selected_files:
        elements = cap_elements_by_type(list(pf.elements), max_per_type)
        result.append({
            "path": pf.file_info.relative_path,
            "elements": elements,
        })

    return result


def _save_benchmark_markdown(
    repo_path: str,
    models_tested: list[str],
    eval_models: list[str],
    elements: list,
    results: dict,
    evaluation_results: dict,
    output_dir: str = "plans/benchmarks/data",
) -> str:
    """Save benchmark results to markdown file.

    Args:
        repo_path: Repository path.
        models_tested: List of models tested.
        eval_models: List of models used for evaluation (LLM-as-judge).
        elements: List of CodeElements.
        results: Dict mapping model -> list of BenchmarkResult.
        evaluation_results: Dict mapping element_index -> {eval_model -> EvaluationResult}.
        output_dir: Directory to save markdown file.

    Returns:
        Path to saved markdown file.
    """
    from collections import defaultdict
    from datetime import datetime as dt
    from pathlib import Path

    from shared.ai.summarization import clean_summary
    from shared.ai.ollama_benchmark import EVALUATION_CRITERIA

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Generate filename with timestamp
    timestamp = dt.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{timestamp}_benchmark.md"
    filepath = Path(output_dir) / filename

    lines = []

    # Helper to get scaled score
    def get_average(elem_idx: int, model: str, eval_model: str) -> float | None:
        eval_res = evaluation_results[elem_idx].get(eval_model)
        if eval_res and model in eval_res.evaluations:
            return eval_res.evaluations[model].average
        return None

    # Helper to get criteria scores
    def get_criteria_scores(elem_idx: int, model: str, eval_model: str) -> dict[str, int]:
        eval_res = evaluation_results[elem_idx].get(eval_model)
        if eval_res and model in eval_res.evaluations:
            return eval_res.evaluations[model].scores
        return {}

    # Helper to get notes
    def get_notes(elem_idx: int, model: str, eval_model: str) -> str:
        eval_res = evaluation_results[elem_idx].get(eval_model)
        if eval_res and model in eval_res.evaluations:
            return eval_res.evaluations[model].notes
        return ""

    # Header
    lines.append(f"# Benchmark Results - {dt.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")
    lines.append(f"**Repository:** `{repo_path}`")
    lines.append(f"**Evaluation Models:** `{', '.join(eval_models)}`")
    lines.append(f"**Models Tested:** {len(models_tested)}")
    lines.append(f"**Elements Evaluated:** {len(elements)}")
    lines.append(f"**Evaluation:** Multi-criteria JSON-based (element-type-specific)")
    lines.append("")

    # Pre-compute real successes for each model
    real_success_indices_by_model: dict[str, list[int]] = {}
    real_successes_by_model: dict[str, list] = {}
    for model in models_tested:
        model_results = results[model]
        real_successes = []
        real_success_indices = []
        for i, r in enumerate(model_results):
            if r.success and r.response.strip():
                cleaned = clean_summary(r.response)
                if cleaned.strip():
                    real_successes.append(r)
                    real_success_indices.append(i)
        real_success_indices_by_model[model] = real_success_indices
        real_successes_by_model[model] = real_successes

    # Summary table per evaluator
    for eval_model in eval_models:
        lines.append(f"## Summary (Evaluator: {eval_model})")
        lines.append("")
        lines.append("| Model | Avg Score | Success | Avg Time | tok/s |")
        lines.append("|-------|-----------|---------|----------|-------|")

        for model in models_tested:
            real_successes = real_successes_by_model[model]
            real_success_indices = real_success_indices_by_model[model]

            # Calculate average scaled score for real successes
            model_scores = [
                get_average(i, model, eval_model)
                for i in real_success_indices
            ]
            valid_scores = [s for s in model_scores if s is not None]
            avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0

            if real_successes:
                avg_wall = sum(r.total_time for r in real_successes) / len(real_successes)
                avg_tps = sum(r.tokens_per_second for r in real_successes) / len(real_successes)
                lines.append(
                    f"| {model} | {avg_score:.1f}/10 | {len(real_successes)}/{len(results[model])} | {avg_wall:.2f}s | {avg_tps:.0f} |"
                )
            else:
                lines.append(f"| {model} | - | 0/{len(results[model])} | - | - |")

        lines.append("")

    # Criteria breakdown by element type
    elements_by_type: dict[str, list[int]] = defaultdict(list)
    for i, elem in enumerate(elements):
        elements_by_type[elem.element_type].append(i)

    lines.append("## Criteria Scores by Element Type")
    lines.append("")

    for elem_type in sorted(elements_by_type.keys()):
        indices = elements_by_type[elem_type]
        criteria = EVALUATION_CRITERIA.get(elem_type, {})

        lines.append(f"### {elem_type} ({len(indices)} elements)")
        lines.append("")

        # Header
        header = "| Criterion |"
        separator = "|-----------|"
        for model in models_tested:
            header += f" {model} |"
            separator += "-" * (len(model) + 2) + "|"
        lines.append(header)
        lines.append(separator)

        for criterion in criteria.keys():
            row = f"| {criterion} |"
            for model in models_tested:
                criterion_scores = []
                for i in indices:
                    r = results[model][i]
                    if r.success and r.response.strip():
                        cleaned = clean_summary(r.response)
                        if cleaned.strip():
                            for em in eval_models:
                                scores = get_criteria_scores(i, model, em)
                                if criterion in scores:
                                    criterion_scores.append(scores[criterion])
                if criterion_scores:
                    avg = sum(criterion_scores) / len(criterion_scores)
                    row += f" {avg:.1f} |"
                else:
                    row += " - |"
            lines.append(row)

        # Overall row
        row = "| **Overall** |"
        for model in models_tested:
            model_scores = []
            for i in indices:
                r = results[model][i]
                if r.success and r.response.strip():
                    cleaned = clean_summary(r.response)
                    if cleaned.strip():
                        for em in eval_models:
                            score = get_average(i, model, em)
                            if score is not None:
                                model_scores.append(score)
            if model_scores:
                avg = sum(model_scores) / len(model_scores)
                row += f" **{avg:.1f}/10** |"
            else:
                row += " - |"
        lines.append(row)

        lines.append("")

    # Averaged summary across all evaluators (if multiple)
    if len(eval_models) > 1:
        lines.append("## Summary (Averaged Across Evaluators)")
        lines.append("")
        lines.append("| Model | Avg Score | Success | Avg Time | tok/s |")
        lines.append("|-------|-----------|---------|----------|-------|")

        for model in models_tested:
            real_successes = real_successes_by_model[model]
            real_success_indices = real_success_indices_by_model[model]

            all_scores = []
            for i in real_success_indices:
                for em in eval_models:
                    score = get_average(i, model, em)
                    if score is not None:
                        all_scores.append(score)
            avg_score = sum(all_scores) / len(all_scores) if all_scores else 0

            if real_successes:
                avg_wall = sum(r.total_time for r in real_successes) / len(real_successes)
                avg_tps = sum(r.tokens_per_second for r in real_successes) / len(real_successes)
                lines.append(
                    f"| {model} | {avg_score:.1f}/10 | {len(real_successes)}/{len(results[model])} | {avg_wall:.2f}s | {avg_tps:.0f} |"
                )
            else:
                lines.append(f"| {model} | - | 0/{len(results[model])} | - | - |")

        lines.append("")

    # Detailed results per element
    lines.append("## Detailed Results")
    lines.append("")

    for i, elem in enumerate(elements):
        elem_name = f"{elem.element_type}:{elem.name}"
        lines.append(f"### {elem_name}")
        lines.append("")
        lines.append(f"**File:** `{elem.relative_path}`")
        lines.append(f"**Lines:** {elem.line_start}-{elem.line_end}")
        lines.append("")

        # Source code snippet
        if elem.raw_code:
            lines.append("<details>")
            lines.append("<summary>Source Code</summary>")
            lines.append("")
            lines.append("```python")
            code_lines = elem.raw_code.strip().split('\n')[:30]
            for line in code_lines:
                lines.append(line)
            if len(elem.raw_code.strip().split('\n')) > 30:
                lines.append(f"# ... ({len(elem.raw_code.strip().split(chr(10)))} lines total)")
            lines.append("```")
            lines.append("</details>")
            lines.append("")

        # Summaries by model with criteria scores
        lines.append("| Model | Score | Summary |")
        lines.append("|-------|-------|---------|")

        for model in models_tested:
            model_result = results[model][i]

            summary = ""
            generation_ok = False
            if model_result.success and model_result.response.strip():
                summary = clean_summary(model_result.response)
                generation_ok = bool(summary.strip())

            if generation_ok:
                # Get averaged score across evaluators
                scores = [get_average(i, model, em) for em in eval_models]
                valid_scores = [s for s in scores if s is not None]
                avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0

                # Escape pipes and truncate for table
                summary_escaped = summary.replace("|", "\\|").replace("\n", " ")
                if len(summary_escaped) > 200:
                    summary_escaped = summary_escaped[:200] + "..."
                lines.append(f"| {model} | {avg_score:.1f}/10 | {summary_escaped} |")
            else:
                error_msg = model_result.error if model_result.error else "empty after cleaning"
                lines.append(f"| {model} | N/A | *(failed: {error_msg[:30]})* |")

        lines.append("")

        # Full summaries with criteria breakdown (expandable)
        lines.append("<details>")
        lines.append("<summary>Full Summaries & Criteria</summary>")
        lines.append("")

        for model in models_tested:
            model_result = results[model][i]

            summary = ""
            generation_ok = False
            if model_result.success and model_result.response.strip():
                summary = clean_summary(model_result.response)
                generation_ok = bool(summary.strip())

            if generation_ok:
                # Get scores and notes from evaluators
                score_parts = []
                for em in eval_models:
                    score = get_average(i, model, em)
                    notes = get_notes(i, model, em)
                    em_short = em.split('/')[0] if '/' in em else em
                    if score is not None:
                        if notes:
                            score_parts.append(f"{em_short}: {score:.1f}/10 - {notes}")
                        else:
                            score_parts.append(f"{em_short}: {score:.1f}/10")
                    else:
                        score_parts.append(f"{em_short}: ?")

                lines.append(f"**{model}**")
                lines.append("")
                lines.append(f"Scores: {' | '.join(score_parts)}")
                lines.append("")

                # Show criteria breakdown
                criteria = EVALUATION_CRITERIA.get(elem.element_type, {})
                if criteria:
                    lines.append("Criteria:")
                    for em in eval_models:
                        scores = get_criteria_scores(i, model, em)
                        if scores:
                            em_short = em.split('/')[0] if '/' in em else em
                            criteria_str = ", ".join(f"{k}={v}" for k, v in scores.items())
                            lines.append(f"- {em_short}: {criteria_str}")
                    lines.append("")

                lines.append(f"> {summary}")
            else:
                error_msg = model_result.error if model_result.error else "empty after cleaning"
                lines.append(f"**{model}** (N/A - failed: {error_msg[:40]})")
                lines.append("")
                lines.append("> *(no valid output)*")
            lines.append("")

        lines.append("</details>")
        lines.append("")
        lines.append("---")
        lines.append("")

    # Write file
    with open(filepath, "w") as f:
        f.write("\n".join(lines))

    return str(filepath)


def _build_summarization_prompt(
    element: "CodeElement",
    parent_summaries: dict[str, str] | None = None,
) -> str:
    """Build a summarization prompt for an element using the same prompts as parse CLI.

    Args:
        element: CodeElement from parsing.
        parent_summaries: Dict with 'file' and/or 'class' summaries for context.

    Returns:
        Prompt string for summarization.
    """
    from shared.ai.summarization import build_prompt

    if parent_summaries is None:
        parent_summaries = {}

    return build_prompt(element, parent_summaries, max_code_tokens=4000)


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
