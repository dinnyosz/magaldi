"""Magaldi CLI - Code discovery engine for AI agents and developers.

Commands:
    magaldi parse /path/to/repo --user <username>
"""

from __future__ import annotations

import sys

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from rich.table import Table

from magaldi.config import MagaldiConfig, load_config
from magaldi.parser.change_detection import (
    ChangeManifest,
    InMemoryFileStateRepository,
    detect_changes,
)
from magaldi.parser.code_parser import ParsingResult, parse_files
from magaldi.parser.discovery import DiscoveryError, DiscoveryResult, discover
from magaldi.embedding.embedding import (
    EmbeddingConfig,
    InMemoryEmbeddingJobRepository,
    InMemoryEmbeddingStore,
    OllamaEmbedClient,
    process_embedding_job,
)
from magaldi.storage.storage import (
    InMemoryDatabaseRepository,
    InMemorySearchRepository,
    StorageResult,
    store_storage_result,
)
from magaldi.summarization.summarization import (
    InMemoryJobRepository,
    InMemorySummaryStore,
    OllamaClient,
    SummarizationConfig,
    process_summarization_job,
)

console = Console()


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
@click.option("--skip-ai", is_flag=True, help="Skip summarization and embedding (phases 5-6)")
@click.option("--dry-run", is_flag=True, help="Use in-memory storage (no database required)")
@click.option("--ollama-url", default=None, help="Ollama API URL (default: from config)")
def parse(
    repo_path: str, user: str, skip_ai: bool, dry_run: bool, ollama_url: str | None
) -> None:
    """Parse a repository and index its code elements.

    REPO_PATH is the path to the repository to parse.
    """
    # Load configuration (skip validation in dry-run mode)
    config = load_config(skip_validation=dry_run)
    if ollama_url:
        config.ollama.url = ollama_url

    if dry_run:
        console.print("[yellow]Dry run mode:[/] Using in-memory storage\n")

    try:
        # Phase 1: Discovery
        console.print("[bold blue]Phase 1:[/] Discovery")
        discovery_result = run_discovery(repo_path, user)
        print_discovery_result(discovery_result)

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

        # Phase 4: Storage
        console.print("\n[bold blue]Phase 4:[/] Storage")
        storage_result = run_storage(parsing_result, manifest, config, dry_run)
        print_storage_result(storage_result)

        if skip_ai:
            console.print("\n[yellow]Skipping AI processing (--skip-ai flag)[/]")
            print_summary(discovery_result, manifest, storage_result, None, None)
            return

        # Phase 5: Summarization
        console.print("\n[bold blue]Phase 5:[/] Summarization")
        summarization_count = run_summarization(parsing_result, config, dry_run)

        # Phase 6: Embedding
        console.print("\n[bold blue]Phase 6:[/] Embedding")
        embedding_count = run_embedding(parsing_result, config, dry_run)

        print_summary(
            discovery_result, manifest, storage_result, summarization_count, embedding_count
        )

    except DiscoveryError as e:
        console.print(f"\n[red]Discovery error:[/] {e}")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[red]Error:[/] {e}")
        if "--dry-run" not in sys.argv:
            console.print("[dim]Hint: Use --dry-run to test without database[/]")
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
        from magaldi.db.elasticsearch import ElasticsearchFileStateRepository
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
        transient=True,
    ) as progress:
        task = progress.add_task("parsing", total=total_files)

        def on_progress(completed: int, total: int) -> None:
            progress.update(task, completed=completed, total=total)

        return parse_files(manifest, on_progress)


def run_storage(
    parsing_result: ParsingResult,
    manifest: ChangeManifest,
    config: MagaldiConfig,
    dry_run: bool,
) -> StorageResult:
    """Run Phase 4: Storage."""
    if dry_run:
        db_repo = InMemoryDatabaseRepository()
        search_repo = InMemorySearchRepository()
    else:
        from magaldi.db.elasticsearch import ElasticsearchRepository
        db_repo = InMemoryDatabaseRepository()  # Not used for persistence
        search_repo = ElasticsearchRepository(config)

    total_elements = parsing_result.total_elements

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]Indexing[/]"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("({task.completed}/{task.total})"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("indexing", total=total_elements)

        def on_progress(completed: int, total: int) -> None:
            progress.update(task, completed=completed, total=total)

        return store_storage_result(parsing_result, manifest, db_repo, search_repo, on_progress)


def run_summarization(
    parsing_result: ParsingResult,
    config: MagaldiConfig,
    dry_run: bool,
) -> int:
    """Run Phase 5: Summarization."""
    sum_config = SummarizationConfig(
        ollama_url=config.ollama.url,
        model=config.ollama.summarize_model,
    )

    # Check Ollama availability
    ollama = OllamaClient(sum_config.ollama_url, sum_config.model)
    if not ollama.verify_model():
        console.print(f"[yellow]Warning:[/] Model {sum_config.model} not available. Skipping summarization.")
        return 0

    # Create repositories
    if dry_run:
        job_repo = InMemoryJobRepository()
        summary_store = InMemorySummaryStore()
    else:
        from magaldi.db.redis import RedisSummarizationJobRepository, RedisSummaryStore
        from magaldi.db.elasticsearch import ElasticsearchRepository
        job_repo = RedisSummarizationJobRepository(config)
        summary_store = RedisSummaryStore(config)
        es_repo = ElasticsearchRepository(config)

    # Populate stores from parsing result
    all_elements = []
    for pf in parsing_result.parsed_files:
        for elem in pf.elements:
            all_elements.append(elem)
            summary_store.store_element(elem)
            job_repo.add_job(
                element_id=elem.element_id,
                level=elem.level,
                parent_id=elem.parent_id,
                dependencies_met=(elem.level == 0),
            )

    # Process jobs level by level
    completed = 0
    total = len(all_elements)

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]Summarizing[/]"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("({task.completed}/{task.total})"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("summarizing", total=total)

        while True:
            jobs = job_repo.claim_pending_jobs("cli-worker", batch_size=10)
            if not jobs:
                break

            for job in jobs:
                element_id = job["element_id"]
                success = process_summarization_job(
                    element_id, job_repo, summary_store, ollama, sum_config
                )
                if success:
                    completed += 1
                    if not dry_run:
                        summary = summary_store.get_summary(element_id)
                        if summary:
                            es_repo.store_summary(element_id, summary)
                progress.update(task, completed=completed)

    console.print(f"  Summarized: [green]{completed}[/] elements")
    return completed


def run_embedding(
    parsing_result: ParsingResult,
    config: MagaldiConfig,
    dry_run: bool,
) -> int:
    """Run Phase 6: Embedding."""
    emb_config = EmbeddingConfig(
        ollama_url=config.ollama.url,
        model=config.ollama.embed_model,
    )

    # Check Ollama availability
    ollama = OllamaEmbedClient(emb_config.ollama_url, emb_config.model)
    if not ollama.verify_model():
        console.print(f"[yellow]Warning:[/] Model {emb_config.model} not available. Skipping embedding.")
        return 0

    # Create repositories
    if dry_run:
        job_repo = InMemoryEmbeddingJobRepository()
        embedding_store = InMemoryEmbeddingStore()
    else:
        from magaldi.db.redis import RedisEmbeddingJobRepository
        from magaldi.db.elasticsearch import ElasticsearchEmbeddingStore
        job_repo = RedisEmbeddingJobRepository(config)
        embedding_store = ElasticsearchEmbeddingStore(config)

    # Populate stores from parsing result
    embeddable = []
    for pf in parsing_result.parsed_files:
        for elem in pf.elements:
            embedding_store.store_element(elem)
            if elem.element_type in ("file", "class", "function", "method"):
                embeddable.append(elem)
                job_repo.add_job(elem.element_id)

    # Process jobs
    completed = 0
    total = len(embeddable)

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]Embedding[/]"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("({task.completed}/{task.total})"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("embedding", total=total)

        while True:
            jobs = job_repo.claim_pending_jobs("cli-worker", batch_size=10)
            if not jobs:
                break

            for job in jobs:
                element_id = job["element_id"]
                success = process_embedding_job(
                    element_id, job_repo, embedding_store, ollama, emb_config
                )
                if success:
                    completed += 1
                progress.update(task, completed=completed)

    console.print(f"  Embedded: [green]{completed}[/] elements")
    return completed


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
    types = ", ".join(f"{t}:{c}" for t, c in sorted(result.elements_by_type.items()))
    failed = f" | [red]{len(result.failed_files)} failed[/]" if result.failed_files else ""
    console.print(f"  {len(result.parsed_files)} files → {result.total_elements} elements ({types}){failed}")


def print_storage_result(result: StorageResult) -> None:
    """Print storage results."""
    parts = [f"{result.files_stored} files", f"{result.elements_stored} stored", f"{result.elements_indexed} indexed"]
    if result.elements_deleted: parts.append(f"[yellow]{result.elements_deleted} deleted[/]")
    if result.storage_errors: parts.append(f"[red]{len(result.storage_errors)} errors[/]")
    console.print(f"  {' | '.join(parts)}")


def print_summary(
    discovery: DiscoveryResult,
    manifest: ChangeManifest,
    storage: StorageResult,
    summarized: int | None,
    embedded: int | None,
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
    table.add_row("Elements stored", str(storage.elements_stored))

    if summarized is not None:
        table.add_row("Elements summarized", str(summarized))
    if embedded is not None:
        table.add_row("Elements embedded", str(embedded))

    console.print(table)
    console.print()


# =============================================================================
# ENTRY POINT
# =============================================================================


if __name__ == "__main__":
    main()
