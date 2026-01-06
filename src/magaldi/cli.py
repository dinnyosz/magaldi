"""Magaldi CLI - Code discovery engine for AI agents and developers.

Commands:
    magaldi parse /path/to/repo --user <username>
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from magaldi.parser.change_detection import (
    ChangeManifest,
    InMemoryFileStateRepository,
    detect_changes,
)
from magaldi.parser.code_parser import parse_files
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
@click.option("--ollama-url", default="http://localhost:11434", help="Ollama API URL")
def parse(repo_path: str, user: str, skip_ai: bool, ollama_url: str) -> None:
    """Parse a repository and index its code elements.

    REPO_PATH is the path to the repository to parse.
    """
    try:
        # Phase 1: Discovery
        console.print("\n[bold blue]Phase 1:[/] Discovery")
        discovery_result = run_discovery(repo_path, user)
        print_discovery_result(discovery_result)

        # Phase 2: Change Detection
        console.print("\n[bold blue]Phase 2:[/] Change Detection")
        file_state_repo = InMemoryFileStateRepository()
        manifest = run_change_detection(discovery_result, file_state_repo)
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
        db_repo = InMemoryDatabaseRepository()
        search_repo = InMemorySearchRepository()
        storage_result = run_storage(parsing_result, manifest, db_repo, search_repo)
        print_storage_result(storage_result)

        if skip_ai:
            console.print("\n[yellow]Skipping AI processing (--skip-ai flag)[/]")
            print_summary(discovery_result, manifest, storage_result, None, None)
            return

        # Phase 5: Summarization
        console.print("\n[bold blue]Phase 5:[/] Summarization")
        summarization_count = run_summarization(
            parsing_result, db_repo, ollama_url
        )

        # Phase 6: Embedding
        console.print("\n[bold blue]Phase 6:[/] Embedding")
        embedding_count = run_embedding(
            parsing_result, db_repo, ollama_url
        )

        print_summary(
            discovery_result, manifest, storage_result, summarization_count, embedding_count
        )

    except DiscoveryError as e:
        console.print(f"\n[red]Discovery error:[/] {e}")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[red]Error:[/] {e}")
        sys.exit(1)


# =============================================================================
# PHASE RUNNERS
# =============================================================================


def run_discovery(repo_path: str, username: str) -> DiscoveryResult:
    """Run Phase 1: Discovery."""
    with console.status("Discovering repository..."):
        return discover(repo_path, username)


def run_change_detection(
    discovery_result: DiscoveryResult,
    file_state_repo: InMemoryFileStateRepository,
) -> ChangeManifest:
    """Run Phase 2: Change Detection."""
    with console.status("Detecting changes..."):
        return detect_changes(discovery_result, file_state_repo)


def run_parsing(manifest: ChangeManifest):
    """Run Phase 3: Parsing."""
    with console.status(f"Parsing {manifest.files_to_parse} files..."):
        return parse_files(manifest)


def run_storage(
    parsing_result,
    manifest: ChangeManifest,
    db_repo: InMemoryDatabaseRepository,
    search_repo: InMemorySearchRepository,
) -> StorageResult:
    """Run Phase 4: Storage."""
    with console.status("Storing elements..."):
        return store_storage_result(parsing_result, manifest, db_repo, search_repo)


def run_summarization(
    parsing_result,
    db_repo: InMemoryDatabaseRepository,
    ollama_url: str,
) -> int:
    """Run Phase 5: Summarization."""
    config = SummarizationConfig(ollama_url=ollama_url)

    # Check Ollama availability
    ollama = OllamaClient(config.ollama_url, config.model)
    if not ollama.verify_model():
        console.print(f"[yellow]Warning:[/] Model {config.model} not available. Skipping summarization.")
        return 0

    # Build job and summary stores from parsed elements
    job_repo = InMemoryJobRepository()
    summary_store = InMemorySummaryStore()

    # Populate stores from parsing result
    all_elements = []
    for pf in parsing_result.parsed_files:
        for elem in pf.elements:
            all_elements.append(elem)
            summary_store.store_element(elem)
            # Add job (level 0 has no dependencies)
            job_repo.add_job(
                element_id=elem.element_id,
                level=elem.level,
                parent_id=elem.parent_id,
                dependencies_met=(elem.level == 0),
            )

    # Process jobs level by level
    completed = 0
    total = len(all_elements)

    with console.status(f"Summarizing {total} elements...") as status:
        while True:
            jobs = job_repo.claim_pending_jobs("cli-worker", batch_size=10)
            if not jobs:
                break

            for job in jobs:
                element_id = job["element_id"]
                status.update(f"Summarizing: {element_id.split(':')[-2]}...")

                success = process_summarization_job(
                    element_id, job_repo, summary_store, ollama, config
                )
                if success:
                    completed += 1

    console.print(f"  Summarized: [green]{completed}[/] elements")
    return completed


def run_embedding(
    parsing_result,
    db_repo: InMemoryDatabaseRepository,
    ollama_url: str,
) -> int:
    """Run Phase 6: Embedding."""
    config = EmbeddingConfig(ollama_url=ollama_url)

    # Check Ollama availability
    ollama = OllamaEmbedClient(config.ollama_url, config.model)
    if not ollama.verify_model():
        console.print(f"[yellow]Warning:[/] Model {config.model} not available. Skipping embedding.")
        return 0

    # Build job and embedding stores
    job_repo = InMemoryEmbeddingJobRepository()
    embedding_store = InMemoryEmbeddingStore()

    # Populate stores from parsing result
    embeddable = []
    for pf in parsing_result.parsed_files:
        for elem in pf.elements:
            embedding_store.store_element(elem)
            # Only embed certain element types
            if elem.element_type in ("file", "class", "function", "method"):
                embeddable.append(elem)
                job_repo.add_job(elem.element_id)

    # Process jobs
    completed = 0
    total = len(embeddable)

    with console.status(f"Embedding {total} elements...") as status:
        while True:
            jobs = job_repo.claim_pending_jobs("cli-worker", batch_size=10)
            if not jobs:
                break

            for job in jobs:
                element_id = job["element_id"]
                status.update(f"Embedding: {element_id.split(':')[-2]}...")

                success = process_embedding_job(
                    element_id, job_repo, embedding_store, ollama, config
                )
                if success:
                    completed += 1

    console.print(f"  Embedded: [green]{completed}[/] elements")
    return completed


# =============================================================================
# OUTPUT FORMATTING
# =============================================================================


def print_discovery_result(result: DiscoveryResult) -> None:
    """Print discovery phase results."""
    console.print(f"  Scope: [cyan]{result.scope}[/]")
    console.print(f"  Repository: [cyan]{result.repository}[/]")
    console.print(f"  User: [cyan]{result.username}[/]")
    console.print(f"  Total files: [green]{result.total_files}[/]")
    console.print(f"  Total lines: [green]{result.total_lines:,}[/]")

    if result.languages:
        console.print("  Languages:")
        for lang, stats in sorted(result.languages.items(), key=lambda x: -x[1].files):
            console.print(f"    {lang}: {stats.files} files, {stats.lines:,} lines")


def print_change_manifest(manifest: ChangeManifest) -> None:
    """Print change detection results."""
    console.print(f"  Files scanned: [green]{manifest.total_files_scanned}[/]")
    console.print(f"  New: [green]{len(manifest.new_files)}[/]")
    console.print(f"  Modified: [yellow]{len(manifest.modified_files)}[/]")
    console.print(f"  Deleted: [red]{len(manifest.deleted_files)}[/]")
    console.print(f"  Unchanged: {manifest.unchanged_count}")
    if manifest.skipped_count > 0:
        console.print(f"  Skipped (same as main): {manifest.skipped_count}")


def print_parsing_result(result) -> None:
    """Print parsing results."""
    console.print(f"  Files parsed: [green]{len(result.parsed_files)}[/]")
    console.print(f"  Total elements: [green]{result.total_elements}[/]")

    if result.elements_by_type:
        console.print("  By type:")
        for etype, count in sorted(result.elements_by_type.items()):
            console.print(f"    {etype}: {count}")

    if result.failed_files:
        console.print(f"  [red]Failed: {len(result.failed_files)}[/]")


def print_storage_result(result: StorageResult) -> None:
    """Print storage results."""
    console.print(f"  Files stored: [green]{result.files_stored}[/]")
    console.print(f"  Elements stored: [green]{result.elements_stored}[/]")
    console.print(f"  Elements indexed: [green]{result.elements_indexed}[/]")
    if result.elements_deleted > 0:
        console.print(f"  Elements deleted: [yellow]{result.elements_deleted}[/]")

    if result.storage_errors:
        console.print(f"  [red]Errors: {len(result.storage_errors)}[/]")


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
