"""Watch command for the Magaldi CLI.

Continuously monitors a repository for file changes and auto-parses modified files.
"""

from __future__ import annotations

import fnmatch
import queue
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import click
from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from shared.cli._printers import print_parsing_result
from shared.cli._runners import run_change_detection, run_discovery, run_processing
from shared.cli._shared import check_model_availability, console, main
from shared.config import load_config

if TYPE_CHECKING:
    from magaldi_core.discovery import DiscoveryResult
    from shared.config import MagaldiConfig

# Supported file extensions (from discovery.py)
SUPPORTED_EXTENSIONS: set[str] = {
    ".py", ".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx", ".php", ".rs"
}


class MagaldiFileHandler(FileSystemEventHandler):
    """Handle file system events and queue changes for processing."""

    def __init__(
        self,
        change_queue: queue.Queue,
        discovery_result: DiscoveryResult,
    ):
        self.change_queue = change_queue
        self.discovery_result = discovery_result
        self.repo_path = discovery_result.repo_path

    def _should_process(self, path: str) -> bool:
        """Check if file should be processed based on extension and excludes."""
        try:
            file_path = Path(path)
            rel_path = file_path.relative_to(self.repo_path)
        except ValueError:
            # Path is not relative to repo
            return False

        # Check extension
        if file_path.suffix not in SUPPORTED_EXTENSIONS:
            return False

        # Check excluded directories
        for part in rel_path.parts[:-1]:  # Exclude filename from directory check
            for exclude in self.discovery_result.exclude_directories:
                if fnmatch.fnmatch(part, exclude):
                    return False

        # Check excluded files
        for exclude in self.discovery_result.exclude_files:
            if fnmatch.fnmatch(rel_path.name, exclude):
                return False

        return True

    def on_modified(self, event: FileSystemEvent) -> None:
        if not event.is_directory and self._should_process(event.src_path):
            self.change_queue.put(("modified", event.src_path))

    def on_created(self, event: FileSystemEvent) -> None:
        if not event.is_directory and self._should_process(event.src_path):
            self.change_queue.put(("created", event.src_path))

    def on_deleted(self, event: FileSystemEvent) -> None:
        if not event.is_directory and self._should_process(event.src_path):
            self.change_queue.put(("deleted", event.src_path))


def process_file_changes(
    changed_files: set[str],
    deleted_files: set[str],
    discovery_result: DiscoveryResult,
    config: MagaldiConfig,
    user: str,
    skip_ai: bool,
    skip_resolve: bool,
    workers: int,
    features: bool,
    glossary: bool,
) -> None:
    """Process accumulated file changes through the parse pipeline."""
    from magaldi_core.change_detection import ChangeManifest, FileInfo
    from magaldi_core.code_parser import parse_files
    from shared.cli.extract import run_feature_extraction, run_glossary_extraction

    # Build FileInfo objects for changed files
    new_or_modified = []
    for file_path in changed_files:
        path = Path(file_path)
        rel_path = str(path.relative_to(discovery_result.repo_path))

        # Determine language from extension
        ext = path.suffix
        lang_map = {
            ".py": "python",
            ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript", ".cjs": "javascript",
            ".ts": "typescript", ".tsx": "tsx",
            ".php": "php",
            ".rs": "rust",
        }
        language = lang_map.get(ext, "unknown")

        # Compute file hash
        import hashlib
        try:
            content = path.read_bytes()
            file_hash = hashlib.sha256(content).hexdigest()
        except Exception:
            continue  # Skip files we can't read

        file_info = FileInfo(
            absolute_path=path,
            relative_path=rel_path,
            language=language,
            hash=file_hash,
        )
        new_or_modified.append(file_info)

    # Build FileInfo for deleted files
    deleted_infos = []
    for file_path in deleted_files:
        path = Path(file_path)
        try:
            rel_path = str(path.relative_to(discovery_result.repo_path))
        except ValueError:
            continue

        ext = path.suffix
        lang_map = {
            ".py": "python",
            ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript", ".cjs": "javascript",
            ".ts": "typescript", ".tsx": "tsx",
            ".php": "php",
            ".rs": "rust",
        }
        language = lang_map.get(ext, "unknown")

        file_info = FileInfo(
            absolute_path=path,
            relative_path=rel_path,
            language=language,
            hash="",  # No hash for deleted files
        )
        deleted_infos.append(file_info)

    if not new_or_modified and not deleted_infos:
        console.print("  [dim]No valid files to process[/]")
        return

    # Handle deleted files first (no parsing needed)
    if deleted_infos:
        from shared.db.elasticsearch import ElasticsearchRepository
        es_repo = ElasticsearchRepository(config)
        total_deleted = 0
        for file_info in deleted_infos:
            count = es_repo.delete_by_file(
                discovery_result.scope, discovery_result.repository, user, file_info.relative_path
            )
            if count > 0:
                console.print(f"  [red]Deleted[/] {count} elements from {file_info.relative_path}")
                total_deleted += count
        if total_deleted == 0 and not new_or_modified:
            console.print("  [dim]No indexed elements found for deleted files[/]")

    # If no files to parse, we're done
    if not new_or_modified:
        return

    # Create a minimal change manifest for parsing
    manifest = ChangeManifest(
        scope=discovery_result.scope,
        repository=discovery_result.repository,
        username=user,
        timestamp=datetime.now(),
        total_files_scanned=len(new_or_modified),
        new_files=[],  # Treat all as modified for simplicity
        modified_files=new_or_modified,
        deleted_files=[],  # Already handled above
        unchanged_count=0,
    )

    console.print(f"  [bold blue]Parsing[/] {len(new_or_modified)} file(s)")

    # Phase 3: Parsing
    parsing_result = parse_files(manifest)
    print_parsing_result(parsing_result)

    if parsing_result.total_elements == 0:
        console.print("  [dim]No elements extracted[/]")
        return

    # Pre-flight check for AI processing
    if not skip_ai:
        model_errors = check_model_availability(config, skip_ai)
        if model_errors:
            console.print("  [yellow]AI models unavailable, skipping AI processing[/]")
            skip_ai = True

    # Phase 4: Processing
    console.print("  [bold blue]Processing[/]")

    # Build file hashes dict
    file_hashes: dict[str, str] = {}
    for fi in new_or_modified:
        file_hashes[fi.relative_path] = fi.hash

    processed, skipped, indexed, avg_wall, avg_summ, avg_embed, elapsed, timing_stats, failed_elements, deleted = run_processing(
        parsing_result, manifest, config, dry_run=False, skip_ai=skip_ai, workers=workers, skip_resolve=skip_resolve
    )

    console.print(f"  Processed {processed} elements, indexed {indexed}")

    # Feature extraction (if requested and we processed elements)
    if features and not skip_ai and processed > 0:
        console.print("  [bold blue]Feature Extraction[/]")
        run_feature_extraction(
            discovery_result.scope,
            discovery_result.repository,
            user,
            config,
            workers=workers,
        )

    # Glossary extraction (if requested)
    if glossary and not skip_ai and processed > 0:
        console.print("  [bold blue]Glossary Extraction[/]")
        run_glossary_extraction(
            scope=discovery_result.scope,
            repository=discovery_result.repository,
            username=user,
            config=config,
            workers=workers,
        )


def watch_loop(
    change_queue: queue.Queue,
    discovery_result: DiscoveryResult,
    config: MagaldiConfig,
    user: str,
    debounce: float,
    skip_ai: bool,
    skip_resolve: bool,
    workers: int,
    features: bool,
    glossary: bool,
    stop_event: threading.Event,
) -> None:
    """Main watch loop - collect changes and process in batches."""
    changed_files: set[str] = set()
    deleted_files: set[str] = set()
    last_change_time = 0.0

    while not stop_event.is_set():
        try:
            # Wait for changes (with timeout for debounce check)
            event_type, path = change_queue.get(timeout=0.1)

            timestamp = datetime.now().strftime("%H:%M:%S")

            if event_type == "deleted":
                deleted_files.add(path)
                changed_files.discard(path)
                rel_path = Path(path).relative_to(discovery_result.repo_path)
                console.print(f"  [dim]{timestamp}[/] [red]Deleted:[/] {rel_path}")
            else:
                changed_files.add(path)
                deleted_files.discard(path)
                rel_path = Path(path).relative_to(discovery_result.repo_path)
                action = "Created" if event_type == "created" else "Modified"
                console.print(f"  [dim]{timestamp}[/] [cyan]{action}:[/] {rel_path}")

            last_change_time = time.time()

        except queue.Empty:
            # Check if debounce period has passed and we have changes to process
            if (
                (changed_files or deleted_files)
                and last_change_time > 0
                and time.time() - last_change_time >= debounce
            ):
                # Process batch
                total = len(changed_files) + len(deleted_files)
                console.print(f"\n[bold cyan]Processing {total} file(s)...[/]")

                try:
                    process_file_changes(
                        changed_files.copy(),
                        deleted_files.copy(),
                        discovery_result,
                        config,
                        user,
                        skip_ai,
                        skip_resolve,
                        workers,
                        features,
                        glossary,
                    )
                except Exception as e:
                    console.print(f"  [red]Error processing changes:[/] {e}")

                changed_files.clear()
                deleted_files.clear()
                last_change_time = 0.0

                console.print("\n[green]✓ Watching for changes...[/]\n")


@main.command()
@click.argument("repo_path", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option("--user", "-u", required=True, help="Username/branch (use 'main' for primary parse)")
@click.option("--skip-ai", is_flag=True, help="Skip AI processing (summarization and embedding)")
@click.option("--features", is_flag=True, help="Run feature extraction after changes")
@click.option("--glossary", is_flag=True, help="Run glossary extraction after changes")
@click.option("--skip-tests", is_flag=True, help="Skip test files and directories")
@click.option("--skip-resolve", is_flag=True, help="Skip call resolution phase")
@click.option("--llm-url", default=None, help="LLM API URL (default: from config)")
@click.option("--workers", "-w", default=0, type=int, help="Max parallel workers (0=auto)")
@click.option("--debounce", default=1.0, type=float, help="Seconds to wait after last change before processing")
@click.option("--no-initial-scan", is_flag=True, help="Skip initial change detection scan")
def watch(
    repo_path: str,
    user: str,
    skip_ai: bool,
    features: bool,
    glossary: bool,
    skip_tests: bool,
    skip_resolve: bool,
    llm_url: str | None,
    workers: int,
    debounce: float,
    no_initial_scan: bool,
) -> None:
    """Watch a repository for changes and auto-parse.

    REPO_PATH is the path to the repository to watch.

    This command continuously monitors the repository for file changes
    and automatically re-parses modified files. Changes are batched
    using a debounce period (default 1 second) to avoid processing
    on every keystroke.

    Example:
        magaldi watch . --user main
        magaldi watch /path/to/repo --user dev --skip-ai
    """
    from magaldi_core.discovery import DiscoveryError

    # Load configuration
    config = load_config()
    if llm_url:
        for model in config.llm.models.values():
            model.url = llm_url

    console.print("[bold blue]Magaldi Watch Mode[/]\n")

    try:
        # Phase 1: Discovery
        console.print("[bold blue]Discovery[/]")
        discovery_result = run_discovery(repo_path, user, skip_tests=skip_tests)
        console.print(f"  Repository: [cyan]{discovery_result.scope}/{discovery_result.repository}[/]")
        console.print(f"  User: [cyan]{user}[/]")
        console.print(f"  Path: [dim]{discovery_result.repo_path}[/]")
        console.print(f"  Files: [cyan]{discovery_result.total_files}[/]")

        # Initial scan and processing (optional)
        if not no_initial_scan:
            console.print("\n[bold blue]Change Detection[/]")
            manifest = run_change_detection(discovery_result, config, dry_run=False)

            if manifest.files_to_parse > 0:
                console.print(f"  Found {manifest.files_to_parse} file(s) to parse")

                # Pre-flight check for AI processing
                if not skip_ai:
                    model_errors = check_model_availability(config, skip_ai)
                    if model_errors:
                        console.print("  [yellow]AI models unavailable, skipping AI processing[/]")
                        skip_ai = True

                # Phase 3: Parsing
                from shared.cli._printers import print_parsing_result, print_processing_result
                from shared.cli._runners import run_hierarchy_extraction, run_parsing

                console.print("\n[bold blue]Parsing[/]")
                parsing_result = run_parsing(manifest)
                print_parsing_result(parsing_result)

                if parsing_result.total_elements > 0:
                    # Phase 4: Processing
                    console.print("\n[bold blue]Processing[/]")
                    processed, skipped, indexed, avg_wall, avg_summ, avg_embed, elapsed, timing_stats, failed_elements, deleted = run_processing(
                        parsing_result, manifest, config, dry_run=False, skip_ai=skip_ai, workers=workers, skip_resolve=skip_resolve
                    )
                    print_processing_result(processed, skipped, indexed, skip_ai, avg_wall, avg_summ, avg_embed, elapsed, timing_stats, workers, deleted)

                    # Hierarchy Extraction
                    if indexed > 0:
                        from shared.db.elasticsearch import ElasticsearchRepository
                        es_repo = ElasticsearchRepository(config)
                        console.print("\n  [bold]Hierarchy Extraction[/]")
                        try:
                            cli_entry_point = discovery_result.repository
                            rel_indexed, ref_indexed = run_hierarchy_extraction(
                                discovery_result.scope,
                                discovery_result.repository,
                                user,
                                es_repo,
                                cli_entry_point=cli_entry_point,
                            )
                            if rel_indexed > 0 or ref_indexed > 0:
                                console.print(f"  Indexed {rel_indexed} relationships, {ref_indexed} external refs")
                        except Exception as e:
                            console.print(f"  [yellow]Warning: Hierarchy extraction failed: {e}[/]")

                    # Feature extraction (if requested)
                    if features and not skip_ai and processed > 0:
                        from shared.cli.extract import run_feature_extraction
                        console.print("\n[bold blue]Feature Extraction[/]")
                        run_feature_extraction(
                            discovery_result.scope,
                            discovery_result.repository,
                            user,
                            config,
                            workers=workers,
                        )

                    # Glossary extraction (if requested)
                    if glossary and not skip_ai and processed > 0:
                        from shared.cli.extract import run_glossary_extraction
                        console.print("\n[bold blue]Glossary Extraction[/]")
                        run_glossary_extraction(
                            discovery_result.scope,
                            discovery_result.repository,
                            user,
                            config,
                            workers=workers,
                        )
            else:
                console.print("  [green]Repository is up to date[/]")

        # Set up file watcher
        change_queue: queue.Queue = queue.Queue()
        event_handler = MagaldiFileHandler(change_queue, discovery_result)
        observer = Observer()
        observer.schedule(event_handler, str(discovery_result.repo_path), recursive=True)

        # Start watching
        console.print(f"\n[green]✓ Watching for changes...[/] [dim](debounce: {debounce}s)[/]\n")

        stop_event = threading.Event()
        observer.start()

        try:
            watch_loop(
                change_queue,
                discovery_result,
                config,
                user,
                debounce,
                skip_ai,
                skip_resolve,
                workers,
                features,
                glossary,
                stop_event,
            )
        except KeyboardInterrupt:
            console.print("\n[yellow]Stopping watch...[/]")
            stop_event.set()
            observer.stop()
            observer.join(timeout=2.0)

    except DiscoveryError as e:
        console.print(f"\n[red]Discovery error:[/] {e}")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[red]Error:[/] {e}")
        sys.exit(1)
