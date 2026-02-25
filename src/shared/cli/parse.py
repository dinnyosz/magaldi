"""Parse command for the Magaldi CLI.

This module contains the parse command which runs the full parsing pipeline:
Discovery -> Change Detection -> Parsing -> Variable Scoring -> Processing ->
Call Resolution -> Feature Extraction -> Glossary.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import click
import yaml

from shared.cli._printers import (
    print_change_manifest,
    print_discovery_result,
    print_feature_result,
    print_parsing_result,
    print_processing_result,
    print_scoring_result,
    print_summary,
)
from shared.cli._runners import (
    run_call_resolution,
    run_change_detection,
    run_discovery,
    run_hierarchy_extraction,
    run_parsing,
    run_processing,
    run_variable_scoring,
)
from shared.cli._shared import check_model_availability, console, main
from shared.config import load_config

if TYPE_CHECKING:
    from magaldi_core.discovery import DiscoveryResult
    from shared.config import MagaldiConfig


def _ensure_repo_config(repo_path: str) -> None:
    """Check for magaldi.yaml and offer to create it if missing.

    Auto-detects scope and repository name, shows the user the detected values,
    and lets them edit before confirming creation. Exits if the user declines.
    """
    repo_path_obj = Path(repo_path).resolve()
    config_path = repo_path_obj / "magaldi.yaml"

    if config_path.exists():
        return

    console.print("[yellow]No magaldi.yaml found[/] in this repository.\n")

    # Auto-detect values
    detected_repo = repo_path_obj.name
    detected_scope = repo_path_obj.parent.name

    # Let the user confirm or edit the detected values
    scope = click.prompt("  Scope (groups related repos, e.g. org name)", default=detected_scope)
    repository = click.prompt("  Repository name", default=detected_repo)

    console.print()
    console.print(f"  scope: [bold]{scope}[/]")
    console.print(f"  repository: [bold]{repository}[/]")
    console.print()

    if not click.confirm("  Create magaldi.yaml with these values?", default=True):
        console.print("\n[red]Aborted.[/] Create a magaldi.yaml manually to continue.")
        sys.exit(1)

    # Write config file
    config_data = {"scope": scope, "repository": repository}
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(f"# Magaldi configuration for {repository}\n")
        f.write("# Scope groups related repositories (e.g., org name, username)\n\n")
        yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)

    console.print(f"\n[green]Created[/] {config_path}\n")


def _run_extraction_only(
    discovery_result: DiscoveryResult,
    config: MagaldiConfig,
    user: str,
    features: bool,
    glossary: bool,
    skip_ai: bool,
    workers: int,
) -> None:
    """Run feature and/or glossary extraction without parsing.

    Called when --features or --glossary is specified but no code changes detected.
    """
    from shared.cli.feature_commands import run_feature_extraction
    from shared.cli.glossary_commands import run_glossary_extraction

    if skip_ai:
        console.print("[yellow]Skipping extraction: --skip-ai is set[/]")
        return

    if features:
        console.print("\n[bold blue]Feature Extraction[/] (no code changes)")
        feature_result = run_feature_extraction(
            discovery_result.scope,
            discovery_result.repository,
            user,
            config,
            workers=workers,
        )
        if feature_result:
            print_feature_result(feature_result)

    if glossary:
        console.print("\n[bold blue]Glossary Extraction[/] (no code changes)")
        run_glossary_extraction(
            scope=discovery_result.scope,
            repository=discovery_result.repository,
            username=user,
            config=config,
            workers=workers,
        )


@main.command()
@click.argument("repo_path", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option("--user", "-u", required=True, help="Username/branch (use 'main' for primary parse)")
@click.option("--skip-ai", is_flag=True, help="Skip AI processing (summarization and embedding)")
@click.option("--features", is_flag=True, help="Run feature/subfeature extraction")
@click.option("--glossary", is_flag=True, help="Run glossary extraction")
@click.option("--skip-tests", is_flag=True, help="Skip test files and directories")
@click.option("--skip-resolve", is_flag=True, help="Skip call resolution phase (faster for large repos)")
@click.option("--dry-run", is_flag=True, help="Use in-memory storage (no database required)")
@click.option("--llm-url", default=None, help="LLM API URL (default: from config)")
@click.option("--workers", "-w", default=0, type=int, help="Max parallel workers (0=auto based on context tier)")
@click.option("--force-clean", is_flag=True, help="Delete all indexed data for this repo/user before parsing")
def parse(
    repo_path: str, user: str, skip_ai: bool, features: bool, glossary: bool, skip_tests: bool, skip_resolve: bool,
    dry_run: bool, llm_url: str | None, workers: int, force_clean: bool
) -> None:
    """Parse a repository and index its code elements.

    REPO_PATH is the path to the repository to parse.
    """
    from magaldi_core.discovery import DiscoveryError

    # Import feature/glossary runners here to avoid circular imports
    from shared.cli.feature_commands import run_feature_extraction
    from shared.cli.glossary_commands import run_glossary_extraction

    # Load configuration (skip validation in dry-run mode)
    config = load_config(skip_validation=dry_run)
    if llm_url:
        # Override URL for all models
        for model in config.llm.models.values():
            model.url = llm_url

    if dry_run:
        console.print("[yellow]Dry run mode:[/] Using in-memory storage\n")

    # Check for magaldi.yaml and offer to create it
    _ensure_repo_config(repo_path)

    try:
        # Phase 1: Discovery
        console.print("[bold blue]Phase 1:[/] Discovery")
        discovery_result = run_discovery(repo_path, user, skip_tests=skip_tests)
        print_discovery_result(discovery_result)

        # Force clean: Delete existing index data before change detection
        if force_clean and not dry_run:
            console.print("[yellow]Force clean:[/] Deleting existing index data...")
            from shared.db.store import Repository
            repo = Repository(config)
            deleted = repo.delete_by_repository(
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
            # If --features or --glossary requested, still run those phases
            if (features or glossary) and not dry_run:
                _run_extraction_only(
                    discovery_result, config, user, features, glossary, skip_ai, workers
                )
            return

        # Phase 3: Parsing
        console.print("\n[bold blue]Phase 3:[/] Parsing")
        parsing_result = run_parsing(manifest)
        print_parsing_result(parsing_result)

        # Pre-flight check: Verify Ollama models are available
        if not dry_run and not skip_ai:
            model_errors = check_model_availability(config, skip_ai)
            if model_errors:
                console.print("\n[red]Pre-flight check failed:[/]")
                for err in model_errors:
                    console.print(f"  [red]✗[/] {err}")
                sys.exit(1)

        # Phase 4: Variable Scoring (LLM-based preflight)
        if not dry_run and not skip_ai:
            scoring_model = config.llm.get_summarize_model().name
            console.print(f"\n[bold blue]Phase 4:[/] Variable Scoring [dim]({scoring_model})[/]")
            scoring_result = run_variable_scoring(parsing_result, config, workers)
            print_scoring_result(scoring_result)
        elif skip_ai:
            console.print("\n[bold blue]Phase 4:[/] Variable Scoring [dim](skipped: --skip-ai)[/]")

        # Phase 5: Processing (summarize -> embed -> index)
        console.print("\n[bold blue]Phase 5:[/] Processing")
        processed, skipped, indexed, avg_wall, avg_summ, avg_embed, elapsed, timing_stats, failed_elements, deleted = run_processing(
            parsing_result, manifest, config, dry_run, skip_ai, workers
        )
        print_processing_result(processed, skipped, indexed, skip_ai, avg_wall, avg_summ, avg_embed, elapsed, timing_stats, workers, deleted)
        # Display errors if any
        if failed_elements:
            console.print(f"\n  [red]Errors ({len(failed_elements)}):[/]")
            for element_id, error in failed_elements[:10]:  # Show first 10
                # Shorten element_id for display
                short_id = element_id.split(":")[-3:]  # format: type:name:line
                console.print(f"    [dim]{':'.join(short_id)}[/] → [red]{error}[/]")
            if len(failed_elements) > 10:
                console.print(f"    [dim]... and {len(failed_elements) - 10} more errors[/]")

        # Hierarchy Extraction (CLI commands, routes)
        if not dry_run and indexed > 0:
            from shared.db.store import Repository
            repo = Repository(config)
            console.print("\n  [bold]Hierarchy Extraction[/]")
            try:
                # Use repository name as CLI entry point fallback
                cli_entry_point = discovery_result.repository
                rel_indexed, ref_indexed = run_hierarchy_extraction(
                    discovery_result.scope,
                    discovery_result.repository,
                    user,
                    repo,
                    cli_entry_point=cli_entry_point,
                )
                if rel_indexed > 0 or ref_indexed > 0:
                    console.print(f"  Indexed {rel_indexed} relationships, {ref_indexed} external refs")
                else:
                    console.print("  [dim]No CLI/route hierarchies found[/]")
            except Exception as e:
                console.print(f"  [yellow]Warning: Hierarchy extraction failed: {e}[/]")

        # Phase 6: Call Resolution (static + embedding + semantic relationships)
        if not dry_run and indexed > 0:
            console.print("\n[bold blue]Phase 6:[/] Call Resolution")
            run_call_resolution(
                repo,
                discovery_result.scope,
                discovery_result.repository,
                user,
                skip_resolve=skip_resolve,
                console=console,
            )

        # Phase 7: Feature Extraction (opt-in with --features)
        if features and not skip_ai and not dry_run and processed > 0:
            console.print("\n[bold blue]Phase 7:[/] Feature Extraction")
            feature_result = run_feature_extraction(
                discovery_result.scope,
                discovery_result.repository,
                user,
                config,
                workers=workers,
            )
            if feature_result:
                print_feature_result(feature_result)

        # Phase 8: Glossary Extraction (opt-in with --glossary)
        if glossary and not skip_ai and not dry_run and processed > 0:
            console.print("\n[bold blue]Phase 8:[/] Glossary Extraction")
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
