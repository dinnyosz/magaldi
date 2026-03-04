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
from rich.markup import escape as rich_escape

from shared.cli._printers import (
    print_change_manifest,
    print_discovery_result,
    print_feature_result,
    print_parsing_result,
    print_phase_timings,
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
from shared.cli.parse_logger import ParseRunLogger
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
    run_logger: ParseRunLogger | None = None,
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
        if run_logger:
            run_logger.start_phase("Feature Extraction")
        console.print("\n[bold blue]Feature Extraction[/] (no code changes)")
        try:
            feature_result = run_feature_extraction(
                discovery_result.scope,
                discovery_result.repository,
                user,
                config,
                workers=workers,
            )
            if feature_result:
                print_feature_result(feature_result)
        except Exception as e:
            if run_logger:
                run_logger.log_error("feature_extraction", str(e))
            console.print(f"  [yellow]Warning: Feature extraction failed: {rich_escape(str(e))}[/]")
        finally:
            if run_logger:
                run_logger.end_phase()

    if glossary:
        if run_logger:
            run_logger.start_phase("Glossary Extraction")
        console.print("\n[bold blue]Glossary Extraction[/] (no code changes)")
        try:
            run_glossary_extraction(
                scope=discovery_result.scope,
                repository=discovery_result.repository,
                username=user,
                config=config,
                workers=workers,
            )
        except Exception as e:
            if run_logger:
                run_logger.log_error("glossary_extraction", str(e))
            console.print(f"  [yellow]Warning: Glossary extraction failed: {rich_escape(str(e))}[/]")
        finally:
            if run_logger:
                run_logger.end_phase()


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
@click.option("--use-docstrings/--no-use-docstrings", default=True, help="Use docstrings as summaries instead of LLM (default: enabled)")
def parse(
    repo_path: str, user: str, skip_ai: bool, features: bool, glossary: bool, skip_tests: bool, skip_resolve: bool,
    dry_run: bool, llm_url: str | None, workers: int, force_clean: bool, use_docstrings: bool
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

    # Initialize run logger (will be populated with scope/repo after discovery)
    run_logger: ParseRunLogger | None = None

    try:
        # Phase 1: Discovery
        console.print("[bold blue]Phase 1:[/] Discovery")
        run_logger = ParseRunLogger(repo_path, scope="unknown", repository="unknown", username=user, mode="parse")
        run_logger.start_phase("Phase 1: Discovery")
        discovery_result = run_discovery(repo_path, user, skip_tests=skip_tests)
        print_discovery_result(discovery_result)
        run_logger.end_phase({"files": discovery_result.total_files, "lines": discovery_result.total_lines})
        run_logger.log_discovery(discovery_result)

        # Now that we know scope/repo, update the logger
        run_logger.scope = discovery_result.scope
        run_logger.repository = discovery_result.repository
        # Update log path with actual scope/repo
        ts = run_logger.run_start_dt.strftime("%Y%m%d_%H%M%S")
        safe_scope = discovery_result.scope.replace("/", "_").replace("\\", "_")
        safe_repo = discovery_result.repository.replace("/", "_").replace("\\", "_")
        log_dir = Path(repo_path) / "logs"
        log_dir.mkdir(exist_ok=True)
        run_logger.log_path = log_dir / f"parse_{safe_scope}_{safe_repo}_{ts}.log"

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
        run_logger.start_phase("Phase 2: Change Detection")
        manifest = run_change_detection(discovery_result, config, dry_run)
        print_change_manifest(manifest)
        run_logger.end_phase({
            "new": len(manifest.new_files),
            "modified": len(manifest.modified_files),
            "deleted": len(manifest.deleted_files),
        })
        run_logger.log_manifest(manifest)

        if manifest.files_to_parse == 0:
            console.print("\n[green]No files to parse. Repository is up to date.[/]")
            # If --features or --glossary requested, still run those phases
            if (features or glossary) and not dry_run:
                _run_extraction_only(
                    discovery_result, config, user, features, glossary, skip_ai, workers, run_logger
                )
            # Print phase timings even for no-op runs
            print_phase_timings(run_logger)
            # Write log if there were errors
            if run_logger.has_errors:
                log_path = run_logger.write()
                console.print(f"\n  [dim]Log:[/] {log_path}")
            return

        # Phase 3: Parsing
        console.print("\n[bold blue]Phase 3:[/] Parsing")
        run_logger.start_phase("Phase 3: Parsing")
        parsing_result = run_parsing(manifest)
        print_parsing_result(parsing_result)
        run_logger.end_phase({
            "files": len(parsing_result.parsed_files),
            "elements": parsing_result.total_elements,
            "failed_files": len(parsing_result.failed_files),
        })

        # Log parsing failures
        for file_info, error in parsing_result.failed_files:
            run_logger.log_error(
                "parsing",
                f"Failed to parse: {error}",
                {"file": file_info.relative_path},
            )

        # Pre-flight check: Verify Ollama models are available
        if not dry_run and not skip_ai:
            model_errors = check_model_availability(config, skip_ai)
            if model_errors:
                console.print("\n[red]Pre-flight check failed:[/]")
                for err in model_errors:
                    console.print(f"  [red]✗[/] {err}")
                for err in model_errors:
                    run_logger.log_error("preflight", err)
                log_path = run_logger.write()
                console.print(f"\n  [dim]Log:[/] {log_path}")
                sys.exit(1)

        # Phase 4: Variable Scoring (LLM-based preflight)
        if not dry_run and not skip_ai:
            scoring_model = config.llm.get_summarize_model().name
            console.print(f"\n[bold blue]Phase 4:[/] Variable Scoring [dim]({scoring_model})[/]")
            run_logger.start_phase("Phase 4: Variable Scoring")
            scoring_result = run_variable_scoring(parsing_result, config, workers)
            print_scoring_result(scoring_result)
            run_logger.end_phase({
                "kept": scoring_result.kept,
                "dropped": scoring_result.dropped,
                "errors": scoring_result.errors,
            })
            run_logger.log_scoring_stats(scoring_result)
            # Log token usage for variable scoring
            if scoring_result.prompt_tokens > 0 or scoring_result.response_tokens > 0:
                run_logger.log_token_usage("Phase 4: Variable Scoring", {
                    "by_type": {
                        "variable": {
                            "input": scoring_result.prompt_tokens,
                            "output": scoring_result.response_tokens,
                            "count": scoring_result.batch_count,
                        },
                    },
                    "by_model": {
                        scoring_model: {
                            "input": scoring_result.prompt_tokens,
                            "output": scoring_result.response_tokens,
                            "count": scoring_result.batch_count,
                        },
                    },
                    "totals": {
                        "input": scoring_result.prompt_tokens,
                        "output": scoring_result.response_tokens,
                        "count": scoring_result.batch_count,
                    },
                })
        elif skip_ai:
            console.print("\n[bold blue]Phase 4:[/] Variable Scoring [dim](skipped: --skip-ai)[/]")

        # Phase 5: Processing (summarize -> embed -> index)
        console.print("\n[bold blue]Phase 5:[/] Processing")
        run_logger.start_phase("Phase 5: Processing")
        processed, skipped, indexed, avg_wall, avg_summ, avg_embed, elapsed, timing_stats, failed_elements, deleted = run_processing(
            parsing_result, manifest, config, dry_run, skip_ai, workers, use_docstrings=use_docstrings
        )
        run_logger.end_phase({
            "processed": processed,
            "skipped": skipped,
            "indexed": indexed,
            "deleted": deleted,
            "failed": len(failed_elements),
        })
        run_logger.log_processing_stats(
            processed, skipped, indexed, deleted, failed_elements,
            avg_wall, avg_summ, avg_embed, elapsed,
        )
        print_processing_result(processed, skipped, indexed, skip_ai, avg_wall, avg_summ, avg_embed, elapsed, timing_stats, workers, deleted)

        # Check for token budget exceeded and log details + token usage
        if timing_stats is not None:
            tier_summary = timing_stats.get_tier_accuracy_summary()
            if tier_summary.get("has_issues"):
                input_rows = tier_summary.get("input", [])
                output_rows = tier_summary.get("output", [])
                run_logger.log_budget_exceeded(input_rows, output_rows)
            # Log token usage for this phase
            run_logger.log_token_usage("Phase 5: Processing", timing_stats.get_token_usage_summary())

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
            run_logger.start_phase("Hierarchy Extraction")
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
                run_logger.end_phase({"relationships": rel_indexed, "external_refs": ref_indexed})
            except Exception as e:
                console.print(f"  [yellow]Warning: Hierarchy extraction failed: {rich_escape(str(e))}[/]")
                run_logger.log_error("hierarchy_extraction", str(e))
                run_logger.end_phase({"error": str(e)})

        # Phase 6: Call Resolution (static + embedding + semantic relationships)
        if not dry_run and indexed > 0:
            console.print("\n[bold blue]Phase 6:[/] Call Resolution")
            run_logger.start_phase("Phase 6: Call Resolution")
            try:
                run_call_resolution(
                    repo,
                    discovery_result.scope,
                    discovery_result.repository,
                    user,
                    skip_resolve=skip_resolve,
                    console=console,
                )
                run_logger.end_phase()
            except Exception as e:
                console.print(f"  [yellow]Warning: Call resolution failed: {rich_escape(str(e))}[/]")
                run_logger.log_error("call_resolution", str(e))
                run_logger.end_phase({"error": str(e)})

        # Phase 7: Feature Extraction (opt-in with --features)
        if features and not skip_ai and not dry_run and processed > 0:
            console.print("\n[bold blue]Phase 7:[/] Feature Extraction")
            run_logger.start_phase("Phase 7: Feature Extraction")
            try:
                feature_result = run_feature_extraction(
                    discovery_result.scope,
                    discovery_result.repository,
                    user,
                    config,
                    workers=workers,
                )
                if feature_result:
                    print_feature_result(feature_result)
                    # Log token usage for feature extraction
                    token_usage = feature_result.get("token_usage")
                    if token_usage:
                        run_logger.log_token_usage("Phase 7: Feature Extraction", token_usage)
                run_logger.end_phase()
            except Exception as e:
                console.print(f"  [yellow]Warning: Feature extraction failed: {rich_escape(str(e))}[/]")
                run_logger.log_error("feature_extraction", str(e))
                run_logger.end_phase({"error": str(e)})

        # Phase 8: Glossary Extraction (opt-in with --glossary)
        if glossary and not skip_ai and not dry_run and processed > 0:
            console.print("\n[bold blue]Phase 8:[/] Glossary Extraction")
            run_logger.start_phase("Phase 8: Glossary Extraction")
            try:
                glossary_result = run_glossary_extraction(
                    scope=discovery_result.scope,
                    repository=discovery_result.repository,
                    username=user,
                    config=config,
                    workers=workers,
                )
                # Log token usage for glossary extraction
                if glossary_result:
                    token_usage = glossary_result.get("token_usage")
                    if token_usage:
                        run_logger.log_token_usage("Phase 8: Glossary Extraction", token_usage)
                run_logger.end_phase()
            except Exception as e:
                console.print(f"  [yellow]Warning: Glossary extraction failed: {rich_escape(str(e))}[/]")
                run_logger.log_error("glossary_extraction", str(e))
                run_logger.end_phase({"error": str(e)})

        print_summary(discovery_result, manifest, processed, indexed, skip_ai)
        print_phase_timings(run_logger)

        # Always write log file; show path when there are errors or budget issues
        log_path = run_logger.write()
        if run_logger.has_errors or run_logger.has_budget_exceeded:
            console.print(f"  [dim]Log:[/] {log_path}")

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted, cancelling workers...[/]")
        if run_logger is not None:
            run_logger.log_error("parse", "Interrupted by user (Ctrl+C)")
            import contextlib
            with contextlib.suppress(Exception):
                run_logger.write()
        # Force immediate exit - os._exit skips Python cleanup (no threading errors)
        import os
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(130)
    except DiscoveryError as e:
        console.print(f"\n[red]Discovery error:[/] {rich_escape(str(e))}")
        if run_logger is not None:
            run_logger.log_error("discovery", str(e))
            log_path = run_logger.write()
            console.print(f"  [dim]Log:[/] {log_path}")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[red]Error:[/] {rich_escape(str(e))}")
        if run_logger is not None:
            run_logger.log_error("parse", str(e))
            log_path = run_logger.write()
            console.print(f"  [dim]Log:[/] {log_path}")
        if "--dry-run" not in sys.argv:
            console.print("[dim]Hint: Use --dry-run to test without database[/]")
        sys.exit(1)
