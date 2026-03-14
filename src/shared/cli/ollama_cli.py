"""Ollama management commands for the Magaldi CLI.

Commands for managing Ollama models, tiers, and service health.

Usage:
    magaldi ollama status      # Health check + list loaded models
    magaldi ollama models      # Detailed model listing
    magaldi ollama pull        # Pull configured Ollama models
    magaldi ollama warmup      # Create tiered model aliases
"""

from __future__ import annotations

import click
import requests
from rich.table import Table

from shared.cli._shared import console, main
from shared.config import load_config


# =============================================================================
# CLI COMMANDS
# =============================================================================


@main.group()
def ollama() -> None:
    """Manage Ollama models and tiers."""
    pass


@ollama.command("status")
@click.option(
    "--url", type=str, default=None,
    help="Ollama API URL (default: from config or http://localhost:11434)",
)
def ollama_status(url: str | None) -> None:
    """Show Ollama connection status and loaded models."""
    base_url = url or _get_ollama_url()

    # Health check
    try:
        r = requests.get(f"{base_url}/api/tags", timeout=5)
        if r.status_code != 200:
            console.print(f"[red]○ Ollama not healthy[/] (HTTP {r.status_code})")
            return
    except requests.ConnectionError:
        console.print(f"[red]○ Ollama not running[/] at {base_url}")
        console.print("  Start with: [bold]ollama serve[/]")
        return
    except Exception as e:
        console.print(f"[red]Error: {e}[/]")
        return

    data = r.json()
    models = data.get("models", [])

    console.print(f"[green]● Ollama running[/] at {base_url}")
    console.print(f"  Models available: {len(models)}")
    console.print()

    if not models:
        console.print("  [dim]No models installed.[/]")
        console.print("  Run: [bold]ollama pull <model>[/] or [bold]magaldi ollama pull[/]")
        return

    # Show summary table
    table = Table(title="Ollama Models")
    table.add_column("Model", style="cyan")
    table.add_column("Size", justify="right")

    for model in models:
        name = model.get("name", "unknown")
        size_bytes = model.get("size", 0)
        size_str = _format_size(size_bytes) if size_bytes else "-"
        table.add_row(name, size_str)

    console.print(table)


@ollama.command("models")
@click.option(
    "--url", type=str, default=None,
    help="Ollama API URL (default: from config or http://localhost:11434)",
)
def ollama_models(url: str | None) -> None:
    """List Ollama models with details.

    Shows all installed models and highlights those referenced in magaldi.yaml.
    """
    base_url = url or _get_ollama_url()

    try:
        r = requests.get(f"{base_url}/api/tags", timeout=5)
        r.raise_for_status()
    except requests.ConnectionError:
        console.print(f"[red]Cannot connect to Ollama at {base_url}[/]")
        return
    except Exception as e:
        console.print(f"[red]Error: {e}[/]")
        return

    data = r.json()
    models = data.get("models", [])

    if not models:
        console.print("[yellow]No models installed in Ollama.[/]")
        console.print("  Run: [bold]ollama pull <model>[/]")
        return

    # Get configured model names for highlighting
    config = load_config()
    configured_models = {
        cfg.name
        for cfg in config.llm.models.values()
        if cfg.provider == "ollama"
    }

    table = Table(title="Ollama Models (detailed)")
    table.add_column("Model", style="cyan")
    table.add_column("Size", justify="right")
    table.add_column("Modified", style="dim")
    table.add_column("Config", justify="center")

    for model in models:
        name = model.get("name", "unknown")
        size_bytes = model.get("size", 0)
        size_str = _format_size(size_bytes) if size_bytes else "-"
        modified = model.get("modified_at", "-")
        if modified and "T" in modified:
            modified = modified.split("T")[0]

        in_config = "[green]✓[/]" if name in configured_models else ""
        table.add_row(name, size_str, modified, in_config)

    console.print(table)

    if configured_models:
        missing = configured_models - {m.get("name", "") for m in models}
        if missing:
            console.print()
            console.print("[yellow]Missing configured models:[/]")
            for m in sorted(missing):
                console.print(f"  - {m}")


@ollama.command("pull")
@click.option(
    "--model", "-m", type=str, default=None,
    help="Pull a specific model by name (e.g., qwen3-embedding:0.6b)",
)
@click.option(
    "--url", type=str, default=None,
    help="Ollama API URL (default: from config or http://localhost:11434)",
)
def ollama_pull(model: str | None, url: str | None) -> None:
    """Pull configured Ollama models.

    Downloads models configured with provider: ollama in magaldi.yaml.
    Use --model to pull a specific model by name.
    """
    base_url = url or _get_ollama_url()
    config = load_config()

    # Collect models to pull
    if model:
        models_to_pull = [model]
    else:
        models_to_pull = [
            cfg.name
            for cfg in config.llm.models.values()
            if cfg.provider == "ollama"
        ]

    if not models_to_pull:
        console.print("[yellow]No Ollama models found in config.[/]")
        console.print("  Add models with `provider: ollama` to magaldi.yaml")
        return

    console.print(f"[bold blue]Pulling {len(models_to_pull)} Ollama model(s)[/]")
    console.print()

    for model_name in models_to_pull:
        console.print(f"  Pulling {model_name}...", end=" ")
        try:
            r = requests.post(
                f"{base_url}/api/pull",
                json={"name": model_name, "stream": False},
                timeout=600,  # Models can be large
            )
            if r.status_code == 200:
                console.print("[green]Done[/]")
            else:
                console.print(f"[red]Failed (HTTP {r.status_code})[/]")
        except requests.ConnectionError:
            console.print(f"[red]Cannot connect to Ollama at {base_url}[/]")
            return
        except Exception as e:
            console.print(f"[red]Error: {e}[/]")

    console.print()
    console.print("[green]Done.[/]")


@ollama.command("warmup")
@click.option(
    "--url", type=str, default=None,
    help="Ollama API URL (default: from config or http://localhost:11434)",
)
def ollama_warmup(url: str | None) -> None:
    """Create tiered model aliases for context optimization.

    Pre-creates Ollama model aliases with fixed context sizes (2k, 4k, 8k, etc.)
    to avoid model reload overhead when context size changes between requests.
    """
    from shared.ai.ollama_models import warmup_ollama_tiers_verbose

    config = load_config()
    base_url = url or _get_ollama_url()

    # Find Ollama models in config
    ollama_models = [
        cfg for cfg in config.llm.models.values()
        if cfg.provider == "ollama"
    ]

    if not ollama_models:
        console.print("[yellow]No Ollama models found in config.[/]")
        return

    console.print("[bold blue]Creating tiered model aliases[/]")
    console.print()

    for cfg in ollama_models:
        litellm_model = cfg.get_litellm_model()
        console.print(f"  Model: {cfg.name}")

        result = warmup_ollama_tiers_verbose(
            litellm_model=litellm_model,
            api_base=base_url,
        )

        if result is None:
            console.print("    [dim]Skipped (not an Ollama model or already tiered)[/]")
            continue

        console.print(f"    Base model: {result.base_model}")
        console.print(f"    Created: {result.created} aliases")
        console.print(f"    Skipped: {result.skipped} (already exist)")
        if result.failed:
            console.print(f"    [red]Failed: {result.failed}[/]")
        console.print()

    console.print("[green]Warmup complete.[/]")


# =============================================================================
# HELPERS
# =============================================================================


def _get_ollama_url() -> str:
    """Get Ollama URL from config, or use default."""
    config = load_config()
    for cfg in config.llm.models.values():
        if cfg.provider == "ollama" and cfg.url:
            return cfg.url.rstrip("/")
    return "http://localhost:11434"


def _format_size(size_bytes: int) -> str:
    """Format file size in human-readable form."""
    if size_bytes >= 1_073_741_824:
        return f"{size_bytes / 1_073_741_824:.1f} GB"
    elif size_bytes >= 1_048_576:
        return f"{size_bytes / 1_048_576:.0f} MB"
    else:
        return f"{size_bytes / 1024:.0f} KB"
