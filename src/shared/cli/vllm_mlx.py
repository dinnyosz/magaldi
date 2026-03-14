"""vllm-mlx server commands for the Magaldi CLI.

Manages a vllm-mlx inference server for Apple Silicon (MLX) models with
continuous batching and OpenAI-compatible API.

Usage:
    magaldi vllm-mlx serve     # Start vllm-mlx server
    magaldi vllm-mlx stop      # Stop the running server
    magaldi vllm-mlx status    # Show server status + loaded model
    magaldi vllm-mlx logs      # Follow server logs
"""

from __future__ import annotations

import shutil
import subprocess
import time

import click
import requests
from rich.markup import escape as rich_escape
from rich.table import Table

from shared.cli._server_utils import (
    PIDFILE_DIR,
    get_pid,
    get_port_from_config,
    is_healthy,
    logfile,
    pidfile,
    stop_server,
    tail_log,
)
from shared.cli._shared import console, main
from shared.config import load_config

# Prefix for PID/log files
PREFIX = "vllm-mlx"
DEFAULT_PORT = 8000


# =============================================================================
# CLI COMMANDS
# =============================================================================


@main.group()
def vllm_mlx() -> None:
    """Manage vllm-mlx server (MLX inference)."""
    pass


@vllm_mlx.command("serve")
@click.option(
    "--port", "-p", type=int, default=None,
    help=f"Server port (default: {DEFAULT_PORT})",
)
@click.option(
    "--model", "-m", type=str, default=None,
    help="HuggingFace model ID (e.g., mlx-community/Qwen3-4B-4bit)",
)
@click.option(
    "--embedding-model", type=str, default=None,
    help="Embedding model ID (optional)",
)
def vllm_mlx_serve(
    port: int | None,
    model: str | None,
    embedding_model: str | None,
) -> None:
    """Start vllm-mlx inference server.

    Serves MLX models with continuous batching via an OpenAI-compatible API.
    Requires vllm-mlx to be installed (pip install vllm-mlx).

    \b
    Example:
        magaldi vllm-mlx serve --model mlx-community/Qwen3-4B-4bit
    """
    # Check if vllm-mlx is installed
    if not shutil.which("vllm-mlx"):
        console.print("[red]vllm-mlx not found.[/]")
        console.print("  Install with: [bold]pip install vllm-mlx[/]")
        return

    # Resolve port and model from config if not specified
    config = load_config()
    port = port or get_port_from_config("vllm-mlx", config.llm.models, DEFAULT_PORT)

    if model is None:
        # Find first vllm-mlx model in config
        for cfg in config.llm.models.values():
            if cfg.provider == "vllm-mlx":
                model = cfg.name
                break

    if model is None:
        console.print("[red]No model specified.[/]")
        console.print("  Use --model or configure a vllm-mlx model in magaldi.yaml")
        return

    # Check if already running
    existing_pid = get_pid(PREFIX, port)
    if existing_pid is not None:
        if is_healthy(port):
            console.print(f"[green]vllm-mlx already running[/] on port {port} (PID {existing_pid})")
            return
        else:
            console.print(f"[yellow]Stale process detected (PID {existing_pid}), cleaning up...[/]")
            stop_server(PREFIX, port, console)
            time.sleep(1)

    # Build command
    # --enable-prefix-cache is already on by default
    # NOTE: --chunked-prefill-tokens causes 'list has no attribute shape'
    #       crash in vllm-mlx 0.2.6. Omitted until upstream fix.
    cmd = [
        "vllm-mlx", "serve", model,
        "--port", str(port),
        "--continuous-batching",
    ]

    if embedding_model:
        cmd.extend(["--embedding-model", embedding_model])

    # Ensure directories exist
    PIDFILE_DIR.mkdir(parents=True, exist_ok=True)

    # Print startup info
    console.print("[bold blue]Starting vllm-mlx server[/]")
    console.print()
    console.print(f"  Model:          {model}")
    console.print(f"  Port:           {port}")
    if embedding_model:
        console.print(f"  Embeddings:     {embedding_model}")
    console.print("  Cont. batching: on")
    console.print(f"  Log:            {logfile(PREFIX, port)}")
    console.print()

    # Start as background process
    log_path = logfile(PREFIX, port)
    with open(log_path, "w") as log_f:
        proc = subprocess.Popen(
            cmd,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    # Write PID
    pidfile(PREFIX, port).write_text(str(proc.pid))

    # Wait for health check
    console.print("  Waiting for server...", end=" ")
    for _attempt in range(60):  # up to 60 seconds (model download can be slow)
        time.sleep(1)
        if is_healthy(port):
            console.print("[green]Ready![/]")
            console.print()
            console.print(f"  API: http://localhost:{port}/v1/models")
            console.print(f"  Health: http://localhost:{port}/health")
            return
        # Check if process died
        if proc.poll() is not None:
            console.print("[red]Process exited![/]")
            console.print(f"  Check log: {log_path}")
            pidfile(PREFIX, port).unlink(missing_ok=True)
            return

    console.print("[yellow]Timeout (model may still be downloading)[/]")
    console.print(f"  Check: curl http://localhost:{port}/health")


@vllm_mlx.command("stop")
@click.option(
    "--port", "-p", type=int, default=None,
    help="Port of the server to stop",
)
def vllm_mlx_stop(port: int | None) -> None:
    """Stop the vllm-mlx server."""
    if port is None:
        config = load_config()
        port = get_port_from_config("vllm-mlx", config.llm.models, DEFAULT_PORT)

    stop_server(PREFIX, port, console)


@vllm_mlx.command("status")
@click.option(
    "--port", "-p", type=int, default=None,
    help="Port of the server to check",
)
def vllm_mlx_status(port: int | None) -> None:
    """Show vllm-mlx server status and loaded models."""
    if port is None:
        config = load_config()
        port = get_port_from_config("vllm-mlx", config.llm.models, DEFAULT_PORT)

    pid = get_pid(PREFIX, port)
    healthy = is_healthy(port)

    # Server status
    if pid and healthy:
        console.print(f"[green]● vllm-mlx running[/] on port {port} (PID {pid})")
    elif pid:
        console.print(f"[yellow]● vllm-mlx starting[/] on port {port} (PID {pid})")
    else:
        console.print(f"[red]○ vllm-mlx not running[/] on port {port}")
        console.print("  Run: [bold]magaldi vllm-mlx serve[/]")
        return

    console.print()

    # Query loaded models
    try:
        r = requests.get(f"http://localhost:{port}/v1/models", timeout=5)
        if r.status_code != 200:
            console.print(f"  [yellow]Could not fetch models (HTTP {r.status_code})[/]")
            return

        data = r.json()
        models = data.get("data", [])

        if not models:
            console.print("  No models loaded.")
            return

        table = Table(title="Models")
        table.add_column("Model ID", style="cyan")
        table.add_column("Type", style="dim")

        for model in models:
            model_id = model.get("id", "unknown")
            obj_type = model.get("object", "model")
            table.add_row(model_id, obj_type)

        console.print(table)

    except requests.ConnectionError:
        console.print("  [red]Could not connect to server[/]")
    except Exception as e:
        console.print(f"  [red]Error: {rich_escape(str(e))}[/]")


@vllm_mlx.command("logs")
@click.option(
    "--port", "-p", type=int, default=None,
    help="Port of the server to show logs for",
)
@click.option(
    "--follow", "-f", is_flag=True,
    help="Follow log output (like tail -f)",
)
@click.option(
    "--lines", "-n", type=int, default=50,
    help="Number of lines to show (default: 50)",
)
def vllm_mlx_logs(port: int | None, follow: bool, lines: int) -> None:
    """Show logs for the vllm-mlx server."""
    if port is None:
        config = load_config()
        port = get_port_from_config("vllm-mlx", config.llm.models, DEFAULT_PORT)

    tail_log(PREFIX, port, console, follow=follow, lines=lines)
