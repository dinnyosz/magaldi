"""LLM server commands for the Magaldi CLI.

Manages vllm-mlx servers based on the magaldi.yaml configuration.
Groups models by URL (port) and spawns one vllm-mlx server per unique port,
with the LLM model as the primary and embedding models pre-loaded.

Usage:
    magaldi llm serve          # Start all configured vllm-mlx servers
    magaldi llm serve --port 8000  # Start only the server on port 8000
    magaldi llm status         # Show status of running servers
    magaldi llm stop           # Stop all managed servers
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

import click
import requests

from shared.cli._shared import console, main
from shared.config import LLMConfig, ModelConfig, load_config

# PID file directory
PIDFILE_DIR = Path.home() / ".magaldi" / "pids"


@dataclass
class ServerPlan:
    """Plan for a single vllm-mlx server instance."""

    port: int
    host: str
    llm_model: str  # HuggingFace MLX model repo for the LLM
    embedding_model: str | None = None  # HuggingFace MLX model repo for embeddings
    continuous_batching: bool = True
    model_configs: list[ModelConfig] = field(default_factory=list)  # source configs

    @property
    def pidfile(self) -> Path:
        return PIDFILE_DIR / f"vllm-mlx-{self.port}.pid"

    @property
    def logfile(self) -> Path:
        return PIDFILE_DIR / f"vllm-mlx-{self.port}.log"

    def health_url(self) -> str:
        return f"http://{self.host}:{self.port}/health"

    def is_running(self) -> bool:
        """Check if server is already running on this port."""
        try:
            r = requests.get(self.health_url(), timeout=3)
            return r.status_code == 200
        except Exception:
            return False

    def get_pid(self) -> int | None:
        """Read PID from pidfile."""
        if self.pidfile.exists():
            try:
                pid = int(self.pidfile.read_text().strip())
                # Check if process is actually alive
                os.kill(pid, 0)
                return pid
            except (ValueError, ProcessLookupError, PermissionError):
                # Stale pidfile
                self.pidfile.unlink(missing_ok=True)
        return None


def _build_server_plans(llm_config: LLMConfig) -> list[ServerPlan]:
    """Build server plans from LLM config.

    Groups vllm-mlx models by port. Each unique port gets one server.
    The first LLM model (non-embedding) on a port becomes the primary model.
    Embedding models on the same port are pre-loaded via --embedding-model.
    """
    # Collect all vllm-mlx models
    vllm_models: list[tuple[str, ModelConfig]] = []
    for ref_name, model_cfg in llm_config.models.items():
        if model_cfg.provider == "vllm-mlx":
            vllm_models.append((ref_name, model_cfg))

    if not vllm_models:
        return []

    # Group by (host, port)
    groups: dict[tuple[str, int], list[tuple[str, ModelConfig]]] = {}
    for ref_name, model_cfg in vllm_models:
        parsed = urlparse(model_cfg.url)
        host = parsed.hostname or "0.0.0.0"
        port = parsed.port or 8000
        key = (host, port)
        groups.setdefault(key, []).append((ref_name, model_cfg))

    plans: list[ServerPlan] = []
    for (host, port), models in groups.items():
        # Separate LLM and embedding models
        llm_models = []
        embed_models = []
        for ref_name, cfg in models:
            if cfg.dimensions is not None:
                embed_models.append((ref_name, cfg))
            else:
                llm_models.append((ref_name, cfg))

        if not llm_models:
            console.print(
                f"  [yellow]Warning: Port {port} has only embedding models, "
                f"needs at least one LLM model[/]"
            )
            continue

        # Primary LLM model — name is the HuggingFace MLX repo
        primary_ref, primary_cfg = llm_models[0]

        # Embedding model (first one, if any)
        embed_name = None
        if embed_models:
            _embed_ref, embed_cfg = embed_models[0]
            embed_name = embed_cfg.name

        plan = ServerPlan(
            port=port,
            host=host,
            llm_model=primary_cfg.name,
            embedding_model=embed_name,
            model_configs=[cfg for _, cfg in models],
        )
        plans.append(plan)

        if len(llm_models) > 1:
            extra = [ref for ref, _ in llm_models[1:]]
            console.print(
                f"  [dim]Note: Port {port} has multiple LLM models. "
                f"Primary: {primary_ref}. Others ignored: {', '.join(extra)}[/]"
            )
            console.print(
                "  [dim](vllm-mlx serves one LLM per process — "
                "use different ports for multiple models)[/]"
            )

    return plans


def _find_vllm_python() -> str:
    """Find a Python interpreter that has vllm-mlx installed.

    Checks sys.executable first, then looks for a .venv in the project root.
    Falls back to sys.executable if nothing better is found.
    """
    # Check if current Python has vllm_mlx
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import vllm_mlx"],
            capture_output=True, timeout=5,
        )
        if result.returncode == 0:
            return sys.executable
    except Exception:
        pass

    # Look for .venv in common locations
    for candidate_dir in [Path.cwd(), Path(__file__).resolve().parent.parent.parent.parent]:
        venv_python = candidate_dir / ".venv" / "bin" / "python"
        if venv_python.exists():
            try:
                result = subprocess.run(
                    [str(venv_python), "-c", "import vllm_mlx"],
                    capture_output=True, timeout=5,
                )
                if result.returncode == 0:
                    return str(venv_python)
            except Exception:
                pass

    # Fallback
    return sys.executable


def _start_server(plan: ServerPlan) -> bool:
    """Start a vllm-mlx server as a background process."""
    if plan.is_running():
        console.print(f"  [green]Already running[/] on port {plan.port}")
        return True

    python_bin = _find_vllm_python()

    # Build command using the newer vllm_mlx.cli entrypoint which supports
    # KV cache quantization, paged cache, chunked prefill, and other options.
    cmd = [
        python_bin, "-m", "vllm_mlx.cli", "serve",
        plan.llm_model,
        "--port", str(plan.port),
        "--host", plan.host,
    ]

    if plan.continuous_batching:
        cmd.append("--continuous-batching")

    if plan.embedding_model:
        cmd.extend(["--embedding-model", plan.embedding_model])

    # -- KV cache quantization --
    # 4-bit quantization reduces KV cache memory ~4x vs fp16.
    # Lower min-quantize-tokens from default 256→0 so even short sequences
    # (variable scoring batches) get quantized.
    cmd.extend([
        "--kv-cache-quantization",
        "--kv-cache-quantization-bits", "4",
        "--kv-cache-min-quantize-tokens", "0",
    ])

    # -- Paged KV cache --
    # Allocates KV cache in small blocks instead of monolithically per
    # sequence.  Blocks are reused/freed dynamically, putting a hard ceiling
    # on total memory.  max-cache-blocks × block-size = max cached tokens.
    # 500 blocks × 64 tokens = 32K tokens max across all sequences.
    cmd.extend([
        "--use-paged-cache",
        "--max-cache-blocks", "500",
    ])

    # -- Prefix cache --
    # Cap stored prefix cache at 4GB.  The default (20% of RAM) is too
    # aggressive when multiple servers share the same machine.
    cmd.extend(["--cache-memory-mb", "4096"])

    # -- Reasoning parser --
    # Extract <think>...</think> into reasoning_content field server-side.
    _REASONING_PARSERS = {"qwen3": "qwen3", "deepseek-r1": "deepseek_r1"}
    primary_cfg = next((c for c in plan.model_configs if c.dimensions is None), None)
    if primary_cfg and primary_cfg.is_thinking_model():
        base = primary_cfg.name.split("/")[-1].lower()
        for family, parser in _REASONING_PARSERS.items():
            if base.startswith(family):
                cmd.extend(["--reasoning-parser", parser])
                break

    # -- Concurrency & scheduling --
    # Limit concurrent sequences to prevent Metal memory explosion.
    cmd.extend(["--max-num-seqs", "2"])

    # Chunked prefill: split long prompts into chunks so active requests
    # aren't starved while a big prompt is being prefilled.
    cmd.extend(["--chunked-prefill-tokens", "512"])

    # Ensure pid directory exists
    PIDFILE_DIR.mkdir(parents=True, exist_ok=True)

    # Start as background process
    console.print(f"  Starting vllm-mlx on port {plan.port}...")
    # Show the full command so the user can see (and override) all parameters
    cmd_display = " ".join(cmd).replace(python_bin, "python -m")
    console.print(f"    [dim]{cmd_display}[/]")
    console.print(f"    Log: {plan.logfile}")

    with open(plan.logfile, "w") as log_f:
        proc = subprocess.Popen(
            cmd,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            start_new_session=True,  # Detach from parent
        )

    # Write PID
    plan.pidfile.write_text(str(proc.pid))

    # Wait for health check
    console.print("    Waiting for server to be ready...", end=" ")
    for _attempt in range(60):  # up to 60 seconds
        time.sleep(1)
        if plan.is_running():
            console.print("[green]Ready![/]")
            return True
        # Check if process died
        if proc.poll() is not None:
            console.print("[red]Process exited![/]")
            console.print(f"    Check log: {plan.logfile}")
            plan.pidfile.unlink(missing_ok=True)
            return False

    console.print("[yellow]Timeout (server may still be loading model)[/]")
    console.print(f"    Check: curl {plan.health_url()}")
    return True  # Process is alive, just slow to load


def _stop_server(plan: ServerPlan) -> bool:
    """Stop a vllm-mlx server by PID."""
    pid = plan.get_pid()
    if pid is None:
        if plan.is_running():
            console.print(
                f"  [yellow]Port {plan.port}: running but no PID file "
                f"(not managed by magaldi)[/]"
            )
            return False
        console.print(f"  [dim]Port {plan.port}: not running[/]")
        return True

    console.print(f"  Stopping vllm-mlx on port {plan.port} (PID {pid})...", end=" ")
    try:
        os.kill(pid, signal.SIGTERM)
        # Wait for graceful shutdown
        for _ in range(10):
            time.sleep(0.5)
            try:
                os.kill(pid, 0)  # Check if still alive
            except ProcessLookupError:
                break
        else:
            # Force kill if still alive
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, signal.SIGKILL)
        console.print("[green]Stopped[/]")
    except ProcessLookupError:
        console.print("[dim]Already stopped[/]")
    except PermissionError:
        console.print("[red]Permission denied[/]")
        return False

    plan.pidfile.unlink(missing_ok=True)
    return True


# =============================================================================
# CLI COMMANDS
# =============================================================================


@main.group()
def llm() -> None:
    """Manage local LLM servers (vllm-mlx)."""
    pass


@llm.command("serve")
@click.option(
    "--port", "-p", type=int, default=None,
    help="Only start the server on this port",
)
@click.option(
    "--no-batch", is_flag=True,
    help="Disable continuous batching (single-user mode)",
)
def llm_serve(port: int | None, no_batch: bool) -> None:
    """Start vllm-mlx servers based on config.

    Reads vllm-mlx model definitions from magaldi.yaml, groups them by port,
    and spawns one vllm-mlx server per unique port. Servers run as background
    processes with PID files in ~/.magaldi/pids/.

    \b
    Example magaldi.yaml:
        llm:
          models:
            qwen3-4b:
              name: mlx-community/Qwen3-4B-Instruct-2507-4bit
              provider: vllm-mlx
              url: http://localhost:8000
            qwen3-embed:
              name: mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ
              provider: vllm-mlx
              url: http://localhost:8000
              dimensions: 1024
    """
    config = load_config()
    plans = _build_server_plans(config.llm)

    if not plans:
        console.print("[yellow]No vllm-mlx models found in config.[/]")
        console.print("Set provider: vllm-mlx and name to the HuggingFace MLX repo in your magaldi.yaml")
        return

    if port is not None:
        plans = [p for p in plans if p.port == port]
        if not plans:
            console.print(f"[red]No vllm-mlx models configured for port {port}[/]")
            return

    if no_batch:
        for plan in plans:
            plan.continuous_batching = False

    console.print(f"[bold blue]Starting {len(plans)} vllm-mlx server(s)[/]")
    console.print()

    success_count = 0
    for plan in plans:
        if _start_server(plan):
            success_count += 1
        console.print()

    if success_count == len(plans):
        console.print(f"[bold green]All {success_count} server(s) started[/]")
    else:
        console.print(
            f"[yellow]{success_count}/{len(plans)} server(s) started[/]"
        )


@llm.command("stop")
@click.option(
    "--port", "-p", type=int, default=None,
    help="Only stop the server on this port",
)
def llm_stop(port: int | None) -> None:
    """Stop managed vllm-mlx servers."""
    config = load_config()
    plans = _build_server_plans(config.llm)

    if port is not None:
        plans = [p for p in plans if p.port == port]

    if not plans:
        # Also check for any orphaned pid files
        if PIDFILE_DIR.exists():
            pidfiles = list(PIDFILE_DIR.glob("vllm-mlx-*.pid"))
            if pidfiles:
                console.print("[yellow]Found orphaned PID files:[/]")
                for pf in pidfiles:
                    port_str = pf.stem.replace("vllm-mlx-", "")
                    plan = ServerPlan(port=int(port_str), host="0.0.0.0", llm_model="")
                    _stop_server(plan)
                return
        console.print("[dim]No vllm-mlx servers to stop[/]")
        return

    console.print(f"[bold blue]Stopping {len(plans)} vllm-mlx server(s)[/]")
    for plan in plans:
        _stop_server(plan)


@llm.command("status")
def llm_status() -> None:
    """Show status of vllm-mlx servers."""
    config = load_config()
    plans = _build_server_plans(config.llm)

    if not plans:
        console.print("[dim]No vllm-mlx models configured[/]")
        console.print("Set provider: vllm-mlx and name to the HuggingFace MLX repo in your magaldi.yaml")
        return

    from rich.table import Table

    table = Table(title="vllm-mlx Servers")
    table.add_column("Port", style="cyan")
    table.add_column("Status")
    table.add_column("PID", style="dim")
    table.add_column("LLM Model")
    table.add_column("Embed Model", style="dim")
    table.add_column("Engine")

    for plan in plans:
        pid = plan.get_pid()
        running = plan.is_running()

        if running:
            status = "[green]● Running[/]"
            # Try to get engine info from health endpoint
            try:
                r = requests.get(plan.health_url(), timeout=3)
                health = r.json()
                engine = health.get("engine_type", "unknown")
            except Exception:
                engine = "unknown"
        else:
            status = "[red]○ Stopped[/]"
            engine = "-"

        table.add_row(
            str(plan.port),
            status,
            str(pid) if pid else "-",
            plan.llm_model,
            plan.embedding_model or "-",
            engine,
        )

    console.print(table)


@llm.command("logs")
@click.option(
    "--port", "-p", type=int, required=True,
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
def llm_logs(port: int, follow: bool, lines: int) -> None:
    """Show logs for a vllm-mlx server."""
    logfile = PIDFILE_DIR / f"vllm-mlx-{port}.log"

    if not logfile.exists():
        console.print(f"[red]No log file found for port {port}[/]")
        return

    if follow:
        # Use tail -f
        with contextlib.suppress(KeyboardInterrupt):
            subprocess.run(
                ["tail", "-f", "-n", str(lines), str(logfile)],
            )
    else:
        # Show last N lines
        try:
            result = subprocess.run(
                ["tail", "-n", str(lines), str(logfile)],
                capture_output=True, text=True,
            )
            console.print(result.stdout, end="")
        except Exception as e:
            console.print(f"[red]Error reading log: {e}[/]")
