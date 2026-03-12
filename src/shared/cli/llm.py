"""LLM server commands for the Magaldi CLI.

Manages a llama-server (llama.cpp) instance in router mode, serving all GGUF
models from a shared directory with automatic hot-swapping and LRU eviction.

Usage:
    magaldi llm serve          # Start llama-server in router mode
    magaldi llm stop           # Stop the running server
    magaldi llm status         # Show server + loaded models
    magaldi llm logs           # Follow server logs
    magaldi llm models         # List available GGUF models
    magaldi llm pull           # Download configured models from HuggingFace

Example magaldi.yaml:
    llm:
      models:
        qwen3.5-4b:
          name: Qwen3.5-4B-Q4_K_M
          provider: llamacpp
          url: http://localhost:8090
          gguf: unsloth/Qwen3.5-4B-GGUF:Qwen3.5-4B-Q4_K_M.gguf
        qwen3.5-2b:
          name: Qwen3.5-2B-Q4_K_M
          provider: llamacpp
          url: http://localhost:8090
          gguf: unsloth/Qwen3.5-2B-GGUF:Qwen3.5-2B-Q4_K_M.gguf
        qwen3-embed:
          name: qwen3-embedding:0.6b
          provider: ollama
          url: http://localhost:11434
          dimensions: 1024
"""

from __future__ import annotations

import configparser
import contextlib
import os
import signal
import subprocess
import time
from pathlib import Path

import click
import requests
from rich.markup import escape as rich_escape
from rich.table import Table

from shared.cli._shared import console, main
from shared.config import LLMConfig, ModelConfig, load_config

# Directories
PIDFILE_DIR = Path.home() / ".magaldi" / "pids"
PRESETS_DIR = Path.home() / ".magaldi"
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "tools" / "models"
LLAMA_SERVER = PROJECT_ROOT / "tools" / "llama.cpp" / "build" / "bin" / "llama-server"

# Defaults
DEFAULT_PORT = 8090
DEFAULT_PARALLEL = 4
DEFAULT_MODELS_MAX = 2
DEFAULT_CTX_SIZE = 8192


# =============================================================================
# HELPERS
# =============================================================================


def _pidfile(port: int) -> Path:
    return PIDFILE_DIR / f"llama-server-{port}.pid"


def _logfile(port: int) -> Path:
    return PIDFILE_DIR / f"llama-server-{port}.log"


def _presets_file() -> Path:
    return PRESETS_DIR / "llama-presets.ini"


def _get_pid(port: int) -> int | None:
    """Read PID from pidfile, return None if stale or missing."""
    pf = _pidfile(port)
    if pf.exists():
        try:
            pid = int(pf.read_text().strip())
            os.kill(pid, 0)  # Check if alive
            return pid
        except (ValueError, ProcessLookupError, PermissionError):
            pf.unlink(missing_ok=True)
    return None


def _is_healthy(port: int) -> bool:
    """Check if llama-server is responding on the given port."""
    try:
        r = requests.get(f"http://localhost:{port}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def _get_llamacpp_models(llm_config: LLMConfig) -> list[ModelConfig]:
    """Get all llamacpp models from config."""
    return [
        cfg for cfg in llm_config.models.values()
        if cfg.provider == "llamacpp"
    ]


def _get_llamacpp_port(llm_config: LLMConfig) -> int:
    """Get the port from the first llamacpp model, or default."""
    for cfg in llm_config.models.values():
        if cfg.provider == "llamacpp":
            from urllib.parse import urlparse
            parsed = urlparse(cfg.url)
            if parsed.port:
                return parsed.port
    return DEFAULT_PORT


def _generate_presets(llm_config: LLMConfig) -> Path:
    """Generate a llama-server presets INI file from config.

    The presets file lets us set per-model context sizes and other params.
    Format: [model:<model-id>] sections with key=value pairs.
    """
    presets = configparser.ConfigParser()

    for cfg in _get_llamacpp_models(llm_config):
        section = f"model:{cfg.name}"
        presets[section] = {}
        if cfg.num_ctx:
            presets[section]["n_ctx"] = str(cfg.num_ctx)

    presets_path = _presets_file()
    presets_path.parent.mkdir(parents=True, exist_ok=True)
    with open(presets_path, "w") as f:
        presets.write(f)

    return presets_path


def _list_gguf_files() -> list[Path]:
    """List all .gguf files in the models directory."""
    if not MODELS_DIR.exists():
        return []
    return sorted(MODELS_DIR.glob("*.gguf"))


def _format_size(size_bytes: int) -> str:
    """Format file size in human-readable form."""
    if size_bytes >= 1_073_741_824:
        return f"{size_bytes / 1_073_741_824:.1f} GB"
    elif size_bytes >= 1_048_576:
        return f"{size_bytes / 1_048_576:.0f} MB"
    else:
        return f"{size_bytes / 1024:.0f} KB"


# =============================================================================
# CLI COMMANDS
# =============================================================================


@main.group()
def llm() -> None:
    """Manage local LLM server (llama.cpp)."""
    pass


@llm.command("serve")
@click.option(
    "--port", "-p", type=int, default=None,
    help=f"Server port (default: {DEFAULT_PORT})",
)
@click.option(
    "--parallel", type=int, default=None,
    help=f"Number of parallel request slots (default: {DEFAULT_PARALLEL})",
)
@click.option(
    "--models-max", type=int, default=None,
    help=f"Max models loaded simultaneously (default: {DEFAULT_MODELS_MAX})",
)
@click.option(
    "--ctx-size", type=int, default=None,
    help=f"Default context size per slot (default: {DEFAULT_CTX_SIZE})",
)
def llm_serve(
    port: int | None,
    parallel: int | None,
    models_max: int | None,
    ctx_size: int | None,
) -> None:
    """Start llama-server in router mode.

    Serves all GGUF models from tools/models/ with automatic hot-swapping.
    Models are loaded on first request and evicted LRU when --models-max
    is reached. Use `magaldi llm pull` to download models first.

    \b
    Prerequisites:
        make llama-setup     # Build llama.cpp + download models
    """
    # Check llama-server binary
    if not LLAMA_SERVER.exists():
        console.print("[red]llama-server not found.[/]")
        console.print(f"  Expected: {LLAMA_SERVER}")
        console.print("  Run: [bold]make llama-setup[/]")
        return

    # Check models directory
    gguf_files = _list_gguf_files()
    if not gguf_files:
        console.print("[red]No GGUF models found.[/]")
        console.print(f"  Expected in: {MODELS_DIR}")
        console.print("  Run: [bold]make llama-pull[/] or [bold]magaldi llm pull[/]")
        return

    # Load config for port and presets
    config = load_config()
    port = port or _get_llamacpp_port(config.llm)
    parallel = parallel or DEFAULT_PARALLEL
    models_max = models_max or DEFAULT_MODELS_MAX
    ctx_size = ctx_size or DEFAULT_CTX_SIZE

    # Check if already running
    existing_pid = _get_pid(port)
    if existing_pid is not None:
        if _is_healthy(port):
            console.print(f"[green]llama-server already running[/] on port {port} (PID {existing_pid})")
            return
        else:
            # PID exists but not healthy — stale process
            console.print(f"[yellow]Stale process detected (PID {existing_pid}), cleaning up...[/]")
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.kill(existing_pid, signal.SIGTERM)
            _pidfile(port).unlink(missing_ok=True)
            time.sleep(1)

    # Generate presets file for per-model config
    presets_path = _generate_presets(config.llm)

    # Build command
    cmd = [
        str(LLAMA_SERVER),
        "--models-dir", str(MODELS_DIR),
        "--port", str(port),
        "--host", "0.0.0.0",
        "--ctx-size", str(ctx_size),
        "--parallel", str(parallel),
        "--models-max", str(models_max),
        "--flash-attn", "on",
        "--n-gpu-layers", "99",
    ]

    # Add presets if we generated any model-specific config
    if presets_path.stat().st_size > 0:
        cmd.extend(["--models-preset", str(presets_path)])

    # Ensure directories exist
    PIDFILE_DIR.mkdir(parents=True, exist_ok=True)

    # Print startup info
    console.print("[bold blue]Starting llama-server (router mode)[/]")
    console.print()
    console.print(f"  Port:           {port}")
    console.print(f"  Models dir:     {MODELS_DIR}")
    console.print(f"  Models found:   {len(gguf_files)}")
    for gf in gguf_files:
        console.print(f"    - {gf.stem}  ({_format_size(gf.stat().st_size)})")
    console.print(f"  Max loaded:     {models_max}")
    console.print(f"  Parallel slots: {parallel}")
    console.print(f"  Context size:   {ctx_size}")
    console.print("  Flash attention: on")
    console.print("  GPU layers:     99 (full offload)")
    console.print(f"  Log:            {_logfile(port)}")
    console.print()

    # Start as background process
    logfile = _logfile(port)
    with open(logfile, "w") as log_f:
        proc = subprocess.Popen(
            cmd,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    # Write PID
    _pidfile(port).write_text(str(proc.pid))

    # Wait for health check
    console.print("  Waiting for server...", end=" ")
    for _attempt in range(30):  # up to 30 seconds
        time.sleep(1)
        if _is_healthy(port):
            console.print("[green]Ready![/]")
            console.print()
            console.print(f"  API: http://localhost:{port}/v1/models")
            console.print(f"  Health: http://localhost:{port}/health")
            return
        # Check if process died
        if proc.poll() is not None:
            console.print("[red]Process exited![/]")
            console.print(f"  Check log: {logfile}")
            _pidfile(port).unlink(missing_ok=True)
            return

    console.print("[yellow]Timeout (server may still be loading)[/]")
    console.print(f"  Check: curl http://localhost:{port}/health")


@llm.command("stop")
@click.option(
    "--port", "-p", type=int, default=None,
    help="Port of the server to stop",
)
def llm_stop(port: int | None) -> None:
    """Stop the llama-server."""
    if port is None:
        config = load_config()
        port = _get_llamacpp_port(config.llm)

    pid = _get_pid(port)
    if pid is None:
        # Check for orphaned pidfiles
        if PIDFILE_DIR.exists():
            pidfiles = list(PIDFILE_DIR.glob("llama-server-*.pid"))
            if pidfiles:
                console.print("[yellow]Found orphaned PID files, cleaning up...[/]")
                for pf in pidfiles:
                    port_str = pf.stem.replace("llama-server-", "")
                    orphan_pid = _get_pid(int(port_str))
                    if orphan_pid:
                        _stop_pid(orphan_pid, int(port_str))
                    else:
                        pf.unlink(missing_ok=True)
                return
        console.print(f"[dim]No llama-server running on port {port}[/]")
        return

    _stop_pid(pid, port)


def _stop_pid(pid: int, port: int) -> None:
    """Stop a process by PID with graceful shutdown."""
    console.print(f"  Stopping llama-server on port {port} (PID {pid})...", end=" ")
    try:
        os.kill(pid, signal.SIGTERM)
        for _ in range(10):
            time.sleep(0.5)
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
        else:
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, signal.SIGKILL)
        console.print("[green]Stopped[/]")
    except ProcessLookupError:
        console.print("[dim]Already stopped[/]")
    except PermissionError:
        console.print("[red]Permission denied[/]")
        return

    _pidfile(port).unlink(missing_ok=True)


@llm.command("status")
@click.option(
    "--port", "-p", type=int, default=None,
    help="Port of the server to check",
)
def llm_status(port: int | None) -> None:
    """Show llama-server status and loaded models."""
    if port is None:
        config = load_config()
        port = _get_llamacpp_port(config.llm)

    pid = _get_pid(port)
    healthy = _is_healthy(port)

    # Server status
    if pid and healthy:
        console.print(f"[green]● llama-server running[/] on port {port} (PID {pid})")
    elif pid:
        console.print(f"[yellow]● llama-server starting[/] on port {port} (PID {pid})")
    else:
        console.print(f"[red]○ llama-server not running[/] on port {port}")
        console.print("  Run: [bold]magaldi llm serve[/]")
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
            console.print("  No models available (is --models-dir correct?)")
            return

        table = Table(title="Models")
        table.add_column("Model ID", style="cyan")
        table.add_column("Status")
        table.add_column("Backend", style="dim")

        for model in models:
            model_id = model.get("id", "unknown")
            # Router mode provides status and path fields
            status = model.get("status", "loaded")

            if status == "loaded":
                status_str = "[green]● loaded[/]"
            elif status == "loading":
                status_str = "[yellow]◌ loading[/]"
            else:
                # Unloaded but available
                status_str = "[dim]○ available[/]"

            backend = model.get("meta", {}).get("ggml.backend", "metal") if isinstance(model.get("meta"), dict) else "-"

            table.add_row(model_id, status_str, str(backend))

        console.print(table)

    except requests.ConnectionError:
        console.print("  [red]Could not connect to server[/]")
    except Exception as e:
        console.print(f"  [red]Error: {rich_escape(str(e))}[/]")


@llm.command("logs")
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
def llm_logs(port: int | None, follow: bool, lines: int) -> None:
    """Show logs for the llama-server."""
    if port is None:
        config = load_config()
        port = _get_llamacpp_port(config.llm)

    logfile = _logfile(port)

    if not logfile.exists():
        console.print(f"[red]No log file found for port {port}[/]")
        console.print(f"  Expected: {logfile}")
        return

    if follow:
        with contextlib.suppress(KeyboardInterrupt):
            subprocess.run(
                ["tail", "-f", "-n", str(lines), str(logfile)],
            )
    else:
        try:
            result = subprocess.run(
                ["tail", "-n", str(lines), str(logfile)],
                capture_output=True, text=True,
            )
            console.print(result.stdout, end="")
        except Exception as e:
            console.print(f"[red]Error reading log: {rich_escape(str(e))}[/]")


@llm.command("models")
def llm_models() -> None:
    """List available GGUF models in tools/models/."""
    gguf_files = _list_gguf_files()

    if not gguf_files:
        console.print("[yellow]No GGUF models found.[/]")
        console.print(f"  Directory: {MODELS_DIR}")
        console.print("  Run: [bold]magaldi llm pull[/] or [bold]make llama-pull[/]")
        return

    # Check if server is running to show loaded status
    config = load_config()
    port = _get_llamacpp_port(config.llm)
    loaded_models: set[str] = set()

    if _is_healthy(port):
        try:
            r = requests.get(f"http://localhost:{port}/v1/models", timeout=3)
            if r.status_code == 200:
                for model in r.json().get("data", []):
                    if model.get("status") == "loaded":
                        loaded_models.add(model.get("id", ""))
        except Exception:
            pass

    table = Table(title=f"GGUF Models ({MODELS_DIR})")
    table.add_column("Model", style="cyan")
    table.add_column("Size", justify="right")
    table.add_column("Status")

    for gf in gguf_files:
        stem = gf.stem
        size = _format_size(gf.stat().st_size)
        if stem in loaded_models:
            status = "[green]● loaded[/]"
        elif _is_healthy(port):
            status = "[dim]○ available[/]"
        else:
            status = "[dim]- server offline[/]"

        table.add_row(stem, size, status)

    console.print(table)


@llm.command("pull")
@click.option(
    "--model", "-m", type=str, default=None,
    help="Pull a specific model by config name (e.g., qwen3.5-4b)",
)
def llm_pull(model: str | None) -> None:
    """Download GGUF models from HuggingFace.

    Downloads models configured with `gguf:` field in magaldi.yaml.
    The gguf field format is: <repo_id>:<filename>

    \b
    Example config:
        llm:
          models:
            qwen3.5-4b:
              name: Qwen3.5-4B-Q4_K_M
              provider: llamacpp
              url: http://localhost:8090
              gguf: unsloth/Qwen3.5-4B-GGUF:Qwen3.5-4B-Q4_K_M.gguf
    """
    config = load_config()

    # Collect models with gguf field
    to_pull: list[tuple[str, ModelConfig]] = []
    for ref_name, cfg in config.llm.models.items():
        if cfg.provider == "llamacpp" and hasattr(cfg, "gguf") and cfg.gguf and (model is None or ref_name == model):
            to_pull.append((ref_name, cfg))

    if not to_pull:
        if model:
            console.print(f"[red]Model '{model}' not found or has no gguf field[/]")
        else:
            console.print("[yellow]No models with gguf field found in config.[/]")
        console.print()
        console.print("Add gguf field to your model config:")
        console.print('  gguf: "unsloth/Qwen3.5-4B-GGUF:Qwen3.5-4B-Q4_K_M.gguf"')
        return

    # Ensure models directory
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        console.print("[red]huggingface_hub not installed.[/]")
        console.print("  Run: pip install huggingface-hub")
        return

    console.print(f"[bold blue]Downloading {len(to_pull)} model(s) to {MODELS_DIR}[/]")
    console.print()

    for ref_name, cfg in to_pull:
        gguf_spec = cfg.gguf  # type: ignore[attr-defined]
        if ":" not in gguf_spec:
            console.print(f"  [red]{ref_name}: invalid gguf format '{gguf_spec}' (expected repo:filename)[/]")
            continue

        repo_id, filename = gguf_spec.rsplit(":", 1)

        # Check if already downloaded
        target = MODELS_DIR / filename
        if target.exists():
            console.print(f"  [dim]{filename} already exists ({_format_size(target.stat().st_size)})[/]")
            continue

        console.print(f"  Downloading {filename} from {repo_id}...")
        try:
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=str(MODELS_DIR),
            )
            if target.exists():
                console.print(f"  [green]  {filename} ({_format_size(target.stat().st_size)})[/]")
            else:
                console.print(f"  [green]  {filename} downloaded[/]")
        except Exception as e:
            console.print(f"  [red]  Failed: {rich_escape(str(e))}[/]")

    console.print()
    console.print("[green]Done.[/] Start with: [bold]magaldi llm serve[/]")
