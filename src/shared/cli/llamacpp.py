"""llama.cpp server commands for the Magaldi CLI.

Manages a llama-server (llama.cpp) instance in router mode, serving all GGUF
models from a shared directory with automatic hot-swapping and LRU eviction.

Usage:
    magaldi llamacpp serve     # Start llama-server in router mode
    magaldi llamacpp stop      # Stop the running server
    magaldi llamacpp status    # Show server + loaded models
    magaldi llamacpp logs      # Follow server logs
    magaldi llamacpp models    # List available GGUF models
    magaldi llamacpp pull      # Download configured models from HuggingFace
    magaldi llamacpp update    # Clone/update llama.cpp and rebuild

Example magaldi.yaml:
    llm:
      models:
        qwen3-4b:
          name: Qwen3-4B-Instruct-2507-Q4_K_M
          provider: llamacpp
          url: http://localhost:8090
          num_ctx: 32768
          parallel: 4
          gguf: unsloth/Qwen3-4B-Instruct-2507-GGUF:Qwen3-4B-Instruct-2507-Q4_K_M.gguf
        qwen3-1.7b:
          name: Qwen3-1.7B-Q4_K_M
          provider: llamacpp
          url: http://localhost:8090
          num_ctx: 32768
          parallel: 16
          gguf: unsloth/Qwen3-1.7B-GGUF:Qwen3-1.7B-Q4_K_M.gguf
        qwen3-embed:
          name: Qwen3-Embedding-0.6B-q8_0
          provider: llamacpp
          url: http://localhost:8090
          dimensions: 1024
          embedding: true
          num_ctx: 40960
          parallel: 3
          gguf: Mungert/Qwen3-Embedding-0.6B-GGUF:Qwen3-Embedding-0.6B-q8_0.gguf
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

# Defaults — tuned for Apple Silicon M4 Pro (48GB)
DEFAULT_PORT = 8090
DEFAULT_PARALLEL = 4
DEFAULT_MODELS_MAX = 2
DEFAULT_CTX_SIZE = 40960


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


def _generate_presets(llm_config: LLMConfig, create_symlinks: bool = True) -> Path:
    """Generate a llama-server presets INI file from config.

    The presets file lets us set per-model context sizes and other params.
    Format: [model:<model-id>] sections with key=value pairs.

    Per-model overrides (from ModelConfig fields):
      n_ctx = <num_ctx>   (explicit context size, avoids inheriting global default)
      parallel = <N>      (per-model parallel slots override)

    Embedding models get additional settings:
      embedding = true    (enables /v1/embeddings endpoint)
      pooling = mean      (average token embeddings into single vector)

    For non-embedding LLM models, also generates per-tier preset sections
    and creates symlinked GGUF files so each context tier gets its own
    model alias with an appropriately sized KV cache.
    """
    from shared.ai.context_size import CONTEXT_TIERS, TIER_PARALLEL
    from shared.ai.llamacpp_models import (
        ensure_all_tier_symlinks,
        get_tiered_model_name,
    )

    presets = configparser.ConfigParser()

    # Global defaults — lower priority than per-model sections,
    # but avoids passing these as CLI args (which override everything).
    presets["*"] = {
        "ctx-size": str(DEFAULT_CTX_SIZE),
        "parallel": str(DEFAULT_PARALLEL),
        # ubatch=2048 enables 4-stage pipeline parallelism for LLM models
        # (batch 8192 / ubatch 2048 = 4 stages across parallel slots).
        # Embedding models override this to 8192 (pooling needs full input).
        "ubatch-size": "2048",
    }

    for cfg in _get_llamacpp_models(llm_config):
        # Section name must match the model ID from --models-dir
        # (GGUF filename without .gguf extension)
        section = cfg.name
        presets[section] = {}
        if cfg.num_ctx:
            presets[section]["ctx-size"] = str(cfg.num_ctx)
        if cfg.parallel:
            presets[section]["parallel"] = str(cfg.parallel)
        if cfg.embedding:
            presets[section]["embedding"] = "true"
            presets[section]["pooling"] = "mean"
            # For embedding models with pooling, the entire input must fit in
            # a single ubatch (physical batch). Default ubatch-size (512) and
            # default batch-size (2048) are too small for large code files.
            # Both must be large enough: ubatch <= batch.
            # See: https://github.com/ggml-org/llama.cpp/issues/11105
            # We use embed_max_context (12288 = 12k) as the ceiling — that's the
            # actual max input size after truncation in the processing pipeline
            # (see ProcessingConfig.embed_max_context in processor/models.py).
            embed_batch = 12288
            presets[section]["batch-size"] = str(embed_batch)
            presets[section]["ubatch-size"] = str(embed_batch)
        else:
            # Non-embedding LLM model: generate per-tier preset sections.
            # Each tier gets a symlinked GGUF with its own ctx-size so the
            # KV cache is allocated per-tier (like Ollama tiered aliases).
            base_gguf = MODELS_DIR / f"{cfg.name}.gguf"
            if create_symlinks and base_gguf.exists():
                ensure_all_tier_symlinks(base_gguf, MODELS_DIR)

            for tier in CONTEXT_TIERS:
                tier_name = get_tiered_model_name(cfg.name, tier)
                presets[tier_name] = {}
                # Each tier gets its own ctx-size matching the tier value.
                # With unified KV cache, this is the total context budget
                # shared across all parallel slots for this tier alias.
                presets[tier_name]["ctx-size"] = str(tier)
                parallel_slots = TIER_PARALLEL.get(tier, DEFAULT_PARALLEL)
                presets[tier_name]["parallel"] = str(parallel_slots)

    presets_path = _presets_file()
    presets_path.parent.mkdir(parents=True, exist_ok=True)
    with open(presets_path, "w") as f:
        presets.write(f)

    return presets_path


def _list_gguf_files(include_tier_symlinks: bool = True) -> list[Path]:
    """List .gguf files in the models directory.

    Args:
        include_tier_symlinks: If False, excludes tier symlinks (e.g., Model-4k.gguf).
    """
    if not MODELS_DIR.exists():
        return []
    files = sorted(MODELS_DIR.glob("*.gguf"))
    if not include_tier_symlinks:
        from shared.ai.llamacpp_models import is_tier_symlink
        files = [f for f in files if not is_tier_symlink(f)]
    return files


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
def llamacpp() -> None:
    """Manage llama.cpp server."""
    pass


@llamacpp.command("serve")
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
    is reached. Use `magaldi llamacpp pull` to download models first.

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
        console.print("  Run: [bold]make llama-pull[/] or [bold]magaldi llamacpp pull[/]")
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
    # NOTE: --parallel and --ctx-size are set in the presets file (global [*]
    # section) instead of here. CLI args have highest priority in llama.cpp
    # and would override per-model preset values, preventing models from
    # having different parallel slot counts.
    cmd = [
        str(LLAMA_SERVER),
        "--models-dir", str(MODELS_DIR),
        "--port", str(port),
        "--host", "0.0.0.0",
        "--models-max", str(models_max),
        "--threads", "6",
        "--flash-attn", "on",
        "--n-gpu-layers", "99",
        # Batch size: n_batch is the logical max batch (prompt tokens per
        # iteration). n_ubatch is set in presets (not here!) because CLI args
        # override per-model presets — embedding models need ubatch=8192
        # (pooling requires entire input in one ubatch) while LLM models
        # benefit from ubatch=2048 (4-stage pipeline across parallel slots).
        "--batch-size", "8192",
        "-sps", "0",
        # KV cache: unified mode shares a single KV cache pool across all
        # parallel slots for each model. With per-tier ctx-size in presets,
        # each tier alias gets an appropriately sized shared cache instead
        # of allocating ctx-size/parallel per slot. This significantly
        # reduces memory waste for tiers with many parallel slots.
        # q8_0 KV cache halves memory vs f16 with negligible quality loss,
        # allowing more parallel slots or larger contexts in the same budget.
        "--kv-unified",
        "--cache-type-k", "q8_0",
        "--cache-type-v", "q8_0",
        "--cache-reuse", "256",
        # Disable thinking globally — Qwen3 thinks by default, wasting
        # 2-5× tokens on <think> reasoning before producing actual output.
        # --reasoning off: disables thinking in the chat template (the actual switch)
        # --reasoning-budget 0: safety net, immediately ends any thinking
        # --reasoning-format none: leaves any leaked think tags in content
        #   (where _strip_think_tags removes them) instead of splitting
        #   into reasoning_content
        "--reasoning", "off",
        "--reasoning-budget", "0",
        "--reasoning-format", "none",
        # Monitoring endpoints
        "--slots",       # expose /slots (on by default, but be explicit)
        "--metrics",     # enable /metrics (Prometheus-compatible)
        "--props",       # enable /props
    ]

    # Add presets if we generated any model-specific config
    if presets_path.stat().st_size > 0:
        cmd.extend(["--models-preset", str(presets_path)])

    # Ensure directories exist
    PIDFILE_DIR.mkdir(parents=True, exist_ok=True)

    # Print startup info
    base_gguf_files = _list_gguf_files(include_tier_symlinks=False)
    tier_symlink_count = len(gguf_files) - len(base_gguf_files)
    console.print("[bold blue]Starting llama-server (router mode)[/]")
    console.print()
    console.print(f"  Port:           {port}")
    console.print(f"  Models dir:     {MODELS_DIR}")
    console.print(f"  Models found:   {len(base_gguf_files)} (+{tier_symlink_count} tier aliases)")
    for gf in base_gguf_files:
        console.print(f"    - {gf.stem}  ({_format_size(gf.stat().st_size)})")
    console.print(f"  Max loaded:     {models_max}")
    console.print(f"  Parallel slots: {parallel} (default, per-model override via presets)")
    console.print(f"  Context size:   {ctx_size} (default, per-model override via presets)")
    console.print("  Flash attention: on")
    console.print("  GPU layers:     99 (full offload)")
    console.print("  KV unified:     on (shared cache pool per model)")
    console.print("  Batch size:     8192 (batch) / 2048 (ubatch, per-model override via presets)")
    console.print("  KV cache type:  q8_0 (K + V)")
    console.print("  Cache reuse:    256 tokens")
    console.print("  Endpoints:      slots, metrics, props")
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
            console.print(f"  API:     http://localhost:{port}/v1/models")
            console.print(f"  Health:  http://localhost:{port}/health")
            console.print(f"  Slots:   http://localhost:{port}/slots")
            console.print(f"  Metrics: http://localhost:{port}/metrics")
            console.print(f"  Props:   http://localhost:{port}/props")
            return
        # Check if process died
        if proc.poll() is not None:
            console.print("[red]Process exited![/]")
            console.print(f"  Check log: {logfile}")
            _pidfile(port).unlink(missing_ok=True)
            return

    console.print("[yellow]Timeout (server may still be loading)[/]")
    console.print(f"  Check: curl http://localhost:{port}/health")


@llamacpp.command("stop")
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


@llamacpp.command("status")
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
        console.print("  Run: [bold]magaldi llamacpp serve[/]")
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


@llamacpp.command("logs")
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


@llamacpp.command("models")
@click.option(
    "--all", "-a", "show_all", is_flag=True,
    help="Include tier symlinks in listing",
)
def llm_models(show_all: bool) -> None:
    """List available GGUF models in tools/models/."""
    from shared.ai.llamacpp_models import is_tier_symlink

    base_files = _list_gguf_files(include_tier_symlinks=False)
    all_files = _list_gguf_files(include_tier_symlinks=True)
    tier_count = len(all_files) - len(base_files)

    if not all_files:
        console.print("[yellow]No GGUF models found.[/]")
        console.print(f"  Directory: {MODELS_DIR}")
        console.print("  Run: [bold]magaldi llamacpp pull[/] or [bold]make llama-pull[/]")
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

    display_files = all_files if show_all else base_files

    table = Table(title=f"GGUF Models ({MODELS_DIR})")
    table.add_column("Model", style="cyan")
    table.add_column("Size", justify="right")
    table.add_column("Type")
    table.add_column("Status")

    for gf in display_files:
        stem = gf.stem
        size = _format_size(gf.stat().st_size) if not gf.is_symlink() else "[dim]symlink[/]"
        model_type = "[dim]tier alias[/]" if is_tier_symlink(gf) else "base"
        if stem in loaded_models:
            status = "[green]● loaded[/]"
        elif _is_healthy(port):
            status = "[dim]○ available[/]"
        else:
            status = "[dim]- server offline[/]"

        table.add_row(stem, size, model_type, status)

    console.print(table)

    if not show_all and tier_count > 0:
        console.print(f"\n  [dim]{tier_count} tier aliases hidden. Use --all to show.[/]")


@llamacpp.command("pull")
@click.option(
    "--model", "-m", type=str, default=None,
    help="Pull a specific model by config name (e.g., qwen3-4b)",
)
def llm_pull(model: str | None) -> None:
    """Download GGUF models from HuggingFace.

    Downloads models configured with `gguf:` field in magaldi.yaml.
    The gguf field format is: <repo_id>:<filename>

    \b
    Example config:
        llm:
          models:
            qwen3-4b:
              name: Qwen3-4B-Instruct-2507-Q4_K_M
              provider: llamacpp
              url: http://localhost:8090
              gguf: unsloth/Qwen3-4B-Instruct-2507-GGUF:Qwen3-4B-Instruct-2507-Q4_K_M.gguf
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
    console.print("[green]Done.[/] Start with: [bold]magaldi llamacpp serve[/]")


@llamacpp.command("warmup")
def llm_warmup() -> None:
    """Create tiered model symlinks for context optimization.

    Creates symlinked GGUF files with tier suffixes (e.g., Model-2k.gguf,
    Model-4k.gguf) and generates matching preset sections. This enables
    per-tier KV cache allocation — small functions use less memory than
    large files.

    \b
    The symlinks share weights via mmap (no extra disk/memory), only the
    KV cache differs. Run before `magaldi llamacpp serve` or let serve
    create them automatically.
    """
    from shared.ai.llamacpp_models import ensure_all_tier_symlinks

    config = load_config()
    models = _get_llamacpp_models(config.llm)

    if not models:
        console.print("[yellow]No llamacpp models found in config.[/]")
        return

    console.print("[bold blue]Creating tiered model symlinks[/]")
    console.print()

    for cfg in models:
        if cfg.embedding:
            console.print(f"  {cfg.name}: [dim]skipped (embedding model)[/]")
            continue

        base_gguf = MODELS_DIR / f"{cfg.name}.gguf"
        if not base_gguf.exists():
            console.print(f"  {cfg.name}: [red]GGUF not found[/] ({base_gguf})")
            continue

        result = ensure_all_tier_symlinks(base_gguf, MODELS_DIR)

        console.print(f"  {cfg.name}:")
        if result.total_created:
            console.print(f"    Created: {result.total_created} symlinks")
        if result.existed:
            console.print(f"    Existed: {len(result.existed)} (already present)")
        if result.failed:
            console.print(f"    [red]Failed: {len(result.failed)}[/]")
        console.print()

    # Regenerate presets with tier sections
    presets_path = _generate_presets(config.llm, create_symlinks=False)
    console.print(f"  Presets: {presets_path}")
    console.print()
    console.print("[green]Warmup complete.[/]")

    # Hint about restart if server is running
    port = _get_llamacpp_port(config.llm)
    if _get_pid(port) is not None:
        console.print()
        console.print("[yellow]Note:[/] Server is running. Restart to use tiered models:")
        console.print("  [bold]magaldi llamacpp stop && magaldi llamacpp serve[/]")


@llamacpp.command("update")
@click.option(
    "--jobs", "-j", type=int, default=None,
    help="Parallel build jobs (default: number of CPU cores)",
)
def llm_update(jobs: int | None) -> None:
    """Update llama.cpp to the latest version and rebuild.

    Clones the llama.cpp repository if it doesn't exist, or pulls the
    latest changes from master. Then rebuilds llama-server with Metal
    support (Apple Silicon GPU offload).

    \b
    What this does:
        1. Clone or update llama.cpp in tools/llama.cpp/
        2. Build with cmake (Metal + Release)
        3. Verify llama-server binary exists

    The server must be restarted after updating to use the new binary.
    """
    llama_dir = PROJECT_ROOT / "tools" / "llama.cpp"

    # Step 1: Clone or pull
    if not llama_dir.exists():
        console.print("[bold blue]Cloning llama.cpp...[/]")
        result = subprocess.run(
            ["git", "clone", "https://github.com/ggml-org/llama.cpp.git", str(llama_dir)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            console.print(f"[red]Clone failed:[/]\n{result.stderr.strip()}")
            return
        console.print("[green]  Cloned successfully[/]")
    else:
        console.print("[bold blue]Updating llama.cpp...[/]")

        # Fetch and show what changed
        fetch_result = subprocess.run(
            ["git", "-C", str(llama_dir), "fetch", "origin", "master"],
            capture_output=True,
            text=True,
        )
        if fetch_result.returncode != 0:
            console.print(f"[red]Fetch failed:[/]\n{fetch_result.stderr.strip()}")
            return

        # Check current vs remote
        local_sha = subprocess.run(
            ["git", "-C", str(llama_dir), "rev-parse", "HEAD"],
            capture_output=True, text=True,
        ).stdout.strip()
        remote_sha = subprocess.run(
            ["git", "-C", str(llama_dir), "rev-parse", "origin/master"],
            capture_output=True, text=True,
        ).stdout.strip()

        if local_sha == remote_sha:
            console.print("[green]  Already up to date[/]")
            # Still rebuild if binary is missing
            if LLAMA_SERVER.exists():
                console.print(f"  Binary: {LLAMA_SERVER}")
                return
            console.print("[yellow]  Binary missing, rebuilding...[/]")
        else:
            # Show short log of incoming changes
            log_result = subprocess.run(
                ["git", "-C", str(llama_dir), "log", "--oneline",
                 f"{local_sha}..{remote_sha}", "--max-count=50"],
                capture_output=True, text=True,
            )
            if log_result.stdout.strip():
                console.print("  Latest commits:")
                for line in log_result.stdout.strip().splitlines():
                    console.print(f"    {line}")

            # Pull
            pull_result = subprocess.run(
                ["git", "-C", str(llama_dir), "pull", "origin", "master"],
                capture_output=True, text=True,
            )
            if pull_result.returncode != 0:
                console.print(f"[red]Pull failed:[/]\n{pull_result.stderr.strip()}")
                return
            console.print("[green]  Updated successfully[/]")

    console.print()

    # Step 2: Build
    # Detect CPU count for parallel build
    if jobs is None:
        try:
            result = subprocess.run(
                ["sysctl", "-n", "hw.logicalcpu"],
                capture_output=True, text=True,
            )
            jobs = int(result.stdout.strip())
        except (ValueError, FileNotFoundError):
            jobs = os.cpu_count() or 4

    console.print(f"[bold blue]Building llama.cpp (Metal, {jobs} jobs)...[/]")

    # cmake configure
    cmake_configure = subprocess.run(
        ["cmake", "-B", "build", "-DGGML_METAL=ON", "-DLLAMA_CURL=OFF"],
        cwd=str(llama_dir),
        capture_output=True,
        text=True,
    )
    if cmake_configure.returncode != 0:
        console.print("[red]cmake configure failed:[/]")
        # Show last few lines of stderr
        for line in cmake_configure.stderr.strip().splitlines()[-10:]:
            console.print(f"  {line}")
        return

    # cmake build
    cmake_build = subprocess.run(
        ["cmake", "--build", "build", "--config", "Release", f"-j{jobs}"],
        cwd=str(llama_dir),
        capture_output=True,
        text=True,
    )
    if cmake_build.returncode != 0:
        console.print("[red]Build failed:[/]")
        for line in cmake_build.stderr.strip().splitlines()[-10:]:
            console.print(f"  {line}")
        return

    console.print("[green]  Build complete[/]")
    console.print()

    # Step 3: Verify
    if LLAMA_SERVER.exists():
        console.print(f"[green]✓[/] llama-server: {LLAMA_SERVER}")
    else:
        console.print("[red]✗ llama-server binary not found after build[/]")
        console.print(f"  Expected: {LLAMA_SERVER}")
        return

    # Hint about restart if server is running
    config = load_config()
    port = _get_llamacpp_port(config.llm)
    if _get_pid(port) is not None:
        console.print()
        console.print("[yellow]Note:[/] Server is running. Restart to use the new binary:")
        console.print("  [bold]magaldi llamacpp stop && magaldi llamacpp serve[/]")


# =============================================================================
# DEPRECATED ALIAS: magaldi llm → magaldi llamacpp
# =============================================================================


@main.group(hidden=True, deprecated=True)
def llm() -> None:
    """[Deprecated] Use 'magaldi llamacpp' instead."""
    pass


for _name, _cmd in llamacpp.commands.items():
    llm.add_command(_cmd, _name)
