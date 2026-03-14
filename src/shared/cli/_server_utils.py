"""Shared server management utilities for CLI backends.

Provides PID file management, health checking, log tailing, and
graceful shutdown for locally managed inference servers (llama.cpp,
vllm-mlx, etc.).
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import time
from pathlib import Path
from urllib.parse import urlparse

import requests
from rich.console import Console
from rich.markup import escape as rich_escape

# Default directory for PID and log files
PIDFILE_DIR = Path.home() / ".magaldi" / "pids"


def pidfile(prefix: str, port: int) -> Path:
    """Get the PID file path for a server.

    Args:
        prefix: Server type prefix (e.g., "llama-server", "vllm-mlx").
        port: Server port number.
    """
    return PIDFILE_DIR / f"{prefix}-{port}.pid"


def logfile(prefix: str, port: int) -> Path:
    """Get the log file path for a server.

    Args:
        prefix: Server type prefix (e.g., "llama-server", "vllm-mlx").
        port: Server port number.
    """
    return PIDFILE_DIR / f"{prefix}-{port}.log"


def get_pid(prefix: str, port: int) -> int | None:
    """Read PID from pidfile, return None if stale or missing.

    Automatically cleans up stale PID files where the process no longer exists.

    Args:
        prefix: Server type prefix.
        port: Server port number.

    Returns:
        Process ID if alive, None otherwise.
    """
    pf = pidfile(prefix, port)
    if pf.exists():
        try:
            pid = int(pf.read_text().strip())
            os.kill(pid, 0)  # Check if alive
            return pid
        except (ValueError, ProcessLookupError, PermissionError):
            pf.unlink(missing_ok=True)
    return None


def is_healthy(port: int, path: str = "/health") -> bool:
    """Check if a server is responding on the given port.

    Args:
        port: Server port number.
        path: Health check endpoint path.

    Returns:
        True if the server responds with HTTP 200.
    """
    try:
        r = requests.get(f"http://localhost:{port}{path}", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def stop_server(prefix: str, port: int, console: Console) -> None:
    """Stop a server by PID with graceful shutdown.

    Sends SIGTERM first, waits up to 5 seconds, then SIGKILL as fallback.
    Cleans up the PID file after stopping.

    Args:
        prefix: Server type prefix.
        port: Server port number.
        console: Rich console for output.
    """
    pid = get_pid(prefix, port)
    if pid is None:
        # Check for orphaned pidfiles
        if PIDFILE_DIR.exists():
            pattern = f"{prefix}-*.pid"
            pidfiles = list(PIDFILE_DIR.glob(pattern))
            if pidfiles:
                console.print("[yellow]Found orphaned PID files, cleaning up...[/]")
                for pf in pidfiles:
                    # Extract port from filename: prefix-PORT.pid
                    port_str = pf.stem.replace(f"{prefix}-", "")
                    try:
                        orphan_port = int(port_str)
                    except ValueError:
                        pf.unlink(missing_ok=True)
                        continue
                    orphan_pid = get_pid(prefix, orphan_port)
                    if orphan_pid:
                        _stop_pid(orphan_pid, prefix, orphan_port, console)
                    else:
                        pf.unlink(missing_ok=True)
                return
        console.print(f"[dim]No {prefix} running on port {port}[/]")
        return

    _stop_pid(pid, prefix, port, console)


def _stop_pid(pid: int, prefix: str, port: int, console: Console) -> None:
    """Stop a process by PID with graceful shutdown."""
    console.print(f"  Stopping {prefix} on port {port} (PID {pid})...", end=" ")
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

    pidfile(prefix, port).unlink(missing_ok=True)


def tail_log(
    prefix: str,
    port: int,
    console: Console,
    follow: bool = False,
    lines: int = 50,
) -> None:
    """Show or follow server log output.

    Args:
        prefix: Server type prefix.
        port: Server port number.
        console: Rich console for output.
        follow: If True, follow log output (like tail -f).
        lines: Number of lines to show.
    """
    log = logfile(prefix, port)

    if not log.exists():
        console.print(f"[red]No log file found for port {port}[/]")
        console.print(f"  Expected: {log}")
        return

    if follow:
        with contextlib.suppress(KeyboardInterrupt):
            subprocess.run(
                ["tail", "-f", "-n", str(lines), str(log)],
            )
    else:
        try:
            result = subprocess.run(
                ["tail", "-n", str(lines), str(log)],
                capture_output=True,
                text=True,
            )
            console.print(result.stdout, end="")
        except Exception as e:
            console.print(f"[red]Error reading log: {rich_escape(str(e))}[/]")


def format_size(size_bytes: int) -> str:
    """Format file size in human-readable form."""
    if size_bytes >= 1_073_741_824:
        return f"{size_bytes / 1_073_741_824:.1f} GB"
    elif size_bytes >= 1_048_576:
        return f"{size_bytes / 1_048_576:.0f} MB"
    else:
        return f"{size_bytes / 1024:.0f} KB"


def get_port_from_config(
    provider: str,
    models: dict,
    default_port: int,
) -> int:
    """Extract port from the first model config matching a provider.

    Args:
        provider: Provider name to match (e.g., "llamacpp", "vllm-mlx").
        models: Dict of model configs from LLMConfig.models.
        default_port: Fallback port if none found in config.

    Returns:
        Port number from the first matching model URL, or default_port.
    """
    for cfg in models.values():
        if cfg.provider == provider:
            parsed = urlparse(cfg.url)
            if parsed.port:
                return parsed.port
    return default_port


def wait_for_healthy(
    port: int,
    console: Console,
    proc: subprocess.Popen,
    timeout_seconds: int = 30,
    path: str = "/health",
) -> bool:
    """Wait for a server to become healthy.

    Args:
        port: Server port number.
        console: Rich console for output.
        proc: The server subprocess to monitor for early exit.
        timeout_seconds: Maximum seconds to wait.
        path: Health check endpoint path.

    Returns:
        True if server became healthy, False if timeout or process died.
    """
    console.print("  Waiting for server...", end=" ")
    for _ in range(timeout_seconds):
        time.sleep(1)
        if is_healthy(port, path):
            console.print("[green]Ready![/]")
            return True
        # Check if process died
        if proc.poll() is not None:
            console.print("[red]Process exited![/]")
            return False

    console.print("[yellow]Timeout (server may still be loading)[/]")
    return False
