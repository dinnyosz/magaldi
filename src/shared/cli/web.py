"""Web commands for the Magaldi CLI.

This module contains the web command group for running the web UI server.
"""

from __future__ import annotations

import click

from shared.cli._shared import console, main
from shared.config import load_config


@main.group()
def web() -> None:
    """Web UI commands."""
    pass


@web.command("serve")
@click.option("--host", "-h", default=None, help="Host to bind to (default: from config)")
@click.option("--port", "-p", default=None, type=int, help="Port to bind to (default: from config)")
@click.option("--reload", "-r", is_flag=True, help="Enable auto-reload for development")
def web_serve(host: str | None, port: int | None, reload: bool) -> None:
    """Start the web server."""
    from magaldi_web.app import run_server

    config = load_config()
    host = host or config.web.host
    port = port or config.web.port

    console.print(f"[bold blue]Starting Magaldi Web UI[/]")
    console.print(f"  URL: http://{host}:{port}")
    console.print(f"  Auto-reload: {'enabled' if reload else 'disabled'}")
    console.print()

    run_server(host=host, port=port, reload=reload)
