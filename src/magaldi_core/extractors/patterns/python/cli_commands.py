"""Python CLI framework command patterns.

Pattern definitions for detecting CLI commands from decorator information
in Python CLI frameworks:
- Click
- Typer
"""

from __future__ import annotations

# =============================================================================
# CLI COMMAND PATTERNS
# =============================================================================

# CLI command decorator patterns (exact matches)
# Maps decorator name -> framework
PYTHON_CLI_COMMAND_PATTERNS: dict[str, str] = {
    "click.command": "click",
    "click.group": "click",
    "app.command": "typer",
    "typer.command": "typer",
}

# CLI decorator suffixes that indicate a command (for patterns like main.command, web.group)
PYTHON_CLI_COMMAND_SUFFIXES: set[str] = {".command", ".group"}
