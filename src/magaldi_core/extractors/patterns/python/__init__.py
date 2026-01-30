"""Python framework pattern extractors.

This package contains pattern definitions for Python web and CLI frameworks:
- web_routes.py - FastAPI, Flask, Django REST, Starlette, Litestar
- cli_commands.py - Click, Typer
"""

from magaldi_core.extractors.patterns.python.web_routes import (
    PYTHON_HTTP_ROUTE_PATTERNS,
    extract_method_from_flask_args,
    extract_method_from_drf_args,
)
from magaldi_core.extractors.patterns.python.cli_commands import (
    PYTHON_CLI_COMMAND_PATTERNS,
    PYTHON_CLI_COMMAND_SUFFIXES,
)

__all__ = [
    # Web routes
    "PYTHON_HTTP_ROUTE_PATTERNS",
    "extract_method_from_flask_args",
    "extract_method_from_drf_args",
    # CLI commands
    "PYTHON_CLI_COMMAND_PATTERNS",
    "PYTHON_CLI_COMMAND_SUFFIXES",
]
