"""Python framework pattern extractors.

This package contains detection functions for Python web and CLI frameworks:
- web_routes.py - FastAPI, Flask, Django REST, Starlette, Litestar
- cli_commands.py - Click, Typer
"""

from magaldi_core.extractors.patterns.python.web_routes import (
    detect_python_http_routes,
)
from magaldi_core.extractors.patterns.python.cli_commands import (
    detect_python_cli_commands,
)

__all__ = [
    "detect_python_http_routes",
    "detect_python_cli_commands",
]
