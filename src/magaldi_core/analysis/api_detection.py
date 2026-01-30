"""API surface detection functions.

This module re-exports detection functions from magaldi_core.extractors.detection
for backwards compatibility. New code should import directly from:
    magaldi_core.extractors.detection

This module provides functions for detecting:
- HTTP routes (FastAPI, Flask, Express, NestJS)
- CLI commands (Click, Typer)
- Public API elements
- Design patterns (Singleton, Builder, Factory, Repository)
"""

# Re-export everything from the new location for backwards compatibility
from magaldi_core.extractors.detection import (
    detect_http_routes,
    detect_cli_commands,
    detect_public_api,
    detect_patterns,
)

__all__ = [
    "detect_http_routes",
    "detect_cli_commands",
    "detect_public_api",
    "detect_patterns",
]
