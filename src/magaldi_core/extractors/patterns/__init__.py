"""Pattern-specific extractors organized by language.

This package contains framework/library pattern extractors:

patterns/
  php/          - PHP frameworks (Slim, Laravel, Symfony)
  python/       - Python frameworks (FastAPI, Flask, Django, Click, Typer)
  javascript/   - JS/TS frameworks (Express, NestJS, Hono)
  rust/         - Rust frameworks (Actix, Axum, Rocket)

These extractors build on top of language extractors to detect
framework/library-specific code structures like routes and CLI commands.
"""

# PHP patterns
from magaldi_core.extractors.patterns.php import (
    extract_slim_routes,
    extract_slim_route_groups,
)

# Python patterns
from magaldi_core.extractors.patterns.python import (
    PYTHON_HTTP_ROUTE_PATTERNS,
    PYTHON_CLI_COMMAND_PATTERNS,
    PYTHON_CLI_COMMAND_SUFFIXES,
    extract_method_from_flask_args,
    extract_method_from_drf_args,
)

# JavaScript/TypeScript patterns
from magaldi_core.extractors.patterns.javascript import (
    JS_HTTP_ROUTE_PATTERNS,
)

__all__ = [
    # PHP
    "extract_slim_routes",
    "extract_slim_route_groups",
    # Python
    "PYTHON_HTTP_ROUTE_PATTERNS",
    "PYTHON_CLI_COMMAND_PATTERNS",
    "PYTHON_CLI_COMMAND_SUFFIXES",
    "extract_method_from_flask_args",
    "extract_method_from_drf_args",
    # JavaScript/TypeScript
    "JS_HTTP_ROUTE_PATTERNS",
]
