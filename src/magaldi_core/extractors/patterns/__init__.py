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

__all__ = [
    # PHP
    "extract_slim_routes",
    "extract_slim_route_groups",
]
