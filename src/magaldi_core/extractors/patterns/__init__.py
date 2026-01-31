"""Pattern-specific extractors organized by language.

This package contains framework/library pattern extractors:

patterns/
  php/          - PHP frameworks (Slim, Laravel, Symfony)
  python/       - Python frameworks (FastAPI, Flask, Django, Click, Typer)
  javascript/   - JS/TS frameworks (Express, NestJS, Hono)
  rust/         - Rust frameworks (Actix-web, Rocket)

These extractors build on top of language extractors to detect
framework/library-specific code structures like routes and CLI commands.
"""

# PHP patterns (AST-level extraction)
from magaldi_core.extractors.patterns.php import (
    extract_slim_routes,
    extract_slim_route_groups,
)

# Python patterns (decorator-based detection)
from magaldi_core.extractors.patterns.python import (
    detect_python_http_routes,
    detect_python_cli_commands,
)

# JavaScript/TypeScript patterns (decorator-based detection)
from magaldi_core.extractors.patterns.javascript import (
    detect_javascript_http_routes,
)

# Rust patterns (attribute-based detection)
from magaldi_core.extractors.patterns.rust import (
    detect_rust_http_routes,
)

__all__ = [
    # PHP
    "extract_slim_routes",
    "extract_slim_route_groups",
    # Python
    "detect_python_http_routes",
    "detect_python_cli_commands",
    # JavaScript/TypeScript
    "detect_javascript_http_routes",
    # Rust
    "detect_rust_http_routes",
]
