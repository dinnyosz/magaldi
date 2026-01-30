"""Pattern-specific extractors.

This package contains extractors for:
- Web framework patterns (slim.py, laravel.py, express.py, etc.)
- CLI framework patterns (artisan.py, symfony_console.py, click.py, etc.)
- Library-specific patterns
- Design patterns

These extractors build on top of language extractors to detect
framework/library-specific code structures.
"""

from magaldi_core.extractors.patterns.slim import (
    extract_slim_routes,
    extract_slim_route_groups,
)

__all__ = [
    "extract_slim_routes",
    "extract_slim_route_groups",
]
