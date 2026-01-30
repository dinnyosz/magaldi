"""PHP framework pattern extractors.

This package contains extractors for PHP web frameworks:
- slim.py - Slim framework routes and middleware
- (future) laravel.py - Laravel routes, controllers, middleware
- (future) symfony.py - Symfony routes and controllers
"""

from magaldi_core.extractors.patterns.php.slim import (
    extract_slim_routes,
    extract_slim_route_groups,
    SlimRouteGroup,
)

__all__ = [
    "extract_slim_routes",
    "extract_slim_route_groups",
    "SlimRouteGroup",
]
