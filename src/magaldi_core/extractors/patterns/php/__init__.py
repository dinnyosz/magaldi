"""PHP framework pattern extractors.

This package contains extractors for PHP web frameworks:
- slim.py - Slim framework routes and middleware
- laravel.py - Laravel routes, controllers, middleware
- symfony.py - Symfony routes and controllers
- cli_commands.py - Symfony Console and Laravel Artisan commands
"""

from magaldi_core.extractors.patterns.php.cli_commands import (
    extract_php_cli_commands,
)
from magaldi_core.extractors.patterns.php.laravel import (
    LaravelRouteGroup,
    extract_laravel_route_groups,
    extract_laravel_routes,
)
from magaldi_core.extractors.patterns.php.slim import (
    SlimRouteGroup,
    extract_slim_route_groups,
    extract_slim_routes,
)
from magaldi_core.extractors.patterns.php.symfony import (
    SymfonyController,
    extract_symfony_controllers,
    extract_symfony_routes,
)

__all__ = [
    # Slim
    "extract_slim_routes",
    "extract_slim_route_groups",
    "SlimRouteGroup",
    # Laravel
    "extract_laravel_routes",
    "extract_laravel_route_groups",
    "LaravelRouteGroup",
    # Symfony
    "extract_symfony_routes",
    "extract_symfony_controllers",
    "SymfonyController",
    # CLI Commands
    "extract_php_cli_commands",
]
