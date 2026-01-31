"""PHP framework pattern extractors.

This package contains extractors for PHP web frameworks:
- slim.py - Slim framework routes and middleware
- laravel.py - Laravel routes, controllers, middleware
- symfony.py - Symfony routes and controllers
- cli_commands.py - Symfony Console and Laravel Artisan commands
"""

from magaldi_core.extractors.patterns.php.slim import (
    extract_slim_routes,
    extract_slim_route_groups,
    SlimRouteGroup,
)
from magaldi_core.extractors.patterns.php.laravel import (
    extract_laravel_routes,
    extract_laravel_route_groups,
    LaravelRouteGroup,
)
from magaldi_core.extractors.patterns.php.symfony import (
    extract_symfony_routes,
    extract_symfony_controllers,
    SymfonyController,
)
from magaldi_core.extractors.patterns.php.cli_commands import (
    extract_php_cli_commands,
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
