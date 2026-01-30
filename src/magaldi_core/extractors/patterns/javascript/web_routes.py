"""JavaScript/TypeScript web framework route patterns and detection.

Pattern definitions and detection functions for HTTP routes
in JS/TS web frameworks:
- NestJS
- Hono
- Express
"""

from __future__ import annotations

import re

from magaldi_core.extractors.types import DecoratorInfo, HttpRoute

# =============================================================================
# HTTP ROUTE PATTERNS
# =============================================================================

# Maps decorator name -> (HTTP method, framework)
# Use "*" for method when it needs to be extracted from args
_JAVASCRIPT_HTTP_ROUTE_PATTERNS: dict[str, tuple[str, str]] = {
    # NestJS patterns (decorators)
    "Get": ("GET", "nestjs"),
    "Post": ("POST", "nestjs"),
    "Put": ("PUT", "nestjs"),
    "Delete": ("DELETE", "nestjs"),
    "Patch": ("PATCH", "nestjs"),
    "Head": ("HEAD", "nestjs"),
    "Options": ("OPTIONS", "nestjs"),
    "All": ("*", "nestjs"),
    # Hono patterns (method calls, similar to Express)
    "c.get": ("GET", "hono"),
    "c.post": ("POST", "hono"),
    "c.put": ("PUT", "hono"),
    "c.delete": ("DELETE", "hono"),
    "c.patch": ("PATCH", "hono"),
    "app.get": ("GET", "hono"),
    "app.post": ("POST", "hono"),
    "app.put": ("PUT", "hono"),
    "app.delete": ("DELETE", "hono"),
    "app.patch": ("PATCH", "hono"),
    # Express patterns (method calls)
    # Note: Express uses method calls like app.get(), router.get()
    # These overlap with Hono but work the same way
    "router.get": ("GET", "express"),
    "router.post": ("POST", "express"),
    "router.put": ("PUT", "express"),
    "router.delete": ("DELETE", "express"),
    "router.patch": ("PATCH", "express"),
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _extract_path_params(path: str) -> list[str]:
    """Extract path parameters from a route path."""
    params = []
    # Express/NestJS style: :param
    for match in re.finditer(r":(\w+)", path):
        params.append(match.group(1))
    # OpenAPI style: {param} (used by some frameworks)
    for match in re.finditer(r"\{(\w+)\}", path):
        params.append(match.group(1))
    return params


def _extract_path_from_args(args: str | None) -> str | None:
    """Extract path from decorator arguments.

    Handles both raw strings and parenthesized argument lists:
    - '"/users"' -> '/users'
    - '("/users")' -> '/users'
    """
    if not args:
        return None

    # Strip whitespace and outer parentheses if present
    cleaned = args.strip()
    if cleaned.startswith("(") and cleaned.endswith(")"):
        cleaned = cleaned[1:-1].strip()

    # Extract first quoted string (the path)
    match = re.match(r'["\']([^"\']+)["\']', cleaned)
    if match:
        return match.group(1)
    return None


# =============================================================================
# DETECTION FUNCTION
# =============================================================================


def detect_javascript_http_routes(
    decorators: list[DecoratorInfo],
) -> list[HttpRoute]:
    """Detect HTTP routes from JavaScript/TypeScript decorator information.

    Args:
        decorators: List of decorator information extracted from a function.

    Returns:
        List of detected HTTP routes.
    """
    routes = []

    for dec in decorators:
        if dec.name in _JAVASCRIPT_HTTP_ROUTE_PATTERNS:
            method, framework = _JAVASCRIPT_HTTP_ROUTE_PATTERNS[dec.name]
            path = _extract_path_from_args(dec.args)

            # NestJS @All() matches all HTTP methods
            if method == "*":
                method = "ALL"

            if path:
                routes.append(
                    HttpRoute(
                        method=method,
                        path=path,
                        path_params=_extract_path_params(path),
                        framework=framework,
                    )
                )

    return routes
