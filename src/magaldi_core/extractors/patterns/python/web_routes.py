"""Python web framework route patterns and detection.

Pattern definitions and detection functions for HTTP routes
in Python web frameworks:
- FastAPI
- Flask
- Django REST Framework
- Starlette
- Litestar
"""

from __future__ import annotations

import re

from magaldi_core.extractors.types import DecoratorInfo, HttpRoute

# =============================================================================
# HTTP ROUTE PATTERNS
# =============================================================================

# Maps decorator name -> (HTTP method, framework)
# Use "*" for method when it needs to be extracted from args
_PYTHON_HTTP_ROUTE_PATTERNS: dict[str, tuple[str, str]] = {
    # FastAPI patterns
    "router.get": ("GET", "fastapi"),
    "router.post": ("POST", "fastapi"),
    "router.put": ("PUT", "fastapi"),
    "router.delete": ("DELETE", "fastapi"),
    "router.patch": ("PATCH", "fastapi"),
    "router.head": ("HEAD", "fastapi"),
    "router.options": ("OPTIONS", "fastapi"),
    "app.get": ("GET", "fastapi"),
    "app.post": ("POST", "fastapi"),
    "app.put": ("PUT", "fastapi"),
    "app.delete": ("DELETE", "fastapi"),
    "app.patch": ("PATCH", "fastapi"),
    # Flask patterns
    "app.route": ("*", "flask"),
    "blueprint.route": ("*", "flask"),
    "bp.route": ("*", "flask"),
    # Django REST Framework patterns
    "api_view": ("*", "django-rest"),
    "action": ("*", "django-rest"),
    # Starlette patterns (same syntax as FastAPI)
    "route": ("*", "starlette"),
    # Litestar patterns
    "get": ("GET", "litestar"),
    "post": ("POST", "litestar"),
    "put": ("PUT", "litestar"),
    "delete": ("DELETE", "litestar"),
    "patch": ("PATCH", "litestar"),
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def extract_method_from_flask_args(args: str | None) -> str:
    """Extract HTTP method from Flask route arguments.

    Handles parenthesized argument lists:
    - 'methods=["POST"]' -> 'POST'
    - '("/users", methods=["POST"])' -> 'POST'

    Args:
        args: Decorator arguments string.

    Returns:
        HTTP method string (defaults to "GET").
    """
    if not args:
        return "GET"

    # Strip outer parentheses if present
    cleaned = args.strip()
    if cleaned.startswith("(") and cleaned.endswith(")"):
        cleaned = cleaned[1:-1]

    match = re.search(r'methods\s*=\s*\[([^\]]+)\]', cleaned)
    if match:
        methods_str = match.group(1)
        method_match = re.search(r'["\'](\w+)["\']', methods_str)
        if method_match:
            return method_match.group(1).upper()
    return "GET"


def extract_method_from_drf_args(args: str | None) -> str:
    """Extract HTTP method from Django REST Framework @api_view args.

    Handles:
    - '(["GET"])' -> 'GET'
    - '(["GET", "POST"])' -> 'GET'
    - '["GET"]' -> 'GET'

    Args:
        args: Decorator arguments string.

    Returns:
        HTTP method string (defaults to "GET").
    """
    if not args:
        return "GET"

    # Strip outer parentheses if present
    cleaned = args.strip()
    if cleaned.startswith("(") and cleaned.endswith(")"):
        cleaned = cleaned[1:-1].strip()

    # Match a list of methods as the first argument
    match = re.match(r'\[([^\]]+)\]', cleaned)
    if match:
        methods_str = match.group(1)
        method_match = re.search(r'["\'](\w+)["\']', methods_str)
        if method_match:
            return method_match.group(1).upper()
    return "GET"


def _extract_path_params(path: str) -> list[str]:
    """Extract path parameters from a route path."""
    params = []
    # FastAPI/OpenAPI style: {param}
    for match in re.finditer(r"\{(\w+)\}", path):
        params.append(match.group(1))
    # Flask style: <param>
    for match in re.finditer(r"<(\w+)>", path):
        params.append(match.group(1))
    return params


def _extract_path_from_args(args: str | None) -> str | None:
    """Extract path from decorator arguments.

    Handles both raw strings and parenthesized argument lists:
    - '"/users"' -> '/users'
    - '("/users")' -> '/users'
    - '("/users", methods=["GET"])' -> '/users'
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


def detect_python_http_routes(
    decorators: list[DecoratorInfo],
) -> list[HttpRoute]:
    """Detect HTTP routes from Python decorator information.

    Args:
        decorators: List of decorator information extracted from a function.

    Returns:
        List of detected HTTP routes.
    """
    routes = []

    for dec in decorators:
        if dec.name in _PYTHON_HTTP_ROUTE_PATTERNS:
            method, framework = _PYTHON_HTTP_ROUTE_PATTERNS[dec.name]
            path = _extract_path_from_args(dec.args)

            # Handle method extraction based on framework
            if method == "*":
                if framework == "django-rest":
                    method = extract_method_from_drf_args(dec.args)
                else:
                    method = extract_method_from_flask_args(dec.args)

            # For DRF, path may not be in decorator (comes from urls.py)
            # Still record the route with placeholder path
            if framework == "django-rest" and not path:
                path = "<url-pattern>"  # Placeholder for DRF routes

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
