"""Python web framework route patterns.

Pattern definitions for detecting HTTP routes from decorator information
in Python web frameworks:
- FastAPI
- Flask
- Django REST Framework
- Starlette
- Litestar
"""

from __future__ import annotations

import re

# =============================================================================
# HTTP ROUTE PATTERNS
# =============================================================================

# Maps decorator name -> (HTTP method, framework)
# Use "*" for method when it needs to be extracted from args
PYTHON_HTTP_ROUTE_PATTERNS: dict[str, tuple[str, str]] = {
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
# FRAMEWORK-SPECIFIC EXTRACTION HELPERS
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
