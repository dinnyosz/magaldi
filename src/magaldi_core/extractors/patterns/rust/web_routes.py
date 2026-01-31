"""Rust web framework route patterns and detection.

Pattern definitions and detection functions for HTTP routes
in Rust web frameworks:
- Actix-web (#[get("/")], #[post("/")])
- Rocket (#[get("/")], #[post("/")], #[launch])

Also detects entry point macros:
- #[tokio::main], #[actix_web::main] - async main
- #[test], #[tokio::test] - test functions
"""

from __future__ import annotations

import re

from magaldi_core.extractors.types import DecoratorInfo, HttpRoute


# =============================================================================
# HTTP ROUTE PATTERNS
# =============================================================================

# Maps attribute name -> (HTTP method, framework)
# Use "WEBSOCKET" for WebSocket endpoints
_RUST_HTTP_ROUTE_PATTERNS: dict[str, tuple[str, str]] = {
    # Actix-web HTTP patterns
    "get": ("GET", "actix-web"),
    "post": ("POST", "actix-web"),
    "put": ("PUT", "actix-web"),
    "delete": ("DELETE", "actix-web"),
    "patch": ("PATCH", "actix-web"),
    "head": ("HEAD", "actix-web"),
    "options": ("OPTIONS", "actix-web"),
    "trace": ("TRACE", "actix-web"),
    "connect": ("CONNECT", "actix-web"),
    # Rocket HTTP patterns (same attribute names as Actix)
    # Detection framework will be refined based on imports
    "route": ("*", "rocket"),  # Generic route with method in args
    # WebSocket patterns
    "websocket": ("WEBSOCKET", "actix-web"),
}

# Entry point/runtime macros
_RUST_ENTRY_POINT_PATTERNS: dict[str, str] = {
    # Actix-web entry point
    "actix_web::main": "actix-web",
    "actix_rt::main": "actix-web",
    # Rocket entry point
    "launch": "rocket",
    "rocket::launch": "rocket",
    # Tokio runtime
    "tokio::main": "tokio",
    # Test patterns
    "test": "test",
    "tokio::test": "tokio-test",
    "actix_rt::test": "actix-test",
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _extract_path_params(path: str) -> list[str]:
    """Extract path parameters from a Rust route path.

    Rust frameworks use different styles:
    - Actix-web: {param} or {param:regex}
    - Rocket: <param> or <param..>

    Args:
        path: Route path string.

    Returns:
        List of parameter names.
    """
    params = []
    # Actix-web style: {param}
    for match in re.finditer(r"\{(\w+)(?::[^}]*)?\}", path):
        params.append(match.group(1))
    # Rocket style: <param>
    for match in re.finditer(r"<(\w+)(?:\.\.)?(?::[^>]*)?>", path):
        params.append(match.group(1))
    return params


def _extract_path_from_args(args: str | None) -> str | None:
    """Extract path from attribute arguments.

    Handles:
    - '"/users"' -> '/users'
    - '("/users")' -> '/users'
    - '("/users", method = "POST")' -> '/users'

    Args:
        args: Decorator arguments string.

    Returns:
        Path string if found, None otherwise.
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


def _extract_method_from_route_args(args: str | None) -> str:
    """Extract HTTP method from #[route] attribute arguments.

    Handles: method = "POST" or method = POST

    Args:
        args: Attribute arguments string.

    Returns:
        HTTP method string (defaults to "GET").
    """
    if not args:
        return "GET"

    # Look for method = "METHOD" or method = METHOD
    match = re.search(r'method\s*=\s*["\']?(\w+)["\']?', args)
    if match:
        return match.group(1).upper()
    return "GET"


# =============================================================================
# DETECTION FUNCTION
# =============================================================================


def detect_rust_http_routes(
    decorators: list[DecoratorInfo],
) -> list[HttpRoute]:
    """Detect HTTP routes from Rust attribute information.

    Args:
        decorators: List of decorator/attribute information extracted from a function.

    Returns:
        List of detected HTTP routes.
    """
    routes = []

    for dec in decorators:
        # Normalize decorator name (remove crate prefixes like actix_web::)
        attr_name = dec.name.split("::")[-1].lower()

        if attr_name in _RUST_HTTP_ROUTE_PATTERNS:
            method, framework = _RUST_HTTP_ROUTE_PATTERNS[attr_name]
            path = _extract_path_from_args(dec.args)

            # Handle generic #[route] attribute
            if method == "*":
                method = _extract_method_from_route_args(dec.args)

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


def is_rust_entry_point(decorators: list[DecoratorInfo]) -> tuple[bool, str | None]:
    """Check if a function is a Rust entry point based on its attributes.

    Args:
        decorators: List of decorator/attribute information.

    Returns:
        Tuple of (is_entry_point, framework_name).
    """
    for dec in decorators:
        # Check full attribute name
        if dec.name in _RUST_ENTRY_POINT_PATTERNS:
            return True, _RUST_ENTRY_POINT_PATTERNS[dec.name]

        # Check without crate prefix
        attr_name = dec.name.split("::")[-1].lower()
        for pattern, framework in _RUST_ENTRY_POINT_PATTERNS.items():
            if pattern.split("::")[-1].lower() == attr_name:
                return True, framework

    return False, None
