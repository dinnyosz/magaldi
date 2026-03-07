"""Java web framework route patterns and detection.

Pattern definitions and detection functions for HTTP routes
in Java web frameworks:
- Spring MVC / Spring Boot (@GetMapping, @PostMapping, @RequestMapping)
- JAX-RS / Jakarta REST (@GET, @POST, @Path)
- Micronaut (@Get, @Post, @Controller)
"""

from __future__ import annotations

import re

from magaldi_core.extractors.types import DecoratorInfo, HttpRoute

# =============================================================================
# HTTP ROUTE PATTERNS
# =============================================================================

# Maps annotation name -> (HTTP method, framework)
# Use "*" for method when it needs to be extracted from args
_JAVA_HTTP_ROUTE_PATTERNS: dict[str, tuple[str, str]] = {
    # Spring MVC / Spring Boot
    "@GetMapping": ("GET", "spring"),
    "@PostMapping": ("POST", "spring"),
    "@PutMapping": ("PUT", "spring"),
    "@DeleteMapping": ("DELETE", "spring"),
    "@PatchMapping": ("PATCH", "spring"),
    "@RequestMapping": ("*", "spring"),
    # JAX-RS / Jakarta REST
    "@GET": ("GET", "jax-rs"),
    "@POST": ("POST", "jax-rs"),
    "@PUT": ("PUT", "jax-rs"),
    "@DELETE": ("DELETE", "jax-rs"),
    "@PATCH": ("PATCH", "jax-rs"),
    "@HEAD": ("HEAD", "jax-rs"),
    "@OPTIONS": ("OPTIONS", "jax-rs"),
    "@Path": ("*", "jax-rs"),
    # Micronaut
    "@Get": ("GET", "micronaut"),
    "@Post": ("POST", "micronaut"),
    "@Put": ("PUT", "micronaut"),
    "@Delete": ("DELETE", "micronaut"),
    "@Patch": ("PATCH", "micronaut"),
    "@Head": ("HEAD", "micronaut"),
    "@Options": ("OPTIONS", "micronaut"),
}

# Path parameter regex: {paramName} or {paramName:regex}
_JAVA_PATH_PARAM_RE = re.compile(r"\{(\w+)(?::[^}]*)?\}")


def _extract_path_from_args(args: str | None) -> str | None:
    """Extract the path string from annotation arguments.

    Handles:
    - @GetMapping("/users")
    - @GetMapping(value = "/users")
    - @GetMapping(path = "/users")
    - @RequestMapping(value = "/users", method = RequestMethod.GET)
    - @Path("/users/{id}")
    """
    if not args:
        return None

    args = args.strip()

    # Direct string: "/users"
    if args.startswith('"') and args.endswith('"'):
        return args[1:-1]

    # Single-element array: {"/users"}
    if args.startswith("{") and args.endswith("}"):
        inner = args[1:-1].strip()
        if inner.startswith('"') and inner.endswith('"'):
            return inner[1:-1]

    # Named parameter: value = "/users" or path = "/users"
    for key in ("value", "path"):
        pattern = rf'{key}\s*=\s*"([^"]*)"'
        match = re.search(pattern, args)
        if match:
            return match.group(1)

    # Fallback: find any quoted string
    match = re.search(r'"([^"]*)"', args)
    if match:
        return match.group(1)

    return None


def _extract_method_from_args(args: str | None) -> str:
    """Extract HTTP method from @RequestMapping args.

    @RequestMapping(value = "/users", method = RequestMethod.GET)
    """
    if not args:
        return "*"

    match = re.search(r"method\s*=\s*RequestMethod\.(\w+)", args)
    if match:
        return match.group(1).upper()

    return "*"


def _extract_path_params(path: str) -> list[str]:
    """Extract path parameters from a route path.

    /users/{id} -> ["id"]
    /users/{userId}/orders/{orderId} -> ["userId", "orderId"]
    """
    return _JAVA_PATH_PARAM_RE.findall(path)


def detect_java_http_routes(
    decorators: list[DecoratorInfo],
) -> list[HttpRoute]:
    """Detect HTTP routes from Java annotation information.

    Args:
        decorators: List of decorator (annotation) info from a method.

    Returns:
        List of detected HTTP routes.
    """
    routes: list[HttpRoute] = []
    path_prefix: str | None = None

    # First pass: check for @Path annotation (JAX-RS class-level path)
    for dec in decorators:
        if dec.name == "@Path":
            path_prefix = _extract_path_from_args(dec.args)

    for dec in decorators:
        pattern = _JAVA_HTTP_ROUTE_PATTERNS.get(dec.name)
        if not pattern:
            continue

        method, framework = pattern
        path = _extract_path_from_args(dec.args)

        # For @RequestMapping, extract method from args
        if dec.name == "@RequestMapping" and method == "*":
            method = _extract_method_from_args(dec.args)

        # JAX-RS HTTP annotations (@GET, @POST) may not have path;
        # use @Path annotation if present
        if not path and dec.name in ("@GET", "@POST", "@PUT", "@DELETE", "@PATCH", "@HEAD", "@OPTIONS"):
            path = path_prefix or "/"

        # Skip if still no path (e.g., bare @Path without HTTP method)
        if dec.name == "@Path" and method == "*":
            # @Path alone just defines prefix, not a route
            continue

        if path:
            routes.append(HttpRoute(
                method=method,
                path=path,
                path_params=_extract_path_params(path),
                framework=framework,
            ))

    return routes
