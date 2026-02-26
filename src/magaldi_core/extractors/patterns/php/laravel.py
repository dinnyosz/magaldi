"""PHP Laravel framework route extractor.

This module provides functions for extracting Laravel framework routes:
- HTTP routes (Route::get(), Route::post(), etc.)
- Route groups (Route::prefix()->group(), Route::middleware()->group())
- Resource routes (Route::resource(), Route::apiResource())

Laravel routing patterns:
https://laravel.com/docs/routing
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from tree_sitter import Node, Tree

from magaldi_core.extractors.base import get_node_text, walk_tree
from magaldi_core.extractors.types import HttpRoute

# =============================================================================
# CONSTANTS
# =============================================================================

# HTTP method names used by Laravel
_LARAVEL_HTTP_METHODS = {"get", "post", "put", "patch", "delete", "options", "any"}

# Resource route methods that generate multiple routes
_LARAVEL_RESOURCE_METHODS = {"resource", "apiResource"}

# All Laravel route-defining methods
_LARAVEL_ROUTE_METHODS = _LARAVEL_HTTP_METHODS | _LARAVEL_RESOURCE_METHODS | {"match"}


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass
class LaravelRouteGroup:
    """A Laravel route group with prefix/middleware."""

    prefix: str
    routes: list[HttpRoute]
    line_start: int
    line_end: int
    middleware: list[str] | None = None


# =============================================================================
# ROUTE EXTRACTION
# =============================================================================


def extract_laravel_routes(tree: Tree, lines: list[str]) -> list[HttpRoute]:
    """Extract HTTP routes from PHP Laravel framework code.

    Detects patterns like:
    - Route::get('/path', function ...)
    - Route::get('/path', [Controller::class, 'method'])
    - Route::post('/users', [UserController::class, 'store'])
    - Route::match(['get', 'post'], '/path', ...)
    - Route::any('/path', ...)

    Args:
        tree: Parsed tree-sitter tree.
        lines: Source code lines.

    Returns:
        List of detected HTTP routes.
    """
    routes: list[HttpRoute] = []

    for node in walk_tree(tree.root_node):
        if node.type == "scoped_call_expression":
            route = _extract_laravel_route_from_call(node, lines)
            if route:
                routes.append(route)

    return routes


def extract_laravel_route_groups(
    tree: Tree, lines: list[str]
) -> list[LaravelRouteGroup]:
    """Extract route groups from PHP Laravel framework code.

    Detects patterns like:
    - Route::prefix('admin')->group(function () { ... })
    - Route::middleware(['auth'])->group(function () { ... })

    Args:
        tree: Parsed tree-sitter tree.
        lines: Source code lines.

    Returns:
        List of detected route groups with their nested routes.
    """
    groups: list[LaravelRouteGroup] = []

    for node in walk_tree(tree.root_node):
        if node.type == "member_call_expression":
            group = _extract_laravel_group_from_call(node, lines)
            if group:
                groups.append(group)

    return groups


# =============================================================================
# INTERNAL HELPERS
# =============================================================================


def _extract_laravel_route_from_call(
    node: Node, _lines: list[str]
) -> HttpRoute | None:
    """Extract a Laravel route from a scoped_call_expression node.

    Laravel uses static method calls like Route::get('/path', callback).

    Args:
        node: A scoped_call_expression node from tree-sitter.
        lines: Source code lines.

    Returns:
        HttpRoute if this is a Laravel route call, None otherwise.
    """
    # Check if this is a Route:: call
    scope_node = None
    name_node = None
    args_node = None

    for child in node.children:
        if child.type == "name":
            if scope_node is None:
                scope_node = child
            else:
                name_node = child
        elif child.type == "arguments":
            args_node = child

    if not scope_node or not name_node:
        return None

    scope_text = get_node_text(scope_node)
    method_name = get_node_text(name_node).lower()

    # Must be a Route:: call with a valid method
    if scope_text != "Route" or method_name not in _LARAVEL_ROUTE_METHODS:
        return None

    if not args_node:
        return None

    # Extract path and method based on the route method type
    if method_name in _LARAVEL_HTTP_METHODS:
        return _extract_simple_route(method_name, args_node, node)
    elif method_name == "match":
        return _extract_match_route(args_node, node)
    elif method_name in _LARAVEL_RESOURCE_METHODS:
        return _extract_resource_route(method_name, args_node, node)

    return None


def _extract_simple_route(
    method: str, args_node: Node, node: Node
) -> HttpRoute | None:
    """Extract a simple HTTP route (get, post, put, etc.)."""
    path = _get_first_string_arg(args_node)
    if not path:
        return None

    return HttpRoute(
        method=method.upper() if method != "any" else "ANY",
        path=path,
        path_params=_extract_path_params(path),
        framework="laravel",
        line=node.start_point[0] + 1,
    )


def _extract_match_route(args_node: Node, node: Node) -> HttpRoute | None:
    """Extract a Route::match(['GET', 'POST'], '/path', ...) route."""
    # First arg is array of methods, second is path
    args = list(args_node.children)
    methods_arr = None
    path = None

    for arg in args:
        if arg.type == "argument":
            for child in arg.children:
                if child.type == "array_creation_expression" and methods_arr is None:
                    methods_arr = child
                elif child.type in ("string", "encapsed_string") and path is None:
                    path = _get_string_content(child)

    if not path:
        return None

    # Get first method from array
    method = "GET"
    if methods_arr:
        method = _extract_first_method_from_array(methods_arr) or "GET"

    return HttpRoute(
        method=method.upper(),
        path=path,
        path_params=_extract_path_params(path),
        framework="laravel",
        line=node.start_point[0] + 1,
    )


def _extract_resource_route(
    resource_type: str, args_node: Node, node: Node
) -> HttpRoute | None:
    """Extract a resource route (Route::resource(), Route::apiResource()).

    Resource routes generate multiple routes, we capture the base path.
    """
    path = _get_first_string_arg(args_node)
    if not path:
        return None

    # Ensure path starts with /
    if not path.startswith("/"):
        path = "/" + path

    return HttpRoute(
        method="RESOURCE" if resource_type == "resource" else "API_RESOURCE",
        path=path,
        path_params=[],
        framework="laravel",
        line=node.start_point[0] + 1,
    )


def _extract_laravel_group_from_call(
    node: Node, lines: list[str]
) -> LaravelRouteGroup | None:
    """Extract a Laravel route group from a member_call_expression.

    Detects: Route::prefix('admin')->group(function () { ... })
    """
    # Check if this is a ->group() call
    method_name = None
    for child in node.children:
        if child.type == "name":
            method_name = get_node_text(child)

    if method_name != "group":
        return None

    # Find the prefix from the chain
    prefix = _find_prefix_in_chain(node)
    if not prefix:
        prefix = ""

    # Find the callback and extract nested routes
    nested_routes: list[HttpRoute] = []
    for child in node.children:
        if child.type == "arguments":
            for arg in child.children:
                if arg.type == "argument":
                    for inner in arg.children:
                        if inner.type == "anonymous_function":
                            # Parse nested routes from callback body
                            nested_routes = _extract_routes_from_callback(inner, prefix, lines)

    if not nested_routes and not prefix:
        return None

    return LaravelRouteGroup(
        prefix=prefix,
        routes=nested_routes,
        line_start=node.start_point[0] + 1,
        line_end=node.end_point[0] + 1,
    )


def _find_prefix_in_chain(node: Node) -> str | None:
    """Find prefix in a chained call like Route::prefix('admin')->group()."""
    current = node
    while current:
        if current.type == "scoped_call_expression":
            # Check if this is a prefix() call
            for child in current.children:
                if child.type == "name":
                    name = get_node_text(child)
                    if name == "prefix":
                        # Get the prefix argument
                        for sibling in current.children:
                            if sibling.type == "arguments":
                                return _get_first_string_arg(sibling)
        elif current.type == "member_call_expression":
            # Check the object part of the call chain
            for child in current.children:
                if child.type in ("scoped_call_expression", "member_call_expression"):
                    result = _find_prefix_in_chain(child)
                    if result:
                        return result

        # Move up the chain
        if hasattr(current, 'children') and current.children:
            for child in current.children:
                if child.type in ("scoped_call_expression", "member_call_expression"):
                    current = child
                    break
            else:
                break
        else:
            break

    return None


def _extract_routes_from_callback(
    callback_node: Node, prefix: str, lines: list[str]
) -> list[HttpRoute]:
    """Extract routes from an anonymous function callback."""
    routes: list[HttpRoute] = []

    for node in walk_tree(callback_node):
        if node.type == "scoped_call_expression":
            route = _extract_laravel_route_from_call(node, lines)
            if route:
                # Prepend prefix to path
                full_path = prefix + route.path if not route.path.startswith(prefix) else route.path
                routes.append(HttpRoute(
                    method=route.method,
                    path=full_path,
                    path_params=_extract_path_params(full_path),
                    framework="laravel",
                    line=route.line,
                ))

    return routes


def _get_first_string_arg(args_node: Node) -> str | None:
    """Get the first string argument from an arguments node."""
    for child in args_node.children:
        if child.type == "argument":
            for inner in child.children:
                if inner.type in ("string", "encapsed_string"):
                    return _get_string_content(inner)
    return None


def _get_string_content(node: Node) -> str:
    """Extract string content from a string node."""
    for child in node.children:
        if child.type == "string_content":
            return get_node_text(child)  # type: ignore[no-any-return]
    # Fallback: strip quotes from the whole text
    text = get_node_text(node)
    return text.strip("'\"")  # type: ignore[no-any-return]


def _extract_first_method_from_array(array_node: Node) -> str | None:
    """Extract the first method string from an array like ['GET', 'POST']."""
    for child in array_node.children:
        if child.type == "array_element_initializer":
            for inner in child.children:
                if inner.type in ("string", "encapsed_string"):
                    return _get_string_content(inner)
    return None


def _extract_path_params(path: str) -> list[str]:
    """Extract path parameters from a Laravel route path.

    Laravel uses {param} syntax for path parameters.
    """
    params = []
    for match in re.finditer(r"\{(\w+)\??}", path):
        params.append(match.group(1))
    return params
