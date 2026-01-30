"""PHP Slim framework extractor.

This module provides functions for extracting Slim framework patterns:
- HTTP routes ($app->get(), $app->post(), etc.)
- Route groups ($app->group())
- Middleware chains

Slim PHP 4.x patterns:
https://www.slimframework.com/docs/v4/
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

# HTTP method names used by Slim framework
SLIM_HTTP_METHODS = {"get", "post", "put", "delete", "patch", "options", "any"}

# All Slim route-defining methods
SLIM_ROUTE_METHODS = SLIM_HTTP_METHODS | {"map", "group"}


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass
class SlimRouteGroup:
    """A Slim route group with a path prefix."""

    prefix: str
    routes: list[HttpRoute]
    line_start: int
    line_end: int
    middleware: list[str] | None = None


# =============================================================================
# ROUTE EXTRACTION
# =============================================================================


def extract_slim_routes(tree: Tree, lines: list[str]) -> list[HttpRoute]:
    """Extract HTTP routes from PHP Slim framework code.

    Detects patterns like:
    - $app->get('/path', function ...)
    - $app->post('/path', function ...)
    - $app->map(['GET', 'POST'], '/path', function ...)
    - $group->get('/path', function ...)

    Args:
        tree: Parsed tree-sitter tree.
        lines: Source code lines.

    Returns:
        List of detected HTTP routes.
    """
    routes: list[HttpRoute] = []

    for node in walk_tree(tree.root_node):
        if node.type == "member_call_expression":
            route = _extract_slim_route_from_call(node, lines)
            if route:
                routes.append(route)

    return routes


def extract_slim_route_groups(
    tree: Tree, lines: list[str]
) -> list[SlimRouteGroup]:
    """Extract route groups from PHP Slim framework code.

    Detects patterns like:
    - $app->group('/api', function ($group) { ... })
    - $app->group('/v1', function ($group) { ... })->add($middleware)

    Args:
        tree: Parsed tree-sitter tree.
        lines: Source code lines.

    Returns:
        List of detected route groups with their nested routes.
    """
    groups: list[SlimRouteGroup] = []

    for node in walk_tree(tree.root_node):
        if node.type == "member_call_expression":
            group = _extract_slim_group_from_call(node, lines)
            if group:
                groups.append(group)

    return groups


def _extract_slim_route_from_call(
    node: Node, lines: list[str]
) -> HttpRoute | None:
    """Extract a Slim route from a member_call_expression node.

    Args:
        node: A member_call_expression node.
        lines: Source code lines.

    Returns:
        HttpRoute if this is a Slim route call, None otherwise.
    """
    method_name = None
    path = None
    http_methods: list[str] = []

    # Extract the method name (get, post, etc.)
    for child in node.children:
        if child.type == "name":
            method_name = get_node_text(child).lower()
        elif child.type == "arguments":
            # Extract path and HTTP methods from arguments
            args = _extract_call_arguments(child)
            if args:
                if method_name == "map" and len(args) >= 2:
                    # $app->map(['GET', 'POST'], '/path', ...)
                    http_methods = _extract_methods_from_array(args[0])
                    path = _extract_string_value(args[1])
                elif method_name in SLIM_HTTP_METHODS and len(args) >= 1:
                    # $app->get('/path', ...)
                    path = _extract_string_value(args[0])
                elif method_name == "any" and len(args) >= 1:
                    # $app->any('/path', ...) - responds to any HTTP method
                    path = _extract_string_value(args[0])
                    http_methods = ["ANY"]

    if not path:
        return None

    # Determine HTTP method
    if method_name in SLIM_HTTP_METHODS:
        http_method = method_name.upper()
    elif method_name == "map" and http_methods:
        http_method = http_methods[0]  # Use first method for single value
    elif method_name == "any":
        http_method = "ANY"
    else:
        return None

    # Extract path parameters like {id}
    path_params = _extract_path_params(path)

    return HttpRoute(
        method=http_method,
        path=path,
        path_params=path_params,
        framework="slim",
    )


def _extract_slim_group_from_call(
    node: Node, lines: list[str]
) -> SlimRouteGroup | None:
    """Extract a Slim route group from a member_call_expression node.

    Args:
        node: A member_call_expression node.
        lines: Source code lines.

    Returns:
        SlimRouteGroup if this is a Slim group call, None otherwise.
    """
    method_name = None
    prefix = None

    # Extract the method name
    for child in node.children:
        if child.type == "name":
            method_name = get_node_text(child).lower()
        elif child.type == "arguments":
            if method_name == "group":
                args = _extract_call_arguments(child)
                if args:
                    prefix = _extract_string_value(args[0])

    if method_name != "group" or not prefix:
        return None

    # Extract nested routes from the group callback
    nested_routes: list[HttpRoute] = []
    for child in walk_tree(node):
        if child.type == "anonymous_function":
            # Parse routes inside the callback
            for nested in walk_tree(child):
                if nested.type == "member_call_expression" and nested != node:
                    route = _extract_slim_route_from_call(nested, lines)
                    if route:
                        # Prepend group prefix to route path
                        route = HttpRoute(
                            method=route.method,
                            path=prefix + route.path,
                            path_params=route.path_params,
                            framework=route.framework,
                        )
                        nested_routes.append(route)

    return SlimRouteGroup(
        prefix=prefix,
        routes=nested_routes,
        line_start=node.start_point[0] + 1,
        line_end=node.end_point[0] + 1,
    )


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _extract_call_arguments(args_node: Node) -> list[Node]:
    """Extract argument nodes from an arguments node."""
    args = []
    for child in args_node.children:
        if child.type == "argument":
            # Get the actual value inside the argument
            for arg_child in child.children:
                if arg_child.type not in ("(", ")", ","):
                    args.append(arg_child)
                    break
        elif child.type not in ("(", ")", ","):
            args.append(child)
    return args


def _extract_string_value(node: Node) -> str | None:
    """Extract string value from a string node."""
    if node.type == "string":
        text = get_node_text(node)
        # Remove quotes
        if text.startswith(("'", '"')) and text.endswith(("'", '"')):
            return text[1:-1]
        return text
    elif node.type == "encapsed_string":
        # Double-quoted string with potential interpolation
        text = get_node_text(node)
        if text.startswith('"') and text.endswith('"'):
            return text[1:-1]
    return None


def _extract_methods_from_array(node: Node) -> list[str]:
    """Extract HTTP method strings from an array literal."""
    methods = []
    if node.type == "array_creation_expression":
        for child in walk_tree(node):
            if child.type in ("string", "encapsed_string"):
                text = get_node_text(child)
                if text.startswith(("'", '"')) and text.endswith(("'", '"')):
                    methods.append(text[1:-1].upper())
            elif child.type == "string_content":
                # Direct string content without quotes
                text = get_node_text(child)
                if text:
                    methods.append(text.upper())
    return methods


def _extract_path_params(path: str) -> list[str]:
    """Extract path parameters from a route path.

    Handles Slim/FastAPI style: {param}
    """
    params = []
    for match in re.finditer(r"\{(\w+)\}", path):
        params.append(match.group(1))
    return params
