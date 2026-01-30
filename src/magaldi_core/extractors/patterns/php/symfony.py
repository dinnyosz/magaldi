"""PHP Symfony framework route extractor.

This module provides functions for extracting Symfony framework routes:
- PHP 8 Attribute routes (#[Route('/path', methods: ['GET'])])
- DocBlock annotation routes (@Route("/path", methods={"GET"}))

Symfony routing patterns:
https://symfony.com/doc/current/routing.html
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

# HTTP methods supported by Symfony Route attribute
_SYMFONY_HTTP_METHODS = {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"}


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass
class SymfonyController:
    """A Symfony controller with its routes."""

    name: str
    routes: list[HttpRoute]
    line_start: int
    line_end: int
    base_path: str | None = None  # From class-level #[Route] attribute


# =============================================================================
# ROUTE EXTRACTION
# =============================================================================


def extract_symfony_routes(tree: Tree, lines: list[str]) -> list[HttpRoute]:
    """Extract HTTP routes from PHP Symfony framework code.

    Detects patterns like:
    - #[Route('/path')]
    - #[Route('/path', methods: ['GET', 'POST'])]
    - #[Route('/path', name: 'route_name', methods: ['GET'])]

    Args:
        tree: Parsed tree-sitter tree.
        lines: Source code lines.

    Returns:
        List of detected HTTP routes.
    """
    routes: list[HttpRoute] = []

    for node in walk_tree(tree.root_node):
        if node.type == "method_declaration":
            method_routes = _extract_routes_from_method(node, lines)
            routes.extend(method_routes)

    return routes


def extract_symfony_controllers(tree: Tree, lines: list[str]) -> list[SymfonyController]:
    """Extract Symfony controllers with their routes.

    Args:
        tree: Parsed tree-sitter tree.
        lines: Source code lines.

    Returns:
        List of detected controllers with their routes.
    """
    controllers: list[SymfonyController] = []

    for node in walk_tree(tree.root_node):
        if node.type == "class_declaration":
            controller = _extract_controller(node, lines)
            if controller:
                controllers.append(controller)

    return controllers


# =============================================================================
# INTERNAL HELPERS
# =============================================================================


def _extract_routes_from_method(node: Node, lines: list[str]) -> list[HttpRoute]:
    """Extract routes from a method's PHP 8 attributes.

    Args:
        node: A method_declaration node.
        lines: Source code lines.

    Returns:
        List of HTTP routes found on this method.
    """
    routes: list[HttpRoute] = []

    # Find attribute_list in method children
    for child in node.children:
        if child.type == "attribute_list":
            for attr_group in child.children:
                if attr_group.type == "attribute_group":
                    route = _extract_route_from_attribute_group(attr_group, node)
                    if route:
                        routes.append(route)

    return routes


def _extract_route_from_attribute_group(
    attr_group: Node, method_node: Node
) -> HttpRoute | None:
    """Extract a route from an attribute_group node.

    Handles: #[Route('/path', methods: ['GET'])]

    Args:
        attr_group: An attribute_group node.
        method_node: The parent method_declaration node (for line info).

    Returns:
        HttpRoute if this is a Route attribute, None otherwise.
    """
    for child in attr_group.children:
        if child.type == "attribute":
            return _extract_route_from_attribute(child, method_node)
    return None


def _extract_route_from_attribute(
    attr_node: Node, method_node: Node
) -> HttpRoute | None:
    """Extract route info from an attribute node.

    Args:
        attr_node: An attribute node.
        method_node: The parent method_declaration node.

    Returns:
        HttpRoute if this is a Route attribute, None otherwise.
    """
    attr_name = None
    args_node = None

    for child in attr_node.children:
        if child.type == "name" or child.type == "qualified_name":
            attr_name = get_node_text(child)
            # Handle qualified names like Symfony\...\Route
            if "\\" in attr_name:
                attr_name = attr_name.split("\\")[-1]
        elif child.type == "arguments":
            args_node = child

    # Check if this is a Route attribute
    if attr_name != "Route":
        return None

    if not args_node:
        return None

    # Extract path and methods from arguments
    path = None
    methods: list[str] = []

    for arg in args_node.children:
        if arg.type == "argument":
            arg_name, arg_value = _extract_argument(arg)

            if arg_name is None and path is None:
                # First positional argument is the path
                path = arg_value
            elif arg_name == "methods":
                methods = _extract_methods_array(arg)
            elif arg_name == "path":
                path = arg_value

    if not path:
        return None

    # Default to GET if no methods specified
    method = methods[0] if methods else "GET"

    return HttpRoute(
        method=method.upper(),
        path=path,
        path_params=_extract_path_params(path),
        framework="symfony",
        line=method_node.start_point[0] + 1,
    )


def _extract_argument(arg_node: Node) -> tuple[str | None, str | None]:
    """Extract argument name and value from an argument node.

    Args:
        arg_node: An argument node.

    Returns:
        Tuple of (argument_name, argument_value). Name is None for positional args.
    """
    arg_name = None
    arg_value = None

    for child in arg_node.children:
        if child.type == "name":
            # This is a named argument: name: value
            arg_name = get_node_text(child)
        elif child.type in ("string", "encapsed_string"):
            arg_value = _get_string_content(child)

    return arg_name, arg_value


def _extract_methods_array(arg_node: Node) -> list[str]:
    """Extract HTTP methods from a methods argument.

    Handles: methods: ['GET', 'POST']

    Args:
        arg_node: An argument node containing the methods array.

    Returns:
        List of HTTP method strings.
    """
    methods: list[str] = []

    for child in arg_node.children:
        if child.type == "array_creation_expression":
            for arr_child in child.children:
                if arr_child.type == "array_element_initializer":
                    for elem in arr_child.children:
                        if elem.type in ("string", "encapsed_string"):
                            method = _get_string_content(elem)
                            if method.upper() in _SYMFONY_HTTP_METHODS:
                                methods.append(method.upper())

    return methods


def _extract_controller(node: Node, lines: list[str]) -> SymfonyController | None:
    """Extract a Symfony controller from a class declaration.

    Args:
        node: A class_declaration node.
        lines: Source code lines.

    Returns:
        SymfonyController if this class has Route attributes, None otherwise.
    """
    class_name = None
    base_path = None
    routes: list[HttpRoute] = []

    for child in node.children:
        if child.type == "name":
            class_name = get_node_text(child)
        elif child.type == "attribute_list":
            # Check for class-level Route attribute (base path)
            for attr_group in child.children:
                if attr_group.type == "attribute_group":
                    path = _extract_base_path_from_attribute_group(attr_group)
                    if path:
                        base_path = path
        elif child.type == "declaration_list":
            # Extract routes from methods
            for decl_child in child.children:
                if decl_child.type == "method_declaration":
                    method_routes = _extract_routes_from_method(decl_child, lines)
                    routes.extend(method_routes)

    if not class_name or not routes:
        return None

    # Prepend base path to all routes if present
    if base_path:
        for route in routes:
            if not route.path.startswith(base_path):
                route.path = base_path.rstrip("/") + "/" + route.path.lstrip("/")
                route.path_params = _extract_path_params(route.path)

    return SymfonyController(
        name=class_name,
        routes=routes,
        line_start=node.start_point[0] + 1,
        line_end=node.end_point[0] + 1,
        base_path=base_path,
    )


def _extract_base_path_from_attribute_group(attr_group: Node) -> str | None:
    """Extract base path from a class-level Route attribute.

    Args:
        attr_group: An attribute_group node.

    Returns:
        Base path string if found, None otherwise.
    """
    for child in attr_group.children:
        if child.type == "attribute":
            attr_name = None
            args_node = None

            for attr_child in child.children:
                if attr_child.type == "name" or attr_child.type == "qualified_name":
                    attr_name = get_node_text(attr_child)
                    if "\\" in attr_name:
                        attr_name = attr_name.split("\\")[-1]
                elif attr_child.type == "arguments":
                    args_node = attr_child

            if attr_name == "Route" and args_node:
                # Get first string argument as base path
                for arg in args_node.children:
                    if arg.type == "argument":
                        for arg_child in arg.children:
                            if arg_child.type in ("string", "encapsed_string"):
                                return _get_string_content(arg_child)

    return None


def _get_string_content(node: Node) -> str:
    """Extract string content from a string node."""
    for child in node.children:
        if child.type == "string_content":
            return get_node_text(child)
    # Fallback: strip quotes from the whole text
    text = get_node_text(node)
    return text.strip("'\"")


def _extract_path_params(path: str) -> list[str]:
    """Extract path parameters from a Symfony route path.

    Symfony uses {param} syntax for path parameters.
    """
    params = []
    for match in re.finditer(r"\{(\w+)\}", path):
        params.append(match.group(1))
    return params
