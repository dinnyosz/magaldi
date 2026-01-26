"""API surface detection functions.

This module provides functions for detecting:
- HTTP routes (FastAPI, Flask, Express)
- CLI commands (Click, Typer, argparse)
- Public API elements
- Design patterns (Singleton, Builder, Factory, Repository)
"""

from __future__ import annotations

import re
from typing import Any

from magaldi_core.extractors.types import (
    CliCommand,
    DecoratorInfo,
    ExtractedCall,
    HttpRoute,
)

# =============================================================================
# HTTP ROUTE DETECTION
# =============================================================================

# HTTP route decorator patterns
_HTTP_ROUTE_PATTERNS: dict[str, tuple[str, str]] = {
    # FastAPI patterns
    "router.get": ("GET", "fastapi"),
    "router.post": ("POST", "fastapi"),
    "router.put": ("PUT", "fastapi"),
    "router.delete": ("DELETE", "fastapi"),
    "router.patch": ("PATCH", "fastapi"),
    "app.get": ("GET", "fastapi"),
    "app.post": ("POST", "fastapi"),
    "app.put": ("PUT", "fastapi"),
    "app.delete": ("DELETE", "fastapi"),
    # Flask patterns
    "app.route": ("*", "flask"),
    "blueprint.route": ("*", "flask"),
}


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
    """Extract path from decorator arguments."""
    if not args:
        return None
    match = re.match(r'["\']([^"\']+)["\']', args.strip())
    if match:
        return match.group(1)
    return None


def _extract_method_from_flask_args(args: str | None) -> str:
    """Extract HTTP method from Flask route arguments."""
    if not args:
        return "GET"
    match = re.search(r'methods\s*=\s*\[([^\]]+)\]', args)
    if match:
        methods_str = match.group(1)
        method_match = re.search(r'["\'](\w+)["\']', methods_str)
        if method_match:
            return method_match.group(1).upper()
    return "GET"


def detect_http_routes(
    decorators: list[DecoratorInfo],
    language: str,  # noqa: ARG001 - reserved for future language-specific patterns
) -> list[HttpRoute]:
    """Detect HTTP routes from decorator information.

    Args:
        decorators: List of decorator information extracted from a function.
        language: The programming language (e.g., "python", "javascript").

    Returns:
        List of detected HTTP routes.
    """
    routes = []

    for dec in decorators:
        if dec.name in _HTTP_ROUTE_PATTERNS:
            method, framework = _HTTP_ROUTE_PATTERNS[dec.name]
            path = _extract_path_from_args(dec.args)

            if path:
                if method == "*":
                    method = _extract_method_from_flask_args(dec.args)

                routes.append(
                    HttpRoute(
                        method=method,
                        path=path,
                        path_params=_extract_path_params(path),
                        framework=framework,
                    )
                )

    return routes


# =============================================================================
# CLI COMMAND DETECTION
# =============================================================================

# CLI command decorator patterns
_CLI_COMMAND_PATTERNS: dict[str, str] = {
    "click.command": "click",
    "click.group": "click",
    "app.command": "typer",
    "typer.command": "typer",
}


def detect_cli_commands(
    decorators: list[DecoratorInfo],
    function_name: str,
    language: str,  # noqa: ARG001 - reserved for future language-specific patterns
) -> list[CliCommand]:
    """Detect CLI commands from decorator information.

    Args:
        decorators: List of decorator information extracted from a function.
        function_name: Name of the decorated function.
        language: The programming language (e.g., "python").

    Returns:
        List of detected CLI commands.
    """
    commands = []

    for dec in decorators:
        if dec.name in _CLI_COMMAND_PATTERNS:
            framework = _CLI_COMMAND_PATTERNS[dec.name]

            # Extract options from sibling decorators
            options = []
            for other_dec in decorators:
                if other_dec.name in (
                    "click.option",
                    "click.argument",
                    "typer.Option",
                    "typer.Argument",
                ):
                    option_name = ""
                    if other_dec.args:
                        # Extract the first argument as the option name
                        args_parts = other_dec.args.split(",")
                        if args_parts:
                            option_name = args_parts[0].strip().strip("\"'")

                    options.append(
                        {
                            "name": option_name,
                            "type": None,
                            "required": "required=True" in (other_dec.full or ""),
                        }
                    )

            commands.append(
                CliCommand(
                    name=function_name,
                    options=options,
                    framework=framework,
                )
            )

    return commands


# =============================================================================
# PUBLIC API DETECTION
# =============================================================================

# Decorators that indicate public API
_PUBLIC_API_DECORATORS = {
    "api_endpoint",
    "public",
    "export",
    "exposed",
    "router.get",
    "router.post",
    "router.put",
    "router.delete",
    "router.patch",
    "app.get",
    "app.post",
    "app.put",
    "app.delete",
    "app.route",
    "blueprint.route",
    "click.command",
    "click.group",
    "app.command",
}


def detect_public_api(
    name: str,
    decorators: list[DecoratorInfo],
    visibility: str,
    language: str,  # noqa: ARG001 - reserved for future language-specific patterns
) -> bool:
    """Detect if an element is a public API.

    Args:
        name: Name of the element (function, method, class).
        decorators: List of decorator information.
        visibility: Visibility level ("public", "private", "protected").
        language: The programming language.

    Returns:
        True if the element is considered a public API, False otherwise.
    """
    # Private/protected are not public API
    if visibility != "public":
        return False

    # Dunder methods are not public API
    if name.startswith("__") and name.endswith("__"):
        return False

    # Check for public API decorators
    for dec in decorators:
        if dec.name in _PUBLIC_API_DECORATORS:
            return True

    # Default: public visibility and not private naming = public API
    return True


# =============================================================================
# DESIGN PATTERN DETECTION
# =============================================================================


def detect_patterns(
    class_info: dict[str, Any],
    calls: list[ExtractedCall],
    _language: str,
) -> tuple[list[str], dict[str, float]]:
    """Detect design patterns in a class.

    Args:
        class_info: Dict with class name, attributes, methods, etc.
        calls: Calls made within the class methods.
        language: Programming language.

    Returns:
        Tuple of (detected pattern names, confidence scores).
    """
    patterns = []
    confidence = {}

    # Singleton detection
    singleton_score = _detect_singleton(class_info)
    if singleton_score >= 0.6:
        patterns.append("singleton")
        confidence["singleton"] = singleton_score

    # Builder detection
    builder_score = _detect_builder(class_info)
    if builder_score >= 0.6:
        patterns.append("builder")
        confidence["builder"] = builder_score

    # Factory detection
    factory_score = _detect_factory(class_info, calls)
    if factory_score >= 0.6:
        patterns.append("factory")
        confidence["factory"] = factory_score

    # Repository detection
    repository_score = _detect_repository(class_info)
    if repository_score >= 0.6:
        patterns.append("repository")
        confidence["repository"] = repository_score

    return patterns, confidence


def _detect_singleton(class_info: dict[str, Any]) -> float:
    """Detect singleton pattern."""
    score = 0.0
    attributes = class_info.get("attributes", [])
    methods = class_info.get("methods", [])

    # Has _instance attribute
    if "_instance" in attributes or "instance" in attributes:
        score += 0.3

    # Has get_instance method
    if "get_instance" in methods or "getInstance" in methods:
        score += 0.3

    # Has __new__ method (Python singleton pattern)
    if "__new__" in methods:
        score += 0.2

    # Returns self/instance from get_instance
    if class_info.get("method_returns_self"):
        score += 0.2

    return score


def _detect_builder(class_info: dict[str, Any]) -> float:
    """Detect builder pattern."""
    score = 0.0
    methods = class_info.get("methods", [])
    returns_self = class_info.get("methods_return_self", [])

    # Multiple methods that return self (method chaining)
    if len(returns_self) >= 2:
        score += 0.4

    # Has a build() method
    if "build" in methods:
        score += 0.3

    # Name ends with Builder
    if class_info.get("name", "").endswith("Builder"):
        score += 0.3

    return score


def _detect_factory(class_info: dict[str, Any], calls: list[ExtractedCall]) -> float:
    """Detect factory pattern."""
    score = 0.0
    methods = class_info.get("methods", [])
    name = class_info.get("name", "")

    # Name contains Factory
    if "Factory" in name or "factory" in name.lower():
        score += 0.3

    # Has create* methods
    create_methods = [m for m in methods if m.startswith("create") or m.startswith("make")]
    if create_methods:
        score += 0.3

    # Methods instantiate other classes
    instantiation_calls = [c for c in calls if c.receiver is None and c.name[0].isupper()]
    if instantiation_calls:
        score += 0.4

    return score


def _detect_repository(class_info: dict[str, Any]) -> float:
    """Detect repository pattern."""
    score = 0.0
    methods = class_info.get("methods", [])
    name = class_info.get("name", "")

    # Name contains Repository
    if "Repository" in name or "Repo" in name:
        score += 0.3

    # Has CRUD-like methods
    crud_methods = {"get", "find", "save", "update", "delete", "create", "add", "remove"}
    found_crud = [m for m in methods if any(crud in m.lower() for crud in crud_methods)]
    if len(found_crud) >= 2:
        score += 0.4

    # Has find_by_* methods
    find_by_methods = [m for m in methods if m.startswith("find_by") or m.startswith("get_by")]
    if find_by_methods:
        score += 0.3

    return score
