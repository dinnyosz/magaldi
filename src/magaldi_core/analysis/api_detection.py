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
# Maps decorator name -> (HTTP method, framework)
# Use "*" for method when it needs to be extracted from args
_HTTP_ROUTE_PATTERNS: dict[str, tuple[str, str]] = {
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
    # Starlette patterns (same as FastAPI)
    "route": ("*", "starlette"),
    # Litestar patterns
    "get": ("GET", "litestar"),
    "post": ("POST", "litestar"),
    "put": ("PUT", "litestar"),
    "delete": ("DELETE", "litestar"),
    "patch": ("PATCH", "litestar"),
    # NestJS patterns (TypeScript/JavaScript)
    "Get": ("GET", "nestjs"),
    "Post": ("POST", "nestjs"),
    "Put": ("PUT", "nestjs"),
    "Delete": ("DELETE", "nestjs"),
    "Patch": ("PATCH", "nestjs"),
    # Hono patterns (JavaScript)
    "c.get": ("GET", "hono"),
    "c.post": ("POST", "hono"),
    "c.put": ("PUT", "hono"),
    "c.delete": ("DELETE", "hono"),
    "app.get": ("GET", "hono"),  # Also used by Hono
    "app.post": ("POST", "hono"),
    # Generic patterns that could be any framework
    "route": ("*", "generic"),
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


def _extract_method_from_flask_args(args: str | None) -> str:
    """Extract HTTP method from Flask route arguments.

    Handles parenthesized argument lists:
    - 'methods=["POST"]' -> 'POST'
    - '("/users", methods=["POST"])' -> 'POST'
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


def _extract_method_from_drf_args(args: str | None) -> str:
    """Extract HTTP method from Django REST Framework @api_view args.

    Handles:
    - '(["GET"])' -> 'GET'
    - '(["GET", "POST"])' -> 'GET'
    - '["GET"]' -> 'GET'
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

            # Handle method extraction based on framework
            if method == "*":
                if framework == "django-rest":
                    method = _extract_method_from_drf_args(dec.args)
                else:
                    method = _extract_method_from_flask_args(dec.args)

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
    """Detect singleton pattern.

    Looks for:
    - _instance attribute (instance or class level)
    - get_instance/getInstance/instance methods
    - __new__ method override
    - @classmethod or @staticmethod on instance getter
    """
    score = 0.0
    attributes = class_info.get("attributes", [])
    methods = class_info.get("methods", [])
    class_variables = class_info.get("class_variables", [])
    decorators = class_info.get("decorators", [])

    # Has _instance attribute (instance or class level)
    instance_attrs = ["_instance", "instance", "_singleton", "_shared_instance"]
    if any(attr in attributes for attr in instance_attrs):
        score += 0.3
    if any(attr in class_variables for attr in instance_attrs):
        score += 0.4

    # Has get_instance/getInstance/instance method
    instance_methods = ["get_instance", "getInstance", "instance", "shared", "default"]
    if any(m in methods for m in instance_methods):
        score += 0.4

    # Has __new__ method (Python singleton pattern)
    if "__new__" in methods:
        score += 0.3

    # Uses classmethod decorator (common for singletons)
    if "classmethod" in decorators:
        score += 0.2

    # Returns self/instance from get_instance
    if class_info.get("method_returns_self"):
        score += 0.1

    return min(score, 1.0)  # Cap at 1.0


def _detect_builder(class_info: dict[str, Any]) -> float:
    """Detect builder pattern.

    Looks for:
    - Multiple methods returning self (method chaining)
    - build() method
    - with_*/set_*/add_* method naming
    - *Builder class name suffix
    """
    score = 0.0
    methods = class_info.get("methods", [])
    returns_self = class_info.get("methods_return_self", [])
    name = class_info.get("name", "")

    # Multiple methods that return self (method chaining)
    if len(returns_self) >= 2:
        score += 0.4

    # Has a build() method
    if "build" in methods:
        score += 0.3

    # Name ends with Builder
    if name.endswith("Builder"):
        score += 0.3

    # Has with_* methods (fluent interface)
    with_methods = [m for m in methods if m.startswith("with_")]
    if len(with_methods) >= 2:
        score += 0.3

    # Has set_* methods (common builder pattern)
    set_methods = [m for m in methods if m.startswith("set_")]
    if len(set_methods) >= 2:
        score += 0.2

    # Has add_* methods (collection builders)
    add_methods = [m for m in methods if m.startswith("add_")]
    if len(add_methods) >= 2:
        score += 0.2

    return min(score, 1.0)  # Cap at 1.0


def _detect_factory(class_info: dict[str, Any], calls: list[ExtractedCall]) -> float:
    """Detect factory pattern.

    Looks for:
    - *Factory class name
    - create_*/make_*/build_*/from_*/new_* methods
    - Methods that instantiate other classes
    - @classmethod/@staticmethod decorators
    """
    score = 0.0
    methods = class_info.get("methods", [])
    name = class_info.get("name", "")
    decorators = class_info.get("decorators", [])

    # Name contains Factory
    if "Factory" in name or "factory" in name.lower():
        score += 0.3

    # Has create*/make*/build*/from_*/new_* methods
    factory_prefixes = ("create", "make", "build", "from_", "new_")
    factory_methods = [
        m
        for m in methods
        if any(m.startswith(p) or m.startswith(p.title()) for p in factory_prefixes)
    ]
    if factory_methods:
        score += 0.3
    if len(factory_methods) >= 2:
        score += 0.3
    if len(factory_methods) >= 3:
        score += 0.1

    # Methods instantiate other classes (uppercase call = class constructor)
    instantiation_calls = [
        c for c in calls if c.receiver is None and c.name and c.name[0].isupper()
    ]
    if instantiation_calls:
        score += 0.3

    # Uses classmethod/staticmethod (common for factories)
    if "classmethod" in decorators or "staticmethod" in decorators:
        score += 0.2

    return min(score, 1.0)


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
