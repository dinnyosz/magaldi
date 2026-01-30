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
from magaldi_core.extractors.patterns.python import (
    PYTHON_HTTP_ROUTE_PATTERNS,
    PYTHON_CLI_COMMAND_PATTERNS,
    PYTHON_CLI_COMMAND_SUFFIXES,
    extract_method_from_flask_args,
    extract_method_from_drf_args,
)
from magaldi_core.extractors.patterns.javascript import (
    JS_HTTP_ROUTE_PATTERNS,
)

# =============================================================================
# HTTP ROUTE DETECTION
# =============================================================================

# Generic patterns that could be any framework
_GENERIC_HTTP_ROUTE_PATTERNS: dict[str, tuple[str, str]] = {
    "route": ("*", "generic"),
}


def _get_http_route_patterns(language: str) -> dict[str, tuple[str, str]]:
    """Get HTTP route patterns for a specific language.

    Args:
        language: Programming language (e.g., "python", "javascript", "typescript").

    Returns:
        Dict mapping decorator names to (method, framework) tuples.
    """
    if language == "python":
        return {**PYTHON_HTTP_ROUTE_PATTERNS, **_GENERIC_HTTP_ROUTE_PATTERNS}
    elif language in ("javascript", "typescript", "tsx"):
        return {**JS_HTTP_ROUTE_PATTERNS, **_GENERIC_HTTP_ROUTE_PATTERNS}
    else:
        # For unknown languages, try all patterns
        return {
            **PYTHON_HTTP_ROUTE_PATTERNS,
            **JS_HTTP_ROUTE_PATTERNS,
            **_GENERIC_HTTP_ROUTE_PATTERNS,
        }

# Reference CLI patterns from Python module
_CLI_COMMAND_PATTERNS = PYTHON_CLI_COMMAND_PATTERNS
_CLI_COMMAND_SUFFIXES = PYTHON_CLI_COMMAND_SUFFIXES


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


def detect_http_routes(
    decorators: list[DecoratorInfo],
    language: str,
) -> list[HttpRoute]:
    """Detect HTTP routes from decorator information.

    Args:
        decorators: List of decorator information extracted from a function.
        language: The programming language (e.g., "python", "javascript").

    Returns:
        List of detected HTTP routes.
    """
    routes = []
    patterns = _get_http_route_patterns(language)

    for dec in decorators:
        if dec.name in patterns:
            method, framework = patterns[dec.name]
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


# =============================================================================
# CLI COMMAND DETECTION
# =============================================================================

# CLI patterns are imported from magaldi_core.extractors.patterns.python
# and aliased as _CLI_COMMAND_PATTERNS and _CLI_COMMAND_SUFFIXES at the top


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
        framework = None

        # Check exact pattern match first
        if dec.name in _CLI_COMMAND_PATTERNS:
            framework = _CLI_COMMAND_PATTERNS[dec.name]
        else:
            # Check for suffix patterns like main.command, web.group, etc.
            for suffix in _CLI_COMMAND_SUFFIXES:
                if dec.name.endswith(suffix):
                    framework = "click"  # Assume click for .command/.group patterns
                    break

        if framework:
            # Extract command name from decorator args if available
            # e.g., @web.command("serve") -> command_name = "serve"
            command_name = function_name
            if dec.args:
                # Extract first string argument as command name
                args_stripped = dec.args.strip().strip("()")
                if args_stripped.startswith(("'", '"')):
                    # Extract quoted string
                    quote_char = args_stripped[0]
                    end_quote = args_stripped.find(quote_char, 1)
                    if end_quote > 0:
                        command_name = args_stripped[1:end_quote]

            # Extract options from sibling decorators
            options = []
            for other_dec in decorators:
                # Check for click/typer options - also handle suffix patterns
                is_option = other_dec.name in (
                    "click.option",
                    "click.argument",
                    "typer.Option",
                    "typer.Argument",
                ) or other_dec.name.endswith((".option", ".argument"))

                if is_option:
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
                    name=command_name,
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
