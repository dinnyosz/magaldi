"""API surface detection functions.

This module provides functions for detecting:
- HTTP routes (FastAPI, Flask, Express, NestJS)
- CLI commands (Click, Typer)
- Public API elements
- Design patterns (Singleton, Builder, Factory, Repository)
"""

from __future__ import annotations

from typing import Any

from magaldi_core.extractors.types import (
    CliCommand,
    DecoratorInfo,
    ExtractedCall,
    HttpRoute,
)
from magaldi_core.extractors.patterns.python import (
    detect_python_http_routes,
    detect_python_cli_commands,
)
from magaldi_core.extractors.patterns.javascript import (
    detect_javascript_http_routes,
)


# =============================================================================
# UNIFIED HTTP ROUTE DETECTION
# =============================================================================


def detect_http_routes(
    decorators: list[DecoratorInfo],
    language: str,
) -> list[HttpRoute]:
    """Detect HTTP routes from decorator information.

    Dispatches to the appropriate language-specific detection function.

    Args:
        decorators: List of decorator information extracted from a function.
        language: The programming language (e.g., "python", "javascript").

    Returns:
        List of detected HTTP routes.
    """
    if language == "python":
        return detect_python_http_routes(decorators)
    elif language in ("javascript", "typescript", "tsx"):
        return detect_javascript_http_routes(decorators)
    else:
        # For unknown languages, try both
        routes = detect_python_http_routes(decorators)
        if not routes:
            routes = detect_javascript_http_routes(decorators)
        return routes


# =============================================================================
# UNIFIED CLI COMMAND DETECTION
# =============================================================================


def detect_cli_commands(
    decorators: list[DecoratorInfo],
    function_name: str,
    language: str,
) -> list[CliCommand]:
    """Detect CLI commands from decorator information.

    Args:
        decorators: List of decorator information extracted from a function.
        function_name: Name of the decorated function.
        language: The programming language (e.g., "python").

    Returns:
        List of detected CLI commands.
    """
    if language == "python":
        return detect_python_cli_commands(decorators, function_name)
    # Future: add JavaScript CLI frameworks like Commander.js, Yargs
    return []


# =============================================================================
# PUBLIC API DETECTION
# =============================================================================

# Decorators that indicate public API
# Combines route and CLI decorators from all supported frameworks
_PUBLIC_API_DECORATORS = {
    # Generic markers
    "api_endpoint",
    "public",
    "export",
    "exposed",
    # FastAPI (including WebSocket)
    "router.get", "router.post", "router.put", "router.delete", "router.patch",
    "router.head", "router.options", "router.trace", "router.websocket",
    "app.get", "app.post", "app.put", "app.delete", "app.patch",
    "app.head", "app.options", "app.trace", "app.websocket",
    # Flask
    "app.route", "blueprint.route", "bp.route",
    # Django REST
    "api_view", "action",
    # Starlette
    "route", "websocket_route",
    # Litestar
    "get", "post", "put", "delete", "patch", "head", "websocket",
    # Quart (async Flask)
    "quart.route", "quart.websocket",
    # Sanic
    "sanic.route", "sanic.websocket",
    # NestJS HTTP
    "Get", "Post", "Put", "Delete", "Patch", "Head", "Options", "All",
    # NestJS WebSocket
    "WebSocketGateway", "SubscribeMessage",
    # NestJS Microservices
    "MessagePattern", "EventPattern",
    # NestJS SSE
    "Sse",
    # NestJS/Angular Dependency Injection
    "Injectable", "Inject", "Controller", "Module", "Component",
    "UseGuards", "UseInterceptors", "UsePipes",
    # Express/Fastify/Hono/Koa (method calls used as decorators)
    "router.get", "router.post", "router.put", "router.delete", "router.patch",
    "router.head", "router.options", "router.all",
    "express.get", "express.post", "express.put", "express.delete", "express.patch",
    "fastify.get", "fastify.post", "fastify.put", "fastify.delete", "fastify.patch",
    "hono.get", "hono.post", "hono.put", "hono.delete", "hono.patch",
    "c.get", "c.post", "c.put", "c.delete", "c.patch",
    "server.get", "server.post", "server.put", "server.delete", "server.patch",
    "koaRouter.get", "koaRouter.post", "koaRouter.put", "koaRouter.delete", "koaRouter.patch",
    # Click/Typer CLI (Python)
    "click.command", "click.group", "app.command", "typer.command", "app.callback",
    # Python Dependency Injection (FastAPI, dataclass, attrs)
    "dataclass", "attrs", "define", "frozen",
    # Symfony Console CLI (PHP)
    "AsCommand",
    # Symfony Route (PHP)
    "Route",
    # Rust Actix-web/Rocket routes
    "get", "post", "put", "delete", "patch", "head", "options", "trace", "route",
    "actix_web::main", "actix_rt::main", "launch", "rocket::launch",
    # Rust Tokio/test
    "tokio::main", "test", "tokio::test", "actix_rt::test",
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

    return min(score, 1.0)


def _detect_builder(class_info: dict[str, Any]) -> float:
    """Detect builder pattern."""
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

    return min(score, 1.0)


def _detect_factory(class_info: dict[str, Any], calls: list[ExtractedCall]) -> float:
    """Detect factory pattern."""
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
