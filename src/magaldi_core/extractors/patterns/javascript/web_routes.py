"""JavaScript/TypeScript web framework route patterns and detection.

Pattern definitions and detection functions for HTTP routes
in JS/TS web frameworks:
- NestJS (HTTP, WebSocket, Microservices, SSE)
- Hono
- Express
- Fastify
- Koa
"""

from __future__ import annotations

import re

from magaldi_core.extractors.types import DecoratorInfo, HttpRoute

# =============================================================================
# HTTP ROUTE PATTERNS
# =============================================================================

# Maps decorator name -> (HTTP method, framework)
# Use "*" for method when it needs to be extracted from args
# Use "WEBSOCKET" for WebSocket endpoints
# Use "MESSAGE" for microservice message patterns
# Use "EVENT" for microservice event patterns
# Use "SSE" for Server-Sent Events
_JAVASCRIPT_HTTP_ROUTE_PATTERNS: dict[str, tuple[str, str]] = {
    # NestJS HTTP patterns (decorators)
    "Get": ("GET", "nestjs"),
    "Post": ("POST", "nestjs"),
    "Put": ("PUT", "nestjs"),
    "Delete": ("DELETE", "nestjs"),
    "Patch": ("PATCH", "nestjs"),
    "Head": ("HEAD", "nestjs"),
    "Options": ("OPTIONS", "nestjs"),
    "All": ("*", "nestjs"),
    # NestJS WebSocket patterns
    "WebSocketGateway": ("WEBSOCKET", "nestjs"),
    "SubscribeMessage": ("WEBSOCKET", "nestjs"),
    # NestJS Microservices patterns
    "MessagePattern": ("MESSAGE", "nestjs"),
    "EventPattern": ("EVENT", "nestjs"),
    # NestJS Server-Sent Events
    "Sse": ("SSE", "nestjs"),
    # Hono patterns (method calls, similar to Express)
    "c.get": ("GET", "hono"),
    "c.post": ("POST", "hono"),
    "c.put": ("PUT", "hono"),
    "c.delete": ("DELETE", "hono"),
    "c.patch": ("PATCH", "hono"),
    "app.get": ("GET", "hono"),
    "app.post": ("POST", "hono"),
    "app.put": ("PUT", "hono"),
    "app.delete": ("DELETE", "hono"),
    "app.patch": ("PATCH", "hono"),
    "hono.get": ("GET", "hono"),
    "hono.post": ("POST", "hono"),
    "hono.put": ("PUT", "hono"),
    "hono.delete": ("DELETE", "hono"),
    "hono.patch": ("PATCH", "hono"),
    # Express patterns (method calls)
    # Note: Express uses method calls like app.get(), router.get()
    "router.get": ("GET", "express"),
    "router.post": ("POST", "express"),
    "router.put": ("PUT", "express"),
    "router.delete": ("DELETE", "express"),
    "router.patch": ("PATCH", "express"),
    "router.head": ("HEAD", "express"),
    "router.options": ("OPTIONS", "express"),
    "router.all": ("ALL", "express"),
    "express.get": ("GET", "express"),
    "express.post": ("POST", "express"),
    "express.put": ("PUT", "express"),
    "express.delete": ("DELETE", "express"),
    "express.patch": ("PATCH", "express"),
    # Fastify patterns (method calls)
    "fastify.get": ("GET", "fastify"),
    "fastify.post": ("POST", "fastify"),
    "fastify.put": ("PUT", "fastify"),
    "fastify.delete": ("DELETE", "fastify"),
    "fastify.patch": ("PATCH", "fastify"),
    "fastify.head": ("HEAD", "fastify"),
    "fastify.options": ("OPTIONS", "fastify"),
    "server.get": ("GET", "fastify"),
    "server.post": ("POST", "fastify"),
    "server.put": ("PUT", "fastify"),
    "server.delete": ("DELETE", "fastify"),
    "server.patch": ("PATCH", "fastify"),
    # Koa patterns (via koa-router)
    "koaRouter.get": ("GET", "koa"),
    "koaRouter.post": ("POST", "koa"),
    "koaRouter.put": ("PUT", "koa"),
    "koaRouter.delete": ("DELETE", "koa"),
    "koaRouter.patch": ("PATCH", "koa"),
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _extract_path_params(path: str) -> list[str]:
    """Extract path parameters from a route path."""
    params = []
    # Express/NestJS style: :param
    for match in re.finditer(r":(\w+)", path):
        params.append(match.group(1))
    # OpenAPI style: {param} (used by some frameworks)
    for match in re.finditer(r"\{(\w+)\}", path):
        params.append(match.group(1))
    return params


def _extract_path_from_args(args: str | None) -> str | None:
    """Extract path from decorator arguments.

    Handles both raw strings and parenthesized argument lists:
    - '"/users"' -> '/users'
    - '("/users")' -> '/users'
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


# =============================================================================
# DETECTION FUNCTION
# =============================================================================


def detect_javascript_http_routes(
    decorators: list[DecoratorInfo],
) -> list[HttpRoute]:
    """Detect HTTP routes from JavaScript/TypeScript decorator information.

    Args:
        decorators: List of decorator information extracted from a function.

    Returns:
        List of detected HTTP routes.
    """
    routes = []

    for dec in decorators:
        if dec.name in _JAVASCRIPT_HTTP_ROUTE_PATTERNS:
            method, framework = _JAVASCRIPT_HTTP_ROUTE_PATTERNS[dec.name]
            path = _extract_path_from_args(dec.args)

            # NestJS @All() matches all HTTP methods
            if method == "*":
                method = "ALL"

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
