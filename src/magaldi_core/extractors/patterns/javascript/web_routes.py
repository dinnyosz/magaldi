"""JavaScript/TypeScript web framework route patterns.

Pattern definitions for detecting HTTP routes from decorator information
in JS/TS web frameworks:
- NestJS
- Hono
- Express (future)
"""

from __future__ import annotations

# =============================================================================
# HTTP ROUTE PATTERNS
# =============================================================================

# Maps decorator name -> (HTTP method, framework)
# Use "*" for method when it needs to be extracted from args
JS_HTTP_ROUTE_PATTERNS: dict[str, tuple[str, str]] = {
    # NestJS patterns (decorators)
    "Get": ("GET", "nestjs"),
    "Post": ("POST", "nestjs"),
    "Put": ("PUT", "nestjs"),
    "Delete": ("DELETE", "nestjs"),
    "Patch": ("PATCH", "nestjs"),
    "Head": ("HEAD", "nestjs"),
    "Options": ("OPTIONS", "nestjs"),
    "All": ("*", "nestjs"),
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
    # Express patterns (method calls)
    # Note: Express uses method calls like app.get(), router.get()
    # These are detected via call extraction, not decorators
    # Adding here for reference and future decorator-style detection
    "router.get": ("GET", "express"),
    "router.post": ("POST", "express"),
    "router.put": ("PUT", "express"),
    "router.delete": ("DELETE", "express"),
    "router.patch": ("PATCH", "express"),
}
