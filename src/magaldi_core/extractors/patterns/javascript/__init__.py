"""JavaScript/TypeScript framework pattern extractors.

This package contains detection functions for JS/TS web frameworks:
- web_routes.py - NestJS, Hono, Express
"""

from magaldi_core.extractors.patterns.javascript.web_routes import (
    detect_javascript_http_routes,
)

__all__ = [
    "detect_javascript_http_routes",
]
