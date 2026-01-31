"""Rust framework pattern extractors.

This package contains extractors for Rust web frameworks:
- web_routes.py - Actix-web and Rocket route detection
"""

from magaldi_core.extractors.patterns.rust.web_routes import (
    detect_rust_http_routes,
)

__all__ = [
    "detect_rust_http_routes",
]
