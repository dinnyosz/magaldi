"""Tests for Rust web framework route detection (Actix-web, Rocket)."""

import pytest

from magaldi_core.extractors.types import DecoratorInfo
from magaldi_core.extractors.patterns.rust.web_routes import (
    detect_rust_http_routes,
    is_rust_entry_point,
)


class TestActixWebRoutes:
    """Test detection of Actix-web HTTP routes."""

    def test_detect_actix_get(self):
        """Test detection of #[get] attribute."""
        decorators = [
            DecoratorInfo(
                name="get",
                args='"/users/{id}"',
                full='#[get("/users/{id}")]',
            )
        ]
        routes = detect_rust_http_routes(decorators)
        assert len(routes) == 1
        assert routes[0].method == "GET"
        assert routes[0].path == "/users/{id}"
        assert routes[0].path_params == ["id"]
        assert routes[0].framework == "actix-web"

    def test_detect_actix_post(self):
        """Test detection of #[post] attribute."""
        decorators = [
            DecoratorInfo(
                name="post",
                args='"/users"',
                full='#[post("/users")]',
            )
        ]
        routes = detect_rust_http_routes(decorators)
        assert len(routes) == 1
        assert routes[0].method == "POST"
        assert routes[0].path == "/users"

    def test_detect_actix_delete(self):
        """Test detection of #[delete] attribute."""
        decorators = [
            DecoratorInfo(
                name="delete",
                args='"/users/{id}"',
                full='#[delete("/users/{id}")]',
            )
        ]
        routes = detect_rust_http_routes(decorators)
        assert len(routes) == 1
        assert routes[0].method == "DELETE"


class TestRocketRoutes:
    """Test detection of Rocket HTTP routes."""

    def test_detect_rocket_get(self):
        """Test detection of Rocket #[get] attribute."""
        decorators = [
            DecoratorInfo(
                name="get",
                args='"/<name>"',
                full='#[get("/<name>")]',
            )
        ]
        routes = detect_rust_http_routes(decorators)
        assert len(routes) == 1
        assert routes[0].method == "GET"
        assert routes[0].path == "/<name>"
        assert routes[0].path_params == ["name"]

    def test_detect_rocket_route_with_method(self):
        """Test detection of #[route] with explicit method."""
        decorators = [
            DecoratorInfo(
                name="route",
                args='"/custom", method = "POST"',
                full='#[route("/custom", method = "POST")]',
            )
        ]
        routes = detect_rust_http_routes(decorators)
        assert len(routes) == 1
        assert routes[0].method == "POST"
        assert routes[0].path == "/custom"


class TestRustEntryPoints:
    """Test detection of Rust entry point macros."""

    def test_actix_main_entry_point(self):
        """Test detection of #[actix_web::main]."""
        decorators = [
            DecoratorInfo(
                name="actix_web::main",
                args=None,
                full="#[actix_web::main]",
            )
        ]
        is_entry, framework = is_rust_entry_point(decorators)
        assert is_entry is True
        assert framework == "actix-web"

    def test_tokio_main_entry_point(self):
        """Test detection of #[tokio::main]."""
        decorators = [
            DecoratorInfo(
                name="tokio::main",
                args=None,
                full="#[tokio::main]",
            )
        ]
        is_entry, framework = is_rust_entry_point(decorators)
        assert is_entry is True
        assert framework == "tokio"

    def test_rocket_launch_entry_point(self):
        """Test detection of #[launch]."""
        decorators = [
            DecoratorInfo(
                name="launch",
                args=None,
                full="#[launch]",
            )
        ]
        is_entry, framework = is_rust_entry_point(decorators)
        assert is_entry is True
        assert framework == "rocket"

    def test_test_attribute(self):
        """Test detection of #[test]."""
        decorators = [
            DecoratorInfo(
                name="test",
                args=None,
                full="#[test]",
            )
        ]
        is_entry, framework = is_rust_entry_point(decorators)
        assert is_entry is True
        assert framework == "test"

    def test_tokio_test_attribute(self):
        """Test detection of #[tokio::test]."""
        decorators = [
            DecoratorInfo(
                name="tokio::test",
                args=None,
                full="#[tokio::test]",
            )
        ]
        is_entry, framework = is_rust_entry_point(decorators)
        assert is_entry is True
        assert framework == "tokio-test"


class TestNoRustRoutes:
    """Test that non-route attributes are not detected."""

    def test_derive_not_detected(self):
        """Test that #[derive] is not detected as route."""
        decorators = [
            DecoratorInfo(
                name="derive",
                args="Debug, Clone",
                full="#[derive(Debug, Clone)]",
            )
        ]
        routes = detect_rust_http_routes(decorators)
        assert len(routes) == 0

    def test_regular_function_not_entry_point(self):
        """Test that regular function without attributes is not entry point."""
        decorators = []
        is_entry, framework = is_rust_entry_point(decorators)
        assert is_entry is False
        assert framework is None

    def test_serde_not_entry_point(self):
        """Test that #[serde] is not an entry point."""
        decorators = [
            DecoratorInfo(
                name="serde",
                args="rename_all = \"camelCase\"",
                full='#[serde(rename_all = "camelCase")]',
            )
        ]
        is_entry, framework = is_rust_entry_point(decorators)
        assert is_entry is False
