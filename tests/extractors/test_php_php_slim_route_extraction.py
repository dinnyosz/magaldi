"""Tests for PHP Slim framework route extraction."""

import pytest
from pathlib import Path

from magaldi_core.code_parser import PhpParser
from magaldi_core.change_detection import FileInfo
from magaldi_core.tree_sitter_manager import get_manager
from magaldi_core.extractors.patterns.slim import extract_slim_routes, extract_slim_route_groups


class TestPhpSlimRouteExtraction:
    """Tests for PHP Slim route extraction."""

    def test_extracts_imports(self):
        """Test that PHP use statements are extracted as imports."""
        code = "<?php\nuse Slim\\Factory\\AppFactory;\n\n$app = AppFactory::create();\n$app->run();\n"

        parser = PhpParser()
        file_info = FileInfo(
            relative_path="test.php",
            absolute_path=Path("/fake/test.php"),
            language="php",
        )

        elements = parser.parse(code, file_info, "scope", "repo", "main")

        # Check for import
        import_elem = next(
            (e for e in elements if e.element_type == "import"),
            None
        )
        assert import_elem is not None, "Expected import element not found"
        assert "AppFactory" in import_elem.name or "Slim" in import_elem.name

    def test_extracts_slim_routes(self):
        """Test that Slim HTTP routes are extracted correctly."""
        code = '''<?php
$app->get('/', function ($request, $response) {
    return $response;
});

$app->get('/users/{id}', function ($request, $response, $args) {
    return $response;
});

$app->post('/users', function ($request, $response) {
    return $response;
});

$app->delete('/books/{id}', function ($request, $response, $args) {
    return $response;
});
'''

        manager = get_manager()
        tree = manager.parse(code.encode('utf-8'), 'php')
        lines = code.split('\n')

        routes = extract_slim_routes(tree, lines)

        assert len(routes) == 4, f"Expected 4 routes, got {len(routes)}"

        # Check route methods
        methods = [r.method for r in routes]
        assert "GET" in methods
        assert "POST" in methods
        assert "DELETE" in methods

        # Check route paths
        paths = [r.path for r in routes]
        assert "/" in paths
        assert "/users/{id}" in paths
        assert "/users" in paths
        assert "/books/{id}" in paths

        # Check path params extraction
        users_id_route = next(r for r in routes if r.path == "/users/{id}")
        assert users_id_route.path_params == ["id"]

    def test_extracts_slim_map_routes(self):
        """Test that $app->map() multi-method routes are extracted."""
        code = '''<?php
$app->map(["GET", "POST"], "/items", function ($request, $response) {
    return $response;
});
'''

        manager = get_manager()
        tree = manager.parse(code.encode('utf-8'), 'php')
        lines = code.split('\n')

        routes = extract_slim_routes(tree, lines)

        assert len(routes) == 1
        assert routes[0].method == "GET"  # First method from array
        assert routes[0].path == "/items"

    def test_extracts_slim_any_routes(self):
        """Test that $app->any() catch-all routes are extracted."""
        code = '''<?php
$app->any("/fallback", function ($request, $response) {
    return $response;
});
'''

        manager = get_manager()
        tree = manager.parse(code.encode('utf-8'), 'php')
        lines = code.split('\n')

        routes = extract_slim_routes(tree, lines)

        assert len(routes) == 1
        assert routes[0].method == "ANY"
        assert routes[0].path == "/fallback"

    def test_extracts_slim_route_groups(self):
        """Test that Slim route groups are extracted with nested routes."""
        code = '''<?php
$app->group("/api", function ($group) {
    $group->get("/users", function ($request, $response) {
        return $response;
    });

    $group->post("/users", function ($request, $response) {
        return $response;
    });
});
'''

        manager = get_manager()
        tree = manager.parse(code.encode('utf-8'), 'php')
        lines = code.split('\n')

        groups = extract_slim_route_groups(tree, lines)

        assert len(groups) == 1
        assert groups[0].prefix == "/api"
        assert len(groups[0].routes) == 2

        # Check that routes have the prefix prepended
        paths = [r.path for r in groups[0].routes]
        assert "/api/users" in paths

    def test_route_framework_is_slim(self):
        """Test that detected routes have framework='slim'."""
        code = '''<?php
$app->get("/test", function ($request, $response) {
    return $response;
});
'''

        manager = get_manager()
        tree = manager.parse(code.encode('utf-8'), 'php')
        lines = code.split('\n')

        routes = extract_slim_routes(tree, lines)

        assert len(routes) == 1
        assert routes[0].framework == "slim"
