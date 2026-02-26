"""Tests for hierarchy extractors (CLI commands, HTTP routes)."""

from __future__ import annotations

from magaldi_core.analysis.hierarchy_extractors import (
    CliHierarchyExtractor,
    ElementInfo,
    RouteElementInfo,
    RouteHierarchyExtractor,
    elements_to_element_info,
    elements_to_route_info,
)
from magaldi_core.analysis.relationships import (
    ExternalRefType,
    RelationshipType,
)

# =============================================================================
# CLI Hierarchy Extractor Tests
# =============================================================================


class TestCliHierarchyExtractor:
    """Tests for CLI command hierarchy extraction."""

    def test_extract_simple_command(self):
        """Test extracting a simple command under a group."""
        elements = [
            # Define a group
            ElementInfo(
                hash_id="abc123",
                element_id="test:repo:main:cli.py:function:main:1",
                name="main",
                file_path="cli.py",
                line=1,
                decorators=["click.group"],
                decorator_details=[{"name": "click.group", "args": None}],
            ),
            # Define a command under the group
            ElementInfo(
                hash_id="def456",
                element_id="test:repo:main:cli.py:function:serve:10",
                name="serve",
                file_path="cli.py",
                line=10,
                decorators=["main.command"],
                decorator_details=[
                    {"name": "main.command", "args": '("serve")'}
                ],
            ),
        ]

        extractor = CliHierarchyExtractor(
            scope="test",
            repository="repo",
            username="main",
            cli_entry_point="myapp",
        )
        relationships, external_refs = extractor.extract(elements)

        # Should have one relationship: serve -> main
        assert len(relationships) == 1
        rel = relationships[0]
        assert rel.source_hash == "def456"  # serve
        assert rel.target_hash == "abc123"  # main
        assert rel.relationship_type == RelationshipType.BELONGS_TO_GROUP
        assert rel.details["command_name"] == "serve"
        assert rel.details["full_command"] == "myapp serve"
        assert rel.details["is_group"] is False

        # Should have one external ref for CLI entry point
        assert len(external_refs) == 1
        ref = external_refs[0]
        assert ref.ref_type == ExternalRefType.CLI_ENTRY_POINT
        assert ref.name == "myapp"

    def test_extract_nested_groups(self):
        """Test extracting nested group structure."""
        elements = [
            # Root group
            ElementInfo(
                hash_id="root_hash",
                element_id="test:repo:main:cli.py:function:main:1",
                name="main",
                file_path="cli.py",
                line=1,
                decorators=["click.group"],
                decorator_details=[{"name": "click.group", "args": None}],
            ),
            # Sub-group under main
            ElementInfo(
                hash_id="web_hash",
                element_id="test:repo:main:cli.py:function:web:10",
                name="web",
                file_path="cli.py",
                line=10,
                decorators=["main.group"],
                decorator_details=[{"name": "main.group", "args": '("web")'}],
            ),
            # Command under web
            ElementInfo(
                hash_id="serve_hash",
                element_id="test:repo:main:cli.py:function:serve:20",
                name="serve",
                file_path="cli.py",
                line=20,
                decorators=["web.command"],
                decorator_details=[
                    {"name": "web.command", "args": '("serve")'}
                ],
            ),
        ]

        extractor = CliHierarchyExtractor(
            scope="test",
            repository="repo",
            username="main",
            cli_entry_point="magaldi",
        )
        relationships, _ = extractor.extract(elements)

        # Should have two relationships: web -> main, serve -> web
        assert len(relationships) == 2

        # Find the serve -> web relationship
        serve_rel = next(r for r in relationships if r.source_hash == "serve_hash")
        assert serve_rel.target_hash == "web_hash"
        assert serve_rel.details["full_command"] == "magaldi web serve"
        assert serve_rel.details["group_path"] == ["main", "web"]

    def test_build_command_tree(self):
        """Test building command tree structure."""
        elements = [
            ElementInfo(
                hash_id="main_hash",
                element_id="test:repo:main:cli.py:function:main:1",
                name="main",
                file_path="cli.py",
                line=1,
                decorators=["click.group"],
                decorator_details=[{"name": "click.group", "args": None}],
            ),
            ElementInfo(
                hash_id="parse_hash",
                element_id="test:repo:main:cli.py:function:parse:10",
                name="parse",
                file_path="cli.py",
                line=10,
                decorators=["main.command"],
                decorator_details=[
                    {"name": "main.command", "args": '("parse")'}
                ],
            ),
        ]

        extractor = CliHierarchyExtractor(
            scope="test",
            repository="repo",
            username="main",
            cli_entry_point="myapp",
        )
        tree = extractor.build_command_tree(elements)

        assert tree.name == "myapp"
        assert tree.is_group is True
        assert len(tree.children) >= 1

    def test_extract_name_from_args_various_formats(self):
        """Test extracting command name from various decorator arg formats."""
        extractor = CliHierarchyExtractor(
            scope="test",
            repository="repo",
            username="main",
        )

        # Double quotes
        assert extractor._extract_name_from_args('("serve")') == "serve"
        # Single quotes
        assert extractor._extract_name_from_args("('parse')") == "parse"
        # With additional args
        assert extractor._extract_name_from_args('("run", help="Run server")') == "run"
        # None
        assert extractor._extract_name_from_args(None) is None
        # Empty
        assert extractor._extract_name_from_args("") is None


# =============================================================================
# Route Hierarchy Extractor Tests
# =============================================================================


class TestRouteHierarchyExtractor:
    """Tests for HTTP route hierarchy extraction."""

    def test_extract_simple_routes(self):
        """Test extracting routes from a single router file."""
        elements = [
            RouteElementInfo(
                hash_id="get_users_hash",
                element_id="test:repo:main:routes/users.py:function:get_users:10",
                name="get_users",
                file_path="routes/users.py",
                line=10,
                http_routes=[
                    {"method": "GET", "path": "/users", "path_params": [], "framework": "fastapi"}
                ],
            ),
            RouteElementInfo(
                hash_id="create_user_hash",
                element_id="test:repo:main:routes/users.py:function:create_user:20",
                name="create_user",
                file_path="routes/users.py",
                line=20,
                http_routes=[
                    {"method": "POST", "path": "/users", "path_params": [], "framework": "fastapi"}
                ],
            ),
        ]

        extractor = RouteHierarchyExtractor(
            scope="test",
            repository="repo",
            username="main",
            api_prefix="/api/v1",
        )
        relationships, external_refs = extractor.extract(elements)

        # Should have relationships for each route -> router
        assert len(relationships) == 2

        # Check first relationship
        rel = relationships[0]
        assert rel.relationship_type == RelationshipType.BELONGS_TO_ROUTER
        assert rel.details["method"] == "GET"
        assert rel.details["path"] == "/users"
        assert rel.details["full_url"] == "/api/v1/users"
        assert rel.details["router_name"] == "users"

        # Should have one external ref for the router
        assert len(external_refs) == 1
        ref = external_refs[0]
        assert ref.ref_type == ExternalRefType.HTTP_ROUTER
        assert ref.name == "users"
        assert ref.metadata["file_path"] == "routes/users.py"
        assert ref.metadata["route_count"] == 2
        assert ref.metadata["api_prefix"] == "/api/v1"

    def test_extract_routes_with_params(self):
        """Test extracting routes with path parameters."""
        elements = [
            RouteElementInfo(
                hash_id="get_user_hash",
                element_id="test:repo:main:routes/users.py:function:get_user:10",
                name="get_user",
                file_path="routes/users.py",
                line=10,
                http_routes=[
                    {
                        "method": "GET",
                        "path": "/users/{user_id}",
                        "path_params": ["user_id"],
                        "framework": "fastapi",
                    }
                ],
            ),
        ]

        extractor = RouteHierarchyExtractor(
            scope="test",
            repository="repo",
            username="main",
            api_prefix="/api/v2",
        )
        relationships, _ = extractor.extract(elements)

        assert len(relationships) == 1
        rel = relationships[0]
        assert rel.details["full_url"] == "/api/v2/users/{user_id}"
        assert rel.details["path_params"] == ["user_id"]

    def test_extract_router_name_from_path(self):
        """Test router name extraction from various file paths."""
        extractor = RouteHierarchyExtractor(
            scope="test",
            repository="repo",
            username="main",
        )

        assert extractor._extract_router_name("routes/users.py") == "users"
        assert extractor._extract_router_name("src/api/v1/admin.py") == "admin"
        assert extractor._extract_router_name("api/__init__.py") == "api"
        assert extractor._extract_router_name("routes.py") == "routes"

    def test_detect_common_prefix(self):
        """Test detecting common URL prefix for routes."""
        extractor = RouteHierarchyExtractor(
            scope="test",
            repository="repo",
            username="main",
        )

        # Routes with common prefix
        elements = [
            RouteElementInfo(
                hash_id="h1",
                element_id="e1",
                name="f1",
                file_path="routes.py",
                line=1,
                http_routes=[{"method": "GET", "path": "/users", "path_params": []}],
            ),
            RouteElementInfo(
                hash_id="h2",
                element_id="e2",
                name="f2",
                file_path="routes.py",
                line=2,
                http_routes=[{"method": "GET", "path": "/users/{id}", "path_params": ["id"]}],
            ),
        ]
        assert extractor._detect_common_prefix(elements) == "/users"

        # Routes without common prefix
        elements2 = [
            RouteElementInfo(
                hash_id="h1",
                element_id="e1",
                name="f1",
                file_path="routes.py",
                line=1,
                http_routes=[{"method": "GET", "path": "/users", "path_params": []}],
            ),
            RouteElementInfo(
                hash_id="h2",
                element_id="e2",
                name="f2",
                file_path="routes.py",
                line=2,
                http_routes=[{"method": "GET", "path": "/products", "path_params": []}],
            ),
        ]
        assert extractor._detect_common_prefix(elements2) == ""

    def test_multiple_routers(self):
        """Test extracting routes from multiple router files."""
        elements = [
            RouteElementInfo(
                hash_id="users_hash",
                element_id="test:repo:main:routes/users.py:function:get_users:10",
                name="get_users",
                file_path="routes/users.py",
                line=10,
                http_routes=[{"method": "GET", "path": "/users", "path_params": []}],
            ),
            RouteElementInfo(
                hash_id="products_hash",
                element_id="test:repo:main:routes/products.py:function:get_products:10",
                name="get_products",
                file_path="routes/products.py",
                line=10,
                http_routes=[{"method": "GET", "path": "/products", "path_params": []}],
            ),
        ]

        extractor = RouteHierarchyExtractor(
            scope="test",
            repository="repo",
            username="main",
            api_prefix="/api",
        )
        relationships, external_refs = extractor.extract(elements)

        # Should have 2 relationships (one per route)
        assert len(relationships) == 2

        # Should have 2 external refs (one per router)
        assert len(external_refs) == 2
        router_names = {ref.name for ref in external_refs}
        assert router_names == {"users", "products"}


# =============================================================================
# Helper Function Tests
# =============================================================================


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_elements_to_element_info(self):
        """Test converting ES documents to ElementInfo."""
        es_docs = [
            {
                "_source": {
                    "hash_id": "abc123",
                    "element_id": "test:repo:main:cli.py:function:main:1",
                    "name": "main",
                    "relative_path": "cli.py",
                    "line_start": 1,
                    "decorators": ["click.group"],
                    "decorator_details": [{"name": "click.group"}],
                    "summary": "Main CLI group",
                }
            },
            {
                "_source": {
                    "hash_id": "def456",
                    "element_id": "test:repo:main:utils.py:function:helper:10",
                    "name": "helper",
                    "relative_path": "utils.py",
                    "line_start": 10,
                    # No decorators - should be filtered out
                }
            },
        ]

        result = elements_to_element_info(es_docs)

        # Should only include the element with decorators
        assert len(result) == 1
        assert result[0].name == "main"
        assert result[0].summary == "Main CLI group"

    def test_elements_to_route_info(self):
        """Test converting ES documents to RouteElementInfo."""
        es_docs = [
            {
                "_source": {
                    "hash_id": "abc123",
                    "element_id": "test:repo:main:routes.py:function:get_users:1",
                    "name": "get_users",
                    "relative_path": "routes.py",
                    "line_start": 1,
                    "http_routes": [{"method": "GET", "path": "/users"}],
                    "summary": "Get all users",
                }
            },
            {
                "_source": {
                    "hash_id": "def456",
                    "element_id": "test:repo:main:utils.py:function:helper:10",
                    "name": "helper",
                    "relative_path": "utils.py",
                    "line_start": 10,
                    # No http_routes - should be filtered out
                }
            },
        ]

        result = elements_to_route_info(es_docs)

        # Should only include the element with http_routes
        assert len(result) == 1
        assert result[0].name == "get_users"
        assert result[0].http_routes == [{"method": "GET", "path": "/users"}]
