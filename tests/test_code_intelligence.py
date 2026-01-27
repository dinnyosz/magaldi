"""Tests for extended code intelligence extraction."""

from magaldi_core.tree_sitter_manager import (
    Comment,
    DecoratorInfo,
    ExtractedCall,
    TreeSitterManager,
    analyze_purity,
    associate_comments,
    detect_cli_commands,
    detect_http_routes,
    detect_patterns,
    detect_public_api,
    extract_comments,
    extract_section_markers,
    extract_side_effects,
    extract_todos,
    extract_type_annotations,
)


class TestTodoExtraction:
    """Tests for TODO comment extraction."""

    def test_extract_simple_todo(self):
        source = "# TODO: fix this bug"
        todos = extract_todos(source)
        assert len(todos) == 1
        assert todos[0].kind == "TODO"
        assert todos[0].text == "fix this bug"
        assert todos[0].line == 1

    def test_extract_todo_with_assignee(self):
        source = "# TODO(alice): review this code"
        todos = extract_todos(source)
        assert len(todos) == 1
        assert todos[0].assignee == "alice"
        assert todos[0].text == "review this code"

    def test_extract_todo_with_issue_ref(self):
        source = "# TODO #123: implement feature"
        todos = extract_todos(source)
        assert len(todos) == 1
        assert todos[0].issue_ref == "#123"

    def test_extract_fixme(self):
        source = "# FIXME: memory leak here"
        todos = extract_todos(source)
        assert len(todos) == 1
        assert todos[0].kind == "FIXME"

    def test_extract_multiple_todos(self):
        source = """# TODO: first thing
def foo():
    # FIXME: second thing
    pass
"""
        todos = extract_todos(source)
        assert len(todos) == 2
        assert todos[0].line == 1
        assert todos[1].line == 3

    def test_extract_todo_with_priority(self):
        source = "# TODO!!: urgent fix needed"
        todos = extract_todos(source)
        assert len(todos) == 1
        assert todos[0].priority == "high"


class TestSectionMarkerExtraction:
    """Tests for section marker extraction."""

    def test_extract_equals_style(self):
        source = "# === HELPERS ==="
        markers = extract_section_markers(source)
        assert len(markers) == 1
        assert markers[0].label == "HELPERS"
        assert markers[0].style == "equals"

    def test_extract_dashes_style(self):
        source = "# --- PRIVATE METHODS ---"
        markers = extract_section_markers(source)
        assert len(markers) == 1
        assert markers[0].label == "PRIVATE METHODS"
        assert markers[0].style == "dashes"

    def test_extract_multiple_markers(self):
        source = """# === IMPORTS ===
import os

# === HELPERS ===
def helper():
    pass
"""
        markers = extract_section_markers(source)
        assert len(markers) == 2
        assert markers[0].label == "IMPORTS"
        assert markers[1].label == "HELPERS"


class TestCommentExtraction:
    """Tests for comment extraction and association."""

    def test_extract_inline_comment(self):
        source = "x = 1  # set x to one"
        comments = extract_comments(source)
        assert len(comments) == 1
        assert comments[0].kind == "inline"
        assert "set x to one" in comments[0].text

    def test_extract_block_comment(self):
        source = "# This is a block comment\ndef foo(): pass"
        comments = extract_comments(source)
        assert len(comments) == 1
        assert comments[0].kind == "block"

    def test_associate_comment_above(self):
        comments = [Comment(text="Helper function", line=5, kind="block", position="above")]
        element_line = 6
        associated = associate_comments(element_line, comments, max_distance=3)
        assert len(associated) == 1

    def test_no_associate_distant_comment(self):
        comments = [Comment(text="Far away", line=1, kind="block", position="above")]
        element_line = 10
        associated = associate_comments(element_line, comments, max_distance=3)
        assert len(associated) == 0


class TestPurityAnalysis:
    """Tests for function purity analysis."""

    def test_pure_function(self):
        calls = []
        mutations = []
        purity = analyze_purity(calls, mutations, "python")
        assert purity.level == "pure"
        assert purity.confidence == "high"

    def test_console_impure(self):
        calls = [ExtractedCall(name="print", receiver=None, line=5)]
        mutations = []
        purity = analyze_purity(calls, mutations, "python")
        assert purity.level == "mutates_external"
        assert "print" in purity.reasons[0]

    def test_self_mutation(self):
        calls = []
        mutations = ["self.cache"]
        purity = analyze_purity(calls, mutations, "python")
        assert purity.level == "mutates_self"

    def test_file_io_impure(self):
        calls = [ExtractedCall(name="open", receiver=None, line=10)]
        mutations = []
        purity = analyze_purity(calls, mutations, "python")
        assert purity.level == "mutates_external"
        assert "io_file" in str(purity.reasons)

    def test_extract_side_effects(self):
        calls = [ExtractedCall(name="print", receiver=None, line=5)]
        mutations = ["self.value"]
        effects = extract_side_effects(calls, mutations, "python")
        assert len(effects) == 2
        effect_kinds = [e.kind for e in effects]
        assert "console" in effect_kinds
        assert "state_mutation" in effect_kinds


class TestTypeAnnotationExtraction:
    """Tests for type annotation extraction."""

    def test_extract_parameter_types(self):
        source = "def foo(x: int, y: str) -> bool: pass"
        manager = TreeSitterManager()
        tree = manager.parse(source.encode(), "python")
        annotations = extract_type_annotations(tree.root_node, "python")

        param_types = [a for a in annotations if a.kind == "parameter"]
        assert len(param_types) == 2
        assert any(a.name == "int" and a.location == "param:x" for a in param_types)

    def test_extract_return_type(self):
        source = "def foo() -> str: pass"
        manager = TreeSitterManager()
        tree = manager.parse(source.encode(), "python")
        annotations = extract_type_annotations(tree.root_node, "python")

        return_types = [a for a in annotations if a.kind == "return"]
        assert len(return_types) == 1
        assert return_types[0].name == "str"

    def test_extract_generic_types(self):
        source = "def foo(items: List[str]) -> Dict[str, int]: pass"
        manager = TreeSitterManager()
        tree = manager.parse(source.encode(), "python")
        annotations = extract_type_annotations(tree.root_node, "python")

        list_type = next((a for a in annotations if "List" in a.name), None)
        assert list_type is not None
        assert list_type.generic_args == ["str"]


class TestHttpRouteDetection:
    """Tests for HTTP route detection."""

    def test_detect_fastapi_route(self):
        decorators = [
            DecoratorInfo(
                name="router.get", args='"/users/{id}"', full='router.get("/users/{id}")'
            )
        ]
        routes = detect_http_routes(decorators, "python")
        assert len(routes) == 1
        assert routes[0].method == "GET"
        assert routes[0].path == "/users/{id}"
        assert routes[0].path_params == ["id"]
        assert routes[0].framework == "fastapi"

    def test_detect_flask_route(self):
        decorators = [
            DecoratorInfo(
                name="app.route",
                args='"/api/items", methods=["POST"]',
                full='app.route("/api/items", methods=["POST"])',
            )
        ]
        routes = detect_http_routes(decorators, "python")
        assert len(routes) == 1
        assert routes[0].method == "POST"
        assert routes[0].path == "/api/items"

    def test_no_route_decorators(self):
        decorators = [DecoratorInfo(name="staticmethod", args=None, full="staticmethod")]
        routes = detect_http_routes(decorators, "python")
        assert len(routes) == 0

    def test_detect_fastapi_post(self):
        decorators = [
            DecoratorInfo(
                name="router.post", args='"/items"', full='router.post("/items")'
            )
        ]
        routes = detect_http_routes(decorators, "python")
        assert len(routes) == 1
        assert routes[0].method == "POST"
        assert routes[0].framework == "fastapi"

    def test_detect_flask_route_default_get(self):
        decorators = [
            DecoratorInfo(name="app.route", args='"/index"', full='app.route("/index")')
        ]
        routes = detect_http_routes(decorators, "python")
        assert len(routes) == 1
        assert routes[0].method == "GET"
        assert routes[0].framework == "flask"

    def test_detect_multiple_path_params(self):
        decorators = [
            DecoratorInfo(
                name="router.get",
                args='"/users/{user_id}/posts/{post_id}"',
                full='router.get("/users/{user_id}/posts/{post_id}")',
            )
        ]
        routes = detect_http_routes(decorators, "python")
        assert len(routes) == 1
        assert routes[0].path_params == ["user_id", "post_id"]

    def test_detect_flask_path_params(self):
        decorators = [
            DecoratorInfo(
                name="app.route",
                args='"/users/<user_id>"',
                full='app.route("/users/<user_id>")',
            )
        ]
        routes = detect_http_routes(decorators, "python")
        assert len(routes) == 1
        assert routes[0].path_params == ["user_id"]


class TestCliCommandDetection:
    """Tests for CLI command detection."""

    def test_detect_click_command(self):
        decorators = [DecoratorInfo(name="click.command", args=None, full="click.command()")]
        commands = detect_cli_commands(decorators, "parse", "python")
        assert len(commands) == 1
        assert commands[0].name == "parse"
        assert commands[0].framework == "click"

    def test_detect_typer_command(self):
        decorators = [DecoratorInfo(name="app.command", args=None, full="app.command()")]
        commands = detect_cli_commands(decorators, "run", "python")
        assert len(commands) == 1
        assert commands[0].framework == "typer"

    def test_no_cli_decorators(self):
        decorators = [DecoratorInfo(name="property", args=None, full="property")]
        commands = detect_cli_commands(decorators, "getter", "python")
        assert len(commands) == 0

    def test_detect_click_group(self):
        decorators = [DecoratorInfo(name="click.group", args=None, full="click.group()")]
        commands = detect_cli_commands(decorators, "main", "python")
        assert len(commands) == 1
        assert commands[0].framework == "click"

    def test_detect_command_with_options(self):
        decorators = [
            DecoratorInfo(name="click.command", args=None, full="click.command()"),
            DecoratorInfo(
                name="click.option",
                args='"--verbose", "-v", is_flag=True',
                full='click.option("--verbose", "-v", is_flag=True)',
            ),
            DecoratorInfo(
                name="click.argument",
                args='"path", required=True',
                full='click.argument("path", required=True)',
            ),
        ]
        commands = detect_cli_commands(decorators, "process", "python")
        assert len(commands) == 1
        assert len(commands[0].options) == 2
        # Check that required option is detected
        option_names = [opt["name"] for opt in commands[0].options]
        assert "--verbose" in option_names or "path" in option_names


class TestPublicApiDetection:
    """Tests for public API detection."""

    def test_public_function(self):
        assert detect_public_api("process_data", [], "public", "python") is True

    def test_private_function(self):
        assert detect_public_api("_helper", [], "private", "python") is False

    def test_dunder_method(self):
        assert detect_public_api("__init__", [], "public", "python") is False

    def test_api_decorator(self):
        decorators = [DecoratorInfo(name="api_endpoint", args=None, full="api_endpoint")]
        assert detect_public_api("handler", decorators, "public", "python") is True

    def test_route_is_public_api(self):
        decorators = [
            DecoratorInfo(name="router.get", args='"/users"', full='router.get("/users")')
        ]
        assert detect_public_api("get_users", decorators, "public", "python") is True

    def test_cli_command_is_public_api(self):
        decorators = [DecoratorInfo(name="click.command", args=None, full="click.command()")]
        assert detect_public_api("run_task", decorators, "public", "python") is True

    def test_protected_function(self):
        assert detect_public_api("_internal_helper", [], "protected", "python") is False

    def test_public_with_export_decorator(self):
        decorators = [DecoratorInfo(name="export", args=None, full="export")]
        assert detect_public_api("exported_func", decorators, "public", "python") is True


class TestPatternDetection:
    """Tests for design pattern detection."""

    def test_detect_singleton(self):
        class_info = {
            "name": "DatabaseConnection",
            "attributes": ["_instance"],
            "methods": ["get_instance", "__new__"],
            "method_returns_self": True,
        }
        patterns, confidence = detect_patterns(class_info, [], "python")
        assert "singleton" in patterns
        assert confidence.get("singleton", 0) >= 0.6

    def test_detect_builder(self):
        class_info = {
            "name": "QueryBuilder",
            "attributes": ["_query"],
            "methods": ["select", "where", "order_by", "build"],
            "methods_return_self": ["select", "where", "order_by"],
        }
        patterns, confidence = detect_patterns(class_info, [], "python")
        assert "builder" in patterns

    def test_detect_factory(self):
        class_info = {
            "name": "UserFactory",
            "attributes": [],
            "methods": ["create_admin", "create_guest"],
        }
        # Simulate calls that instantiate other classes
        calls = [
            ExtractedCall(name="AdminUser", receiver=None, line=10),
            ExtractedCall(name="GuestUser", receiver=None, line=15),
        ]
        patterns, confidence = detect_patterns(class_info, calls, "python")
        assert "factory" in patterns

    def test_detect_repository(self):
        class_info = {
            "name": "UserRepository",
            "attributes": ["_db"],
            "methods": ["find_by_id", "find_all", "save", "delete"],
        }
        patterns, confidence = detect_patterns(class_info, [], "python")
        assert "repository" in patterns

    def test_no_pattern(self):
        class_info = {
            "name": "SimpleClass",
            "attributes": ["value"],
            "methods": ["get_value", "set_value"],
        }
        patterns, confidence = detect_patterns(class_info, [], "python")
        assert len(patterns) == 0

    def test_detect_singleton_with_class_variable(self):
        """Detect singleton with class-level _instance variable."""
        class_info = {
            "name": "Logger",
            "attributes": [],  # No instance attributes
            "methods": ["__new__", "log", "error"],
            "class_variables": ["_instance"],
        }
        patterns, confidence = detect_patterns(class_info, [], "python")
        assert "singleton" in patterns

    def test_detect_singleton_with_instance_method(self):
        """Detect singleton with instance() class method."""
        class_info = {
            "name": "Configuration",
            "attributes": ["_settings"],
            "methods": ["instance", "get", "set"],
            "decorators": ["classmethod"],
        }
        patterns, confidence = detect_patterns(class_info, [], "python")
        assert "singleton" in patterns
