"""Tests for extended code intelligence extraction."""

from magaldi_core.analysis.concurrency import detect_concurrency, detect_env_vars
from magaldi_core.analysis.metrics import (
    analyze_docstring,
    compute_code_metrics,
    compute_complexity,
)
from magaldi_core.analysis.security import detect_security_issues
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

    def test_detect_fastapi_websocket(self):
        """Test FastAPI WebSocket route detection."""
        decorators = [
            DecoratorInfo(
                name="router.websocket",
                args='"/ws"',
                full='router.websocket("/ws")',
            )
        ]
        routes = detect_http_routes(decorators, "python")
        assert len(routes) == 1
        assert routes[0].method == "WEBSOCKET"
        assert routes[0].path == "/ws"
        assert routes[0].framework == "fastapi"

    def test_detect_app_websocket(self):
        """Test FastAPI app.websocket detection."""
        decorators = [
            DecoratorInfo(
                name="app.websocket",
                args='"/ws/{channel}"',
                full='app.websocket("/ws/{channel}")',
            )
        ]
        routes = detect_http_routes(decorators, "python")
        assert len(routes) == 1
        assert routes[0].method == "WEBSOCKET"
        assert routes[0].path_params == ["channel"]

    def test_detect_litestar_websocket(self):
        """Test Litestar websocket decorator detection."""
        decorators = [
            DecoratorInfo(
                name="websocket",
                args='"/stream"',
                full='websocket("/stream")',
            )
        ]
        routes = detect_http_routes(decorators, "python")
        assert len(routes) == 1
        assert routes[0].method == "WEBSOCKET"
        assert routes[0].framework == "litestar"

    def test_detect_sanic_route(self):
        """Test Sanic route detection."""
        decorators = [
            DecoratorInfo(
                name="sanic.get",
                args='"/api/data"',
                full='sanic.get("/api/data")',
            )
        ]
        routes = detect_http_routes(decorators, "python")
        assert len(routes) == 1
        assert routes[0].method == "GET"
        assert routes[0].framework == "sanic"


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

    def test_detect_typer_callback(self):
        """Test Typer callback (root command) detection."""
        decorators = [DecoratorInfo(name="app.callback", args=None, full="app.callback()")]
        commands = detect_cli_commands(decorators, "main", "python")
        assert len(commands) == 1
        assert commands[0].framework == "typer"

    def test_detect_command_suffix_pattern(self):
        """Test detection of commands with custom group names like main.command."""
        decorators = [DecoratorInfo(name="web.command", args=None, full="web.command()")]
        commands = detect_cli_commands(decorators, "serve", "python")
        assert len(commands) == 1
        assert commands[0].framework == "click"


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

    def test_detect_builder_with_fluent_methods(self):
        """Detect builder with with_* methods."""
        class_info = {
            "name": "RequestBuilder",
            "attributes": ["_url", "_headers", "_body"],
            "methods": ["with_url", "with_header", "with_body", "send"],
        }
        patterns, confidence = detect_patterns(class_info, [], "python")
        assert "builder" in patterns

    def test_detect_builder_with_set_methods(self):
        """Detect builder with set_* chained methods."""
        class_info = {
            "name": "ConfigBuilder",
            "attributes": ["_config"],
            "methods": ["set_timeout", "set_retries", "set_base_url", "build"],
            "methods_return_self": ["set_timeout", "set_retries", "set_base_url"],
        }
        patterns, confidence = detect_patterns(class_info, [], "python")
        assert "builder" in patterns

    def test_detect_factory_with_from_methods(self):
        """Detect factory with from_* class methods."""
        class_info = {
            "name": "Parser",
            "attributes": [],
            "methods": ["from_string", "from_file", "from_dict", "parse"],
            "decorators": ["classmethod"],
        }
        patterns, confidence = detect_patterns(class_info, [], "python")
        assert "factory" in patterns

    def test_detect_factory_with_build_methods(self):
        """Detect factory with build_* methods."""
        class_info = {
            "name": "ConnectionManager",
            "attributes": ["_pool"],
            "methods": ["build_connection", "build_pool", "close"],
        }
        patterns, confidence = detect_patterns(class_info, [], "python")
        assert "factory" in patterns


# =============================================================================
# CODE METRICS TESTS
# =============================================================================


class TestComplexityAnalysis:
    """Tests for cyclomatic complexity computation."""

    def test_simple_function_complexity(self):
        """Simple function with no branches has complexity 1."""
        source = """def foo():
    return 1
"""
        manager = TreeSitterManager()
        tree = manager.parse(source.encode(), "python")
        complexity = compute_complexity(tree.root_node, "python")
        assert complexity["cyclomatic"] == 1
        assert complexity["branch_count"] == 0

    def test_if_statement_adds_complexity(self):
        """If statement adds to complexity."""
        source = """def foo(x):
    if x > 0:
        return 1
    return 0
"""
        manager = TreeSitterManager()
        tree = manager.parse(source.encode(), "python")
        complexity = compute_complexity(tree.root_node, "python")
        assert complexity["cyclomatic"] >= 2
        assert complexity["branch_count"] >= 1

    def test_multiple_branches(self):
        """Multiple if/elif statements increase complexity."""
        source = """def foo(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0
"""
        manager = TreeSitterManager()
        tree = manager.parse(source.encode(), "python")
        complexity = compute_complexity(tree.root_node, "python")
        assert complexity["cyclomatic"] >= 3

    def test_boolean_operators_add_complexity(self):
        """Boolean operators (and, or) add to complexity."""
        source = """def foo(x, y):
    if x > 0 and y > 0:
        return 1
    return 0
"""
        manager = TreeSitterManager()
        tree = manager.parse(source.encode(), "python")
        complexity = compute_complexity(tree.root_node, "python")
        # 1 (base) + 1 (if) + 1 (and) = 3
        assert complexity["cyclomatic"] >= 3

    def test_nesting_depth(self):
        """Nested control structures increase nesting depth."""
        source = """def foo(x, y):
    if x > 0:
        if y > 0:
            return 1
    return 0
"""
        manager = TreeSitterManager()
        tree = manager.parse(source.encode(), "python")
        complexity = compute_complexity(tree.root_node, "python")
        assert complexity["nesting_depth"] >= 2


class TestCodeMetrics:
    """Tests for code size metrics computation."""

    def test_basic_metrics(self):
        """Test basic code metrics."""
        code = """def foo(x, y):
    return x + y
"""
        params = [{"name": "x"}, {"name": "y"}]
        metrics = compute_code_metrics(code, params)
        assert metrics["line_count"] == 2  # Non-empty lines
        assert metrics["param_count"] == 2
        assert metrics["char_count"] == len(code)

    def test_empty_function(self):
        """Test metrics for empty function."""
        code = ""
        params = []
        metrics = compute_code_metrics(code, params)
        assert metrics["line_count"] == 0
        assert metrics["param_count"] == 0
        assert metrics["char_count"] == 0

    def test_metrics_with_no_params(self):
        """Test metrics for function with no parameters."""
        code = """def foo():
    return 42
"""
        params = None
        metrics = compute_code_metrics(code, params)
        assert metrics["param_count"] == 0


class TestDocstringQuality:
    """Tests for docstring quality analysis."""

    def test_no_docstring(self):
        """Function without docstring."""
        result = analyze_docstring(None, [], None)
        assert result["has_docstring"] is False
        assert result["coverage"] == 0.0

    def test_empty_docstring(self):
        """Empty docstring."""
        result = analyze_docstring("", [], None)
        assert result["has_docstring"] is False

    def test_simple_docstring(self):
        """Simple docstring without param docs."""
        result = analyze_docstring("This function does something.", [], None)
        assert result["has_docstring"] is True
        assert result["has_params"] is False
        assert result["coverage"] == 1.0  # No params or return to document

    def test_docstring_with_params(self):
        """Docstring with parameter documentation."""
        docstring = """Process data.

        Args:
            x: The input value.
            y: Another value.
        """
        params = [{"name": "x"}, {"name": "y"}]
        result = analyze_docstring(docstring, params, None)
        assert result["has_docstring"] is True
        assert result["has_params"] is True

    def test_docstring_with_return(self):
        """Docstring with return documentation."""
        docstring = """Process data.

        Returns:
            The processed result.
        """
        result = analyze_docstring(docstring, [], "str")
        assert result["has_docstring"] is True
        assert result["has_return"] is True

    def test_sphinx_style_docstring(self):
        """Sphinx-style docstring."""
        docstring = """Process data.

        :param x: The input.
        :returns: The output.
        """
        result = analyze_docstring(docstring, [{"name": "x"}], "str")
        assert result["has_params"] is True
        assert result["has_return"] is True
        assert result["coverage"] == 1.0


# =============================================================================
# SECURITY ANALYSIS TESTS
# =============================================================================


class TestSecurityDetection:
    """Tests for security issue detection."""

    def test_no_issues_in_safe_code(self):
        """Safe code should have no security issues."""
        code = """def add(x, y):
    return x + y
"""
        issues = detect_security_issues(code, "python")
        assert len(issues) == 0

    def test_detect_hardcoded_password(self):
        """Detect hardcoded password."""
        code = 'password = "secret123"'
        issues = detect_security_issues(code, "python")
        assert len(issues) >= 1
        assert any(i["kind"] == "hardcoded_secret" for i in issues)

    def test_detect_hardcoded_api_key(self):
        """Detect hardcoded API key."""
        code = 'api_key = "sk_live_abc123def456"'
        issues = detect_security_issues(code, "python")
        assert len(issues) >= 1
        assert any(i["kind"] == "hardcoded_secret" for i in issues)

    def test_detect_sql_injection_fstring(self):
        """Detect SQL injection with f-string."""
        code = '''cursor.execute(f"SELECT * FROM users WHERE id = {user_id}")'''
        issues = detect_security_issues(code, "python")
        assert len(issues) >= 1
        assert any(i["kind"] == "sql_injection" for i in issues)

    def test_detect_command_injection(self):
        """Detect command injection with os.system."""
        code = '''os.system(f"rm -rf {path}")'''
        issues = detect_security_issues(code, "python")
        assert len(issues) >= 1
        assert any(i["kind"] == "command_injection" for i in issues)

    def test_detect_subprocess_shell_true(self):
        """Detect subprocess with shell=True."""
        code = 'subprocess.run(cmd, shell=True)'
        issues = detect_security_issues(code, "python")
        assert len(issues) >= 1
        assert any(i["kind"] == "command_injection" for i in issues)

    def test_detect_eval(self):
        """Detect eval usage."""
        code = "result = eval(user_input)"
        issues = detect_security_issues(code, "python")
        assert len(issues) >= 1
        assert any(i["kind"] == "command_injection" for i in issues)

    def test_detect_pickle_load(self):
        """Detect pickle deserialization."""
        code = "data = pickle.loads(untrusted_data)"
        issues = detect_security_issues(code, "python")
        assert len(issues) >= 1
        assert any(i["kind"] == "deserialization" for i in issues)

    def test_skip_env_var_lookup(self):
        """Skip false positive for env var lookup."""
        code = 'password = os.environ.get("PASSWORD")'
        issues = detect_security_issues(code, "python")
        # Should not flag this as hardcoded secret
        hardcoded_issues = [i for i in issues if i["kind"] == "hardcoded_secret"]
        assert len(hardcoded_issues) == 0

    def test_skip_empty_string(self):
        """Skip false positive for empty string."""
        code = 'password = ""'
        issues = detect_security_issues(code, "python")
        hardcoded_issues = [i for i in issues if i["kind"] == "hardcoded_secret"]
        assert len(hardcoded_issues) == 0


# =============================================================================
# CONCURRENCY AND ENV VAR TESTS
# =============================================================================


class TestEnvVarDetection:
    """Tests for environment variable detection."""

    def test_detect_os_environ_get(self):
        """Detect os.environ.get()."""
        code = 'db_host = os.environ.get("DATABASE_HOST")'
        env_vars = detect_env_vars(None, code, "python")
        assert len(env_vars) >= 1
        assert any(e["name"] == "DATABASE_HOST" for e in env_vars)

    def test_detect_os_getenv(self):
        """Detect os.getenv()."""
        code = 'port = os.getenv("PORT")'
        env_vars = detect_env_vars(None, code, "python")
        assert len(env_vars) >= 1
        assert any(e["name"] == "PORT" for e in env_vars)

    def test_detect_os_environ_bracket(self):
        """Detect os.environ[] access."""
        code = 'secret = os.environ["SECRET_KEY"]'
        env_vars = detect_env_vars(None, code, "python")
        assert len(env_vars) >= 1
        assert any(e["name"] == "SECRET_KEY" for e in env_vars)

    def test_detect_multiple_env_vars(self):
        """Detect multiple env vars."""
        code = """
host = os.environ.get("HOST")
port = os.getenv("PORT")
"""
        env_vars = detect_env_vars(None, code, "python")
        assert len(env_vars) >= 2

    def test_no_env_vars(self):
        """No env vars in code."""
        code = 'name = "hello"'
        env_vars = detect_env_vars(None, code, "python")
        assert len(env_vars) == 0


class TestConcurrencyDetection:
    """Tests for concurrency pattern detection."""

    def test_detect_async_function(self):
        """Detect async function."""
        code = """async def fetch_data():
    await something()
"""
        result = detect_concurrency(None, code, [], True, "python")
        assert result["is_async"] is True
        assert "async" in result["patterns"]

    def test_detect_threading(self):
        """Detect threading usage."""
        code = """
thread = threading.Thread(target=worker)
thread.start()
"""
        result = detect_concurrency(None, code, [], False, "python")
        assert result["uses_threads"] is True
        assert "threading" in result["patterns"]

    def test_detect_lock_usage(self):
        """Detect lock usage."""
        code = """
lock = Lock()
with lock:
    shared_data += 1
"""
        result = detect_concurrency(None, code, [], False, "python")
        assert result["uses_locks"] is True
        assert "locking" in result["patterns"]

    def test_detect_multiprocessing(self):
        """Detect multiprocessing usage."""
        code = """
pool = multiprocessing.Pool(4)
results = pool.map(worker, items)
"""
        result = detect_concurrency(None, code, [], False, "python")
        assert result["uses_threads"] is True
        assert "multiprocessing" in result["patterns"]

    def test_no_concurrency(self):
        """No concurrency patterns."""
        code = """def add(x, y):
    return x + y
"""
        result = detect_concurrency(None, code, [], False, "python")
        assert result["is_async"] is False
        assert result["uses_threads"] is False
        assert result["uses_locks"] is False
        assert len(result["patterns"]) == 0

    def test_async_from_decorators(self):
        """Detect async from decorators."""
        code = "def handler(): pass"
        decorators = ["asynccontextmanager"]
        result = detect_concurrency(None, code, decorators, False, "python")
        assert "async" in result["patterns"]
