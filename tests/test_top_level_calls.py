"""Tests for top-level call extraction across all languages.

Tests that calls at module/file scope are extracted correctly while
calls inside function/class bodies are excluded.
"""

from __future__ import annotations

from magaldi_core.code_parser import CodeElement
from magaldi_core.extractors.bash import extract_top_level_bash_calls
from magaldi_core.extractors.javascript.call_extractor import (
    extract_top_level_javascript_calls,
)
from magaldi_core.extractors.php import extract_top_level_php_calls
from magaldi_core.extractors.python.call_extractor import (
    extract_top_level_python_calls,
)
from magaldi_core.extractors.rust import extract_top_level_rust_calls
from magaldi_core.parsers.base import Call
from magaldi_core.tree_sitter_manager import get_manager
from shared.ai.embedding import (
    InMemoryEmbeddingStore,
    build_summary_embedding_text,
)


def _parse(code: str, language: str):
    manager = get_manager()
    return manager.parse(code.encode("utf-8"), language)


# =============================================================================
# PYTHON TOP-LEVEL CALLS
# =============================================================================


class TestPythonTopLevelCalls:
    def test_extract_top_level_calls(self):
        code = """\
import logging
logging.basicConfig(level=logging.DEBUG)
app = Flask(__name__)
setup()
"""
        tree = _parse(code, "python")
        calls = extract_top_level_python_calls(tree)

        names = [(c.name, c.receiver) for c in calls]
        assert ("basicConfig", "logging") in names
        assert ("Flask", None) in names
        assert ("setup", None) in names

    def test_top_level_skips_function_bodies(self):
        code = """\
setup()

def my_func():
    inner_call()
    obj.method()
"""
        tree = _parse(code, "python")
        calls = extract_top_level_python_calls(tree)

        names = [c.name for c in calls]
        assert "setup" in names
        assert "inner_call" not in names
        assert "method" not in names

    def test_top_level_skips_class_bodies(self):
        code = """\
setup()

class MyClass:
    x = some_func()
    def method(self):
        self.call()
"""
        tree = _parse(code, "python")
        calls = extract_top_level_python_calls(tree)

        names = [c.name for c in calls]
        assert "setup" in names
        assert "some_func" not in names
        assert "call" not in names

    def test_top_level_in_if_main(self):
        code = """\
if __name__ == "__main__":
    main()
    app.run(debug=True)
"""
        tree = _parse(code, "python")
        calls = extract_top_level_python_calls(tree)

        names = [(c.name, c.receiver) for c in calls]
        assert ("main", None) in names
        assert ("run", "app") in names

    def test_top_level_empty(self):
        code = """\
def foo():
    bar()

class MyClass:
    pass
"""
        tree = _parse(code, "python")
        calls = extract_top_level_python_calls(tree)
        assert calls == []

    def test_top_level_method_calls(self):
        code = """\
app = Flask(__name__)
app.register_blueprint(api_bp)
db.init_app(app)
"""
        tree = _parse(code, "python")
        calls = extract_top_level_python_calls(tree)

        names = [(c.name, c.receiver) for c in calls]
        assert ("Flask", None) in names
        assert ("register_blueprint", "app") in names
        assert ("init_app", "db") in names

    def test_top_level_decorated_definition_skipped(self):
        code = """\
setup()

@decorator
def decorated_func():
    inner()
"""
        tree = _parse(code, "python")
        calls = extract_top_level_python_calls(tree)

        names = [c.name for c in calls]
        assert "setup" in names
        assert "inner" not in names

    def test_deduplication(self):
        code = """\
setup()
setup()
"""
        tree = _parse(code, "python")
        calls = extract_top_level_python_calls(tree)
        # Each call is on a different line, so both should be kept
        setup_calls = [c for c in calls if c.name == "setup"]
        assert len(setup_calls) == 2


# =============================================================================
# JAVASCRIPT TOP-LEVEL CALLS
# =============================================================================


class TestJavaScriptTopLevelCalls:
    def test_extract_top_level_calls(self):
        code = """\
const app = express();
app.use(cors());
configure();
"""
        tree = _parse(code, "javascript")
        calls = extract_top_level_javascript_calls(tree)

        names = [(c.name, c.receiver) for c in calls]
        assert ("express", None) in names
        assert ("use", "app") in names
        assert ("cors", None) in names
        assert ("configure", None) in names

    def test_top_level_skips_function_bodies(self):
        code = """\
setup();

function myFunc() {
    innerCall();
}

const arrow = () => {
    arrowCall();
};
"""
        tree = _parse(code, "javascript")
        calls = extract_top_level_javascript_calls(tree)

        names = [c.name for c in calls]
        assert "setup" in names
        assert "innerCall" not in names
        assert "arrowCall" not in names

    def test_top_level_skips_class_bodies(self):
        code = """\
configure();

class MyClass {
    method() {
        this.call();
    }
}
"""
        tree = _parse(code, "javascript")
        calls = extract_top_level_javascript_calls(tree)

        names = [c.name for c in calls]
        assert "configure" in names
        assert "call" not in names

    def test_top_level_empty(self):
        code = """\
function foo() {
    bar();
}

class MyClass {}
"""
        tree = _parse(code, "javascript")
        calls = extract_top_level_javascript_calls(tree)
        assert calls == []

    def test_typescript_top_level_calls(self):
        code = """\
const app = createApp();
app.listen(3000);
"""
        tree = _parse(code, "typescript")
        calls = extract_top_level_javascript_calls(tree)

        names = [(c.name, c.receiver) for c in calls]
        assert ("createApp", None) in names
        assert ("listen", "app") in names


# =============================================================================
# PHP TOP-LEVEL CALLS
# =============================================================================


class TestPhpTopLevelCalls:
    def test_extract_top_level_calls(self):
        code = """\
<?php
session_start();
$app = new Application();
configure();
"""
        tree = _parse(code, "php")
        calls = extract_top_level_php_calls(tree)

        names = [(c.name, c.receiver) for c in calls]
        assert ("session_start", None) in names
        assert ("configure", None) in names

    def test_top_level_skips_function_bodies(self):
        code = """\
<?php
setup();

function myFunc() {
    innerCall();
}
"""
        tree = _parse(code, "php")
        calls = extract_top_level_php_calls(tree)

        names = [c.name for c in calls]
        assert "setup" in names
        assert "innerCall" not in names

    def test_top_level_skips_class_bodies(self):
        code = """\
<?php
setup();

class MyClass {
    public function doStuff() {
        innerCall();
    }
}
"""
        tree = _parse(code, "php")
        calls = extract_top_level_php_calls(tree)

        names = [c.name for c in calls]
        assert "setup" in names
        assert "innerCall" not in names
        assert "doStuff" not in names

    def test_top_level_empty(self):
        code = """\
<?php
function foo() {
    bar();
}

class MyClass {}
"""
        tree = _parse(code, "php")
        calls = extract_top_level_php_calls(tree)
        assert calls == []

    def test_top_level_method_call(self):
        code = """\
<?php
$router->get('/users', 'UserController@index');
"""
        tree = _parse(code, "php")
        calls = extract_top_level_php_calls(tree)

        names = [(c.name, c.receiver) for c in calls]
        assert ("get", "router") in names


# =============================================================================
# RUST TOP-LEVEL CALLS
# =============================================================================


class TestRustTopLevelCalls:
    def test_extract_top_level_calls(self):
        # Rust top-level calls are rare but can exist in macros / static init
        # The most common pattern is in fn main() but that's inside a function.
        # For testing, we use static initializers.
        code = """\
static X: i32 = compute_value();
"""
        tree = _parse(code, "rust")
        calls = extract_top_level_rust_calls(tree)

        names = [c.name for c in calls]
        assert "compute_value" in names

    def test_top_level_skips_function_bodies(self):
        code = """\
fn main() {
    setup();
    app.run();
}
"""
        tree = _parse(code, "rust")
        calls = extract_top_level_rust_calls(tree)
        assert calls == []

    def test_top_level_skips_impl_bodies(self):
        code = """\
impl MyStruct {
    fn new() -> Self {
        init();
        Self {}
    }
}
"""
        tree = _parse(code, "rust")
        calls = extract_top_level_rust_calls(tree)
        assert calls == []

    def test_top_level_empty(self):
        code = """\
fn foo() {
    bar();
}

struct MyStruct {
    x: i32,
}
"""
        tree = _parse(code, "rust")
        calls = extract_top_level_rust_calls(tree)
        assert calls == []


# =============================================================================
# BASH TOP-LEVEL CALLS
# =============================================================================


class TestBashTopLevelCalls:
    def test_extract_top_level_calls(self):
        code = """\
#!/bin/bash
setup_env
deploy_app
"""
        tree = _parse(code, "bash")
        calls = extract_top_level_bash_calls(tree)

        names = [c.name for c in calls]
        assert "setup_env" in names
        assert "deploy_app" in names

    def test_top_level_skips_function_bodies(self):
        code = """\
setup_env

deploy() {
    kubectl apply -f manifests/
    notify_slack
}
"""
        tree = _parse(code, "bash")
        calls = extract_top_level_bash_calls(tree)

        names = [c.name for c in calls]
        assert "setup_env" in names
        assert "kubectl" not in names
        assert "notify_slack" not in names

    def test_top_level_skips_builtins(self):
        code = """\
#!/bin/bash
echo "hello"
cd /tmp
setup_env
"""
        tree = _parse(code, "bash")
        calls = extract_top_level_bash_calls(tree)

        names = [c.name for c in calls]
        assert "setup_env" in names
        assert "echo" not in names
        assert "cd" not in names

    def test_top_level_empty(self):
        code = """\
deploy() {
    kubectl apply
}
"""
        tree = _parse(code, "bash")
        calls = extract_top_level_bash_calls(tree)
        assert calls == []

    def test_top_level_skips_source_imports(self):
        code = """\
source ./lib.sh
. ./utils.sh
setup_env
"""
        tree = _parse(code, "bash")
        calls = extract_top_level_bash_calls(tree)

        names = [c.name for c in calls]
        assert "setup_env" in names
        assert "source" not in names


# =============================================================================
# EMBEDDING TEXT WITH FILE CALLS
# =============================================================================


class TestFileEmbeddingWithCalls:
    def test_file_element_includes_top_level_calls(self):
        file_element = CodeElement(
            element_id="scope:repo:main:src/app.py:file:app.py:1",
            scope="scope",
            repository="repo",
            username="main",
            relative_path="src/app.py",
            element_type="file",
            name="app.py",
            language="python",
            line_start=1,
            line_end=100,
            level=0,
            calls=[
                Call(name="Flask", receiver=None, line=3),
                Call(name="register_blueprint", receiver="app", line=5),
                Call(name="run", receiver="app", line=10),
            ],
        )

        store = InMemoryEmbeddingStore()
        store.store_element(file_element)
        store.store_summary(file_element.element_id, "Flask web application module.")

        text = build_summary_embedding_text(file_element, store)

        assert "Calls: Flask, app.register_blueprint, app.run" in text

    def test_file_element_no_calls(self):
        file_element = CodeElement(
            element_id="scope:repo:main:src/utils.py:file:utils.py:1",
            scope="scope",
            repository="repo",
            username="main",
            relative_path="src/utils.py",
            element_type="file",
            name="utils.py",
            language="python",
            line_start=1,
            line_end=50,
            level=0,
        )

        store = InMemoryEmbeddingStore()
        store.store_element(file_element)
        store.store_summary(file_element.element_id, "Utility functions.")

        text = build_summary_embedding_text(file_element, store)

        assert "Calls:" not in text
        assert "Utility functions" in text

    def test_file_element_calls_deduplicated(self):
        file_element = CodeElement(
            element_id="scope:repo:main:src/app.py:file:app.py:1",
            scope="scope",
            repository="repo",
            username="main",
            relative_path="src/app.py",
            element_type="file",
            name="app.py",
            language="python",
            line_start=1,
            line_end=100,
            level=0,
            calls=[
                Call(name="setup", receiver=None, line=1),
                Call(name="setup", receiver=None, line=5),  # Duplicate name
                Call(name="run", receiver="app", line=10),
            ],
        )

        store = InMemoryEmbeddingStore()
        store.store_element(file_element)
        store.store_summary(file_element.element_id, "App module.")

        text = build_summary_embedding_text(file_element, store)

        # Should deduplicate "setup" — only one occurrence
        assert "Calls: setup, app.run" in text

    def test_file_element_empty_calls_list(self):
        file_element = CodeElement(
            element_id="scope:repo:main:src/app.py:file:app.py:1",
            scope="scope",
            repository="repo",
            username="main",
            relative_path="src/app.py",
            element_type="file",
            name="app.py",
            language="python",
            line_start=1,
            line_end=100,
            level=0,
            calls=[],
        )

        store = InMemoryEmbeddingStore()
        store.store_element(file_element)
        store.store_summary(file_element.element_id, "App module.")

        text = build_summary_embedding_text(file_element, store)

        assert "Calls:" not in text
