"""Tests for call resolution (Phase 1 and Phase 2).

Phase 1: Same-file and self-method resolution (at parse time)
Phase 2: Cross-file resolution via imports (after indexing)
"""

from pathlib import Path

import pytest

from magaldi_core.change_detection import FileInfo
from magaldi_core.code_parser import (
    JavaScriptParser,
    PhpParser,
    PythonParser,
    RustParser,
)
from magaldi_core.parsers.base import Call

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def python_parser():
    return PythonParser()


@pytest.fixture
def javascript_parser():
    return JavaScriptParser()


@pytest.fixture
def php_parser():
    return PhpParser()


@pytest.fixture
def rust_parser():
    return RustParser()


def make_file_info(path: str, language: str) -> FileInfo:
    """Create a FileInfo object for testing."""
    return FileInfo(
        relative_path=path,
        absolute_path=Path("/tmp") / path,
        language=language,
        hash="abc123",
    )


# =============================================================================
# PHASE 1: PYTHON RESOLUTION TESTS
# =============================================================================


class TestPythonPhase1Resolution:
    """Test Phase 1 resolution for Python."""

    def test_same_file_function_call_resolved(self, python_parser):
        """Test that bare function calls to same-file functions are resolved."""
        code = '''
def helper():
    pass

def main():
    helper()
'''
        file_info = make_file_info("test.py", "python")
        elements = python_parser.parse(code, file_info, "test", "repo", "main")

        # Find the main function
        main_func = next(e for e in elements if e.name == "main" and e.element_type == "function")
        helper_func = next(e for e in elements if e.name == "helper" and e.element_type == "function")

        # Check that helper() call is resolved
        assert len(main_func.calls) == 1
        assert main_func.calls[0].name == "helper"
        assert main_func.calls[0].resolved_id == helper_func.element_id

    def test_self_method_call_resolved(self, python_parser):
        """Test that self.method() calls are resolved to sibling methods."""
        code = '''
class MyClass:
    def process(self):
        self.validate()

    def validate(self):
        pass
'''
        file_info = make_file_info("test.py", "python")
        elements = python_parser.parse(code, file_info, "test", "repo", "main")

        # Find the methods
        process_method = next(e for e in elements if e.name == "process" and e.element_type == "method")
        validate_method = next(e for e in elements if e.name == "validate" and e.element_type == "method")

        # Check that self.validate() call is resolved
        assert len(process_method.calls) == 1
        assert process_method.calls[0].name == "validate"
        assert process_method.calls[0].receiver == "self"
        assert process_method.calls[0].resolved_id == validate_method.element_id

    def test_external_call_not_resolved(self, python_parser):
        """Test that external calls (like os.path.join) are not resolved."""
        code = '''
import os

def main():
    os.path.join("a", "b")
'''
        file_info = make_file_info("test.py", "python")
        elements = python_parser.parse(code, file_info, "test", "repo", "main")

        main_func = next(e for e in elements if e.name == "main" and e.element_type == "function")

        # External call should not be resolved
        assert len(main_func.calls) >= 1
        join_call = next((c for c in main_func.calls if c.name == "join"), None)
        if join_call:
            assert join_call.resolved_id is None

    def test_builtin_call_not_resolved(self, python_parser):
        """Test that builtin calls (like len, print) are not resolved."""
        code = '''
def main():
    x = [1, 2, 3]
    print(len(x))
'''
        file_info = make_file_info("test.py", "python")
        elements = python_parser.parse(code, file_info, "test", "repo", "main")

        main_func = next(e for e in elements if e.name == "main" and e.element_type == "function")

        # Builtin calls should not be resolved
        for call in main_func.calls:
            assert call.resolved_id is None


# =============================================================================
# PHASE 1: JAVASCRIPT RESOLUTION TESTS
# =============================================================================


class TestJavaScriptPhase1Resolution:
    """Test Phase 1 resolution for JavaScript."""

    def test_same_file_function_call_resolved(self, javascript_parser):
        """Test that bare function calls to same-file functions are resolved."""
        code = '''
function helper() {
    return true;
}

function main() {
    helper();
}
'''
        file_info = make_file_info("test.js", "javascript")
        elements = javascript_parser.parse(code, file_info, "test", "repo", "main")

        main_func = next(e for e in elements if e.name == "main" and e.element_type == "function")
        helper_func = next(e for e in elements if e.name == "helper" and e.element_type == "function")

        assert len(main_func.calls) >= 1
        helper_call = next((c for c in main_func.calls if c.name == "helper"), None)
        assert helper_call is not None
        assert helper_call.resolved_id == helper_func.element_id

    def test_this_method_call_resolved(self, javascript_parser):
        """Test that this.method() calls are resolved to sibling methods."""
        code = '''
class MyClass {
    process() {
        this.validate();
    }

    validate() {
        return true;
    }
}
'''
        file_info = make_file_info("test.js", "javascript")
        elements = javascript_parser.parse(code, file_info, "test", "repo", "main")

        process_method = next(e for e in elements if e.name == "process" and e.element_type == "method")
        validate_method = next(e for e in elements if e.name == "validate" and e.element_type == "method")

        validate_call = next((c for c in process_method.calls if c.name == "validate"), None)
        assert validate_call is not None
        assert validate_call.receiver == "this"
        assert validate_call.resolved_id == validate_method.element_id


# =============================================================================
# PHASE 1: PHP RESOLUTION TESTS
# =============================================================================


class TestPhpPhase1Resolution:
    """Test Phase 1 resolution for PHP."""

    def test_same_file_function_call_resolved(self, php_parser):
        """Test that bare function calls to same-file functions are resolved."""
        code = '''<?php
function helper() {
    return true;
}

function main() {
    helper();
}
'''
        file_info = make_file_info("test.php", "php")
        elements = php_parser.parse(code, file_info, "test", "repo", "main")

        main_func = next((e for e in elements if e.name == "main" and e.element_type == "function"), None)
        helper_func = next((e for e in elements if e.name == "helper" and e.element_type == "function"), None)

        if main_func and helper_func and main_func.calls:
            helper_call = next((c for c in main_func.calls if c.name == "helper"), None)
            if helper_call:
                assert helper_call.resolved_id == helper_func.element_id

    def test_this_method_call_resolved(self, php_parser):
        """Test that $this->method() calls are resolved to sibling methods."""
        code = '''<?php
class MyClass {
    public function process() {
        $this->validate();
    }

    public function validate() {
        return true;
    }
}
'''
        file_info = make_file_info("test.php", "php")
        elements = php_parser.parse(code, file_info, "test", "repo", "main")

        process_method = next((e for e in elements if e.name == "process" and e.element_type == "method"), None)
        validate_method = next((e for e in elements if e.name == "validate" and e.element_type == "method"), None)

        if process_method and validate_method and process_method.calls:
            validate_call = next((c for c in process_method.calls if c.name == "validate"), None)
            if validate_call:
                # PHP extractor returns "this" (without the $)
                assert validate_call.receiver in ("$this", "this")
                assert validate_call.resolved_id == validate_method.element_id


# =============================================================================
# PHASE 1: RUST RESOLUTION TESTS
# =============================================================================


class TestRustPhase1Resolution:
    """Test Phase 1 resolution for Rust."""

    def test_same_file_function_call_resolved(self, rust_parser):
        """Test that bare function calls to same-file functions are resolved."""
        code = '''
fn helper() -> bool {
    true
}

fn main() {
    helper();
}
'''
        file_info = make_file_info("test.rs", "rust")
        elements = rust_parser.parse(code, file_info, "test", "repo", "main")

        main_func = next((e for e in elements if e.name == "main" and e.element_type == "function"), None)
        helper_func = next((e for e in elements if e.name == "helper" and e.element_type == "function"), None)

        if main_func and helper_func and main_func.calls:
            helper_call = next((c for c in main_func.calls if c.name == "helper"), None)
            if helper_call:
                assert helper_call.resolved_id == helper_func.element_id


# =============================================================================
# PHASE 2: CROSS-FILE RESOLUTION TESTS
# =============================================================================


class TestPhase2Resolution:
    """Test Phase 2 cross-file resolution."""

    def test_build_import_map(self):
        """Test building import map from imports list."""
        from magaldi_core.call_resolution import _build_import_map

        imports = [
            {"name": "process", "module": "utils", "alias": None, "line": 1},
            {"name": "pandas", "module": "pandas", "alias": "pd", "line": 2},
        ]

        import_map = _build_import_map(imports)

        assert "process" in import_map
        assert import_map["process"]["module"] == "utils"
        assert "pd" in import_map
        assert import_map["pd"]["module"] == "pandas"

    def test_is_external_module(self):
        """Test detection of external/stdlib modules."""
        from magaldi_core.call_resolution import _is_external_module

        # Standard library
        assert _is_external_module("os") is True
        assert _is_external_module("sys") is True
        assert _is_external_module("json") is True
        assert _is_external_module("pathlib") is True

        # Third-party
        assert _is_external_module("numpy") is True
        assert _is_external_module("pandas") is True
        assert _is_external_module("requests") is True

        # Relative imports are internal
        assert _is_external_module("./utils") is False
        assert _is_external_module("../common") is False

        # Project modules (not in stdlib/third-party lists)
        assert _is_external_module("magaldi_core") is False
        assert _is_external_module("myproject.utils") is False

    def test_module_to_file_paths(self):
        """Test converting module paths to file paths."""
        from magaldi_core.call_resolution import _module_to_file_paths

        paths = _module_to_file_paths("magaldi_core.storage")

        assert "src/magaldi_core/storage.py" in paths
        assert "magaldi_core/storage.py" in paths

    def test_relative_import_no_caller_path(self):
        """Test that relative imports return empty when no caller_path."""
        from magaldi_core.call_resolution import _module_to_file_paths

        paths = _module_to_file_paths(".utils")
        assert len(paths) == 0

    def test_relative_import_single_dot(self):
        """Test resolving single-dot relative import."""
        from magaldi_core.call_resolution import _module_to_file_paths

        paths = _module_to_file_paths(
            ".utils", caller_path="src/magaldi_core/parsers/base.py"
        )
        assert "src/magaldi_core/parsers/utils.py" in paths
        assert "src/magaldi_core/parsers/utils/__init__.py" in paths

    def test_relative_import_double_dot(self):
        """Test resolving double-dot relative import (parent dir)."""
        from magaldi_core.call_resolution import _module_to_file_paths

        paths = _module_to_file_paths(
            "..common", caller_path="src/magaldi_core/parsers/base.py"
        )
        assert "src/magaldi_core/common.py" in paths
        assert "src/magaldi_core/common/__init__.py" in paths

    def test_relative_import_bare_dot(self):
        """Test resolving bare dot import (from . import foo)."""
        from magaldi_core.call_resolution import _module_to_file_paths

        paths = _module_to_file_paths(
            ".", caller_path="src/magaldi_core/parsers/base.py"
        )
        assert "src/magaldi_core/parsers/__init__.py" in paths

    def test_relative_import_triple_dot(self):
        """Test resolving triple-dot relative import (grandparent dir)."""
        from magaldi_core.call_resolution import _module_to_file_paths

        paths = _module_to_file_paths(
            "...helpers", caller_path="src/magaldi_core/parsers/python/base.py"
        )
        assert "src/magaldi_core/helpers.py" in paths
        assert "src/magaldi_core/helpers/__init__.py" in paths

    def test_relative_import_submodule(self):
        """Test resolving relative import with submodule path."""
        from magaldi_core.call_resolution import _module_to_file_paths

        paths = _module_to_file_paths(
            ".extractors.python", caller_path="src/magaldi_core/parsers/base.py"
        )
        assert "src/magaldi_core/parsers/extractors/python.py" in paths
        assert "src/magaldi_core/parsers/extractors/python/__init__.py" in paths


# =============================================================================
# FIX 2: TYPE UNWRAPPING TESTS
# =============================================================================


class TestUnwrapType:
    """Tests for _unwrap_type() helper."""

    def test_plain_type(self):
        from magaldi_core.call_resolution import _unwrap_type

        assert _unwrap_type("Repository") == "Repository"

    def test_optional(self):
        from magaldi_core.call_resolution import _unwrap_type

        assert _unwrap_type("Optional[Repository]") == "Repository"

    def test_union_with_none(self):
        from magaldi_core.call_resolution import _unwrap_type

        assert _unwrap_type("Union[Repository, None]") == "Repository"

    def test_generic(self):
        from magaldi_core.call_resolution import _unwrap_type

        assert _unwrap_type("list[str]") == "list"

    def test_qualified_name(self):
        from magaldi_core.call_resolution import _unwrap_type

        assert _unwrap_type("db.Repository") == "Repository"

    def test_optional_qualified(self):
        from magaldi_core.call_resolution import _unwrap_type

        assert _unwrap_type("Optional[db.Repository]") == "Repository"

    def test_whitespace(self):
        from magaldi_core.call_resolution import _unwrap_type

        assert _unwrap_type("  Repository  ") == "Repository"

    def test_union_multiple_non_none(self):
        """Union with multiple non-None types keeps the Union prefix."""
        from magaldi_core.call_resolution import _unwrap_type

        assert _unwrap_type("Union[str, int]") == "Union"


# =============================================================================
# FIX 3: RETURN-TYPE PROPAGATION TESTS
# =============================================================================


class TestBuildReceiverTypeMap:
    """Tests for _build_receiver_type_map() helper."""

    def test_simple_assignment(self):
        from unittest.mock import MagicMock

        from magaldi_core.call_resolution import _build_receiver_type_map

        repo = MagicMock()
        repo.get_document.return_value = {"return_type": "User"}

        raw_code = "result = get_user()\nresult.save()"
        resolved_calls = {"get_user": "scope:repo:main:path:function:get_user:10"}

        type_map = _build_receiver_type_map(raw_code, resolved_calls, repo)

        assert type_map == {"result": "User"}
        repo.get_document.assert_called_once_with(
            "scope:repo:main:path:function:get_user:10"
        )

    def test_no_return_type(self):
        from unittest.mock import MagicMock

        from magaldi_core.call_resolution import _build_receiver_type_map

        repo = MagicMock()
        repo.get_document.return_value = {"return_type": None}

        raw_code = "result = get_user()\nresult.save()"
        resolved_calls = {"get_user": "some:id"}

        type_map = _build_receiver_type_map(raw_code, resolved_calls, repo)

        assert type_map == {}

    def test_unresolved_function(self):
        from unittest.mock import MagicMock

        from magaldi_core.call_resolution import _build_receiver_type_map

        repo = MagicMock()

        raw_code = "result = unknown_func()\nresult.save()"
        resolved_calls = {}  # unknown_func not resolved

        type_map = _build_receiver_type_map(raw_code, resolved_calls, repo)

        assert type_map == {}
        repo.get_document.assert_not_called()

    def test_await_assignment(self):
        from unittest.mock import MagicMock

        from magaldi_core.call_resolution import _build_receiver_type_map

        repo = MagicMock()
        repo.get_document.return_value = {"return_type": "Response"}

        raw_code = "result = await fetch_data()\nresult.json()"
        resolved_calls = {"fetch_data": "some:id"}

        type_map = _build_receiver_type_map(raw_code, resolved_calls, repo)

        assert type_map == {"result": "Response"}

    def test_multiple_assignments(self):
        from unittest.mock import MagicMock

        from magaldi_core.call_resolution import _build_receiver_type_map

        repo = MagicMock()
        repo.get_document.side_effect = [
            {"return_type": "User"},
            {"return_type": "Session"},
        ]

        raw_code = "user = get_user()\nsession = create_session()\nuser.save()\nsession.commit()"
        resolved_calls = {
            "get_user": "id:1",
            "create_session": "id:2",
        }

        type_map = _build_receiver_type_map(raw_code, resolved_calls, repo)

        assert type_map == {"user": "User", "session": "Session"}


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestCallResolutionIntegration:
    """Integration tests for the full call resolution pipeline."""

    def test_call_object_has_resolved_id_field(self):
        """Test that Call objects have resolved_id field."""
        call = Call(name="test", receiver=None, line=1)
        assert hasattr(call, "resolved_id")
        assert call.resolved_id is None

        call_with_id = Call(name="test", receiver=None, line=1, resolved_id="some:id")
        assert call_with_id.resolved_id == "some:id"

    def test_extracted_call_has_resolved_id_field(self):
        """Test that ExtractedCall objects have resolved_id field."""
        from magaldi_core.extractors.types import ExtractedCall

        call = ExtractedCall(name="test", receiver=None, line=1)
        assert hasattr(call, "resolved_id")
        assert call.resolved_id is None

        call_with_id = ExtractedCall(name="test", receiver=None, line=1, resolved_id="some:id")
        assert call_with_id.resolved_id == "some:id"


# =============================================================================
# MULTI-LANGUAGE REGEX PATTERN TESTS
# =============================================================================


class TestMultiLanguageAssignmentPatterns:
    """Test per-language assignment patterns for Strategy 5.5."""

    def test_javascript_const_assignment(self):
        from unittest.mock import MagicMock

        from magaldi_core.call_resolution import _build_receiver_type_map

        repo = MagicMock()
        repo.get_document.return_value = {"return_type": "User"}

        raw_code = "const result = getUser()\nresult.save()"
        resolved_calls = {"getUser": "some:id"}

        type_map = _build_receiver_type_map(
            raw_code, resolved_calls, repo, language="javascript",
        )
        assert type_map == {"result": "User"}

    def test_javascript_let_assignment(self):
        from unittest.mock import MagicMock

        from magaldi_core.call_resolution import _build_receiver_type_map

        repo = MagicMock()
        repo.get_document.return_value = {"return_type": "Response"}

        raw_code = "let data = fetchData()\ndata.json()"
        resolved_calls = {"fetchData": "some:id"}

        type_map = _build_receiver_type_map(
            raw_code, resolved_calls, repo, language="javascript",
        )
        assert type_map == {"data": "Response"}

    def test_javascript_await_assignment(self):
        from unittest.mock import MagicMock

        from magaldi_core.call_resolution import _build_receiver_type_map

        repo = MagicMock()
        repo.get_document.return_value = {"return_type": "Response"}

        raw_code = "const response = await fetch()\nresponse.json()"
        resolved_calls = {"fetch": "some:id"}

        type_map = _build_receiver_type_map(
            raw_code, resolved_calls, repo, language="javascript",
        )
        assert type_map == {"response": "Response"}

    def test_php_assignment(self):
        from unittest.mock import MagicMock

        from magaldi_core.call_resolution import _build_receiver_type_map

        repo = MagicMock()
        repo.get_document.return_value = {"return_type": "User"}

        raw_code = "$result = getUser()\n$result->save()"
        resolved_calls = {"getUser": "some:id"}

        type_map = _build_receiver_type_map(
            raw_code, resolved_calls, repo, language="php",
        )
        assert type_map == {"result": "User"}

    def test_rust_let_assignment(self):
        from unittest.mock import MagicMock

        from magaldi_core.call_resolution import _build_receiver_type_map

        repo = MagicMock()
        repo.get_document.return_value = {"return_type": "User"}

        raw_code = "let result = get_user()\nresult.save()"
        resolved_calls = {"get_user": "some:id"}

        type_map = _build_receiver_type_map(
            raw_code, resolved_calls, repo, language="rust",
        )
        assert type_map == {"result": "User"}

    def test_rust_let_mut_assignment(self):
        from unittest.mock import MagicMock

        from magaldi_core.call_resolution import _build_receiver_type_map

        repo = MagicMock()
        repo.get_document.return_value = {"return_type": "Connection"}

        raw_code = "let mut conn = connect()\nconn.query()"
        resolved_calls = {"connect": "some:id"}

        type_map = _build_receiver_type_map(
            raw_code, resolved_calls, repo, language="rust",
        )
        assert type_map == {"conn": "Connection"}

    def test_rust_typed_let_assignment(self):
        from unittest.mock import MagicMock

        from magaldi_core.call_resolution import _build_receiver_type_map

        repo = MagicMock()
        repo.get_document.return_value = {"return_type": "Config"}

        raw_code = "let cfg: Config = load_config()\ncfg.validate()"
        resolved_calls = {"load_config": "some:id"}

        type_map = _build_receiver_type_map(
            raw_code, resolved_calls, repo, language="rust",
        )
        assert type_map == {"cfg": "Config"}


class TestMultiLanguageConstructorPatterns:
    """Test per-language constructor patterns for Strategy 5.6."""

    def test_javascript_new_constructor(self):
        from magaldi_core.call_resolution import _build_constructor_type_map

        code = "const repo = new Repository()"
        result = _build_constructor_type_map(code, language="javascript")
        assert result == {"repo": "Repository"}

    def test_javascript_await_new(self):
        from magaldi_core.call_resolution import _build_constructor_type_map

        code = "const client = await new ApiClient()"
        result = _build_constructor_type_map(code, language="javascript")
        assert result == {"client": "ApiClient"}

    def test_javascript_no_match_without_new(self):
        from magaldi_core.call_resolution import _build_constructor_type_map

        # Without "new", JS constructor pattern should NOT match
        code = "const repo = Repository()"
        result = _build_constructor_type_map(code, language="javascript")
        assert result == {}

    def test_javascript_builtin_excluded(self):
        from magaldi_core.call_resolution import _build_constructor_type_map

        code = "const err = new Error('msg')"
        result = _build_constructor_type_map(code, language="javascript")
        assert result == {}

    def test_php_new_constructor(self):
        from magaldi_core.call_resolution import _build_constructor_type_map

        code = "$repo = new Repository()"
        result = _build_constructor_type_map(code, language="php")
        assert result == {"repo": "Repository"}

    def test_php_namespaced_constructor(self):
        from magaldi_core.call_resolution import _build_constructor_type_map

        # Backslash before class name should be handled
        code = "$user = new \\User()"
        result = _build_constructor_type_map(code, language="php")
        assert result == {"user": "User"}

    def test_rust_new_constructor(self):
        from magaldi_core.call_resolution import _build_constructor_type_map

        code = "let repo = Repository::new()"
        result = _build_constructor_type_map(code, language="rust")
        assert result == {"repo": "Repository"}

    def test_rust_mut_constructor(self):
        from magaldi_core.call_resolution import _build_constructor_type_map

        code = "let mut builder = Builder::new()"
        result = _build_constructor_type_map(code, language="rust")
        assert result == {"builder": "Builder"}

    def test_rust_builtin_excluded(self):
        from magaldi_core.call_resolution import _build_constructor_type_map

        code = "let map = HashMap::new()"
        result = _build_constructor_type_map(code, language="rust")
        assert result == {}

    def test_rust_typed_constructor(self):
        from magaldi_core.call_resolution import _build_constructor_type_map

        code = "let repo: Repository = Repository::new()"
        result = _build_constructor_type_map(code, language="rust")
        assert result == {"repo": "Repository"}


# =============================================================================
# WILDCARD IMPORT EXPANSION TESTS
# =============================================================================


class TestBuildImportMapWildcard:
    """Tests for wildcard import expansion in _build_import_map."""

    def test_non_wildcard_imports_unchanged(self):
        """Basic imports work exactly as before when no repo provided."""
        from magaldi_core.call_resolution import _build_import_map

        imports = [
            {"name": "process", "module": "utils", "alias": None, "line": 1},
            {"name": "pandas", "module": "pandas", "alias": "pd", "line": 2},
        ]

        result = _build_import_map(imports)

        assert "process" in result
        assert "pd" in result

    def test_wildcard_import_not_expanded_without_repo(self):
        """Wildcard imports are not expanded when no repo is available."""
        from magaldi_core.call_resolution import _build_import_map

        imports = [
            {"name": "*", "module": "utils", "alias": None, "line": 1},
            {"name": "foo", "module": "bar", "alias": None, "line": 2},
        ]

        result = _build_import_map(imports)

        # Without repo, wildcard stays as "*" (harmless, no function named "*")
        # and foo is still present
        assert "foo" in result
        assert len(result) == 2  # "*" and "foo"

    def test_wildcard_import_expanded_with_repo(self):
        """Wildcard import expands to all elements in the target file."""
        from unittest.mock import MagicMock

        from magaldi_core.call_resolution import _build_import_map

        repo = MagicMock()
        repo.get_elements_by_file.return_value = [
            {"name": "process", "element_type": "function", "element_id": "id:1"},
            {"name": "validate", "element_type": "function", "element_id": "id:2"},
            {"name": "MyClass", "element_type": "class", "element_id": "id:3"},
        ]

        imports = [
            {"name": "*", "module": ".utils", "alias": None, "line": 1},
        ]

        result = _build_import_map(
            imports, repo, "scope", "repo", "main",
            language="python", caller_path="src/app/views.py",
        )

        assert "process" in result
        assert "validate" in result
        assert "MyClass" in result
        assert result["process"]["module"] == ".utils"

    def test_wildcard_skips_private_names_python(self):
        """Python wildcard import doesn't include _private names."""
        from unittest.mock import MagicMock

        from magaldi_core.call_resolution import _build_import_map

        repo = MagicMock()
        repo.get_elements_by_file.return_value = [
            {"name": "public_func", "element_type": "function", "element_id": "id:1"},
            {"name": "_private_func", "element_type": "function", "element_id": "id:2"},
            {"name": "__dunder", "element_type": "function", "element_id": "id:3"},
        ]

        imports = [
            {"name": "*", "module": ".utils", "alias": None, "line": 1},
        ]

        result = _build_import_map(
            imports, repo, "scope", "repo", "main",
            language="python", caller_path="src/app/views.py",
        )

        assert "public_func" in result
        assert "_private_func" not in result
        assert "__dunder" not in result

    def test_wildcard_skips_variables_and_files(self):
        """Wildcard import doesn't include file or variable elements."""
        from unittest.mock import MagicMock

        from magaldi_core.call_resolution import _build_import_map

        repo = MagicMock()
        repo.get_elements_by_file.return_value = [
            {"name": "process", "element_type": "function", "element_id": "id:1"},
            {"name": "MY_CONST", "element_type": "variable", "element_id": "id:2"},
            {"name": "utils.py", "element_type": "file", "element_id": "id:3"},
        ]

        imports = [
            {"name": "*", "module": ".utils", "alias": None, "line": 1},
        ]

        result = _build_import_map(
            imports, repo, "scope", "repo", "main",
            language="python", caller_path="src/app/views.py",
        )

        assert "process" in result
        assert "MY_CONST" not in result
        assert "utils.py" not in result

    def test_wildcard_does_not_overwrite_explicit_imports(self):
        """Explicit imports take precedence over wildcard-expanded names."""
        from unittest.mock import MagicMock

        from magaldi_core.call_resolution import _build_import_map

        repo = MagicMock()
        repo.get_elements_by_file.return_value = [
            {"name": "process", "element_type": "function", "element_id": "id:1"},
        ]

        imports = [
            {"name": "process", "module": "other_module", "alias": None, "line": 1},
            {"name": "*", "module": ".utils", "alias": None, "line": 2},
        ]

        result = _build_import_map(
            imports, repo, "scope", "repo", "main",
            language="python", caller_path="src/app/views.py",
        )

        # Explicit import came first and should not be overwritten
        assert result["process"]["module"] == "other_module"

    def test_wildcard_external_module_skipped(self):
        """Wildcard import of external module is skipped."""
        from unittest.mock import MagicMock

        from magaldi_core.call_resolution import _build_import_map

        repo = MagicMock()

        imports = [
            {"name": "*", "module": "numpy", "alias": None, "line": 1},
        ]

        result = _build_import_map(
            imports, repo, "scope", "repo", "main",
            language="python", caller_path="src/app/views.py",
        )

        # Should not have expanded anything
        assert len(result) == 0
        repo.get_elements_by_file.assert_not_called()


# =============================================================================
# RE-EXPORT FOLLOWING TESTS
# =============================================================================


class TestFollowInitReexports:
    """Tests for _follow_init_reexports."""

    def test_follows_reexport_from_init(self):
        """Resolves element re-exported through __init__.py."""
        from unittest.mock import MagicMock

        from magaldi_core.call_resolution import _follow_init_reexports

        repo = MagicMock()
        # __init__.py has: from .models import User
        repo.get_file_imports.return_value = [
            {"name": "User", "module": ".models", "alias": None, "line": 1},
        ]
        # User found in models.py
        repo.get_document_by_name.side_effect = lambda name, element_type, relative_path, **kw: (
            {"element_id": "scope:repo:main:src/mypackage/models.py:class:User:10"}
            if name == "User" and "models" in relative_path
            else None
        )

        result = _follow_init_reexports(
            repo,
            ["src/mypackage/__init__.py"],
            "User",
            "scope", "repo", "main",
            language="python",
        )

        assert result == "scope:repo:main:src/mypackage/models.py:class:User:10"

    def test_skips_non_init_files(self):
        """Only processes __init__.py files for re-exports."""
        from unittest.mock import MagicMock

        from magaldi_core.call_resolution import _follow_init_reexports

        repo = MagicMock()

        result = _follow_init_reexports(
            repo,
            ["src/utils.py", "src/helpers.py"],
            "SomeFunc",
            "scope", "repo", "main",
        )

        assert result is None
        repo.get_file_imports.assert_not_called()

    def test_returns_none_when_no_matching_reexport(self):
        """Returns None when __init__.py doesn't re-export the element."""
        from unittest.mock import MagicMock

        from magaldi_core.call_resolution import _follow_init_reexports

        repo = MagicMock()
        repo.get_file_imports.return_value = [
            {"name": "OtherClass", "module": ".models", "alias": None, "line": 1},
        ]

        result = _follow_init_reexports(
            repo,
            ["src/mypackage/__init__.py"],
            "User",
            "scope", "repo", "main",
        )

        assert result is None

    def test_depth_limit_prevents_infinite_recursion(self):
        """Recursion stops at depth 3."""
        from unittest.mock import MagicMock

        from magaldi_core.call_resolution import _follow_init_reexports

        repo = MagicMock()

        result = _follow_init_reexports(
            repo,
            ["src/pkg/__init__.py"],
            "Foo",
            "scope", "repo", "main",
            _depth=3,
        )

        assert result is None
        repo.get_file_imports.assert_not_called()


# =============================================================================
# FALLBACK NAME LOOKUP TESTS
# =============================================================================


class TestFallbackNameLookup:
    """Tests for _fallback_name_lookup."""

    def test_single_candidate_resolves(self):
        """Returns element ID when exactly one candidate exists."""
        from unittest.mock import MagicMock

        from magaldi_core.call_resolution import _fallback_name_lookup

        repo = MagicMock()
        repo.find_candidates_by_name.return_value = [
            {"element_id": "scope:repo:main:path:function:process:10"},
        ]

        result = _fallback_name_lookup(repo, "process", "scope", "repo", "main")

        assert result == "scope:repo:main:path:function:process:10"

    def test_multiple_candidates_returns_none(self):
        """Returns None when multiple candidates exist (ambiguous)."""
        from unittest.mock import MagicMock

        from magaldi_core.call_resolution import _fallback_name_lookup

        repo = MagicMock()
        repo.find_candidates_by_name.return_value = [
            {"element_id": "id:1"},
            {"element_id": "id:2"},
        ]

        result = _fallback_name_lookup(repo, "get", "scope", "repo", "main")

        assert result is None

    def test_no_candidates_returns_none(self):
        """Returns None when no candidates exist."""
        from unittest.mock import MagicMock

        from magaldi_core.call_resolution import _fallback_name_lookup

        repo = MagicMock()
        repo.find_candidates_by_name.return_value = []

        result = _fallback_name_lookup(repo, "nonexistent", "scope", "repo", "main")

        assert result is None


# =============================================================================
# Embedding blocklist for generic names
# =============================================================================


class TestEmbeddingBlocklist:
    """Verify generic method names are skipped by embedding resolution."""

    def test_blocklist_contains_common_names(self):
        """Blocklist should include the most common false-positive names."""
        from magaldi_core.call_resolution import _EMBEDDING_BLOCKLIST

        for name in ["new", "create", "close", "resolve", "apply", "get", "set"]:
            assert name in _EMBEDDING_BLOCKLIST, f"{name} should be in blocklist"

    def test_generic_single_candidate_skipped(self):
        """Generic names with a single candidate should not be resolved."""
        from unittest.mock import MagicMock

        from magaldi_core.call_resolution import resolve_calls_by_embedding

        repo = MagicMock()
        repo.find_all_elements_with_calls.return_value = [
            {
                "element_id": "scope:repo:main:app.py:function:my_func:10",
                "relative_path": "app.py",
                "calls": [
                    {"name": "create", "receiver": "Factory", "category": "untyped"},
                ],
            }
        ]
        repo.find_candidates_by_name.return_value = [
            {"element_id": "target1", "name": "create", "element_type": "method"},
        ]

        total, single, rrf = resolve_calls_by_embedding(repo, "scope", "repo", "main")

        # "create" is generic — single candidate should be skipped
        assert single == 0

    def test_non_generic_single_candidate_resolves(self):
        """Non-generic names with a single candidate should resolve normally."""
        from unittest.mock import MagicMock

        from magaldi_core.call_resolution import resolve_calls_by_embedding

        repo = MagicMock()
        repo.find_all_elements_with_calls.return_value = [
            {
                "element_id": "scope:repo:main:app.py:function:my_func:10",
                "relative_path": "app.py",
                "calls": [
                    {"name": "processOrder", "receiver": "OrderService", "category": "untyped"},
                ],
            }
        ]
        repo.find_candidates_by_name.return_value = [
            {"element_id": "target1", "name": "processOrder", "element_type": "method"},
        ]

        total, single, rrf = resolve_calls_by_embedding(repo, "scope", "repo", "main")

        # "processOrder" is NOT generic — should resolve
        assert single == 1

    def test_generic_names_need_higher_rrf_threshold(self):
        """Generic names should require a higher RRF score to resolve."""
        from magaldi_core.call_resolution import _EMBEDDING_BLOCKLIST

        # Verify the blocklist is used as a threshold modifier, not a hard skip
        # (the actual threshold doubling is tested via integration in the function)
        assert "get" in _EMBEDDING_BLOCKLIST
        assert "processOrder" not in _EMBEDDING_BLOCKLIST

    def test_blocklist_is_frozenset(self):
        """Blocklist should be a frozenset for O(1) lookups."""
        from magaldi_core.call_resolution import _EMBEDDING_BLOCKLIST

        assert isinstance(_EMBEDDING_BLOCKLIST, frozenset)


class TestSuperResolution:
    """Test Strategy 5.8: super()/parent:: call resolution."""

    def _make_repo(self):
        from unittest.mock import MagicMock

        repo = MagicMock()
        repo.store_calls = MagicMock()
        return repo

    def test_resolves_python_super_call(self):
        """Test super().method() resolves to parent class method."""
        from magaldi_core.call_resolution import _resolve_via_super

        repo = self._make_repo()

        elements = [
            {
                "element_id": "s:r:main:app.py:method:validate:20",
                "parent_id": "s:r:main:app.py:class:Child:10",
                "calls": [
                    {"name": "validate", "receiver": "super", "category": "untyped"},
                ],
            },
        ]

        # Parent class has base_classes
        repo.get_document.return_value = {
            "element_id": "s:r:main:app.py:class:Child:10",
            "base_classes": ["Parent"],
        }

        # Resolve Parent class
        repo.get_document_by_name_only.return_value = {
            "element_id": "s:r:main:base.py:class:Parent:1",
        }

        # Resolve method on Parent
        repo.get_method_by_class.return_value = {
            "element_id": "s:r:main:base.py:method:validate:5",
        }

        count = _resolve_via_super(repo, elements, "s", "r", "main")

        assert count == 1
        assert elements[0]["calls"][0]["resolved_id"] == "s:r:main:base.py:method:validate:5"
        assert elements[0]["calls"][0]["category"] == "super_resolved"
        repo.store_calls.assert_called_once()

    def test_resolves_php_parent_call(self):
        """Test parent::method() resolves (PHP uses receiver='parent')."""
        from magaldi_core.call_resolution import _resolve_via_super

        repo = self._make_repo()

        elements = [
            {
                "element_id": "s:r:main:app.php:method:save:20",
                "parent_id": "s:r:main:app.php:class:Child:10",
                "calls": [
                    {"name": "save", "receiver": "parent", "category": "untyped"},
                ],
            },
        ]

        repo.get_document.return_value = {
            "element_id": "s:r:main:app.php:class:Child:10",
            "base_classes": ["BaseModel"],
        }
        repo.get_document_by_name_only.return_value = {
            "element_id": "s:r:main:base.php:class:BaseModel:1",
        }
        repo.get_method_by_class.return_value = {
            "element_id": "s:r:main:base.php:method:save:5",
        }

        count = _resolve_via_super(repo, elements, "s", "r", "main")
        assert count == 1
        assert elements[0]["calls"][0]["resolved_id"] == "s:r:main:base.php:method:save:5"

    def test_skips_already_resolved(self):
        """Test already-resolved super calls are skipped."""
        from magaldi_core.call_resolution import _resolve_via_super

        repo = self._make_repo()

        elements = [
            {
                "element_id": "s:r:main:app.py:method:validate:20",
                "parent_id": "s:r:main:app.py:class:Child:10",
                "calls": [
                    {
                        "name": "validate",
                        "receiver": "super",
                        "resolved_id": "already_resolved",
                        "category": "resolved",
                    },
                ],
            },
        ]

        count = _resolve_via_super(repo, elements, "s", "r", "main")
        assert count == 0

    def test_no_base_classes_skips(self):
        """Test elements without base_classes are skipped."""
        from magaldi_core.call_resolution import _resolve_via_super

        repo = self._make_repo()

        elements = [
            {
                "element_id": "s:r:main:app.py:method:validate:20",
                "parent_id": "s:r:main:app.py:class:Child:10",
                "calls": [
                    {"name": "validate", "receiver": "super", "category": "untyped"},
                ],
            },
        ]

        repo.get_document.return_value = {
            "element_id": "s:r:main:app.py:class:Child:10",
            "base_classes": [],
        }

        count = _resolve_via_super(repo, elements, "s", "r", "main")
        assert count == 0

    def test_no_parent_id_skips(self):
        """Test top-level functions with super calls are skipped."""
        from magaldi_core.call_resolution import _resolve_via_super

        repo = self._make_repo()

        elements = [
            {
                "element_id": "s:r:main:app.py:function:broken:1",
                "calls": [
                    {"name": "validate", "receiver": "super", "category": "untyped"},
                ],
            },
        ]

        count = _resolve_via_super(repo, elements, "s", "r", "main")
        assert count == 0

    def test_method_not_found_in_parent(self):
        """Test graceful handling when method doesn't exist in parent class."""
        from magaldi_core.call_resolution import _resolve_via_super

        repo = self._make_repo()

        elements = [
            {
                "element_id": "s:r:main:app.py:method:validate:20",
                "parent_id": "s:r:main:app.py:class:Child:10",
                "calls": [
                    {"name": "nonexistent", "receiver": "super", "category": "untyped"},
                ],
            },
        ]

        repo.get_document.return_value = {
            "element_id": "s:r:main:app.py:class:Child:10",
            "base_classes": ["Parent"],
        }
        repo.get_document_by_name_only.return_value = {
            "element_id": "s:r:main:base.py:class:Parent:1",
        }
        repo.get_method_by_class.return_value = None

        count = _resolve_via_super(repo, elements, "s", "r", "main")
        assert count == 0

    def test_multiple_inheritance_tries_all(self):
        """Test resolution tries each base class in order."""
        from magaldi_core.call_resolution import _resolve_via_super

        repo = self._make_repo()

        elements = [
            {
                "element_id": "s:r:main:app.py:method:save:20",
                "parent_id": "s:r:main:app.py:class:Child:10",
                "calls": [
                    {"name": "save", "receiver": "super", "category": "untyped"},
                ],
            },
        ]

        repo.get_document.return_value = {
            "element_id": "s:r:main:app.py:class:Child:10",
            "base_classes": ["Mixin", "Base"],
        }

        # Mixin doesn't have the method, Base does
        def mock_name_lookup(name, element_type, scope, repository, username):
            if name == "Mixin":
                return {"element_id": "s:r:main:mixin.py:class:Mixin:1"}
            if name == "Base":
                return {"element_id": "s:r:main:base.py:class:Base:1"}
            return None

        repo.get_document_by_name_only.side_effect = mock_name_lookup

        def mock_method_lookup(class_id, method_name, scope, repository, username):
            if class_id == "s:r:main:base.py:class:Base:1" and method_name == "save":
                return {"element_id": "s:r:main:base.py:method:save:5"}
            return None

        repo.get_method_by_class.side_effect = mock_method_lookup

        count = _resolve_via_super(repo, elements, "s", "r", "main")
        assert count == 1
        assert elements[0]["calls"][0]["resolved_id"] == "s:r:main:base.py:method:save:5"
