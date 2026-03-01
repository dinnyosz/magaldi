"""Tests for per-language module resolution."""

from __future__ import annotations

import pytest

from magaldi_core.module_resolver import (
    JavaScriptModuleResolver,
    PHPModuleResolver,
    PythonModuleResolver,
    RustModuleResolver,
    get_module_resolver,
)

# =============================================================================
# REGISTRY
# =============================================================================


class TestGetModuleResolver:
    """Test the resolver registry."""

    def test_python(self) -> None:
        resolver = get_module_resolver("python")
        assert resolver is not None
        assert isinstance(resolver, PythonModuleResolver)

    def test_javascript(self) -> None:
        resolver = get_module_resolver("javascript")
        assert resolver is not None
        assert isinstance(resolver, JavaScriptModuleResolver)

    def test_typescript_shares_js_resolver(self) -> None:
        js = get_module_resolver("javascript")
        ts = get_module_resolver("typescript")
        tsx = get_module_resolver("tsx")
        assert js is ts
        assert js is tsx

    def test_php(self) -> None:
        resolver = get_module_resolver("php")
        assert resolver is not None
        assert isinstance(resolver, PHPModuleResolver)

    def test_rust(self) -> None:
        resolver = get_module_resolver("rust")
        assert resolver is not None
        assert isinstance(resolver, RustModuleResolver)

    def test_unsupported_language(self) -> None:
        assert get_module_resolver("bash") is None
        assert get_module_resolver("unknown") is None


# =============================================================================
# PYTHON
# =============================================================================


class TestPythonExternalModule:
    """Test Python stdlib/third-party detection."""

    @pytest.fixture
    def resolver(self) -> PythonModuleResolver:
        return PythonModuleResolver()

    def test_stdlib_module(self, resolver: PythonModuleResolver) -> None:
        assert resolver.is_external_module("os") is True
        assert resolver.is_external_module("os.path") is True
        assert resolver.is_external_module("json") is True
        assert resolver.is_external_module("pathlib") is True
        assert resolver.is_external_module("collections.abc") is True

    def test_third_party_module(self, resolver: PythonModuleResolver) -> None:
        assert resolver.is_external_module("numpy") is True
        assert resolver.is_external_module("pandas.DataFrame") is True
        assert resolver.is_external_module("requests") is True
        assert resolver.is_external_module("pydantic") is True

    def test_relative_import_is_internal(self, resolver: PythonModuleResolver) -> None:
        assert resolver.is_external_module(".utils") is False
        assert resolver.is_external_module("..models") is False
        assert resolver.is_external_module(".") is False

    def test_local_module_is_internal(self, resolver: PythonModuleResolver) -> None:
        assert resolver.is_external_module("magaldi_core") is False
        assert resolver.is_external_module("shared.db") is False
        assert resolver.is_external_module("myapp.utils") is False


class TestPythonModulePaths:
    """Test Python module → file path conversion."""

    @pytest.fixture
    def resolver(self) -> PythonModuleResolver:
        return PythonModuleResolver()

    def test_absolute_import(self, resolver: PythonModuleResolver) -> None:
        paths = resolver.module_to_file_paths("magaldi_core.storage")
        assert "src/magaldi_core/storage.py" in paths
        assert "src/magaldi_core/storage/__init__.py" in paths
        assert "magaldi_core/storage.py" in paths

    def test_single_dot_relative(self, resolver: PythonModuleResolver) -> None:
        paths = resolver.module_to_file_paths(".utils", "src/myapp/views.py")
        assert "src/myapp/utils.py" in paths
        assert "src/myapp/utils/__init__.py" in paths

    def test_double_dot_relative(self, resolver: PythonModuleResolver) -> None:
        paths = resolver.module_to_file_paths("..models", "src/myapp/views/base.py")
        assert "src/myapp/models.py" in paths

    def test_bare_relative(self, resolver: PythonModuleResolver) -> None:
        paths = resolver.module_to_file_paths(".", "src/myapp/views.py")
        assert "src/myapp/__init__.py" in paths

    def test_no_caller_path_for_relative(self, resolver: PythonModuleResolver) -> None:
        paths = resolver.module_to_file_paths(".utils")
        assert paths == []


# =============================================================================
# JAVASCRIPT / TYPESCRIPT
# =============================================================================


class TestJavaScriptExternalModule:
    """Test JS/TS external module detection."""

    @pytest.fixture
    def resolver(self) -> JavaScriptModuleResolver:
        return JavaScriptModuleResolver()

    def test_node_builtin(self, resolver: JavaScriptModuleResolver) -> None:
        assert resolver.is_external_module("fs") is True
        assert resolver.is_external_module("path") is True
        assert resolver.is_external_module("http") is True
        assert resolver.is_external_module("crypto") is True

    def test_node_protocol(self, resolver: JavaScriptModuleResolver) -> None:
        assert resolver.is_external_module("node:fs") is True
        assert resolver.is_external_module("node:path") is True

    def test_npm_package(self, resolver: JavaScriptModuleResolver) -> None:
        assert resolver.is_external_module("react") is True
        assert resolver.is_external_module("express") is True
        assert resolver.is_external_module("lodash") is True

    def test_scoped_package(self, resolver: JavaScriptModuleResolver) -> None:
        assert resolver.is_external_module("@scope/package") is True

    def test_bare_specifier_assumed_external(self, resolver: JavaScriptModuleResolver) -> None:
        # Unknown bare specifiers are npm packages by convention
        assert resolver.is_external_module("some-unknown-package") is True

    def test_relative_is_internal(self, resolver: JavaScriptModuleResolver) -> None:
        assert resolver.is_external_module("./utils") is False
        assert resolver.is_external_module("../lib/foo") is False
        assert resolver.is_external_module("/absolute/path") is False

    def test_tilde_import_is_internal(self, resolver: JavaScriptModuleResolver) -> None:
        """Tilde imports (common path alias) are internal."""
        assert resolver.is_external_module("~/utils") is False
        assert resolver.is_external_module("~/components/Button") is False

    def test_hash_import_is_internal(self, resolver: JavaScriptModuleResolver) -> None:
        """Hash imports (Node.js subpath imports) are internal."""
        assert resolver.is_external_module("#utils") is False
        assert resolver.is_external_module("#lib/helpers") is False

    def test_import_with_extension_is_internal(self, resolver: JavaScriptModuleResolver) -> None:
        """Imports ending with a known extension are internal."""
        assert resolver.is_external_module("utils.ts") is False
        assert resolver.is_external_module("helpers.js") is False
        assert resolver.is_external_module("component.tsx") is False

    def test_node_builtins_and_npm_checks_not_shortcircuited(self, resolver: JavaScriptModuleResolver) -> None:
        """Builtins and npm checks are actually evaluated (not short-circuited by or True)."""
        # These should match _NODE_BUILTINS specifically
        assert resolver.is_external_module("fs") is True
        # These should match _NPM_PACKAGES specifically
        assert resolver.is_external_module("react") is True


class TestJavaScriptModulePaths:
    """Test JS/TS module → file path conversion."""

    @pytest.fixture
    def resolver(self) -> JavaScriptModuleResolver:
        return JavaScriptModuleResolver()

    def test_relative_import_tries_extensions(self, resolver: JavaScriptModuleResolver) -> None:
        paths = resolver.module_to_file_paths("./utils", "src/app.ts")
        assert "src/utils.ts" in paths
        assert "src/utils.tsx" in paths
        assert "src/utils.js" in paths
        assert "src/utils.jsx" in paths

    def test_relative_import_tries_index(self, resolver: JavaScriptModuleResolver) -> None:
        paths = resolver.module_to_file_paths("./utils", "src/app.ts")
        assert "src/utils/index.ts" in paths
        assert "src/utils/index.js" in paths

    def test_parent_relative(self, resolver: JavaScriptModuleResolver) -> None:
        paths = resolver.module_to_file_paths("../lib/foo", "src/app/main.ts")
        assert "src/lib/foo.ts" in paths
        assert "src/lib/foo.js" in paths

    def test_import_with_extension(self, resolver: JavaScriptModuleResolver) -> None:
        paths = resolver.module_to_file_paths("./utils.js", "src/app.ts")
        # Should include the exact path and also try other extensions
        assert "src/utils.js" in paths
        assert "src/utils.ts" in paths

    def test_bare_specifier_returns_empty(self, resolver: JavaScriptModuleResolver) -> None:
        # External packages can't be resolved to file paths
        paths = resolver.module_to_file_paths("react", "src/app.ts")
        assert paths == []

    def test_no_caller_path(self, resolver: JavaScriptModuleResolver) -> None:
        paths = resolver.module_to_file_paths("./utils")
        assert paths == []


# =============================================================================
# PHP
# =============================================================================


class TestPHPExternalModule:
    """Test PHP external namespace detection."""

    @pytest.fixture
    def resolver(self) -> PHPModuleResolver:
        return PHPModuleResolver()

    def test_illuminate_is_external(self, resolver: PHPModuleResolver) -> None:
        assert resolver.is_external_module("Illuminate\\Support\\Collection") is True

    def test_symfony_is_external(self, resolver: PHPModuleResolver) -> None:
        assert resolver.is_external_module("Symfony\\Component\\HttpFoundation") is True

    def test_doctrine_is_external(self, resolver: PHPModuleResolver) -> None:
        assert resolver.is_external_module("Doctrine\\ORM\\EntityManager") is True

    def test_app_namespace_is_internal(self, resolver: PHPModuleResolver) -> None:
        assert resolver.is_external_module("App\\Models\\User") is False
        assert resolver.is_external_module("App\\Controllers\\UserController") is False

    def test_custom_namespace_is_internal(self, resolver: PHPModuleResolver) -> None:
        assert resolver.is_external_module("MyCompany\\Auth\\Provider") is False

    def test_leading_backslash_stripped(self, resolver: PHPModuleResolver) -> None:
        assert resolver.is_external_module("\\Illuminate\\Support\\Collection") is True
        assert resolver.is_external_module("\\App\\Models\\User") is False


class TestPHPModulePaths:
    """Test PHP namespace → file path conversion."""

    @pytest.fixture
    def resolver(self) -> PHPModuleResolver:
        return PHPModuleResolver()

    def test_app_namespace_psr4(self, resolver: PHPModuleResolver) -> None:
        paths = resolver.module_to_file_paths("App\\Models\\User")
        assert "app/Models/User.php" in paths
        assert "src/Models/User.php" in paths

    def test_tests_namespace(self, resolver: PHPModuleResolver) -> None:
        paths = resolver.module_to_file_paths("Tests\\Unit\\UserTest")
        assert "tests/Unit/UserTest.php" in paths

    def test_custom_namespace(self, resolver: PHPModuleResolver) -> None:
        paths = resolver.module_to_file_paths("MyCompany\\Auth\\Provider")
        assert "src/MyCompany/Auth/Provider.php" in paths
        assert "MyCompany/Auth/Provider.php" in paths

    def test_leading_backslash_stripped(self, resolver: PHPModuleResolver) -> None:
        paths = resolver.module_to_file_paths("\\App\\Models\\User")
        assert "app/Models/User.php" in paths


# =============================================================================
# RUST
# =============================================================================


class TestRustExternalModule:
    """Test Rust external crate detection."""

    @pytest.fixture
    def resolver(self) -> RustModuleResolver:
        return RustModuleResolver()

    def test_std_is_external(self, resolver: RustModuleResolver) -> None:
        assert resolver.is_external_module("std::collections::HashMap") is True
        assert resolver.is_external_module("std::io::Read") is True

    def test_core_is_external(self, resolver: RustModuleResolver) -> None:
        assert resolver.is_external_module("core::fmt::Display") is True

    def test_known_crate_is_external(self, resolver: RustModuleResolver) -> None:
        assert resolver.is_external_module("serde::Serialize") is True
        assert resolver.is_external_module("tokio::sync::Mutex") is True
        assert resolver.is_external_module("anyhow::Result") is True

    def test_crate_path_is_internal(self, resolver: RustModuleResolver) -> None:
        assert resolver.is_external_module("crate::storage::Backend") is False
        assert resolver.is_external_module("crate::utils") is False

    def test_super_path_is_internal(self, resolver: RustModuleResolver) -> None:
        assert resolver.is_external_module("super::utils") is False

    def test_self_path_is_internal(self, resolver: RustModuleResolver) -> None:
        assert resolver.is_external_module("self::helper") is False

    def test_unknown_bare_crate_is_external(self, resolver: RustModuleResolver) -> None:
        # Unknown bare names assumed to be external crates
        assert resolver.is_external_module("some_crate::Thing") is True


class TestRustModulePaths:
    """Test Rust crate path → file path conversion."""

    @pytest.fixture
    def resolver(self) -> RustModuleResolver:
        return RustModuleResolver()

    def test_crate_module_path(self, resolver: RustModuleResolver) -> None:
        paths = resolver.module_to_file_paths("crate::storage::Backend")
        assert "src/storage/Backend.rs" in paths or "src/storage.rs" in paths
        # Should try both the full path and parent module
        assert "src/storage.rs" in paths
        assert "src/storage/mod.rs" in paths

    def test_crate_single_module(self, resolver: RustModuleResolver) -> None:
        paths = resolver.module_to_file_paths("crate::utils")
        assert "src/utils.rs" in paths
        assert "src/utils/mod.rs" in paths

    def test_super_path(self, resolver: RustModuleResolver) -> None:
        paths = resolver.module_to_file_paths(
            "super::utils", "src/storage/backend.rs"
        )
        assert "src/utils.rs" in paths
        assert "src/utils/mod.rs" in paths

    def test_self_path(self, resolver: RustModuleResolver) -> None:
        paths = resolver.module_to_file_paths(
            "self::helper", "src/storage/mod.rs"
        )
        assert "src/storage/helper.rs" in paths
        assert "src/storage/helper/mod.rs" in paths

    def test_external_returns_empty(self, resolver: RustModuleResolver) -> None:
        paths = resolver.module_to_file_paths("std::collections::HashMap")
        assert paths == []

    def test_no_caller_for_super(self, resolver: RustModuleResolver) -> None:
        paths = resolver.module_to_file_paths("super::utils")
        assert paths == []
