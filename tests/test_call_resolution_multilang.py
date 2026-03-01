"""Multi-language integration tests for call resolution.

Tests per-language paths through Strategies 3-5.7:
- Import-based resolution via module resolvers (3-4)
- Return-type propagation via per-language patterns (5.5)
- Constructor inference via per-language patterns (5.6)
- Scope binding extraction via per-language AST (5.7)
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from magaldi_core.call_resolution import (
    _build_constructor_type_map,
    _build_receiver_type_map,
    _lookup_element_by_import,
)
from magaldi_core.module_resolver import get_module_resolver
from magaldi_core.scope_bindings import (
    SOURCE_ASSIGNMENT_CALL,
    SOURCE_ASSIGNMENT_METHOD_CALL,
    SOURCE_CONSTRUCTOR,
    SOURCE_EXCEPT_AS,
    SOURCE_FOR_IN,
    extract_variable_bindings,
)

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_repo():
    """Create a mock Repository for testing cross-file resolution."""
    repo = MagicMock()
    repo.get_document_by_name.return_value = None
    repo.get_document.return_value = None
    return repo


# =============================================================================
# STRATEGY 3-4: IMPORT-BASED RESOLUTION (per language)
# =============================================================================


class TestJSImportResolution:
    """Test import resolution for JavaScript/TypeScript."""

    def test_relative_import_resolves_to_file(self, mock_repo):
        """JS relative import: import { func } from './utils' resolves to utils.ts."""
        elem_id = "scope:repo:main:src/utils.ts:function:processData:5"
        mock_repo.get_document_by_name.return_value = {
            "element_id": elem_id,
        }

        result = _lookup_element_by_import(
            mock_repo,
            import_info={"module": "./utils", "name": "processData"},
            element_name="processData",
            scope="scope",
            repository="repo",
            username="main",
            caller_path="src/app.ts",
            language="javascript",
        )

        assert result == elem_id
        # Verify it tried JS file paths via get_document_by_name
        call_args = mock_repo.get_document_by_name.call_args_list
        paths_tried = [c.kwargs.get("relative_path", c[1].get("relative_path", ""))
                       if c.kwargs else "" for c in call_args]
        assert any("utils.ts" in p for p in paths_tried)

    def test_bare_specifier_skipped_as_external(self, mock_repo):
        """JS bare specifier (npm package) is skipped as external."""
        result = _lookup_element_by_import(
            mock_repo,
            import_info={"module": "react", "name": "useState"},
            element_name="useState",
            scope="scope",
            repository="repo",
            username="main",
            language="javascript",
        )

        assert result is None
        # Should NOT have searched the repo — react is external
        mock_repo.get_document_by_name.assert_not_called()


class TestPHPImportResolution:
    """Test import resolution for PHP."""

    def test_app_namespace_resolves(self, mock_repo):
        """PHP: use App\\Models\\User resolves to app/Models/User.php."""
        elem_id = "scope:repo:main:app/Models/User.php:class:User:10"
        mock_repo.get_document_by_name.return_value = {
            "element_id": elem_id,
        }

        result = _lookup_element_by_import(
            mock_repo,
            import_info={"module": "App\\Models\\User", "name": "User"},
            element_name="User",
            scope="scope",
            repository="repo",
            username="main",
            language="php",
        )

        assert result == elem_id
        call_args = mock_repo.get_document_by_name.call_args_list
        paths_tried = [c.kwargs.get("relative_path", "") for c in call_args]
        assert any("app/Models/User.php" in p for p in paths_tried)

    def test_third_party_skipped(self, mock_repo):
        """PHP: Illuminate namespace is skipped as external."""
        result = _lookup_element_by_import(
            mock_repo,
            import_info={
                "module": "Illuminate\\Support\\Collection",
                "name": "Collection",
            },
            element_name="Collection",
            scope="scope",
            repository="repo",
            username="main",
            language="php",
        )

        assert result is None
        mock_repo.get_document_by_name.assert_not_called()


class TestRustImportResolution:
    """Test import resolution for Rust."""

    def test_crate_path_resolves(self, mock_repo):
        """Rust: use crate::storage::Backend resolves via Rust file paths."""
        elem_id = "scope:repo:main:src/storage.rs:function:Backend:5"

        # Return None for all paths except the one we want
        def side_effect(**kwargs):
            path = kwargs.get("relative_path", "")
            if "storage" in path:
                return {"element_id": elem_id}
            return None

        mock_repo.get_document_by_name.side_effect = side_effect

        result = _lookup_element_by_import(
            mock_repo,
            import_info={
                "module": "crate::storage::Backend",
                "name": "Backend",
            },
            element_name="Backend",
            scope="scope",
            repository="repo",
            username="main",
            language="rust",
        )

        assert result == elem_id
        # Verify Rust-style paths were attempted
        call_args = mock_repo.get_document_by_name.call_args_list
        paths_tried = [c.kwargs.get("relative_path", "") for c in call_args]
        assert any(p.startswith("src/storage") for p in paths_tried)

    def test_std_skipped(self, mock_repo):
        """Rust: std::collections is skipped as external."""
        result = _lookup_element_by_import(
            mock_repo,
            import_info={
                "module": "std::collections::HashMap",
                "name": "HashMap",
            },
            element_name="HashMap",
            scope="scope",
            repository="repo",
            username="main",
            language="rust",
        )

        assert result is None
        mock_repo.get_document_by_name.assert_not_called()


# =============================================================================
# STRATEGY 5.5: RETURN-TYPE PROPAGATION (per language)
# =============================================================================


class TestMultiLangReturnTypePropagation:
    """Test per-language receiver type map building (Strategy 5.5)."""

    def test_js_const_assignment(self, mock_repo):
        """JS: const result = getUser() maps result -> return type of getUser."""
        mock_repo.get_document.return_value = {"return_type": "User"}
        code = "const result = getUser();"
        resolved_calls = {"getUser": "scope:repo:main:utils.ts:function:getUser:1"}
        type_map = _build_receiver_type_map(
            code, resolved_calls, mock_repo, language="javascript",
        )
        assert type_map.get("result") == "User"

    def test_php_assignment(self, mock_repo):
        """PHP: $result = getUser() maps result -> return type of getUser."""
        mock_repo.get_document.return_value = {"return_type": "User"}
        code = "$result = getUser();"
        resolved_calls = {"getUser": "scope:repo:main:utils.php:function:getUser:1"}
        type_map = _build_receiver_type_map(
            code, resolved_calls, mock_repo, language="php",
        )
        assert type_map.get("result") == "User"

    def test_rust_let_assignment(self, mock_repo):
        """Rust: let result = get_user() maps result -> return type of get_user."""
        mock_repo.get_document.return_value = {"return_type": "User"}
        code = "let result = get_user();"
        resolved_calls = {"get_user": "scope:repo:main:utils.rs:function:get_user:1"}
        type_map = _build_receiver_type_map(
            code, resolved_calls, mock_repo, language="rust",
        )
        assert type_map.get("result") == "User"


# =============================================================================
# STRATEGY 5.6: CONSTRUCTOR INFERENCE (per language)
# =============================================================================


class TestMultiLangConstructorInference:
    """Test per-language constructor type map building (Strategy 5.6)."""

    def test_js_new_constructor(self):
        """JS: const repo = new Repository() maps repo -> Repository."""
        code = "const repo = new Repository();"
        type_map = _build_constructor_type_map(code, language="javascript")
        assert type_map.get("repo") == "Repository"

    def test_php_new_constructor(self):
        """PHP: $repo = new Repository() maps repo -> Repository."""
        code = "$repo = new Repository();"
        type_map = _build_constructor_type_map(code, language="php")
        assert type_map.get("repo") == "Repository"

    def test_rust_new_constructor(self):
        """Rust: let repo = Repository::new() maps repo -> Repository."""
        code = "let repo = Repository::new();"
        type_map = _build_constructor_type_map(code, language="rust")
        assert type_map.get("repo") == "Repository"


# =============================================================================
# STRATEGY 5.7: SCOPE BINDINGS (per language, end-to-end)
# =============================================================================


class TestMultiLangScopeBindings:
    """End-to-end tests for per-language scope binding extraction.

    These test full function bodies, verifying that all binding patterns
    are correctly extracted from realistic code for each language.
    """

    def test_js_full_function(self):
        """JS function with all binding patterns."""
        code = (
            "function processOrder(orderId) {\n"
            "    const order = getOrder(orderId);\n"
            "    const items = order.getItems();\n"
            "    const customer = new Customer();\n"
            "    for (const item of items) {}\n"
            "    try {\n"
            "        throw new Error();\n"
            "    } catch (err) {}\n"
            "}\n"
        )
        bindings = extract_variable_bindings(code, "javascript")
        binding_map = {b.variable: b for b in bindings}

        assert "order" in binding_map
        assert binding_map["order"].source == SOURCE_ASSIGNMENT_CALL
        assert binding_map["order"].call_name == "getOrder"

        assert "items" in binding_map
        assert binding_map["items"].source == SOURCE_ASSIGNMENT_METHOD_CALL
        assert binding_map["items"].call_name == "getItems"
        assert binding_map["items"].call_receiver == "order"

        assert "customer" in binding_map
        assert binding_map["customer"].source == SOURCE_CONSTRUCTOR
        assert binding_map["customer"].type_name == "Customer"

        assert "item" in binding_map
        assert binding_map["item"].source == SOURCE_FOR_IN

        assert "err" in binding_map
        assert binding_map["err"].source == SOURCE_EXCEPT_AS

    def test_php_full_function(self):
        """PHP function with all binding patterns."""
        code = (
            "<?php\n"
            "function processOrder($orderId) {\n"
            "    $order = getOrder($orderId);\n"
            "    $items = $order->getItems();\n"
            "    $customer = new Customer();\n"
            "    foreach (getProducts() as $product) {}\n"
            "    try {} catch (OrderException $e) {}\n"
            "}\n"
        )
        bindings = extract_variable_bindings(code, "php")
        binding_map = {b.variable: b for b in bindings}

        assert "order" in binding_map
        assert binding_map["order"].source == SOURCE_ASSIGNMENT_CALL
        assert binding_map["order"].call_name == "getOrder"

        assert "items" in binding_map
        assert binding_map["items"].source == SOURCE_ASSIGNMENT_METHOD_CALL
        assert binding_map["items"].call_name == "getItems"
        assert binding_map["items"].call_receiver == "order"

        assert "customer" in binding_map
        assert binding_map["customer"].source == SOURCE_CONSTRUCTOR
        assert binding_map["customer"].type_name == "Customer"

        assert "product" in binding_map
        assert binding_map["product"].source == SOURCE_FOR_IN
        assert binding_map["product"].call_name == "getProducts"

        assert "e" in binding_map
        assert binding_map["e"].source == SOURCE_EXCEPT_AS
        assert binding_map["e"].type_name == "OrderException"

    def test_rust_full_function(self):
        """Rust function with all binding patterns."""
        code = (
            "fn process_order(order_id: u64) {\n"
            "    let order = get_order(order_id);\n"
            "    let items = order.get_items();\n"
            "    let customer = Customer::new();\n"
            "    let config = Config::load();\n"
            "    for item in items.iter() {}\n"
            "}\n"
        )
        bindings = extract_variable_bindings(code, "rust")
        binding_map = {b.variable: b for b in bindings}

        assert "order" in binding_map
        assert binding_map["order"].source == SOURCE_ASSIGNMENT_CALL
        assert binding_map["order"].call_name == "get_order"

        assert "items" in binding_map
        assert binding_map["items"].source == SOURCE_ASSIGNMENT_METHOD_CALL
        assert binding_map["items"].call_name == "get_items"
        assert binding_map["items"].call_receiver == "order"

        assert "customer" in binding_map
        assert binding_map["customer"].source == SOURCE_CONSTRUCTOR
        assert binding_map["customer"].type_name == "Customer"

        assert "config" in binding_map
        assert binding_map["config"].source == SOURCE_ASSIGNMENT_METHOD_CALL
        assert binding_map["config"].call_name == "load"
        assert binding_map["config"].call_receiver == "Config"

        assert "item" in binding_map
        assert binding_map["item"].source == SOURCE_FOR_IN
        assert binding_map["item"].call_name == "iter"
        assert binding_map["item"].call_receiver == "items"


# =============================================================================
# MODULE RESOLVER INTEGRATION
# =============================================================================


class TestModuleResolverIntegration:
    """Test that module resolvers correctly classify imports per language."""

    @pytest.mark.parametrize("language,module,expected_external", [
        # Python
        ("python", "os", True),
        ("python", "numpy", True),
        ("python", ".utils", False),
        ("python", "myapp.models", False),
        # JavaScript
        ("javascript", "react", True),
        ("javascript", "fs", True),
        ("javascript", "node:path", True),
        ("javascript", "./utils", False),
        ("javascript", "../lib/foo", False),
        # PHP
        ("php", "Illuminate\\Support\\Collection", True),
        ("php", "Symfony\\Component\\HttpFoundation", True),
        ("php", "App\\Models\\User", False),
        ("php", "MyCompany\\Auth\\Provider", False),
        # Rust
        ("rust", "std::collections::HashMap", True),
        ("rust", "serde::Serialize", True),
        ("rust", "crate::storage::Backend", False),
        ("rust", "super::utils", False),
        ("rust", "self::helper", False),
    ])
    def test_external_module_detection(self, language, module, expected_external):
        """Verify external module detection across all languages."""
        resolver = get_module_resolver(language)
        assert resolver is not None
        assert resolver.is_external_module(module) is expected_external

    @pytest.mark.parametrize("language,module,caller,expected_substr", [
        # Python
        ("python", "magaldi_core.storage", None, "magaldi_core/storage.py"),
        ("python", ".utils", "src/myapp/views.py", "src/myapp/utils.py"),
        # JavaScript
        ("javascript", "./utils", "src/app.ts", "src/utils.ts"),
        ("javascript", "../lib/foo", "src/app/main.ts", "src/lib/foo.ts"),
        # PHP
        ("php", "App\\Models\\User", None, "app/Models/User.php"),
        # Rust
        ("rust", "crate::storage", None, "src/storage.rs"),
        ("rust", "super::utils", "src/storage/backend.rs", "src/utils.rs"),
    ])
    def test_module_to_file_paths(self, language, module, caller, expected_substr):
        """Verify module-to-file-path resolution across all languages."""
        resolver = get_module_resolver(language)
        assert resolver is not None
        paths = resolver.module_to_file_paths(module, caller)
        assert any(expected_substr in p for p in paths), (
            f"Expected '{expected_substr}' in paths {paths}"
        )

    @pytest.mark.parametrize("language,module", [
        ("javascript", "react"),
        ("rust", "std::collections::HashMap"),
    ])
    def test_external_module_returns_empty_paths(self, language, module):
        """External modules should return empty file paths (JS, Rust).

        Note: PHP resolver generates file paths even for external namespaces,
        because the externality check happens in _lookup_element_by_import
        before calling module_to_file_paths.
        """
        resolver = get_module_resolver(language)
        assert resolver is not None
        paths = resolver.module_to_file_paths(module)
        assert paths == []

    def test_unsupported_language_returns_none(self):
        """Languages without resolvers return None."""
        assert get_module_resolver("bash") is None
        assert get_module_resolver("unknown") is None
