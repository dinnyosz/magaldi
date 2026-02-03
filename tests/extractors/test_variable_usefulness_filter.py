"""Tests for the variable/constant usefulness filter.

This module tests that the usefulness filter correctly:
- Keeps useful variables (constants, literals, type aliases, etc.)
- Skips transient variables (instance creations, function call results, etc.)
"""

import pytest
from magaldi_core.extractors.usefulness_filter import (
    is_useful_name,
    is_useful_value_type,
    is_useful_factory_call,
    should_skip_variable,
    SKIP_NAMES,
    USEFUL_FACTORIES,
)


class TestIsUsefulName:
    """Test the is_useful_name function."""

    def test_skips_temp_names(self):
        """Short/temp names like i, j, tmp should be skipped."""
        for name in ["i", "j", "k", "tmp", "temp", "res", "val", "_"]:
            is_useful, reason = is_useful_name(name)
            assert not is_useful, f"{name} should be skipped"
            assert reason == "temp_name"

    def test_keeps_constants(self):
        """UPPER_CASE names are always useful."""
        for name in ["MAX_RETRIES", "API_URL", "CONFIG_PATH"]:
            is_useful, reason = is_useful_name(name)
            assert is_useful, f"{name} should be kept"
            assert reason == ""

    def test_keeps_dunder_names(self):
        """Dunder names like __all__, __version__ are useful."""
        for name in ["__all__", "__version__", "__author__"]:
            is_useful, reason = is_useful_name(name)
            assert is_useful, f"{name} should be kept"
            assert reason == ""

    def test_regular_names_pass_through(self):
        """Regular names pass through (usefulness determined by value)."""
        for name in ["client", "service", "handler"]:
            is_useful, reason = is_useful_name(name)
            assert is_useful, f"{name} should pass through"


class TestIsUsefulValueType:
    """Test the is_useful_value_type function."""

    def test_python_literals_are_useful(self):
        """Python literal types should be useful."""
        for value_type in ["string", "integer", "float", "list", "dictionary"]:
            is_useful, reason = is_useful_value_type(value_type, "python")
            assert is_useful, f"Python {value_type} should be useful"

    def test_javascript_literals_are_useful(self):
        """JavaScript literal types should be useful."""
        for value_type in ["string", "number", "array", "object", "regex"]:
            is_useful, reason = is_useful_value_type(value_type, "javascript")
            assert is_useful, f"JS {value_type} should be useful"

    def test_php_literals_are_useful(self):
        """PHP literal types should be useful."""
        for value_type in ["string", "integer", "float", "array_creation_expression"]:
            is_useful, reason = is_useful_value_type(value_type, "php")
            assert is_useful, f"PHP {value_type} should be useful"

    def test_rust_literals_are_useful(self):
        """Rust literal types should be useful."""
        for value_type in ["string_literal", "integer_literal", "array_expression"]:
            is_useful, reason = is_useful_value_type(value_type, "rust")
            assert is_useful, f"Rust {value_type} should be useful"

    def test_arrow_function_is_useful(self):
        """Arrow functions should always be useful."""
        is_useful, reason = is_useful_value_type("arrow_function", "javascript")
        assert is_useful


class TestIsUsefulFactoryCall:
    """Test the is_useful_factory_call function."""

    def test_enum_factories_are_useful(self):
        """Enum, IntEnum, StrEnum calls should be useful."""
        for func_name in ["Enum", "IntEnum", "StrEnum", "Flag"]:
            is_useful, reason = is_useful_factory_call(func_name)
            assert is_useful, f"{func_name} should be useful"

    def test_typevar_is_useful(self):
        """TypeVar and similar typing constructs should be useful."""
        for func_name in ["TypeVar", "ParamSpec", "TypeVarTuple", "NewType"]:
            is_useful, reason = is_useful_factory_call(func_name)
            assert is_useful, f"{func_name} should be useful"

    def test_namedtuple_is_useful(self):
        """namedtuple calls should be useful."""
        for func_name in ["namedtuple", "NamedTuple"]:
            is_useful, reason = is_useful_factory_call(func_name)
            assert is_useful, f"{func_name} should be useful"

    def test_re_compile_is_useful(self):
        """re.compile calls should be useful."""
        is_useful, reason = is_useful_factory_call("re.compile")
        assert is_useful

    def test_logging_getlogger_is_useful(self):
        """logging.getLogger calls should be useful."""
        is_useful, reason = is_useful_factory_call("logging.getLogger")
        assert is_useful

    def test_threading_lock_is_useful(self):
        """threading.Lock and similar should be useful."""
        for func_name in ["Lock", "RLock", "Semaphore", "threading.Lock"]:
            is_useful, reason = is_useful_factory_call(func_name)
            assert is_useful, f"{func_name} should be useful"

    def test_instance_creation_is_skipped(self):
        """PascalCase class instantiations should be skipped."""
        for func_name in ["HttpClient", "ServiceFactory", "UserRepository"]:
            is_useful, reason = is_useful_factory_call(func_name)
            assert not is_useful, f"{func_name} should be skipped"
            assert reason == "instance_creation"

    def test_method_call_results_are_skipped(self):
        """Method call results like obj.method() should be skipped."""
        for func_name in ["requests.get", "factory.create", "self.build"]:
            is_useful, reason = is_useful_factory_call(func_name)
            assert not is_useful, f"{func_name} should be skipped"
            assert reason == "method_call_result"

    def test_function_call_results_are_skipped(self):
        """Lowercase function call results should be skipped."""
        for func_name in ["process_items", "fetch_data", "get_config"]:
            is_useful, reason = is_useful_factory_call(func_name)
            assert not is_useful, f"{func_name} should be skipped"
            assert reason == "function_call_result"


class TestShouldSkipVariable:
    """Test the main should_skip_variable function."""

    def test_keeps_upper_case_constants(self):
        """UPPER_CASE names are always kept regardless of value."""
        skip, reason = should_skip_variable(
            name="MAX_RETRIES",
            value_type="call",
            func_name="get_max_retries",
            language="python",
            is_module_level=True,
        )
        assert not skip, "UPPER_CASE constant should be kept"

    def test_keeps_dunder_names(self):
        """Dunder names are always kept."""
        skip, reason = should_skip_variable(
            name="__all__",
            value_type="list",
            func_name=None,
            language="python",
            is_module_level=True,
        )
        assert not skip, "__all__ should be kept"

    def test_keeps_literal_values(self):
        """Literal values like strings, numbers, lists should be kept."""
        skip, reason = should_skip_variable(
            name="default_timeout",
            value_type="integer",
            func_name=None,
            language="python",
            is_module_level=True,
        )
        assert not skip, "Literal value should be kept"

    def test_keeps_type_aliases(self):
        """Type aliases (identifier values) should be kept."""
        skip, reason = should_skip_variable(
            name="UserId",
            value_type="identifier",
            func_name=None,
            language="python",
            is_module_level=True,
        )
        assert not skip, "Type alias should be kept"

    def test_keeps_enum_definitions(self):
        """Enum() calls should be kept."""
        skip, reason = should_skip_variable(
            name="Status",
            value_type="call",
            func_name="Enum",
            language="python",
            is_module_level=True,
        )
        assert not skip, "Enum definition should be kept"

    def test_keeps_typevar_definitions(self):
        """TypeVar() calls should be kept."""
        skip, reason = should_skip_variable(
            name="T",
            value_type="call",
            func_name="TypeVar",
            language="python",
            is_module_level=True,
        )
        assert not skip, "TypeVar definition should be kept"

    def test_skips_instance_creation(self):
        """Instance creations like SomeClass() should be skipped."""
        skip, reason = should_skip_variable(
            name="client",
            value_type="call",
            func_name="HttpClient",
            language="python",
            is_module_level=True,
        )
        assert skip, "Instance creation should be skipped"
        assert reason == "instance_creation"

    def test_skips_method_call_results(self):
        """Method call results like requests.get() should be skipped."""
        skip, reason = should_skip_variable(
            name="response",
            value_type="call",
            func_name="requests.get",
            language="python",
            is_module_level=True,
        )
        assert skip, "Method call result should be skipped"
        assert reason == "method_call_result"

    def test_skips_function_call_results(self):
        """Function call results like process_data() should be skipped."""
        skip, reason = should_skip_variable(
            name="data",
            value_type="call",
            func_name="process_data",
            language="python",
            is_module_level=True,
        )
        assert skip, "Function call result should be skipped"
        assert reason == "function_call_result"

    def test_skips_temp_names(self):
        """Temp names like i, j, tmp should be skipped."""
        for name in ["i", "j", "tmp", "temp"]:
            skip, reason = should_skip_variable(
                name=name,
                value_type="integer",
                func_name=None,
                language="python",
                is_module_level=True,
            )
            assert skip, f"{name} should be skipped"
            assert reason == "temp_name"


class TestJavaScriptSpecificCases:
    """Test JavaScript-specific usefulness criteria."""

    def test_keeps_arrow_functions(self):
        """Arrow functions should be kept."""
        skip, reason = should_skip_variable(
            name="handleClick",
            value_type="arrow_function",
            func_name=None,
            language="javascript",
            is_module_level=True,
        )
        assert not skip, "Arrow function should be kept"

    def test_keeps_regex_literals(self):
        """Regex literals should be kept."""
        skip, reason = should_skip_variable(
            name="PATTERN",
            value_type="regex",
            func_name=None,
            language="javascript",
            is_module_level=True,
        )
        assert not skip, "Regex literal should be kept"

    def test_keeps_template_strings(self):
        """Template strings should be kept."""
        skip, reason = should_skip_variable(
            name="template",
            value_type="template_string",
            func_name=None,
            language="javascript",
            is_module_level=True,
        )
        assert not skip, "Template string should be kept"

    def test_keeps_react_context(self):
        """React.createContext should be kept."""
        skip, reason = should_skip_variable(
            name="ThemeContext",
            value_type="call",
            func_name="createContext",
            language="javascript",
            is_module_level=True,
        )
        assert not skip, "createContext should be kept"


class TestPhpSpecificCases:
    """Test PHP-specific usefulness criteria."""

    def test_keeps_array_creation(self):
        """Array creation expressions should be kept."""
        skip, reason = should_skip_variable(
            name="config",
            value_type="array_creation_expression",
            func_name=None,
            language="php",
            is_module_level=True,
        )
        assert not skip, "Array creation should be kept"

    def test_skips_object_creation(self):
        """Object creations with new should be skipped."""
        skip, reason = should_skip_variable(
            name="client",
            value_type="object_creation_expression",
            func_name="HttpClient",
            language="php",
            is_module_level=True,
        )
        assert skip, "Object creation should be skipped"


class TestASTBasedInstanceDetection:
    """Test that instance creation is detected via AST node type, not naming."""

    def test_js_new_expression_skipped_regardless_of_case(self):
        """JS new expressions should be skipped even with non-PascalCase names."""
        # camelCase constructor
        skip, reason = should_skip_variable(
            name="client",
            value_type="new_expression",
            func_name="httpClient",  # camelCase
            language="javascript",
            is_module_level=True,
        )
        assert skip, "new httpClient() should be skipped"
        assert reason == "instance_creation"

        # snake_case constructor
        skip, reason = should_skip_variable(
            name="client",
            value_type="new_expression",
            func_name="http_client",  # snake_case
            language="javascript",
            is_module_level=True,
        )
        assert skip, "new http_client() should be skipped"
        assert reason == "instance_creation"

    def test_php_object_creation_skipped_regardless_of_case(self):
        """PHP object creations should be skipped even with non-PascalCase names."""
        skip, reason = should_skip_variable(
            name="service",
            value_type="object_creation_expression",
            func_name="myService",  # camelCase
            language="php",
            is_module_level=True,
        )
        assert skip, "new myService() should be skipped"
        assert reason == "instance_creation"

    def test_js_new_map_is_kept(self):
        """new Map() should be kept as a useful built-in."""
        skip, reason = should_skip_variable(
            name="cache",
            value_type="new_expression",
            func_name="Map",
            language="javascript",
            is_module_level=True,
        )
        assert not skip, "new Map() should be kept"

    def test_js_new_set_is_kept(self):
        """new Set() should be kept as a useful built-in."""
        skip, reason = should_skip_variable(
            name="items",
            value_type="new_expression",
            func_name="Set",
            language="javascript",
            is_module_level=True,
        )
        assert not skip, "new Set() should be kept"

    def test_js_new_regexp_is_kept(self):
        """new RegExp() should be kept as a useful built-in."""
        skip, reason = should_skip_variable(
            name="pattern",
            value_type="new_expression",
            func_name="RegExp",
            language="javascript",
            is_module_level=True,
        )
        assert not skip, "new RegExp() should be kept"


class TestRustSpecificCases:
    """Test Rust-specific usefulness criteria."""

    def test_keeps_string_literals(self):
        """String literals should be kept."""
        skip, reason = should_skip_variable(
            name="DEFAULT_URL",
            value_type="string_literal",
            func_name=None,
            language="rust",
            is_module_level=True,
        )
        assert not skip, "String literal should be kept"

    def test_keeps_mutex_new(self):
        """Mutex::new and similar should be kept."""
        skip, reason = should_skip_variable(
            name="INSTANCE",
            value_type="call",
            func_name="Mutex::new",
            language="rust",
            is_module_level=True,
        )
        assert not skip, "Mutex::new should be kept"

    def test_keeps_arc_new(self):
        """Arc::new should be kept."""
        skip, reason = should_skip_variable(
            name="shared",
            value_type="call",
            func_name="Arc::new",
            language="rust",
            is_module_level=True,
        )
        assert not skip, "Arc::new should be kept"
