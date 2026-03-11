"""Tests for the heuristic pre-filter for variable scoring."""

from __future__ import annotations

from magaldi_core.variable_scoring.heuristic_filter import (
    _normalize_raw_code,
    apply_heuristic_filter,
    should_drop_variable,
)


class TestSingleLetterDrop:
    """Rule 1: Single-character names should be dropped."""

    def test_single_letter_lowercase(self):
        drop, reason = should_drop_variable("i", "i = 0")
        assert drop
        assert reason == "single_letter"

    def test_single_letter_underscore(self):
        drop, reason = should_drop_variable("_", "_ = foo()")
        assert drop
        assert reason == "single_letter"

    def test_single_letter_various(self):
        for letter in "abcdefghjklmnpqrstuvwxyz":
            drop, _ = should_drop_variable(letter, f"{letter} = something()")
            assert drop, f"Expected {letter!r} to be dropped"

    def test_uppercase_single_kept(self):
        """Single uppercase letters are still dropped (they're rarely meaningful)."""
        drop, _ = should_drop_variable("X", "X = 42")
        assert drop


class TestThrowawayNameDrop:
    """Rule 2: Known throwaway names with simple assignments should drop."""

    def test_tmp_simple(self):
        drop, reason = should_drop_variable("tmp", "tmp = process(data)")
        assert drop
        assert reason == "throwaway_name"

    def test_result_simple(self):
        drop, reason = should_drop_variable("result", "result = compute(x)")
        assert drop
        assert reason == "throwaway_name"

    def test_data_simple(self):
        drop, reason = should_drop_variable("data", "data = load()")
        assert drop
        assert reason == "throwaway_name"

    def test_err_simple(self):
        drop, reason = should_drop_variable("err", "err = validate(input)")
        assert drop
        assert reason == "throwaway_name"

    def test_idx_simple(self):
        drop, reason = should_drop_variable("idx", "idx = find_index(items)")
        assert drop
        assert reason == "throwaway_name"

    def test_throwaway_multiline_dropped(self):
        """Multi-line assignments with throwaway names are still dropped."""
        code = """data = {
            'host': 'localhost',
            'port': 8080,
            'debug': True,
        }"""
        drop, reason = should_drop_variable("data", code)
        assert drop
        assert reason == "throwaway_name"

    def test_throwaway_with_type_annotation(self):
        drop, _ = should_drop_variable("result", "result: str = compute()")
        assert drop

    def test_request_dropped(self):
        drop, reason = should_drop_variable("req", "req = build_request()")
        assert drop
        assert reason == "throwaway_name"

    def test_request_full_dropped(self):
        drop, reason = should_drop_variable("request", "request = make_request()")
        assert drop
        assert reason == "throwaway_name"

    def test_params_dropped(self):
        drop, reason = should_drop_variable("params", "params = parse_args()")
        assert drop
        assert reason == "throwaway_name"

    def test_props_dropped(self):
        drop, reason = should_drop_variable("props", "props = get_properties()")
        assert drop
        assert reason == "throwaway_name"

    def test_instance_dropped(self):
        drop, reason = should_drop_variable("instance", "instance = create()")
        assert drop
        assert reason == "throwaway_name"

    def test_len_dropped(self):
        drop, reason = should_drop_variable("len", "len = len(items)")
        assert drop
        assert reason == "throwaway_name"

    def test_length_dropped(self):
        drop, reason = should_drop_variable("length", "length = len(items)")
        assert drop
        assert reason == "throwaway_name"

    def test_offset_dropped(self):
        drop, reason = should_drop_variable("offset", "offset = calculate_offset()")
        assert drop
        assert reason == "throwaway_name"

    def test_inner_dropped(self):
        drop, reason = should_drop_variable("inner", "inner = unwrap(outer)")
        assert drop
        assert reason == "throwaway_name"

    def test_parsed_dropped(self):
        drop, reason = should_drop_variable("parsed", "parsed = parse(raw)")
        assert drop
        assert reason == "throwaway_name"

    def test_func_dropped(self):
        drop, reason = should_drop_variable("func", "func = get_handler()")
        assert drop
        assert reason == "throwaway_name"

    def test_foo_dropped(self):
        drop, reason = should_drop_variable("foo", "foo = 42")
        assert drop
        assert reason == "throwaway_name"

    def test_meta_dropped(self):
        drop, reason = should_drop_variable("meta", "meta = extract_metadata()")
        assert drop
        assert reason == "throwaway_name"

    def test_metadata_dropped(self):
        drop, reason = should_drop_variable("metadata", "metadata = {}")
        assert drop
        assert reason == "throwaway_name"

    def test_desc_dropped(self):
        drop, reason = should_drop_variable("desc", 'desc = "some description"')
        assert drop
        assert reason == "throwaway_name"

    def test_expected_dropped(self):
        drop, reason = should_drop_variable("expected", "expected = compute_expected()")
        assert drop
        assert reason == "throwaway_name"

    def test_actual_dropped(self):
        drop, reason = should_drop_variable("actual", "actual = run_test()")
        assert drop
        assert reason == "throwaway_name"


class TestFunctionCallResultDrop:
    """Rule 3: Generic function-call results with lowercase names."""

    def test_generic_short_name_call(self):
        """Short generic name that isn't a throwaway — caught by Rule 3."""
        drop, reason = should_drop_variable("zap", "zap = bar()")
        assert drop
        assert "call_result" in reason

    def test_throwaway_name_call(self):
        drop, _ = should_drop_variable("resp", "resp = requests.get(url)")
        assert drop

    def test_framework_app_kept(self):
        drop, _ = should_drop_variable("app", "app = Flask(__name__)")
        assert not drop

    def test_framework_logger_kept(self):
        drop, _ = should_drop_variable("logger", "logger = logging.getLogger(__name__)")
        assert not drop

    def test_framework_client_kept(self):
        drop, _ = should_drop_variable("client", "client = OpenSearch()")
        assert not drop

    def test_framework_db_kept(self):
        drop, _ = should_drop_variable("db", "db = create_engine(url)")
        assert not drop

    def test_framework_suffix_kept(self):
        """Names with framework suffixes like redis_client are kept."""
        drop, _ = should_drop_variable("redis_client", "redis_client = Redis()")
        assert not drop

    def test_framework_prefix_kept(self):
        drop, _ = should_drop_variable("cache_pool", "cache_pool = ConnectionPool()")
        assert not drop

    def test_uppercase_call_kept(self):
        """Any uppercase in name suggests it's intentional."""
        drop, _ = should_drop_variable("MyClass", "MyClass = create_class()")
        assert not drop

    def test_longer_name_call_kept(self):
        """Names longer than 4 chars that aren't throwaway are kept."""
        drop, _ = should_drop_variable("processor", "processor = build_pipeline()")
        assert not drop


class TestAlwaysKeep:
    """Variables that should always be kept regardless of rules."""

    def test_upper_case_constants(self):
        drop, _ = should_drop_variable("MAX_RETRIES", "MAX_RETRIES = 3")
        assert not drop

    def test_upper_case_config(self):
        drop, _ = should_drop_variable("DATABASE_URL", 'DATABASE_URL = "postgres://..."')
        assert not drop

    def test_dunder_all(self):
        drop, _ = should_drop_variable("__all__", '__all__ = ["foo", "bar"]')
        assert not drop

    def test_dunder_version(self):
        drop, _ = should_drop_variable("__version__", '__version__ = "1.0.0"')
        assert not drop

    def test_multiline_dict(self):
        code = """SETTINGS = {
            'timeout': 30,
            'retries': 3,
            'debug': False,
        }"""
        drop, _ = should_drop_variable("SETTINGS", code)
        assert not drop

    def test_multiline_list(self):
        code = """ALLOWED_HOSTS = [
            'localhost',
            '127.0.0.1',
            '0.0.0.0',
        ]"""
        drop, _ = should_drop_variable("ALLOWED_HOSTS", code)
        assert not drop


class TestApplyHeuristicFilter:
    """Test the batch filter function."""

    def test_mixed_variables(self):
        variables = [
            ("id1", "app.py", "i", "i = 0"),
            ("id2", "app.py", "MAX_RETRIES", "MAX_RETRIES = 3"),
            ("id3", "app.py", "tmp", "tmp = process()"),
            ("id4", "app.py", "app", "app = Flask(__name__)"),
            ("id5", "app.py", "__all__", '__all__ = ["foo"]'),
        ]

        kept, drops = apply_heuristic_filter(variables)

        # i and tmp should be dropped
        assert len(drops) == 2
        assert "id1" in drops
        assert "id3" in drops

        # MAX_RETRIES, app, __all__ should be kept
        kept_ids = [eid for eid, _, _, _ in kept]
        assert "id2" in kept_ids
        assert "id4" in kept_ids
        assert "id5" in kept_ids

    def test_empty_input(self):
        kept, drops = apply_heuristic_filter([])
        assert kept == []
        assert drops == {}

    def test_all_kept(self):
        variables = [
            ("id1", "config.py", "DATABASE_URL", 'DATABASE_URL = "postgres://..."'),
            ("id2", "config.py", "SECRET_KEY", 'SECRET_KEY = "abc123"'),
        ]
        kept, drops = apply_heuristic_filter(variables)
        assert len(kept) == 2
        assert len(drops) == 0

    def test_all_dropped(self):
        variables = [
            ("id1", "main.py", "i", "i = 0"),
            ("id2", "main.py", "j", "j = 1"),
            ("id3", "main.py", "tmp", "tmp = foo()"),
        ]
        kept, drops = apply_heuristic_filter(variables)
        assert len(kept) == 0
        assert len(drops) == 3

    def test_drop_reasons_populated(self):
        variables = [
            ("id1", "main.py", "x", "x = 0"),
            ("id2", "main.py", "result", "result = compute()"),
        ]
        _, drops = apply_heuristic_filter(variables)
        assert drops["id1"] == "single_letter"
        assert "throwaway" in drops["id2"]


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_empty_raw_code(self):
        drop, _ = should_drop_variable("processor", "")
        assert not drop  # Can't determine pattern, keep it

    def test_short_name_call_dropped(self):
        """Short names (<=4 chars) that are function call results get dropped."""
        drop, _ = should_drop_variable("x1", "x1 = compute()")
        assert drop  # Short name + function call result

    def test_short_name_non_call_kept(self):
        """Short names that are NOT function calls are kept."""
        drop, _ = should_drop_variable("x1", "x1 = 42")
        assert not drop

    def test_mixed_case_name(self):
        drop, _ = should_drop_variable("myVar", "myVar = getValue()")
        assert not drop  # Has uppercase, kept

    def test_snake_case_longer(self):
        drop, _ = should_drop_variable("user_name", "user_name = get_name()")
        assert not drop  # Longer than 4, not throwaway

    def test_raw_code_no_assignment(self):
        """Variable with no = sign — still dropped if throwaway name."""
        drop, reason = should_drop_variable("tmp", "tmp: int")
        assert drop
        assert reason == "throwaway_name"

    def test_call_with_multiline_args(self):
        """Function call with multi-line arguments should not match simple call."""
        code = """config = create_config(
            host='localhost',
            port=8080,
        )"""
        drop, _ = should_drop_variable("config", code)
        assert not drop  # Multi-line, not a simple call


class TestNormalizeRawCode:
    """Test _normalize_raw_code strips language-specific prefixes."""

    # Python — already starts with name, no-op
    def test_python_no_change(self):
        assert _normalize_raw_code("data", "data = load()") == "data = load()"

    def test_python_type_annotation(self):
        assert _normalize_raw_code("result", "result: str = compute()") == "result: str = compute()"

    # Rust
    def test_rust_let(self):
        assert _normalize_raw_code("data", "let data = load();") == "data = load();"

    def test_rust_let_mut(self):
        assert _normalize_raw_code("tmp", "let mut tmp = process(x);") == "tmp = process(x);"

    def test_rust_let_with_type(self):
        assert _normalize_raw_code("count", "let count: usize = 0;") == "count: usize = 0;"

    # PHP
    def test_php_dollar_prefix(self):
        assert _normalize_raw_code("data", "$data = load()") == "data = load()"

    def test_php_dollar_with_call(self):
        assert _normalize_raw_code("result", "$result = $db->query($sql)") == "result = $db->query($sql)"

    # Java
    def test_java_private_int(self):
        assert _normalize_raw_code("count", "private int count = 0;") == "count = 0;"

    def test_java_public_static_final(self):
        assert _normalize_raw_code("data", "public static final String data = load();") == 'data = load();'

    def test_java_private_field(self):
        assert _normalize_raw_code("name", 'private String name = "default";') == 'name = "default";'

    # JS/TS — variable_declarator already strips const/let/var
    def test_js_already_stripped(self):
        assert _normalize_raw_code("data", "data = load()") == "data = load()"

    # Edge cases
    def test_no_match_returns_original(self):
        """If name isn't found in code, return as-is."""
        code = "something_else = 42"
        assert _normalize_raw_code("data", code) == code

    def test_empty_raw_code(self):
        assert _normalize_raw_code("foo", "") == ""

    def test_name_with_whitespace_prefix(self):
        """Leading whitespace is stripped."""
        assert _normalize_raw_code("data", "  data = 1") == "data = 1"


class TestRustVariableDrops:
    """Verify heuristic rules work with Rust raw_code format."""

    def test_rust_let_throwaway_dropped(self):
        drop, reason = should_drop_variable("tmp", "let tmp = process(data);")
        assert drop
        assert reason == "throwaway_name"

    def test_rust_let_mut_throwaway_dropped(self):
        drop, reason = should_drop_variable("result", "let mut result = compute();")
        assert drop
        assert reason == "throwaway_name"

    def test_rust_let_single_letter_dropped(self):
        drop, reason = should_drop_variable("x", "let x = 0;")
        assert drop
        assert reason == "single_letter"

    def test_rust_let_framework_kept(self):
        drop, _ = should_drop_variable("client", "let client = HttpClient::new();")
        assert not drop

    def test_rust_let_constant_kept(self):
        drop, _ = should_drop_variable("MAX_RETRIES", "let MAX_RETRIES = 3;")
        assert not drop

    def test_rust_short_call_result_dropped(self):
        drop, reason = should_drop_variable("zap", "let zap = bar();")
        assert drop
        assert "call_result" in reason


class TestPHPVariableDrops:
    """Verify heuristic rules work with PHP raw_code format."""

    def test_php_throwaway_dropped(self):
        drop, reason = should_drop_variable("data", "$data = load()")
        assert drop
        assert reason == "throwaway_name"

    def test_php_result_dropped(self):
        drop, reason = should_drop_variable("result", "$result = compute()")
        assert drop
        assert reason == "throwaway_name"

    def test_php_single_letter_dropped(self):
        drop, reason = should_drop_variable("i", "$i = 0")
        assert drop
        assert reason == "single_letter"

    def test_php_framework_kept(self):
        drop, _ = should_drop_variable("app", "$app = new Application()")
        assert not drop

    def test_php_short_call_result_dropped(self):
        drop, reason = should_drop_variable("zap", "$zap = bar()")
        assert drop
        assert "call_result" in reason


class TestJavaVariableDrops:
    """Verify heuristic rules work with Java raw_code format."""

    def test_java_throwaway_dropped(self):
        drop, reason = should_drop_variable("result", "int result = compute();")
        assert drop
        assert reason == "throwaway_name"

    def test_java_private_throwaway_dropped(self):
        drop, reason = should_drop_variable("data", "private String data = load();")
        assert drop
        assert reason == "throwaway_name"

    def test_java_framework_kept(self):
        drop, _ = should_drop_variable("logger", "private static final Logger logger = LoggerFactory.getLogger(Foo.class);")
        assert not drop

    def test_java_constant_kept(self):
        drop, _ = should_drop_variable("MAX_SIZE", "public static final int MAX_SIZE = 100;")
        assert not drop

    def test_java_short_call_result_dropped(self):
        drop, reason = should_drop_variable("zap", "final Zap zap = create();")
        assert drop
        assert "call_result" in reason


class TestGenericTypeAnnotations:
    """Verify type annotations with generics work correctly."""

    def test_python_generic_type(self):
        drop, _ = should_drop_variable("data", "data: list[str] = load()")
        assert drop

    def test_typescript_generic_type(self):
        drop, _ = should_drop_variable("data", "data: Map<string, number> = load()")
        assert drop

    def test_java_generic_field(self):
        drop, _ = should_drop_variable("items", "private List<String> items = new ArrayList<>();")
        assert drop

    def test_rust_generic_type(self):
        drop, _ = should_drop_variable("data", "let data: Vec<String> = load();")
        assert drop
