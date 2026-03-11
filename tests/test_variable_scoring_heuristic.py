"""Tests for the heuristic pre-filter for variable scoring."""

from __future__ import annotations

from magaldi_core.variable_scoring.heuristic_filter import (
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

    def test_throwaway_multiline_kept(self):
        """Multi-line assignments with throwaway names are kept (might be dicts/lists)."""
        code = """data = {
            'host': 'localhost',
            'port': 8080,
            'debug': True,
        }"""
        drop, _ = should_drop_variable("data", code)
        assert not drop

    def test_throwaway_with_type_annotation(self):
        drop, _ = should_drop_variable("result", "result: str = compute()")
        assert drop


class TestFunctionCallResultDrop:
    """Rule 3: Generic function-call results with lowercase names."""

    def test_generic_short_name_call(self):
        drop, reason = should_drop_variable("foo", "foo = bar()")
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
        drop, _ = should_drop_variable("foo", "")
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
        """Variable with no = sign (e.g. declaration only)."""
        drop, _ = should_drop_variable("tmp", "tmp: int")
        assert not drop  # Not a simple assignment pattern

    def test_call_with_multiline_args(self):
        """Function call with multi-line arguments should not match simple call."""
        code = """config = create_config(
            host='localhost',
            port=8080,
        )"""
        drop, _ = should_drop_variable("config", code)
        assert not drop  # Multi-line, not a simple call
