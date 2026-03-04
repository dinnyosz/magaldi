"""Tests for shared.text_utils module."""

from shared.text_utils import humanize_name


class TestHumanizeName:
    """Tests for humanize_name identifier splitting."""

    def test_snake_case(self):
        assert humanize_name("test_parse_html") == "test parse html"

    def test_camel_case(self):
        assert humanize_name("parseHTMLContent") == "parse html content"

    def test_pascal_case(self):
        assert humanize_name("TestUserAuthentication") == "test user authentication"

    def test_screaming_snake(self):
        assert humanize_name("MAX_RETRY_COUNT") == "max retry count"

    def test_mixed_snake_and_camel(self):
        assert humanize_name("test_parseHTMLContent") == "test parse html content"

    def test_acronym_at_end(self):
        assert humanize_name("parseHTML") == "parse html"

    def test_acronym_at_start(self):
        assert humanize_name("HTMLParser") == "html parser"

    def test_single_word_lowercase(self):
        assert humanize_name("test") == "test"

    def test_single_word_uppercase(self):
        assert humanize_name("TEST") == "test"

    def test_empty_string(self):
        assert humanize_name("") == ""

    def test_leading_underscore(self):
        assert humanize_name("_private_method") == "private method"

    def test_double_underscore_dunder(self):
        assert humanize_name("__init__") == "init"

    def test_trailing_underscore(self):
        assert humanize_name("class_") == "class"

    def test_numbers_in_name(self):
        assert humanize_name("test_v2_parser") == "test v2 parser"

    def test_numbers_camel(self):
        assert humanize_name("getV2Parser") == "get v2 parser"

    def test_single_letter_parts(self):
        assert humanize_name("a_b_c") == "a b c"

    def test_consecutive_underscores(self):
        assert humanize_name("test__double") == "test double"

    def test_simple_camel_case(self):
        assert humanize_name("getUserById") == "get user by id"

    def test_all_caps_single_word(self):
        assert humanize_name("HTTP") == "http"

    def test_file_stem(self):
        """Test with typical test file names (without extension)."""
        assert humanize_name("test_user_auth") == "test user auth"

    def test_conftest(self):
        assert humanize_name("conftest") == "conftest"

    def test_complex_mixed(self):
        """Test a realistic complex identifier."""
        assert humanize_name("test_shouldParseJSON_whenValid") == "test should parse json when valid"
