"""Tests for Bash function signature extraction.

Bash functions don't have formal parameters, but we capture the function
definition line as the signature (e.g., "get_os()", "function setup()").
"""

from pathlib import Path

from magaldi_core.change_detection import FileInfo
from magaldi_core.code_parser import BashParser


class TestBashFunctionSignature:
    """Tests for Bash function signature extraction."""

    def test_name_parens_syntax(self):
        """Functions defined as 'name() {' get signature 'name()'."""
        code = 'get_os() {\n    echo "os"\n}'
        parser = BashParser()
        file_info = FileInfo(
            relative_path="test.sh",
            absolute_path=Path("/fake/test.sh"),
            language="bash",
        )
        elements = parser.parse(code, file_info, "scope", "repo", "main")
        func = next(e for e in elements if e.name == "get_os")
        assert func.signature == "get_os()"

    def test_function_keyword_parens_syntax(self):
        """Functions defined as 'function name() {' get signature 'function name()'."""
        code = 'function setup() {\n    apt-get update\n}'
        parser = BashParser()
        file_info = FileInfo(
            relative_path="test.sh",
            absolute_path=Path("/fake/test.sh"),
            language="bash",
        )
        elements = parser.parse(code, file_info, "scope", "repo", "main")
        func = next(e for e in elements if e.name == "setup")
        assert func.signature == "function setup()"

    def test_function_keyword_no_parens_syntax(self):
        """Functions defined as 'function name {' get signature 'function name'."""
        code = 'function teardown {\n    rm -rf /tmp/test-*\n}'
        parser = BashParser()
        file_info = FileInfo(
            relative_path="test.sh",
            absolute_path=Path("/fake/test.sh"),
            language="bash",
        )
        elements = parser.parse(code, file_info, "scope", "repo", "main")
        func = next(e for e in elements if e.name == "teardown")
        assert func.signature == "function teardown"

    def test_one_liner_function(self):
        """One-liner functions get correct signature."""
        code = 'die() { echo "$@" >&2; exit 1; }'
        parser = BashParser()
        file_info = FileInfo(
            relative_path="test.sh",
            absolute_path=Path("/fake/test.sh"),
            language="bash",
        )
        elements = parser.parse(code, file_info, "scope", "repo", "main")
        func = next(e for e in elements if e.name == "die")
        assert func.signature == "die()"

    def test_recovered_function_signature(self):
        """Recovered functions (from tree-sitter parse errors) also get signatures."""
        # ANSI-C quoting breaks tree-sitter, triggering recovery
        code = (
            "broken() {\n"
            "    theme=\"${theme// '[GTK'[0-9]']'}\"\n"
            "}\n"
        )
        parser = BashParser()
        file_info = FileInfo(
            relative_path="test.sh",
            absolute_path=Path("/fake/test.sh"),
            language="bash",
        )
        elements = parser.parse(code, file_info, "scope", "repo", "main")
        func = next(e for e in elements if e.name == "broken")
        assert func.signature == "broken()"
