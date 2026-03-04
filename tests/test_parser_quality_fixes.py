"""Tests for parser quality fixes from test repo reports.

Covers:
- Python dual function/method extraction for decorated class members
- JS/TS arrow function signature and parameter extraction
- JS/TS new ClassName() constructor call extraction
- Multi-line Python signature docstring detection
- Rust type_alias extraction
- clean_summary() prompt leak, signature echo, and ThisFunction detection
"""

from pathlib import Path

import pytest

from magaldi_core.change_detection import FileInfo
from magaldi_core.parsers.base import extract_docstring, extract_preceding_doc_comment
from shared.ai.prompts import clean_summary


# =============================================================================
# Python: Decorated class member dual-extraction bug
# =============================================================================


class TestPythonDecoratedClassMembers:
    """Verify decorated methods inside classes are NOT extracted as top-level functions."""

    def _parse_python(self, code: str):
        from magaldi_core.code_parser import PythonParser

        parser = PythonParser()
        file_info = FileInfo(
            relative_path="test.py",
            absolute_path=Path("/fake/test.py"),
            language="python",
        )
        return parser.parse(code, file_info, "scope", "repo", "main")

    def test_property_not_extracted_as_function(self):
        """@property def ok(self) inside a class should only be a method, not a function."""
        code = """\
class Response:
    @property
    def ok(self):
        return self.status_code < 400

    @staticmethod
    def from_dict(data):
        return Response()
"""
        elements = self._parse_python(code)

        # Should have exactly one 'ok' element, as a method
        ok_elems = [e for e in elements if e.name == "ok"]
        assert len(ok_elems) == 1, f"Expected 1 'ok' element, got {len(ok_elems)}: {[(e.name, e.element_type) for e in ok_elems]}"
        assert ok_elems[0].element_type == "method"

        # from_dict should also be a method, not a function
        fd_elems = [e for e in elements if e.name == "from_dict"]
        assert len(fd_elems) == 1
        assert fd_elems[0].element_type == "method"

    def test_decorated_top_level_function_still_extracted(self):
        """Decorated functions at module level should still be extracted as functions."""
        code = """\
import functools

@functools.lru_cache
def expensive_compute(n):
    return n * n
"""
        elements = self._parse_python(code)

        func_elems = [e for e in elements if e.name == "expensive_compute"]
        assert len(func_elems) == 1
        assert func_elems[0].element_type == "function"


# =============================================================================
# JS/TS: Arrow function signature and parameter extraction
# =============================================================================


class TestJSArrowFunctionSignature:
    """Verify arrow functions get proper signature, parameters, and return type."""

    def _parse_js(self, code: str, language: str = "javascript"):
        from magaldi_core.code_parser import JavaScriptParser

        parser = JavaScriptParser(language)
        file_info = FileInfo(
            relative_path="test.js",
            absolute_path=Path("/fake/test.js"),
            language=language,
        )
        return parser.parse(code, file_info, "scope", "repo", "main")

    def test_arrow_function_has_signature(self):
        code = "const add = (a, b) => a + b;\n"
        elements = self._parse_js(code)

        add_elem = next((e for e in elements if e.name == "add"), None)
        assert add_elem is not None
        assert add_elem.signature is not None
        assert "add" in add_elem.signature
        assert "=>" in add_elem.signature

    def test_arrow_function_has_parameters(self):
        code = "const greet = (name, greeting) => `${greeting}, ${name}`;\n"
        elements = self._parse_js(code)

        greet_elem = next((e for e in elements if e.name == "greet"), None)
        assert greet_elem is not None
        assert greet_elem.parameters is not None
        param_names = [p["name"] for p in greet_elem.parameters]
        assert "name" in param_names
        assert "greeting" in param_names

    def test_async_arrow_function_signature(self):
        code = "const fetchData = async (url) => { return await fetch(url); };\n"
        elements = self._parse_js(code)

        func = next((e for e in elements if e.name == "fetchData"), None)
        assert func is not None
        assert func.signature is not None
        assert "async" in func.signature

    def test_ts_arrow_function_return_type(self):
        code = "const parse = (input: string): number => parseInt(input);\n"
        elements = self._parse_js(code, "typescript")

        func = next((e for e in elements if e.name == "parse"), None)
        assert func is not None
        assert func.return_type is not None


# =============================================================================
# JS/TS: new ClassName() constructor call extraction
# =============================================================================


class TestJSNewExpressionCalls:
    """Verify new ClassName() calls are extracted."""

    def test_new_expression_extracted(self):
        from magaldi_core.extractors.javascript.call_extractor import extract_javascript_calls
        from magaldi_core.tree_sitter_manager import get_manager

        code = """\
function createClient() {
    const client = new HttpClient();
    const adapter = new pkg.Adapter();
    return client;
}
"""
        manager = get_manager()
        tree = manager.parse(code.encode("utf-8"), "javascript")

        # Find the function node
        func_node = None
        for node in tree.root_node.children:
            if node.type == "function_declaration":
                func_node = node
                break

        assert func_node is not None
        calls = extract_javascript_calls(func_node)
        call_names = [(c.name, c.receiver) for c in calls]

        assert ("HttpClient", None) in call_names, f"Expected HttpClient call, got {call_names}"
        assert ("Adapter", "pkg") in call_names, f"Expected pkg.Adapter call, got {call_names}"


# =============================================================================
# Python: Multi-line signature docstring detection
# =============================================================================


class TestMultiLineSignatureDocstring:
    """Verify docstrings are found after multi-line function signatures."""

    def test_single_line_signature(self):
        lines = [
            "def simple(self):",
            '    """A simple docstring."""',
            "    pass",
        ]
        result = extract_docstring(lines, 0)
        assert result == "A simple docstring."

    def test_multi_line_signature(self):
        lines = [
            "def request(",
            "    self,",
            "    method,",
            "    url,",
            "    params=None,",
            "):",
            '    """Send a request."""',
            "    pass",
        ]
        result = extract_docstring(lines, 0)
        assert result == "Send a request."

    def test_multi_line_signature_with_return_type(self):
        lines = [
            "def process(",
            "    data: dict,",
            "    options: Options,",
            ") -> Result:",
            '    """Process data with options."""',
            "    return Result()",
        ]
        result = extract_docstring(lines, 0)
        assert result == "Process data with options."

    def test_multi_line_docstring_after_multi_line_sig(self):
        lines = [
            "def long_func(",
            "    arg1,",
            "    arg2,",
            "    arg3,",
            "):",
            '    """',
            "    This is a longer",
            "    multi-line docstring.",
            '    """',
            "    pass",
        ]
        result = extract_docstring(lines, 0)
        assert result is not None
        assert "longer" in result

    def test_no_docstring(self):
        lines = [
            "def no_doc(self):",
            "    return 42",
        ]
        result = extract_docstring(lines, 0)
        assert result is None

    def test_signature_with_noqa_comment(self):
        """Inline # noqa comment after colon should not break docstring extraction."""
        lines = [
            "def __init__(self, window_seconds: float = 300.0):  # noqa: ARG002",
            '    """Initialize throughput-by-level tracker.',
            "",
            "    Data is kept for the entire tier lifetime.",
            '    """',
            "    self._levels = {}",
        ]
        result = extract_docstring(lines, 0)
        assert result is not None
        assert result.startswith("Initialize throughput-by-level tracker.")
        assert "self._levels" not in result

    def test_signature_with_type_ignore_comment(self):
        """Inline # type: ignore comment after colon should not break extraction."""
        lines = [
            "def process(self, data: Any):  # type: ignore[override]",
            '    """Process the incoming data."""',
            "    return data",
        ]
        result = extract_docstring(lines, 0)
        assert result == "Process the incoming data."

    def test_multi_line_signature_with_noqa_comment(self):
        """Multi-line signature where closing paren line has inline comment."""
        lines = [
            "def complex_func(",
            "    arg1: str,",
            "    arg2: int,",
            "):  # noqa: PLR0913",
            '    """Handle complex arguments."""',
            "    pass",
        ]
        result = extract_docstring(lines, 0)
        assert result == "Handle complex arguments."


# =============================================================================
# Rust: type_alias extraction
# =============================================================================


class TestRustTypeAlias:
    """Verify Rust type aliases are extracted."""

    def _parse_rust(self, code: str):
        from magaldi_core.code_parser import RustParser

        parser = RustParser()
        file_info = FileInfo(
            relative_path="test.rs",
            absolute_path=Path("/fake/test.rs"),
            language="rust",
        )
        return parser.parse(code, file_info, "scope", "repo", "main")

    def test_simple_type_alias(self):
        code = "type Result<T> = std::result::Result<T, Error>;\n"
        elements = self._parse_rust(code)

        ta = next((e for e in elements if e.name == "Result" and e.element_type == "type_alias"), None)
        assert ta is not None, f"Expected type_alias 'Result', got: {[(e.name, e.element_type) for e in elements]}"
        assert ta.signature is not None

    def test_multiple_type_aliases(self):
        code = """\
type Callback = fn(i32) -> bool;
type BoxedFuture<T> = Box<dyn Future<Output = T>>;
"""
        elements = self._parse_rust(code)

        ta_names = [e.name for e in elements if e.element_type == "type_alias"]
        assert "Callback" in ta_names
        assert "BoxedFuture" in ta_names


# =============================================================================
# clean_summary(): prompt leak, signature echo, ThisFunction
# =============================================================================


class TestCleanSummaryArtifacts:
    """Verify clean_summary strips LLM artifacts."""

    def test_prompt_leak_describe(self):
        assert clean_summary("Describe this function in 2 sentences...") == ""

    def test_prompt_leak_write_only(self):
        assert clean_summary("Write ONLY the 2-3 sentence summary.") == ""

    def test_prompt_leak_answer_these(self):
        assert clean_summary("Answer these questions about the code...") == ""

    def test_prompt_leak_focus_on(self):
        assert clean_summary("Focus on the function itself.") == ""

    def test_prompt_leak_summarize(self):
        assert clean_summary("Summarize this class for AI agents.") == ""

    def test_signature_echo_python(self):
        assert clean_summary("def process(data, options):") == ""

    def test_signature_echo_js(self):
        assert clean_summary("function handleRequest(req, res) {") == ""

    def test_signature_echo_rust(self):
        assert clean_summary("fn parse(input: &str) -> Result<AST>") == ""

    def test_signature_echo_async(self):
        assert clean_summary("async def fetch_data(url):") == ""

    def test_signature_echo_const(self):
        assert clean_summary("const processItem = (item) => {") == ""

    def test_this_function_prefix(self):
        result = clean_summary("ThisFunction processes data and returns results.")
        assert not result.lower().startswith("thisfunction")
        assert "processes" in result.lower() or "data" in result.lower()

    def test_this_method_prefix(self):
        result = clean_summary("ThisMethod validates the input parameters.")
        assert not result.lower().startswith("thismethod")

    def test_valid_summary_unchanged(self):
        result = clean_summary("Processes incoming HTTP requests and routes them to handlers.")
        assert result.startswith("Processes")

    def test_think_tag_removal(self):
        result = clean_summary("<think>reasoning here</think>Handles database connections.")
        assert "think" not in result.lower()
        assert result.startswith("Handles")

    def test_orphaned_think_tag(self):
        result = clean_summary("</think>Manages the request lifecycle.")
        assert result.startswith("Manages")


# =============================================================================
# Call categorizer: resolved calls preserved
# =============================================================================


class TestCallCategorizerResolved:
    """Verify already-resolved calls are categorized as RESOLVED."""

    def test_resolved_call_categorized(self):
        from magaldi_core.extractors.call_categorizer import categorize_call
        from magaldi_core.extractors.types import CallCategory

        class MockCall:
            name = "process"
            receiver = None
            resolved_id = "scope:repo:main:utils.py:function:process:42"
            category = "unknown"

        result = categorize_call(MockCall(), "python")
        assert result == CallCategory.RESOLVED

    def test_unresolved_bare_call_unknown(self):
        from magaldi_core.extractors.call_categorizer import categorize_call
        from magaldi_core.extractors.types import CallCategory

        class MockCall:
            name = "custom_func"
            receiver = None
            resolved_id = None
            category = "unknown"

        result = categorize_call(MockCall(), "python")
        assert result == CallCategory.UNKNOWN


# =============================================================================
# Preceding doc comment extraction (all languages)
# =============================================================================


class TestPrecedingDocComment:
    """Tests for extract_preceding_doc_comment across all languages."""

    # --- JSDoc (JavaScript/TypeScript) ---

    def test_jsdoc_multiline_block(self):
        lines = [
            "/**",
            " * Handle ticket purchase for a show.",
            " * @param {string} showId - The show identifier",
            " * @returns {Ticket} The purchased ticket",
            " */",
            "function purchaseTicket(showId) {",
        ]
        result = extract_preceding_doc_comment(lines, 6, "javascript")
        assert result is not None
        assert "Handle ticket purchase" in result
        assert "@param" in result

    def test_jsdoc_single_line(self):
        lines = [
            "/** Short description */",
            "function foo() {",
        ]
        result = extract_preceding_doc_comment(lines, 2, "javascript")
        assert result == "Short description"

    def test_jsdoc_typescript(self):
        lines = [
            "/**",
            " * User configuration interface.",
            " */",
            "interface UserConfig {",
        ]
        result = extract_preceding_doc_comment(lines, 4, "typescript")
        assert result == "User configuration interface."

    def test_js_regular_comment_not_extracted(self):
        """// line comments are not extracted for JS/TS."""
        lines = [
            "// This is a regular comment",
            "function foo() {",
        ]
        result = extract_preceding_doc_comment(lines, 2, "javascript")
        assert result is None

    def test_js_non_doc_block_comment_extracted(self):
        """/* */ directly before an element is still a useful comment."""
        lines = [
            "/* Handles user authentication */",
            "function foo() {",
        ]
        result = extract_preceding_doc_comment(lines, 2, "javascript")
        assert result == "Handles user authentication"

    def test_js_multiline_non_doc_block_extracted(self):
        """Multi-line /* ... */ directly before an element is extracted."""
        lines = [
            "/*",
            " * Non-doc block comment",
            " * with multiple lines",
            " */",
            "function foo() {",
        ]
        result = extract_preceding_doc_comment(lines, 5, "javascript")
        assert result == "Non-doc block comment\nwith multiple lines"

    def test_block_comment_doesnt_bleed_across_boundaries(self):
        """A */ closing must not scan past another */ and pick up an older block."""
        lines = [
            "/** @deprecated Use newHelper instead */",
            "function oldHelper() {}",
            "",
            "/* Regular block comment */",
            "function regularComment() {}",
        ]
        # Line 4 (/* Regular... */) is directly above line 5
        result = extract_preceding_doc_comment(lines, 5, "javascript")
        assert result == "Regular block comment"

    def test_stray_closing_doesnt_bleed_into_previous_block(self):
        """A bare */ line must not scan past code into a previous block."""
        lines = [
            "/** Old JSDoc */",
            "function old() {}",
            "",
            "some code here",
            " */",  # stray closing without matching opening
            "function broken() {}",
        ]
        result = extract_preceding_doc_comment(lines, 6, "javascript")
        assert result is None

    # --- Rust doc comments ---

    def test_rust_triple_slash(self):
        lines = [
            "/// Validate instrument serial number.",
            "/// Returns true if valid.",
            "fn validate_serial(s: &str) -> bool {",
        ]
        result = extract_preceding_doc_comment(lines, 3, "rust")
        assert result is not None
        assert "Validate instrument" in result
        assert "Returns true" in result

    def test_rust_with_derive_attribute(self):
        """Doc comment above #[derive(...)] should still be extracted."""
        lines = [
            "/// A musical instrument in the orchestra.",
            "#[derive(Debug, Clone)]",
            "struct Instrument {",
        ]
        result = extract_preceding_doc_comment(lines, 3, "rust")
        assert result == "A musical instrument in the orchestra."

    def test_rust_inner_doc_not_extracted(self):
        """//! inner doc comments should NOT be extracted as element docs."""
        lines = [
            "//! Module-level documentation",
            "fn foo() {",
        ]
        result = extract_preceding_doc_comment(lines, 2, "rust")
        assert result is None

    def test_rust_block_doc_comment(self):
        lines = [
            "/**",
            " * A Rust block doc comment.",
            " * Second line.",
            " */",
            "fn documented() {",
        ]
        result = extract_preceding_doc_comment(lines, 5, "rust")
        assert result is not None
        assert "block doc comment" in result

    def test_rust_multiple_attributes_between_doc_and_element(self):
        lines = [
            "/// The main configuration.",
            "#[derive(Debug)]",
            "#[serde(rename_all = \"camelCase\")]",
            "struct Config {",
        ]
        result = extract_preceding_doc_comment(lines, 4, "rust")
        assert result == "The main configuration."

    # --- PHPDoc ---

    def test_phpdoc_block(self):
        lines = [
            "/**",
            " * Manage backstage operations.",
            " * @param string $area The backstage area",
            " */",
            "class BackstageManager {",
        ]
        result = extract_preceding_doc_comment(lines, 5, "php")
        assert result is not None
        assert "Manage backstage" in result

    def test_php_attribute_between_doc_and_class(self):
        """PHP 8 #[Attribute] between PHPDoc and class should be skipped."""
        lines = [
            "/**",
            " * A controller class.",
            " */",
            "#[Route('/api')]",
            "class ApiController {",
        ]
        result = extract_preceding_doc_comment(lines, 5, "php")
        assert result == "A controller class."

    # --- Bash ---

    def test_bash_comment_block(self):
        lines = [
            "# Check if all stage equipment is ready.",
            "# Returns 0 on success, 1 on failure.",
            "check_stage_equipment() {",
        ]
        result = extract_preceding_doc_comment(lines, 3, "bash")
        assert result is not None
        assert "stage equipment" in result

    def test_bash_skips_shebang(self):
        lines = [
            "#!/bin/bash",
            "do_thing() {",
        ]
        result = extract_preceding_doc_comment(lines, 2, "bash")
        assert result is None

    def test_bash_skips_section_markers(self):
        lines = [
            "# ========================",
            "do_thing() {",
        ]
        result = extract_preceding_doc_comment(lines, 2, "bash")
        assert result is None

    def test_bash_stops_at_section_marker(self):
        """Section marker should stop backward scanning, not include content above."""
        lines = [
            "# Some unrelated comment",
            "# --- Section ---",
            "# This function does X.",
            "do_thing() {",
        ]
        result = extract_preceding_doc_comment(lines, 4, "bash")
        assert result == "This function does X."
        assert "unrelated" not in result

    # --- Python (# comments for variables) ---

    def test_python_hash_comment_for_variable(self):
        lines = [
            "# Maximum retry count before giving up.",
            "MAX_RETRIES = 3",
        ]
        result = extract_preceding_doc_comment(lines, 2, "python")
        assert result == "Maximum retry count before giving up."

    def test_python_multiline_hash_comment(self):
        lines = [
            "# Thread-safe cache for storing summaries.",
            "# Uses LRU eviction with a configurable max size.",
            "_summary_cache: dict[str, str] = {}",
        ]
        result = extract_preceding_doc_comment(lines, 3, "python")
        assert result is not None
        assert "Thread-safe" in result
        assert "LRU eviction" in result

    # --- Edge cases ---

    def test_no_comment_returns_none(self):
        lines = [
            "x = 1",
            "y = 2",
        ]
        result = extract_preceding_doc_comment(lines, 2, "python")
        assert result is None

    def test_one_blank_line_gap_allowed(self):
        lines = [
            "/** Short desc */",
            "",
            "function foo() {",
        ]
        result = extract_preceding_doc_comment(lines, 3, "javascript")
        assert result == "Short desc"

    def test_two_blank_lines_gap_returns_none(self):
        lines = [
            "/** Short desc */",
            "",
            "",
            "function foo() {",
        ]
        result = extract_preceding_doc_comment(lines, 4, "javascript")
        assert result is None

    def test_element_at_line_1_returns_none(self):
        lines = [
            "function foo() {",
        ]
        result = extract_preceding_doc_comment(lines, 1, "javascript")
        assert result is None

    def test_empty_lines_list(self):
        result = extract_preceding_doc_comment([], 1, "javascript")
        assert result is None

    def test_truncates_long_doc_comment(self):
        """Doc comments longer than 2000 chars should be truncated."""
        long_line = "x" * 2100
        lines = [
            f"/// {long_line}",
            "fn foo() {",
        ]
        result = extract_preceding_doc_comment(lines, 2, "rust")
        assert result is not None
        assert len(result) == 2000


# =============================================================================
# Integration tests: doc comment extraction via full parser pipeline
# =============================================================================

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "languages"


class TestDocCommentIntegration:
    """Integration tests parsing real fixture files and checking docstring field."""

    def test_javascript_jsdoc_on_class(self):
        """JSDoc on BoxOffice class in teatro_ticketing.js."""
        from magaldi_core.parsers.javascript import JavaScriptParser

        fixture = FIXTURES_DIR / "teatro_ticketing.js"
        content = fixture.read_text()
        file_info = FileInfo(
            relative_path="teatro_ticketing.js",
            absolute_path=fixture,
            language="javascript",
        )
        parser = JavaScriptParser()
        elements = parser.parse(content, file_info, "test", "teatro", "main")

        box_office = next(
            (e for e in elements if e.name == "BoxOffice" and e.element_type == "class"),
            None,
        )
        assert box_office is not None
        assert box_office.docstring is not None
        assert "heart of the teatro" in box_office.docstring

    def test_rust_triple_slash_on_function(self):
        """/// doc comment on validate_instrument_serial in teatro_orchestra.rs."""
        from magaldi_core.parsers.rust import RustParser

        fixture = FIXTURES_DIR / "teatro_orchestra.rs"
        content = fixture.read_text()
        file_info = FileInfo(
            relative_path="teatro_orchestra.rs",
            absolute_path=fixture,
            language="rust",
        )
        parser = RustParser()
        elements = parser.parse(content, file_info, "test", "teatro", "main")

        fn_elem = next(
            (e for e in elements if e.name == "validate_instrument_serial"),
            None,
        )
        assert fn_elem is not None
        assert fn_elem.docstring is not None
        assert "Phantom" in fn_elem.docstring

    def test_rust_enum_without_doc_after_section_marker(self):
        """Enum after '// --- Enums ---' section marker should have no docstring."""
        from magaldi_core.parsers.rust import RustParser

        fixture = FIXTURES_DIR / "teatro_orchestra.rs"
        content = fixture.read_text()
        file_info = FileInfo(
            relative_path="teatro_orchestra.rs",
            absolute_path=fixture,
            language="rust",
        )
        parser = RustParser()
        elements = parser.parse(content, file_info, "test", "teatro", "main")

        # TuningStatus is right after #[derive] which is after "// --- Enums ---"
        tuning = next(
            (e for e in elements if e.name == "TuningStatus"),
            None,
        )
        assert tuning is not None
        # No doc comment — only #[derive(Debug, Clone, PartialEq)] above it
        assert tuning.docstring is None
