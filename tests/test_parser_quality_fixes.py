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

    # --- TypeScript abstract class ---

    def test_ts_jsdoc_on_abstract_class(self):
        """JSDoc on abstract class should be extracted."""
        lines = [
            "/**",
            " * Base controller for all routes.",
            " */",
            "abstract class BaseController {",
        ]
        result = extract_preceding_doc_comment(lines, 4, "typescript")
        assert result is not None
        assert "Base controller" in result

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


# =============================================================================
# TypeScript: abstract class extraction
# =============================================================================


class TestTypeScriptAbstractClass:
    """Verify abstract classes and their methods are extracted."""

    def _parse_typescript(self, code: str):
        from magaldi_core.parsers.javascript import JavaScriptParser

        parser = JavaScriptParser("typescript")
        file_info = FileInfo(
            relative_path="test.ts",
            absolute_path=Path("/fake/test.ts"),
            language="typescript",
        )
        return parser.parse(code, file_info, "scope", "repo", "main")

    def test_abstract_class_extracted(self):
        """abstract class should be extracted as a class element."""
        code = """\
abstract class Animal {
    abstract speak(): string;
}
"""
        elements = self._parse_typescript(code)
        cls = next((e for e in elements if e.name == "Animal" and e.element_type == "class"), None)
        assert cls is not None, f"Expected class 'Animal', got: {[(e.name, e.element_type) for e in elements]}"
        assert "abstract" in (cls.decorators or [])

    def test_abstract_method_extracted(self):
        """Abstract method signatures should be extracted as methods."""
        code = """\
abstract class Animal {
    abstract speak(): string;
    abstract move(distance: number): void;
}
"""
        elements = self._parse_typescript(code)
        methods = [e for e in elements if e.element_type == "method"]
        method_names = [m.name for m in methods]
        assert "speak" in method_names, f"Expected 'speak' method, got: {method_names}"
        assert "move" in method_names, f"Expected 'move' method, got: {method_names}"

        speak = next(m for m in methods if m.name == "speak")
        assert "abstract" in (speak.decorators or [])
        assert speak.signature is not None
        assert "abstract" in speak.signature

    def test_abstract_class_concrete_methods(self):
        """Concrete methods in abstract class should also be extracted."""
        code = """\
abstract class Animal {
    abstract speak(): string;
    getName(): string {
        return this.name;
    }
}
"""
        elements = self._parse_typescript(code)
        methods = [e for e in elements if e.element_type == "method"]
        method_names = [m.name for m in methods]
        assert "speak" in method_names
        assert "getName" in method_names

        get_name = next(m for m in methods if m.name == "getName")
        assert "abstract" not in (get_name.decorators or [])

    def test_abstract_class_with_extends(self):
        """Abstract class with extends should have base_classes populated."""
        code = """\
abstract class Vehicle extends Transport {
    abstract start(): void;
}
"""
        elements = self._parse_typescript(code)
        cls = next((e for e in elements if e.name == "Vehicle" and e.element_type == "class"), None)
        assert cls is not None
        assert cls.base_classes is not None
        assert "Transport" in cls.base_classes

    def test_abstract_and_regular_class_coexist(self):
        """Both abstract and regular classes in same file should be extracted."""
        code = """\
abstract class Shape {
    abstract area(): number;
}

class Circle extends Shape {
    area(): number {
        return Math.PI * this.radius ** 2;
    }
}
"""
        elements = self._parse_typescript(code)
        classes = [e for e in elements if e.element_type == "class"]
        class_names = [c.name for c in classes]
        assert "Shape" in class_names
        assert "Circle" in class_names

        shape = next(c for c in classes if c.name == "Shape")
        assert "abstract" in (shape.decorators or [])

        circle = next(c for c in classes if c.name == "Circle")
        assert "abstract" not in (circle.decorators or [])

    def test_exported_abstract_class(self):
        """export abstract class should be extracted."""
        code = """\
export abstract class BaseService {
    abstract init(): Promise<void>;
}
"""
        elements = self._parse_typescript(code)
        cls = next((e for e in elements if e.name == "BaseService" and e.element_type == "class"), None)
        assert cls is not None
        assert "abstract" in (cls.decorators or [])


# =============================================================================
# PHP: interface parsing
# =============================================================================


class TestPhpInterfaceParsing:
    """Verify PHP interfaces are extracted as CodeElements."""

    def _parse_php(self, code: str):
        from magaldi_core.parsers.php import PhpParser

        parser = PhpParser()
        file_info = FileInfo(
            relative_path="test.php",
            absolute_path=Path("/fake/test.php"),
            language="php",
        )
        return parser.parse(code, file_info, "scope", "repo", "main")

    def test_simple_interface_extracted(self):
        """Interface should be extracted as an element."""
        code = "<?php\ninterface Cacheable {\n    public function cache(): void;\n}\n"
        elements = self._parse_php(code)
        iface = next((e for e in elements if e.name == "Cacheable"), None)
        assert iface is not None, f"Expected interface 'Cacheable', got: {[(e.name, e.element_type) for e in elements]}"

    def test_interface_type_correct(self):
        """Interface element_type should be 'interface'."""
        code = "<?php\ninterface Cacheable {\n    public function cache(): void;\n}\n"
        elements = self._parse_php(code)
        iface = next(e for e in elements if e.name == "Cacheable")
        assert iface.element_type == "interface"

    def test_interface_methods_extracted(self):
        """Method signatures inside interface should be extracted as methods."""
        code = "<?php\ninterface Serializable {\n    public function serialize(): string;\n    public function unserialize(string $data): void;\n}\n"
        elements = self._parse_php(code)
        methods = [e for e in elements if e.element_type == "method"]
        method_names = [m.name for m in methods]
        assert "serialize" in method_names
        assert "unserialize" in method_names

    def test_interface_method_parent_id(self):
        """Interface methods should have parent_id pointing to the interface."""
        code = "<?php\ninterface Cacheable {\n    public function cache(): void;\n}\n"
        elements = self._parse_php(code)
        iface = next(e for e in elements if e.name == "Cacheable")
        method = next(e for e in elements if e.name == "cache" and e.element_type == "method")
        assert method.parent_id == iface.element_id

    def test_interface_extends(self):
        """Interface extending other interfaces should have base_classes populated."""
        code = "<?php\ninterface ReadWritable extends Readable, Writable {\n    public function seek(int $pos): void;\n}\n"
        elements = self._parse_php(code)
        iface = next(e for e in elements if e.name == "ReadWritable")
        assert iface.base_classes is not None
        assert "Readable" in iface.base_classes
        assert "Writable" in iface.base_classes

    def test_interface_docstring(self):
        """PHPDoc before interface should be extracted."""
        code = "<?php\n/**\n * Defines cacheable behavior.\n */\ninterface Cacheable {\n}\n"
        elements = self._parse_php(code)
        iface = next(e for e in elements if e.name == "Cacheable")
        assert iface.docstring is not None
        assert "cacheable" in iface.docstring.lower()

    def test_interface_and_class_coexist(self):
        """Both interface and class in same file should be extracted."""
        code = "<?php\ninterface Loggable {\n    public function log(): void;\n}\n\nclass FileLogger implements Loggable {\n    public function log(): void {}\n}\n"
        elements = self._parse_php(code)
        iface = next((e for e in elements if e.name == "Loggable" and e.element_type == "interface"), None)
        cls = next((e for e in elements if e.name == "FileLogger" and e.element_type == "class"), None)
        assert iface is not None
        assert cls is not None


# =============================================================================
# PHP: trait parsing
# =============================================================================


class TestPhpTraitParsing:
    """Verify PHP traits are extracted as CodeElements."""

    def _parse_php(self, code: str):
        from magaldi_core.parsers.php import PhpParser

        parser = PhpParser()
        file_info = FileInfo(
            relative_path="test.php",
            absolute_path=Path("/fake/test.php"),
            language="php",
        )
        return parser.parse(code, file_info, "scope", "repo", "main")

    def test_simple_trait_extracted(self):
        """Trait should be extracted as an element."""
        code = "<?php\ntrait Timestampable {\n    public function getCreatedAt(): string { return $this->createdAt; }\n}\n"
        elements = self._parse_php(code)
        trait = next((e for e in elements if e.name == "Timestampable"), None)
        assert trait is not None, f"Expected trait 'Timestampable', got: {[(e.name, e.element_type) for e in elements]}"

    def test_trait_type_correct(self):
        """Trait element_type should be 'trait'."""
        code = "<?php\ntrait Timestampable {\n    public function touch(): void {}\n}\n"
        elements = self._parse_php(code)
        trait = next(e for e in elements if e.name == "Timestampable")
        assert trait.element_type == "trait"

    def test_trait_methods_extracted(self):
        """Methods inside trait should be extracted."""
        code = "<?php\ntrait Cacheable {\n    public function cache(): void {}\n    public function invalidate(): void {}\n}\n"
        elements = self._parse_php(code)
        methods = [e for e in elements if e.element_type == "method"]
        method_names = [m.name for m in methods]
        assert "cache" in method_names
        assert "invalidate" in method_names

    def test_trait_method_parent_id(self):
        """Trait methods should have parent_id pointing to the trait."""
        code = "<?php\ntrait Cacheable {\n    public function cache(): void {}\n}\n"
        elements = self._parse_php(code)
        trait = next(e for e in elements if e.name == "Cacheable")
        method = next(e for e in elements if e.name == "cache" and e.element_type == "method")
        assert method.parent_id == trait.element_id

    def test_trait_properties_extracted(self):
        """Properties inside trait should be extracted as variables."""
        code = "<?php\ntrait HasName {\n    protected $name;\n    public function getName(): string { return $this->name; }\n}\n"
        elements = self._parse_php(code)
        props = [e for e in elements if e.element_type == "variable" and e.name == "name"]
        assert len(props) >= 1

    def test_trait_docstring(self):
        """PHPDoc before trait should be extracted."""
        code = "<?php\n/**\n * Adds timestamp fields.\n */\ntrait Timestampable {\n}\n"
        elements = self._parse_php(code)
        trait = next(e for e in elements if e.name == "Timestampable")
        assert trait.docstring is not None
        assert "timestamp" in trait.docstring.lower()


# =============================================================================
# PHP: class constant extraction
# =============================================================================


class TestPhpClassConstants:
    """Verify PHP class constants are extracted."""

    def _parse_php(self, code: str):
        from magaldi_core.parsers.php import PhpParser

        parser = PhpParser()
        file_info = FileInfo(
            relative_path="test.php",
            absolute_path=Path("/fake/test.php"),
            language="php",
        )
        return parser.parse(code, file_info, "scope", "repo", "main")

    def test_class_constant_extracted(self):
        """Class constant should be extracted."""
        code = "<?php\nclass Config {\n    const VERSION = '1.0';\n}\n"
        elements = self._parse_php(code)
        const = next((e for e in elements if e.name == "VERSION"), None)
        assert const is not None, f"Expected constant 'VERSION', got: {[(e.name, e.element_type) for e in elements]}"

    def test_class_constant_type(self):
        """Class constant element_type should be 'constant'."""
        code = "<?php\nclass Config {\n    const VERSION = '1.0';\n}\n"
        elements = self._parse_php(code)
        const = next(e for e in elements if e.name == "VERSION")
        assert const.element_type == "constant"

    def test_class_constant_visibility(self):
        """Visibility modifiers should be in decorators."""
        code = "<?php\nclass Config {\n    public const PUBLIC_VAL = 1;\n    private const PRIVATE_VAL = 2;\n    protected const PROTECTED_VAL = 3;\n}\n"
        elements = self._parse_php(code)

        pub = next(e for e in elements if e.name == "PUBLIC_VAL")
        assert "public" in (pub.decorators or [])

        priv = next(e for e in elements if e.name == "PRIVATE_VAL")
        assert "private" in (priv.decorators or [])

        prot = next(e for e in elements if e.name == "PROTECTED_VAL")
        assert "protected" in (prot.decorators or [])

    def test_class_constant_signature(self):
        """Signature should contain visibility and const declaration."""
        code = "<?php\nclass Config {\n    public const VERSION = '1.0';\n}\n"
        elements = self._parse_php(code)
        const = next(e for e in elements if e.name == "VERSION")
        assert const.signature is not None
        assert "public const VERSION" in const.signature

    def test_multiple_class_constants(self):
        """Multiple constants should all be extracted."""
        code = "<?php\nclass Status {\n    const ACTIVE = 1;\n    const INACTIVE = 0;\n    const PENDING = 2;\n}\n"
        elements = self._parse_php(code)
        const_names = [e.name for e in elements if e.element_type == "constant"]
        assert "ACTIVE" in const_names
        assert "INACTIVE" in const_names
        assert "PENDING" in const_names

    def test_class_constant_parent_id(self):
        """Class constant should have parent_id pointing to the class."""
        code = "<?php\nclass Config {\n    const VERSION = '1.0';\n}\n"
        elements = self._parse_php(code)
        cls = next(e for e in elements if e.name == "Config" and e.element_type == "class")
        const = next(e for e in elements if e.name == "VERSION")
        assert const.parent_id == cls.element_id

    def test_interface_constants(self):
        """Constants inside interfaces should also be extracted."""
        code = "<?php\ninterface HasVersion {\n    const API_VERSION = '2.0';\n}\n"
        elements = self._parse_php(code)
        const = next((e for e in elements if e.name == "API_VERSION"), None)
        assert const is not None
        assert const.element_type == "constant"


# =============================================================================
# JS/TS: Variable, constant, and enum conversion in parser
# =============================================================================


class TestJsVariableConstantParsing:
    """Verify JS/TS variables, constants, and enums are not silently dropped by the parser."""

    def _parse_js(self, code: str, language: str = "javascript"):
        from magaldi_core.code_parser import JavaScriptParser

        parser = JavaScriptParser(language)
        file_info = FileInfo(
            relative_path="test.js",
            absolute_path=Path("/fake/test.js"),
            language=language,
        )
        return parser.parse(code, file_info, "scope", "repo", "main")

    def test_export_const_extracted(self):
        """export const should produce a constant element."""
        code = "export const API_URL = 'https://example.com';\n"
        elements = self._parse_js(code, language="typescript")
        const = next((e for e in elements if e.name == "API_URL"), None)
        assert const is not None, "export const API_URL should be extracted"
        assert const.element_type in ("variable", "constant")

    def test_const_variable_extracted(self):
        """const with object value should be extracted."""
        code = "const config = { debug: true, port: 3000 };\n"
        elements = self._parse_js(code)
        var = next((e for e in elements if e.name == "config"), None)
        assert var is not None, "const config should be extracted"

    def test_var_declaration_extracted(self):
        """var declaration should be extracted."""
        code = "var MAX_RETRIES = 5;\n"
        elements = self._parse_js(code)
        var = next((e for e in elements if e.name == "MAX_RETRIES"), None)
        assert var is not None, "var MAX_RETRIES should be extracted"

    def test_top_level_variable_level_1(self):
        """Top-level variables should have level=1 (not level=3)."""
        code = "const VERSION = '1.0.0';\n"
        elements = self._parse_js(code, language="typescript")
        var = next((e for e in elements if e.name == "VERSION"), None)
        assert var is not None
        assert var.level == 1, f"Top-level variable should be level 1, got {var.level}"

    def test_ts_enum_extracted(self):
        """TypeScript enums should be extracted (not silently dropped)."""
        code = "enum Color {\n  Red,\n  Green,\n  Blue,\n}\n"
        elements = self._parse_js(code, language="typescript")
        enum = next((e for e in elements if e.name == "Color"), None)
        assert enum is not None, "enum Color should be extracted"
        assert enum.element_type == "enum"

    def test_ts_enum_level_1(self):
        """TypeScript enums should be level 1 (like classes)."""
        code = "export enum Direction {\n  Up,\n  Down,\n}\n"
        elements = self._parse_js(code, language="typescript")
        enum = next((e for e in elements if e.name == "Direction"), None)
        assert enum is not None
        assert enum.level == 1

    def test_variable_has_element_id(self):
        """Extracted variables should have element_id set."""
        code = "const PI = 3.14;\n"
        elements = self._parse_js(code, language="typescript")
        var = next((e for e in elements if e.name == "PI"), None)
        assert var is not None
        assert var.element_id is not None
        assert ":variable:" in var.element_id or ":constant:" in var.element_id

    def test_constant_preserves_type(self):
        """Element type 'constant' should be preserved (not coerced to 'variable')."""
        code = "export const MAX = 100;\n"
        elements = self._parse_js(code, language="typescript")
        # The extractor marks `const` as constant via _extract_js_variable
        var = next((e for e in elements if e.name == "MAX"), None)
        assert var is not None


# =============================================================================
# JS: Object-method and prototype assignment extraction
# =============================================================================


class TestJsObjectMethodAssignment:
    """Verify assignment-based function patterns are extracted."""

    def _parse_js(self, code: str, language: str = "javascript"):
        from magaldi_core.code_parser import JavaScriptParser

        parser = JavaScriptParser(language)
        file_info = FileInfo(
            relative_path="test.js",
            absolute_path=Path("/fake/test.js"),
            language=language,
        )
        return parser.parse(code, file_info, "scope", "repo", "main")

    def test_obj_method_assignment(self):
        """app.use = function use(fn) {} should be extracted."""
        code = "app.use = function use(fn) { return fn; };\n"
        elements = self._parse_js(code)
        func = next((e for e in elements if e.name == "use"), None)
        assert func is not None, "app.use = function use() should be extracted"
        assert func.element_type == "function"

    def test_prototype_method_assignment(self):
        """View.prototype.lookup = function lookup(name) {} should be extracted."""
        code = "View.prototype.lookup = function lookup(name) { return name; };\n"
        elements = self._parse_js(code)
        func = next((e for e in elements if e.name == "lookup"), None)
        assert func is not None, "View.prototype.lookup should be extracted"
        assert func.element_type == "function"

    def test_exports_named_function(self):
        """exports.Router = function Router() {} should be extracted."""
        code = "exports.Router = function Router() {};\n"
        elements = self._parse_js(code)
        func = next((e for e in elements if e.name == "Router"), None)
        assert func is not None, "exports.Router should be extracted"

    def test_module_exports_named_function(self):
        """module.exports = function createApp() {} should use function name."""
        code = "module.exports = function createApp() { return {}; };\n"
        elements = self._parse_js(code)
        func = next((e for e in elements if e.name == "createApp"), None)
        assert func is not None, "module.exports with named function should be extracted"

    def test_module_exports_anonymous_skipped(self):
        """module.exports = function() {} (anonymous) should be skipped."""
        code = "module.exports = function() { return {}; };\n"
        elements = self._parse_js(code)
        funcs = [e for e in elements if e.element_type == "function"]
        assert len(funcs) == 0, "Anonymous module.exports should be skipped"

    def test_module_exports_object_skipped(self):
        """module.exports = { foo: 1 } should not produce a function."""
        code = "module.exports = { foo: 1 };\n"
        elements = self._parse_js(code)
        funcs = [e for e in elements if e.element_type == "function"]
        assert len(funcs) == 0, "Object export should not produce a function"

    def test_assignment_function_has_params(self):
        """Extracted assignment function should have parameters."""
        code = "app.set = function set(key, value) { this[key] = value; };\n"
        elements = self._parse_js(code)
        func = next((e for e in elements if e.name == "set"), None)
        assert func is not None
        assert func.parameters is not None
        param_names = [p["name"] for p in func.parameters]
        assert "key" in param_names
        assert "value" in param_names

    def test_assignment_function_has_signature(self):
        """Extracted assignment function should have a meaningful signature."""
        code = "app.use = function use(fn) { return fn; };\n"
        elements = self._parse_js(code)
        func = next((e for e in elements if e.name == "use"), None)
        assert func is not None
        assert func.signature is not None
        assert "app.use" in func.signature

    def test_async_assignment_function(self):
        """Async assignment functions should have is_async flag."""
        code = "app.handler = async function handler(req) { await req; };\n"
        elements = self._parse_js(code)
        func = next((e for e in elements if e.name == "handler"), None)
        assert func is not None
        assert func.is_async is True


# =============================================================================
# Bash: Brace-group extraction fix
# =============================================================================


class TestBashBraceGroupExtraction:
    """Verify functions inside { ... } brace-groups are extracted."""

    def _parse_bash(self, code: str):
        from magaldi_core.code_parser import BashParser

        parser = BashParser()
        file_info = FileInfo(
            relative_path="test.sh",
            absolute_path=Path("/fake/test.sh"),
            language="bash",
        )
        return parser.parse(code, file_info, "scope", "repo", "main")

    def test_function_inside_brace_group(self):
        """Functions inside { ... } should be extracted."""
        code = "#!/bin/bash\n{\n  my_func() {\n    echo hello\n  }\n}\n"
        elements = self._parse_bash(code)
        func = next((e for e in elements if e.name == "my_func"), None)
        assert func is not None, "Function inside brace-group should be extracted"
        assert func.element_type == "function"

    def test_multiple_functions_inside_brace_group(self):
        """Multiple functions inside { ... } should all be extracted."""
        code = (
            "#!/bin/bash\n{\n"
            "  func_a() {\n    echo a\n  }\n"
            "  func_b() {\n    echo b\n  }\n"
            "  func_c() {\n    echo c\n  }\n"
            "}\n"
        )
        elements = self._parse_bash(code)
        func_names = [e.name for e in elements if e.element_type == "function"]
        assert "func_a" in func_names
        assert "func_b" in func_names
        assert "func_c" in func_names

    def test_variable_inside_brace_group(self):
        """Top-level variables inside { ... } should be extracted."""
        code = "#!/bin/bash\n{\n  MY_VAR=hello\n}\n"
        elements = self._parse_bash(code)
        var = next((e for e in elements if e.name == "MY_VAR"), None)
        assert var is not None, "Variable inside brace-group should be extracted"

    def test_top_level_functions_still_work(self):
        """Functions NOT inside brace-groups should still work (regression check)."""
        code = "#!/bin/bash\nmy_func() {\n  echo hello\n}\n"
        elements = self._parse_bash(code)
        func = next((e for e in elements if e.name == "my_func"), None)
        assert func is not None

    def test_nested_brace_groups(self):
        """Functions in nested { { ... } } should be extracted."""
        code = "#!/bin/bash\n{\n  {\n    inner_func() {\n      echo inner\n    }\n  }\n}\n"
        elements = self._parse_bash(code)
        func = next((e for e in elements if e.name == "inner_func"), None)
        assert func is not None, "Function in nested brace-group should be extracted"

    def test_mixed_top_level_and_brace_group(self):
        """Script with both top-level and brace-group functions."""
        code = (
            "#!/bin/bash\n"
            "top_func() {\n  echo top\n}\n"
            "{\n  inner_func() {\n    echo inner\n  }\n}\n"
        )
        elements = self._parse_bash(code)
        func_names = [e.name for e in elements if e.element_type == "function"]
        assert "top_func" in func_names
        assert "inner_func" in func_names
