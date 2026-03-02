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
from magaldi_core.parsers.base import extract_docstring
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
