"""Tests for Bash function recovery when tree-sitter produces parse errors.

Tree-sitter-bash cannot parse certain valid Bash syntax (e.g., ANSI-C quoting
inside parameter expansions, glob character classes in variable substitutions).
When such syntax appears inside a function body, tree-sitter destroys the
function_definition node and may cascade the error into subsequent functions.

The recovery pass in BashExtractor.extract_elements() uses regex + brace
counting to detect and reconstruct these missing functions.
"""

from pathlib import Path

from magaldi_core.change_detection import FileInfo
from magaldi_core.code_parser import BashParser
from magaldi_core.extractors.bash import BashExtractor
from magaldi_core.tree_sitter_manager import TreeSitterManager


class TestBashFunctionRecoveryAfterParseError:
    """Tests for recovering functions broken by tree-sitter parse errors."""

    def _extract_functions(self, code: str) -> list:
        """Helper: parse code and return only function elements."""
        mgr = TreeSitterManager()
        tree = mgr.parse(code.encode("utf-8"), "bash")
        lines = code.split("\n")
        ext = BashExtractor()
        elements = ext.extract_elements(tree, lines)
        return [e for e in elements if e.element_type == "function"]

    def test_glob_pattern_in_variable_substitution(self):
        """Functions after glob patterns like [0-9] in ${var//pattern} are recovered."""
        code = (
            'get_style() {\n'
            '    if [[ "$gtk_shorthand" == "on" ]]; then\n'
            "        theme=\"${theme// '[GTK'[0-9]']'}\"\n"
            '    fi\n'
            '}\n'
            '\n'
            'get_theme() {\n'
            '    name="gtk-theme-name"\n'
            '    get_style\n'
            '}\n'
            '\n'
            'get_icons() {\n'
            '    name="gtk-icon-theme-name"\n'
            '    get_style\n'
            '}'
        )
        funcs = self._extract_functions(code)
        names = {f.name for f in funcs}
        assert "get_style" in names, "get_style should be recovered"
        assert "get_theme" in names, "get_theme should be recovered"
        assert "get_icons" in names, "get_icons should be recovered"
        assert len(funcs) == 3

    def test_ansi_c_quoting_in_parameter_expansion(self):
        r"""Functions with ANSI-C quoting ($'...') in parameter expansion are recovered."""
        # Bash line: kde_config_dir="${kde_config_dir/$'/:'*}"
        code = "kde_config_dir() {\n    kde_config_dir=\"${kde_config_dir/$'/:'*}\"\n}\n\nnext_func() {\n    echo \"hello\"\n}"
        funcs = self._extract_functions(code)
        names = {f.name for f in funcs}
        assert "kde_config_dir" in names, "kde_config_dir should be recovered"
        assert "next_func" in names, "next_func should be found (tree-sitter or recovered)"
        assert len(funcs) == 2

    def test_no_recovery_when_no_errors(self):
        """No recovery pass is triggered when tree-sitter has no errors."""
        code = (
            "func_a() {\n"
            '    echo "a"\n'
            "}\n"
            "\n"
            "func_b() {\n"
            '    echo "b"\n'
            "}"
        )
        funcs = self._extract_functions(code)
        names = {f.name for f in funcs}
        assert "func_a" in names
        assert "func_b" in names
        assert len(funcs) == 2
        # All should have nodes (from tree-sitter, not recovered)
        for f in funcs:
            assert f.node is not None, f"{f.name} should have a tree-sitter node"

    def test_recovered_functions_have_no_node(self):
        """Recovered functions have node=None to distinguish from tree-sitter parsed ones."""
        # Use ANSI-C quoting ($'/') which consistently breaks tree-sitter
        code = "broken() {\n    x=\"${x/$'/:'*}\"\n}\n\nalso_broken() {\n    echo \"after\"\n}"
        funcs = self._extract_functions(code)
        recovered = [f for f in funcs if f.node is None]
        assert len(recovered) >= 1, "At least one function should be recovered (node=None)"

    def test_no_duplicate_functions(self):
        """Functions found by both tree-sitter and regex are not duplicated."""
        code = (
            "good_func() {\n"
            '    echo "works"\n'
            "}\n"
            "\n"
            "broken() {\n"
            "    theme=\"${theme// '[GTK'[0-9]']'}\"\n"
            "}\n"
            "\n"
            "another_good() {\n"
            '    echo "also works"\n'
            "}"
        )
        funcs = self._extract_functions(code)
        names = [f.name for f in funcs]
        # Check no duplicates
        assert len(names) == len(set(names)), f"Duplicate functions: {names}"
        assert len(funcs) == 3

    def test_correct_line_boundaries(self):
        """Recovered function line boundaries match actual source positions."""
        code = (
            "# line 1\n"
            "broken() {\n"
            "    theme=\"${theme// '[GTK'[0-9]']'}\"\n"
            "}\n"
            "\n"
            "after() {\n"
            '    echo "hi"\n'
            "}"
        )
        funcs = self._extract_functions(code)
        broken = next(f for f in funcs if f.name == "broken")
        after = next(f for f in funcs if f.name == "after")

        assert broken.line_start == 2, f"broken should start at line 2, got {broken.line_start}"
        assert broken.line_end == 4, f"broken should end at line 4, got {broken.line_end}"
        assert after.line_start == 6, f"after should start at line 6, got {after.line_start}"
        assert after.line_end == 8, f"after should end at line 8, got {after.line_end}"

    def test_function_keyword_syntax(self):
        """Functions using 'function name() {' syntax are also recovered."""
        code = (
            "function broken_func() {\n"
            "    theme=\"${theme// '[GTK'[0-9]']'}\"\n"
            "}\n"
        )
        funcs = self._extract_functions(code)
        assert len(funcs) == 1
        assert funcs[0].name == "broken_func"

    def test_nested_braces_in_recovered_function(self):
        """Recovery handles nested braces (if/case/for blocks) correctly."""
        code = (
            "complex() {\n"
            "    theme=\"${theme// '[GTK'[0-9]']'}\"\n"
            "    if true; then\n"
            "        for x in a b c; do\n"
            '            echo "$x"\n'
            "        done\n"
            "    fi\n"
            "}\n"
            "\n"
            "simple() {\n"
            '    echo "after"\n'
            "}"
        )
        funcs = self._extract_functions(code)
        names = {f.name for f in funcs}
        assert "complex" in names
        assert "simple" in names

        complex_fn = next(f for f in funcs if f.name == "complex")
        assert complex_fn.line_end == 8, f"complex should end at line 8, got {complex_fn.line_end}"

    def test_heredoc_in_recovered_function(self):
        """Recovery correctly skips heredoc content when counting braces."""
        code = (
            "with_heredoc() {\n"
            "    theme=\"${theme// '[GTK'[0-9]']'}\"\n"
            "    cat <<'EOF'\n"
            "    This has { and } inside\n"
            "    } more braces {\n"
            "EOF\n"
            "}\n"
            "\n"
            "after_heredoc() {\n"
            '    echo "ok"\n'
            "}"
        )
        funcs = self._extract_functions(code)
        names = {f.name for f in funcs}
        assert "with_heredoc" in names
        assert "after_heredoc" in names

        heredoc_fn = next(f for f in funcs if f.name == "with_heredoc")
        assert heredoc_fn.line_end == 7, f"with_heredoc should end at line 7, got {heredoc_fn.line_end}"

    def test_cascading_error_multiple_functions(self):
        """Multiple consecutive functions after a parse error are all recovered."""
        code = (
            "first() {\n"
            "    theme=\"${theme// '[GTK'[0-9]']'}\"\n"
            "}\n"
            "\n"
            "second() {\n"
            '    echo "b"\n'
            "}\n"
            "\n"
            "third() {\n"
            '    echo "c"\n'
            "}\n"
            "\n"
            "fourth() {\n"
            '    echo "d"\n'
            "}"
        )
        funcs = self._extract_functions(code)
        names = {f.name for f in funcs}
        assert names == {"first", "second", "third", "fourth"}, f"Expected all 4 functions, got: {names}"


class TestBashFunctionRecoveryIntegration:
    """Integration tests using BashParser.parse() (full pipeline)."""

    def test_recovered_functions_get_element_ids(self):
        """Recovered functions get valid element IDs via BashParser.parse()."""
        code = (
            "broken() {\n"
            "    theme=\"${theme// '[GTK'[0-9]']'}\"\n"
            "}\n"
            "\n"
            "after() {\n"
            '    echo "ok"\n'
            "}"
        )
        parser = BashParser()
        file_info = FileInfo(
            relative_path="test.sh",
            absolute_path=Path("/fake/test.sh"),
            language="bash",
        )
        elements = parser.parse(code, file_info, "scope", "repo", "main")
        func_elements = [e for e in elements if e.element_type == "function"]

        assert len(func_elements) == 2
        for elem in func_elements:
            assert elem.element_id, f"{elem.name} should have an element_id"
            assert "scope" in elem.element_id
            assert "repo" in elem.element_id

    def test_recovered_functions_have_correct_parent(self):
        """Recovered functions have parent_id pointing to the file element."""
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
        file_elem = next(e for e in elements if e.element_type == "file")
        func_elem = next(e for e in elements if e.element_type == "function")

        assert func_elem.parent_id == file_elem.element_id
