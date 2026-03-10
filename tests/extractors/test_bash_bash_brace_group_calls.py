"""Tests for Bash brace-group wrapper extraction.

Verifies that functions, variables, calls, and imports inside { ... } brace
groups and ( ... ) subshell wrappers are extracted correctly. Many Bash scripts
(e.g., nvm.sh, install scripts) wrap their entire body in a brace group to
ensure the script is fully downloaded before execution.
"""

from pathlib import Path

from magaldi_core.change_detection import FileInfo
from magaldi_core.code_parser import BashParser
from magaldi_core.extractors.bash import extract_top_level_bash_calls
from magaldi_core.tree_sitter_manager import get_manager


def _parse_bash(code: str):
    parser = BashParser()
    file_info = FileInfo(
        relative_path="test.sh",
        absolute_path=Path("/fake/test.sh"),
        language="bash",
    )
    return parser.parse(code, file_info, "scope", "repo", "main")


def _parse_tree(code: str):
    manager = get_manager()
    return manager.parse(code.encode("utf-8"), "bash")


class TestBashBraceGroupCalls:
    """Verify calls inside brace-group wrapped scripts are extracted."""

    def test_function_calls_inside_brace_group(self):
        """Calls from functions inside { ... } should be extracted."""
        code = (
            "#!/usr/bin/env bash\n"
            "{\n"
            "nvm_has() {\n"
            '  type "$1" > /dev/null 2>&1\n'
            "}\n"
            "\n"
            "nvm_install() {\n"
            "  nvm_has curl\n"
            '  echo "Installing..."\n'
            "}\n"
            "}\n"
        )
        elements = _parse_bash(code)

        # Functions should be extracted
        func_names = [e.name for e in elements if e.element_type == "function"]
        assert "nvm_has" in func_names
        assert "nvm_install" in func_names

        # nvm_install should have a call to nvm_has
        nvm_install = next(e for e in elements if e.name == "nvm_install")
        call_names = [c.name for c in nvm_install.calls]
        assert "nvm_has" in call_names, "nvm_install should call nvm_has"

    def test_top_level_calls_inside_brace_group(self):
        """Top-level calls inside { ... } should appear on file element."""
        code = (
            "#!/usr/bin/env bash\n"
            "{\n"
            "setup() {\n"
            "  echo setup\n"
            "}\n"
            "deploy() {\n"
            "  echo deploy\n"
            "}\n"
            "setup\n"
            "deploy\n"
            "}\n"
        )
        tree = _parse_tree(code)
        calls = extract_top_level_bash_calls(tree)
        names = [c.name for c in calls]
        assert "setup" in names, "Top-level call 'setup' inside brace group"
        assert "deploy" in names, "Top-level call 'deploy' inside brace group"

    def test_top_level_calls_skip_function_bodies_in_brace_group(self):
        """Function-body calls inside { ... } should NOT appear as top-level."""
        code = (
            "#!/usr/bin/env bash\n"
            "{\n"
            "deploy() {\n"
            "  kubectl apply\n"
            "  notify_slack\n"
            "}\n"
            "deploy\n"
            "}\n"
        )
        tree = _parse_tree(code)
        calls = extract_top_level_bash_calls(tree)
        names = [c.name for c in calls]
        assert "deploy" in names
        assert "kubectl" not in names, "kubectl is inside function body"
        assert "notify_slack" not in names, "notify_slack is inside function body"

    def test_call_resolution_inside_brace_group(self):
        """Calls to sibling functions inside { ... } should resolve."""
        code = (
            "#!/usr/bin/env bash\n"
            "{\n"
            "helper() {\n"
            "  echo help\n"
            "}\n"
            "main_func() {\n"
            "  helper\n"
            "}\n"
            "}\n"
        )
        elements = _parse_bash(code)
        main_func = next(e for e in elements if e.name == "main_func")
        helper_call = next((c for c in main_func.calls if c.name == "helper"), None)
        assert helper_call is not None, "main_func should call helper"
        assert helper_call.resolved_id is not None, "Call to helper should be resolved"

    def test_file_level_calls_resolved_in_brace_group(self):
        """Top-level calls inside { ... } should resolve to functions."""
        code = (
            "#!/usr/bin/env bash\n"
            "{\n"
            "setup() {\n"
            "  echo setup\n"
            "}\n"
            "setup\n"
            "}\n"
        )
        elements = _parse_bash(code)
        file_elem = next(e for e in elements if e.element_type == "file")
        setup_call = next((c for c in file_elem.calls if c.name == "setup"), None)
        assert setup_call is not None, "File should have top-level call to setup"
        assert setup_call.resolved_id is not None, "Call to setup should be resolved"


class TestBashSubshellExtraction:
    """Verify extraction inside ( ... ) subshell wrappers."""

    def test_functions_inside_subshell(self):
        """Functions inside ( ... ) should be extracted."""
        code = (
            "#!/usr/bin/env bash\n"
            "(\n"
            "my_func() {\n"
            "  echo hello\n"
            "}\n"
            ")\n"
        )
        elements = _parse_bash(code)
        func = next((e for e in elements if e.name == "my_func"), None)
        assert func is not None, "Function inside subshell should be extracted"
        assert func.element_type == "function"

    def test_top_level_calls_inside_subshell(self):
        """Top-level calls inside ( ... ) should be extracted."""
        code = (
            "#!/usr/bin/env bash\n"
            "(\n"
            "setup() {\n"
            "  echo setup\n"
            "}\n"
            "setup\n"
            ")\n"
        )
        tree = _parse_tree(code)
        calls = extract_top_level_bash_calls(tree)
        names = [c.name for c in calls]
        assert "setup" in names, "Top-level call inside subshell should be found"


class TestBashBraceGroupNvmStyle:
    """Test with a realistic nvm.sh-style brace-group wrapped script."""

    def test_nvm_style_script(self):
        """Simulated nvm.sh pattern: all functions inside { ... } download guard."""
        code = (
            "#!/usr/bin/env bash\n"
            "{ # this ensures the entire script is downloaded\n"
            "\n"
            "nvm_has() {\n"
            '  type "$1" > /dev/null 2>&1\n'
            "}\n"
            "\n"
            "nvm_echo() {\n"
            '  command printf "%s\\n" "$*" 2>/dev/null\n'
            "}\n"
            "\n"
            "nvm_default_install_dir() {\n"
            '  [ -z "${XDG_CONFIG_HOME-}" ] && printf %s "${HOME}/.nvm" || printf %s "${XDG_CONFIG_HOME}/nvm"\n'
            "}\n"
            "\n"
            "nvm_install_dir() {\n"
            "  if nvm_has nvm; then\n"
            "    nvm_echo nvm_default_install_dir\n"
            "  fi\n"
            "}\n"
            "\n"
            "nvm_install_dir\n"
            "\n"
            "} # closing brace\n"
        )
        elements = _parse_bash(code)

        # All functions should be extracted
        func_names = [e.name for e in elements if e.element_type == "function"]
        assert "nvm_has" in func_names
        assert "nvm_echo" in func_names
        assert "nvm_default_install_dir" in func_names
        assert "nvm_install_dir" in func_names
        assert len(func_names) == 4

        # nvm_install_dir should have calls
        nvm_install_dir = next(e for e in elements if e.name == "nvm_install_dir")
        call_names = [c.name for c in nvm_install_dir.calls]
        assert "nvm_has" in call_names
        assert "nvm_echo" in call_names

        # File element should have top-level call
        file_elem = next(e for e in elements if e.element_type == "file")
        file_call_names = [c.name for c in file_elem.calls]
        assert "nvm_install_dir" in file_call_names

        # All functions should be children of the file element
        for func in (e for e in elements if e.element_type == "function"):
            assert func.parent_id == file_elem.element_id, (
                f"Function {func.name} should be child of file element"
            )
