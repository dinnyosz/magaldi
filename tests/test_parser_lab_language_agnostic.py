"""Tests for parser_lab language-agnostic refactoring.

Verifies that:
- _detect_language uses SUPPORTED_EXTENSIONS/SUPPORTED_FILENAMES from discovery
- _detect_language handles shebangs for code snippets
- _get_extension derives mappings from SUPPORTED_EXTENSIONS
- _find_uncaptured_nodes returns empty for unknown languages
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from magaldi_core.discovery import SUPPORTED_EXTENSIONS
from magaldi_mcp.tools.parser_lab import (
    _detect_language,
    _find_uncaptured_nodes,
    _get_extension,
)

# =============================================================================
# _detect_language from file paths
# =============================================================================


class TestDetectLanguageFromPath:
    """Test _detect_language with file paths."""

    @pytest.mark.parametrize(
        "ext,expected_lang",
        [(ext, lang) for ext, lang in SUPPORTED_EXTENSIONS.items()],
    )
    def test_all_supported_extensions(self, ext: str, expected_lang: str) -> None:
        """Every extension in SUPPORTED_EXTENSIONS should be detected."""
        result = _detect_language(f"/fake/file{ext}", None)
        assert result == expected_lang

    def test_dockerfile_detection(self) -> None:
        assert _detect_language("/fake/Dockerfile", None) == "dockerfile"

    def test_dockerfile_variant_detection(self) -> None:
        assert _detect_language("/fake/Dockerfile.prod", None) == "dockerfile"

    def test_unknown_extension_returns_none(self) -> None:
        assert _detect_language("/fake/file.unknown", None) is None

    def test_sh_extension(self) -> None:
        """If .sh is in SUPPORTED_EXTENSIONS, it should be detected."""
        if ".sh" in SUPPORTED_EXTENSIONS:
            assert _detect_language("/fake/script.sh", None) == SUPPORTED_EXTENSIONS[".sh"]


# =============================================================================
# _detect_language from code snippets (shebangs)
# =============================================================================


class TestDetectLanguageFromShebang:
    """Test _detect_language shebang detection for code snippets."""

    def test_bash_shebang(self) -> None:
        code = "#!/bin/bash\necho hello"
        assert _detect_language(None, code) == "bash"

    def test_sh_shebang(self) -> None:
        code = "#!/bin/sh\necho hello"
        assert _detect_language(None, code) == "bash"

    def test_env_bash_shebang(self) -> None:
        code = "#!/usr/bin/env bash\necho hello"
        assert _detect_language(None, code) == "bash"

    def test_python_shebang(self) -> None:
        code = "#!/usr/bin/env python3\nimport sys"
        assert _detect_language(None, code) == "python"

    def test_node_shebang(self) -> None:
        code = "#!/usr/bin/env node\nconsole.log('hi')"
        assert _detect_language(None, code) == "javascript"

    def test_php_shebang(self) -> None:
        code = "#!/usr/bin/env php\n<?php echo 'hi';"
        assert _detect_language(None, code) == "php"

    def test_shebang_takes_priority_over_heuristics(self) -> None:
        """Shebang should be checked before heuristic patterns."""
        # This code has 'def' and 'import' (python heuristic), but shebang says bash
        code = "#!/bin/bash\n# def import\necho hello"
        assert _detect_language(None, code) == "bash"


# =============================================================================
# _detect_language heuristics (existing behavior preserved)
# =============================================================================


class TestDetectLanguageHeuristics:
    """Test that existing code heuristics still work."""

    def test_python_heuristic(self) -> None:
        assert _detect_language(None, "import os\ndef main():\n    pass") == "python"

    def test_javascript_heuristic(self) -> None:
        assert _detect_language(None, "function hello() {}") == "javascript"

    def test_rust_heuristic(self) -> None:
        assert _detect_language(None, "fn main() {\n    let x = 1;\n}") == "rust"

    def test_php_heuristic(self) -> None:
        assert _detect_language(None, "<?php echo 'hello';") == "php"

    def test_yaml_heuristic(self) -> None:
        assert _detect_language(None, "---\nkey: value") == "yaml"

    def test_dockerfile_heuristic(self) -> None:
        assert _detect_language(None, "FROM python:3.12") == "dockerfile"

    def test_none_for_empty(self) -> None:
        assert _detect_language(None, None) is None

    def test_none_for_unrecognized(self) -> None:
        assert _detect_language(None, "random text that matches nothing specific") is None


# =============================================================================
# _get_extension
# =============================================================================


class TestGetExtension:
    """Test _get_extension reverse lookup from SUPPORTED_EXTENSIONS."""

    def test_python(self) -> None:
        assert _get_extension("python") == "py"

    def test_javascript(self) -> None:
        assert _get_extension("javascript") == "js"

    def test_typescript(self) -> None:
        assert _get_extension("typescript") == "ts"

    def test_tsx(self) -> None:
        assert _get_extension("tsx") == "tsx"

    def test_php(self) -> None:
        assert _get_extension("php") == "php"

    def test_rust(self) -> None:
        assert _get_extension("rust") == "rs"

    def test_markdown(self) -> None:
        assert _get_extension("markdown") == "md"

    def test_yaml(self) -> None:
        assert _get_extension("yaml") == "yaml"

    def test_toml(self) -> None:
        assert _get_extension("toml") == "toml"

    def test_text(self) -> None:
        assert _get_extension("text") == "txt"

    def test_dockerfile(self) -> None:
        """Dockerfile should come from SUPPORTED_FILENAMES fallback."""
        assert _get_extension("dockerfile") == "dockerfile"

    def test_unknown_language_falls_back_to_txt(self) -> None:
        assert _get_extension("unknown_language_xyz") == "txt"

    def test_all_supported_languages_have_extension(self) -> None:
        """Every language in SUPPORTED_EXTENSIONS should produce a non-txt result."""
        all_languages = set(SUPPORTED_EXTENSIONS.values())
        for lang in all_languages:
            ext = _get_extension(lang)
            assert ext != "txt" or lang == "text", f"Language '{lang}' mapped to 'txt' unexpectedly"


# =============================================================================
# _find_uncaptured_nodes with unknown languages
# =============================================================================


class TestFindUncapturedNodesUnknown:
    """Test that _find_uncaptured_nodes handles unknown languages gracefully."""

    def test_unknown_language_returns_empty(self) -> None:
        """For a language not in interesting_types, should return empty list."""
        mock_node = MagicMock()
        mock_node.type = "some_node"
        mock_node.children = []

        gaps = _find_uncaptured_nodes(mock_node, set(), "bash")
        assert gaps == []

    def test_unknown_language_no_crash(self) -> None:
        """Completely unknown language should not crash."""
        mock_node = MagicMock()
        mock_node.type = "root"
        mock_node.children = []

        gaps = _find_uncaptured_nodes(mock_node, set(), "totally_unknown_lang")
        assert gaps == []

    def test_known_language_still_works(self) -> None:
        """Python (a known language) should still find gaps."""
        # Create a mock AST node that looks like an uncaptured function_definition
        mock_func = MagicMock()
        mock_func.type = "function_definition"
        mock_func.start_byte = 0
        mock_func.end_byte = 50
        mock_func.start_point = (0, 0)
        mock_func.text = b"def hello(): pass"
        mock_func.children = []

        mock_root = MagicMock()
        mock_root.type = "module"
        mock_root.children = [mock_func]

        # No captured ranges → this function is a gap
        gaps = _find_uncaptured_nodes(mock_root, set(), "python")
        assert len(gaps) == 1
        assert gaps[0].node_type == "function_definition"
