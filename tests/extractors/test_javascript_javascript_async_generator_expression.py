"""Auto-generated parser test for javascript_async_generator_expression."""

import pytest
from pathlib import Path

from magaldi_core.code_parser import JavaScriptParser
from magaldi_core.change_detection import FileInfo


class TestJavascriptAsyncGeneratorExpression:
    """Tests for javascript_async_generator_expression extraction."""

    def test_javascript_async_generator_expression(self):
        code = 'const paginateEach = async function* (response) {\n  let normalizedOptions;\n  while (response) {\n    normalizedOptions = response.request?.options;\n    yield response;\n    response = await response._pagination();\n  }\n};'

        parser = JavaScriptParser()
        file_info = FileInfo(
            relative_path="test.js",
            absolute_path=Path("/fake/test.js"),
            language="javascript",
        )

        elements = parser.parse(code, file_info, "scope", "repo", "main")

        # Check for paginateEach
        javascript_async_generator_expression_elem = next(
            (e for e in elements if e.name == "paginateEach" and e.element_type == "function"),
            None
        )
        assert javascript_async_generator_expression_elem is not None, f"Expected function 'paginateEach' not found"
        assert javascript_async_generator_expression_elem.is_async is True
