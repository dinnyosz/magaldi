"""Auto-generated parser test for tsx_import_extraction."""

from pathlib import Path

from magaldi_core.change_detection import FileInfo
from magaldi_core.code_parser import JavaScriptParser


class TestTsxImportExtraction:
    """Tests for tsx_import_extraction extraction."""

    def test_tsx_import_extraction(self):
        code = "import React from 'react';\nimport { useState, useEffect } from 'react';\nimport * as d3 from 'd3';\nimport type { MyType } from './types';\n"

        parser = JavaScriptParser("tsx")
        file_info = FileInfo(
            relative_path="test.tsx",
            absolute_path=Path("/fake/test.tsx"),
            language="tsx",
        )

        elements = parser.parse(code, file_info, "scope", "repo", "main")

        # Get all import elements
        import_elements = [e for e in elements if e.element_type == "import"]
        import_names = [e.name for e in import_elements]

        # Should have imports for each unique module
        assert "react" in import_names, f"Expected import 'react' not found. Got: {import_names}"
        assert "d3" in import_names, f"Expected import 'd3' not found. Got: {import_names}"
        assert "./types" in import_names, f"Expected import './types' not found. Got: {import_names}"

        # Should have at least 3 unique imports (react may appear once or twice depending on deduplication)
        assert len(import_elements) >= 3, f"Expected at least 3 imports, got {len(import_elements)}"
