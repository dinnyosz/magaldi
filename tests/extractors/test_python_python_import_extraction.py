"""Auto-generated parser test for python_import_extraction."""

from pathlib import Path

from magaldi_core.change_detection import FileInfo
from magaldi_core.code_parser import PythonParser


class TestPythonImportExtraction:
    """Tests for python_import_extraction extraction."""

    def test_python_import_extraction(self):
        code = 'import os\nimport sys\nfrom pathlib import Path\nfrom typing import Optional, List\nfrom flask import Flask, request, jsonify\n'

        parser = PythonParser()
        file_info = FileInfo(
            relative_path="test.py",
            absolute_path=Path("/fake/test.py"),
            language="python",
        )

        elements = parser.parse(code, file_info, "scope", "repo", "main")

        # Check for os
        python_import_extraction_elem = next(
            (e for e in elements if e.name == "os" and e.element_type == "import"),
            None
        )
        assert python_import_extraction_elem is not None, "Expected import 'os' not found"

        # Check for sys
        python_import_extraction_elem = next(
            (e for e in elements if e.name == "sys" and e.element_type == "import"),
            None
        )
        assert python_import_extraction_elem is not None, "Expected import 'sys' not found"

        # Check for pathlib
        python_import_extraction_elem = next(
            (e for e in elements if e.name == "pathlib" and e.element_type == "import"),
            None
        )
        assert python_import_extraction_elem is not None, "Expected import 'pathlib' not found"

        # Check for typing
        python_import_extraction_elem = next(
            (e for e in elements if e.name == "typing" and e.element_type == "import"),
            None
        )
        assert python_import_extraction_elem is not None, "Expected import 'typing' not found"

        # Check for flask
        python_import_extraction_elem = next(
            (e for e in elements if e.name == "flask" and e.element_type == "import"),
            None
        )
        assert python_import_extraction_elem is not None, "Expected import 'flask' not found"
