"""Auto-generated parser test for bash_variable_types."""

from pathlib import Path

from magaldi_core.change_detection import FileInfo
from magaldi_core.code_parser import BashParser


class TestBashVariableTypes:
    """Tests for bash_variable_types extraction."""

    def test_bash_variable_types(self):
        code = 'VERSION="1.0.0"\nreadonly DB_HOST="localhost"\nexport DB_PORT=5432\ndeclare -r APP_NAME="myapp"\ndeclare -a ITEMS=("one" "two")\ndeclare -A MAP=([key1]="val1")\nTIMESTAMP=$(date +%Y%m%d)\n'

        parser = BashParser()
        file_info = FileInfo(
            relative_path="test.sh",
            absolute_path=Path("/fake/test.sh"),
            language="bash",
        )

        elements = parser.parse(code, file_info, "scope", "repo", "main")

        # Check for VERSION
        bash_variable_types_elem = next(
            (e for e in elements if e.name == "VERSION" and e.element_type == "constant"),
            None
        )
        assert bash_variable_types_elem is not None, "Expected constant 'VERSION' not found"

        # Check for DB_HOST
        bash_variable_types_elem = next(
            (e for e in elements if e.name == "DB_HOST" and e.element_type == "constant"),
            None
        )
        assert bash_variable_types_elem is not None, "Expected constant 'DB_HOST' not found"

        # Check for DB_PORT
        bash_variable_types_elem = next(
            (e for e in elements if e.name == "DB_PORT" and e.element_type == "constant"),
            None
        )
        assert bash_variable_types_elem is not None, "Expected constant 'DB_PORT' not found"

        # Check for APP_NAME
        bash_variable_types_elem = next(
            (e for e in elements if e.name == "APP_NAME" and e.element_type == "constant"),
            None
        )
        assert bash_variable_types_elem is not None, "Expected constant 'APP_NAME' not found"

        # Check for ITEMS
        bash_variable_types_elem = next(
            (e for e in elements if e.name == "ITEMS" and e.element_type == "variable"),
            None
        )
        assert bash_variable_types_elem is not None, "Expected variable 'ITEMS' not found"

        # Check for MAP
        bash_variable_types_elem = next(
            (e for e in elements if e.name == "MAP" and e.element_type == "variable"),
            None
        )
        assert bash_variable_types_elem is not None, "Expected variable 'MAP' not found"

        # Check for TIMESTAMP
        bash_variable_types_elem = next(
            (e for e in elements if e.name == "TIMESTAMP" and e.element_type == "constant"),
            None
        )
        assert bash_variable_types_elem is not None, "Expected constant 'TIMESTAMP' not found"

        # Check element count
        # 7 variables/constants + 1 file element = 8
        assert len(elements) == 8, f"Expected 8 elements, got {len(elements)}"
