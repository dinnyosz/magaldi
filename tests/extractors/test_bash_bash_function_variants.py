"""Auto-generated parser test for bash_function_variants."""

import pytest
from pathlib import Path

from magaldi_core.code_parser import BashParser
from magaldi_core.change_detection import FileInfo


class TestBashFunctionVariants:
    """Tests for bash_function_variants extraction."""

    def test_bash_function_variants(self):
        code = 'die() { echo "$@" >&2; exit 1; }\n\nfunction setup() {\n    apt-get update\n    apt-get install -y curl\n}\n\nfunction teardown {\n    rm -rf /tmp/test-*\n}\n\nprocess_logs() {\n    cat /var/log/syslog | grep -i error | sort | uniq -c\n}\n'

        parser = BashParser()
        file_info = FileInfo(
            relative_path="test.sh",
            absolute_path=Path("/fake/test.sh"),
            language="bash",
        )

        elements = parser.parse(code, file_info, "scope", "repo", "main")

        # Check for die
        bash_function_variants_elem = next(
            (e for e in elements if e.name == "die" and e.element_type == "function"),
            None
        )
        assert bash_function_variants_elem is not None, f"Expected function 'die' not found"

        # Check for setup
        bash_function_variants_elem = next(
            (e for e in elements if e.name == "setup" and e.element_type == "function"),
            None
        )
        assert bash_function_variants_elem is not None, f"Expected function 'setup' not found"

        # Check for teardown
        bash_function_variants_elem = next(
            (e for e in elements if e.name == "teardown" and e.element_type == "function"),
            None
        )
        assert bash_function_variants_elem is not None, f"Expected function 'teardown' not found"

        # Check for process_logs
        bash_function_variants_elem = next(
            (e for e in elements if e.name == "process_logs" and e.element_type == "function"),
            None
        )
        assert bash_function_variants_elem is not None, f"Expected function 'process_logs' not found"

        # Check element count
        # 4 functions + 1 file element = 5
        assert len(elements) == 5, f"Expected 5 elements, got {len(elements)}"
