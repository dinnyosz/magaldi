"""Auto-generated parser test for php_static_call_receiver."""

import pytest
from pathlib import Path

from magaldi_core.code_parser import PhpParser
from magaldi_core.change_detection import FileInfo


class TestPhpStaticCallReceiver:
    """Tests for php_static_call_receiver extraction."""

    def test_php_static_call_receiver(self):
        code = '<?php\nfunction dispatch() {\n    Utils::chooseHandler();\n    parent::__construct();\n    self::getInstance();\n}\n'

        parser = PhpParser()
        file_info = FileInfo(
            relative_path="test.php",
            absolute_path=Path("/fake/test.php"),
            language="php",
        )

        elements = parser.parse(code, file_info, "scope", "repo", "main")

        # Check for dispatch
        php_static_call_receiver_elem = next(
            (e for e in elements if e.name == "dispatch" and e.element_type == "function"),
            None
        )
        assert php_static_call_receiver_elem is not None, f"Expected function 'dispatch' not found"

        # Check static call extraction: receiver and name must be correct
        calls = php_static_call_receiver_elem.calls
        assert len(calls) == 3, f"Expected 3 calls, got {len(calls)}: {[(c.name, c.receiver) for c in calls]}"

        # Utils::chooseHandler() - class name as receiver
        utils_call = next((c for c in calls if c.name == "chooseHandler"), None)
        assert utils_call is not None, "Missing call to chooseHandler"
        assert utils_call.receiver == "Utils", \
            f"Expected receiver 'Utils' but got '{utils_call.receiver}'"

        # parent::__construct() - relative scope as receiver
        parent_call = next((c for c in calls if c.name == "__construct"), None)
        assert parent_call is not None, "Missing call to __construct"
        assert parent_call.receiver == "parent", \
            f"Expected receiver 'parent' but got '{parent_call.receiver}'"

        # self::getInstance() - relative scope as receiver
        self_call = next((c for c in calls if c.name == "getInstance"), None)
        assert self_call is not None, "Missing call to getInstance"
        assert self_call.receiver == "self", \
            f"Expected receiver 'self' but got '{self_call.receiver}'"
