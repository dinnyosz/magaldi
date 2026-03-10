"""Auto-generated parser test for php_static_method_signature."""

import pytest
from pathlib import Path

from magaldi_core.code_parser import PhpParser
from magaldi_core.change_detection import FileInfo


class TestPhpStaticMethodSignature:
    """Tests for php_static_method_signature extraction."""

    def test_php_static_method_signature(self):
        code = '<?php\nclass Utils {\n    public static function create(string $name): self {}\n    public function normal(): void {}\n}\n'

        parser = PhpParser()
        file_info = FileInfo(
            relative_path="test.php",
            absolute_path=Path("/fake/test.php"),
            language="php",
        )

        elements = parser.parse(code, file_info, "scope", "repo", "main")

        # Check for Utils
        php_static_method_signature_elem = next(
            (e for e in elements if e.name == "Utils" and e.element_type == "class"),
            None
        )
        assert php_static_method_signature_elem is not None, f"Expected class 'Utils' not found"

        # Check for create
        php_static_method_signature_elem = next(
            (e for e in elements if e.name == "create" and e.element_type == "method"),
            None
        )
        assert php_static_method_signature_elem is not None, f"Expected method 'create' not found"
        assert "public" in php_static_method_signature_elem.decorators, "Missing decorator public"
        assert "static" in php_static_method_signature_elem.decorators, "Missing decorator static"
        assert "static" in php_static_method_signature_elem.signature, \
            f"Signature should contain 'static' but was: {php_static_method_signature_elem.signature}"

        # Check for normal
        php_static_method_signature_elem = next(
            (e for e in elements if e.name == "normal" and e.element_type == "method"),
            None
        )
        assert php_static_method_signature_elem is not None, f"Expected method 'normal' not found"
        assert "public" in php_static_method_signature_elem.decorators, "Missing decorator public"
