"""Auto-generated parser test for php_static_method_signature."""

import pytest
from pathlib import Path

from magaldi_core.code_parser import PhpParser
from magaldi_core.change_detection import FileInfo


class TestPhpStaticMethodSignature:
    """Tests for php_static_method_signature extraction."""

    def test_php_static_method_signature(self):
        code = '<?php\n\nclass RetryMiddleware {\n    public static function exponentialDelay(int $retries): int {\n        return (int) pow(2, $retries - 1);\n    }\n    \n    public function normalMethod(): void {\n        // nothing\n    }\n    \n    protected static function helperMethod(string $key, int $value = 0): array {\n        return [$key => $value];\n    }\n}\n'

        parser = PhpParser()
        file_info = FileInfo(
            relative_path="test.php",
            absolute_path=Path("/fake/test.php"),
            language="php",
        )

        elements = parser.parse(code, file_info, "scope", "repo", "main")

        # Check for RetryMiddleware
        php_static_method_signature_elem = next(
            (e for e in elements if e.name == "RetryMiddleware" and e.element_type == "class"),
            None
        )
        assert php_static_method_signature_elem is not None, f"Expected class 'RetryMiddleware' not found"

        # Check for exponentialDelay
        exp_delay = next(
            (e for e in elements if e.name == "exponentialDelay" and e.element_type == "method"),
            None
        )
        assert exp_delay is not None, "Expected method 'exponentialDelay' not found"
        assert "public" in exp_delay.decorators, "Missing decorator public"
        assert "static" in exp_delay.decorators, "Missing decorator static"
        # Signature must include 'static' keyword
        assert "static" in exp_delay.signature, (
            f"Signature should contain 'static': {exp_delay.signature}"
        )
        assert exp_delay.signature.startswith("public static function exponentialDelay"), (
            f"Signature should start with 'public static function exponentialDelay': {exp_delay.signature}"
        )

        # Check for normalMethod (no static)
        normal = next(
            (e for e in elements if e.name == "normalMethod" and e.element_type == "method"),
            None
        )
        assert normal is not None, "Expected method 'normalMethod' not found"
        assert "public" in normal.decorators, "Missing decorator public"
        assert "static" not in normal.decorators, "normalMethod should not have static decorator"
        # Signature should NOT contain 'static'
        assert "static" not in normal.signature, (
            f"Signature should not contain 'static': {normal.signature}"
        )
        assert normal.signature.startswith("public function normalMethod"), (
            f"Signature should start with 'public function normalMethod': {normal.signature}"
        )

        # Check for helperMethod
        helper = next(
            (e for e in elements if e.name == "helperMethod" and e.element_type == "method"),
            None
        )
        assert helper is not None, "Expected method 'helperMethod' not found"
        assert "protected" in helper.decorators, "Missing decorator protected"
        assert "static" in helper.decorators, "Missing decorator static"
        # Signature must include 'static' keyword
        assert "static" in helper.signature, (
            f"Signature should contain 'static': {helper.signature}"
        )
        assert helper.signature.startswith("protected static function helperMethod"), (
            f"Signature should start with 'protected static function helperMethod': {helper.signature}"
        )
