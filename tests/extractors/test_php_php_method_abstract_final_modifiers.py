"""Auto-generated parser test for php_method_abstract_final_modifiers."""

import pytest
from pathlib import Path

from magaldi_core.code_parser import PhpParser
from magaldi_core.change_detection import FileInfo


class TestPhpMethodAbstractFinalModifiers:
    """Tests for php_method_abstract_final_modifiers extraction."""

    def test_php_method_abstract_final_modifiers(self):
        code = '<?php\nabstract class BaseService {\n    abstract public function execute(): void;\n    final public function run(): void {}\n    public function normal(): void {}\n}\n'

        parser = PhpParser()
        file_info = FileInfo(
            relative_path="test.php",
            absolute_path=Path("/fake/test.php"),
            language="php",
        )

        elements = parser.parse(code, file_info, "scope", "repo", "main")

        # Check for BaseService
        php_method_abstract_final_modifiers_elem = next(
            (e for e in elements if e.name == "BaseService" and e.element_type == "class"),
            None
        )
        assert php_method_abstract_final_modifiers_elem is not None, f"Expected class 'BaseService' not found"

        # Check for execute
        php_method_abstract_final_modifiers_elem = next(
            (e for e in elements if e.name == "execute" and e.element_type == "method"),
            None
        )
        assert php_method_abstract_final_modifiers_elem is not None, f"Expected method 'execute' not found"
        assert "public" in php_method_abstract_final_modifiers_elem.decorators, "Missing decorator public"
        assert "abstract" in php_method_abstract_final_modifiers_elem.decorators, "Missing decorator abstract"

        # Check for run
        php_method_abstract_final_modifiers_elem = next(
            (e for e in elements if e.name == "run" and e.element_type == "method"),
            None
        )
        assert php_method_abstract_final_modifiers_elem is not None, f"Expected method 'run' not found"
        assert "public" in php_method_abstract_final_modifiers_elem.decorators, "Missing decorator public"
        assert "final" in php_method_abstract_final_modifiers_elem.decorators, "Missing decorator final"

        # Check for normal
        php_method_abstract_final_modifiers_elem = next(
            (e for e in elements if e.name == "normal" and e.element_type == "method"),
            None
        )
        assert php_method_abstract_final_modifiers_elem is not None, f"Expected method 'normal' not found"
        assert "public" in php_method_abstract_final_modifiers_elem.decorators, "Missing decorator public"
