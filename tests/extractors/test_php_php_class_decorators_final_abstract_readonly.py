"""Auto-generated parser test for php_class_decorators_final_abstract_readonly."""

import pytest
from pathlib import Path

from magaldi_core.code_parser import PhpParser
from magaldi_core.change_detection import FileInfo


class TestPhpClassDecoratorsFinalAbstractReadonly:
    """Tests for php_class_decorators_final_abstract_readonly extraction."""

    def test_php_class_decorators_final_abstract_readonly(self):
        code = '<?php\nfinal class Utils {\n    public function helper(): void {}\n}\n\nabstract class BaseService {\n    abstract public function execute(): void;\n}\n\nreadonly class ValueObject {\n    public function __construct(public string $name) {}\n}\n'

        parser = PhpParser()
        file_info = FileInfo(
            relative_path="test.php",
            absolute_path=Path("/fake/test.php"),
            language="php",
        )

        elements = parser.parse(code, file_info, "scope", "repo", "main")

        # Check for Utils
        php_class_decorators_final_abstract_readonly_elem = next(
            (e for e in elements if e.name == "Utils" and e.element_type == "class"),
            None
        )
        assert php_class_decorators_final_abstract_readonly_elem is not None, f"Expected class 'Utils' not found"
        assert "final" in php_class_decorators_final_abstract_readonly_elem.decorators, "Missing decorator final"

        # Check for BaseService
        php_class_decorators_final_abstract_readonly_elem = next(
            (e for e in elements if e.name == "BaseService" and e.element_type == "class"),
            None
        )
        assert php_class_decorators_final_abstract_readonly_elem is not None, f"Expected class 'BaseService' not found"
        assert "abstract" in php_class_decorators_final_abstract_readonly_elem.decorators, "Missing decorator abstract"

        # Check for ValueObject
        php_class_decorators_final_abstract_readonly_elem = next(
            (e for e in elements if e.name == "ValueObject" and e.element_type == "class"),
            None
        )
        assert php_class_decorators_final_abstract_readonly_elem is not None, f"Expected class 'ValueObject' not found"
        assert "readonly" in php_class_decorators_final_abstract_readonly_elem.decorators, "Missing decorator readonly"
