"""Auto-generated parser test for rust_let_bindings_are_variables_not_constants."""

import pytest
from pathlib import Path

from magaldi_core.code_parser import RustParser
from magaldi_core.change_detection import FileInfo


class TestRustLetBindingsAreVariablesNotConstants:
    """Tests for rust_let_bindings_are_variables_not_constants extraction."""

    def test_rust_let_bindings_are_variables_not_constants(self):
        code = 'const MAX_SIZE: usize = 100;\nstatic GLOBAL_NAME: &str = "hello";\n\nfn process() {\n    let x = 42;\n    let mut name = String::from("world");\n    let result: i32 = x + 1;\n}'

        parser = RustParser()
        file_info = FileInfo(
            relative_path="test.rs",
            absolute_path=Path("/fake/test.rs"),
            language="rust",
        )

        elements = parser.parse(code, file_info, "scope", "repo", "main")

        # Check for MAX_SIZE
        rust_let_bindings_are_variables_not_constants_elem = next(
            (e for e in elements if e.name == "MAX_SIZE" and e.element_type == "constant"),
            None
        )
        assert rust_let_bindings_are_variables_not_constants_elem is not None, f"Expected constant 'MAX_SIZE' not found"

        # Check for GLOBAL_NAME
        rust_let_bindings_are_variables_not_constants_elem = next(
            (e for e in elements if e.name == "GLOBAL_NAME" and e.element_type == "variable"),
            None
        )
        assert rust_let_bindings_are_variables_not_constants_elem is not None, f"Expected variable 'GLOBAL_NAME' not found"

        # Check for process
        rust_let_bindings_are_variables_not_constants_elem = next(
            (e for e in elements if e.name == "process" and e.element_type == "function"),
            None
        )
        assert rust_let_bindings_are_variables_not_constants_elem is not None, f"Expected function 'process' not found"

        # Check for x
        rust_let_bindings_are_variables_not_constants_elem = next(
            (e for e in elements if e.name == "x" and e.element_type == "variable"),
            None
        )
        assert rust_let_bindings_are_variables_not_constants_elem is not None, f"Expected variable 'x' not found"

        # Check for name
        rust_let_bindings_are_variables_not_constants_elem = next(
            (e for e in elements if e.name == "name" and e.element_type == "variable"),
            None
        )
        assert rust_let_bindings_are_variables_not_constants_elem is not None, f"Expected variable 'name' not found"

        # Check for result
        rust_let_bindings_are_variables_not_constants_elem = next(
            (e for e in elements if e.name == "result" and e.element_type == "variable"),
            None
        )
        assert rust_let_bindings_are_variables_not_constants_elem is not None, f"Expected variable 'result' not found"
