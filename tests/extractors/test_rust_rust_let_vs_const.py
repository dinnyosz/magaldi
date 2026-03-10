"""Auto-generated parser test for rust_let_vs_const."""

import pytest
from pathlib import Path

from magaldi_core.code_parser import RustParser
from magaldi_core.change_detection import FileInfo


class TestRustLetVsConst:
    """Tests for rust_let_vs_const extraction."""

    def test_rust_let_vs_const(self):
        code = 'const MAX_SIZE: usize = 1024;\nstatic COUNTER: u64 = 0;\nstatic mut GLOBAL_FLAG: bool = false;\n\nfn process_data(input: &str) -> bool {\n    let path = get_path();\n    let mut opts = Options::default();\n    const INNER_CONST: u32 = 42;\n    true\n}\n\nstruct MyService {\n    name: String,\n}\n\nimpl MyService {\n    pub fn new(name: String) -> Self {\n        let trimmed = name.trim().to_string();\n        let mut result = MyService { name: trimmed };\n        result\n    }\n\n    fn private_method(&self) -> bool {\n        let config = load_config();\n        const LIMIT: usize = 100;\n        config.len() < LIMIT\n    }\n}\n'

        parser = RustParser()
        file_info = FileInfo(
            relative_path="test.rs",
            absolute_path=Path("/fake/test.rs"),
            language="rust",
        )

        elements = parser.parse(code, file_info, "scope", "repo", "main")

        # Check for MAX_SIZE
        rust_let_vs_const_elem = next(
            (e for e in elements if e.name == "MAX_SIZE" and e.element_type == "constant"),
            None
        )
        assert rust_let_vs_const_elem is not None, f"Expected constant 'MAX_SIZE' not found"

        # Check for COUNTER
        rust_let_vs_const_elem = next(
            (e for e in elements if e.name == "COUNTER" and e.element_type == "variable"),
            None
        )
        assert rust_let_vs_const_elem is not None, f"Expected variable 'COUNTER' not found"
        assert "static" in rust_let_vs_const_elem.decorators, "Missing decorator static"

        # Check for GLOBAL_FLAG
        rust_let_vs_const_elem = next(
            (e for e in elements if e.name == "GLOBAL_FLAG" and e.element_type == "variable"),
            None
        )
        assert rust_let_vs_const_elem is not None, f"Expected variable 'GLOBAL_FLAG' not found"
        assert "static" in rust_let_vs_const_elem.decorators, "Missing decorator static"
        assert "mut" in rust_let_vs_const_elem.decorators, "Missing decorator mut"

        # Check for process_data
        rust_let_vs_const_elem = next(
            (e for e in elements if e.name == "process_data" and e.element_type == "function"),
            None
        )
        assert rust_let_vs_const_elem is not None, f"Expected function 'process_data' not found"

        # Check for path
        rust_let_vs_const_elem = next(
            (e for e in elements if e.name == "path" and e.element_type == "variable"),
            None
        )
        assert rust_let_vs_const_elem is not None, f"Expected variable 'path' not found"

        # Check for opts
        rust_let_vs_const_elem = next(
            (e for e in elements if e.name == "opts" and e.element_type == "variable"),
            None
        )
        assert rust_let_vs_const_elem is not None, f"Expected variable 'opts' not found"
        assert "mut" in rust_let_vs_const_elem.decorators, "Missing decorator mut"

        # Check for INNER_CONST
        rust_let_vs_const_elem = next(
            (e for e in elements if e.name == "INNER_CONST" and e.element_type == "constant"),
            None
        )
        assert rust_let_vs_const_elem is not None, f"Expected constant 'INNER_CONST' not found"

        # Check for MyService
        rust_let_vs_const_elem = next(
            (e for e in elements if e.name == "MyService" and e.element_type == "class"),
            None
        )
        assert rust_let_vs_const_elem is not None, f"Expected class 'MyService' not found"

        # Check for MyService impl block (second MyService class, with impl decorator)
        rust_let_vs_const_elem = next(
            (e for e in elements if e.name == "MyService" and e.element_type == "class"
             and "impl" in (e.decorators or [])),
            None
        )
        assert rust_let_vs_const_elem is not None, f"Expected class 'MyService' with impl decorator not found"
        assert "impl" in rust_let_vs_const_elem.decorators, "Missing decorator impl"

        # Check for new
        rust_let_vs_const_elem = next(
            (e for e in elements if e.name == "new" and e.element_type == "function"),
            None
        )
        assert rust_let_vs_const_elem is not None, f"Expected function 'new' not found"

        # Check for trimmed
        rust_let_vs_const_elem = next(
            (e for e in elements if e.name == "trimmed" and e.element_type == "variable"),
            None
        )
        assert rust_let_vs_const_elem is not None, f"Expected variable 'trimmed' not found"

        # Check for result
        rust_let_vs_const_elem = next(
            (e for e in elements if e.name == "result" and e.element_type == "variable"),
            None
        )
        assert rust_let_vs_const_elem is not None, f"Expected variable 'result' not found"
        assert "mut" in rust_let_vs_const_elem.decorators, "Missing decorator mut"

        # Check for private_method
        rust_let_vs_const_elem = next(
            (e for e in elements if e.name == "private_method" and e.element_type == "method"),
            None
        )
        assert rust_let_vs_const_elem is not None, f"Expected method 'private_method' not found"

        # Check for config
        rust_let_vs_const_elem = next(
            (e for e in elements if e.name == "config" and e.element_type == "variable"),
            None
        )
        assert rust_let_vs_const_elem is not None, f"Expected variable 'config' not found"

        # Check for LIMIT
        rust_let_vs_const_elem = next(
            (e for e in elements if e.name == "LIMIT" and e.element_type == "constant"),
            None
        )
        assert rust_let_vs_const_elem is not None, f"Expected constant 'LIMIT' not found"
