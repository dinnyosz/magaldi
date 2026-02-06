"""Tests for the Bash parser and extractor."""

from pathlib import Path

from magaldi_core.change_detection import FileInfo
from magaldi_core.extractors.bash import (
    BashExtractor,
    extract_bash_calls,
    extract_bash_imports,
)
from magaldi_core.parsers.bash import BashParser
from magaldi_core.tree_sitter_manager import get_manager


class TestBashExtractor:
    """Tests for BashExtractor element extraction."""

    def _extract(self, code: str):
        manager = get_manager()
        tree = manager.parse(code.encode("utf-8"), "bash")
        extractor = BashExtractor()
        return extractor.extract_elements(tree, code.split("\n"))

    def test_extract_function(self):
        code = 'deploy() {\n    kubectl apply -f manifests/\n}\n'
        elements = self._extract(code)
        funcs = [e for e in elements if e.element_type == "function"]
        assert len(funcs) == 1
        assert funcs[0].name == "deploy"
        assert funcs[0].line_start == 1
        assert funcs[0].line_end == 3

    def test_extract_function_with_keyword(self):
        code = 'function deploy {\n    echo "hi"\n}\n'
        elements = self._extract(code)
        funcs = [e for e in elements if e.element_type == "function"]
        assert len(funcs) == 1
        assert funcs[0].name == "deploy"

    def test_extract_multiple_functions(self):
        code = 'foo() {\n    echo 1\n}\n\nbar() {\n    echo 2\n}\n'
        elements = self._extract(code)
        funcs = [e for e in elements if e.element_type == "function"]
        assert len(funcs) == 2
        assert funcs[0].name == "foo"
        assert funcs[1].name == "bar"

    def test_extract_readonly_constant(self):
        code = 'readonly MY_VAR="hello"\n'
        elements = self._extract(code)
        constants = [e for e in elements if e.element_type == "constant"]
        assert len(constants) == 1
        assert constants[0].name == "MY_VAR"

    def test_extract_export_constant(self):
        code = 'export API_URL="https://example.com"\n'
        elements = self._extract(code)
        constants = [e for e in elements if e.element_type == "constant"]
        assert len(constants) == 1
        assert constants[0].name == "API_URL"

    def test_extract_plain_variable_as_constant(self):
        code = 'VERSION="1.0.0"\n'
        elements = self._extract(code)
        constants = [e for e in elements if e.element_type == "constant"]
        assert len(constants) == 1
        assert constants[0].name == "VERSION"

    def test_extract_declare_readonly(self):
        code = 'declare -r CONST_VAL="fixed"\n'
        elements = self._extract(code)
        constants = [e for e in elements if e.element_type == "constant"]
        assert len(constants) == 1
        assert constants[0].name == "CONST_VAL"

    def test_extract_declare_array_as_variable(self):
        code = 'declare -A MY_ARRAY\n'
        elements = self._extract(code)
        variables = [e for e in elements if e.element_type == "variable"]
        assert len(variables) == 1
        assert variables[0].name == "MY_ARRAY"

    def test_skip_local_declarations(self):
        """Local declarations inside functions should not appear at top level."""
        code = 'my_func() {\n    local x="hello"\n}\n'
        elements = self._extract(code)
        # Should only have the function, no variable
        assert len(elements) == 1
        assert elements[0].element_type == "function"

    def test_empty_script(self):
        code = ""
        elements = self._extract(code)
        assert len(elements) == 0

    def test_shebang_only(self):
        code = "#!/bin/bash\n"
        elements = self._extract(code)
        assert len(elements) == 0

    def test_comments_only(self):
        code = "#!/bin/bash\n# This is a comment\n# Another comment\n"
        elements = self._extract(code)
        assert len(elements) == 0

    def test_raw_code_content(self):
        code = 'deploy() {\n    kubectl apply -f .\n    echo "done"\n}\n'
        elements = self._extract(code)
        func = elements[0]
        assert "kubectl apply" in func.raw_code
        assert 'echo "done"' in func.raw_code


class TestBashImports:
    """Tests for bash source/. import extraction."""

    def _extract_imports(self, code: str):
        manager = get_manager()
        tree = manager.parse(code.encode("utf-8"), "bash")
        return extract_bash_imports(tree, code.split("\n"))

    def test_source_import(self):
        code = 'source ./lib/common.sh\n'
        imports = self._extract_imports(code)
        assert len(imports) == 1
        assert imports[0].module == "./lib/common.sh"
        assert imports[0].name == "common.sh"

    def test_dot_import(self):
        code = '. ./lib/utils.sh\n'
        imports = self._extract_imports(code)
        assert len(imports) == 1
        assert imports[0].module == "./lib/utils.sh"
        assert imports[0].name == "utils.sh"

    def test_multiple_imports(self):
        code = 'source ./a.sh\nsource ./b.sh\n. ./c.sh\n'
        imports = self._extract_imports(code)
        assert len(imports) == 3

    def test_quoted_import_path(self):
        code = 'source "./lib/common.sh"\n'
        imports = self._extract_imports(code)
        assert len(imports) == 1
        assert imports[0].module == "./lib/common.sh"

    def test_no_imports(self):
        code = 'echo "hello"\n'
        imports = self._extract_imports(code)
        assert len(imports) == 0


class TestBashCalls:
    """Tests for bash call extraction within functions."""

    def _extract_calls(self, code: str):
        manager = get_manager()
        tree = manager.parse(code.encode("utf-8"), "bash")
        # Find the function_definition node
        for child in tree.root_node.children:
            if child.type == "function_definition":
                return extract_bash_calls(child)
        return []

    def test_extract_command_calls(self):
        code = 'deploy() {\n    kubectl apply -f .\n    helm upgrade app\n}\n'
        calls = self._extract_calls(code)
        names = [c.name for c in calls]
        assert "kubectl" in names
        assert "helm" in names

    def test_skip_builtins(self):
        code = 'my_func() {\n    echo "hi"\n    cd /tmp\n    local x=1\n    custom_cmd\n}\n'
        calls = self._extract_calls(code)
        names = [c.name for c in calls]
        assert "echo" not in names
        assert "cd" not in names
        assert "local" not in names
        assert "custom_cmd" in names

    def test_skip_source_imports(self):
        code = 'my_func() {\n    source ./lib.sh\n    . ./util.sh\n    real_cmd\n}\n'
        calls = self._extract_calls(code)
        names = [c.name for c in calls]
        assert "source" not in names
        assert "." not in names
        assert "real_cmd" in names

    def test_no_calls(self):
        code = 'my_func() {\n    echo "just echo"\n}\n'
        calls = self._extract_calls(code)
        assert len(calls) == 0

    def test_calls_have_no_receiver(self):
        """Bash commands don't have receivers like OOP methods."""
        code = 'my_func() {\n    kubectl get pods\n}\n'
        calls = self._extract_calls(code)
        assert len(calls) == 1
        assert calls[0].receiver is None

    def test_no_duplicate_calls(self):
        code = 'my_func() {\n    kubectl get pods\n    kubectl apply -f .\n}\n'
        calls = self._extract_calls(code)
        # Both should appear since they're on different lines
        assert len(calls) == 2


class TestBashParser:
    """Tests for the full BashParser."""

    def _parse(self, content: str, filename: str = "deploy.sh") -> list:
        parser = BashParser()
        file_info = FileInfo(
            relative_path=f"scripts/{filename}",
            absolute_path=Path(f"/fake/scripts/{filename}"),
            language="bash",
        )
        return parser.parse(content, file_info, "scope", "repo", "main")

    def test_creates_file_element(self):
        code = '#!/bin/bash\necho "hello"\n'
        elements = self._parse(code)
        file_elems = [e for e in elements if e.element_type == "file"]
        assert len(file_elems) == 1
        assert file_elems[0].name == "deploy.sh"
        assert file_elems[0].language == "bash"
        assert file_elems[0].level == 0

    def test_functions_have_correct_level(self):
        code = 'deploy() {\n    echo "deploying"\n}\n'
        elements = self._parse(code)
        funcs = [e for e in elements if e.element_type == "function"]
        assert len(funcs) == 1
        assert funcs[0].level == 2

    def test_constants_have_correct_level(self):
        code = 'readonly VERSION="1.0"\n'
        elements = self._parse(code)
        constants = [e for e in elements if e.element_type == "constant"]
        assert len(constants) == 1
        assert constants[0].level == 3

    def test_elements_have_parent_id(self):
        code = '#!/bin/bash\nreadonly VAR="x"\ndeploy() { echo ok; }\n'
        elements = self._parse(code)
        file_elem = next(e for e in elements if e.element_type == "file")
        for elem in elements:
            if elem.element_type == "file":
                continue
            assert elem.parent_id == file_elem.element_id

    def test_element_ids_are_unique(self):
        code = '#!/bin/bash\nVAR1="a"\nVAR2="b"\nfoo() { echo 1; }\nbar() { echo 2; }\n'
        elements = self._parse(code)
        ids = [e.element_id for e in elements]
        assert len(ids) == len(set(ids))

    def test_imports_on_file_element(self):
        code = '#!/bin/bash\nsource ./lib/common.sh\n. ./lib/utils.sh\n'
        elements = self._parse(code)
        file_elem = next(e for e in elements if e.element_type == "file")
        assert len(file_elem.imports) == 2
        modules = [i.module for i in file_elem.imports]
        assert "./lib/common.sh" in modules
        assert "./lib/utils.sh" in modules

    def test_calls_on_function_elements(self):
        code = 'deploy() {\n    kubectl apply -f .\n    helm upgrade app\n}\n'
        elements = self._parse(code)
        func = next(e for e in elements if e.element_type == "function")
        call_names = [c.name for c in func.calls]
        assert "kubectl" in call_names
        assert "helm" in call_names

    def test_same_file_call_resolution(self):
        code = 'helper() {\n    echo "helping"\n}\nmain() {\n    helper\n}\n'
        elements = self._parse(code)
        main_func = next(e for e in elements if e.name == "main")
        helper_call = next(c for c in main_func.calls if c.name == "helper")
        assert helper_call.resolved_id is not None

    def test_empty_script(self):
        elements = self._parse("")
        assert len(elements) == 1
        assert elements[0].element_type == "file"

    def test_complex_script(self):
        """Test a realistic script with mixed elements."""
        code = '''#!/bin/bash
set -euo pipefail

readonly DEPLOY_ENV="production"
export API_URL="https://api.example.com"

source ./lib/common.sh

deploy_to_k8s() {
    local namespace="$1"
    kubectl apply -f manifests/
    helm upgrade --install myapp ./chart
    notify_team "Deployed to $namespace"
}

run_tests() {
    pytest tests/
    deploy_to_k8s staging
}

cleanup() {
    rm -rf /tmp/build-*
    docker system prune -f
}
'''
        elements = self._parse(code)

        # File element
        file_elems = [e for e in elements if e.element_type == "file"]
        assert len(file_elems) == 1

        # Constants
        constants = [e for e in elements if e.element_type == "constant"]
        const_names = [c.name for c in constants]
        assert "DEPLOY_ENV" in const_names
        assert "API_URL" in const_names

        # Functions
        funcs = [e for e in elements if e.element_type == "function"]
        func_names = [f.name for f in funcs]
        assert "deploy_to_k8s" in func_names
        assert "run_tests" in func_names
        assert "cleanup" in func_names

        # Imports on file element
        assert len(file_elems[0].imports) == 1
        assert file_elems[0].imports[0].module == "./lib/common.sh"

        # Calls within functions
        deploy_func = next(f for f in funcs if f.name == "deploy_to_k8s")
        deploy_calls = [c.name for c in deploy_func.calls]
        assert "kubectl" in deploy_calls
        assert "helm" in deploy_calls
        assert "notify_team" in deploy_calls

        # Cross-function call resolution
        tests_func = next(f for f in funcs if f.name == "run_tests")
        deploy_call = next((c for c in tests_func.calls if c.name == "deploy_to_k8s"), None)
        assert deploy_call is not None
        assert deploy_call.resolved_id is not None

    def test_language_is_bash(self):
        code = '#!/bin/bash\nfoo() { echo 1; }\n'
        elements = self._parse(code)
        for elem in elements:
            assert elem.language == "bash"
