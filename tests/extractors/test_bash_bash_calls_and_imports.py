"""Auto-generated parser test for bash_calls_and_imports."""

from pathlib import Path

from magaldi_core.change_detection import FileInfo
from magaldi_core.code_parser import BashParser


class TestBashCallsAndImports:
    """Tests for bash_calls_and_imports extraction."""

    def test_bash_calls_and_imports(self):
        code = 'source ./lib/helpers.sh\n. /etc/profile.d/custom.sh\n\ndeploy() {\n    kubectl apply -f manifests/\n    helm upgrade --install myapp ./chart\n    notify_team "Deployed"\n    echo "done"\n}\n'

        parser = BashParser()
        file_info = FileInfo(
            relative_path="test.sh",
            absolute_path=Path("/fake/test.sh"),
            language="bash",
        )

        elements = parser.parse(code, file_info, "scope", "repo", "main")

        # Check for deploy
        bash_calls_and_imports_elem = next(
            (e for e in elements if e.name == "deploy" and e.element_type == "function"),
            None
        )
        assert bash_calls_and_imports_elem is not None, "Expected function 'deploy' not found"
