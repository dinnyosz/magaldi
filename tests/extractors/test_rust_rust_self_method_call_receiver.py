"""Auto-generated parser test for rust_self_method_call_receiver."""

from pathlib import Path

from magaldi_core.code_parser import RustParser
from magaldi_core.change_detection import FileInfo


class TestRustSelfMethodCallReceiver:
    """Tests for self.method() calls having receiver='self'."""

    def test_self_method_calls_have_self_receiver(self):
        code = (
            'struct Walker {\n'
            '    config: Config,\n'
            '}\n'
            '\n'
            'impl Walker {\n'
            '    fn scan(&self, paths: &[PathBuf]) -> Result<ExitCode> {\n'
            '        let walker = self.build_walker(paths)?;\n'
            '        let receiver = self.receive(rx);\n'
            '        self.spawn_senders(walker, tx);\n'
            '        process_results(walker);\n'
            '        Config::new();\n'
            '    }\n'
            '}\n'
        )

        parser = RustParser()
        file_info = FileInfo(
            relative_path="test.rs",
            absolute_path=Path("/fake/test.rs"),
            language="rust",
        )

        elements = parser.parse(code, file_info, "scope", "repo", "main")

        scan_elem = next(
            (e for e in elements if e.name == "scan"),
            None,
        )
        assert scan_elem is not None, "Expected method 'scan' not found"
        assert scan_elem.calls, "Expected calls in scan method"

        calls_by_name = {c.name: c for c in scan_elem.calls}

        # self.build_walker() -> receiver should be "self"
        assert "build_walker" in calls_by_name, "Expected call to build_walker"
        assert calls_by_name["build_walker"].receiver == "self", (
            f"Expected receiver='self' for build_walker, got {calls_by_name['build_walker'].receiver!r}"
        )

        # self.receive() -> receiver should be "self"
        assert "receive" in calls_by_name, "Expected call to receive"
        assert calls_by_name["receive"].receiver == "self", (
            f"Expected receiver='self' for receive, got {calls_by_name['receive'].receiver!r}"
        )

        # self.spawn_senders() -> receiver should be "self"
        assert "spawn_senders" in calls_by_name, "Expected call to spawn_senders"
        assert calls_by_name["spawn_senders"].receiver == "self", (
            f"Expected receiver='self' for spawn_senders, got {calls_by_name['spawn_senders'].receiver!r}"
        )

        # process_results() -> bare call, no receiver
        assert "process_results" in calls_by_name, "Expected call to process_results"
        assert calls_by_name["process_results"].receiver is None, (
            f"Expected receiver=None for process_results, got {calls_by_name['process_results'].receiver!r}"
        )

        # Config::new() -> receiver should be "Config"
        assert "new" in calls_by_name, "Expected call to new (Config::new)"
        assert calls_by_name["new"].receiver == "Config", (
            f"Expected receiver='Config' for new, got {calls_by_name['new'].receiver!r}"
        )

    def test_chained_self_calls(self):
        """Test that chained self calls like self.foo().bar() are handled."""
        code = (
            'impl Builder {\n'
            '    fn build(&self) {\n'
            '        self.validate();\n'
            '        other.process();\n'
            '    }\n'
            '}\n'
        )

        parser = RustParser()
        file_info = FileInfo(
            relative_path="test.rs",
            absolute_path=Path("/fake/test.rs"),
            language="rust",
        )

        elements = parser.parse(code, file_info, "scope", "repo", "main")

        build_elem = next(
            (e for e in elements if e.name == "build"),
            None,
        )
        assert build_elem is not None, "Expected method 'build' not found"

        calls_by_name = {c.name: c for c in build_elem.calls}

        # self.validate() -> receiver should be "self"
        assert "validate" in calls_by_name, "Expected call to validate"
        assert calls_by_name["validate"].receiver == "self", (
            f"Expected receiver='self', got {calls_by_name['validate'].receiver!r}"
        )

        # other.process() -> receiver should be "other"
        assert "process" in calls_by_name, "Expected call to process"
        assert calls_by_name["process"].receiver == "other", (
            f"Expected receiver='other', got {calls_by_name['process'].receiver!r}"
        )
