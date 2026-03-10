"""Parser tests for TypeScript visibility modifier extraction.

Covers:
- Standard method visibility (private/protected/public)
- Field visibility
- Arrow function field methods with visibility
- Getter/setter with visibility
- Static methods with visibility
- Readonly fields with visibility
- Abstract methods with visibility
"""

from pathlib import Path

from magaldi_core.code_parser import JavaScriptParser
from magaldi_core.change_detection import FileInfo


def _parse_ts(code: str) -> list:
    """Helper to parse TypeScript code and return elements."""
    parser = JavaScriptParser("typescript")
    file_info = FileInfo(
        relative_path="test.ts",
        absolute_path=Path("/fake/test.ts"),
        language="typescript",
    )
    return parser.parse(code, file_info, "scope", "repo", "main")


def _lookup(elements, element_type, name):
    """Find element by type and name."""
    for e in elements:
        if e.element_type == element_type and e.name == name:
            return e
    available = [(e.element_type, e.name) for e in elements]
    raise AssertionError(f"Element ({element_type}, {name}) not found. Available: {available}")


class TestTypescriptMethodVisibility:
    """Tests for standard method visibility modifiers."""

    def test_private_method(self):
        elements = _parse_ts(
            'class Foo {\n'
            '  private open(): void {\n'
            '    console.log("open");\n'
            '  }\n'
            '}'
        )
        method = _lookup(elements, "method", "open")
        assert method.visibility == "private"

    def test_protected_method(self):
        elements = _parse_ts(
            'class Foo {\n'
            '  protected handle(msg: string): void {\n'
            '    console.log(msg);\n'
            '  }\n'
            '}'
        )
        method = _lookup(elements, "method", "handle")
        assert method.visibility == "protected"

    def test_public_method(self):
        elements = _parse_ts(
            'class Foo {\n'
            '  public send(data: string): void {\n'
            '    console.log(data);\n'
            '  }\n'
            '}'
        )
        method = _lookup(elements, "method", "send")
        assert method.visibility == "public"

    def test_no_modifier_defaults_public(self):
        elements = _parse_ts(
            'class Foo {\n'
            '  close(): void {\n'
            '    console.log("close");\n'
            '  }\n'
            '}'
        )
        method = _lookup(elements, "method", "close")
        assert method.visibility == "public"


class TestTypescriptFieldVisibility:
    """Tests for field/property visibility modifiers."""

    def test_private_field(self):
        elements = _parse_ts(
            'class Foo {\n'
            '  private url: string;\n'
            '}'
        )
        field = _lookup(elements, "variable", "url")
        assert field.visibility == "private"

    def test_protected_field(self):
        elements = _parse_ts(
            'class Foo {\n'
            '  protected cache: Map<string, any>;\n'
            '}'
        )
        field = _lookup(elements, "variable", "cache")
        assert field.visibility == "protected"

    def test_public_field(self):
        elements = _parse_ts(
            'class Foo {\n'
            '  public name: string;\n'
            '}'
        )
        field = _lookup(elements, "variable", "name")
        assert field.visibility == "public"

    def test_private_readonly_field(self):
        elements = _parse_ts(
            'class Foo {\n'
            '  private readonly url: string;\n'
            '}'
        )
        field = _lookup(elements, "variable", "url")
        assert field.visibility == "private"

    def test_protected_static_field(self):
        elements = _parse_ts(
            'class Foo {\n'
            '  protected static instance: Foo;\n'
            '}'
        )
        field = _lookup(elements, "variable", "instance")
        assert field.visibility == "protected"


class TestTypescriptArrowMethodVisibility:
    """Tests for arrow function fields with visibility (common in trpc, etc.)."""

    def test_private_arrow_method(self):
        """Arrow function as class field with private modifier.

        This pattern is very common in trpc's WsClient:
            private open = (): void => { ... }
        """
        elements = _parse_ts(
            'class WsClient {\n'
            '  private open = (): void => {\n'
            '    console.log("open");\n'
            '  };\n'
            '}'
        )
        method = _lookup(elements, "method", "open")
        assert method.visibility == "private", (
            f"Expected private arrow method 'open', got '{method.visibility}'"
        )

    def test_private_async_arrow_method(self):
        """Async arrow function with private modifier."""
        elements = _parse_ts(
            'class WsClient {\n'
            '  private reconnect = async (): Promise<void> => {\n'
            '    await this.open();\n'
            '  };\n'
            '}'
        )
        method = _lookup(elements, "method", "reconnect")
        assert method.visibility == "private", (
            f"Expected private async arrow method 'reconnect', got '{method.visibility}'"
        )

    def test_protected_arrow_method(self):
        elements = _parse_ts(
            'class Foo {\n'
            '  protected handler = (ev: Event): void => {\n'
            '    console.log(ev);\n'
            '  };\n'
            '}'
        )
        method = _lookup(elements, "method", "handler")
        assert method.visibility == "protected"


class TestTypescriptGetterSetterVisibility:
    """Tests for getter/setter with visibility modifiers."""

    def test_private_getter(self):
        elements = _parse_ts(
            'class Foo {\n'
            '  private get connected(): boolean {\n'
            '    return true;\n'
            '  }\n'
            '}'
        )
        method = _lookup(elements, "method", "connected")
        assert method.visibility == "private"

    def test_protected_setter(self):
        elements = _parse_ts(
            'class Foo {\n'
            '  protected set timeout(val: number) {\n'
            '    console.log(val);\n'
            '  }\n'
            '}'
        )
        method = _lookup(elements, "method", "timeout")
        assert method.visibility == "protected"


class TestTypescriptStaticMethodVisibility:
    """Tests for static methods with visibility modifiers."""

    def test_private_static_method(self):
        elements = _parse_ts(
            'class Foo {\n'
            '  private static create(): Foo {\n'
            '    return new Foo();\n'
            '  }\n'
            '}'
        )
        method = _lookup(elements, "method", "create")
        assert method.visibility == "private"

    def test_protected_static_method(self):
        elements = _parse_ts(
            'class Foo {\n'
            '  protected static getInstance(): Foo {\n'
            '    return new Foo();\n'
            '  }\n'
            '}'
        )
        method = _lookup(elements, "method", "getInstance")
        assert method.visibility == "protected"


class TestTypescriptTrpcWsClientPattern:
    """Test the exact pattern from trpc's WsClient that triggered this bug."""

    TRPC_WSCLIENT_CODE = (
        'class WsClient {\n'
        '  private pendingRequests: Map<string, any>;\n'
        '  private connection: WebSocket | null;\n'
        '\n'
        '  constructor(opts: WsClientOptions) {\n'
        '    this.pendingRequests = new Map();\n'
        '    this.connection = null;\n'
        '  }\n'
        '\n'
        '  private open(): void {\n'
        '    this.connection = new WebSocket(this.opts.url);\n'
        '  }\n'
        '\n'
        '  private reconnect(): void {\n'
        '    this.open();\n'
        '  }\n'
        '\n'
        '  private setupWebSocketListeners(): void {\n'
        '    console.log("setup");\n'
        '  }\n'
        '\n'
        '  private handleResponseMessage(msg: any): void {\n'
        '    console.log(msg);\n'
        '  }\n'
        '\n'
        '  private handleIncomingRequest(req: any): void {\n'
        '    console.log(req);\n'
        '  }\n'
        '\n'
        '  private send(data: string): void {\n'
        '    console.log(data);\n'
        '  }\n'
        '\n'
        '  private batchSend(): void {\n'
        '    console.log("batch");\n'
        '  }\n'
        '\n'
        '  public request(op: any): void {\n'
        '    console.log(op);\n'
        '  }\n'
        '\n'
        '  public close(): void {\n'
        '    console.log("close");\n'
        '  }\n'
        '\n'
        '  public getConnection(): WebSocket | null {\n'
        '    return this.connection;\n'
        '  }\n'
        '\n'
        '  requestAsPromise(op: any): Promise<any> {\n'
        '    return new Promise(() => {});\n'
        '  }\n'
        '}'
    )

    def test_private_methods_count(self):
        """7 of 11 methods should be private (matching the bug report)."""
        elements = _parse_ts(self.TRPC_WSCLIENT_CODE)
        methods = [e for e in elements if e.element_type == "method" and e.name != "constructor"]
        private_methods = [m for m in methods if m.visibility == "private"]
        assert len(private_methods) == 7, (
            f"Expected 7 private methods, got {len(private_methods)}. "
            f"Methods: {[(m.name, m.visibility) for m in methods]}"
        )

    def test_all_private_methods(self):
        elements = _parse_ts(self.TRPC_WSCLIENT_CODE)
        lookup = {(e.element_type, e.name): e for e in elements}

        private_methods = ["open", "reconnect", "setupWebSocketListeners",
                          "handleResponseMessage", "handleIncomingRequest",
                          "send", "batchSend"]
        for name in private_methods:
            method = lookup[("method", name)]
            assert method.visibility == "private", (
                f"Expected '{name}' to be private, got '{method.visibility}'"
            )

    def test_public_methods(self):
        elements = _parse_ts(self.TRPC_WSCLIENT_CODE)
        lookup = {(e.element_type, e.name): e for e in elements}

        public_methods = ["request", "close", "getConnection", "requestAsPromise"]
        for name in public_methods:
            method = lookup[("method", name)]
            assert method.visibility == "public", (
                f"Expected '{name}' to be public, got '{method.visibility}'"
            )

    def test_private_fields(self):
        elements = _parse_ts(self.TRPC_WSCLIENT_CODE)
        lookup = {(e.element_type, e.name): e for e in elements}

        assert lookup[("variable", "pendingRequests")].visibility == "private"
        assert lookup[("variable", "connection")].visibility == "private"


class TestTypescriptRealTrpcWsClient:
    """Test parsing the actual trpc wsClient.ts file from test_repos.

    This verifies visibility extraction works on a real-world TypeScript file,
    not just synthetic test snippets.
    """

    TRPC_FILE = Path(__file__).parent.parent.parent / "test_repos" / "trpc" / \
        "packages" / "client" / "src" / "links" / "wsLink" / "wsClient" / "wsClient.ts"

    @staticmethod
    def _parse_real_file(file_path: Path) -> list:
        """Parse a real TypeScript file."""
        parser = JavaScriptParser("typescript")
        file_info = FileInfo(
            relative_path=str(file_path.relative_to(file_path.parent.parent.parent.parent.parent.parent.parent.parent)),
            absolute_path=file_path,
            language="typescript",
        )
        code = file_path.read_text(encoding="utf-8")
        return parser.parse(code, file_info, "test-repo", "trpc", "main")

    def test_real_trpc_private_methods(self):
        """Verify actual trpc WsClient private methods are extracted correctly."""
        if not self.TRPC_FILE.exists():
            import pytest
            pytest.skip("trpc test repo not available")

        elements = self._parse_real_file(self.TRPC_FILE)
        methods = {e.name: e for e in elements if e.element_type == "method"}

        # These methods are private in the actual trpc source
        private_method_names = [
            "open", "reconnect", "setupWebSocketListeners",
            "handleResponseMessage", "handleIncomingRequest",
            "send", "batchSend",
        ]
        for name in private_method_names:
            assert name in methods, f"Method '{name}' not extracted"
            assert methods[name].visibility == "private", (
                f"Expected '{name}' to be private, got '{methods[name].visibility}'"
            )

    def test_real_trpc_public_methods(self):
        """Verify actual trpc WsClient public methods are extracted correctly."""
        if not self.TRPC_FILE.exists():
            import pytest
            pytest.skip("trpc test repo not available")

        elements = self._parse_real_file(self.TRPC_FILE)
        methods = {e.name: e for e in elements if e.element_type == "method"}

        # close and request are explicitly public
        assert methods["close"].visibility == "public"
        assert methods["request"].visibility == "public"

    def test_real_trpc_private_fields(self):
        """Verify actual trpc WsClient private fields are extracted correctly."""
        if not self.TRPC_FILE.exists():
            import pytest
            pytest.skip("trpc test repo not available")

        elements = self._parse_real_file(self.TRPC_FILE)
        fields = {e.name: e for e in elements if e.element_type == "variable"}

        # These fields are private in the actual trpc source
        private_field_names = [
            "allowReconnect", "requestManager", "activeConnection",
            "reconnectRetryDelay", "inactivityTimeout", "callbacks",
            "lazyMode", "encoder",
        ]
        for name in private_field_names:
            if name in fields:
                assert fields[name].visibility == "private", (
                    f"Expected field '{name}' to be private, got '{fields[name].visibility}'"
                )
