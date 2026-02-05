"""Shared fixtures for parser tests."""

from pathlib import Path

import pytest

from magaldi_core.change_detection import FileInfo


@pytest.fixture
def python_code() -> str:
    return '''"""Module docstring."""

CONSTANT = 42
MAX_SIZE = 100

def standalone_function(x: int, y: int = 0) -> int:
    """Add two numbers."""
    return x + y

async def async_function():
    """Async example."""
    pass

class MyClass:
    """A sample class."""

    def __init__(self, name: str):
        """Initialize."""
        self.name = name

    def greet(self) -> str:
        """Return greeting."""
        return f"Hello, {self.name}!"

    @staticmethod
    def static_method():
        """Static method."""
        pass

    def _private_method(self):
        """Private method."""
        pass

class _PrivateClass:
    """A private class."""
    pass
'''


@pytest.fixture
def javascript_code() -> str:
    return '''
function regularFunction(a, b) {
    return a + b;
}

async function asyncFunction() {
    await something();
}

const arrowFunc = (x) => x * 2;

class MyClass {
    constructor(name) {
        this.name = name;
    }

    greet() {
        return `Hello, ${this.name}!`;
    }
}

export class ExportedClass {
    doSomething() {}
}
'''


@pytest.fixture
def temp_python_file(tmp_path: Path, python_code: str) -> FileInfo:
    file = tmp_path / "test_module.py"
    file.write_text(python_code)
    return FileInfo(
        relative_path="test_module.py",
        absolute_path=file,
        language="python",
        hash="abc123",
    )


@pytest.fixture
def temp_js_file(tmp_path: Path, javascript_code: str) -> FileInfo:
    file = tmp_path / "test_module.js"
    file.write_text(javascript_code)
    return FileInfo(
        relative_path="test_module.js",
        absolute_path=file,
        language="javascript",
        hash="def456",
    )
