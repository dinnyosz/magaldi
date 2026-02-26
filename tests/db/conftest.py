"""Shared fixtures for search backend integration tests.

These tests require a running search backend instance (via Docker).
Run with: pytest -m integration tests/db/
"""

from __future__ import annotations

import os

import pytest

from magaldi_core.code_parser import CodeElement
from shared.config import MagaldiConfig, SearchBackendConfig, reset_config
from shared.db.store import INDEX_NAME

# Skip all tests in this directory if search backend is not available
pytestmark = pytest.mark.integration


@pytest.fixture(scope="function")
def config():
    """Load config for search backend connection."""
    # Reset config singleton
    reset_config()

    # Create config directly with known good values for integration tests
    return MagaldiConfig(
        search_backend=SearchBackendConfig(
            host=os.environ.get("MAGALDI_SEARCH_HOST", "localhost"),
            port=int(os.environ.get("MAGALDI_SEARCH_PORT", "9200")),
            scheme=os.environ.get("MAGALDI_SEARCH_SCHEME", "http"),
        ),
    )


@pytest.fixture
def repo(config):
    """Create repository and clean up test data."""
    from shared.db.store import Repository

    repo = Repository(config)

    # Delete test documents before test
    try:
        client = repo._get_client()
        client.delete_by_query(
            index=INDEX_NAME,
            body={"query": {"prefix": {"scope": "test-"}}},
            refresh=True,
            ignore=[404],
        )
    except Exception:
        pass

    yield repo

    # Delete test documents after test
    try:
        client = repo._get_client()
        client.delete_by_query(
            index=INDEX_NAME,
            body={"query": {"prefix": {"scope": "test-"}}},
            refresh=True,
            ignore=[404],
        )
    except Exception:
        pass

    repo.close()


@pytest.fixture
def sample_element() -> CodeElement:
    """Create a sample code element for testing."""
    return CodeElement(
        element_id="test-es:test-repo:main:src/app.py:function:process:10",
        scope="test-es",
        repository="test-repo",
        username="main",
        relative_path="src/app.py",
        element_type="function",
        name="process",
        language="python",
        line_start=10,
        line_end=25,
        raw_code="def process(data):\n    return data.upper()",
        signature="def process(data)",
        docstring="Process input data.",
        decorators=["staticmethod"],
        is_async=False,
        visibility="public",
        level=2,
        parent_id=None,
    )


@pytest.fixture
def multiple_elements() -> list[CodeElement]:
    """Create multiple code elements for search testing."""
    return [
        CodeElement(
            element_id="test-es:test-repo:main:src/utils.py:file:utils.py:1",
            scope="test-es",
            repository="test-repo",
            username="main",
            relative_path="src/utils.py",
            element_type="file",
            name="utils.py",
            language="python",
            line_start=1,
            line_end=100,
            level=0,
        ),
        CodeElement(
            element_id="test-es:test-repo:main:src/utils.py:function:calculate:10",
            scope="test-es",
            repository="test-repo",
            username="main",
            relative_path="src/utils.py",
            element_type="function",
            name="calculate",
            language="python",
            line_start=10,
            line_end=20,
            raw_code="def calculate(x, y):\n    return x + y",
            signature="def calculate(x, y)",
            docstring="Calculate the sum of two numbers.",
            level=2,
        ),
        CodeElement(
            element_id="test-es:test-repo:main:src/utils.py:function:validate:30",
            scope="test-es",
            repository="test-repo",
            username="main",
            relative_path="src/utils.py",
            element_type="function",
            name="validate",
            language="python",
            line_start=30,
            line_end=45,
            raw_code="def validate(input):\n    return bool(input)",
            signature="def validate(input)",
            docstring="Validate the input data is not empty.",
            level=2,
        ),
        CodeElement(
            element_id="test-es:test-repo:main:src/models.py:class:User:1",
            scope="test-es",
            repository="test-repo",
            username="main",
            relative_path="src/models.py",
            element_type="class",
            name="User",
            language="python",
            line_start=1,
            line_end=50,
            raw_code="class User:\n    def __init__(self, name):\n        self.name = name",
            docstring="User model for authentication.",
            level=1,
        ),
    ]
