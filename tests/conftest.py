"""Pytest configuration and shared fixtures for Magaldi tests."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Generator

import pytest

from magaldi import config as config_module
from magaldi.config import (
    ElasticsearchConfig,
    LoggingConfig,
    MagaldiConfig,
    MCPConfig,
    OllamaConfig,
    ParserConfig,
    SearchConfig,
    UserDataConfig,
    WebConfig,
    WorkerConfig,
    WorkersConfig,
)


# =============================================================================
# PATHS
# =============================================================================

TESTS_DIR = Path(__file__).parent
FIXTURES_DIR = TESTS_DIR / "fixtures"
CONFIG_FIXTURES_DIR = FIXTURES_DIR / "config"


# =============================================================================
# CONFIG FIXTURES
# =============================================================================


@pytest.fixture(autouse=True)
def reset_global_config() -> Generator[None, None, None]:
    """Automatically reset global config state before and after each test.

    This ensures tests don't leak state to each other.
    """
    config_module._config = None
    yield
    config_module._config = None


@pytest.fixture
def test_elasticsearch_config() -> ElasticsearchConfig:
    """Provide a test Elasticsearch configuration."""
    return ElasticsearchConfig(
        host="localhost",
        port=9200,
        scheme="http",
        index="magaldi_test_elements",
        timeout=10,
        retry_on_timeout=False,
        max_retries=1,
    )


@pytest.fixture
def test_ollama_config() -> OllamaConfig:
    """Provide a test Ollama configuration."""
    return OllamaConfig(
        url="http://localhost:11434",
        summarize_model="test-model",
        summarize_temperature=0.3,
        summarize_max_tokens=128,
        summarize_context_window=4096,
        embed_model="test-embed",
        embed_dimensions=1024,
        embed_context_window=4096,
    )


@pytest.fixture
def test_workers_config() -> WorkersConfig:
    """Provide a test workers configuration."""
    return WorkersConfig(
        summarization=WorkerConfig(
            count=2,
            batch_size=5,
            claim_timeout_seconds=30,
            max_retries=1,
            retry_delay_seconds=1.0,
        ),
        embedding=WorkerConfig(
            count=2,
            batch_size=10,
            claim_timeout_seconds=30,
            max_retries=1,
            retry_delay_seconds=1.0,
        ),
    )


@pytest.fixture
def test_config(
    test_elasticsearch_config: ElasticsearchConfig,
    test_ollama_config: OllamaConfig,
    test_workers_config: WorkersConfig,
) -> MagaldiConfig:
    """Provide a complete test configuration."""
    return MagaldiConfig(
        elasticsearch=test_elasticsearch_config,
        ollama=test_ollama_config,
        workers=test_workers_config,
        parser=ParserConfig(
            parallel_workers=2,
            parse_timeout_seconds=10,
        ),
        search=SearchConfig(
            default_limit=5,
            max_limit=20,
        ),
        web=WebConfig(
            host="127.0.0.1",
            port=8888,
        ),
        mcp=MCPConfig(
            server_name="magaldi-test",
            server_version="0.0.1",
        ),
        logging=LoggingConfig(
            level="DEBUG",
        ),
        user_data=UserDataConfig(
            expiration_days=7,
            cleanup_batch_size=100,
        ),
    )


# =============================================================================
# ENVIRONMENT FIXTURES
# =============================================================================


@pytest.fixture
def clean_magaldi_env() -> Generator[None, None, None]:
    """Remove all MAGALDI_ environment variables for the duration of the test."""
    original_env: dict[str, str] = {}
    magaldi_vars = [k for k in os.environ if k.startswith("MAGALDI_")]

    for var in magaldi_vars:
        original_env[var] = os.environ.pop(var)

    yield

    # Restore original environment
    os.environ.update(original_env)


# =============================================================================
# PATH FIXTURES
# =============================================================================


@pytest.fixture
def valid_config_path() -> Path:
    """Path to valid test config file."""
    return CONFIG_FIXTURES_DIR / "valid.yaml"


@pytest.fixture
def minimal_config_path() -> Path:
    """Path to minimal test config file."""
    return CONFIG_FIXTURES_DIR / "minimal.yaml"


# =============================================================================
# TEMPORARY FILE FIXTURES
# =============================================================================


@pytest.fixture
def temp_config_file(tmp_path: Path) -> Generator[Path, None, None]:
    """Create a temporary config file for testing."""
    config_file = tmp_path / "magaldi.yaml"
    config_file.write_text(
        """
elasticsearch:
  host: temphost
  port: 9200
"""
    )
    yield config_file


# =============================================================================
# MARKERS
# =============================================================================


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line(
        "markers", "integration: marks tests that require external services (ES, Ollama, etc.)"
    )
