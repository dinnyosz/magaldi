"""Tests for the configuration module (TDD - written before implementation)."""

import os
from pathlib import Path

import pytest

from magaldi.config import (
    ConfigurationError,
    ElasticsearchConfig,
    LoggingConfig,
    MagaldiConfig,
    MCPConfig,
    MySQLConfig,
    OllamaConfig,
    ParserConfig,
    SearchConfig,
    UserDataConfig,
    WebConfig,
    WorkerConfig,
    WorkersConfig,
    get_config,
    load_config,
    reset_config,
)


# =============================================================================
# FIXTURES
# =============================================================================

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "config"


@pytest.fixture(autouse=True)
def clean_config():
    """Reset config state before and after each test."""
    reset_config()
    yield
    reset_config()


@pytest.fixture
def clean_env():
    """Remove MAGALDI_ environment variables for clean testing."""
    original_env = {}
    magaldi_vars = [k for k in os.environ if k.startswith("MAGALDI_")]
    for var in magaldi_vars:
        original_env[var] = os.environ.pop(var)
    yield
    os.environ.update(original_env)


# =============================================================================
# DATACLASS DEFAULTS
# =============================================================================


class TestMySQLConfigDefaults:
    """Test MySQLConfig dataclass defaults."""

    def test_default_host(self):
        config = MySQLConfig()
        assert config.host == "localhost"

    def test_default_port(self):
        config = MySQLConfig()
        assert config.port == 3306

    def test_default_database(self):
        config = MySQLConfig()
        assert config.database == "magaldi"

    def test_default_user(self):
        config = MySQLConfig()
        assert config.user == "magaldi"

    def test_default_password_empty(self):
        config = MySQLConfig()
        assert config.password == ""

    def test_default_pool_size(self):
        config = MySQLConfig()
        assert config.pool_size == 10

    def test_default_pool_timeout(self):
        config = MySQLConfig()
        assert config.pool_timeout == 30


class TestElasticsearchConfigDefaults:
    """Test ElasticsearchConfig dataclass defaults."""

    def test_default_url(self):
        config = ElasticsearchConfig()
        assert config.url == "http://localhost:9200"

    def test_default_index(self):
        config = ElasticsearchConfig()
        assert config.index == "magaldi_code_elements"

    def test_default_timeout(self):
        config = ElasticsearchConfig()
        assert config.timeout == 30

    def test_default_retry_on_timeout(self):
        config = ElasticsearchConfig()
        assert config.retry_on_timeout is True

    def test_default_max_retries(self):
        config = ElasticsearchConfig()
        assert config.max_retries == 3


class TestOllamaConfigDefaults:
    """Test OllamaConfig dataclass defaults."""

    def test_default_url(self):
        config = OllamaConfig()
        assert config.url == "http://localhost:11434"

    def test_default_summarize_model(self):
        config = OllamaConfig()
        assert config.summarize_model == "qwen2.5-coder:7b"

    def test_default_summarize_temperature(self):
        config = OllamaConfig()
        assert config.summarize_temperature == 0.3

    def test_default_embed_model(self):
        config = OllamaConfig()
        assert config.embed_model == "snowflake-arctic-embed2"

    def test_default_embed_dimensions(self):
        config = OllamaConfig()
        assert config.embed_dimensions == 1024


class TestWorkerConfigDefaults:
    """Test WorkerConfig dataclass defaults."""

    def test_default_count(self):
        config = WorkerConfig()
        assert config.count == 4

    def test_default_batch_size(self):
        config = WorkerConfig()
        assert config.batch_size == 10

    def test_default_claim_timeout(self):
        config = WorkerConfig()
        assert config.claim_timeout_seconds == 300

    def test_default_max_retries(self):
        config = WorkerConfig()
        assert config.max_retries == 3

    def test_default_retry_delay(self):
        config = WorkerConfig()
        assert config.retry_delay_seconds == 5.0


# =============================================================================
# LOADING FROM FILE
# =============================================================================


class TestLoadFromFile:
    """Test loading configuration from YAML files."""

    def test_load_valid_config(self, clean_env):
        """Load a fully specified config file."""
        config = load_config(FIXTURES_DIR / "valid.yaml")

        assert config.mysql.host == "testhost"
        assert config.mysql.port == 3307
        assert config.mysql.password == "testpassword"
        assert config.elasticsearch.url == "http://testhost:9200"
        assert config.ollama.summarize_model == "test-model"
        assert config.workers.summarization.count == 2
        assert config.logging.level == "DEBUG"

    def test_load_minimal_config(self, clean_env):
        """Load minimal config - other values should use defaults."""
        config = load_config(FIXTURES_DIR / "minimal.yaml")

        # Specified value
        assert config.mysql.password == "minimalpassword"

        # Default values
        assert config.mysql.host == "localhost"
        assert config.mysql.port == 3306
        assert config.elasticsearch.url == "http://localhost:9200"

    def test_load_nonexistent_file_uses_defaults(self, clean_env):
        """Loading nonexistent file should use defaults (but fail validation)."""
        with pytest.raises(ConfigurationError) as exc_info:
            load_config("/nonexistent/path/config.yaml")

        # Should fail because password is required
        assert "password" in str(exc_info.value).lower()

    def test_load_caches_config(self, clean_env):
        """Loading config should cache the result."""
        config1 = load_config(FIXTURES_DIR / "valid.yaml")
        config2 = load_config(FIXTURES_DIR / "valid.yaml")
        assert config1 is config2


# =============================================================================
# ENVIRONMENT VARIABLE OVERRIDES
# =============================================================================


class TestEnvOverrides:
    """Test environment variable overrides."""

    def test_env_overrides_mysql_host(self, clean_env):
        os.environ["MAGALDI_MYSQL_HOST"] = "envhost"
        os.environ["MAGALDI_MYSQL_PASSWORD"] = "envpassword"
        config = load_config(FIXTURES_DIR / "valid.yaml")
        assert config.mysql.host == "envhost"

    def test_env_overrides_mysql_port_converts_to_int(self, clean_env):
        os.environ["MAGALDI_MYSQL_PORT"] = "3308"
        os.environ["MAGALDI_MYSQL_PASSWORD"] = "envpassword"
        config = load_config(FIXTURES_DIR / "valid.yaml")
        assert config.mysql.port == 3308
        assert isinstance(config.mysql.port, int)

    def test_env_overrides_mysql_password(self, clean_env):
        os.environ["MAGALDI_MYSQL_PASSWORD"] = "secretpassword"
        config = load_config(FIXTURES_DIR / "minimal.yaml")
        assert config.mysql.password == "secretpassword"

    def test_env_overrides_elasticsearch_url(self, clean_env):
        os.environ["MAGALDI_ELASTICSEARCH_URL"] = "http://eshost:9201"
        os.environ["MAGALDI_MYSQL_PASSWORD"] = "test"
        config = load_config(FIXTURES_DIR / "minimal.yaml")
        assert config.elasticsearch.url == "http://eshost:9201"

    def test_env_overrides_ollama_url(self, clean_env):
        os.environ["MAGALDI_OLLAMA_URL"] = "http://ollama:11435"
        os.environ["MAGALDI_MYSQL_PASSWORD"] = "test"
        config = load_config(FIXTURES_DIR / "minimal.yaml")
        assert config.ollama.url == "http://ollama:11435"

    def test_env_overrides_log_level(self, clean_env):
        os.environ["MAGALDI_LOG_LEVEL"] = "WARNING"
        os.environ["MAGALDI_MYSQL_PASSWORD"] = "test"
        config = load_config(FIXTURES_DIR / "minimal.yaml")
        assert config.logging.level == "WARNING"

    def test_env_overrides_web_port(self, clean_env):
        os.environ["MAGALDI_WEB_PORT"] = "9000"
        os.environ["MAGALDI_MYSQL_PASSWORD"] = "test"
        config = load_config(FIXTURES_DIR / "minimal.yaml")
        assert config.web.port == 9000


# =============================================================================
# PRIORITY CHAIN
# =============================================================================


class TestPriorityChain:
    """Test that env > file > defaults priority chain works."""

    def test_env_overrides_file(self, clean_env):
        """Environment should override file values."""
        os.environ["MAGALDI_MYSQL_HOST"] = "env-host"
        config = load_config(FIXTURES_DIR / "valid.yaml")

        # File says "testhost", env says "env-host"
        assert config.mysql.host == "env-host"

    def test_file_overrides_defaults(self, clean_env):
        """File should override default values."""
        config = load_config(FIXTURES_DIR / "valid.yaml")

        # Default is "localhost", file says "testhost"
        assert config.mysql.host == "testhost"

    def test_defaults_used_when_not_specified(self, clean_env):
        """Defaults should be used when not specified anywhere."""
        os.environ["MAGALDI_MYSQL_PASSWORD"] = "test"
        config = load_config(FIXTURES_DIR / "minimal.yaml")

        # Not in file, not in env, should be default
        assert config.elasticsearch.timeout == 30


# =============================================================================
# VALIDATION
# =============================================================================


class TestValidation:
    """Test configuration validation."""

    def test_missing_mysql_password_raises_error(self, clean_env):
        """Missing MySQL password should raise ConfigurationError."""
        with pytest.raises(ConfigurationError) as exc_info:
            load_config(FIXTURES_DIR / "invalid_missing_password.yaml")

        assert "password" in str(exc_info.value).lower()
        assert "required" in str(exc_info.value).lower() or "mysql" in str(exc_info.value).lower()

    def test_valid_config_passes_validation(self, clean_env):
        """Valid config should not raise."""
        config = load_config(FIXTURES_DIR / "valid.yaml")
        assert config is not None


# =============================================================================
# GET_CONFIG
# =============================================================================


class TestGetConfig:
    """Test the get_config() function."""

    def test_get_config_before_load_raises_error(self, clean_env):
        """get_config() before load_config() should raise RuntimeError."""
        with pytest.raises(RuntimeError) as exc_info:
            get_config()

        assert "not loaded" in str(exc_info.value).lower()

    def test_get_config_after_load_returns_config(self, clean_env):
        """get_config() after load_config() should return the config."""
        load_config(FIXTURES_DIR / "valid.yaml")
        config = get_config()
        assert isinstance(config, MagaldiConfig)

    def test_get_config_returns_same_instance(self, clean_env):
        """get_config() should return the same cached instance."""
        load_config(FIXTURES_DIR / "valid.yaml")
        config1 = get_config()
        config2 = get_config()
        assert config1 is config2


# =============================================================================
# RESET_CONFIG
# =============================================================================


class TestResetConfig:
    """Test the reset_config() function."""

    def test_reset_clears_cached_config(self, clean_env):
        """reset_config() should clear the cached config."""
        load_config(FIXTURES_DIR / "valid.yaml")
        reset_config()

        with pytest.raises(RuntimeError):
            get_config()

    def test_can_reload_after_reset(self, clean_env):
        """Should be able to load a different config after reset."""
        load_config(FIXTURES_DIR / "valid.yaml")
        config1 = get_config()
        assert config1.mysql.host == "testhost"

        reset_config()

        os.environ["MAGALDI_MYSQL_PASSWORD"] = "test"
        load_config(FIXTURES_DIR / "minimal.yaml")
        config2 = get_config()
        assert config2.mysql.host == "localhost"  # Default


# =============================================================================
# MAGALDI CONFIG ROOT
# =============================================================================


class TestMagaldiConfigRoot:
    """Test the root MagaldiConfig dataclass."""

    def test_has_all_sections(self, clean_env):
        """MagaldiConfig should have all expected sections."""
        config = load_config(FIXTURES_DIR / "valid.yaml")

        assert hasattr(config, "mysql")
        assert hasattr(config, "elasticsearch")
        assert hasattr(config, "ollama")
        assert hasattr(config, "workers")
        assert hasattr(config, "parser")
        assert hasattr(config, "search")
        assert hasattr(config, "web")
        assert hasattr(config, "mcp")
        assert hasattr(config, "logging")
        assert hasattr(config, "user_data")

    def test_section_types(self, clean_env):
        """Each section should be the correct type."""
        config = load_config(FIXTURES_DIR / "valid.yaml")

        assert isinstance(config.mysql, MySQLConfig)
        assert isinstance(config.elasticsearch, ElasticsearchConfig)
        assert isinstance(config.ollama, OllamaConfig)
        assert isinstance(config.workers, WorkersConfig)
        assert isinstance(config.parser, ParserConfig)
        assert isinstance(config.search, SearchConfig)
        assert isinstance(config.web, WebConfig)
        assert isinstance(config.mcp, MCPConfig)
        assert isinstance(config.logging, LoggingConfig)
        assert isinstance(config.user_data, UserDataConfig)


# =============================================================================
# NESTED CONFIG (WORKERS)
# =============================================================================


class TestWorkersConfig:
    """Test nested workers configuration."""

    def test_workers_has_summarization(self, clean_env):
        config = load_config(FIXTURES_DIR / "valid.yaml")
        assert isinstance(config.workers.summarization, WorkerConfig)

    def test_workers_has_embedding(self, clean_env):
        config = load_config(FIXTURES_DIR / "valid.yaml")
        assert isinstance(config.workers.embedding, WorkerConfig)

    def test_workers_summarization_values(self, clean_env):
        config = load_config(FIXTURES_DIR / "valid.yaml")
        assert config.workers.summarization.count == 2
        assert config.workers.summarization.batch_size == 5

    def test_workers_embedding_values(self, clean_env):
        config = load_config(FIXTURES_DIR / "valid.yaml")
        assert config.workers.embedding.count == 2
        assert config.workers.embedding.batch_size == 10
