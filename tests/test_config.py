"""Tests for the configuration module (TDD - written before implementation)."""

import os
from pathlib import Path

import pytest

from shared.config import (
    ConfigurationError,
    ElasticsearchConfig,
    LLMConfig,
    LoggingConfig,
    MagaldiConfig,
    MCPConfig,
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
    # Skip loading .env file during tests
    os.environ["MAGALDI_SKIP_DOTENV"] = "1"
    yield
    # Remove any MAGALDI_ vars that were set during the test
    new_vars = [k for k in os.environ if k.startswith("MAGALDI_")]
    for var in new_vars:
        os.environ.pop(var)
    # Restore original vars
    os.environ.update(original_env)


# =============================================================================
# DATACLASS DEFAULTS
# =============================================================================


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


class TestLLMConfigDefaults:
    """Test LLMConfig dataclass defaults."""

    def test_default_summarize_model_reference(self):
        """Test that summarize_model is a reference key to named model."""
        config = LLMConfig()
        assert config.summarize_model == "qwen3-4b"

    def test_default_summarize_model_name(self):
        """Test that the referenced summarize model has correct name."""
        config = LLMConfig()
        model = config.get_summarize_model()
        assert model.name == "qwen3:4b-instruct"

    def test_default_summarize_temperature(self):
        config = LLMConfig()
        assert config.summarize_temperature == 0.2

    def test_default_embed_model_reference(self):
        """Test that embed_model is a reference key to named model."""
        config = LLMConfig()
        assert config.embed_model == "arctic-embed"

    def test_default_embed_model_name(self):
        """Test that the referenced embed model has correct name."""
        config = LLMConfig()
        model = config.get_embed_model()
        assert model.name == "snowflake-arctic-embed2"

    def test_default_embed_dimensions(self):
        """Test that embed model has correct dimensions."""
        config = LLMConfig()
        model = config.get_embed_model()
        assert model.dimensions == 1024

    def test_aggregation_context_size_default(self):
        """Should have default aggregation context size."""
        config = LLMConfig()
        assert config.aggregation_context_size == 16384


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

        assert config.elasticsearch.host == "testhost"
        assert config.elasticsearch.port == 9200
        assert config.elasticsearch.url == "http://testhost:9200"
        # summarize_model is now a reference key
        assert config.llm.summarize_model == "test-model"
        # The actual model name is accessed via get_summarize_model()
        assert config.llm.get_summarize_model().name == "test-model-name"
        assert config.workers.summarization.count == 2
        assert config.logging.level == "DEBUG"

    def test_load_minimal_config(self, clean_env):
        """Load minimal config - other values should use defaults."""
        config = load_config(FIXTURES_DIR / "minimal.yaml")

        # Specified value
        assert config.elasticsearch.host == "localhost"

        # Default values
        assert config.elasticsearch.port == 9200
        assert config.elasticsearch.url == "http://localhost:9200"

    def test_load_nonexistent_file_uses_defaults(self, clean_env):
        """Loading nonexistent file should use defaults."""
        config = load_config("/nonexistent/path/config.yaml")

        # Should use default values
        assert config.elasticsearch.host == "localhost"
        assert config.redis.host == "localhost"

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

    def test_env_overrides_elasticsearch_host_port(self, clean_env):
        os.environ["MAGALDI_ELASTICSEARCH_HOST"] = "eshost"
        os.environ["MAGALDI_ELASTICSEARCH_PORT"] = "9201"
        os.environ["MAGALDI_ELASTICSEARCH_SCHEME"] = "https"
        config = load_config(FIXTURES_DIR / "minimal.yaml")
        assert config.elasticsearch.host == "eshost"
        assert config.elasticsearch.port == 9201
        assert config.elasticsearch.scheme == "https"
        assert config.elasticsearch.url == "https://eshost:9201"

    def test_env_overrides_llm_model_reference(self, clean_env):
        """Test environment overrides for LLM model references."""
        # First ensure the model exists in the config
        config = load_config(FIXTURES_DIR / "minimal.yaml")
        # The env override changes the model reference key, not the model name
        # Note: This requires the referenced model to exist in config.llm.models
        assert config.llm.summarize_model == "qwen3-4b"  # Default from dataclass

    def test_env_overrides_log_level(self, clean_env):
        os.environ["MAGALDI_LOG_LEVEL"] = "WARNING"
        config = load_config(FIXTURES_DIR / "minimal.yaml")
        assert config.logging.level == "WARNING"

    def test_env_overrides_web_port(self, clean_env):
        os.environ["MAGALDI_WEB_PORT"] = "9000"
        config = load_config(FIXTURES_DIR / "minimal.yaml")
        assert config.web.port == 9000


# =============================================================================
# PRIORITY CHAIN
# =============================================================================


class TestPriorityChain:
    """Test that env > file > defaults priority chain works."""

    def test_env_overrides_file(self, clean_env):
        """Environment should override file values."""
        os.environ["MAGALDI_ELASTICSEARCH_HOST"] = "env-host"
        config = load_config(FIXTURES_DIR / "valid.yaml")

        # File says "testhost", env says "env-host"
        assert config.elasticsearch.host == "env-host"

    def test_file_overrides_defaults(self, clean_env):
        """File should override default values."""
        config = load_config(FIXTURES_DIR / "valid.yaml")

        # Default is "localhost", file says "testhost"
        assert config.elasticsearch.host == "testhost"

    def test_defaults_used_when_not_specified(self, clean_env):
        """Defaults should be used when not specified anywhere."""
        config = load_config(FIXTURES_DIR / "minimal.yaml")

        # Not in file, not in env, should be default
        assert config.elasticsearch.timeout == 30


# =============================================================================
# VALIDATION
# =============================================================================


class TestValidation:
    """Test configuration validation."""

    def test_invalid_embed_dimensions_raises_error(self, clean_env):
        """Invalid embed dimensions for snowflake model should raise ConfigurationError."""
        from shared.config import _validate_config, ModelConfig

        # Load config and manually set invalid dimensions on the embed model
        config = load_config(FIXTURES_DIR / "valid.yaml", skip_validation=True)
        # Replace the embed model with one that has snowflake name but wrong dimensions
        config.llm.models["test-embed"] = ModelConfig(
            name="snowflake-arctic-embed2",
            provider="ollama",
            url="http://localhost:11434",
            dimensions=512,  # Wrong for snowflake-arctic-embed2
        )

        with pytest.raises(ConfigurationError) as exc_info:
            _validate_config(config)

        assert "embed_dimensions" in str(exc_info.value).lower()

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
        assert config1.elasticsearch.host == "testhost"

        reset_config()

        load_config(FIXTURES_DIR / "minimal.yaml")
        config2 = get_config()
        assert config2.elasticsearch.host == "localhost"  # Default


# =============================================================================
# MAGALDI CONFIG ROOT
# =============================================================================


class TestMagaldiConfigRoot:
    """Test the root MagaldiConfig dataclass."""

    def test_has_all_sections(self, clean_env):
        """MagaldiConfig should have all expected sections."""
        config = load_config(FIXTURES_DIR / "valid.yaml")

        assert hasattr(config, "elasticsearch")
        assert hasattr(config, "redis")
        assert hasattr(config, "llm")
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

        assert isinstance(config.elasticsearch, ElasticsearchConfig)
        assert isinstance(config.llm, LLMConfig)
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
