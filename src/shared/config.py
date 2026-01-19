"""Central configuration management for Magaldi.

This module provides a single source of truth for all configuration values.
Configuration is loaded with the following priority (highest to lowest):
1. Environment variables (MAGALDI_*)
2. Configuration file (magaldi.yaml)
3. Dataclass defaults

Usage:
    from shared.config import load_config, get_config

    # At application startup
    load_config()  # Or load_config("/path/to/config.yaml")

    # In any module
    config = get_config()
    print(config.elasticsearch.host)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


class ConfigurationError(Exception):
    """Raised when configuration is invalid or incomplete."""

    pass


# =============================================================================
# CONFIGURATION DATACLASSES
# =============================================================================


@dataclass
class ElasticsearchConfig:
    """Elasticsearch configuration."""

    host: str = "localhost"
    port: int = 9200
    scheme: str = "http"
    index: str = "magaldi_code_elements"
    timeout: int = 30
    retry_on_timeout: bool = True
    max_retries: int = 3

    @property
    def url(self) -> str:
        """Build URL from components."""
        return f"{self.scheme}://{self.host}:{self.port}"


@dataclass
class RedisConfig:
    """Redis configuration."""

    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: str | None = None


@dataclass
class LLMConfig:
    """LLM provider configuration.

    Supports multiple providers through LiteLLM:
    - "ollama": Local Ollama server
    - "openai": OpenAI API
    - "anthropic": Anthropic API
    - And many more (see LiteLLM docs)

    Model format depends on provider:
    - ollama: Just the model name (e.g., "qwen2.5-coder:3b")
    - openai: Model name (e.g., "gpt-4o-mini")
    - anthropic: Model name (e.g., "claude-3-haiku-20240307")
    """

    # Provider selection
    provider: str = "ollama"  # ollama, openai, anthropic, etc.

    # API configuration
    url: str = "http://localhost:11434"  # For Ollama
    api_key: str | None = None  # For cloud providers (or set via env vars)

    # Summarization model (for files, classes, features)
    summarize_model: str = "qwen2.5-coder:3b"
    # Smaller model for functions, methods, variables, constants
    summarize_model_small: str = "qwen2.5-coder:1.5b"
    summarize_temperature: float = 0.3
    summarize_max_tokens: int = 256
    summarize_context_window: int = 8192

    # Embedding model
    embed_model: str = "snowflake-arctic-embed2"
    embed_dimensions: int = 1024
    embed_context_window: int = 8192

    # Embedding provider (can be different from summarization)
    embed_provider: str | None = None  # If None, uses same as provider
    embed_api_key: str | None = None  # If None, uses same as api_key

    def get_model_for_element_type(self, element_type: str) -> str:
        """Get the appropriate model for an element type.

        Uses small model for functions, methods, variables, constants.
        Uses main model for files, classes.
        """
        if element_type in ("function", "method", "variable", "constant"):
            return self.summarize_model_small
        return self.summarize_model

    def get_litellm_model(self, model_name: str) -> str:
        """Get the full LiteLLM model identifier.

        Args:
            model_name: The model name (e.g., "qwen2.5-coder:3b")

        Returns:
            Full model identifier for LiteLLM (e.g., "ollama/qwen2.5-coder:3b")
        """
        if self.provider == "ollama":
            return f"ollama/{model_name}"
        elif self.provider == "openai":
            return model_name  # OpenAI models don't need prefix
        else:
            return f"{self.provider}/{model_name}"

    def get_embed_litellm_model(self) -> str:
        """Get the full LiteLLM embedding model identifier."""
        provider = self.embed_provider or self.provider
        if provider == "ollama":
            return f"ollama/{self.embed_model}"
        elif provider == "openai":
            return self.embed_model
        else:
            return f"{provider}/{self.embed_model}"




@dataclass
class WorkerConfig:
    """Configuration for a worker pool."""

    count: int = 4
    batch_size: int = 10
    claim_timeout_seconds: int = 300
    max_retries: int = 3
    retry_delay_seconds: float = 5.0


@dataclass
class WorkersConfig:
    """Configuration for all worker pools."""

    summarization: WorkerConfig = field(default_factory=WorkerConfig)
    embedding: WorkerConfig = field(
        default_factory=lambda: WorkerConfig(batch_size=20, retry_delay_seconds=2.0)
    )


@dataclass
class ParserConfig:
    """Parser configuration."""

    parallel_workers: int = 4
    parse_timeout_seconds: int = 30
    hash_algorithm: str = "sha256"
    hash_workers: int = 4
    hash_chunk_size: int = 65536
    max_file_size_bytes: int = 10485760  # 10 MB
    encoding_fallback: str = "latin-1"


@dataclass
class SearchConfig:
    """Search configuration."""

    default_limit: int = 10
    max_limit: int = 50
    query_cache_size: int = 1000
    query_cache_ttl_seconds: int = 300


@dataclass
class WebConfig:
    """Web server configuration."""

    host: str = "0.0.0.0"
    port: int = 8080
    cors_origins: list[str] = field(default_factory=lambda: ["http://localhost:3000"])
    stats_cache_ttl_seconds: int = 30
    file_tree_cache_ttl_seconds: int = 300


@dataclass
class MCPConfig:
    """MCP server configuration."""

    server_name: str = "magaldi"
    server_version: str = "1.0.0"


@dataclass
class LoggingConfig:
    """Logging configuration."""

    level: str = "INFO"
    format: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    file: str | None = None


@dataclass
class UserDataConfig:
    """User data management configuration."""

    expiration_days: int = 30
    cleanup_batch_size: int = 1000


@dataclass
class BenchmarkConfig:
    """Benchmark configuration for model comparison.

    Model selection based on arxiv.org/html/2507.03160v2:
    - Qwen2.5-Coder family: best stability and performance/efficiency ratio
    - 3B models offer sweet spot (59% pass@1, ~11GB VRAM)
    - 10% performance gain typically requires 4x VRAM increase
    """

    # Models to benchmark - ordered by size within each category
    # Reference: "Assessing Small Language Models for Code Generation" (2025)
    models: list[str] = field(default_factory=lambda: [
        # Tier 1: Ultra-light (<1.5B) - ~6-8GB VRAM
        "qwen2.5-coder:0.5b",
        "qwen2.5-coder:1.5b",      # 54% pass@1, best efficiency
        "opencoder:1.5b",          # Full transparency model
        "llama3.2:1b",
        # Tier 2: Light (1.5B-3B) - ~10-12GB VRAM, best efficiency/performance
        "qwen2.5-coder:3b",        # 59% pass@1, sweet spot
        "llama3.2:3b",
        # Tier 3: Medium (6B-9B) - ~14-17GB VRAM
        "qwen2.5-coder:7b",        # 65% pass@1, most stable (1.00)
        "opencoder:8b",            # Comparable to top performers
        # MoE models - active params listed
        "granite3.1-moe:1b",       # ~1B active (IBM, 128K context)
        "granite3.1-moe:3b",       # ~3B active (IBM, 128K context)
        "deepseek-coder-v2:lite",  # 2.4B active from 16B total
    ])

    # Model used for evaluating/rating summaries (LLM-as-judge)
    # Qwen2.5-Coder 7B chosen for highest stability score (1.00)
    eval_model: str = "qwen2.5-coder:7b"

    # Ollama API URL (defaults to same as llm.url)
    ollama_url: str | None = None

    # Generation settings (based on paper's methodology)
    temperature: float = 0.2      # Paper used 0.2, top_p=0.95
    max_tokens: int = 256
    timeout: int = 120


@dataclass
class MagaldiConfig:
    """Root configuration object containing all sections."""

    elasticsearch: ElasticsearchConfig = field(default_factory=ElasticsearchConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    workers: WorkersConfig = field(default_factory=WorkersConfig)
    parser: ParserConfig = field(default_factory=ParserConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    web: WebConfig = field(default_factory=WebConfig)
    mcp: MCPConfig = field(default_factory=MCPConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    user_data: UserDataConfig = field(default_factory=UserDataConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)


# =============================================================================
# GLOBAL STATE
# =============================================================================

_config: MagaldiConfig | None = None


# =============================================================================
# PUBLIC API
# =============================================================================


def load_config(
    config_path: str | Path | None = None,
    skip_validation: bool = False,
) -> MagaldiConfig:
    """Load configuration from file and environment variables.

    Configuration priority (highest to lowest):
    1. Environment variables (MAGALDI_*)
    2. .env file
    3. Configuration file
    4. Dataclass defaults

    Args:
        config_path: Path to configuration file. If None, searches standard locations.
        skip_validation: If True, skip validation (useful for dry-run mode).

    Returns:
        Loaded and validated MagaldiConfig instance.

    Raises:
        ConfigurationError: If configuration is invalid.
    """
    global _config

    if _config is not None:
        return _config

    # Load .env file if present (doesn't override existing env vars)
    # Skip in tests by setting MAGALDI_SKIP_DOTENV=1
    if not os.environ.get("MAGALDI_SKIP_DOTENV"):
        load_dotenv()

    # Start with defaults
    config = MagaldiConfig()

    # Find and load config file
    if config_path is not None:
        config_path = Path(config_path)
    else:
        config_path = _find_config_file()

    if config_path and config_path.exists():
        config = _load_from_file(config, config_path)

    # Apply environment variable overrides
    config = _apply_env_overrides(config)

    # Validate (unless skipped)
    if not skip_validation:
        _validate_config(config)

    _config = config
    return config


def get_config() -> MagaldiConfig:
    """Get the loaded configuration.

    Must call load_config() first.

    Returns:
        The cached MagaldiConfig instance.

    Raises:
        RuntimeError: If load_config() has not been called.
    """
    if _config is None:
        raise RuntimeError(
            "Configuration not loaded. Call load_config() first at application startup."
        )
    return _config


def reset_config() -> None:
    """Reset the cached configuration.

    Primarily used for testing to ensure a clean state between tests.
    """
    global _config
    _config = None


# =============================================================================
# PRIVATE HELPERS
# =============================================================================


def _find_config_file() -> Path | None:
    """Find configuration file in standard locations.

    Search order:
    1. MAGALDI_CONFIG environment variable
    2. ~/.magaldi/config.yaml
    3. /etc/magaldi/config.yaml
    4. ./config/magaldi.yaml
    5. ./magaldi.yaml

    Returns:
        Path to config file if found, None otherwise.
    """
    locations: list[Path | None] = [
        Path(os.environ["MAGALDI_CONFIG"]) if "MAGALDI_CONFIG" in os.environ else None,
        Path.home() / ".magaldi" / "config.yaml",
        Path("/etc/magaldi/config.yaml"),
        Path("./config/magaldi.yaml"),
        Path("./magaldi.yaml"),
    ]

    for loc in locations:
        if loc and loc.exists():
            return loc

    return None


def _load_from_file(config: MagaldiConfig, config_path: Path) -> MagaldiConfig:
    """Load configuration from YAML file.

    Args:
        config: Base config with defaults.
        config_path: Path to YAML file.

    Returns:
        Config with file values merged in.
    """
    with open(config_path, encoding="utf-8") as f:
        file_config = yaml.safe_load(f) or {}

    return _merge_config(config, file_config)


def _merge_config(config: MagaldiConfig, file_config: dict[str, Any]) -> MagaldiConfig:
    """Merge file configuration into dataclass config.

    Args:
        config: Base config with defaults.
        file_config: Dictionary from YAML file.

    Returns:
        Config with file values merged in.
    """
    section_mapping = {
        "elasticsearch": config.elasticsearch,
        "redis": config.redis,
        "llm": config.llm,
        "parser": config.parser,
        "search": config.search,
        "web": config.web,
        "mcp": config.mcp,
        "logging": config.logging,
        "user_data": config.user_data,
    }

    for section_name, section_obj in section_mapping.items():
        if section_name in file_config:
            for key, value in file_config[section_name].items():
                if hasattr(section_obj, key):
                    setattr(section_obj, key, value)

    # Handle nested workers config
    if "workers" in file_config:
        workers_config = file_config["workers"]
        if "summarization" in workers_config:
            for key, value in workers_config["summarization"].items():
                if hasattr(config.workers.summarization, key):
                    setattr(config.workers.summarization, key, value)
        if "embedding" in workers_config:
            for key, value in workers_config["embedding"].items():
                if hasattr(config.workers.embedding, key):
                    setattr(config.workers.embedding, key, value)

    return config


def _apply_env_overrides(config: MagaldiConfig) -> MagaldiConfig:
    """Apply environment variable overrides.

    Environment variables follow the pattern: MAGALDI_{SECTION}_{KEY}
    Type conversion is handled automatically.

    Args:
        config: Config to apply overrides to.

    Returns:
        Config with environment overrides applied.
    """
    # Define mappings: env_var -> (section_attr, key, converter)
    env_mappings: dict[str, tuple[str, str] | tuple[str, str, type]] = {
        # Elasticsearch
        "MAGALDI_ELASTICSEARCH_HOST": ("elasticsearch", "host"),
        "MAGALDI_ELASTICSEARCH_PORT": ("elasticsearch", "port", int),
        "MAGALDI_ELASTICSEARCH_SCHEME": ("elasticsearch", "scheme"),
        "MAGALDI_ELASTICSEARCH_INDEX": ("elasticsearch", "index"),
        "MAGALDI_ELASTICSEARCH_TIMEOUT": ("elasticsearch", "timeout", int),
        # Redis
        "MAGALDI_REDIS_HOST": ("redis", "host"),
        "MAGALDI_REDIS_PORT": ("redis", "port", int),
        "MAGALDI_REDIS_DB": ("redis", "db", int),
        "MAGALDI_REDIS_PASSWORD": ("redis", "password"),
        # LLM configuration
        "MAGALDI_LLM_PROVIDER": ("llm", "provider"),
        "MAGALDI_LLM_URL": ("llm", "url"),
        "MAGALDI_LLM_API_KEY": ("llm", "api_key"),
        "MAGALDI_LLM_SUMMARIZE_MODEL": ("llm", "summarize_model"),
        "MAGALDI_LLM_EMBED_MODEL": ("llm", "embed_model"),
        # Logging
        "MAGALDI_LOG_LEVEL": ("logging", "level"),
        # Web
        "MAGALDI_WEB_HOST": ("web", "host"),
        "MAGALDI_WEB_PORT": ("web", "port", int),
    }

    for env_var, mapping in env_mappings.items():
        value = os.environ.get(env_var)
        if value is not None:
            section_name = mapping[0]
            attr_name = mapping[1]
            converter = mapping[2] if len(mapping) > 2 else str

            section = getattr(config, section_name)
            setattr(section, attr_name, converter(value))

    return config


def _validate_config(config: MagaldiConfig) -> None:
    """Validate configuration values.

    Args:
        config: Config to validate.

    Raises:
        ConfigurationError: If validation fails.
    """
    errors: list[str] = []

    # Consistency checks
    if (
        config.llm.embed_dimensions != 1024
        and config.llm.embed_model == "snowflake-arctic-embed2"
    ):
        errors.append(
            f"snowflake-arctic-embed2 requires embed_dimensions=1024, "
            f"got {config.llm.embed_dimensions}"
        )

    # Port range checks
    if not (1 <= config.elasticsearch.port <= 65535):
        errors.append(f"Elasticsearch port must be between 1 and 65535, got {config.elasticsearch.port}")

    if not (1 <= config.redis.port <= 65535):
        errors.append(f"Redis port must be between 1 and 65535, got {config.redis.port}")

    if not (1 <= config.web.port <= 65535):
        errors.append(f"Web port must be between 1 and 65535, got {config.web.port}")

    if errors:
        raise ConfigurationError("\n".join(errors))
