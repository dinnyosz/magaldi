# Magaldi Configuration - Phase 0: Central Configuration

## Overview

All Magaldi components read from a single configuration source. No hardcoded values or class-level defaults - everything flows from this central config.

```
┌─────────────────────────────────────────────────────────────────┐
│                    CONFIGURATION HIERARCHY                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Environment Variables     (highest priority)                │
│     MAGALDI_MYSQL_PASSWORD=secret                               │
│                     ↓                                           │
│  2. Config File                                                 │
│     /etc/magaldi/config.yaml or ~/.magaldi/config.yaml          │
│                     ↓                                           │
│  3. Defaults                  (lowest priority)                 │
│     Built into ConfigSchema                                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Configuration Schema

### Complete Config File

```yaml
# /etc/magaldi/config.yaml
# All values shown are defaults unless marked [REQUIRED]

# =============================================================================
# DATABASE CONNECTIONS
# =============================================================================

mysql:
  host: localhost
  port: 3306
  database: magaldi
  user: magaldi
  password: ${MAGALDI_MYSQL_PASSWORD}  # [REQUIRED] - use env var
  pool_size: 10
  pool_timeout: 30

elasticsearch:
  url: http://localhost:9200
  index: magaldi_code_elements
  timeout: 30
  retry_on_timeout: true
  max_retries: 3

# =============================================================================
# OLLAMA (AI MODELS)
# =============================================================================

ollama:
  url: http://localhost:11434

  # Summarization model
  summarize_model: qwen2.5-coder:7b
  summarize_temperature: 0.3
  summarize_max_tokens: 256
  summarize_context_window: 8192

  # Embedding model
  embed_model: snowflake-arctic-embed2
  embed_dimensions: 1024
  embed_context_window: 8192

# =============================================================================
# WORKER POOLS
# =============================================================================

workers:
  summarization:
    count: 4
    batch_size: 10
    claim_timeout_seconds: 300
    max_retries: 3
    retry_delay_seconds: 5.0

  embedding:
    count: 4
    batch_size: 20
    claim_timeout_seconds: 300
    max_retries: 3
    retry_delay_seconds: 2.0

# =============================================================================
# PARSER SETTINGS
# =============================================================================

parser:
  # Parallel file parsing
  parallel_workers: 4
  parse_timeout_seconds: 30

  # Hash computation
  hash_algorithm: sha256
  hash_workers: 4
  hash_chunk_size: 65536

  # File handling
  max_file_size_bytes: 10485760  # 10 MB
  encoding_fallback: latin-1

# =============================================================================
# SEARCH SETTINGS
# =============================================================================

search:
  default_limit: 10
  max_limit: 50
  query_cache_size: 1000
  query_cache_ttl_seconds: 300

# =============================================================================
# WEB UI
# =============================================================================

web:
  host: 0.0.0.0
  port: 8080
  cors_origins:
    - http://localhost:3000
  stats_cache_ttl_seconds: 30
  file_tree_cache_ttl_seconds: 300

# =============================================================================
# MCP SERVER
# =============================================================================

mcp:
  server_name: magaldi
  server_version: "1.0.0"

# =============================================================================
# LOGGING
# =============================================================================

logging:
  level: INFO  # DEBUG, INFO, WARNING, ERROR
  format: "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
  file: null  # Optional: /var/log/magaldi/magaldi.log

# =============================================================================
# USER DATA MANAGEMENT
# =============================================================================

user_data:
  expiration_days: 30
  cleanup_batch_size: 1000
```

---

## Configuration Loader

### Python Implementation

```python
# src/magaldi/config.py
"""Central configuration management for Magaldi."""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List
import yaml


@dataclass
class MySQLConfig:
    host: str = "localhost"
    port: int = 3306
    database: str = "magaldi"
    user: str = "magaldi"
    password: str = ""  # Required via env
    pool_size: int = 10
    pool_timeout: int = 30


@dataclass
class ElasticsearchConfig:
    url: str = "http://localhost:9200"
    index: str = "magaldi_code_elements"
    timeout: int = 30
    retry_on_timeout: bool = True
    max_retries: int = 3


@dataclass
class OllamaConfig:
    url: str = "http://localhost:11434"

    # Summarization
    summarize_model: str = "qwen2.5-coder:7b"
    summarize_temperature: float = 0.3
    summarize_max_tokens: int = 256
    summarize_context_window: int = 8192

    # Embedding
    embed_model: str = "snowflake-arctic-embed2"
    embed_dimensions: int = 1024
    embed_context_window: int = 8192


@dataclass
class WorkerConfig:
    count: int = 4
    batch_size: int = 10
    claim_timeout_seconds: int = 300
    max_retries: int = 3
    retry_delay_seconds: float = 5.0


@dataclass
class WorkersConfig:
    summarization: WorkerConfig = field(default_factory=lambda: WorkerConfig(batch_size=10))
    embedding: WorkerConfig = field(default_factory=lambda: WorkerConfig(batch_size=20, retry_delay_seconds=2.0))


@dataclass
class ParserConfig:
    parallel_workers: int = 4
    parse_timeout_seconds: int = 30
    hash_algorithm: str = "sha256"
    hash_workers: int = 4
    hash_chunk_size: int = 65536
    max_file_size_bytes: int = 10485760
    encoding_fallback: str = "latin-1"


@dataclass
class SearchConfig:
    default_limit: int = 10
    max_limit: int = 50
    query_cache_size: int = 1000
    query_cache_ttl_seconds: int = 300


@dataclass
class WebConfig:
    host: str = "0.0.0.0"
    port: int = 8080
    cors_origins: List[str] = field(default_factory=lambda: ["http://localhost:3000"])
    stats_cache_ttl_seconds: int = 30
    file_tree_cache_ttl_seconds: int = 300


@dataclass
class MCPConfig:
    server_name: str = "magaldi"
    server_version: str = "1.0.0"


@dataclass
class LoggingConfig:
    level: str = "INFO"
    format: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    file: Optional[str] = None


@dataclass
class UserDataConfig:
    expiration_days: int = 30
    cleanup_batch_size: int = 1000


@dataclass
class MagaldiConfig:
    """Root configuration object."""

    mysql: MySQLConfig = field(default_factory=MySQLConfig)
    elasticsearch: ElasticsearchConfig = field(default_factory=ElasticsearchConfig)
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    workers: WorkersConfig = field(default_factory=WorkersConfig)
    parser: ParserConfig = field(default_factory=ParserConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    web: WebConfig = field(default_factory=WebConfig)
    mcp: MCPConfig = field(default_factory=MCPConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    user_data: UserDataConfig = field(default_factory=UserDataConfig)


# Global config instance
_config: Optional[MagaldiConfig] = None


def load_config(config_path: Optional[str] = None) -> MagaldiConfig:
    """
    Load configuration from file and environment variables.

    Priority: Environment > Config File > Defaults
    """
    global _config

    if _config is not None:
        return _config

    # Find config file
    if config_path is None:
        config_path = _find_config_file()

    # Start with defaults
    config = MagaldiConfig()

    # Load from file if exists
    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            file_config = yaml.safe_load(f)
            config = _merge_config(config, file_config)

    # Override with environment variables
    config = _apply_env_overrides(config)

    # Validate required values
    _validate_config(config)

    _config = config
    return config


def _find_config_file() -> Optional[str]:
    """Find config file in standard locations."""

    locations = [
        os.environ.get("MAGALDI_CONFIG"),
        Path.home() / ".magaldi" / "config.yaml",
        Path("/etc/magaldi/config.yaml"),
        Path("./config.yaml"),
    ]

    for loc in locations:
        if loc and Path(loc).exists():
            return str(loc)

    return None


def _merge_config(config: MagaldiConfig, file_config: dict) -> MagaldiConfig:
    """Merge file config into dataclass config."""

    if "mysql" in file_config:
        for key, value in file_config["mysql"].items():
            if hasattr(config.mysql, key):
                setattr(config.mysql, key, value)

    if "elasticsearch" in file_config:
        for key, value in file_config["elasticsearch"].items():
            if hasattr(config.elasticsearch, key):
                setattr(config.elasticsearch, key, value)

    if "ollama" in file_config:
        for key, value in file_config["ollama"].items():
            if hasattr(config.ollama, key):
                setattr(config.ollama, key, value)

    # ... similar for other sections

    return config


def _apply_env_overrides(config: MagaldiConfig) -> MagaldiConfig:
    """Apply environment variable overrides."""

    env_mappings = {
        "MAGALDI_MYSQL_HOST": ("mysql", "host"),
        "MAGALDI_MYSQL_PORT": ("mysql", "port", int),
        "MAGALDI_MYSQL_DATABASE": ("mysql", "database"),
        "MAGALDI_MYSQL_USER": ("mysql", "user"),
        "MAGALDI_MYSQL_PASSWORD": ("mysql", "password"),

        "MAGALDI_ELASTICSEARCH_URL": ("elasticsearch", "url"),
        "MAGALDI_ELASTICSEARCH_INDEX": ("elasticsearch", "index"),

        "MAGALDI_OLLAMA_URL": ("ollama", "url"),
        "MAGALDI_OLLAMA_SUMMARIZE_MODEL": ("ollama", "summarize_model"),
        "MAGALDI_OLLAMA_EMBED_MODEL": ("ollama", "embed_model"),

        "MAGALDI_LOG_LEVEL": ("logging", "level"),

        "MAGALDI_WEB_PORT": ("web", "port", int),
    }

    for env_var, mapping in env_mappings.items():
        value = os.environ.get(env_var)
        if value is not None:
            section = getattr(config, mapping[0])
            attr = mapping[1]
            converter = mapping[2] if len(mapping) > 2 else str
            setattr(section, attr, converter(value))

    return config


def _validate_config(config: MagaldiConfig):
    """Validate required configuration values."""

    errors = []

    if not config.mysql.password:
        errors.append("MySQL password is required (set MAGALDI_MYSQL_PASSWORD)")

    if config.ollama.embed_dimensions != 1024 and config.ollama.embed_model == "snowflake-arctic-embed2":
        errors.append("snowflake-arctic-embed2 requires embed_dimensions=1024")

    if errors:
        raise ConfigurationError("\n".join(errors))


class ConfigurationError(Exception):
    """Raised when configuration is invalid."""
    pass


def get_config() -> MagaldiConfig:
    """Get the loaded configuration (must call load_config first)."""
    if _config is None:
        raise RuntimeError("Configuration not loaded. Call load_config() first.")
    return _config
```

---

## Usage in Components

### All Components Use Central Config

```python
# src/magaldi/db/mysql.py
from magaldi.config import get_config

class MySQLConnection:
    def __init__(self):
        config = get_config()
        self.host = config.mysql.host
        self.port = config.mysql.port
        self.database = config.mysql.database
        self.user = config.mysql.user
        self.password = config.mysql.password
        self.pool_size = config.mysql.pool_size


# src/magaldi/search/elasticsearch.py
from magaldi.config import get_config

class ElasticsearchClient:
    def __init__(self):
        config = get_config()
        self.url = config.elasticsearch.url
        self.index = config.elasticsearch.index
        self.timeout = config.elasticsearch.timeout


# src/magaldi/ai/summarization.py
from magaldi.config import get_config

class SummarizationWorkerPool:
    def __init__(self):
        config = get_config()
        self.ollama_url = config.ollama.url
        self.model = config.ollama.summarize_model
        self.temperature = config.ollama.summarize_temperature
        self.max_tokens = config.ollama.summarize_max_tokens
        self.num_workers = config.workers.summarization.count
        self.batch_size = config.workers.summarization.batch_size


# src/magaldi/ai/embedding.py
from magaldi.config import get_config

class EmbeddingWorkerPool:
    def __init__(self):
        config = get_config()
        self.ollama_url = config.ollama.url
        self.model = config.ollama.embed_model
        self.dimensions = config.ollama.embed_dimensions
        self.num_workers = config.workers.embedding.count
        self.batch_size = config.workers.embedding.batch_size
```

---

## CLI Configuration Override

```bash
# Use default config locations
magaldi parse /path/to/repo --user main

# Specify config file
magaldi --config /path/to/config.yaml parse /path/to/repo --user main

# Override specific values via env
MAGALDI_LOG_LEVEL=DEBUG magaldi parse /path/to/repo --user main

# Override via CLI flags (highest priority)
magaldi parse /path/to/repo --user main --mysql-host db.example.com
```

---

## Testing Configuration

```python
# tests/conftest.py
import pytest
from magaldi.config import MagaldiConfig, MySQLConfig, load_config

@pytest.fixture
def test_config():
    """Provide test configuration."""
    return MagaldiConfig(
        mysql=MySQLConfig(
            host="localhost",
            port=3307,  # Test database port
            database="magaldi_test",
            user="test",
            password="test",
        ),
        # ... other test overrides
    )

@pytest.fixture(autouse=True)
def reset_config():
    """Reset global config between tests."""
    from magaldi import config
    config._config = None
    yield
    config._config = None
```

---

## Docker Environment

```yaml
# docker-compose.yml
services:
  magaldi:
    image: magaldi:latest
    environment:
      # All config via environment
      - MAGALDI_MYSQL_HOST=mysql
      - MAGALDI_MYSQL_PASSWORD=${MYSQL_PASSWORD}
      - MAGALDI_ELASTICSEARCH_URL=http://elasticsearch:9200
      - MAGALDI_OLLAMA_URL=http://ollama:11434
      - MAGALDI_LOG_LEVEL=INFO
    volumes:
      # Or mount config file
      - ./config.yaml:/etc/magaldi/config.yaml:ro
```

---

## Summary

| Principle | Implementation |
|-----------|----------------|
| Single source of truth | One config file, one schema |
| No hardcoded values | All values from config or defaults |
| Environment overrides | Env vars override file values |
| Type safety | Dataclasses with validation |
| Testability | Config can be injected/reset |
| Documentation | Schema is self-documenting |
