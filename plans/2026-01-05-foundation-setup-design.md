# Foundation Setup Design

**Date:** 2026-01-05
**Status:** Approved

## Overview

Initial project setup covering Docker infrastructure, Python project structure, and central configuration module. This forms the foundation for all subsequent phase implementations.

## Key Decisions

1. **Docker for external services only** - MySQL, Elasticsearch, Redis run in Docker. Python application runs locally in venv for easier file watching and development iteration.

2. **Flexible Ollama** - Supports both local Ollama installation (default) and Docker-based Ollama (via `--profile ollama`).

3. **Modern Python packaging** - Using pyproject.toml, src layout, type hints, ruff for linting.

4. **Central configuration** - All settings flow from `MagaldiConfig`, no hardcoded values in modules.

## Directory Structure

```
magaldi/
├── docker/
│   └── init-db.sql            # MySQL schema initialization
├── src/
│   └── magaldi/
│       ├── __init__.py        # Package marker + version
│       ├── config.py          # Central configuration (phase0)
│       ├── cli.py             # Click-based CLI entry point
│       └── db/                # Database modules (future)
├── tests/
│   ├── __init__.py
│   ├── conftest.py            # Pytest fixtures, config reset
│   ├── test_config.py         # TDD tests for config module
│   └── fixtures/
│       └── config/            # Test config files
├── config/
│   └── magaldi.yaml           # Example application config
├── docker-compose.yml         # External services (mysql, es, redis)
├── .env.example               # Environment template
├── pyproject.toml             # Python project metadata
├── requirements.txt           # All Python dependencies
├── Makefile                   # Dev commands
└── README.md                  # Comprehensive project documentation
```

## Docker Compose Services

### Core Services (always started)
- `mysql` - Percona 8.0, port 3306, healthcheck, persistent volume
- `elasticsearch` - 8.11.0, port 9200, xpack monitoring enabled
- `redis` - 7-alpine, port 6379, for caching

### Optional Services (profile-based)
- `ollama` - Latest, port 11434, `--profile ollama`
- `ollama-init` - One-shot model puller, depends on ollama

## Configuration Module

### Dataclass Hierarchy

```python
@dataclass
class MagaldiConfig:
    mysql: MySQLConfig
    elasticsearch: ElasticsearchConfig
    ollama: OllamaConfig
    workers: WorkersConfig
    parser: ParserConfig
    search: SearchConfig
    web: WebConfig
    mcp: MCPConfig
    logging: LoggingConfig
    user_data: UserDataConfig
```

### Priority Chain
1. Environment variables (highest)
2. Config file (magaldi.yaml)
3. Dataclass defaults (lowest)

### Key Functions
- `load_config(path=None)` - Load and cache configuration
- `get_config()` - Retrieve cached config (fails if not loaded)
- `reset_config()` - Clear cache (for testing)

## Python Project Structure

### Package Layout
```
src/magaldi/
├── __init__.py          # Version, package metadata
├── config.py            # Central configuration
├── cli.py               # Click-based CLI entry point
├── db/                  # Database modules
│   ├── mysql.py         # Connection pool, queries
│   └── elasticsearch.py # ES client, index management
├── parser/              # Phase 1-3 (future)
├── ai/                  # Phase 5-6 (future)
├── mcp/                 # Phase 7 (future)
└── web/                 # Phase 8 (future)
```

### Design Principles
- Each module imports config via `from magaldi.config import get_config()`
- No circular dependencies - config is the root
- Lazy initialization - connections created on first use
- Type hints throughout (mypy strict)
- Ruff for linting/formatting

## Tests

### Structure
```
tests/
├── conftest.py              # Shared fixtures
├── test_config.py           # Config module tests
└── fixtures/
    └── config/
        ├── valid.yaml       # Complete valid config
        ├── minimal.yaml     # Only required fields
        └── invalid.yaml     # For error testing
```

### Test Coverage for Config
- Load from file (valid YAML)
- Load with missing file (uses defaults)
- Environment overrides work
- Priority chain (env > file > defaults)
- Validation errors for missing required values
- Type conversion (string env to int port)
- `get_config()` fails before `load_config()`

## README Structure

1. Quick Start - Prerequisites, one-liner setup
2. Installation - Clone, venv, deps, services
3. Configuration - Env vars, config file options
4. Usage - Parser, MCP server, Web UI, CLI
5. Development - Structure, tests, Makefile
6. Architecture - Links to plans/, diagram
7. Troubleshooting - Common issues, health checks

## Makefile Commands

```makefile
setup          # Create venv, install deps
services       # Start core Docker services
services-full  # Start all services including Ollama
test           # Run pytest
lint           # Run ruff
typecheck      # Run mypy
dev            # Run development server
clean          # Remove generated files
```

## Implementation Order

1. Create directory structure
2. Write docker-compose.yml (core + optional ollama)
3. Write docker/init-db.sql
4. Write pyproject.toml, requirements.txt
5. Write Makefile
6. Write .env.example
7. Write tests/test_config.py (TDD - tests first)
8. Write src/magaldi/config.py (make tests pass)
9. Write tests/conftest.py
10. Write config/magaldi.yaml (example)
11. Write README.md
12. Verify all tests pass
