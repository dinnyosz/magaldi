# Magaldi

**Code discovery engine for AI agents and developers.**

Magaldi helps AI agents and developers navigate codebases through intelligent indexing and semantic search. Named after Agustín Magaldi (who helped launch Eva Perón's career), it launches your code understanding to the next level.

## Features

- **Multi-language parsing** - Python, JavaScript, TypeScript, PHP, Rust via Tree-sitter
- **Semantic search** - Find code by meaning, not just keywords
- **AI-powered summaries** - Automatic summarization of functions, classes, and files
- **MCP integration** - Works directly with Claude Code as an MCP server
- **Multi-user support** - Main branch + user overlays for team development
- **Web UI** - Browse, search, and visualize your codebase

## Quick Start

### Prerequisites

- Python 3.11+
- Docker and Docker Compose
- Ollama (local installation or via Docker)

### Installation

```bash
# Clone the repository
git clone https://github.com/magaldi/magaldi.git
cd magaldi

# Create environment file
cp .env.example .env
# Edit .env and set MAGALDI_MYSQL_PASSWORD

# Create virtual environment and install dependencies
python3 -m venv .venv
.venv/bin/pip install -e .

# Start Docker services (MySQL, Elasticsearch, Redis, Kibana)
docker compose up -d

# Activate virtual environment
source .venv/bin/activate

# Pull Ollama models (if using local Ollama)
ollama pull qwen2.5-coder:7b
ollama pull snowflake-arctic-embed2

# Run tests to verify installation
.venv/bin/pytest tests/ -v
```

### Using Docker Ollama (Optional)

If you don't have Ollama installed locally:

```bash
# Start all services including Ollama
docker compose --profile ollama up -d
```

## Usage

### Parse a Repository

```bash
# Parse as the main branch (typically done by CI)
magaldi parse /path/to/your/repo --user main

# Parse as a specific user (overlays on main)
magaldi parse /path/to/your/repo --user alice
```

### Start the Web UI

```bash
magaldi web serve
# Open http://localhost:8080
```

### Start the MCP Server (for Claude Code)

```bash
magaldi mcp serve
```

### CLI Reference

```bash
magaldi --help              # Show all commands
magaldi parse --help        # Parse command options
magaldi web serve --help    # Web server options
magaldi worker start --help # Worker options
```

## Configuration

Magaldi uses a layered configuration system:

1. **Environment variables** (highest priority)
2. **Config file** (`config/magaldi.yaml`)
3. **Built-in defaults** (lowest priority)

### Environment Variables

```bash
# Required
MAGALDI_MYSQL_PASSWORD=yourpassword

# Optional overrides - MySQL
MAGALDI_MYSQL_HOST=localhost
MAGALDI_MYSQL_PORT=3306

# Optional overrides - Elasticsearch
MAGALDI_ELASTICSEARCH_HOST=localhost
MAGALDI_ELASTICSEARCH_PORT=9200
MAGALDI_ELASTICSEARCH_SCHEME=http

# Optional overrides - Redis
MAGALDI_REDIS_HOST=localhost
MAGALDI_REDIS_PORT=6379
MAGALDI_REDIS_PASSWORD=  # If auth required

# Optional overrides - Other
MAGALDI_OLLAMA_URL=http://localhost:11434
MAGALDI_LOG_LEVEL=INFO
MAGALDI_WEB_PORT=8080
```

### Config File

See `config/magaldi.yaml` for all available options.

### Repository Config

Each repository needs a `magaldi.yaml` in its root:

```yaml
# Required
scope: my-project-scope

# Optional
name: my-project
description: "Project description"
tags:
  - api
  - backend

exclude_directories:
  - node_modules
  - .venv

exclude_files:
  - "*.min.js"
```

## Development

### Project Structure

```
magaldi/
├── src/magaldi/         # Main package
│   ├── config.py        # Central configuration
│   ├── parser/          # Code parsing (Phases 1-3)
│   ├── storage/         # Storage layer (Phase 4)
│   ├── summarization/   # AI summarization (Phase 5)
│   ├── embedding/       # Vector embedding (Phase 6)
│   ├── mcp/             # MCP server (Phase 7)
│   └── web/             # Web UI (Phase 8)
├── tests/               # Test suite
├── config/              # Configuration files
├── docker/              # Docker resources
└── plans/               # Design documents
```

### Direct Commands

**Setup:**
```bash
python3 -m venv .venv
.venv/bin/pip install -e .
```

**Start/Stop Services:**
```bash
docker compose up -d              # Start MySQL, ES, Redis, Kibana
docker compose --profile ollama up -d  # Include Ollama
docker compose down               # Stop all services
docker compose ps                 # Check status
docker compose logs mysql         # View logs
```

**Service URLs:**
- Kibana: http://localhost:5601
- Elasticsearch: http://localhost:9200

**Run Tests:**
```bash
.venv/bin/pytest tests/ -v                    # All tests
.venv/bin/pytest tests/test_config.py -v      # Specific file
.venv/bin/pytest tests/ -v --tb=short         # Short traceback
.venv/bin/pytest tests/ --cov=src/magaldi     # With coverage
```

**Type Checking:**
```bash
.venv/bin/mypy src/
```

**Linting and Formatting:**
```bash
.venv/bin/ruff check src/ tests/      # Check for issues
.venv/bin/ruff check src/ --fix       # Auto-fix issues
.venv/bin/ruff format src/ tests/     # Format code
```

**Ollama Models:**
```bash
ollama pull qwen2.5-coder:7b          # Summarization model
ollama pull snowflake-arctic-embed2   # Embedding model
ollama list                           # List installed models
```

### Code Style

- **Formatter**: Ruff
- **Linter**: Ruff
- **Type checker**: mypy (strict mode)
- **Line length**: 100 characters

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PARSER PIPELINE                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│  Discovery  →  Change Detection  →  Parsing  →  Storage                     │
│  (paths)       (SHA256 hash)        (Tree-sitter) (MySQL + ES)              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                         AI PROCESSING PIPELINE                               │
├─────────────────────────────────────────────────────────────────────────────┤
│  Summarization  →  Embedding  →  MCP Server  →  Web UI                      │
│  (Ollama LLM)      (vectors)     (Claude Code)   (Dashboard)                │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Technology Stack

| Component | Technology |
|-----------|------------|
| Parser | Tree-sitter |
| Database | MySQL 8.0 |
| Vector Store | Elasticsearch 8.11.0 |
| ES Dashboard | Kibana 8.11.0 |
| Cache | Redis |
| AI Models | Ollama (qwen2.5-coder:7b, snowflake-arctic-embed2) |
| Web Framework | FastAPI |
| MCP | Python MCP SDK |

### Design Documents

Detailed design documents are in `plans/`:

- `architecture_overview.md` - High-level architecture
- `phase1_discovery.md` - Path validation, config loading
- `phase2_change_detection.md` - SHA256 hashing, diff logic
- `phase3_parsing.md` - Tree-sitter extraction
- `phase4_storage.md` - MySQL/ES storage
- `phase5_summarization.md` - AI summarization
- `phase6_embedding.md` - Vector generation
- `phase7_mcp_server.md` - MCP tools
- `phase8_web_ui.md` - Web interface

## Troubleshooting

### Services Not Starting

```bash
# Check service status
docker compose ps

# View logs
docker compose logs mysql
docker compose logs elasticsearch
```

### Elasticsearch Memory Issues

```bash
# Increase heap size in .env
ES_HEAP_SIZE=2g

# Or increase host vm.max_map_count
sudo sysctl -w vm.max_map_count=262144
```

### Database Connection Issues

```bash
# Check MySQL is ready
docker exec -it magaldi-mysql mysql -u magaldi -p -e "SELECT 1"

# Reset database (WARNING: deletes all data)
docker compose down -v
docker compose up -d
```

### Ollama Models Not Loading

```bash
# Check Ollama is running
curl http://localhost:11434/api/tags

# Pull models manually
ollama pull qwen2.5-coder:7b
ollama pull snowflake-arctic-embed2
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Write tests (TDD approach)
4. Implement the feature
5. Verify with:
   ```bash
   .venv/bin/pytest tests/ -v
   .venv/bin/mypy src/
   .venv/bin/ruff check src/ tests/
   ```
6. Submit a pull request

## License

MIT License - see LICENSE file for details.
