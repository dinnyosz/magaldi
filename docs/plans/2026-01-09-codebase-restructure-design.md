# Codebase Restructure Design

## Overview

Reorganize the codebase by functionality into three main packages:
- `magaldi_core` - Core indexing pipeline
- `magaldi_mcp` - MCP server for Claude Code integration
- `shared` - Common utilities (db, config, ai)

## Directory Structure

```
src/
├── magaldi_core/            # Core indexing pipeline
│   ├── __init__.py
│   ├── discovery.py
│   ├── change_detection.py
│   ├── code_parser.py
│   ├── tree_sitter_manager.py
│   ├── storage.py
│   └── processor.py
│
├── magaldi_mcp/             # MCP server
│   ├── __init__.py
│   ├── server.py
│   └── tools.py
│
└── shared/                  # Common utilities
    ├── __init__.py
    ├── cli.py
    ├── config.py
    ├── db/
    │   ├── __init__.py
    │   ├── elasticsearch.py
    │   └── redis.py
    └── ai/
        ├── __init__.py
        ├── summarization.py
        ├── embedding.py
        └── clustering/
            ├── __init__.py
            ├── clusterer.py
            └── feature_processor.py
```

## Import Structure

```python
# magaldi_core imports from shared
from shared.db.elasticsearch import ElasticsearchRepository
from shared.config import get_config

# magaldi_mcp imports from shared
from shared.db.elasticsearch import ElasticsearchRepository
from shared.ai.embedding import OllamaEmbedClient

# CLI imports from all
from magaldi_core.processor import run_parse_pipeline
from magaldi_mcp.server import run_server
```

Dependency graph:
```
magaldi_core ──▶ shared
magaldi_mcp  ──▶ shared
     CLI     ──▶ magaldi_core, magaldi_mcp, shared
```

No circular dependencies. `magaldi_core` and `magaldi_mcp` don't import each other.

## CLI Subcommands

```bash
magaldi parse /path/to/repo --user alice    # Index a repository
magaldi mcp --repo-root /path/to/repo       # Start MCP server
magaldi features                             # Process features/clustering
```

Summarization and embedding are triggered automatically by the parse pipeline.

## Package Configuration

```toml
[project]
name = "magaldi"

[project.scripts]
magaldi = "shared.cli:main"

[tool.setuptools.packages.find]
where = ["src"]
include = ["magaldi_core*", "magaldi_mcp*", "shared*"]
```

## Migration Plan

| Current Location | New Location |
|-----------------|--------------|
| `magaldi/parser/*` | `magaldi_core/` |
| `magaldi/processing/processor.py` | `magaldi_core/processor.py` |
| `magaldi/storage/storage.py` | `magaldi_core/storage.py` |
| `magaldi/mcp/*` | `magaldi_mcp/` |
| `magaldi/db/*` | `shared/db/` |
| `magaldi/config.py` | `shared/config.py` |
| `magaldi/summarization/*` | `shared/ai/summarization.py` |
| `magaldi/embedding/*` | `shared/ai/embedding.py` |
| `magaldi/clustering/*` | `shared/ai/clustering/` |
| `magaldi/cli.py` | `shared/cli.py` |

Then update all imports across the codebase.

## Future Extensions

- `magaldi_web/` - Web UI for browsing indexed code
