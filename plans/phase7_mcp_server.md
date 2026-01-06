# Magaldi MCP Server - Phase 7: MCP Server

## Overview

The MCP Server exposes Magaldi's code discovery capabilities to Claude Code via the Model Context Protocol. This enables AI agents to semantically search codebases, understand code structure, and navigate repositories during development tasks.

```
┌─────────────────────────────────────────────────────────────────┐
│                      PHASE 7: MCP SERVER                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  7.1              7.2              7.3              7.4          │
│  ─────            ─────            ─────            ─────        │
│  SERVER       →   TOOLS        →   RESOURCES   →   PROMPTS      │
│  SETUP            DEFINITION       & CONTEXT       & SKILLS     │
│                                                                 │
│  • Python MCP     • search_code    • repo info     • Exploration│
│  • Config         • find_similar   • file tree     • Navigation │
│  • Auth           • get_context    • summaries     • Understand │
│                   • list_repos                                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Integration Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     CLAUDE CODE                                  │
│                                                                 │
│  User: "Find authentication handlers"                           │
│                          │                                      │
│                          ▼                                      │
│  ┌─────────────────────────────────────┐                       │
│  │         MCP Client                   │                       │
│  └──────────────┬──────────────────────┘                       │
│                 │ stdio/SSE                                     │
└─────────────────┼───────────────────────────────────────────────┘
                  │
┌─────────────────┼───────────────────────────────────────────────┐
│                 ▼                                               │
│  ┌─────────────────────────────────────┐                       │
│  │       Magaldi MCP Server            │                       │
│  │                                      │                       │
│  │  Tools:                             │                       │
│  │  • search_code (semantic)           │                       │
│  │  • find_similar (by element)        │                       │
│  │  • get_context (hierarchy)          │                       │
│  │  • list_repos                       │                       │
│  │  • get_file_summary                 │                       │
│  │                                      │                       │
│  └──────────────┬──────────────────────┘                       │
│                 │                                               │
│                 ▼                                               │
│  ┌──────────────────────┐  ┌──────────────────────┐           │
│  │   Elasticsearch      │  │      MySQL           │           │
│  │   (Vector Search)    │  │   (Metadata)         │           │
│  └──────────────────────┘  └──────────────────────┘           │
│                                                                 │
│                     MAGALDI BACKEND                             │
└─────────────────────────────────────────────────────────────────┘
```

---

## Input

Prerequisites from earlier phases:

```python
# Elasticsearch index with embeddings (Phase 6)
# MySQL with summaries and metadata (Phase 5)
# Parsed code elements (Phase 4)

# Required environment:
# - Ollama running with embedding model (for query embedding)
# - MySQL connection
# - Elasticsearch connection
```

---

## 7.1 Server Setup

### Purpose

Initialize MCP server with proper configuration, authentication, and connection management.

### MCP Server Implementation

```python
#!/usr/bin/env python3
"""Magaldi MCP Server - Code Discovery for Claude Code"""

import asyncio
import json
import logging
from typing import Any, Optional
from contextlib import asynccontextmanager

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool,
    TextContent,
    Resource,
    ResourceTemplate,
    Prompt,
    PromptMessage,
    GetPromptResult,
)

from magaldi.db import MySQLConnection
from magaldi.search import ElasticsearchClient
from magaldi.ollama import OllamaEmbedClient
from magaldi.config import load_config

log = logging.getLogger(__name__)


class MagaldiMCPServer:
    """MCP Server for Magaldi code discovery"""

    def __init__(self, config_path: Optional[str] = None):
        self.config = load_config(config_path)
        self.server = Server("magaldi")
        self.db: Optional[MySQLConnection] = None
        self.es: Optional[ElasticsearchClient] = None
        self.ollama: Optional[OllamaEmbedClient] = None

        # Register handlers
        self._register_tools()
        self._register_resources()
        self._register_prompts()

    async def initialize(self):
        """Initialize connections"""

        self.db = MySQLConnection(
            host=self.config.mysql_host,
            port=self.config.mysql_port,
            database=self.config.mysql_database,
            user=self.config.mysql_user,
            password=self.config.mysql_password,
        )
        await self.db.connect()

        self.es = ElasticsearchClient(
            hosts=[self.config.elasticsearch_url],
        )

        self.ollama = OllamaEmbedClient(
            url=self.config.ollama_url,
            model=self.config.embed_model,
        )

        log.info("Magaldi MCP Server initialized")

    async def shutdown(self):
        """Clean up connections"""
        if self.db:
            await self.db.close()
        log.info("Magaldi MCP Server shutdown")

    async def run(self):
        """Run the MCP server"""
        async with stdio_server() as (read_stream, write_stream):
            await self.initialize()
            try:
                await self.server.run(
                    read_stream,
                    write_stream,
                    self.server.create_initialization_options()
                )
            finally:
                await self.shutdown()
```

### Configuration

```python
# config/magaldi_mcp.yaml
server:
  name: magaldi
  version: "1.0.0"
  description: "Code discovery and semantic search for AI agents"

# Database connections
mysql:
  host: localhost
  port: 3306
  database: magaldi
  user: magaldi
  password: ${MAGALDI_MYSQL_PASSWORD}

elasticsearch:
  url: http://localhost:9200
  index: magaldi_code_elements

ollama:
  url: http://localhost:11434
  embed_model: snowflake-arctic-embed2

# Search defaults
search:
  default_limit: 10
  max_limit: 50
  default_scope: null  # null = search all scopes

# User context
user:
  default_username: main
```

### Claude Code Configuration

```json
// ~/.claude/claude_desktop_config.json
{
  "mcpServers": {
    "magaldi": {
      "command": "magaldi-mcp",
      "args": ["--config", "/path/to/magaldi_mcp.yaml"],
      "env": {
        "MAGALDI_MYSQL_PASSWORD": "secret"
      }
    }
  }
}
```

### Alternative: Docker Deployment

```yaml
# docker-compose.yml
services:
  magaldi-mcp:
    image: magaldi/mcp-server:latest
    environment:
      - MYSQL_HOST=mysql
      - MYSQL_PASSWORD=${MYSQL_PASSWORD}
      - ELASTICSEARCH_URL=http://elasticsearch:9200
      - OLLAMA_URL=http://ollama:11434
    depends_on:
      - mysql
      - elasticsearch
      - ollama
```

---

## 7.2 Tool Definitions

### Purpose

Define MCP tools that Claude Code can invoke for code discovery.

### Tool: search_code

Semantic search across the codebase.

```python
def _register_tools(self):
    """Register all MCP tools"""

    @self.server.tool()
    async def search_code(
        query: str,
        scope: Optional[str] = None,
        repository: Optional[str] = None,
        username: str = "main",
        element_types: Optional[list[str]] = None,
        language: Optional[str] = None,
        limit: int = 10
    ) -> list[dict]:
        """
        Semantic search for code elements.

        Args:
            query: Natural language search query (e.g., "authentication handler",
                   "database connection pool", "error handling middleware")
            scope: Filter by scope (e.g., "backend", "frontend")
            repository: Filter by repository name
            username: User branch to search ("main" for base, or username for overlay)
            element_types: Filter by types ["file", "class", "function", "method"]
            language: Filter by language ("python", "javascript", etc.)
            limit: Maximum results (default 10, max 50)

        Returns:
            List of matching code elements with summaries and locations
        """

        # Validate limit
        limit = min(limit, self.config.search.max_limit)

        # Generate query embedding
        query_embedding = await self.ollama.embed_async(query)

        # Build ES query
        es_query = build_semantic_query(
            query_embedding=query_embedding,
            query_text=query,
            scope=scope,
            repository=repository,
            username=username,
            element_types=element_types,
            language=language,
            limit=limit
        )

        # Execute search
        results = await self.es.search_async(
            index="magaldi_code_elements",
            body=es_query
        )

        # Format results
        return [
            {
                "element_id": hit["_source"]["element_id"],
                "name": hit["_source"]["name"],
                "type": hit["_source"]["element_type"],
                "file": hit["_source"]["relative_path"],
                "line": hit["_source"]["line_start"],
                "language": hit["_source"]["language"],
                "summary": hit["_source"].get("summary", ""),
                "signature": hit["_source"].get("signature", ""),
                "score": hit["_score"],
                "repository": hit["_source"]["repository"],
                "scope": hit["_source"]["scope"],
            }
            for hit in results["hits"]["hits"]
        ]


def build_semantic_query(query_embedding: list[float], query_text: str,
                        scope: Optional[str], repository: Optional[str],
                        username: str, element_types: Optional[list[str]],
                        language: Optional[str], limit: int) -> dict:
    """Build Elasticsearch hybrid search query"""

    # Base filters
    filters = [
        {"terms": {"username": ["main", username]}}  # Always include main + user overlay
    ]

    if scope:
        filters.append({"term": {"scope": scope}})
    if repository:
        filters.append({"term": {"repository": repository}})
    if element_types:
        filters.append({"terms": {"element_type": element_types}})
    if language:
        filters.append({"term": {"language": language}})

    return {
        "size": limit,
        "query": {
            "bool": {
                "filter": filters,
                "should": [
                    # Semantic similarity (primary)
                    {
                        "script_score": {
                            "query": {"match_all": {}},
                            "script": {
                                "source": "(cosineSimilarity(params.qv, 'embedding') + 1.0) * 2",
                                "params": {"qv": query_embedding}
                            }
                        }
                    },
                    # Keyword boost on name
                    {"match": {"name": {"query": query_text, "boost": 1.5}}},
                    # Keyword boost on summary
                    {"match": {"summary": {"query": query_text, "boost": 1.0}}},
                ],
                "minimum_should_match": 1
            }
        },
        "_source": [
            "element_id", "name", "element_type", "relative_path",
            "line_start", "language", "summary", "signature",
            "repository", "scope"
        ]
    }
```

### Tool: find_similar

Find code similar to a given element.

```python
@self.server.tool()
async def find_similar(
    element_id: str,
    limit: int = 10,
    same_repo_only: bool = False
) -> list[dict]:
    """
    Find code elements similar to a given element.

    Args:
        element_id: The element ID to find similar code for
        limit: Maximum results (default 10)
        same_repo_only: Only search within same repository

    Returns:
        List of similar code elements ranked by similarity
    """

    # Get element's embedding from ES
    element = await self.es.get_async(
        index="magaldi_code_elements",
        id=element_id,
        _source=["embedding", "scope", "repository", "username"]
    )

    if not element["found"]:
        raise ValueError(f"Element not found: {element_id}")

    embedding = element["_source"]["embedding"]

    # Build filter
    filters = [
        {"terms": {"username": ["main", element["_source"]["username"]]}}
    ]

    if same_repo_only:
        filters.append({"term": {"scope": element["_source"]["scope"]}})
        filters.append({"term": {"repository": element["_source"]["repository"]}})

    # Exclude the source element
    must_not = [{"term": {"element_id": element_id}}]

    # Search for similar
    results = await self.es.search_async(
        index="magaldi_code_elements",
        body={
            "size": limit,
            "query": {
                "bool": {
                    "filter": filters,
                    "must_not": must_not,
                    "must": [
                        {
                            "script_score": {
                                "query": {"match_all": {}},
                                "script": {
                                    "source": "cosineSimilarity(params.qv, 'embedding') + 1.0",
                                    "params": {"qv": embedding}
                                }
                            }
                        }
                    ]
                }
            },
            "_source": [
                "element_id", "name", "element_type", "relative_path",
                "line_start", "language", "summary", "signature",
                "repository", "scope"
            ]
        }
    )

    return [
        {
            "element_id": hit["_source"]["element_id"],
            "name": hit["_source"]["name"],
            "type": hit["_source"]["element_type"],
            "file": hit["_source"]["relative_path"],
            "line": hit["_source"]["line_start"],
            "language": hit["_source"]["language"],
            "summary": hit["_source"].get("summary", ""),
            "similarity": hit["_score"] - 1.0,  # Convert back to cosine similarity
            "repository": hit["_source"]["repository"],
            "scope": hit["_source"]["scope"],
        }
        for hit in results["hits"]["hits"]
    ]
```

### Tool: get_context

Get hierarchical context for an element.

```python
@self.server.tool()
async def get_context(
    element_id: str,
    include_siblings: bool = False,
    include_children: bool = True
) -> dict:
    """
    Get hierarchical context for a code element.

    Returns the element's file summary, parent class (if applicable),
    and optionally siblings and children.

    Args:
        element_id: The element to get context for
        include_siblings: Include sibling elements (same parent)
        include_children: Include child elements (methods in class, etc.)

    Returns:
        Hierarchical context with summaries at each level
    """

    # Get element from MySQL (has full data)
    element = await self.db.query_one_async("""
        SELECT * FROM code_elements WHERE element_id = %s
    """, (element_id,))

    if not element:
        raise ValueError(f"Element not found: {element_id}")

    context = {
        "element": {
            "id": element.element_id,
            "name": element.name,
            "type": element.element_type,
            "file": element.relative_path,
            "line_start": element.line_start,
            "line_end": element.line_end,
            "summary": element.summary,
            "signature": element.signature,
            "docstring": element.docstring,
        },
        "file": None,
        "parent": None,
        "siblings": [],
        "children": [],
    }

    # Get file context
    file_element = await self.db.query_one_async("""
        SELECT element_id, name, summary FROM code_elements
        WHERE scope = %s AND repository = %s AND username = %s
          AND relative_path = %s AND element_type = 'file'
    """, (element.scope, element.repository, element.username, element.relative_path))

    if file_element:
        context["file"] = {
            "id": file_element.element_id,
            "name": file_element.name,
            "summary": file_element.summary,
        }

    # Get parent context
    if element.parent_id:
        parent = await self.db.query_one_async("""
            SELECT element_id, name, element_type, summary, signature
            FROM code_elements WHERE element_id = %s
        """, (element.parent_id,))

        if parent:
            context["parent"] = {
                "id": parent.element_id,
                "name": parent.name,
                "type": parent.element_type,
                "summary": parent.summary,
                "signature": parent.signature,
            }

    # Get siblings
    if include_siblings and element.parent_id:
        siblings = await self.db.query_async("""
            SELECT element_id, name, element_type, line_start, summary
            FROM code_elements
            WHERE parent_id = %s AND element_id != %s
            ORDER BY line_start
        """, (element.parent_id, element_id))

        context["siblings"] = [
            {
                "id": s.element_id,
                "name": s.name,
                "type": s.element_type,
                "line": s.line_start,
                "summary": s.summary,
            }
            for s in siblings
        ]

    # Get children
    if include_children:
        children = await self.db.query_async("""
            SELECT element_id, name, element_type, line_start, summary, signature
            FROM code_elements
            WHERE parent_id = %s
            ORDER BY line_start
        """, (element_id,))

        context["children"] = [
            {
                "id": c.element_id,
                "name": c.name,
                "type": c.element_type,
                "line": c.line_start,
                "summary": c.summary,
                "signature": c.signature,
            }
            for c in children
        ]

    return context
```

### Tool: discover

Discover available Magaldi capabilities.

```python
@self.server.tool()
async def discover() -> dict:
    """
    Discover all available Magaldi capabilities.

    Returns detailed information about available tools, resources,
    and prompts including their parameters and descriptions.

    Returns:
        Dictionary with tools, resources, prompts, and server info
    """

    return {
        "server": {
            "name": self.config.mcp.server_name,
            "version": self.config.mcp.server_version,
            "description": "Magaldi - Semantic code discovery for AI agents",
        },
        "tools": [
            {
                "name": "search_code",
                "description": "Semantic search for code elements",
                "parameters": {
                    "query": {"type": "string", "required": True, "description": "Natural language search query"},
                    "scope": {"type": "string", "required": False, "description": "Filter by scope"},
                    "repository": {"type": "string", "required": False, "description": "Filter by repository"},
                    "element_types": {"type": "array", "required": False, "description": "Filter by types: file, class, function, method"},
                    "language": {"type": "string", "required": False, "description": "Filter by language"},
                    "limit": {"type": "integer", "required": False, "default": 10, "description": "Max results (1-50)"},
                },
                "examples": [
                    {"query": "authentication handler"},
                    {"query": "database connection", "language": "python"},
                    {"query": "API endpoints", "element_types": ["function", "method"]},
                ]
            },
            {
                "name": "find_similar",
                "description": "Find code similar to a given element",
                "parameters": {
                    "element_id": {"type": "string", "required": True, "description": "Element ID to find similar code for"},
                    "limit": {"type": "integer", "required": False, "default": 10},
                    "same_repo_only": {"type": "boolean", "required": False, "default": False},
                },
            },
            {
                "name": "get_context",
                "description": "Get hierarchical context for a code element",
                "parameters": {
                    "element_id": {"type": "string", "required": True},
                    "include_siblings": {"type": "boolean", "required": False, "default": False},
                    "include_children": {"type": "boolean", "required": False, "default": True},
                },
            },
            {
                "name": "list_repos",
                "description": "List all indexed repositories",
                "parameters": {
                    "scope": {"type": "string", "required": False, "description": "Filter by scope"},
                },
            },
            {
                "name": "get_file_summary",
                "description": "Get summary and structure of a file",
                "parameters": {
                    "file_path": {"type": "string", "required": True},
                    "scope": {"type": "string", "required": True},
                    "repository": {"type": "string", "required": True},
                    "username": {"type": "string", "required": False, "default": "main"},
                },
            },
            {
                "name": "discover",
                "description": "This tool - discover available capabilities",
                "parameters": {},
            },
        ],
        "resources": [
            {
                "uri_template": "magaldi://repo/{scope}/{repository}",
                "description": "Repository overview with file list and statistics",
            },
            {
                "uri_template": "magaldi://file/{scope}/{repository}/{path}",
                "description": "File details with element structure",
            },
        ],
        "prompts": [
            {
                "name": "explore-repo",
                "description": "Generate a prompt for exploring a repository",
                "parameters": {"scope": "string", "repository": "string"},
            },
            {
                "name": "find-implementation",
                "description": "Generate a prompt for finding feature implementation",
                "parameters": {"feature": "string", "scope": "string (optional)", "repository": "string (optional)"},
            },
            {
                "name": "understand-code",
                "description": "Generate a prompt for understanding a code element",
                "parameters": {"element_id": "string"},
            },
        ],
        "indexes": await self._get_index_summary(),
    }


async def _get_index_summary(self) -> dict:
    """Get summary of indexed content."""

    repo_count = await self.db.query_one_async(
        "SELECT COUNT(*) as count FROM repositories"
    )
    element_count = await self.db.query_one_async(
        "SELECT COUNT(*) as count FROM code_elements WHERE username = 'main'"
    )
    languages = await self.db.query_async(
        "SELECT DISTINCT language FROM code_elements WHERE username = 'main'"
    )

    return {
        "repositories": repo_count.count,
        "elements": element_count.count,
        "languages": [r.language for r in languages],
    }
```

### Tool: get_stats

Get code statistics for a repository.

```python
@self.server.tool()
async def get_stats(
    scope: str,
    repository: str,
    username: str = "main"
) -> dict:
    """
    Get code statistics for a repository.

    Args:
        scope: Repository scope
        repository: Repository name
        username: User branch (default "main")

    Returns:
        Statistics including element counts, LOC, languages, and complexity metrics
    """

    # Element counts by type
    element_counts = await self.db.query_async("""
        SELECT element_type, COUNT(*) as count
        FROM code_elements
        WHERE scope = %s AND repository = %s AND username = %s
        GROUP BY element_type
    """, (scope, repository, username))

    # Language breakdown
    language_stats = await self.db.query_async("""
        SELECT language, COUNT(*) as files, SUM(line_count) as lines
        FROM file_states
        WHERE scope = %s AND repository = %s AND username = %s
        GROUP BY language
    """, (scope, repository, username))

    # Total lines of code
    total_loc = await self.db.query_one_async("""
        SELECT SUM(line_count) as total
        FROM file_states
        WHERE scope = %s AND repository = %s AND username = %s
    """, (scope, repository, username))

    # Average function size (lines)
    avg_function_size = await self.db.query_one_async("""
        SELECT AVG(line_end - line_start) as avg_lines
        FROM code_elements
        WHERE scope = %s AND repository = %s AND username = %s
          AND element_type IN ('function', 'method')
    """, (scope, repository, username))

    # Files with most elements (complexity indicator)
    complex_files = await self.db.query_async("""
        SELECT relative_path, COUNT(*) as element_count
        FROM code_elements
        WHERE scope = %s AND repository = %s AND username = %s
        GROUP BY relative_path
        ORDER BY element_count DESC
        LIMIT 10
    """, (scope, repository, username))

    return {
        "repository": {"scope": scope, "name": repository},
        "summary": {
            "total_files": sum(l.files for l in language_stats),
            "total_lines": total_loc.total or 0,
            "total_elements": sum(e.count for e in element_counts),
        },
        "elements_by_type": {e.element_type: e.count for e in element_counts},
        "languages": [
            {"language": l.language, "files": l.files, "lines": l.lines}
            for l in language_stats
        ],
        "metrics": {
            "avg_function_lines": round(avg_function_size.avg_lines or 0, 1),
        },
        "complexity_hotspots": [
            {"file": f.relative_path, "elements": f.element_count}
            for f in complex_files
        ]
    }
```

### Tool: get_dependencies

Map dependencies between code elements.

```python
@self.server.tool()
async def get_dependencies(
    element_id: str,
    direction: str = "both",  # "imports", "imported_by", "both"
    depth: int = 1
) -> dict:
    """
    Get dependency relationships for a code element.

    Note: This requires import/call tracking during parsing (Phase 3 enhancement).

    Args:
        element_id: Element to get dependencies for
        direction: "imports" (what this element uses), "imported_by" (what uses this), or "both"
        depth: How many levels deep to traverse (default 1)

    Returns:
        Dependency graph with import and usage relationships
    """

    element = await self.db.query_one_async(
        "SELECT * FROM code_elements WHERE element_id = %s", (element_id,)
    )

    if not element:
        raise ValueError(f"Element not found: {element_id}")

    result = {
        "element": {
            "id": element.element_id,
            "name": element.name,
            "type": element.element_type,
            "file": element.relative_path,
        },
        "imports": [],
        "imported_by": [],
    }

    if direction in ("imports", "both"):
        # Get elements this element imports/calls
        imports = await self.db.query_async("""
            SELECT target_element_id, relationship_type
            FROM element_dependencies
            WHERE source_element_id = %s
        """, (element_id,))

        for imp in imports:
            target = await self.db.query_one_async(
                "SELECT element_id, name, element_type, relative_path FROM code_elements WHERE element_id = %s",
                (imp.target_element_id,)
            )
            if target:
                result["imports"].append({
                    "id": target.element_id,
                    "name": target.name,
                    "type": target.element_type,
                    "file": target.relative_path,
                    "relationship": imp.relationship_type,  # "import", "call", "inherit"
                })

    if direction in ("imported_by", "both"):
        # Get elements that import/call this element
        imported_by = await self.db.query_async("""
            SELECT source_element_id, relationship_type
            FROM element_dependencies
            WHERE target_element_id = %s
        """, (element_id,))

        for dep in imported_by:
            source = await self.db.query_one_async(
                "SELECT element_id, name, element_type, relative_path FROM code_elements WHERE element_id = %s",
                (dep.source_element_id,)
            )
            if source:
                result["imported_by"].append({
                    "id": source.element_id,
                    "name": source.name,
                    "type": source.element_type,
                    "file": source.relative_path,
                    "relationship": dep.relationship_type,
                })

    return result
```

### Tool: get_recent_changes

Get recently modified files and elements.

```python
@self.server.tool()
async def get_recent_changes(
    scope: Optional[str] = None,
    repository: Optional[str] = None,
    days: int = 7,
    limit: int = 20
) -> dict:
    """
    Get recently modified files and elements.

    Args:
        scope: Filter by scope (optional)
        repository: Filter by repository (optional)
        days: Look back this many days (default 7)
        limit: Maximum results (default 20)

    Returns:
        Recently changed files with their modified elements
    """

    # Build filter
    where_clauses = ["parsed_at > DATE_SUB(NOW(), INTERVAL %s DAY)"]
    params = [days]

    if scope:
        where_clauses.append("scope = %s")
        params.append(scope)
    if repository:
        where_clauses.append("repository = %s")
        params.append(repository)

    where_sql = " AND ".join(where_clauses)

    # Get recently changed files
    files = await self.db.query_async(f"""
        SELECT scope, repository, relative_path, language, parsed_at
        FROM file_states
        WHERE {where_sql}
        ORDER BY parsed_at DESC
        LIMIT %s
    """, (*params, limit))

    result = {
        "period_days": days,
        "files_changed": len(files),
        "changes": []
    }

    for f in files:
        # Get elements in this file that changed
        elements = await self.db.query_async("""
            SELECT name, element_type, line_start, summary
            FROM code_elements
            WHERE scope = %s AND repository = %s AND relative_path = %s
              AND username = 'main'
            ORDER BY line_start
        """, (f.scope, f.repository, f.relative_path))

        result["changes"].append({
            "file": f.relative_path,
            "repository": f"{f.scope}/{f.repository}",
            "language": f.language,
            "changed_at": f.parsed_at.isoformat(),
            "elements": [
                {
                    "name": e.name,
                    "type": e.element_type,
                    "line": e.line_start,
                    "summary": e.summary,
                }
                for e in elements[:10]  # Limit elements per file
            ]
        })

    return result
```

### Tool: get_file_contributors

See who has active branches with changes to a file.

```python
@self.server.tool()
async def get_file_contributors(
    file_path: str,
    scope: str,
    repository: str
) -> dict:
    """
    Get users who have active indexed versions of a file.

    Shows who is currently working on a file (has their own branch version indexed).

    Args:
        file_path: Relative path to file
        scope: Repository scope
        repository: Repository name

    Returns:
        List of users with their version info and when it was last parsed
    """

    # Get all user versions of this file (excluding main)
    contributors = await self.db.query_async("""
        SELECT
            fs.username,
            fs.file_hash,
            fs.parsed_at,
            fs.expires_at,
            fs.is_deleted,
            (SELECT file_hash FROM file_states
             WHERE scope = fs.scope AND repository = fs.repository
               AND relative_path = fs.relative_path AND username = 'main') as main_hash
        FROM file_states fs
        WHERE fs.scope = %s
          AND fs.repository = %s
          AND fs.relative_path = %s
          AND fs.username != 'main'
          AND fs.is_deleted = FALSE
          AND (fs.expires_at IS NULL OR fs.expires_at > NOW())
        ORDER BY fs.parsed_at DESC
    """, (scope, repository, file_path))

    # Get main version info
    main_version = await self.db.query_one_async("""
        SELECT file_hash, parsed_at
        FROM file_states
        WHERE scope = %s AND repository = %s AND relative_path = %s AND username = 'main'
    """, (scope, repository, file_path))

    return {
        "file": file_path,
        "repository": f"{scope}/{repository}",
        "main_version": {
            "hash": main_version.file_hash if main_version else None,
            "parsed_at": main_version.parsed_at.isoformat() if main_version else None,
        } if main_version else None,
        "active_contributors": [
            {
                "username": c.username,
                "has_changes": c.file_hash != c.main_hash,  # True if different from main
                "parsed_at": c.parsed_at.isoformat(),
                "expires_at": c.expires_at.isoformat() if c.expires_at else None,
            }
            for c in contributors
        ],
        "contributor_count": len(contributors),
    }
```

### Tool: get_active_users

See all users with active branches in a repository.

```python
@self.server.tool()
async def get_active_users(
    scope: str,
    repository: str
) -> dict:
    """
    Get all users with active indexed branches in a repository.

    Args:
        scope: Repository scope
        repository: Repository name

    Returns:
        List of active users with their file counts and last activity
    """

    users = await self.db.query_async("""
        SELECT
            username,
            COUNT(DISTINCT relative_path) as file_count,
            MAX(parsed_at) as last_activity,
            MIN(expires_at) as earliest_expiry
        FROM file_states
        WHERE scope = %s
          AND repository = %s
          AND username != 'main'
          AND is_deleted = FALSE
          AND (expires_at IS NULL OR expires_at > NOW())
        GROUP BY username
        ORDER BY last_activity DESC
    """, (scope, repository))

    return {
        "repository": f"{scope}/{repository}",
        "active_users": [
            {
                "username": u.username,
                "files_modified": u.file_count,
                "last_activity": u.last_activity.isoformat(),
                "expires_at": u.earliest_expiry.isoformat() if u.earliest_expiry else None,
            }
            for u in users
        ],
        "user_count": len(users),
    }
```

### Tool: list_repos

List available repositories.

```python
@self.server.tool()
async def list_repos(
    scope: Optional[str] = None
) -> list[dict]:
    """
    List all indexed repositories.

    Args:
        scope: Filter by scope (optional)

    Returns:
        List of repositories with statistics
    """

    query = """
        SELECT
            r.scope,
            r.name as repository,
            r.description,
            COUNT(DISTINCT fs.relative_path) as file_count,
            COUNT(ce.element_id) as element_count,
            GROUP_CONCAT(DISTINCT rl.language) as languages
        FROM repositories r
        LEFT JOIN file_states fs ON r.scope = fs.scope AND r.name = fs.repository
        LEFT JOIN code_elements ce ON r.scope = ce.scope AND r.name = ce.repository
        LEFT JOIN repository_languages rl ON r.scope = rl.scope AND r.name = rl.repository
        WHERE fs.username = 'main' AND fs.is_deleted = FALSE
    """

    params = []
    if scope:
        query += " AND r.scope = %s"
        params.append(scope)

    query += " GROUP BY r.scope, r.name, r.description ORDER BY r.scope, r.name"

    repos = await self.db.query_async(query, tuple(params))

    return [
        {
            "scope": r.scope,
            "repository": r.repository,
            "description": r.description,
            "file_count": r.file_count,
            "element_count": r.element_count,
            "languages": r.languages.split(",") if r.languages else [],
        }
        for r in repos
    ]
```

### Tool: get_file_summary

Get summary and structure of a file.

```python
@self.server.tool()
async def get_file_summary(
    file_path: str,
    scope: str,
    repository: str,
    username: str = "main"
) -> dict:
    """
    Get summary and structure of a specific file.

    Args:
        file_path: Relative path to file
        scope: Repository scope
        repository: Repository name
        username: User branch (default "main")

    Returns:
        File summary with list of contained elements
    """

    # Get file element
    file_elem = await self.db.query_one_async("""
        SELECT * FROM code_elements
        WHERE scope = %s AND repository = %s AND username = %s
          AND relative_path = %s AND element_type = 'file'
    """, (scope, repository, username, file_path))

    if not file_elem:
        # Try main if user not found
        if username != "main":
            return await get_file_summary(file_path, scope, repository, "main")
        raise ValueError(f"File not found: {file_path}")

    # Get all elements in file
    elements = await self.db.query_async("""
        SELECT element_id, name, element_type, line_start, line_end,
               summary, signature, level, parent_id
        FROM code_elements
        WHERE scope = %s AND repository = %s AND username = %s
          AND relative_path = %s AND element_type != 'file'
        ORDER BY line_start
    """, (scope, repository, username, file_path))

    # Build tree structure
    def build_tree(elements, parent_id=None):
        children = []
        for e in elements:
            if e.parent_id == parent_id or (parent_id is None and e.level == 1):
                node = {
                    "id": e.element_id,
                    "name": e.name,
                    "type": e.element_type,
                    "line_start": e.line_start,
                    "line_end": e.line_end,
                    "summary": e.summary,
                    "signature": e.signature,
                    "children": build_tree(elements, e.element_id)
                }
                children.append(node)
        return children

    return {
        "file": {
            "path": file_path,
            "language": file_elem.language,
            "summary": file_elem.summary,
            "line_count": file_elem.line_end,
        },
        "structure": build_tree(elements, file_elem.element_id),
        "stats": {
            "classes": sum(1 for e in elements if e.element_type == "class"),
            "functions": sum(1 for e in elements if e.element_type == "function"),
            "methods": sum(1 for e in elements if e.element_type == "method"),
        }
    }
```

---

## 7.3 Resources & Context

### Purpose

Expose repository information as MCP resources for context enrichment.

### Resource: Repository Info

```python
def _register_resources(self):
    """Register MCP resources"""

    @self.server.resource("magaldi://repo/{scope}/{repository}")
    async def get_repo_resource(scope: str, repository: str) -> Resource:
        """Get repository overview as a resource"""

        # Get repo info
        repo = await self.db.query_one_async("""
            SELECT r.*,
                   COUNT(DISTINCT fs.relative_path) as file_count,
                   COUNT(ce.element_id) as element_count
            FROM repositories r
            LEFT JOIN file_states fs ON r.scope = fs.scope AND r.name = fs.repository
            LEFT JOIN code_elements ce ON r.scope = ce.scope AND r.name = ce.repository
            WHERE r.scope = %s AND r.name = %s
              AND fs.username = 'main' AND fs.is_deleted = FALSE
            GROUP BY r.id
        """, (scope, repository))

        if not repo:
            raise ValueError(f"Repository not found: {scope}/{repository}")

        # Get language breakdown
        languages = await self.db.query_async("""
            SELECT language, file_count, line_count
            FROM repository_languages
            WHERE scope = %s AND repository = %s
            ORDER BY line_count DESC
        """, (scope, repository))

        # Get top-level structure (files)
        structure = await self.db.query_async("""
            SELECT relative_path, language, summary
            FROM code_elements
            WHERE scope = %s AND repository = %s AND username = 'main'
              AND element_type = 'file'
            ORDER BY relative_path
            LIMIT 50
        """, (scope, repository))

        content = f"""# Repository: {scope}/{repository}

{repo.description or "No description"}

## Statistics
- Files: {repo.file_count}
- Code Elements: {repo.element_count}

## Languages
"""
        for lang in languages:
            content += f"- {lang.language}: {lang.file_count} files, {lang.line_count} lines\n"

        content += "\n## File Structure\n"
        for f in structure:
            content += f"- `{f.relative_path}` ({f.language}): {f.summary or 'No summary'}\n"

        return Resource(
            uri=f"magaldi://repo/{scope}/{repository}",
            name=f"{scope}/{repository}",
            description=repo.description,
            mimeType="text/markdown",
            text=content
        )
```

### Resource Template: File Content

```python
@self.server.resource_template("magaldi://file/{scope}/{repository}/{path:path}")
async def get_file_resource(scope: str, repository: str, path: str) -> Resource:
    """Get file summary and structure as a resource"""

    file_data = await get_file_summary(path, scope, repository)

    content = f"""# File: {path}

**Language:** {file_data['file']['language']}
**Lines:** {file_data['file']['line_count']}

## Summary
{file_data['file']['summary'] or 'No summary available'}

## Structure
"""

    def render_tree(nodes, indent=0):
        result = ""
        for node in nodes:
            prefix = "  " * indent
            sig = f" - `{node['signature']}`" if node.get('signature') else ""
            result += f"{prefix}- **{node['type']}** `{node['name']}`{sig}\n"
            if node.get('summary'):
                result += f"{prefix}  {node['summary']}\n"
            if node.get('children'):
                result += render_tree(node['children'], indent + 1)
        return result

    content += render_tree(file_data['structure'])

    return Resource(
        uri=f"magaldi://file/{scope}/{repository}/{path}",
        name=path,
        mimeType="text/markdown",
        text=content
    )
```

### List Resources

```python
@self.server.list_resources()
async def list_resources() -> list[Resource]:
    """List available repository resources"""

    repos = await list_repos()

    return [
        Resource(
            uri=f"magaldi://repo/{r['scope']}/{r['repository']}",
            name=f"{r['scope']}/{r['repository']}",
            description=r.get('description', f"{r['file_count']} files, {r['element_count']} elements"),
            mimeType="text/markdown"
        )
        for r in repos
    ]
```

---

## 7.4 Prompts & Skills

### Purpose

Provide pre-built prompts for common code discovery workflows.

### Prompt: Explore Repository

```python
def _register_prompts(self):
    """Register MCP prompts"""

    @self.server.prompt("explore-repo")
    async def explore_repo_prompt(
        scope: str,
        repository: str
    ) -> GetPromptResult:
        """
        Generate a prompt for exploring a repository's structure and purpose.
        """

        # Get repo overview
        repo_resource = await get_repo_resource(scope, repository)

        return GetPromptResult(
            description=f"Explore the {scope}/{repository} repository",
            messages=[
                PromptMessage(
                    role="user",
                    content=TextContent(
                        type="text",
                        text=f"""I'd like to understand the structure and purpose of this repository.

{repo_resource.text}

Please analyze this repository and tell me:
1. What is the main purpose of this codebase?
2. What are the key components/modules?
3. What patterns or architecture does it follow?
4. Where should I start if I want to understand the core functionality?
"""
                    )
                )
            ]
        )
```

### Prompt: Find Implementation

```python
@self.server.prompt("find-implementation")
async def find_implementation_prompt(
    feature: str,
    scope: Optional[str] = None,
    repository: Optional[str] = None
) -> GetPromptResult:
    """
    Generate a prompt for finding where a feature is implemented.
    """

    # Pre-search to provide context
    results = await search_code(
        query=feature,
        scope=scope,
        repository=repository,
        limit=10
    )

    results_text = ""
    for r in results:
        results_text += f"""
- **{r['name']}** ({r['type']}) in `{r['file']}:{r['line']}`
  {r['summary'] or 'No summary'}
"""

    return GetPromptResult(
        description=f"Find implementation of: {feature}",
        messages=[
            PromptMessage(
                role="user",
                content=TextContent(
                    type="text",
                    text=f"""I need to find where "{feature}" is implemented in the codebase.

Here are the most relevant code elements I found:
{results_text}

Please analyze these results and:
1. Identify the main implementation location(s)
2. Explain how the feature works based on the summaries
3. Suggest which files I should read first to understand it
4. Note any related components I should be aware of
"""
                )
            )
        ]
    )
```

### Prompt: Understand Code

```python
@self.server.prompt("understand-code")
async def understand_code_prompt(
    element_id: str
) -> GetPromptResult:
    """
    Generate a prompt for understanding a specific code element.
    """

    # Get full context
    context = await get_context(element_id, include_siblings=True, include_children=True)

    context_text = f"""
## File Context
**File:** {context['element']['file']}
**Summary:** {context['file']['summary'] if context['file'] else 'No file summary'}

"""

    if context['parent']:
        context_text += f"""## Parent: {context['parent']['name']} ({context['parent']['type']})
{context['parent']['summary'] or 'No summary'}

"""

    context_text += f"""## Element: {context['element']['name']} ({context['element']['type']})
**Signature:** `{context['element'].get('signature', 'N/A')}`
**Lines:** {context['element']['line_start']}-{context['element']['line_end']}

**Summary:**
{context['element']['summary'] or 'No summary'}

**Docstring:**
{context['element'].get('docstring') or 'No docstring'}
"""

    if context['children']:
        context_text += "\n## Children:\n"
        for c in context['children']:
            context_text += f"- `{c['name']}` ({c['type']}): {c['summary'] or 'No summary'}\n"

    if context['siblings']:
        context_text += "\n## Siblings (same parent):\n"
        for s in context['siblings']:
            context_text += f"- `{s['name']}` ({s['type']}): {s['summary'] or 'No summary'}\n"

    return GetPromptResult(
        description=f"Understand: {context['element']['name']}",
        messages=[
            PromptMessage(
                role="user",
                content=TextContent(
                    type="text",
                    text=f"""Help me understand this code element:

{context_text}

Please explain:
1. What does this {context['element']['type']} do?
2. How does it fit into the larger system (based on file/parent context)?
3. What are the key responsibilities?
4. Are there any important relationships with siblings or children?
"""
                )
            )
        ]
    )
```

---

## Output

Phase 7 produces:

```python
# Running MCP server accessible to Claude Code
# Tools available:
#   - search_code: Semantic code search
#   - find_similar: Find similar code elements
#   - get_context: Get hierarchical context
#   - list_repos: List available repositories
#   - get_file_summary: Get file structure

# Resources available:
#   - magaldi://repo/{scope}/{repository}
#   - magaldi://file/{scope}/{repository}/{path}

# Prompts available:
#   - explore-repo: Understand repository structure
#   - find-implementation: Find where feature is implemented
#   - understand-code: Deep dive into code element
```

---

## Progress Reporting

```
[MCP Server]
Starting Magaldi MCP Server...
  Connecting to MySQL...              ✓
  Connecting to Elasticsearch...      ✓
  Connecting to Ollama...             ✓
  Verifying embedding model...        ✓ snowflake-arctic-embed2

Registered tools:
  • search_code                       semantic code search
  • find_similar                      find similar elements
  • get_context                       hierarchical context
  • list_repos                        list repositories
  • get_file_summary                  file structure

Registered resources:
  • magaldi://repo/{scope}/{repo}     repository overview
  • magaldi://file/{scope}/{repo}/{path}  file details

Registered prompts:
  • explore-repo                      explore repository
  • find-implementation               find feature implementation
  • understand-code                   understand code element

Server ready on stdio
```

---

## Error Handling

| Error | Action |
|-------|--------|
| Database unavailable | Return error, suggest retry |
| Elasticsearch unavailable | Return error, suggest retry |
| Ollama unavailable | Fall back to keyword search only |
| Element not found | Return clear error with element_id |
| Invalid scope/repo | Return list of valid options |
| Query too broad | Return top results with warning |
| Timeout | Return partial results if available |

### Error Response Format

```python
class MCPError(Exception):
    """Base error for MCP operations"""

    def __init__(self, code: str, message: str, details: Optional[dict] = None):
        self.code = code
        self.message = message
        self.details = details or {}

    def to_response(self) -> dict:
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "details": self.details
            }
        }

# Usage:
raise MCPError(
    code="ELEMENT_NOT_FOUND",
    message=f"Element not found: {element_id}",
    details={"element_id": element_id, "suggestion": "Use list_repos to find valid repositories"}
)
```

---

## Performance Considerations

| Operation | Bottleneck | Optimization |
|-----------|------------|--------------|
| Query embedding | Ollama | Cache recent queries |
| Semantic search | ES script_score | Limit to relevant repos |
| Context loading | MySQL joins | Eager load common paths |
| Resource rendering | String building | Cache rendered resources |

### Query Cache

```python
from functools import lru_cache
import hashlib

class QueryCache:
    """Cache for query embeddings"""

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 300):
        self.cache = {}
        self.max_size = max_size
        self.ttl = ttl_seconds

    def get_embedding(self, query: str) -> Optional[list[float]]:
        key = hashlib.md5(query.encode()).hexdigest()
        if key in self.cache:
            entry = self.cache[key]
            if time.time() - entry['time'] < self.ttl:
                return entry['embedding']
            del self.cache[key]
        return None

    def set_embedding(self, query: str, embedding: list[float]):
        key = hashlib.md5(query.encode()).hexdigest()
        if len(self.cache) >= self.max_size:
            # Remove oldest
            oldest = min(self.cache.items(), key=lambda x: x[1]['time'])
            del self.cache[oldest[0]]
        self.cache[key] = {'embedding': embedding, 'time': time.time()}
```

---

## CLI Interface

```bash
# Start MCP server (stdio mode for Claude Code)
magaldi-mcp serve

# Start with custom config
magaldi-mcp serve --config /path/to/config.yaml

# Test tools directly
magaldi-mcp test search_code --query "authentication" --scope backend

# Health check
magaldi-mcp health

# List registered tools
magaldi-mcp tools
```

---

## Summary of Decisions

| Decision | Value |
|----------|-------|
| MCP transport | stdio (Claude Code standard) |
| Language | Python (mcp SDK) |
| Search type | Hybrid (semantic + keyword) |
| User overlay | Always include main + user branch |
| Query embedding | Ollama (same model as indexing) |
| Cache strategy | LRU cache for query embeddings |
| Resource format | Markdown |
| Error format | Structured with code + message + details |
| Max results | 50 (configurable) |
| Default results | 10 |
