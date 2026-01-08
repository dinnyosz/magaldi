"""Magaldi MCP Server - Code Discovery for Claude Code.

Exposes semantic search and code navigation tools via the Model Context Protocol.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from magaldi.config import get_config, load_config
from magaldi.db.elasticsearch import ElasticsearchRepository
from magaldi.embedding.embedding import OllamaEmbedClient

log = logging.getLogger(__name__)


class MagaldiMCPServer:
    """MCP Server for Magaldi code discovery."""

    def __init__(self, default_username: str = "main") -> None:
        self.server = Server("magaldi")
        self.config = get_config()
        self.default_username = default_username
        self.es_repo: ElasticsearchRepository | None = None
        self.ollama_embed: OllamaEmbedClient | None = None

        # Register handlers
        self._register_tools()

    def _get_es(self) -> ElasticsearchRepository:
        """Get or create Elasticsearch repository."""
        if self.es_repo is None:
            self.es_repo = ElasticsearchRepository(self.config)
        return self.es_repo

    def _get_ollama(self) -> OllamaEmbedClient:
        """Get or create Ollama embedding client."""
        if self.ollama_embed is None:
            ollama_config = self.config.ollama
            self.ollama_embed = OllamaEmbedClient(
                url=ollama_config.url,
                model=ollama_config.embed_model,
            )
        return self.ollama_embed

    def _register_tools(self) -> None:
        """Register all MCP tools."""

        @self.server.list_tools()
        async def list_tools() -> list[Tool]:
            """List available tools."""
            return [
                Tool(
                    name="search_code",
                    description="Semantic search for code elements using natural language. "
                    "Find functions, classes, methods by describing what they do.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Natural language search query (e.g., 'authentication handler', 'database connection')",
                            },
                            "scope": {
                                "type": "string",
                                "description": "Filter by scope (optional)",
                            },
                            "repository": {
                                "type": "string",
                                "description": "Filter by repository name (optional)",
                            },
                            "username": {
                                "type": "string",
                                "description": "User branch to search (default: 'main')",
                                "default": "main",
                            },
                            "element_types": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Filter by element types: file, class, function, method (optional)",
                            },
                            "language": {
                                "type": "string",
                                "description": "Filter by programming language (optional)",
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum results (1-50, default: 20)",
                                "default": 20,
                            },
                            "include_code": {
                                "type": "boolean",
                                "description": "Include source code in results (default: false)",
                                "default": False,
                            },
                        },
                        "required": ["query"],
                    },
                ),
                Tool(
                    name="search_features",
                    description="Search for code features/capabilities. Features are groups of "
                    "related functions that implement a common functionality.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query for features (e.g., 'authentication', 'data validation')",
                            },
                            "scope": {
                                "type": "string",
                                "description": "Filter by scope (optional)",
                            },
                            "repository": {
                                "type": "string",
                                "description": "Filter by repository (optional)",
                            },
                            "username": {
                                "type": "string",
                                "description": "User branch (default: 'main')",
                                "default": "main",
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum results (default: 10)",
                                "default": 10,
                            },
                        },
                        "required": ["query"],
                    },
                ),
                Tool(
                    name="find_similar",
                    description="Find code elements similar to a given element. "
                    "Useful for finding related implementations or patterns.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "element_id": {
                                "type": "string",
                                "description": "Element ID to find similar code for",
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum results (default: 10)",
                                "default": 10,
                            },
                            "same_repo_only": {
                                "type": "boolean",
                                "description": "Only search within same repository (default: false)",
                                "default": False,
                            },
                        },
                        "required": ["element_id"],
                    },
                ),
                Tool(
                    name="get_element",
                    description="Get full details of a code element by its ID.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "element_id": {
                                "type": "string",
                                "description": "The element ID",
                            },
                            "include_code": {
                                "type": "boolean",
                                "description": "Include raw source code (default: false)",
                                "default": False,
                            },
                        },
                        "required": ["element_id"],
                    },
                ),
                Tool(
                    name="get_context",
                    description="Get hierarchical context for a code element. "
                    "Returns file summary, parent class/function, and optionally children/siblings.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "element_id": {
                                "type": "string",
                                "description": "Element ID to get context for",
                            },
                            "include_siblings": {
                                "type": "boolean",
                                "description": "Include sibling elements (default: false)",
                                "default": False,
                            },
                            "include_children": {
                                "type": "boolean",
                                "description": "Include child elements (default: true)",
                                "default": True,
                            },
                        },
                        "required": ["element_id"],
                    },
                ),
                Tool(
                    name="get_file_structure",
                    description="Get the structure of a file with all contained elements.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "scope": {
                                "type": "string",
                                "description": "Repository scope",
                            },
                            "repository": {
                                "type": "string",
                                "description": "Repository name",
                            },
                            "file_path": {
                                "type": "string",
                                "description": "Relative file path",
                            },
                            "username": {
                                "type": "string",
                                "description": "User branch (default: 'main')",
                                "default": "main",
                            },
                        },
                        "required": ["scope", "repository", "file_path"],
                    },
                ),
                Tool(
                    name="list_repos",
                    description="List all indexed repositories.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "scope": {
                                "type": "string",
                                "description": "Filter by scope (optional)",
                            },
                        },
                        "required": [],
                    },
                ),
                Tool(
                    name="list_features",
                    description="List all extracted features for a repository.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "scope": {
                                "type": "string",
                                "description": "Repository scope",
                            },
                            "repository": {
                                "type": "string",
                                "description": "Repository name",
                            },
                            "username": {
                                "type": "string",
                                "description": "User branch (default: 'main')",
                                "default": "main",
                            },
                        },
                        "required": ["scope", "repository"],
                    },
                ),
                Tool(
                    name="get_repo_stats",
                    description="Get statistics for a repository.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "scope": {
                                "type": "string",
                                "description": "Repository scope",
                            },
                            "repository": {
                                "type": "string",
                                "description": "Repository name",
                            },
                            "username": {
                                "type": "string",
                                "description": "User branch (default: 'main')",
                                "default": "main",
                            },
                        },
                        "required": ["scope", "repository"],
                    },
                ),
                Tool(
                    name="get_children",
                    description="Get child elements of a parent element.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "element_id": {
                                "type": "string",
                                "description": "Parent element ID",
                            },
                        },
                        "required": ["element_id"],
                    },
                ),
                Tool(
                    name="get_feature_members",
                    description="Get all members of a feature cluster.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "feature_id": {
                                "type": "string",
                                "description": "Feature ID",
                            },
                        },
                        "required": ["feature_id"],
                    },
                ),
            ]

        @self.server.call_tool()
        async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
            """Handle tool calls."""
            try:
                result = await self._handle_tool(name, arguments)
                return [TextContent(type="text", text=_format_result(result))]
            except Exception as e:
                log.exception(f"Tool {name} failed")
                return [TextContent(type="text", text=f"Error: {e}")]

    async def _handle_tool(self, name: str, args: dict[str, Any]) -> Any:
        """Dispatch tool call to implementation."""
        from magaldi.mcp.tools import (
            find_similar,
            get_children,
            get_context,
            get_element,
            get_feature_members,
            get_file_structure,
            get_repo_stats,
            list_features,
            list_repos,
            search_code,
            search_features,
        )

        es = self._get_es()
        ollama = self._get_ollama()

        if name == "search_code":
            return await asyncio.to_thread(
                search_code,
                es,
                ollama,
                query=args["query"],
                scope=args.get("scope"),
                repository=args.get("repository"),
                username=args.get("username", self.default_username),
                element_types=args.get("element_types"),
                language=args.get("language"),
                limit=args.get("limit", 20),
                include_code=args.get("include_code", False),
            )
        elif name == "search_features":
            return await asyncio.to_thread(
                search_features,
                es,
                ollama,
                query=args["query"],
                scope=args.get("scope"),
                repository=args.get("repository"),
                username=args.get("username", self.default_username),
                limit=args.get("limit", 10),
            )
        elif name == "find_similar":
            return await asyncio.to_thread(
                find_similar,
                es,
                element_id=args["element_id"],
                limit=args.get("limit", 10),
                same_repo_only=args.get("same_repo_only", False),
            )
        elif name == "get_element":
            return await asyncio.to_thread(
                get_element,
                es,
                element_id=args["element_id"],
                include_code=args.get("include_code", False),
            )
        elif name == "get_context":
            return await asyncio.to_thread(
                get_context,
                es,
                element_id=args["element_id"],
                include_siblings=args.get("include_siblings", False),
                include_children=args.get("include_children", True),
            )
        elif name == "get_file_structure":
            return await asyncio.to_thread(
                get_file_structure,
                es,
                scope=args["scope"],
                repository=args["repository"],
                file_path=args["file_path"],
                username=args.get("username", self.default_username),
            )
        elif name == "list_repos":
            return await asyncio.to_thread(
                list_repos,
                es,
                scope=args.get("scope"),
            )
        elif name == "list_features":
            return await asyncio.to_thread(
                list_features,
                es,
                scope=args["scope"],
                repository=args["repository"],
                username=args.get("username", self.default_username),
            )
        elif name == "get_repo_stats":
            return await asyncio.to_thread(
                get_repo_stats,
                es,
                scope=args["scope"],
                repository=args["repository"],
                username=args.get("username", self.default_username),
            )
        elif name == "get_children":
            return await asyncio.to_thread(
                get_children,
                es,
                element_id=args["element_id"],
            )
        elif name == "get_feature_members":
            return await asyncio.to_thread(
                get_feature_members,
                es,
                feature_id=args["feature_id"],
            )
        else:
            raise ValueError(f"Unknown tool: {name}")

    async def run(self) -> None:
        """Run the MCP server on stdio."""
        log.info("Starting Magaldi MCP Server...")

        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options(),
            )


def _format_result(result: Any) -> str:
    """Format tool result as readable text."""
    import json

    # Format search results as readable text
    if isinstance(result, list) and result:
        first = result[0]

        # Code search results
        if "element_id" in first and "type" in first:
            lines = [f"Found {len(result)} results:\n"]
            for r in result:
                loc = f"{r.get('file', '?')}:{r.get('line', '?')}" if r.get('file') else "N/A"
                lines.append(f"[{r.get('type', '?')}] {r.get('name', '?')} ({loc})")
                if r.get('signature'):
                    lines.append(f"  {r['signature']}")
                if r.get('summary'):
                    # Truncate long summaries
                    summary = r['summary']
                    if len(summary) > 200:
                        summary = summary[:200] + "..."
                    lines.append(f"  {summary}")
                if r.get('code'):
                    # Include code block
                    lines.append(f"  ```")
                    lines.append(r['code'])
                    lines.append(f"  ```")
                lines.append("")
            return "\n".join(lines)

        # Feature search results
        if "feature_id" in first:
            lines = [f"Found {len(result)} features:\n"]
            for r in result:
                lines.append(f"[feature] {r.get('label', '?')} ({r.get('member_count', 0)} members)")
                if r.get('summary'):
                    summary = r['summary']
                    if len(summary) > 200:
                        summary = summary[:200] + "..."
                    lines.append(f"  {summary}")
                lines.append("")
            return "\n".join(lines)

        # Repository list
        if "element_count" in first and "scope" in first:
            lines = ["Indexed repositories:\n"]
            for r in result:
                lines.append(f"  {r.get('scope')}/{r.get('repository')}: {r.get('element_count')} elements, {r.get('file_count')} files")
            return "\n".join(lines)

    # Single dict results (get_element, get_context, etc.)
    if isinstance(result, dict):
        # Element details
        if "element_id" in result and "type" in result:
            lines = [f"[{result.get('type')}] {result.get('name')}"]
            lines.append(f"  File: {result.get('file')}:{result.get('line_start')}-{result.get('line_end')}")
            if result.get('signature'):
                lines.append(f"  Signature: {result['signature']}")
            if result.get('summary'):
                lines.append(f"  Summary: {result['summary']}")
            if result.get('code'):
                lines.append(f"  Code:\n{result['code']}")
            return "\n".join(lines)

        # Stats
        if "elements_by_type" in result:
            lines = [f"Repository stats:"]
            lines.append(f"  Total elements: {result.get('total_elements')}")
            lines.append(f"  Total lines: {result.get('total_lines')}")
            lines.append(f"  Features: {result.get('feature_count')}")
            lines.append(f"  By type: {result.get('elements_by_type')}")
            return "\n".join(lines)

        # Fallback to JSON for other dicts
        return json.dumps(result, indent=2, default=str)

    if isinstance(result, list):
        return json.dumps(result, indent=2, default=str)

    return str(result)


def run_server() -> None:
    """Entry point to run the MCP server."""
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Magaldi MCP Server")
    parser.add_argument(
        "--user", "-u",
        default=os.environ.get("MAGALDI_USER", "main"),
        help="Default username for searches (default: MAGALDI_USER env or 'main')",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Load configuration first
    load_config()

    server = MagaldiMCPServer(default_username=args.user)
    asyncio.run(server.run())
