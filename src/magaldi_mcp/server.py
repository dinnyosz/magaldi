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

from shared.config import get_config, load_config
from shared.db.elasticsearch import ElasticsearchRepository
from shared.ai.embedding import CodeEmbeddingClient

log = logging.getLogger(__name__)


class MagaldiMCPServer:
    """MCP Server for Magaldi code discovery."""

    def __init__(
        self,
        default_username: str = "main",
    ) -> None:
        self.server = Server("magaldi")
        self.config = get_config()
        self.default_username = default_username
        self.es_repo: ElasticsearchRepository | None = None
        self.embed_client: CodeEmbeddingClient | None = None

        # Register handlers
        self._register_tools()

    def _get_es(self) -> ElasticsearchRepository:
        """Get or create Elasticsearch repository."""
        if self.es_repo is None:
            self.es_repo = ElasticsearchRepository(self.config)
        return self.es_repo

    def _get_embed_client(self) -> CodeEmbeddingClient:
        """Get or create embedding client."""
        if self.embed_client is None:
            llm_config = self.config.llm
            self.embed_client = CodeEmbeddingClient(
                url=llm_config.url,
                model=llm_config.embed_model,
                provider=llm_config.embed_provider or llm_config.provider,
                api_key=llm_config.embed_api_key or llm_config.api_key,
            )
        return self.embed_client

    def _register_tools(self) -> None:
        """Register all MCP tools."""

        @self.server.list_tools()
        async def list_tools() -> list[Tool]:
            """List available tools."""
            return [
                # =============================================================
                # SEARCH - Start here to find code
                # =============================================================
                Tool(
                    name="search_code",
                    description="FIND CODE: Search for functions, classes, methods by what they do. "
                    "Uses pre-indexed semantic embeddings - finds 'login' when you search 'authentication'. "
                    "Returns AI summaries so you understand code without reading it. "
                    "Use include_code=true to see implementation. Use brief=true for exploration.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "What the code does (e.g., 'handle authentication', 'parse JSON', 'validate email')",
                            },
                            "element_types": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Filter: file, class, function, method",
                            },
                            "include_code": {
                                "type": "boolean",
                                "description": "Include source code in results",
                                "default": False,
                            },
                            "brief": {
                                "type": "boolean",
                                "description": "Minimal output (name, file, line only) - use for broad exploration",
                                "default": False,
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Max results (default: 20)",
                                "default": 20,
                            },
                            "repository": {"type": "string", "description": "Filter by repo"},
                            "scope": {"type": "string", "description": "Filter by scope"},
                            "include_tests": {
                                "type": "boolean",
                                "description": "Include test elements in results. Default: true.",
                                "default": True,
                            },
                        },
                        "required": ["query"],
                    },
                ),
                Tool(
                    name="search_features",
                    description="FIND CAPABILITIES: Search for high-level features (groups of related functions). "
                    "Pre-clustered by AI - 'authentication' returns all auth-related functions grouped together. "
                    "Use to understand what the codebase CAN DO, not just specific implementations.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "What capability you need (e.g., 'authentication', 'caching', 'validation')",
                            },
                            "limit": {"type": "integer", "default": 20},
                            "repository": {"type": "string"},
                            "scope": {"type": "string"},
                            "glossary_term": {
                                "type": "string",
                                "description": "Filter to features where this glossary term appears in members",
                            },
                            "min_percentage": {
                                "type": "number",
                                "description": "Minimum percentage of members containing the term (0-100)",
                                "default": 0,
                            },
                        },
                        "required": ["query"],
                    },
                ),
                Tool(
                    name="find_similar",
                    description="FIND RELATED CODE: Given an element_id, find similar implementations. "
                    "Use after search_code to find related patterns or alternative approaches.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "element_id": {"type": "string", "description": "Element ID from search results"},
                            "limit": {"type": "integer", "default": 10},
                            "same_repo_only": {"type": "boolean", "default": False},
                            "include_tests": {
                                "type": "boolean",
                                "description": "Include test elements in results. Default: true.",
                                "default": True,
                            },
                        },
                        "required": ["element_id"],
                    },
                ),
                # =============================================================
                # INSPECT - Look at specific code
                # =============================================================
                Tool(
                    name="get_element",
                    description="INSPECT ELEMENT: Get full details of a specific element by ID. "
                    "Use after search_code to see complete info. Use include_code=true for source.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "element_id": {"type": "string", "description": "Element ID from search results"},
                            "include_code": {"type": "boolean", "default": False},
                        },
                        "required": ["element_id"],
                    },
                ),
                Tool(
                    name="batch_get_elements",
                    description="INSPECT MULTIPLE: Get several elements at once by their IDs. "
                    "More efficient than multiple get_element calls.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "element_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of element IDs",
                            },
                            "include_code": {"type": "boolean", "default": False},
                        },
                        "required": ["element_ids"],
                    },
                ),
                Tool(
                    name="get_context",
                    description="UNDERSTAND CONTEXT: See where an element fits - its parent class, "
                    "siblings, and children. Use to understand code structure around a function.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "element_id": {"type": "string"},
                            "include_children": {"type": "boolean", "default": True},
                            "include_siblings": {"type": "boolean", "default": False},
                        },
                        "required": ["element_id"],
                    },
                ),
                Tool(
                    name="get_children",
                    description="LIST CHILDREN: Get all child elements (methods in a class, etc). "
                    "Use to explore what a class contains.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "element_id": {"type": "string", "description": "Parent element ID"},
                        },
                        "required": ["element_id"],
                    },
                ),
                # =============================================================
                # FILES - Work with indexed file data
                # =============================================================
                Tool(
                    name="find_files",
                    description="FIND FILES: Search for files by glob pattern. "
                    "USE THIS instead of built-in Glob - searches indexed codebase. "
                    "Discovers file structure (e.g., '**/*.py', 'src/**/*.ts').",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "pattern": {"type": "string", "description": "Glob pattern"},
                            "scope": {"type": "string", "description": "Filter by scope"},
                            "repository": {"type": "string", "description": "Filter by repo"},
                            "limit": {"type": "integer", "default": 50},
                        },
                        "required": ["pattern"],
                    },
                ),
                Tool(
                    name="get_file_structure",
                    description="FILE OVERVIEW: Get the structure of a file (all classes, functions, methods). "
                    "Use to understand what's in a file without reading it.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "file_path": {"type": "string", "description": "Relative file path"},
                            "scope": {"type": "string"},
                            "repository": {"type": "string"},
                        },
                        "required": ["scope", "repository", "file_path"],
                    },
                ),
                # =============================================================
                # FEATURES - High-level code organization
                # =============================================================
                Tool(
                    name="list_features",
                    description="LIST ALL FEATURES: See all extracted features/capabilities in a repo. "
                    "Use to get a high-level understanding of what the codebase does.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "scope": {"type": "string"},
                            "repository": {"type": "string"},
                        },
                        "required": ["scope", "repository"],
                    },
                ),
                Tool(
                    name="get_feature_members",
                    description="FEATURE DETAILS: Get all functions that belong to a feature. "
                    "Use after list_features or search_features to see implementations.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "feature_id": {"type": "string"},
                        },
                        "required": ["feature_id"],
                    },
                ),
                # =============================================================
                # META - Repository information
                # =============================================================
                Tool(
                    name="list_repos",
                    description="LIST REPOSITORIES: See all indexed codebases.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "scope": {"type": "string"},
                        },
                        "required": [],
                    },
                ),
                Tool(
                    name="get_repo_stats",
                    description="REPO OVERVIEW: Get statistics (element counts, languages, etc).",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "scope": {"type": "string"},
                            "repository": {"type": "string"},
                        },
                        "required": ["scope", "repository"],
                    },
                ),
                # =============================================================
                # CODE SEARCH - Regex/grep-based search
                # =============================================================
                Tool(
                    name="grep_code",
                    description="[DEPRECATED: Use pattern_search instead] "
                    "GREP CODE: Search with regex pattern (like ripgrep). "
                    "USE pattern_search for better performance - queries run server-side.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "pattern": {"type": "string", "description": "Regex pattern to search"},
                            "scope": {"type": "string", "description": "Filter by scope"},
                            "repository": {"type": "string", "description": "Filter by repo"},
                            "glob": {"type": "string", "description": "File filter (e.g., '*.py', 'src/**/*.ts')"},
                            "context_lines": {"type": "integer", "default": 0, "description": "Lines of context around match"},
                            "limit": {"type": "integer", "default": 50},
                            "include_tests": {
                                "type": "boolean",
                                "description": "Include test elements in results. Default: true.",
                                "default": True,
                            },
                        },
                        "required": ["pattern"],
                    },
                ),
                Tool(
                    name="pattern_search",
                    description="PATTERN SEARCH: ES-native pattern matching on code. "
                    "Faster than grep_code - query runs server-side. "
                    "Three modes: regexp (Lucene syntax), wildcard (* and ?), proximity (terms near each other).",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "pattern": {
                                "type": "string",
                                "description": "Search pattern. Syntax depends on mode.",
                            },
                            "mode": {
                                "type": "string",
                                "enum": ["regexp", "wildcard", "proximity"],
                                "description": "regexp: Lucene regex (e.g., 'add_column.*Model'). "
                                "wildcard: Simple wildcards (e.g., '*column*'). "
                                "proximity: Terms near each other (e.g., 'add column Model').",
                            },
                            "scope": {"type": "string", "description": "Filter by scope (required)"},
                            "repository": {"type": "string", "description": "Filter by repo (required)"},
                            "username": {"type": "string", "description": "User branch to search"},
                            "slop": {
                                "type": "integer",
                                "default": 5,
                                "description": "For proximity mode: max word distance",
                            },
                            "glob": {"type": "string", "description": "File filter (e.g., '*.py')"},
                            "limit": {"type": "integer", "default": 50},
                            "include_tests": {"type": "boolean", "default": True},
                        },
                        "required": ["pattern", "mode", "scope", "repository"],
                    },
                ),
                Tool(
                    name="find_usages",
                    description="FIND USAGES: Find where a function/class/method is called or referenced. "
                    "USE THIS instead of grepping for 'functionName(' - automatically filters definitions, "
                    "includes context lines, and understands code structure.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "element_id": {"type": "string", "description": "Element ID from search results"},
                            "limit": {"type": "integer", "default": 30},
                        },
                        "required": ["element_id"],
                    },
                ),
                Tool(
                    name="find_implementations",
                    description="FIND IMPLEMENTATIONS: Find classes that inherit from or implement a protocol/base class. "
                    "USE THIS instead of grepping for 'class.*BaseClass' - understands inheritance patterns.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "element_id": {"type": "string", "description": "Element ID of the protocol/base class"},
                            "class_name": {"type": "string", "description": "Or just the class name to search for"},
                            "scope": {"type": "string", "description": "Filter by scope"},
                            "repository": {"type": "string", "description": "Filter by repo"},
                            "limit": {"type": "integer", "default": 20},
                        },
                        "required": [],
                    },
                ),
                Tool(
                    name="get_call_graph",
                    description="CALL GRAPH: Get callers (who calls this) and callees (what it calls) for a function. "
                    "Pre-computed from indexed code - instant dependency analysis for refactoring impact.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "element_id": {"type": "string", "description": "Function/method element ID"},
                            "direction": {"type": "string", "enum": ["callers", "callees", "both"], "default": "both"},
                        },
                        "required": ["element_id"],
                    },
                ),
                Tool(
                    name="find_callers",
                    description="FIND CALLERS: Find all functions that call a given function/method. "
                    "Uses indexed call data for instant results. Returns callers grouped by code/tests.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "element_id": {"type": "string", "description": "Target element ID to find callers of"},
                            "scope": {"type": "string", "description": "Filter by scope"},
                            "repository": {"type": "string", "description": "Filter by repository"},
                            "username": {"type": "string", "description": "Filter by username branch"},
                            "limit": {"type": "integer", "default": 30, "description": "Max results"},
                            "include_tests": {"type": "boolean", "default": True, "description": "Include test functions"},
                        },
                        "required": ["element_id"],
                    },
                ),
                Tool(
                    name="find_call_chain",
                    description="CALL CHAIN: Trace call chains from an element. "
                    "Shows what a function calls (callees) or what calls it (callers) recursively. "
                    "Use for impact analysis before refactoring.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "element_id": {"type": "string", "description": "Starting element ID"},
                            "direction": {
                                "type": "string",
                                "enum": ["callers", "callees", "both"],
                                "default": "callees",
                                "description": "callers: what calls this, callees: what this calls, both: both directions",
                            },
                            "max_depth": {"type": "integer", "default": 5, "description": "Max depth to traverse (1-10)"},
                            "scope": {"type": "string", "description": "Filter by scope"},
                            "repository": {"type": "string", "description": "Filter by repository"},
                            "username": {"type": "string", "description": "Filter by username branch"},
                        },
                        "required": ["element_id"],
                    },
                ),
                Tool(
                    name="find_dead_code",
                    description="DEAD CODE: Find functions/methods that are never called. "
                    "Excludes entry points (routes, CLI commands), magic methods, and main functions. "
                    "Use for codebase cleanup.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "scope": {"type": "string", "description": "Repository scope (required)"},
                            "repository": {"type": "string", "description": "Repository name (required)"},
                            "username": {"type": "string", "description": "Username branch"},
                            "include_tests": {"type": "boolean", "default": False, "description": "Include test functions in check"},
                        },
                        "required": ["scope", "repository"],
                    },
                ),
                Tool(
                    name="find_entry_points",
                    description="ENTRY POINTS: Find HTTP handlers, CLI commands, test fixtures, main functions. "
                    "Detects entry points by decorator patterns (@route, @command, @fixture) and naming. "
                    "Returns grouped by type.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "scope": {"type": "string", "description": "Repository scope (required)"},
                            "repository": {"type": "string", "description": "Repository name (required)"},
                            "username": {"type": "string", "description": "Username branch"},
                        },
                        "required": ["scope", "repository"],
                    },
                ),
                # =============================================================
                # META - Self-documentation
                # =============================================================
                Tool(
                    name="generate_skill",
                    description="GENERATE SKILL: Create a SKILL.md file that teaches LLMs how to use this MCP effectively. "
                    "The skill file documents best practices, workflows, and anti-patterns for token-efficient code discovery. "
                    "ASK THE USER whether to install globally (~/.claude/skills) or project-local (.claude/skills).",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "project_root": {"type": "string", "description": "Project root directory (required for scope='project')"},
                            "skill_name": {"type": "string", "default": "magaldi", "description": "Name of the skill directory"},
                            "scope": {
                                "type": "string",
                                "enum": ["project", "global"],
                                "default": "project",
                                "description": "Where to install: 'project' (this project only) or 'global' (all projects)",
                            },
                        },
                        "required": [],
                    },
                ),
                # =============================================================
                # GLOSSARY - Domain terminology discovery
                # =============================================================
                Tool(
                    name="list_glossary",
                    description="LIST GLOSSARY: List all glossary terms for a repository. "
                    "Shows domain concepts extracted from code element names. "
                    "Use to discover what terminology exists in the codebase.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "scope": {"type": "string", "description": "Filter by scope"},
                            "repository": {"type": "string", "description": "Filter by repo"},
                            "min_count": {"type": "integer", "default": 1, "description": "Minimum occurrence count"},
                        },
                        "required": ["scope", "repository"],
                    },
                ),
                Tool(
                    name="get_glossary_term",
                    description="GET GLOSSARY TERM: Get full details for a specific glossary term. "
                    "Shows element IDs, file paths, and feature associations for the term.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "scope": {"type": "string"},
                            "repository": {"type": "string"},
                            "term": {"type": "string", "description": "The glossary term to look up"},
                        },
                        "required": ["scope", "repository", "term"],
                    },
                ),
                Tool(
                    name="search_glossary",
                    description="SEARCH GLOSSARY: Search glossary terms by partial match. "
                    "Use to find all terms related to a concept (e.g., 'user' matches 'user', 'username').",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "scope": {"type": "string"},
                            "repository": {"type": "string"},
                            "query": {"type": "string", "description": "Partial term to search for"},
                        },
                        "required": ["scope", "repository", "query"],
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
        from magaldi_mcp.tools import (
            batch_get_elements,
            find_call_chain,
            find_callers,
            find_dead_code,
            find_entry_points,
            find_files,
            find_implementations,
            find_similar,
            find_usages,
            generate_skill,
            get_call_graph,
            get_children,
            get_context,
            get_element,
            get_feature_members,
            get_file_structure,
            get_glossary_term,
            get_repo_stats,
            grep_code,
            list_features,
            list_glossary,
            list_repos,
            pattern_search,
            search_code,
            search_features,
            search_glossary,
        )

        es = self._get_es()
        embed_client = self._get_embed_client()

        if name == "search_code":
            return await asyncio.to_thread(
                search_code,
                es,
                embed_client,
                query=args["query"],
                scope=args.get("scope"),
                repository=args.get("repository"),
                username=args.get("username", self.default_username),
                element_types=args.get("element_types"),
                language=args.get("language"),
                limit=args.get("limit", 20),
                include_code=args.get("include_code", False),
                brief=args.get("brief", False),
                include_tests=args.get("include_tests", True),
            )
        elif name == "search_features":
            return await asyncio.to_thread(
                search_features,
                es,
                embed_client,
                query=args["query"],
                scope=args.get("scope"),
                repository=args.get("repository"),
                username=args.get("username", self.default_username),
                limit=args.get("limit", 10),
                glossary_term=args.get("glossary_term"),
                min_percentage=args.get("min_percentage", 0.0),
            )
        elif name == "find_similar":
            return await asyncio.to_thread(
                find_similar,
                es,
                element_id=args["element_id"],
                limit=args.get("limit", 10),
                same_repo_only=args.get("same_repo_only", False),
                include_tests=args.get("include_tests", True),
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
        elif name == "batch_get_elements":
            return await asyncio.to_thread(
                batch_get_elements,
                es,
                element_ids=args["element_ids"],
                include_code=args.get("include_code", False),
            )
        elif name == "find_files":
            # Uses ES - no filesystem access needed
            return await asyncio.to_thread(
                find_files,
                es,
                pattern=args["pattern"],
                scope=args.get("scope"),
                repository=args.get("repository"),
                limit=args.get("limit", 50),
            )
        elif name == "grep_code":
            # Uses ES raw_code field - no filesystem access needed
            return await asyncio.to_thread(
                grep_code,
                es,
                pattern=args["pattern"],
                scope=args.get("scope"),
                repository=args.get("repository"),
                glob=args.get("glob"),
                context_lines=args.get("context_lines", 0),
                limit=args.get("limit", 50),
                include_tests=args.get("include_tests", True),
            )
        elif name == "pattern_search":
            # ES-native pattern search - faster than grep_code
            return await asyncio.to_thread(
                pattern_search,
                es,
                pattern=args["pattern"],
                mode=args["mode"],
                scope=args["scope"],
                repository=args["repository"],
                username=args.get("username"),  # Pass None if not provided
                slop=args.get("slop", 5),
                glob=args.get("glob"),
                limit=args.get("limit", 50),
                include_tests=args.get("include_tests", True),
            )
        elif name == "find_usages":
            # Uses ES - no filesystem access needed
            return await asyncio.to_thread(
                find_usages,
                es,
                element_id=args["element_id"],
                limit=args.get("limit", 30),
            )
        elif name == "find_implementations":
            # Uses ES - no filesystem access needed
            return await asyncio.to_thread(
                find_implementations,
                es,
                element_id=args.get("element_id"),
                class_name=args.get("class_name"),
                scope=args.get("scope"),
                repository=args.get("repository"),
                limit=args.get("limit", 20),
            )
        elif name == "get_call_graph":
            # Uses ES - no filesystem access needed
            return await asyncio.to_thread(
                get_call_graph,
                es,
                element_id=args["element_id"],
                direction=args.get("direction", "both"),
            )
        elif name == "find_callers":
            return await asyncio.to_thread(
                find_callers,
                es,
                element_id=args["element_id"],
                scope=args.get("scope"),
                repository=args.get("repository"),
                username=args.get("username"),
                limit=args.get("limit", 30),
                include_tests=args.get("include_tests", True),
            )
        elif name == "find_call_chain":
            return await asyncio.to_thread(
                find_call_chain,
                es,
                element_id=args["element_id"],
                direction=args.get("direction", "callees"),
                max_depth=args.get("max_depth", 5),
                scope=args.get("scope"),
                repository=args.get("repository"),
                username=args.get("username"),
            )
        elif name == "find_dead_code":
            return await asyncio.to_thread(
                find_dead_code,
                es,
                scope=args["scope"],
                repository=args["repository"],
                username=args.get("username"),
                include_tests=args.get("include_tests", False),
            )
        elif name == "find_entry_points":
            return await asyncio.to_thread(
                find_entry_points,
                es,
                scope=args["scope"],
                repository=args["repository"],
                username=args.get("username"),
            )
        elif name == "generate_skill":
            return await asyncio.to_thread(
                generate_skill,
                project_root=args.get("project_root"),
                skill_name=args.get("skill_name", "magaldi"),
                scope=args.get("scope", "project"),
            )
        elif name == "list_glossary":
            return await asyncio.to_thread(
                list_glossary,
                es,
                scope=args["scope"],
                repository=args["repository"],
                username=args.get("username", self.default_username),
                min_count=args.get("min_count", 1),
            )
        elif name == "get_glossary_term":
            return await asyncio.to_thread(
                get_glossary_term,
                es,
                scope=args["scope"],
                repository=args["repository"],
                term=args["term"],
                username=args.get("username", self.default_username),
            )
        elif name == "search_glossary":
            return await asyncio.to_thread(
                search_glossary,
                es,
                scope=args["scope"],
                repository=args["repository"],
                query=args["query"],
                username=args.get("username", self.default_username),
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

        # File list (find_files)
        if "path" in first and "size" in first and "element_id" not in first:
            lines = [f"Found {len(result)} files:\n"]
            for r in result:
                lines.append(f"  {r.get('path')} ({r.get('size')} bytes)")
            return "\n".join(lines)

        # Grep results
        if "file" in first and "content" in first and "line" in first:
            lines = [f"Found {len(result)} matches:\n"]
            for r in result:
                lines.append(f"{r.get('file')}:{r.get('line')}")
                # Show context before
                for ctx in r.get("context_before", []):
                    lines.append(f"  | {ctx}")
                # Highlight the match line
                lines.append(f"  > {r.get('content')}")
                # Show context after
                for ctx in r.get("context_after", []):
                    lines.append(f"  | {ctx}")
                lines.append("")
            return "\n".join(lines)

        # Usage results (find_usages)
        if "file" in first and "content" in first and "context_before" in first:
            lines = [f"Found {len(result)} usages:\n"]
            for r in result:
                lines.append(f"{r.get('file')}:{r.get('line')}")
                for ctx in r.get("context_before", []):
                    lines.append(f"  | {ctx}")
                lines.append(f"  > {r.get('content')}")
                for ctx in r.get("context_after", []):
                    lines.append(f"  | {ctx}")
                lines.append("")
            return "\n".join(lines)

        # Implementation results
        if "class_name" in first and "definition" in first:
            lines = [f"Found {len(result)} implementations:\n"]
            for r in result:
                lines.append(f"[class] {r.get('class_name')} ({r.get('file')}:{r.get('line')})")
                lines.append(f"  {r.get('definition')}")
                lines.append("")
            return "\n".join(lines)

        # Glossary list results
        if "term" in first and "total_count" in first:
            lines = [f"Found {len(result)} glossary terms:\n"]
            for r in result:
                lines.append(f"  {r.get('term')}: {r.get('total_count')} occurrences")
                if r.get('feature_associations'):
                    assocs = r['feature_associations'][:3]  # Show top 3
                    for a in assocs:
                        lines.append(f"    -> {a.get('feature_label')} ({a.get('percentage'):.0f}%)")
            return "\n".join(lines)

    # Single dict results (get_element, get_context, read_file, etc.)
    if isinstance(result, dict):
        # Grouped search results (code_results and test_results)
        if "code_results" in result and "test_results" in result:
            code_results = result.get("code_results", [])
            test_results = result.get("test_results", [])
            total_code = result.get("total_code", len(code_results))
            total_tests = result.get("total_tests", len(test_results))

            lines = []

            # Format code results
            if code_results:
                lines.append(f"Code Results ({total_code}):\n")
                for r in code_results:
                    loc = f"{r.get('file', '?')}:{r.get('line', '?')}" if r.get('file') else "N/A"
                    lines.append(f"[{r.get('type', '?')}] {r.get('name', '?')} ({loc})")
                    if r.get('signature'):
                        lines.append(f"  {r['signature']}")
                    if r.get('summary'):
                        summary = r['summary']
                        if len(summary) > 200:
                            summary = summary[:200] + "..."
                        lines.append(f"  {summary}")
                    if r.get('code'):
                        lines.append(f"  ```")
                        lines.append(r['code'])
                        lines.append(f"  ```")
                    # For grep results
                    if r.get('content') and r.get('match'):
                        for ctx in r.get("context_before", []):
                            lines.append(f"  | {ctx}")
                        lines.append(f"  > {r.get('content')}")
                        for ctx in r.get("context_after", []):
                            lines.append(f"  | {ctx}")
                    lines.append("")
            else:
                lines.append("No code results.\n")

            # Format test results
            if test_results:
                lines.append(f"Test Results ({total_tests}):\n")
                for r in test_results:
                    loc = f"{r.get('file', '?')}:{r.get('line', '?')}" if r.get('file') else "N/A"
                    lines.append(f"[{r.get('type', '?')}] {r.get('name', '?')} ({loc})")
                    if r.get('signature'):
                        lines.append(f"  {r['signature']}")
                    if r.get('summary'):
                        summary = r['summary']
                        if len(summary) > 200:
                            summary = summary[:200] + "..."
                        lines.append(f"  {summary}")
                    if r.get('code'):
                        lines.append(f"  ```")
                        lines.append(r['code'])
                        lines.append(f"  ```")
                    # For grep results
                    if r.get('content') and r.get('match'):
                        for ctx in r.get("context_before", []):
                            lines.append(f"  | {ctx}")
                        lines.append(f"  > {r.get('content')}")
                        for ctx in r.get("context_after", []):
                            lines.append(f"  | {ctx}")
                    lines.append("")
            elif result.get("total_tests", 0) > 0:
                lines.append(f"(Tests excluded: {total_tests})")

            return "\n".join(lines)

        # Feature members result (get_feature_members)
        if "members" in result and "glossary_terms" in result:
            members = result.get("members", [])
            glossary_terms = result.get("glossary_terms", [])

            lines = []

            # Format members
            if members:
                lines.append(f"Feature Members ({len(members)}):\n")
                for m in members:
                    loc = f"{m.get('file', '?')}:{m.get('line', '?')}" if m.get('file') else "N/A"
                    lines.append(f"[{m.get('type', '?')}] {m.get('name', '?')} ({loc})")
                    if m.get('signature'):
                        lines.append(f"  {m['signature']}")
                    if m.get('summary'):
                        summary = m['summary']
                        if len(summary) > 200:
                            summary = summary[:200] + "..."
                        lines.append(f"  {summary}")
                    lines.append("")
            else:
                lines.append("No members.\n")

            # Format glossary terms
            if glossary_terms:
                lines.append(f"Glossary Terms ({len(glossary_terms)}):\n")
                for t in glossary_terms:
                    lines.append(f"  {t.get('term')}: {t.get('frequency')} occurrences ({t.get('percentage', 0):.1f}%)")
                lines.append("")

            return "\n".join(lines)

        # File read result
        if "content" in result and "path" in result:
            lines = [f"File: {result.get('path')} ({result.get('lines_returned')}/{result.get('total_lines')} lines)"]
            lines.append("```")
            lines.append(result.get("content", ""))
            lines.append("```")
            return "\n".join(lines)

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

        # Call graph (get_call_graph result)
        if "callers" in result and "callees" in result and "element" in result:
            elem = result.get("element", {})
            lines = [f"Call graph for [{elem.get('type')}] {elem.get('name')} ({elem.get('file')}:{elem.get('line')})\n"]

            if result.get("callers"):
                lines.append(f"Callers ({len(result['callers'])}):")
                for c in result["callers"]:
                    loc = f"{c.get('file')}:{c.get('line')}" if c.get('file') else "?"
                    lines.append(f"  [{c.get('type', '?')}] {c.get('name')} ({loc})")
                    if c.get('summary'):
                        summary = c['summary'][:100] + "..." if len(c.get('summary', '')) > 100 else c.get('summary', '')
                        lines.append(f"    {summary}")
                lines.append("")

            if result.get("callees"):
                lines.append(f"Callees ({len(result['callees'])}):")
                for c in result["callees"]:
                    if c.get("element_id"):
                        loc = f"{c.get('file')}:{c.get('target_line')}" if c.get('file') else "?"
                        lines.append(f"  [{c.get('type', '?')}] {c.get('name')} ({loc})")
                        if c.get('summary'):
                            summary = c['summary'][:100] + "..." if len(c.get('summary', '')) > 100 else c.get('summary', '')
                            lines.append(f"    {summary}")
                    else:
                        receiver = f"{c.get('receiver')}." if c.get('receiver') else ""
                        lines.append(f"  {receiver}{c.get('name')}() @ line {c.get('line')} (unresolved)")
                lines.append("")

            return "\n".join(lines)

        # find_callers result (has "target" key)
        if "target" in result and "code_results" in result and "test_results" in result:
            target = result.get("target", {})
            lines = [f"Callers of [{target.get('type')}] {target.get('name')} ({target.get('file')}:{target.get('line')})\n"]

            code_results = result.get("code_results", [])
            test_results = result.get("test_results", [])

            if code_results:
                lines.append(f"Code Callers ({len(code_results)}):")
                for c in code_results:
                    loc = f"{c.get('file')}:{c.get('line')}" if c.get('file') else "?"
                    lines.append(f"  [{c.get('type', '?')}] {c.get('name')} ({loc})")
                    if c.get('summary'):
                        summary = c['summary'][:100] + "..." if len(c.get('summary', '')) > 100 else c.get('summary', '')
                        lines.append(f"    {summary}")
                lines.append("")

            if test_results:
                lines.append(f"Test Callers ({len(test_results)}):")
                for c in test_results:
                    loc = f"{c.get('file')}:{c.get('line')}" if c.get('file') else "?"
                    lines.append(f"  [{c.get('type', '?')}] {c.get('name')} ({loc})")

            return "\n".join(lines)

        # find_call_chain result (has "root" and "direction" keys)
        if "root" in result and "direction" in result:
            root = result.get("root", {})
            direction = result.get("direction")
            max_depth = result.get("max_depth", 5)
            lines = [f"Call chain for [{root.get('type')}] {root.get('name')} ({root.get('file')}:{root.get('line')})"]
            lines.append(f"Direction: {direction}, Max depth: {max_depth}\n")

            def format_tree(nodes: list, indent: int = 0) -> None:
                for node in nodes:
                    prefix = "  " * indent
                    if node.get("cycle"):
                        lines.append(f"{prefix}[CYCLE] {node.get('name')}")
                    elif node.get("unresolved"):
                        receiver = f"{node.get('receiver')}." if node.get('receiver') else ""
                        lines.append(f"{prefix}[unresolved] {receiver}{node.get('name')}()")
                    elif node.get("missing"):
                        lines.append(f"{prefix}[missing] {node.get('name')}")
                    else:
                        loc = f"{node.get('file')}:{node.get('line')}" if node.get('file') else "?"
                        lines.append(f"{prefix}[{node.get('type', '?')}] {node.get('name')} ({loc})")
                        # Recurse into children
                        if node.get("callers"):
                            format_tree(node["callers"], indent + 1)
                        if node.get("callees"):
                            format_tree(node["callees"], indent + 1)

            if result.get("callers"):
                lines.append("Callers:")
                format_tree(result["callers"], 1)
                lines.append("")

            if result.get("callees"):
                lines.append("Callees:")
                format_tree(result["callees"], 1)

            return "\n".join(lines)

        # find_dead_code result (has "potentially_dead" key)
        if "potentially_dead" in result and "stats" in result:
            dead = result.get("potentially_dead", [])
            stats = result.get("stats", {})
            lines = [f"Dead Code Analysis\n"]
            lines.append(f"Stats:")
            lines.append(f"  Total functions: {stats.get('total_functions', 0)}")
            lines.append(f"  Excluded (entry points): {stats.get('excluded_entry_points', 0)}")
            lines.append(f"  Called: {stats.get('called', 0)}")
            lines.append(f"  Potentially dead: {stats.get('potentially_dead', 0)}")
            lines.append("")

            if dead:
                lines.append(f"Potentially Dead Code ({len(dead)}):")
                for d in dead:
                    loc = f"{d.get('file')}:{d.get('line')}" if d.get('file') else "?"
                    lines.append(f"  [{d.get('type', '?')}] {d.get('name')} ({loc})")
                    if d.get('summary'):
                        summary = d['summary'][:100] + "..." if len(d.get('summary', '')) > 100 else d.get('summary', '')
                        lines.append(f"    {summary}")
            else:
                lines.append("No potentially dead code found!")

            return "\n".join(lines)

        # find_entry_points result (has "http", "cli", "test", "main" keys)
        if "http" in result and "cli" in result and "test" in result and "main" in result:
            stats = result.get("stats", {})
            lines = [f"Entry Points (Total: {stats.get('total', 0)})\n"]

            def format_entries(title: str, entries: list) -> None:
                if entries:
                    lines.append(f"{title} ({len(entries)}):")
                    for e in entries:
                        loc = f"{e.get('file')}:{e.get('line')}" if e.get('file') else "?"
                        lines.append(f"  [{e.get('type', '?')}] {e.get('name')} ({loc})")
                        if e.get('decorators'):
                            decs = ", ".join(e['decorators'][:3])
                            if len(e['decorators']) > 3:
                                decs += "..."
                            lines.append(f"    decorators: {decs}")
                    lines.append("")

            format_entries("HTTP Handlers", result.get("http", []))
            format_entries("CLI Commands", result.get("cli", []))
            format_entries("Test Fixtures", result.get("test", []))
            format_entries("Main Functions", result.get("main", []))
            format_entries("Async Tasks", result.get("async_tasks", []))

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

    server = MagaldiMCPServer(
        default_username=args.user,
    )
    asyncio.run(server.run())
