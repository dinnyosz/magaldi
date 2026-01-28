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

from magaldi_mcp.formatters import format_result
from magaldi_mcp.tools.schemas import ALL_TOOL_SCHEMAS
from shared.ai.embedding import CodeEmbeddingClient
from shared.config import get_config, load_config
from shared.db.elasticsearch import ElasticsearchRepository

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
            embed_model = self.config.llm.get_embed_model()
            self.embed_client = CodeEmbeddingClient(
                url=embed_model.url,
                model=embed_model.name,
                provider=embed_model.provider,
                api_key=embed_model.api_key,
            )
        return self.embed_client

    def _register_tools(self) -> None:
        """Register all MCP tools."""

        @self.server.list_tools()
        async def list_tools() -> list[Tool]:
            """List available tools."""
            return list(ALL_TOOL_SCHEMAS)

        @self.server.call_tool()
        async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
            """Handle tool calls."""
            try:
                result = await self._handle_tool(name, arguments)
                return [TextContent(type="text", text=format_result(result))]
            except Exception as e:
                log.exception(f"Tool {name} failed")
                return [TextContent(type="text", text=f"Error: {e}")]

    async def _handle_tool(self, name: str, args: dict[str, Any]) -> Any:
        """Dispatch tool call to implementation."""
        from magaldi_mcp.tools import (
            batch_get_elements,
            dependency_graph,
            explain_element,
            find_async_code,
            find_by_pattern,
            find_call_chain,
            find_callers,
            find_complex_functions,
            find_dead_code,
            find_dependencies,
            find_dependents,
            find_entry_points,
            find_env_usage,
            find_files,
            find_implementations,
            find_security_issues,
            find_similar,
            find_undocumented,
            find_usages,
            generate_config,
            generate_skill,
            get_call_graph,
            get_children,
            get_command_tree,
            get_context,
            get_element,
            get_feature_members,
            get_file_structure,
            get_glossary_term,
            get_repo_stats,
            get_route_tree,
            list_features,
            list_glossary,
            list_patterns,
            list_repos,
            parser_lab_analyze,
            parser_lab_create_test,
            parser_lab_run_tests,
            parser_lab_suggest_fix,
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
        elif name == "pattern_search":
            # ES-native pattern search
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
        elif name == "generate_config":
            return await asyncio.to_thread(
                generate_config,
                repo_path=args["repo_path"],
                scope=args.get("scope"),
                repository=args.get("repository"),
            )
        elif name == "generate_skill":
            return await asyncio.to_thread(
                generate_skill,
                project_root=args.get("project_root"),
                skill_name=args.get("skill_name", "magaldi"),
                scope=args.get("scope", "project"),
            )
        elif name == "explain_element":
            return await asyncio.to_thread(
                explain_element,
                es,
                element_id=args["element_id"],
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
        elif name == "find_dependencies":
            return await asyncio.to_thread(
                find_dependencies,
                es,
                file_path=args.get("file_path"),
                element_id=args.get("element_id"),
                scope=args.get("scope"),
                repository=args.get("repository"),
                username=args.get("username"),
            )
        elif name == "find_dependents":
            return await asyncio.to_thread(
                find_dependents,
                es,
                module=args["module"],
                scope=args["scope"],
                repository=args["repository"],
                username=args.get("username"),
                limit=args.get("limit", 50),
            )
        elif name == "dependency_graph":
            return await asyncio.to_thread(
                dependency_graph,
                es,
                scope=args["scope"],
                repository=args["repository"],
                username=args.get("username"),
                internal_only=args.get("internal_only", True),
            )
        elif name == "list_patterns":
            return await asyncio.to_thread(
                list_patterns,
                es,
                scope=args["scope"],
                repository=args["repository"],
                username=args.get("username", self.default_username),
            )
        elif name == "find_by_pattern":
            return await asyncio.to_thread(
                find_by_pattern,
                es,
                pattern=args["pattern"],
                scope=args["scope"],
                repository=args["repository"],
                username=args.get("username", self.default_username),
                min_confidence=args.get("min_confidence", 0.6),
                limit=args.get("limit", 20),
            )
        # Parser Lab tools (for Magaldi self-improvement)
        elif name == "parser_lab_analyze":
            return await asyncio.to_thread(
                parser_lab_analyze,
                file_path=args.get("file_path"),
                code=args.get("code"),
                context7_query=args.get("context7_query"),
                language=args.get("language"),
                debug=args.get("debug", False),
            )
        elif name == "parser_lab_create_test":
            return await asyncio.to_thread(
                parser_lab_create_test,
                name=args["name"],
                language=args["language"],
                code=args["code"],
                expected=args["expected"],
            )
        elif name == "parser_lab_run_tests":
            return await asyncio.to_thread(
                parser_lab_run_tests,
                filter=args.get("filter"),
                verbose=args.get("verbose", False),
            )
        elif name == "parser_lab_suggest_fix":
            return await asyncio.to_thread(
                parser_lab_suggest_fix,
                gap_description=args["gap_description"],
                language=args["language"],
                failing_test=args.get("failing_test"),
            )
        # Metrics tools
        elif name == "find_complex_functions":
            return await asyncio.to_thread(
                find_complex_functions,
                es,
                scope=args["scope"],
                repository=args["repository"],
                username=args.get("username", self.default_username),
                min_complexity=args.get("min_complexity", 10),
                limit=args.get("limit", 20),
                include_tests=args.get("include_tests", False),
            )
        elif name == "find_security_issues":
            return await asyncio.to_thread(
                find_security_issues,
                es,
                scope=args["scope"],
                repository=args["repository"],
                username=args.get("username", self.default_username),
                severity=args.get("severity", "high"),
                kind=args.get("kind"),
                limit=args.get("limit", 50),
            )
        elif name == "find_undocumented":
            return await asyncio.to_thread(
                find_undocumented,
                es,
                scope=args["scope"],
                repository=args["repository"],
                username=args.get("username", self.default_username),
                max_coverage=args.get("max_coverage", 0.5),
                public_only=args.get("public_only", True),
                limit=args.get("limit", 30),
                include_tests=args.get("include_tests", False),
            )
        elif name == "find_env_usage":
            return await asyncio.to_thread(
                find_env_usage,
                es,
                scope=args["scope"],
                repository=args["repository"],
                username=args.get("username", self.default_username),
                env_name=args.get("env_name"),
                limit=args.get("limit", 50),
            )
        elif name == "find_async_code":
            return await asyncio.to_thread(
                find_async_code,
                es,
                scope=args["scope"],
                repository=args["repository"],
                username=args.get("username", self.default_username),
                pattern=args.get("pattern", "all"),
                limit=args.get("limit", 30),
                include_tests=args.get("include_tests", False),
            )
        elif name == "get_command_tree":
            return await asyncio.to_thread(
                get_command_tree,
                es,
                scope=args["scope"],
                repository=args["repository"],
                username=args.get("username", self.default_username),
            )
        elif name == "get_route_tree":
            return await asyncio.to_thread(
                get_route_tree,
                es,
                scope=args["scope"],
                repository=args["repository"],
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


# Keep _format_result for backward compatibility (used in tests)
def _format_result(result: Any) -> str:
    """Format tool result as readable text.

    This function is kept for backward compatibility.
    New code should use format_result from magaldi_mcp.formatters.
    """
    formatted: str = format_result(result)
    return formatted


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
