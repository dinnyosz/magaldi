"""Magaldi MCP Server - Code Discovery for Claude Code.

Exposes semantic search and code navigation tools via the Model Context Protocol.
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from magaldi_mcp.formatters import format_result
from magaldi_mcp.tools.schemas import ALL_TOOL_SCHEMAS
from magaldi_mcp.tools.schemas._annotations import TOOLS_WITH_OUTPUT_CONTROL
from shared.ai.embedding import CodeEmbeddingClient
from shared.config import get_config, load_config
from shared.db.redis import RedisMCPAnalyticsRepository
from shared.db.store import Repository

log = logging.getLogger(__name__)


_SERVER_INSTRUCTIONS = """\
Magaldi is a code discovery engine. Search for tools to:
- Search code semantically (by meaning, not just text)
- Find files, grep/regex patterns, function usages
- Trace call chains, callers, callees
- Analyze dependencies and imports
- Detect design patterns, dead code, security issues
- Explore features, glossary terms, HTTP routes, CLI commands
- Inspect elements with pre-computed AI summaries

Tool categories: search, navigation, call-graph, features, \
dependencies, code-quality, repo-config, parser-lab.
"""


class MagaldiMCPServer:
    """MCP Server for Magaldi code discovery."""

    def __init__(
        self,
        default_username: str = "main",
        enable_analytics: bool = True,
    ) -> None:
        self.server = Server("magaldi", instructions=_SERVER_INSTRUCTIONS)
        self.config = get_config()
        self.default_username = default_username
        self.repo: Repository | None = None
        self.embed_client: CodeEmbeddingClient | None = None

        # Analytics tracking
        self._enable_analytics = enable_analytics
        self._analytics_repo: RedisMCPAnalyticsRepository | None = None
        # Session ID: use PID + random suffix to identify this MCP session
        self._session_id = f"{os.getpid()}-{uuid.uuid4().hex[:8]}"

        # Register handlers
        self._register_tools()

    def _get_repo(self) -> Repository:
        """Get or create search repository."""
        if self.repo is None:
            self.repo = Repository(self.config)
        return self.repo

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

    def _get_analytics_repo(self) -> RedisMCPAnalyticsRepository | None:
        """Get or create analytics repository."""
        if not self._enable_analytics:
            return None
        if self._analytics_repo is None:
            try:
                self._analytics_repo = RedisMCPAnalyticsRepository(self.config)
            except Exception as e:
                log.warning(f"Failed to initialize analytics: {e}")
                self._enable_analytics = False
                return None
        return self._analytics_repo

    def _record_tool_call(self, tool_name: str, input_args: dict[str, Any] | None = None) -> None:
        """Record a tool call for analytics."""
        try:
            repo = self._get_analytics_repo()
            if repo:
                repo.record_tool_call(tool_name, self._session_id)
                # Also record detailed call info if args provided
                if input_args is not None:
                    repo.record_tool_call_details(tool_name, self._session_id, input_args)
        except Exception as e:
            # Don't let analytics failures affect tool execution
            log.debug(f"Analytics recording failed: {e}")

    def _record_tool_end(self, output_result: str | None = None) -> None:
        """Record that the current tool call has ended."""
        try:
            repo = self._get_analytics_repo()
            if repo:
                repo.record_tool_end(self._session_id)
                # Also record output if provided
                if output_result is not None:
                    repo.record_tool_output(self._session_id, output_result)
        except Exception as e:
            # Don't let analytics failures affect tool execution
            log.debug(f"Analytics end recording failed: {e}")

    def _register_tools(self) -> None:
        """Register all MCP tools."""

        @self.server.list_tools()
        async def list_tools() -> list[Tool]:
            """List available tools."""
            return list(ALL_TOOL_SCHEMAS)

        @self.server.call_tool()
        async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
            """Handle tool calls."""
            # Extract output control params before passing to tool
            max_tokens = arguments.pop("max_tokens", None) if name in TOOLS_WITH_OUTPUT_CONTROL else None
            filename = arguments.pop("filename", None) if name in TOOLS_WITH_OUTPUT_CONTROL else None

            # Record analytics before execution (with input args)
            self._record_tool_call(name, arguments)

            try:
                result = await self._handle_tool(name, arguments)
                formatted_result = format_result(result)

                # Add narrowing hint when results hit the limit
                hint = _build_result_hint(name, result, arguments)
                if hint:
                    formatted_result += f"\n\n{hint}"

                # Apply output control: filename takes precedence over max_tokens
                if filename:
                    formatted_result = _apply_filename(
                        formatted_result, filename, name, result,
                    )
                elif max_tokens:
                    formatted_result = _apply_max_tokens(
                        formatted_result, max_tokens,
                    )

                # Record end time and output for analytics
                self._record_tool_end(formatted_result)
                return [TextContent(type="text", text=formatted_result)]
            except Exception as e:
                log.exception(f"Tool {name} failed")
                error_msg = f"Error: {e}"
                self._record_tool_end(error_msg)
                return [TextContent(type="text", text=error_msg)]

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
            mcp_self_review,
            parser_lab_analyze,
            parser_lab_create_test,
            parser_lab_run_tests,
            parser_lab_suggest_fix,
            pattern_search,
            search_code,
            search_features,
            search_glossary,
        )

        repo = self._get_repo()
        embed_client = self._get_embed_client()

        if name == "search_code":
            return await asyncio.to_thread(
                search_code,
                repo,
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
                include_related=args.get("include_related", True),
            )
        elif name == "search_features":
            return await asyncio.to_thread(
                search_features,
                repo,
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
                repo,
                element_id=args["hash_id"],
                limit=args.get("limit", 10),
                same_repo_only=args.get("same_repo_only", False),
                include_tests=args.get("include_tests", True),
            )
        elif name == "get_element":
            return await asyncio.to_thread(
                get_element,
                repo,
                element_id=args["hash_id"],
                include_code=args.get("include_code", False),
                brief=args.get("brief", True),
            )
        elif name == "get_context":
            return await asyncio.to_thread(
                get_context,
                repo,
                element_id=args["hash_id"],
                include_siblings=args.get("include_siblings", False),
                include_children=args.get("include_children", True),
            )
        elif name == "get_file_structure":
            return await asyncio.to_thread(
                get_file_structure,
                repo,
                scope=args.get("scope"),
                repository=args.get("repository"),
                file_path=args["file_path"],
                username=args.get("username", self.default_username),
            )
        elif name == "list_repos":
            return await asyncio.to_thread(
                list_repos,
                repo,
                scope=args.get("scope"),
            )
        elif name == "list_features":
            return await asyncio.to_thread(
                list_features,
                repo,
                scope=args.get("scope"),
                repository=args.get("repository"),
                username=args.get("username", self.default_username),
                brief=args.get("brief", True),
            )
        elif name == "get_repo_stats":
            return await asyncio.to_thread(
                get_repo_stats,
                repo,
                scope=args.get("scope"),
                repository=args.get("repository"),
                username=args.get("username", self.default_username),
            )
        elif name == "get_children":
            return await asyncio.to_thread(
                get_children,
                repo,
                element_id=args["hash_id"],
            )
        elif name == "get_feature_members":
            return await asyncio.to_thread(
                get_feature_members,
                repo,
                feature_id=args["feature_id"],
            )
        elif name == "batch_get_elements":
            return await asyncio.to_thread(
                batch_get_elements,
                repo,
                element_ids=args["hash_ids"],
                include_code=args.get("include_code", False),
            )
        elif name == "find_files":
            return await asyncio.to_thread(
                find_files,
                repo,
                pattern=args["pattern"],
                scope=args.get("scope"),
                repository=args.get("repository"),
                limit=args.get("limit", 50),
            )
        elif name == "pattern_search":
            return await asyncio.to_thread(
                pattern_search,
                repo,
                pattern=args["pattern"],
                mode=args["mode"],
                scope=args.get("scope"),
                repository=args.get("repository"),
                username=args.get("username"),  # Pass None if not provided
                slop=args.get("slop", 5),
                glob=args.get("glob"),
                limit=args.get("limit", 50),
                include_tests=args.get("include_tests", True),
            )
        elif name == "find_usages":
            return await asyncio.to_thread(
                find_usages,
                repo,
                element_id=args["hash_id"],
                limit=args.get("limit", 30),
            )
        elif name == "find_implementations":
            return await asyncio.to_thread(
                find_implementations,
                repo,
                element_id=args.get("hash_id"),
                class_name=args.get("class_name"),
                scope=args.get("scope"),
                repository=args.get("repository"),
                limit=args.get("limit", 20),
            )
        elif name == "get_call_graph":
            return await asyncio.to_thread(
                get_call_graph,
                repo,
                element_id=args["hash_id"],
                direction=args.get("direction", "both"),
            )
        elif name == "find_callers":
            return await asyncio.to_thread(
                find_callers,
                repo,
                element_id=args["hash_id"],
                scope=args.get("scope"),
                repository=args.get("repository"),
                username=args.get("username"),
                limit=args.get("limit", 30),
                include_tests=args.get("include_tests", True),
            )
        elif name == "find_call_chain":
            return await asyncio.to_thread(
                find_call_chain,
                repo,
                element_id=args["hash_id"],
                direction=args.get("direction", "callees"),
                max_depth=args.get("max_depth", 5),
                scope=args.get("scope"),
                repository=args.get("repository"),
                username=args.get("username"),
            )
        elif name == "find_dead_code":
            return await asyncio.to_thread(
                find_dead_code,
                repo,
                scope=args.get("scope"),
                repository=args.get("repository"),
                username=args.get("username"),
                include_tests=args.get("include_tests", False),
            )
        elif name == "find_entry_points":
            return await asyncio.to_thread(
                find_entry_points,
                repo,
                scope=args.get("scope"),
                repository=args.get("repository"),
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
                repo,
                element_id=args["hash_id"],
            )
        elif name == "list_glossary":
            return await asyncio.to_thread(
                list_glossary,
                repo,
                scope=args.get("scope"),
                repository=args.get("repository"),
                username=args.get("username", self.default_username),
                min_count=args.get("min_count", 1),
            )
        elif name == "get_glossary_term":
            return await asyncio.to_thread(
                get_glossary_term,
                repo,
                scope=args.get("scope"),
                repository=args.get("repository"),
                term=args["term"],
                username=args.get("username", self.default_username),
            )
        elif name == "search_glossary":
            return await asyncio.to_thread(
                search_glossary,
                repo,
                scope=args.get("scope"),
                repository=args.get("repository"),
                query=args["query"],
                username=args.get("username", self.default_username),
            )
        elif name == "find_dependencies":
            return await asyncio.to_thread(
                find_dependencies,
                repo,
                file_path=args.get("file_path"),
                element_id=args.get("hash_id"),
                scope=args.get("scope"),
                repository=args.get("repository"),
                username=args.get("username"),
            )
        elif name == "find_dependents":
            return await asyncio.to_thread(
                find_dependents,
                repo,
                module=args["module"],
                scope=args.get("scope"),
                repository=args.get("repository"),
                username=args.get("username"),
                limit=args.get("limit", 50),
            )
        elif name == "dependency_graph":
            return await asyncio.to_thread(
                dependency_graph,
                repo,
                scope=args.get("scope"),
                repository=args.get("repository"),
                username=args.get("username"),
                internal_only=args.get("internal_only", True),
            )
        elif name == "list_patterns":
            return await asyncio.to_thread(
                list_patterns,
                repo,
                scope=args.get("scope"),
                repository=args.get("repository"),
                username=args.get("username", self.default_username),
            )
        elif name == "find_by_pattern":
            return await asyncio.to_thread(
                find_by_pattern,
                repo,
                pattern=args["pattern"],
                scope=args.get("scope"),
                repository=args.get("repository"),
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
                repo,
                scope=args.get("scope"),
                repository=args.get("repository"),
                username=args.get("username", self.default_username),
                min_complexity=args.get("min_complexity", 10),
                limit=args.get("limit", 20),
                include_tests=args.get("include_tests", False),
            )
        elif name == "find_security_issues":
            return await asyncio.to_thread(
                find_security_issues,
                repo,
                scope=args.get("scope"),
                repository=args.get("repository"),
                username=args.get("username", self.default_username),
                severity=args.get("severity", "high"),
                kind=args.get("kind"),
                limit=args.get("limit", 50),
            )
        elif name == "find_undocumented":
            return await asyncio.to_thread(
                find_undocumented,
                repo,
                scope=args.get("scope"),
                repository=args.get("repository"),
                username=args.get("username", self.default_username),
                max_coverage=args.get("max_coverage", 0.5),
                public_only=args.get("public_only", True),
                limit=args.get("limit", 30),
                include_tests=args.get("include_tests", False),
            )
        elif name == "find_env_usage":
            return await asyncio.to_thread(
                find_env_usage,
                repo,
                scope=args.get("scope"),
                repository=args.get("repository"),
                username=args.get("username", self.default_username),
                env_name=args.get("env_name"),
                limit=args.get("limit", 50),
            )
        elif name == "find_async_code":
            return await asyncio.to_thread(
                find_async_code,
                repo,
                scope=args.get("scope"),
                repository=args.get("repository"),
                username=args.get("username", self.default_username),
                pattern=args.get("pattern", "all"),
                limit=args.get("limit", 30),
                include_tests=args.get("include_tests", False),
            )
        elif name == "get_command_tree":
            return await asyncio.to_thread(
                get_command_tree,
                repo,
                scope=args.get("scope"),
                repository=args.get("repository"),
                username=args.get("username", self.default_username),
            )
        elif name == "get_route_tree":
            return await asyncio.to_thread(
                get_route_tree,
                repo,
                scope=args.get("scope"),
                repository=args.get("repository"),
                username=args.get("username", self.default_username),
            )
        elif name == "mcp_self_review":
            analytics_repo = self._get_analytics_repo()
            return await asyncio.to_thread(
                mcp_self_review,
                analytics_repo,
                context=args["context"],
                include_analytics=args.get("include_analytics", True),
                focus_tools=args.get("focus_tools"),
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


def _estimate_tokens(text: str) -> int:
    """Estimate token count (1 token ~= 4 chars)."""
    return len(text) // 4


# Per-tool narrowing hints: maps tool name to available filter parameters
_TOOL_NARROWING_PARAMS: dict[str, str] = {
    "search_code": "element_types, brief=true, or include_tests=false",
    "pattern_search": "glob filter (e.g. '*.py') or include_tests=false",
    "find_usages": "limit",
    "find_files": "a more specific glob pattern",
    "find_callers": "include_tests=false",
    "find_dead_code": "include_tests, or save to filename",
    "find_entry_points": "save to filename",
    "find_security_issues": "severity filter or kind filter",
    "find_undocumented": "max_coverage threshold or public_only=true",
    "find_env_usage": "env_name filter",
    "find_async_code": "pattern filter (async/threading/locking)",
    "find_complex_functions": "higher min_complexity threshold",
    "find_dependents": "limit",
}


def _count_results(result: Any) -> int | None:
    """Count items in a tool result, handling various result shapes."""
    if isinstance(result, list):
        return len(result)
    if isinstance(result, dict):
        # Grouped results: code_results + test_results
        if "code_results" in result:
            return len(result.get("code_results", [])) + len(result.get("test_results", []))
        # Dead code
        if "potentially_dead" in result:
            return len(result["potentially_dead"])
        # Entry points
        if "entry_points" in result:
            return len(result["entry_points"])
    return None


def _build_result_hint(tool_name: str, result: Any, args: dict[str, Any]) -> str | None:
    """Build a narrowing hint when results hit the requested limit."""
    if tool_name not in _TOOL_NARROWING_PARAMS:
        return None

    limit = args.get("limit")
    if limit is None:
        return None

    count = _count_results(result)
    if count is None or count < limit:
        return None

    params = _TOOL_NARROWING_PARAMS[tool_name]
    return f"Showing {count} results (limit reached). Narrow with {params}, or use max_tokens/filename for output control."


def _apply_max_tokens(formatted: str, max_tokens: int) -> str:
    """Truncate formatted output to fit within a token budget.

    Drops trailing result blocks (separated by blank lines) to stay within budget.
    Each kept result is complete - nothing is cut mid-result.
    """
    if _estimate_tokens(formatted) <= max_tokens:
        return formatted

    # Split on double-newlines to find result boundaries
    blocks = formatted.split("\n\n")
    if len(blocks) <= 1:
        # Single block or no structure - truncate by chars
        char_budget = max_tokens * 4
        if len(formatted) > char_budget:
            return formatted[:char_budget] + f"\n\n... truncated (token budget: {max_tokens})"
        return formatted

    # Keep adding blocks until we exceed the budget
    kept: list[str] = []
    total_chars = 0
    char_budget = max_tokens * 4

    for block in blocks:
        # +2 for the "\n\n" separator
        block_cost = len(block) + (2 if kept else 0)
        if total_chars + block_cost > char_budget and kept:
            break
        kept.append(block)
        total_chars += block_cost

    dropped = len(blocks) - len(kept)
    result = "\n\n".join(kept)
    if dropped > 0:
        result += f"\n\n... {dropped} more results omitted (token budget: {max_tokens}). Use filename param to save full results, or narrow your query."
    return result


def _apply_filename(
    formatted: str,
    filename: str,
    tool_name: str,
    raw_result: Any,
) -> str:
    """Save formatted output to file and return a summary.

    Returns a short summary so the agent can decide whether to read the file.
    """
    from pathlib import Path

    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(formatted, encoding="utf-8")

    # Build a useful summary
    summary = _generate_file_summary(tool_name, raw_result, filename, formatted)
    return summary


def _generate_file_summary(
    _tool_name: str,
    result: Any,
    filename: str,
    formatted: str,
) -> str:
    """Generate a tool-specific summary when results are saved to file."""
    token_count = _estimate_tokens(formatted)
    lines: list[str] = []

    if isinstance(result, list):
        count = len(result)
        # Show top 2-3 names as preview
        preview_names = []
        for item in result[:3]:
            if isinstance(item, dict):
                name = item.get("name") or item.get("element_name") or item.get("term", "")
                if name:
                    preview_names.append(name)
        lines.append(f"{count} results saved to `{filename}` (~{token_count} tokens).")
        if preview_names:
            lines.append(f"Top matches: {', '.join(preview_names)}")

    elif isinstance(result, dict):
        if "element" in result:
            # explain_element style
            el = result["element"]
            name = el.get("name", "unknown") if isinstance(el, dict) else "unknown"
            callers = result.get("callers", [])
            callees = result.get("callees", [])
            lines.append(f"Element `{name}` details saved to `{filename}` (~{token_count} tokens).")
            lines.append(f"Callers: {len(callers)}, Callees: {len(callees)}")
        elif "potentially_dead" in result:
            # find_dead_code
            dead = result.get("potentially_dead", [])
            lines.append(f"{len(dead)} potentially dead functions saved to `{filename}` (~{token_count} tokens).")
        elif "code_results" in result or "test_results" in result:
            # grouped search / find_callers
            code_n = len(result.get("code_results", []))
            test_n = len(result.get("test_results", []))
            lines.append(f"{code_n + test_n} results ({code_n} code, {test_n} test) saved to `{filename}` (~{token_count} tokens).")
        else:
            lines.append(f"Results saved to `{filename}` (~{token_count} tokens).")
    else:
        lines.append(f"Results saved to `{filename}` (~{token_count} tokens).")

    lines.append("Use Read tool to inspect.")
    return "\n".join(lines)


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
