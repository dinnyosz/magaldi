# Decision: MCP Token Optimization via Tool Search

**Date:** 2026-02-12
**Status:** Implemented

## Context

With 44 magaldi tools + playwright (22) + chrome-devtools (22) + slack (8) + context7 (2) = ~98 MCP tools consuming ~22,600 tokens (11.3% of 200K context), we investigated strategies to reduce the static tool schema tax.

## Research Findings

### Available Strategies

1. **`ENABLE_TOOL_SEARCH=true`** — Claude Code's built-in Tool Search defers all MCP tools and loads them on-demand via regex/BM25 search. 85% token reduction. Auto mode has bugs (GitHub #19890), explicit `true` is more reliable.

2. **API-level `defer_loading`** — For direct Anthropic API users, per-tool/per-server deferred loading via `mcp_toolset` config. Requires `advanced-tool-use-2025-11-20` beta header.

3. **MCP Server `instructions`** — Server-level instructions help Claude discover tools when Tool Search is active. Analogous to SKILL.md but at the protocol level.

4. **Code Execution with MCP** — Anthropic's pattern where agents write code to call MCP tools as APIs. 98.7% token reduction but requires sandbox infrastructure. Not actionable for us.

5. **MCP spec proposals** — SEP-1576 (schema deduplication), Discussion #532 (hierarchical tools). Neither merged yet.

## Decision

Implemented strategies 1-3:
- Added `_SERVER_INSTRUCTIONS` to `MagaldiMCPServer` via the MCP SDK `instructions` parameter
- Documented `ENABLE_TOOL_SEARCH=true` for Claude Code users in SKILL.md
- Documented `defer_loading` configuration for API users with recommended 5 always-loaded core tools

## Token Impact

| Scenario | Tokens |
|----------|--------|
| No optimization (all 44 tools loaded) | ~8,700 |
| With Tool Search (Claude Code) | ~200 |
| With defer_loading (API, 5 core tools) | ~1,200 |

## Sources

- [Tool Search Tool - Anthropic Docs](https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-search-tool)
- [Claude Code MCP Docs](https://code.claude.com/docs/en/mcp)
- [Code Execution with MCP](https://www.anthropic.com/engineering/code-execution-with-mcp)
- [GitHub #19890 - auto mode bug](https://github.com/anthropics/claude-code/issues/19890)
