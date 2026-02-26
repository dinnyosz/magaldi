# Deferred Loading & Tool Discoverability: Best Practices Review

## Sources Reviewed

1. [Tool Search Tool - Official Anthropic Docs](https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-search-tool)
2. [Scaling MCP Tools with Anthropic's Defer Loading](https://unified.to/blog/scaling_mcp_tools_with_anthropic_defer_loading)
3. [Claude Code MCP Tool Search: Save 95% Context](https://claudefa.st/blog/tools/mcp-extensions/mcp-tool-search)
4. [Writing Effective Tools for AI Agents](https://www.anthropic.com/engineering/writing-tools-for-agents)
5. [MCP and Context Overload](https://eclipsesource.com/blogs/2026/01/22/mcp-context-overload/)

## Key Numbers

| Metric | Value |
|--------|-------|
| Token cost of 50 tools | ~10-20K tokens |
| Tool Search overhead | ~500 tokens |
| Context reduction | 85% (from ~77K to ~8.7K with 50+ tools) |
| Claude Code auto-trigger threshold | 10% of context window |
| Max tools supported | 10,000 |
| Search results per query | 3-5 tools |
| Regex pattern max length | 200 chars |
| Accuracy improvement (Opus 4) | 49% → 74% with Tool Search |
| Accuracy improvement (Opus 4.5) | 79.5% → 88.1% with Tool Search |
| Optimal response size | Under 25,000 tokens |

## Already Done (what we implemented)

- [x] Added `defer_loading` documentation to SKILL.md and config.py
- [x] Top 5 tools set to `defer_loading: false` based on MCP analytics
- [x] Improved `pattern_search` description (removed "ES-native" jargon)
- [x] Added search keywords to `find_files` and `get_file_structure`

## Actionable Improvements Identified

### HIGH PRIORITY

#### 1. Tool Description Keyword Optimization
**Source**: Anthropic official docs + writing-tools-for-agents
**Finding**: Tool Search matches against tool names, descriptions, argument names, AND argument descriptions. Use semantic keywords that match how users describe tasks.

**Current gaps**:
- `pattern_search`: ✅ Already fixed ("grep-like matching")
- `find_files`: ✅ Already fixed
- `get_file_structure`: ✅ Already fixed
- `search_code`: Good ("semantic search for functions, classes, methods")
- `find_usages`: Good ("find where a function/class/method is called or referenced")
- `get_element`: Could add "inspect" or "view" keyword
- `explain_element`: Good
- `find_similar`: Could clarify "duplicates" or "clone detection"

#### 2. System Prompt Hint for Tool Categories
**Source**: Anthropic official docs (Optimization tips)
**Recommendation**: Add a system prompt section: "You can search for tools to search code, analyze dependencies, find usages, detect patterns, and audit security."
**Impact**: Helps Claude know what categories of tools exist before searching.

#### 3. Error Messages Should Guide Token-Efficient Behavior
**Source**: Anthropic writing-tools-for-agents
**Recommendation**: Instead of generic errors, return messages like "Found 847 results. Please narrow with `glob` filter or `limit` parameter."
**Current state**: Need to audit error messages across all tools.

#### 4. Response Size Control
**Source**: Anthropic writing-tools-for-agents
**Recommendation**: Keep responses under 25K tokens, implement sensible defaults, add filtering and truncation with helpful instructions.
**Current state**: We have `max_tokens` and `filename` params on 9 tools (good), but default `limit` values may be too generous on some tools.

### MEDIUM PRIORITY

#### 5. Parameter Naming Clarity
**Source**: Anthropic writing-tools-for-agents
**Finding**: "Instead of `user`, try `user_id`" — namespacing has non-trivial effects.
**Current state**: We use `hash_id` consistently (good), but `scope` and `repository` could be more descriptive. However changing them would be breaking.

#### 6. Avoid Overlapping Tool Names
**Source**: EclipseSource context overload article
**Finding**: Tools like `get_status`, `fetch_status`, `query_status` cause misfires. Common names like `search` in dozens of MCP servers cause conflicts.
**Current state**: Our tools are well-namespaced under `magaldi` prefix. No conflicts identified.

#### 7. Description Anti-Patterns to Avoid
**Source**: Anthropic writing-tools-for-agents
**Finding**: Avoid low-level technical jargon (uuid, mime_type). Use natural language.
**Current state**: `pattern_search` had "ES-native" (fixed). Some tools mention "hash_id" in descriptions which is internal jargon — but it's also the parameter name so it's needed.

### LOW PRIORITY / ALREADY ADDRESSED

#### 8. Custom Tool Search Implementation
**Source**: Anthropic official docs
**Finding**: You can implement your own tool search using embeddings by returning `tool_reference` blocks from a custom tool.
**Assessment**: Interesting for future — Magaldi already has embeddings and could provide a smarter search than regex/BM25. Not needed now.

#### 9. Prompt Caching Compatibility
**Source**: Anthropic official docs
**Finding**: Tool search works with prompt caching. Use `cache_control` breakpoints.
**Assessment**: API users can leverage this. No action needed from our side.

#### 10. Sub-Agent Isolation
**Source**: EclipseSource article
**Finding**: Sub-agents can have isolated context/tools for specialized tasks.
**Assessment**: Already recommended in SKILL.md with subagent delegation guidance.

## Anti-Patterns to Avoid

1. **All tools deferred** — At least one tool must be non-deferred (API returns 400 error)
2. **Tool descriptions with jargon** — "ES-native", "knn_vector" etc. won't match user queries
3. **Overly generic descriptions** — "Search for things" won't disambiguate from other MCP servers
4. **Too many parameters without defaults** — Agents struggle with many required params
5. **Large response payloads** — Over 25K tokens degrades performance
6. **Missing error guidance** — "ERROR: TOO_MANY_RESULTS" vs "Found 847 results. Use limit or glob to narrow."

## Recommended Next Steps

1. **Add system prompt hint** in SKILL.md describing tool categories
2. **Audit error messages** across all tool implementations for token-efficient guidance
3. **Review default limits** (some tools default to 50 which could produce large responses)
4. **Consider custom tool search** in future (leverage existing embeddings)
