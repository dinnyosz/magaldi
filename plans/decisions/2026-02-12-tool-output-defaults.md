# Decision: Optimize MCP Tool Output Defaults

**Date:** 2026-02-12
**Status:** Implemented

## Context

After auditing all 44 MCP tools, we found that many tools return more data than needed by default, consuming context window tokens unnecessarily. The key insight from best practices: **the server should default to minimal output and let the agent request more when needed**, not the other way around.

Key constraint: Claude Code's `MAX_MCP_OUTPUT_TOKENS` default is 25,000 tokens per tool response. Our tools should aim to stay well under this.

## Changes

### Category A: search_code Smart Defaults (biggest impact)

| Change | Before | After | Why |
|--------|--------|-------|-----|
| `brief` | false | **true** | Agent gets compact list (name/file/line), inspects what it needs |
| `element_types` | all | **[function, method]** | 80%+ of searches want these, not files/imports/variables |
| `include_related` | true | **false** | Triggers extra OpenSearch queries + adds up to 10 related files |
| `include_tests` | true | **false** | Most searches want production code, not test code |

**Token impact:** ~4,000 → ~400 tokens per call (10x reduction)

### Category B: Lower Excessive Default Limits

| Tool | Before | After | Why |
|------|--------|-------|-----|
| `pattern_search` | 50 | **20** | Matches search_code; 50 is overwhelming |
| `find_usages` | 30 | **20** | 20 is enough for impact analysis |
| `find_callers` | 30 | **20** | Same as find_usages |
| `find_similar` | 10 | **5** | 5 is enough to see patterns |
| `find_files` | 50 | **30** | 50 file listings floods context |
| `find_security_issues` | 50 | **20** | 20 is enough to prioritize |
| `find_env_usage` | 50 | **30** | Consistency |
| `find_dependents` | 50 | **30** | Consistency |

### Category C: Cap Unbounded Tools

| Tool | Before | After | Why |
|------|--------|-------|-----|
| `find_dead_code` | unbounded (up to 2000) | **limit=30** | Can return hundreds of dead functions |
| `find_entry_points` | unbounded (up to 2000) | **limit=30** | Same |
| `list_glossary` | unbounded | **limit=50** | Glossaries can have hundreds of terms |

### Category D: Reduce Depth/Noise

| Tool | Change | Before | After | Why |
|------|--------|--------|-------|-----|
| `find_call_chain` | max_depth | 5 | **3** | Depth 5 = exponential tree explosion |

### Category E: No Changes Needed (Already Good)

- `get_element` — brief=true already default
- `search_features` — brief=true already default
- `list_features` — brief=true already default
- `get_call_graph` — already compact
- `find_implementations` — limit=20 reasonable
- `find_complex_functions` — limit=20 reasonable
- `find_undocumented` — limit=30, include_tests=false reasonable
- `find_async_code` — limit=30, include_tests=false reasonable
- `explain_element` — already structured with output control
- `dependency_graph` — formatter already caps at 50 edges, 10 cycles
- `batch_get_elements`, `get_context`, `get_children` — single-element or internally capped

## Implementation Notes

Most changes are **schema-level** (default values in tool schemas) + matching `args.get()` defaults in `server.py`. Category C (unbounded tools) also required post-call slicing in `server.py` since `find_dead_code`, `find_entry_points`, and `list_glossary` don't natively support a `limit` parameter.

All limits remain overridable. An agent can always pass `brief=false`, `limit=50`, `include_tests=true` etc. when it needs more.

## Sources

- [10 strategies to reduce MCP token bloat](https://thenewstack.io/how-to-reduce-mcp-token-bloat/)
- [Optimising MCP Server Context Usage](https://scottspence.com/posts/optimising-mcp-server-context-usage-in-claude-code)
- [Reducing MCP token usage by 100x - Speakeasy](https://www.speakeasy.com/blog/how-we-reduced-token-usage-by-100x-dynamic-toolsets-v2)
- [Ballooning context in the MCP era](https://www.coderabbit.ai/blog/handling-ballooning-context-in-the-mcp-era-context-engineering-on-steroids)
