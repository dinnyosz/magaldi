# MCP Token Optimization Strategy for Magaldi

## Research Date: 2026-02-06

## Executive Summary

Magaldi currently exposes 40+ MCP tools. At ~550-850 tokens per tool definition, this means **22,000-34,000 tokens** are consumed by tool definitions alone on every API turn. Combined with tool response payloads, Magaldi can consume 30-50% of Claude Code's 200K context window before any useful work happens.

This document synthesizes research from industry best practices (2025-2026) and proposes a layered optimization strategy.

---

## Part 1: The Two Token Problems

### Problem A: Tool Definition Overhead (Input Tax)

Every API turn, all tool definitions are re-sent as input tokens. With 40+ tools at ~700 tokens each, that's ~28,000 tokens per turn. In a 10-turn conversation, that's 280,000 cumulative input tokens just for tool definitions.

**Real-world measurements:**
| Setup | Token Cost |
|---|---|
| Magaldi (~40 tools, estimated) | ~22,000-34,000 tokens |
| Chrome DevTools MCP (26 tools) | ~17,512 tokens |
| Docker MCP (135 tools) | ~125,964 tokens |
| Anthropic internal (pre-optimization) | 134,000 tokens |

### Problem B: Response Payload Bloat (Context Pollution)

Tool responses enter the context window and stay there permanently (until auto-compaction). A `search_code` call returning 20 results might consume 3,000-5,000 tokens. After the agent uses one result and moves on, the other 19 results are dead weight.

**The lifecycle problem:**
```
Turn 1: search_code → 20 results (4,000 tokens) → agent picks result #3
Turn 2: get_element on result #3 → 800 tokens
Turn 3: find_usages → 15 results (3,000 tokens) → agent picks 2
Turn 4-20: Coding, editing, testing...
         ↑ Those 7,800 tokens from turns 1-3 are still in context, useless
```

---

## Part 2: Token-Saving Techniques (Ranked by Impact)

### Tier 1: Quick Wins (< 1 day each)

#### 1.1 Subagent Delegation via SKILL.md Guidance
**Savings: 60-80% for multi-step search workflows**
**Effort: 1 hour (documentation change only)**

Update SKILL.md to instruct Claude to delegate multi-step Magaldi workflows to Explore subagents. Search results stay in the subagent's isolated context; only the synthesized answer returns to the main context.

**When to delegate:**
- Multi-step workflows: search → inspect → find_usages
- Exploratory queries: "find all authentication code"
- Large result sets: anything with limit > 10

**When to call inline:**
- Single-call lookups: `get_element(hash_id=known_id)`
- Quick pattern searches with expected small results
- File structure queries

#### 1.2 Add `readOnlyHint` Annotations to All Tools
**Savings: Indirect (faster auto-approval, fewer interaction turns)**
**Effort: 30 minutes**

All Magaldi tools are read-only. Adding `readOnlyHint: true` lets clients auto-approve them without user confirmation prompts, reducing interaction overhead.

#### 1.3 Shorten Tool Descriptions
**Savings: 20-40% on tool definition tokens**
**Effort: 2-3 hours**

Current descriptions are verbose. Follow the 1-2 sentence rule:
- Lead with WHEN to use the tool
- Remove redundant parameter descriptions (self-documenting names suffice)
- Move technical details to parameter descriptions, not tool description

**Before:** "FIND CODE: Search for functions, classes, methods by what they do. Uses pre-indexed semantic embeddings - finds 'login' when you search 'authentication'. Returns AI summaries so you understand code without reading it. Use include_code=true to see implementation. Use brief=true for exploration. USE THIS instead of grepping for keywords."

**After:** "Search code by meaning. Finds 'login' when you search 'authentication'. Returns AI summaries."

### Tier 2: Medium Effort (1-3 days each)

#### 2.1 Token Budget Parameter (`max_tokens`)
**Savings: 60-80% per response**
**Effort: 2-3 days**

Inspired by Context7's `tokens` parameter. Add a `max_tokens` parameter to high-volume tools that truncates results server-side.

```python
search_code(query="authentication", max_tokens=2000)
# Server estimates tokens per result, returns as many as fit
# Always appends: "Showing 8 of 23 results (budget: 2000 tokens)"
```

**Apply to:** `search_code`, `find_usages`, `pattern_search`, `find_callers`, `find_call_chain`, `explain_element`

**Implementation:** Use a simple heuristic (1 token ≈ 4 chars) to estimate response size. Truncate result list when approaching budget. Always include a "N more results available" indicator.

#### 2.2 Save-to-File Pattern (`filename` parameter)
**Savings: 95%+ for large results**
**Effort: 1-2 days**

Add an optional `filename` parameter to high-volume tools. When provided, write full results to disk and return only a summary.

```python
search_code(query="auth", filename="/tmp/magaldi-search.md")
# Returns: "23 results saved to /tmp/magaldi-search.md"
# Agent can Read the file selectively if needed
```

**Apply to:** `search_code`, `find_usages`, `pattern_search`, `find_callers`, `dependency_graph`, `find_dead_code`, `find_entry_points`

**Precedent:** Playwright MCP, Chrome DevTools MCP both implement this pattern.

#### 2.3 Compact Output Format
**Savings: 3-5x per response**
**Effort: 1-2 days**

Add a `format` parameter to search tools:

| Format | Output Style | Use Case |
|---|---|---|
| `full` (default) | Current structured output | When agent needs details |
| `compact` | One line per result: `name | file:line | summary` | Exploration/discovery |
| `ids` | Just hash_ids | For batch_get_elements follow-up |

**Token comparison for 20 search results:**
| Format | Estimated Tokens |
|---|---|
| `full` | 4,000-6,000 |
| `compact` | 800-1,200 |
| `ids` | 200-400 |

**Response format consideration:** Plain text/markdown is ~15-80% more token-efficient than JSON for the same data. Current formatters already output markdown, which is good.

#### 2.4 Tool Consolidation (STRAP Pattern)
**Savings: 30-50% on tool definition tokens**
**Effort: 2-3 days**

Merge related tools into domain-level dispatchers:

**Before (6 tools):**
- `find_similar`
- `find_similar_structure`
- `find_similar_intent`
- `find_duplicates`
- `find_callers`
- `find_call_chain`

**After (2 tools):**
- `find_similar(mode="any|structure|intent|duplicates")`
- `find_calls(direction="callers|callees|chain", ...)`

**Candidates for consolidation:**
| Current Tools | Merged Tool |
|---|---|
| `find_similar`, `find_similar_structure`, `find_similar_intent`, `find_duplicates` | `find_similar(mode=...)` |
| `find_callers`, `find_call_chain`, `get_call_graph` | `call_analysis(mode=...)` |
| `find_dependencies`, `find_dependents`, `dependency_graph` | `dependencies(direction=...)` |
| `find_dead_code`, `find_entry_points`, `find_complex_functions`, `find_undocumented` | `code_quality(check=...)` |
| `list_glossary`, `get_glossary_term`, `search_glossary` | `glossary(action=...)` |

This could reduce from ~40 tools to ~20 tools, saving ~14,000 tokens in definitions.

### Tier 3: Significant Effort (1-2 weeks)

#### 3.1 Cursor-Based Pagination
**Savings: Variable (enables early stopping)**
**Effort: 3-5 days**

Add opaque cursor support for list operations:

```python
results = search_code(query="auth", limit=5)
# Returns 5 results + next_cursor="eyJvZmZzZXQiOjV9"

more = search_code(query="auth", limit=5, cursor="eyJvZmZzZXQiOjV9")
# Returns next 5 results + next_cursor=...
```

Lets the agent fetch small batches and stop when it finds what it needs.

#### 3.2 Structured Content (MCP 2025-06-18 Spec)
**Savings: Moderate (client-dependent)**
**Effort: 3-5 days**

Implement `outputSchema` + `structuredContent` per the June 2025 MCP spec. Return both:
- `structuredContent`: Full typed JSON (for programmatic consumption)
- `content`: Compact human-readable summary (for LLM context)

The client can choose which to send to the model. This is the spec-compliant way to do what `brief` mode does today.

#### 3.3 Progressive Discovery / Meta-Tool Pattern
**Savings: 85-96% on tool definitions**
**Effort: 1-2 weeks**

Replace 40+ individual tool registrations with 2-3 meta-tools:

```
magaldi_discover(query="find authentication code")
→ "Suggested tools: search_code, search_features"

magaldi_describe(tool="search_code")
→ Full schema for search_code only

magaldi_execute(tool="search_code", params={...})
→ Results
```

**Token math:**
- Current: ~28,000 tokens for 40 tool definitions
- Meta-tool: ~1,500 tokens for 3 meta-tools + on-demand schema loading
- Per-task overhead: ~500 tokens for discovering + describing needed tools

**Trade-off:** Adds 1-2 extra round trips per tool call. Best for servers with 30+ tools. Speakeasy achieved 96% reduction with this approach.

**Alternative:** Claude Code now supports `ENABLE_TOOL_SEARCH` which achieves similar results at the client level using Anthropic's Tool Search Tool API. This may be simpler than building it into Magaldi itself.

### Tier 4: Architectural (Long-term)

#### 4.1 Code Execution Mode
**Savings: 98.7% (Anthropic's benchmark)**
**Effort: 2-4 weeks**

Present Magaldi tools as TypeScript code APIs instead of direct MCP tools. The agent writes code that calls Magaldi, filters results in-code, and only the final summary enters the LLM context.

This is Anthropic's recommended long-term approach but requires a code execution environment.

---

## Part 3: Magaldi-Specific Recommendations

### Should Magaldi Have a Custom Subagent?

**No.** Use the built-in Explore subagent instead. Reasons:

| Factor | Custom Subagent | Built-in Explore |
|---|---|---|
| Maintenance | Must maintain prompt separately | Zero maintenance |
| Tool access | Must define allowlist | Inherits all MCP tools |
| Domain knowledge | Fixed in prompt | Inherits CLAUDE.md + SKILL.md |
| Flexibility | Rigid | Adapts to context |
| Write capability | Can have it | Read-only |

The "Master-Clone" pattern (recommended by multiple practitioners in 2026): put domain knowledge in CLAUDE.md/SKILL.md and let built-in agents inherit it. This is exactly what Magaldi already does.

**The one exception:** If you want a write-capable agent that can search → analyze → edit in one shot, a custom subagent would be needed. But for search/discovery (Magaldi's core use case), Explore is sufficient.

### Tool Description Optimization

Current Magaldi tool descriptions average ~100-150 words. Industry best practice is 10-25 words.

**Optimization targets (examples):**

| Tool | Current Description Length | Target |
|---|---|---|
| `search_code` | ~50 words | "Search code by meaning. Returns AI summaries. Use brief=true for exploration." (~12 words) |
| `pattern_search` | ~40 words | "ES-native pattern matching: regexp, wildcard, or proximity mode." (~9 words) |
| `find_usages` | ~30 words | "Find where a function/class is called. Filters out definitions." (~9 words) |
| `explain_element` | ~50 words | "Complete element overview: details, callers, callees, similar code, parent." (~9 words) |

### Error Response Optimization

Keep errors concise and actionable:
```
Bad:  "Error: The element with hash_id 'abc123' was not found in the Elasticsearch index 'magaldi-code-elements'. Please verify that the element exists and that the index has been properly populated."

Good: "Element 'abc123' not found. Run search_code to find valid element IDs."
```

### Response Format

Current markdown format is good. Avoid switching to JSON (15-80% more tokens). Consider:
- CSV for tabular results (search results, file lists): ~30% fewer tokens
- Plain text for single-value responses
- Keep markdown for hierarchical data (call graphs, dependency trees)

---

## Part 4: Implementation Priority

| Priority | Action | Effort | Token Savings | Type |
|---|---|---|---|---|
| **P0** | Update SKILL.md for subagent delegation | 1 hour | 60-80% on workflows | Documentation |
| **P0** | Shorten all tool descriptions | 2-3 hours | 20-40% on definitions | Server |
| **P1** | Add `max_tokens` parameter | 2-3 days | 60-80% per response | Server |
| **P1** | Add `filename` parameter | 1-2 days | 95%+ for large results | Server |
| **P1** | Add `readOnlyHint` annotations | 30 min | Indirect | Server |
| **P2** | Add `compact` output format | 1-2 days | 3-5x per response | Server |
| **P2** | Consolidate tools (40 → ~20) | 2-3 days | ~14K tokens saved | Server |
| **P3** | Cursor-based pagination | 3-5 days | Variable | Server |
| **P3** | Structured content (MCP spec) | 3-5 days | Client-dependent | Server |
| **P4** | Progressive disclosure / meta-tool | 1-2 weeks | 85-96% on definitions | Server |

### Estimated Combined Impact

If P0-P2 are implemented:
- Tool definitions: ~28K → ~14K tokens (50% reduction from consolidation + shorter descriptions)
- Average response: ~4K → ~1K tokens (75% reduction from max_tokens + compact format)
- Multi-step workflows: 80% reduction via subagent delegation
- Large results: 95% reduction via save-to-file

**Total estimated context savings: 60-80% across a typical session.**

---

## Part 5: Key Sources

### Anthropic Official
- [Code execution with MCP: building more efficient AI agents](https://www.anthropic.com/engineering/code-execution-with-mcp) - 98.7% reduction
- [Introducing Advanced Tool Use](https://www.anthropic.com/engineering/advanced-tool-use) - Tool Search Tool
- [Tool Search Tool Documentation](https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-search-tool)
- [Create custom subagents](https://code.claude.com/docs/en/sub-agents)

### Industry Best Practices
- [Reducing MCP token usage by 100x - Speakeasy](https://www.speakeasy.com/blog/how-we-reduced-token-usage-by-100x-dynamic-toolsets-v2) - Dynamic toolsets
- [The Meta-Tool Pattern - Synaptic Labs](https://blog.synapticlabs.ai/bounded-context-packs-meta-tool-pattern) - Progressive disclosure
- [Designing MCP servers for wide schemas - Axiom](https://axiom.co/blog/designing-mcp-servers-for-wide-events) - Cell budget
- [15 Best Practices for MCP Servers - The New Stack](https://thenewstack.io/15-best-practices-for-building-mcp-servers-in-production/)
- [10 Strategies to Reduce MCP Token Bloat - The New Stack](https://thenewstack.io/how-to-reduce-mcp-token-bloat/)
- [MCP Tool Descriptions Best Practices - Merge](https://www.merge.dev/blog/mcp-tool-description)
- [STRAP Pattern: 96 to 10 tools](https://almatuck.com/articles/reduced-mcp-tools-96-to-10-strap-pattern)
- [Cloudflare Code Mode](https://blog.cloudflare.com/code-mode/) - 81% reduction

### MCP Specification
- [MCP Spec 2025-06-18 - outputSchema/structuredContent](https://modelcontextprotocol.io/specification/2025-06-18/server/tools)
- [MCP Spec - Pagination](https://modelcontextprotocol.io/specification/2025-03-26/server/utilities/pagination)
- [SEP-1576: Mitigating Token Bloat](https://github.com/modelcontextprotocol/modelcontextprotocol/issues/1576)

### Token Measurements
- [MCP Token Limits: The Hidden Cost - DEV](https://dev.to/piotr_hajdas/mcp-token-limits-the-hidden-cost-of-tool-overload-2d5)
- [Claude Code's Hidden MCP Flag: 32k Tokens Back](https://paddo.dev/blog/claude-code-hidden-mcp-flag/)
- [Claude Code Cut MCP Context Bloat by 46.9%](https://medium.com/@joe.njenga/claude-code-just-cut-mcp-context-bloat-by-46-9-51k-tokens-down-to-8-5k-with-new-tool-search-ddf9e905f734)

### Response Formatting
- [TOON vs JSON: Reduce LLM Token Costs](https://jsontoon.com/toon-vs-json) - 30-60% savings
- [Format comparison: JSON vs XML vs YAML vs Markdown](https://wonderwhy-er.github.io/format-token-comparison/)
- [Context7 Token Budget](https://upstash.com/blog/new-context7)
