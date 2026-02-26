# MCP Best Practices Review for Magaldi

**Date**: 2026-02-09
**Purpose**: Deep review of current MCP best practices, how LLMs interact with tools, and how Magaldi should evolve.

---

## 0. Executive Summary — What Changed Since v1

This v2 review adds deep dives on:
- **Hierarchical tool patterns** — the 4 proven architectures for tool grouping
- **Server Instructions** — an underutilized MCP feature Magaldi should adopt
- **Anthropic's Context Engineering guide** — prioritization hierarchy and context rot
- **Error recovery patterns** — turning errors into agent learning opportunities
- **GitLab's consolidation principles** — battle-tested tool reduction strategy
- **Modular MCP proxy** — the two-tool (discover + execute) reference implementation
- **Tool Bloat Tipping Point** — hard numbers on where accuracy degrades

---

## 0b. Current Magaldi Tool Description Audit

Before recommendations, here's a concrete gap analysis of every tool description.

### Tools With NO "When to Use" Guidance (All 44)

None of Magaldi's tool descriptions start with "Use this when..." or include negative guidance. Every description follows the pattern "X does Y" rather than "Use this when you need Y. Not for Z."

### Tools With Bare/Missing Parameter Descriptions

`scope`, `repository`, `limit`, and `include_tests` are bare across ~30 tools. These should either have descriptions or be documented in server instructions as "auto-detected, usually omit."

### Overlapping Tools Without Disambiguation

| Overlap Group | Tools | Problem |
|---|---|---|
| Call relationships | `find_callers`, `get_call_graph`, `find_call_chain` | No description says when to pick which |
| Element inspection | `get_element`, `batch_get_elements`, `explain_element` | LLM may use `explain_element` when `get_element` suffices (wasting tokens) |
| Glossary | `list_glossary`, `get_glossary_term`, `search_glossary` | Three tools for one concept |
| Dependencies | `find_dependencies`, `find_dependents`, `dependency_graph` | Directions unclear without reading docs |
| Similarity | `find_similar` | Only one, but the SKILL.md references 3 more that don't exist in schemas |

---

## 0c. Concrete Description Rewrites

Here are before/after rewrites following Anthropic's guidelines for every tool group.

### Search Tools

**search_code** (current):
> "Semantic search for functions, classes, methods by what they do. Returns AI summaries so you understand code without reading it."

**search_code** (proposed):
> "Discover code by describing what it does in natural language. Best when you don't know exact names or file locations. Returns AI summaries so you understand code without reading source. Do NOT use for finding call sites of a known function (use find_usages) or exact text/regex matching (use pattern_search)."

**search_features** (current):
> "Search for high-level features (groups of related functions). Returns pre-clustered capabilities in the codebase."

**search_features** (proposed):
> "Find groups of related functions that implement a capability (e.g., 'authentication', 'caching'). Use when you need a broad overview of how a feature is implemented across the codebase. Do NOT use for finding a single function (use search_code)."

**pattern_search** (current):
> "ES-native pattern matching on code. Three modes: regexp (Lucene syntax), wildcard (* and ?), proximity (terms near each other)."

**pattern_search** (proposed):
> "Search for exact text patterns, regex, or terms near each other in source code. Use when you know the literal text you're looking for (function names, string literals, patterns). Do NOT use for semantic/meaning-based search (use search_code)."

### Call Relationship Tools — The Critical Disambiguation

**get_call_graph** (proposed):
> "Get the immediate callers and callees of a function (1 level deep). Use for a quick overview of direct relationships. Pre-computed, instant. Do NOT use for recursive call chains (use find_call_chain) or for a full caller list with file/test grouping (use find_callers)."

**find_callers** (proposed):
> "Find all functions that call a given function, grouped by production code vs tests. Use for impact analysis. Supports output control (max_tokens, filename). Do NOT use for recursive traces (use find_call_chain) or quick overview (use get_call_graph)."

**find_call_chain** (proposed):
> "Trace call chains recursively (A→B→C→D). Use before refactoring to see full downstream/upstream impact up to N levels deep. Do NOT use for just immediate callers (use get_call_graph or find_callers)."

### Inspect Tools — Disambiguation

**get_element** (proposed):
> "Get details of a specific element by ID. Use when you have a hash_id and need its summary, signature, or code. Lightweight. For multiple elements use batch_get_elements. For comprehensive overview including callers/callees/similar use explain_element."

**explain_element** (proposed):
> "Comprehensive overview of a code element in one call: details, callers, callees, imports, similar code, parent context. Use when you need full understanding. Expensive — much more data than get_element. Do NOT use for quick lookups."

**batch_get_elements** (proposed):
> "Get details of multiple elements by their IDs in one call. Use when you have several hash_ids to avoid N separate get_element calls."

### Glossary Tools — Consolidation Candidate

**list_glossary** (proposed):
> "List all domain concepts extracted from code. Use to understand a codebase's vocabulary. For finding a specific term use search_glossary instead."

**search_glossary** (proposed):
> "Find glossary terms by partial match (e.g., 'user' finds 'user', 'username', 'user_id'). For full details of a known term use get_glossary_term."

### Dependency Tools — Direction Clarity

**find_dependencies** (proposed):
> "Get what a file imports (internal and external). Use to understand a file's inputs. For the reverse (what imports this) use find_dependents."

**find_dependents** (proposed):
> "Find all files that import a given module. Use for impact analysis before changing a module. For forward direction (what does a file import) use find_dependencies."

### Bare Parameter Fixes

For recurring bare params, add descriptions:

| Param | Proposed Description |
|---|---|
| `scope` | `"Organization/project scope. Auto-detected from magaldi.yaml — usually omit."` |
| `repository` | `"Repository name. Auto-detected from magaldi.yaml — usually omit."` |
| `limit` | `"Maximum results to return (default: N). Higher values consume more context window."` |
| `include_tests` | `"Include test files in results. Set false to focus on production code."` |

---

## 1. The Token Budget Crisis

The single most important issue in MCP today. Every tool schema is serialized into the LLM's context window on every turn. With Magaldi exposing **44 tools**, we consume thousands of tokens before any work begins.

### Industry Data Points

| Source | Finding |
|---|---|
| Anthropic (Nov 2025) | Code-mode reduced context from ~150K to ~2K tokens (98.7% reduction) |
| Claude Code Tool Search | Cut MCP context from 51K to 8.5K tokens (46.9% reduction) |
| Speakeasy Dynamic Toolsets | 96-100x token reduction with on-demand loading |
| Synapticlabs Meta-Tool | 85-95% reduction with bounded context packs |
| Spring AI Tool Search | 34-64% token savings with dynamic discovery |
| Dynamic ReAct (paper) | 50% tool loading reduction while maintaining accuracy |
| Anthropic research | **Model accuracy DECREASES as context window size increases** |

### What This Means for Magaldi

Traditional approach with 50+ tools: ~77K tokens before work begins.
With Tool Search Tool: ~8.7K tokens (search tool + 3-5 discovered tools).
**Magaldi's 44 tools are well above the 10-15 tool sweet spot.**

### Solutions Being Adopted

1. **Anthropic's `defer_loading`** (Beta: `advanced-tool-use-2025-11-20`):
   - Mark tools with `defer_loading: true`
   - Claude only sees Tool Search Tool + always-loaded tools
   - When Claude needs a capability, it searches
   - Results: Opus 4 accuracy jumped from 49% → 74%, Opus 4.5 from 79.5% → 88.1%

2. **SEP-1576: Schema Deduplication via `$ref`**:
   - JSON Schema `$ref` references eliminate redundant definitions
   - Magaldi's `scope`, `repository`, `limit`, `include_tests` appear in ~30 tools
   - Not yet in spec, but coming

3. **Meta-Tool Pattern** (Synapticlabs):
   - Two registered tools: `discover` + `execute`
   - Discovery tool's description lists all available agents/tools
   - LLM requests specific schemas on demand, executes through execution tool
   - Three layers: meta-tools → domain agents → individual tools

4. **`allowed_tools` Filtering**:
   - Claude API's `allowedTools` parameter
   - Wildcards: `"mcp__magaldi__search_*"` or specific: `"mcp__magaldi__search_code"`
   - Clients can filter per-request based on task context

### Tool Bloat Tipping Point (Synapticlabs Research)

Hard numbers on where things break:

| Tool Count | Effect |
|---|---|
| 1-15 | LLMs perform well at tool selection |
| 20-30 | Problems begin: slower selection, occasional misuse |
| 40+ | Degraded response quality, wrong tool selection, hallucinated parameters |
| 50+ | ~10K-20K tokens consumed by schemas alone (200-400 tokens per tool) |

**Token math for Magaldi's 44 tools at ~300 tokens each = ~13,200 tokens** consumed every turn just for tool schemas. This is before any conversation, system prompt, or tool results.

The meta-tool pattern (2 tools at ~600 tokens + 3 on-demand at ~450 tokens = ~1,050 tokens) gives **~12x reduction**.

---

## 2. Anthropic's Official Tool Design Principles

From **"Writing Effective Tools for AI Agents"** (Anthropic Engineering, Dec 2025):

### Principle 1: Don't Wrap Every API Endpoint

> "Don't wrap every API endpoint as a tool. Focus on high-impact workflows that match how users (and agents) actually think about tasks."

**Consolidate multi-step operations.** Instead of `list_users` + `list_events` + `create_event`, provide `schedule_event` which finds availability and schedules.

**Magaldi assessment**: We have some tool sprawl:
- `find_callers` vs `get_call_graph` vs `find_call_chain` — three tools for call relationships
- `get_element` vs `batch_get_elements` vs `explain_element` — three tools for element inspection
- `find_dependencies` vs `find_dependents` vs `dependency_graph` — three tools for deps
- `list_glossary` vs `get_glossary_term` vs `search_glossary` — three tools for glossary

Could some of these be consolidated with mode parameters?

### Principle 2: Return Human-Readable Context

> "Agents need human-readable context, not just technical identifiers. When returning data, include descriptions and categories rather than UUIDs that require additional tool calls."

**Magaldi assessment**: Strong here — AI summaries in search results mean the LLM understands code without reading source. The `hash_id` is a necessary identifier but always accompanied by name, file, line, summary.

### Principle 3: Configurable Response Verbosity

> "Make tools' response verbosity configurable with a `response_format` enum: 'concise' or 'detailed'."

Anthropic's recommended pattern:
```
enum ResponseFormat {
  DETAILED = "detailed",   // Full metadata, IDs, everything
  CONCISE = "concise"      // Just the essentials, no IDs
}
```

**Magaldi assessment**: We have `brief` (boolean) and `max_tokens` (integer). The `response_format` enum pattern is more expressive:
- `concise`: name, file, line only (current `brief=true`)
- `detailed`: full summaries, related elements, code (current default)
- Could add: `ids_only` for chaining, `with_code` for source inspection

### Principle 4: Tools Are Contracts With Non-Deterministic Agents

> "Tools represent a fundamentally new software paradigm: contracts between deterministic systems and non-deterministic agents."

Design defensively:
- Clear enough that agents can't easily misuse them
- Informative enough to guide agents toward better strategies
- Efficient enough to preserve context window space

### Principle 5: Keep Responses Under 25K Tokens

For optimal LLM performance. Magaldi's `max_tokens` parameter addresses this, but it's opt-in. Consider making token budgets the default behavior.

---

## 3. Tool Description Quality

### Current Best Practice (SEP-1382, Merge.dev, OpenAI)

**Tool descriptions** help with tool SELECTION (which tool to use).
**Schema descriptions** help with tool EXECUTION (how to call it correctly).

#### Description Format Recommendations

1. **Start with "when to use"**: "Use this to find functions/classes by what they do. Best for discovering code you haven't seen before."
2. **Include negative guidance**: "Do NOT use this for finding where a known function is called — use `find_usages` instead."
3. **Disambiguate similar tools**: When tools overlap, descriptions MUST clarify boundaries.
4. **Keep it 1-2 sentences**: Concise high-level explanation of what the tool accomplishes.

#### Parameter Description Recommendations

1. **Include examples**: `"start_date: ISO date string (YYYY-MM-DD) for the beginning of the search range"`
2. **Explain enums inline**: `"element_types: Filter by type. Options: file, class, function, method, variable, constant, interface, trait, enum, type_alias, import"`
3. **Never leave bare parameters**: Every parameter needs a description, even `limit`.
4. **Describe defaults and behavior**: `"limit: Maximum results to return (default: 20). Higher values use more context window."`

### Magaldi Assessment

**Good:**
- `search_code` description includes examples in `query` parameter
- Verb-first naming: `search_code`, `find_usages`, `find_callers`

**Gaps:**
- No "when to use" / "when NOT to use" in any tool description
- No disambiguation between overlapping tools
- `limit` has no description in most tools — just `{"type": "integer", "default": 20}`
- Tool descriptions describe what the tool IS, not WHEN to use it

**Example rewrite for `search_code`:**
```
Current:  "Semantic search for functions, classes, methods by what they do.
           Returns AI summaries so you understand code without reading it."

Better:   "Use this to discover code by describing what it does in natural language.
           Best for exploration when you don't know exact names or locations.
           Do NOT use for finding where a known function is called (use find_usages)
           or for exact text/regex matching (use pattern_search)."
```

---

## 4. Tool Annotations (MCP Spec)

### The Four Standard Annotations

| Annotation | Magaldi Usage | Assessment |
|---|---|---|
| `readOnlyHint` | ✅ true for queries, false for generators | Correct |
| `destructiveHint` | ✅ false for all tools | Correct — we never destroy data |
| `idempotentHint` | ✅ true for all tools | Correct — same query = same result |
| `openWorldHint` | ✅ false for all tools | Correct — closed system (OpenSearch) |

**Magaldi is ahead of most MCP servers here.** Many servers don't set annotations at all, causing ChatGPT and other hosts to treat all tools as potentially destructive (requiring approval for every call).

---

## 5. Output Design

### Structured Content (MCP 2025-06-18+)

The spec now supports dual output:
- `content`: Text blocks for LLM consumption (backward compatible)
- `structuredContent`: JSON object for machine-readable parsing
- `outputSchema`: JSON Schema for validating structured output

**Magaldi assessment**: We return text content only. Adding `structuredContent` would enable:
- Downstream tool chaining without re-parsing text
- Client-side rendering (IDEs could display results as trees, tables)
- Validation of response format

### Progressive Disclosure Pattern

Best practice for output:
1. **Default**: Brief, actionable summary
2. **On request**: Full details with `response_format=detailed` or `include_code=true`
3. **Overflow**: Save to file with `filename` parameter

**Magaldi assessment**: Already implemented via `brief`, `max_tokens`, `filename`. This is a strength.

### Error Handling

MCP best practice: Return errors inside the result with `isError: true`, not as protocol exceptions.

Error messages should answer three questions:
1. What happened? ("The element was not found.")
2. Why? ("No element exists with hash_id 'abc123' in repository 'myrepo'.")
3. What to do? ("Use search_code to find the element by description, or check the hash_id from previous results.")

**Magaldi assessment**: Need to audit error responses for actionable guidance.

---

## 5b. Server Instructions — An Underutilized Feature

### What Are Server Instructions?

MCP servers can provide `instructions` — text injected into the LLM's system prompt that explains *how to use the server*. This is separate from individual tool descriptions. Think of it as a "user manual" for the entire server.

From the [MCP blog](http://blog.modelcontextprotocol.io/posts/2025-11-03-using-server-instructions/):
> "Server instructions give the server a way to inject information that the LLM should always read in order to understand how to use the server — independent of individual prompts, tools, or messages."

### Best Practices for Server Instructions

1. **Focus on what tools and resources don't convey** — preferences, workflow patterns, disambiguation rules
2. **Keep them concise** — they consume context tokens on every turn
3. **No instructions are better than poorly written instructions** — bad instructions confuse the model
4. **Include tool workflow guidance** — "Start with `search_code` to discover elements, then use `get_element` for details"

### Implementation in Python SDK

```python
from fastmcp import FastMCP
mcp = FastMCP(
    name="magaldi",
    instructions="Magaldi is a code discovery engine. Start with search_code for exploration, ..."
)
```

Or with the low-level SDK:
```python
from mcp.server import Server
server = Server("magaldi")
# Instructions set via server_info capabilities
```

Clients like Claude Code, LibreChat, and VS Code Copilot inject these into the agent's system message when MCP server tools are used. Custom instructions take precedence over server-provided ones.

### Magaldi Assessment

**Current state:** `Server("magaldi")` — no instructions set.
**Opportunity:** Add server instructions that explain:
- The typical workflow: search → inspect → navigate → analyze
- When to use semantic search vs pattern search
- That `scope`/`repository` are auto-detected
- Tool group overview so the LLM knows what categories exist

This is **high impact, very low effort** — a single string on server init.

---

## 5c. Anthropic's Context Engineering Hierarchy

From **"Effective Context Engineering for AI Agents"** (Anthropic, Dec 2025):

### The Priority Stack

When allocating context budget, prioritize in this order:

1. **Current task** (highest priority)
2. **Tools** (schemas and descriptions)
3. **Retrieved documents** (RAG results)
4. **Memory** (persistent state)
5. **History** (conversation turns)

### Context Rot

> "LLMs get confused when given too much information — a phenomenon known as 'context rot' where the model's ability to accurately recall information decreases as token count increases."

This is why Magaldi's 44 tool schemas actively *hurt* accuracy — not just waste tokens, but degrade the model's ability to reason about the tools it has.

### Compression Strategies Applicable to Magaldi

| Strategy | Application |
|---|---|
| **Trimming** | Remove `scope`/`repository` from schemas when auto-detected |
| **Summarization** | Server instructions summarize all tools instead of loading each schema |
| **Budget per query** | Default `max_tokens` on all tools, not just opt-in |
| **Deduplication** | `$ref` for repeated parameter blocks (SEP-1576) |
| **Progressive disclosure** | `defer_loading` for rarely-used tools |

### CodeRabbit's Approach (Relevant Parallel)

CodeRabbit, as an MCP client consuming multiple servers, developed:
1. **Context deduplication** — collapse repeated data
2. **Summarization pipelines** — LLMs summarize retrieved context before final use
3. **Prioritization + truncation** — token budgets per MCP query
4. **"Good context engineering is knowing what to leave out"**

---

## 5d. Error Handling Deep Dive

### The MCP Error Model

Two kinds of errors:
1. **Protocol-level errors** — JSON-RPC errors caught by MCP client, *never reach the LLM*
2. **Tool-level errors** — returned in the result payload with `isError: true`, *injected into LLM context*

**Critical insight:** Tool errors ARE prompts. The LLM reads them like any other text. A well-crafted error message teaches the model to self-correct.

### Magaldi's Current Error Handling

```python
except Exception as e:
    log.exception(f"Tool {name} failed")
    error_msg = f"Error: {e}"
    return [TextContent(type="text", text=error_msg)]
```

Problems:
- Returns raw Python exception strings (not LLM-friendly)
- No `isError: true` flag — LLM can't distinguish errors from results
- No recovery guidance — model has no next step
- No disambiguation — "Error: 404" means nothing to an agent

### Recommended Pattern (from Alpic AI, MCPcat)

Every error response should answer three questions:

```
Error: Element not found.
Reason: No element exists with hash_id 'abc123' in scope 'myorg' repository 'myrepo'.
Recovery: Use search_code to find the element by description instead, or verify
the hash_id from previous search results. Hash IDs are hex strings like 'a1b2c3d4...'.
```

### Specific Error Categories for Magaldi

| Error Type | Current | Recommended |
|---|---|---|
| Element not found | `Error: NoneType` | "Element not found. Use search_code to discover elements by description." |
| Invalid hash_id | Raw exception | "Invalid element ID format. Hash IDs are 64-char hex strings from search results." |
| Repository not found | `Error: index not found` | "Repository 'X' not indexed. Use list_repos to see available repositories." |
| Empty results | Returns `[]` | "No results found. Try broader query terms or remove element_type filters." |
| Embedding service down | Connection error | "Embedding service unavailable. search_code and find_similar won't work. Use pattern_search (regex-based, no embeddings needed) as fallback." |

---

## 6. Hierarchical Tools & Progressive Discovery (Deep Dive)

### The Problem

44 flat tools force the LLM to scan all descriptions every turn. This causes:
- Token waste (~13K for schemas alone)
- Decision paralysis (similar tools confuse selection)
- Hallucinated parameters (too many schemas to keep straight)
- **Context rot** — accuracy degrades with more context, not just wastes it

### The Four Proven Architectures for Hierarchical Tools

Based on research from Dynamic ReAct, Synapticlabs, Klavis AI, Modular MCP, and MCP SDK discussions:

#### Architecture 1: `defer_loading` (Anthropic Native)

**How it works:**
- All 44 tools registered with the API
- ~15 core tools: `defer_loading: false` (always in context)
- ~29 secondary tools: `defer_loading: true` (invisible until searched)
- Claude's built-in Tool Search Tool finds deferred tools when needed
- Searched tools get expanded into context for that turn

**Pros:** Zero MCP server changes needed (just schema annotations). Supported by Claude API natively.
**Cons:** Requires Anthropic's beta header. Only works with Claude. Other clients (ChatGPT, Copilot) don't support it.
**Token savings:** ~46-50% (based on Claude Code benchmarks)

**Implementation for Magaldi:**
```python
# In tool schema definitions
Tool(
    name="find_dead_code",
    description="...",
    inputSchema={...},
    annotations=READONLY_ANNOTATIONS,
    # NEW: mark as deferrable
    defer_loading=True,
)
```

#### Architecture 2: Meta-Tool Pattern (Two-Tool Discovery/Execute)

**How it works:**
- Server registers only 2 tools: `discover_tools` + `call_tool`
- `discover_tools` description lists all available groups and their tools
- LLM calls `discover_tools(group="search")` → gets schemas for search tools
- LLM calls `call_tool(name="search_code", args={...})` → executes

**Reference implementation:** [Modular MCP](https://github.com/d-kimuson/modular-mcp) — a proxy that wraps any MCP server with this pattern.

**Pros:** Works with ALL clients. Maximum token savings (~90%). Client-agnostic.
**Cons:** Extra round-trip for discovery. Loss of native client features (auto-complete, type hints).
**Token savings:** ~85-92% (2 tools at ~600 tokens vs 44 at ~13K)

**Magaldi-specific design:**
```
discover_tools(group?) → If no group: returns group list with descriptions.
                         If group specified: returns full schemas for that group's tools.

call_tool(tool_name, args) → Dispatches to actual tool implementation.
```

#### Architecture 3: Hierarchical Strata (Klavis AI)

**How it works — 4 layers:**
1. LLM sees a list of **services** (e.g., "code_search", "code_analysis", "architecture")
2. Selects a service → receives its **categories** (e.g., code_search → semantic, pattern, file)
3. Selects a category → receives **action names + descriptions** (but NOT full schemas)
4. Selects an action → receives **complete parameter schema** and executes

**Pros:** Minimum context per step. Ideal for 100+ tools.
**Cons:** 3-4 round-trips before first tool execution. Overkill for 44 tools.
**Token savings:** 95%+ but at latency cost

#### Architecture 4: Namespace Grouping (SEP-993)

**How it works:**
- Proposed MCP spec extension for tool namespacing
- Tools named like `search.code`, `search.features`, `inspect.element`
- Clients can discover groups first, then list tools within a group
- [MetaMCP](https://docs.metamcp.com/en/concepts/namespaces) implements `@agent-name/tool-name` convention

**Pros:** Clean, spec-aligned, good client UX.
**Cons:** Not yet in MCP spec. Requires client support. Current naming `mcp__magaldi__search_code` already uses flat namespacing.

### Recommendation for Magaldi: Hybrid Approach

**Short term (now):** Architecture 1 (`defer_loading`) — zero server changes, immediate benefit for Claude users.

**Medium term:** Architecture 2 (meta-tool) as an *opt-in alternative mode* — so non-Claude clients benefit too. Keep the flat tool list as default for backward compat, add meta-tool mode via config flag.

**Not recommended:** Architecture 3 (too many round-trips for our scale) or Architecture 4 (spec not ready).

### Magaldi's Natural Groups (for any architecture)

| Group (10) | Tools | Count | Always-Load? |
|---|---|---|---|
| **Search** | search_code, search_features, find_similar, pattern_search | 4 | Yes |
| **Inspect** | get_element, batch_get_elements, get_context, get_children, explain_element | 5 | Yes |
| **Navigate** | find_usages, find_callers, find_call_chain, get_call_graph, find_implementations | 5 | Yes |
| **Files** | find_files, get_file_structure | 2 | Yes |
| **Analysis** | find_dead_code, find_complex_functions, find_security_issues, find_undocumented, find_async_code, find_env_usage, find_entry_points | 7 | **No** |
| **Architecture** | find_dependencies, find_dependents, dependency_graph, list_patterns, find_by_pattern | 5 | **No** |
| **Domain** | list_glossary, get_glossary_term, search_glossary, list_features, get_feature_members | 5 | **No** |
| **Meta** | list_repos, get_repo_stats, get_command_tree, get_route_tree | 4 | **No** |
| **Config** | generate_skill, generate_config | 2 | **No** |
| **Labs** | parser_lab_analyze, parser_lab_create_test, parser_lab_run_tests, parser_lab_suggest_fix, mcp_self_review | 5 | **No** |
| | | **44 total** | **16 always / 28 deferred** |

### Token Budget Comparison

| Mode | Tools in context | Tokens |
|---|---|---|
| Current (flat 44) | 44 | ~13,200 |
| `defer_loading` (16 core) | 16 + Tool Search | ~5,400 |
| Meta-tool (2 + on-demand 3) | 5 | ~1,500 |
| Ideal (after consolidation to ~30 + defer) | 12 + Tool Search | ~4,200 |

---

## 6b. Tool Consolidation Strategy (GitLab Model)

### GitLab's Principles

GitLab maintains one of the largest production MCP servers. Their [development guidelines](https://docs.gitlab.com/development/mcp_server/) distill hard-won lessons:

1. **Necessity check**: "Is this truly a new capability, or could an existing tool handle it with a parameter adjustment?"
2. **Consolidation potential**: "Can functionality be merged with an existing tool using an enum or parameter?"
3. **Aggregated tools pattern**: "Combine multiple related API tools that serve similar purposes but operate at different scopes into a single unified interface."

Example: GitLab consolidated `search_global`, `search_group`, `search_project` into a single `search` tool with a `scope` parameter.

### Consolidation Candidates for Magaldi

#### Tier 1: Clear merges (reduce by 5-6 tools)

| Current Tools | Proposed | How |
|---|---|---|
| `get_element` + `batch_get_elements` | `get_element` | Accept `hash_id` (string) OR `hash_ids` (array). Single ID → single result, array → batch. |
| `list_glossary` + `search_glossary` | `glossary` | Optional `query` param. No query → list all. With query → search. |
| `list_features` + `get_feature_members` | `features` | Optional `feature_id`. No ID → list features. With ID → show members. |
| `list_patterns` + `find_by_pattern` | `patterns` | Optional `pattern` type param. No filter → list all. With filter → find by pattern. |
| `find_dependencies` + `find_dependents` | `dependencies` | `direction` param: "imports" (what file imports) or "importers" (what imports this file). |

#### Tier 2: Consider merging (reduce by 2-3 more tools)

| Current Tools | Proposed | Notes |
|---|---|---|
| `find_callers` + `get_call_graph` | `call_graph` | `find_callers` is a subset of `get_call_graph(direction="callers")` — already exists! |
| `get_context` + `get_children` | `context` | `get_children` is `get_context(include_children=true, include_siblings=false)` |

#### Tier 3: Leave separate (different enough)

These look similar but serve genuinely different workflows:
- `search_code` vs `pattern_search` — semantic vs regex, different engines entirely
- `find_usages` vs `find_callers` — text references vs call graph edges
- `get_element` vs `explain_element` — brief lookup vs comprehensive analysis
- `find_call_chain` vs `get_call_graph` — recursive traversal vs single-hop graph

### Post-Consolidation Tool Count

| | Count |
|---|---|
| Current | 44 |
| After Tier 1 merges | 38-39 |
| After Tier 1 + 2 | 35-36 |
| After consolidation + `defer_loading` (16 core) | **16 in context** |

---

## 7. The Competition: Code Search MCP Servers

### Serena (oraios)
- **Approach**: Language server integration (LSP) for semantic code understanding
- **Key tools**: `find_symbol`, `find_referencing_symbols`, `insert_after_symbol`
- **Strength**: Symbol-level precision, IDE-like experience
- **Weakness**: Requires language server setup per language

### Claude Context (Zilliz)
- **Approach**: AST-based code chunking + vector embeddings (Milvus)
- **Strength**: Incremental indexing via Merkle trees
- **Weakness**: Less rich metadata than Magaldi

### Code Index MCP (johnhuang316)
- **Approach**: Basic indexing + search
- **Strength**: Simple setup
- **Weakness**: No AI summaries, no call graphs

### Magaldi's Differentiators
- **AI summaries**: LLM understands code without reading source
- **Three embedding types**: summary, code, caller (asymmetric)
- **Call graph resolution**: Static + embedding + semantic strategies
- **Feature extraction**: Pre-clustered capability groups
- **Glossary**: Domain concept extraction
- **Multi-user model**: Main branch + user diffs

---

## 8. MCP Spec Evolution (2025-11-25)

Key changes Magaldi should track:

| Feature | Status | Relevance |
|---|---|---|
| **Tasks** (async execution) | Experimental | Long-running analysis could return task handles |
| **Extensions framework** | Stable | Custom capabilities outside core spec |
| **Server Discovery** | Coming | `.well-known` URLs for browsing servers |
| **Protected Resource Metadata** | New | Auth declarations for enterprise |
| **Structured Content + outputSchema** | Stable (June 2025) | Type-safe tool outputs |
| **Tool Search Tool / defer_loading** | Beta | On-demand tool discovery |

---

## 9. Specific Recommendations for Magaldi

### Priority 1: Tool Description Rewrite (High Impact, Low Effort)

Rewrite all 44 tool descriptions to follow Anthropic's pattern:
- "Use this when..." opening
- Negative guidance for disambiguation
- Examples in parameter descriptions
- No bare parameters

### Priority 2: `defer_loading` Support (High Impact, Medium Effort)

Categorize tools into always-loaded vs deferrable:
- **Always loaded** (~12-15): search_code, find_usages, pattern_search, get_element, explain_element, find_files, get_file_structure, find_callers, find_call_chain, get_call_graph, find_implementations, search_features
- **Deferred** (~29): Everything else

This requires clients to support the beta, but Claude Code already does.

### Priority 3: Response Format Enum (Medium Impact, Low Effort)

Replace `brief` boolean with `response_format` enum:
- `concise`: Name, file, line only
- `detailed` (default): Full summaries, related elements
- `with_code`: Include source code
- `ids_only`: Just hash_ids for chaining

### Priority 4: Tool Consolidation (Medium Impact, Medium Effort)

Candidates for merging:
- `get_element` + `batch_get_elements` → single tool with `hash_id` (string) or `hash_ids` (array)
- `list_glossary` + `search_glossary` → single tool with optional `query` param
- `find_callers` + `get_call_graph` → `get_call_graph` with direction param (already exists)

### Priority 5: Structured Content (Lower Impact, Medium Effort)

Add `outputSchema` to tool definitions and return `structuredContent` alongside `content`. Enables typed tool chaining and client-side rendering.

### Priority 6: Error Message Quality (Medium Impact, Low Effort)

Audit all error responses. Every error should answer:
1. What happened?
2. Why?
3. What should the LLM do instead?

---

## 9b. Server Instructions — An Untapped Feature

### What Are Server Instructions?

MCP servers can provide `serverInstructions` — a free-form text string injected into the LLM's system prompt when the server's tools are active. It acts like a "user manual" for the entire server.

**From the MCP blog:** "Server instructions give the server a way to inject information that the LLM should always read in order to understand how to use the server — independent of individual prompts, tools, or messages."

### When to Use Them

Server instructions are for things that **individual tool descriptions can't convey**:
- **Tool interdependence**: "Always use search_code before find_usages — you need a hash_id first"
- **Multi-tool workflows**: "To review a function: search_code → get_element → find_callers"
- **Global conventions**: "scope and repository are auto-detected from magaldi.yaml — usually omit"
- **Server-wide context**: "This server searches a pre-indexed codebase. All results come from the index, not live filesystem."

### Magaldi's Current State

- **No `serverInstructions` set.** The server initializes as `Server("magaldi")` with no instructions.
- The SKILL.md file contains excellent tool-routing guidance (decision table, workflow examples, anti-patterns) but this is loaded via Claude Code's skill system, not via MCP's native `serverInstructions`.
- **Problem**: Non-Claude clients (ChatGPT, Copilot, Cursor) don't get this guidance at all.

### Recommended Server Instructions for Magaldi

```python
SERVER_INSTRUCTIONS = """
Magaldi is a code discovery engine with a pre-indexed codebase. All results come from the index, not the live filesystem.

TOOL ROUTING:
- Know what you're looking for by name/pattern? → pattern_search
- Exploring by meaning/description? → search_code
- Need call sites of a known element? → find_usages (requires hash_id from search_code)
- Need full understanding of one element? → explain_element (expensive, use get_element for quick lookups)
- Need recursive call traces? → find_call_chain (not get_call_graph which is 1-level only)

CONVENTIONS:
- scope and repository are auto-detected from magaldi.yaml in the working directory — usually omit
- hash_id parameters require element IDs from previous search results (64-char hex strings)
- Use brief=true or max_tokens for exploration to save context window
- Use filename parameter to save large results to disk instead of consuming context

COMMON WORKFLOWS:
1. "Find where X is called": search_code(query="X") → find_usages(hash_id=result_id)
2. "Understand function X": search_code(query="X") → explain_element(hash_id=result_id)
3. "Refactor X safely": search_code → find_usages → find_call_chain(direction="callers")
"""
```

### Why This Matters

Server instructions are loaded once per session, not per turn. They consume ~200-300 tokens, but they **prevent wasted tool calls** (which cost 500-2000 tokens each). Even one prevented wrong tool call pays for the instructions.

---

## 9c. Block's Playbook — Lessons from 60+ MCP Servers

Block (Square/Cash App) has built 60+ MCP servers. Their key principles align with and extend Anthropic's guidance:

### Principle 1: Workflow-First Design

> "Start top-down from the workflow that needs to be automated, and work backwards (in as few steps as possible) to define tools that support that flow."

**Magaldi assessment**: Our tools are designed bottom-up from capabilities ("here's what OpenSearch can do") rather than top-down from workflows ("here's what an agent needs to do when exploring code"). The SKILL.md compensates by documenting workflows, but the tools themselves don't guide the agent toward optimal paths.

### Principle 2: Consolidate Multi-Step Operations

> "LLMs are improving at planning but it's hard for them to chain 20 tool calls. Design tools that require less chaining."

**Magaldi assessment**: `explain_element` is a good example of consolidation (combines get_element + find_callers + find_callees + find_similar + get_context). We should apply this pattern more:
- A `discover_code` tool that does search_code → get_element for the top result
- An `impact_analysis` tool that does find_usages → find_call_chain in one call

### Principle 3: Clean Schema, Tidy Data

> "Think of creating a 'gold dataset' — easy to query for the end user. Denormalize so fewer joins are required."

**Magaldi assessment**: Our search results already include summaries (denormalized). But call graph results still require follow-up calls to understand what the callers/callees do. Pre-including brief summaries of callers in `get_call_graph` results would reduce chaining.

### Principle 4: Prefer Markdown Over JSON

> "Markdown or XML is typically more token-efficient than raw JSON for tool responses."

**Magaldi assessment**: Our formatters already produce markdown. This is correct.

### Principle 5: Phil Schmid's "MCP is a UI for Non-Human Users"

From the ex-AWS, now Google engineer's widely shared post:

> "When building an MCP Server you are not building infrastructure — you are building an interface for AI agents. Build it like one."

Key additions:
- **Service-prefixed names**: Pattern `{service}_{action}_{resource}` — we do this partially (`find_callers`, `search_code`) but inconsistently (`get_call_graph`, `dependency_graph`)
- **Docstrings are instructions**: Every piece of text is part of the agent's context
- **One server, one job**: Split by persona if needed — a "magaldi-explore" server for search/discovery and "magaldi-analyze" for metrics/security

---

## 9d. Tool Consolidation Deep Dive

### Microsoft's "Tool-Space Interference" Research

Microsoft Research (2025) studied ~1,500 MCP servers and found:
- **Tool name collisions** between servers cause failures
- **Overlapping capabilities** across servers cause the LLM to pick wrong tools
- **State divergence** when multiple paths exist to the same outcome

Their **MCP Interviewer** tool automates compatibility analysis and produces "compatibility scores."

### Consolidation Candidates for Magaldi

**Tier 1: Merge immediately (no user-facing behavior change)**

| Current | Proposed | Rationale |
|---|---|---|
| `get_element` + `batch_get_elements` | `get_element(hash_id OR hash_ids)` | Accept string or array; single code path |
| `list_glossary` + `search_glossary` | `search_glossary(query?)` | No query = list all. With query = search. |

**Tier 2: Merge with mode parameter**

| Current | Proposed | Rationale |
|---|---|---|
| `find_callers` + `get_call_graph` | `get_call_graph(direction, detailed?)` | `find_callers` is really `get_call_graph(direction="callers", detailed=true)` |
| `find_dependencies` + `find_dependents` | `find_dependencies(direction="imports"\|"imported_by")` | Symmetric operation, just reversed |

**Tier 3: Consider but risky**

| Current | Proposed | Risk |
|---|---|---|
| `get_context` + `get_children` | `get_context(include_children=true)` | Already nearly the same; `get_children` returns only children |
| `list_features` + `get_feature_members` | `search_features` could include members | Would make search_features response much larger |

**Net effect**: 44 → ~37 tools (7 eliminated). Combined with `defer_loading`, context drops from ~13K to ~4K tokens.

---

## 10. Key Takeaways

1. **Less is more**: 44 tools is too many. Consolidate to ~37, defer ~21, keep ~16 in context. The industry sweet spot is 10-15 always-loaded tools.
2. **Server instructions are free real estate**: A single string on server init guides the LLM through tool workflows. ~200 tokens that prevent thousands in wasted wrong-tool calls. **Magaldi has none today.**
3. **Descriptions are the #1 lever**: The LLM selects tools ENTIRELY from descriptions. "Use this when... Do NOT use for..." is the proven pattern. **None of Magaldi's 44 tools follow this.**
4. **Errors are prompts**: Raw Python exceptions teach the LLM nothing. Structured errors with recovery guidance enable self-correction. **Magaldi returns `Error: {e}` with no `isError` flag.**
5. **Token efficiency is existential**: Context rot means extra tokens actively *reduce* accuracy. This isn't optimization — it's correctness. 44 tools at ~300 tokens each = ~13K tokens before any work begins.
6. **Four hierarchical architectures exist**: `defer_loading` (easiest, Claude-only), meta-tool pattern (client-agnostic, ~90% savings), hierarchical strata (100+ tools), namespaces (spec pending). Magaldi should do `defer_loading` now, meta-tool later.
7. **Consolidate like GitLab**: "Can it be an enum/parameter on an existing tool?" eliminates 5-7 tools immediately. `get_element` + `batch_get_elements`, glossary trio, dependency pair.
8. **Workflow-first design** (Block): Tools should support agent workflows, not mirror API capabilities. `explain_element` is a good example — more tools like it.
9. **Magaldi's strengths are real**: AI summaries, output control (max_tokens/filename/brief), correct annotations, rich semantic search, three embedding types. The foundation is solid — the work is in *presentation*, not substance.

### Quick-Win Implementation Order

| Priority | Action | Effort | Token Impact |
|---|---|---|---|
| 1 | Add server instructions | 30 min | Prevents ~2-5 wasted tool calls/session |
| 2 | Rewrite tool descriptions | 2-3 hrs | Correct tool selection (accuracy) |
| 3 | Structured error messages + isError | 1-2 hrs | Agent self-correction |
| 4 | Add descriptions to bare params | 30 min | Better parameter usage |
| 5 | Tier 1 tool consolidation | 4-6 hrs | 44 → 38 tools (~1.8K saved) |
| 6 | `defer_loading` annotations | 1 hr | 44 → 16 in context (~7.8K saved) |
| 7 | Response format enum | 2-3 hrs | Better output control |
| 8 | Meta-tool mode (optional) | 8-12 hrs | 44 → 2+N in context (~11K saved) |

---

## Sources

- [MCP Specification 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25)
- [MCP Specification 2025-06-18 — Tools](https://modelcontextprotocol.io/specification/2025-06-18/server/tools)
- [Anthropic: Writing Effective Tools for AI Agents](https://www.anthropic.com/engineering/writing-tools-for-agents)
- [Anthropic: Advanced Tool Use](https://www.anthropic.com/engineering/advanced-tool-use)
- [Anthropic: Code Execution with MCP](https://www.anthropic.com/engineering/code-execution-with-mcp)
- [Anthropic: Tool Search Tool Docs](https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-search-tool)
- [SEP-1576: Mitigating Token Bloat](https://github.com/modelcontextprotocol/modelcontextprotocol/issues/1576)
- [SEP-1382: Documentation Best Practices for MCP Tools](https://github.com/modelcontextprotocol/modelcontextprotocol/issues/1382)
- [10 Strategies to Reduce MCP Token Bloat — The New Stack](https://thenewstack.io/how-to-reduce-mcp-token-bloat/)
- [The Meta-Tool Pattern — Synapticlabs](https://blog.synapticlabs.ai/bounded-context-packs-meta-tool-pattern)
- [Dynamic ReAct: Scalable Tool Selection (arXiv)](https://arxiv.org/html/2509.20386v1)
- [MCP-Zero: Active Tool Discovery (arXiv)](https://arxiv.org/html/2506.01056v3)
- [Speakeasy: Dynamic Tool Discovery](https://www.speakeasy.com/mcp/tool-design/dynamic-tool-discovery)
- [Speakeasy: 100x Token Reduction](https://www.speakeasy.com/blog/100x-token-reduction-dynamic-toolsets)
- [Spring AI: Dynamic Tool Discovery (34-64% savings)](https://spring.io/blog/2025/12/11/spring-ai-tool-search-tools-tzolov/)
- [CodeRabbit: Ballooning Context in the MCP Era](https://www.coderabbit.ai/blog/handling-ballooning-context-in-the-mcp-era-context-engineering-on-steroids)
- [Klavis AI: Less is More — 4 MCP Design Patterns](https://www.klavis.ai/blog/less-is-more-mcp-design-patterns-for-ai-agents)
- [Merge.dev: MCP Tool Descriptions Best Practices](https://www.merge.dev/blog/mcp-tool-description)
- [MCPcat: Error Handling Best Practices](https://mcpcat.io/guides/error-handling-custom-mcp-servers/)
- [Claude Code Tool Search (46.9% reduction)](https://medium.com/@joe.njenga/claude-code-just-cut-mcp-context-bloat-by-46-9-51k-tokens-down-to-8-5k-with-new-tool-search-ddf9e905f734)
- [Unified.to: Scaling MCP with Defer Loading](https://unified.to/blog/scaling_mcp_tools_with_anthropic_defer_loading)
- [MCP 2025-11-25 Anniversary Release](http://blog.modelcontextprotocol.io/posts/2025-11-25-first-mcp-anniversary/)
- [Serena: Semantic Code Agent Toolkit](https://github.com/oraios/serena)
- [Zilliz Claude Context](https://github.com/zilliztech/claude-context)
- [Itential: Context as the New Currency](https://www.itential.com/blog/company/ai-networking/context-as-the-new-currency-designing-effective-mcp-servers-for-ai/)
- [OpenAI Cookbook: MCP Tool Guide](https://cookbook.openai.com/examples/mcp/mcp_tool_guide)
- [Progressive Tool Discovery — Agentic Patterns](https://agentic-patterns.com/patterns/progressive-tool-discovery/)
- [Stacklok MCP Optimizer](https://docs.stacklok.com/toolhive/tutorials/mcp-optimizer)
- [Cisco: MCP Elicitation, Structured Content, OAuth](https://blogs.cisco.com/developer/whats-new-in-mcp-elicitation-structured-content-and-oauth-enhancements)
- [WorkOS: MCP 2025-11-25 Spec Update](https://workos.com/blog/mcp-2025-11-25-spec-update)
- [Block's Playbook for Designing MCP Servers](https://engineering.block.xyz/blog/blocks-playbook-for-designing-mcp-servers)
- [Phil Schmid: MCP is Not the Problem, It's Your Server](https://www.philschmid.de/mcp-best-practices)
- [Microsoft: Tool-Space Interference in the MCP Era](https://www.microsoft.com/en-us/research/blog/tool-space-interference-in-the-mcp-era-designing-for-agent-compatibility-at-scale/)
- [Anthropic: Equipping Agents with Agent Skills](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills)
- [Anthropic: Effective Context Engineering for AI Agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)
- [Alpic AI: Better MCP Error Responses](https://alpic.ai/blog/better-mcp-tool-call-error-responses-ai-recover-gracefully)
- [MCP Blog: Server Instructions](http://blog.modelcontextprotocol.io/posts/2025-11-03-using-server-instructions/)
- [Synapticlabs: The Tool Bloat Tipping Point](https://blog.synapticlabs.ai/bounded-context-packs-tool-bloat-tipping-point)
- [MCP Discussion #532: Hierarchical Tool Management](https://github.com/orgs/modelcontextprotocol/discussions/532)
- [SEP-986: Tool Name Format Specification](https://github.com/modelcontextprotocol/modelcontextprotocol/issues/986)
- [MCP Python SDK Issue #829: Tool Grouping Support](https://github.com/modelcontextprotocol/python-sdk/issues/829)
- [ECF/MCPToolGroups: Tool Groups for MCP](https://github.com/ECF/MCPToolGroups)
- [Dynamic ReAct: Five Architectures for Scalable Tool Selection](https://arxiv.org/html/2509.20386v1)
- [kvg.dev: Stop Drowning Your Agent in Tools](https://kvg.dev/posts/20260110-tool-bloat-ai-agents/)
- [Gong MCP Tool Description Example](https://www.merge.dev/blog/mcp-tool-description)
- [Modular MCP: On-Demand Tool Loading Proxy](https://github.com/d-kimuson/modular-mcp)
- [SEP-993: Namespaces for MCP](https://github.com/modelcontextprotocol/modelcontextprotocol/issues/993)
- [MetaMCP: Namespace Documentation](https://docs.metamcp.com/en/concepts/namespaces)
- [GitLab MCP Server Development Guidelines](https://docs.gitlab.com/development/mcp_server/)
- [SEP-1888: Progressive Disclosure for Library Discovery](https://github.com/modelcontextprotocol/modelcontextprotocol/issues/1888)
- [GitHub MCP Server: Consolidation Changelog](https://github.blog/changelog/2025-10-29-github-mcp-server-now-comes-with-better-tools/)
- [Anthropic: Introducing Advanced Tool Use](https://www.anthropic.com/engineering/advanced-tool-use)
- [Anthropic: Defer Loading for MCP Toolsets](https://unified.to/blog/scaling_mcp_tools_with_anthropic_defer_loading)
- [SEP-1624: Structured Content vs Content Usage Guidance](https://github.com/modelcontextprotocol/modelcontextprotocol/issues/1624)
