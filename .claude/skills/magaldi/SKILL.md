---
name: magaldi
description: >
  ALWAYS use for: grep, find usages, search patterns, find implementations,
  call graphs, find where X is used/called, search code by meaning.
  These tools use the PRE-INDEXED codebase for faster, richer results than raw file search.
  Invoke BEFORE using built-in Grep/Glob/Read tools.
---

# Magaldi Code Discovery

**CRITICAL: Use magaldi tools INSTEAD OF built-in Grep/Glob for code search.**

You can search for tools to: search code semantically, find files by pattern,
grep/regex search, find where functions are called, trace call chains,
analyze dependencies and imports, detect design patterns, find dead code,
audit security issues, find complex functions, list HTTP routes and CLI commands,
explore glossary terms, and inspect code elements with AI summaries.

The codebase is pre-indexed with:
- Semantic embeddings (search by meaning)
- Pre-computed summaries (understand without reading)
- Call graphs (who calls what)
- Feature clustering (related functions grouped)

## When to Use Magaldi vs Built-in Tools

| User Request | USE THIS | NOT THIS |
|--------------|----------|----------|
| "grep for X" / "find pattern X" | `mcp__magaldi__pattern_search` (mode="regexp") | Built-in Grep |
| "find where X is used/called" | `mcp__magaldi__find_usages` | Built-in Grep |
| "search for functions that do X" | `mcp__magaldi__search_code` | Built-in Grep |
| "find files matching *.py" | `mcp__magaldi__find_files` | Built-in Glob |
| "what implements Interface X" | `mcp__magaldi__find_implementations` | Built-in Grep |
| "who calls this function" | `mcp__magaldi__get_call_graph` | Built-in Grep |
| "trace call chain from A to B" | `mcp__magaldi__find_call_chain` | N/A |
| "find dead code" | `mcp__magaldi__find_dead_code` | N/A |
| "find entry points" | `mcp__magaldi__find_entry_points` | N/A |
| "what does this file import" | `mcp__magaldi__find_dependencies` | N/A |
| "what files import this module" | `mcp__magaldi__find_dependents` | N/A |
| "show module dependency graph" | `mcp__magaldi__dependency_graph` | N/A |
| "explain this element completely" | `mcp__magaldi__explain_element` | N/A |
| "find similar code to X" | `mcp__magaldi__find_similar` | N/A |
| "find code with same structure" | `mcp__magaldi__find_similar_structure` | N/A |
| "find code with same purpose" | `mcp__magaldi__find_similar_intent` | N/A |
| "find duplicate code" | `mcp__magaldi__find_duplicates` | N/A |
| "what does the codebase do" | `mcp__magaldi__search_features` | N/A |

## Why Magaldi Tools Are Better

| Feature | Magaldi | Built-in Grep/Glob |
|---------|---------|-------------------|
| Pre-indexed | Yes - instant results | No - scans every file |
| Summaries | Every function has AI summary | None |
| Semantic search | "authentication" finds login, auth, verify | Only literal matches |
| Call graphs | Built-in | Must grep manually |
| Context | Parent class, siblings, children | Just file/line |

## Tool Priority (Use in This Order)

### 1. SEMANTIC SEARCH (Start Here for "what does X do")
```
mcp__magaldi__search_code(query="authentication logic", brief=true)
```
- Natural language: "function that validates tokens"
- Returns summaries, not just file:line
- Use `brief=true` for exploration
- Use `search_mode` to control embedding search:
  - `"summary"`: Search by what code DOES (intent/purpose)
  - `"code"`: Search by what code LOOKS LIKE (structure)
  - `"hybrid"` (default): Search both, merge results

### 2. PATTERN SEARCH (For literal patterns, regex, wildcards)
```
mcp__magaldi__pattern_search(pattern="add_job.*\\(", mode="regexp", scope="...", repository="...")
```
- **Three modes:**
  - `regexp`: Lucene regex (e.g., `"add_column.*Model"`)
  - `wildcard`: Simple wildcards (e.g., `"*column*Model*"`)
  - `proximity`: Terms near each other (e.g., `"add column Model"` with slop=5)
- ES-native - queries run server-side for better performance
- Requires `scope` and `repository` parameters

**Note:** `grep_code` is deprecated - use `pattern_search` with `mode="regexp"` instead.

### 3. USAGE TRACKING (For "where is X called")
```
mcp__magaldi__find_usages(element_id="...")
```
- After search_code found the element
- Shows all call sites with context
- Filters out definitions automatically

### 4. RELATIONSHIPS (For refactoring, impact analysis)
```
mcp__magaldi__get_call_graph(element_id="...")
mcp__magaldi__find_implementations(class_name="BaseClass")
```
- Before modifying shared code
- Understanding dependencies

### 5. SIMILARITY SEARCH (For finding related code)
```
mcp__magaldi__find_similar_structure(element_id="...")  # Same structure/patterns
mcp__magaldi__find_similar_intent(element_id="...")     # Same purpose/behavior
mcp__magaldi__find_duplicates(scope="...", repository="...")  # Duplicate code detection
```
- `find_similar_structure`: Find code that LOOKS similar (copy-paste, patterns)
- `find_similar_intent`: Find code that DOES similar things (different implementations)
- `find_duplicates`: Find near-duplicate functions for refactoring

### 6. CALL ANALYSIS (For understanding code flow)
```
mcp__magaldi__find_callers(element_id="...")           # Who calls this function
mcp__magaldi__find_call_chain(element_id="...", direction="both", max_depth=5)  # Trace call chains
mcp__magaldi__find_dead_code(scope="...", repository="...")  # Find uncalled functions
mcp__magaldi__find_entry_points(scope="...", repository="...")  # Find HTTP handlers, CLI commands
```
- `find_callers`: Simple "who calls this" (separates code vs tests)
- `find_call_chain`: Recursive tracing (A → B → C → D)
- `find_dead_code`: Identify functions never called (excludes entry points)
- `find_entry_points`: Find @route, @command, main functions

### 7. DEPENDENCY ANALYSIS (For understanding imports)
```
mcp__magaldi__find_dependencies(file_path="src/utils.py", scope="...", repository="...")
mcp__magaldi__find_dependents(module="utils", scope="...", repository="...")
mcp__magaldi__dependency_graph(scope="...", repository="...")  # Full module graph with cycle detection
```
- `find_dependencies`: What does this file import (internal vs external)
- `find_dependents`: What files import this module (reverse dependency)
- `dependency_graph`: Full module-level graph with circular dependency detection

### 8. META TOOLS (Comprehensive views)
```
mcp__magaldi__explain_element(element_id="...")  # Complete element overview
```
- `explain_element`: One-call comprehensive view:
  - Element details (name, type, signature, summary)
  - Top 5 callers
  - All direct callees
  - Imports (for files)
  - Top 3 similar code
  - Parent context

## Workflow Examples

### "Grep for X" / "Find pattern X"
```
1. mcp__magaldi__pattern_search(pattern="X", mode="regexp", scope="...", repository="...")
   - NOT: built-in Grep tool
   - For wildcards: mode="wildcard" with patterns like "*X*"
   - For proximity: mode="proximity" with slop parameter
```

### "Find where function X is called"
```
1. mcp__magaldi__search_code(query="X", element_types=["function"])
2. mcp__magaldi__find_usages(element_id=result.element_id)
   - NOT: grep for "X("
```

### "What implements interface Y"
```
1. mcp__magaldi__find_implementations(class_name="Y")
   - NOT: grep for "class.*Y"
```

### "How does X work"
```
1. mcp__magaldi__search_code(query="X functionality", brief=true)
2. mcp__magaldi__get_element(element_id=best_match, include_code=true)
   - NOT: grep then read file
```

### "Find all authentication code"
```
1. mcp__magaldi__search_features(query="authentication")
2. mcp__magaldi__get_feature_members(feature_id=result.feature_id)
   - Returns grouped, related functions
```

### "Refactor function Z"
```
1. mcp__magaldi__search_code(query="Z")
2. mcp__magaldi__find_usages(element_id)  # Impact analysis
3. mcp__magaldi__get_call_graph(element_id)  # Dependencies
4. THEN make changes
```

### "Find dead code / cleanup opportunities"
```
1. mcp__magaldi__find_dead_code(scope="...", repository="...")
   - Returns functions with no callers
   - Excludes entry points (@route, @command, main)
   - Review before deleting
```

### "Understand entry points for API/CLI"
```
1. mcp__magaldi__find_entry_points(scope="...", repository="...")
   - Groups by type: HTTP, CLI, test fixtures, main
   - Shows decorators and file locations
```

### "Trace how A ends up calling B"
```
1. mcp__magaldi__find_call_chain(element_id=A, direction="callees", max_depth=5)
   - Shows: A → X → Y → B
   - Detects cycles
```

### "Check circular dependencies"
```
1. mcp__magaldi__dependency_graph(scope="...", repository="...")
   - Returns nodes, edges, cycles
   - Shows circular import chains
```

### "Understand an element completely"
```
1. mcp__magaldi__explain_element(element_id="...")
   - Returns: element details, callers, callees, imports, similar code, parent
   - One call instead of 5 separate calls
```

## Deferred Tool Loading (API Users)

When using Magaldi via the Anthropic API with `mcp_toolset`, configure deferred loading to save context tokens. Default all tools to deferred, keep the top 5 most-used always loaded:

```json
{
  "type": "mcp_toolset",
  "mcp_server_name": "magaldi",
  "default_config": { "defer_loading": true },
  "configs": {
    "search_code": { "defer_loading": false },
    "find_files": { "defer_loading": false },
    "get_element": { "defer_loading": false },
    "pattern_search": { "defer_loading": false },
    "get_file_structure": { "defer_loading": false }
  }
}
```

Claude Code users: deferred loading is automatic when tool descriptions exceed 10K tokens (Magaldi qualifies).

## Anti-Patterns (NEVER Do These)

1. **Using built-in Grep instead of magaldi__pattern_search**
   - Magaldi pattern_search runs queries server-side in OpenSearch
   - Built-in Grep scans files one by one

2. **Using deprecated grep_code**
   - Use `pattern_search` with `mode="regexp"` instead
   - grep_code is deprecated and will be removed

4. **Using built-in Glob instead of magaldi__find_files**
   - Magaldi knows which files are indexed

5. **Grepping for function calls instead of find_usages**
   - find_usages filters definitions, has context

6. **Reading whole files to understand them**
   - Use search_code → get_element with summaries

7. **Skipping semantic search**
   - Summaries save tokens, embeddings find related code

## Available Tools Quick Reference

| Tool | Purpose |
|------|---------|
| `search_code` | Semantic search by meaning (supports `search_mode`: summary/code/hybrid) |
| `search_features` | Find high-level capabilities |
| `pattern_search` | **ES-native pattern search** - regexp, wildcard, or proximity mode |
| `grep_code` | ~~Deprecated~~ - use `pattern_search` with mode="regexp" |
| `find_usages` | Where is this called/used (uses ES regexp internally) |
| `find_implementations` | What implements this interface (uses ES regexp internally) |
| `get_call_graph` | Callers and callees for a single element |
| `find_callers` | Who calls this function (separates code vs tests) |
| `find_call_chain` | Trace call chains recursively (A → B → C) |
| `find_dead_code` | Find functions never called (excludes entry points) |
| `find_entry_points` | Find HTTP handlers, CLI commands, main functions |
| `find_dependencies` | What does a file import (internal vs external) |
| `find_dependents` | What files import this module |
| `dependency_graph` | Full module dependency graph with cycle detection |
| `explain_element` | **Comprehensive element overview** (callers, callees, similar, parent) |
| `find_similar` | Similar code patterns (uses both embeddings) |
| `find_similar_structure` | Find structurally similar code (code embedding) |
| `find_similar_intent` | Find code with similar purpose (summary embedding) |
| `find_duplicates` | Find near-duplicate code for refactoring |
| `get_element` | Full element details |
| `get_context` | Parent, siblings, children |
| `find_files` | Glob pattern search (USE THIS not built-in Glob) |
| `list_features` | All features in repo |
| `get_feature_members` | Functions in a feature |
| `list_repos` | All indexed repos |
| `get_repo_stats` | Repository statistics |

## Token-Saving Parameters

High-volume tools support `max_tokens` and `filename` for output control:

```
# Limit response size (drops trailing results to fit budget)
mcp__magaldi__search_code(query="auth", max_tokens=500)

# Save full results to file, get summary inline
mcp__magaldi__find_dead_code(filename="/tmp/dead_code.md")
```

**Supported tools:** search_code, search_features, find_usages, pattern_search,
find_callers, find_call_chain, explain_element, find_dead_code, find_entry_points.

## Subagent Delegation

**Delegate to Explore subagent when:**
- Multi-step workflows (search -> inspect -> find_usages)
- Exploratory queries with unknown result count
- Results won't be directly referenced in the next action

**Call inline when:**
- Single-call lookups with known hash_id (get_element, get_context)
- Quick searches expected to return <5 results
- Results needed for the immediate next action

## Remember

The index has already done the hard work:
- Code is parsed and structured
- Summaries explain what code does
- **Dual embeddings** enable semantic search:
  - Summary embedding: search by intent/purpose
  - Code embedding: search by structure/patterns
- **Call graphs are pre-indexed**:
  - Imports extracted from file elements
  - Function calls extracted with receivers
  - Call targets resolved to element IDs
- **Dependency graphs** built from imports

**Use magaldi tools. Don't re-grep what's already indexed.**
