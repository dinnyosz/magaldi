# MCP Search Improvements Design

**Date:** 2026-01-23
**Status:** Draft

## Overview

This design improves how Claude Code discovers code through Magaldi's MCP by addressing three gaps:

| Problem | Current State | Solution |
|---------|---------------|----------|
| Semantic search only covers summaries | `search_code` embeds AI summaries, not actual code | Add `code_embedding` field, search both |
| Pattern search is inefficient | `grep_code` pulls all data client-side, applies Python regex | Replace with ES-native `pattern_search` |
| Call graph is regex-based and incomplete | `get_call_graph` uses runtime regex, doesn't resolve calls | Extract calls/imports at parse time via tree-sitter |

## Goals

1. **Semantic code search** - Find code by what it does AND what it looks like
2. **Fast pattern matching** - Server-side ES queries instead of client-side regex
3. **Accurate call graphs** - Pre-computed, resolved call relationships
4. **Dependency analysis** - Track imports and module relationships
5. **Code quality tools** - Dead code detection, duplicate finding
6. **Rich web UI** - Visual exploration of all code relationships

## Multi-User Model

Magaldi uses a layered indexing model where user changes overlay the main branch:

### Branches

| Branch | Purpose | Indexed By |
|--------|---------|------------|
| `main` | Universal truth, full codebase | CI/central system |
| `<username>` | User's local changes only | User's local indexer |

### Query Behavior

When querying with `username="alice"`:

```
1. Search for element in alice's index
2. If found → return alice's version
3. If not found → fall back to main's version
4. Return result (alice's changes shadow main)
```

This allows:
- Users to see their local changes immediately
- Unchanged code to be queried from main
- No need to re-index the entire codebase for local changes

### Implementation

All query methods should implement this fallback logic:

```python
def get_element_with_fallback(element_id: str, username: str) -> dict | None:
    # Try user's version first
    if username != "main":
        result = query_by_username(element_id, username)
        if result:
            return result

    # Fall back to main
    return query_by_username(element_id, "main")
```

For search queries (semantic, pattern), results should merge:
1. Get results from username (if not main)
2. Get results from main
3. Deduplicate by element_id (username version wins)
4. Return merged results

## Context Detection

Tools require `scope`, `repository`, and `username` parameters. These are detected/configured as follows:

### Configuration Sources

| Parameter | Source | Committed? |
|-----------|--------|------------|
| `scope` | `magaldi.yaml` in repo root | Yes |
| `repository` | `magaldi.yaml` in repo root | Yes |
| `username` | OS user detection | No |

### magaldi.yaml

Every indexed repository must have a `magaldi.yaml` in its root:

```yaml
scope: mycompany           # Required: organization/namespace
repository: backend-api    # Required: repository name
```

This file is committed to the repository as it's project-specific.

### Username Detection

Auto-detected from the OS user running Claude Code:

```python
import getpass
import os

def get_username() -> str:
    # Allow explicit override via environment
    if os.environ.get("MAGALDI_USER"):
        return os.environ["MAGALDI_USER"]

    # Auto-detect from OS
    return getpass.getuser()
```

### Tool Behavior

When `scope`/`repository` not provided in a tool call:

```python
# Return error with guidance
return {
    "error": "scope and repository required",
    "hint": "Read magaldi.yaml from repo root, or run init_config to create one"
}
```

### Skill Behavior

The Magaldi skill should:

1. **On startup:** Check for `magaldi.yaml` in current directory
2. **If found:** Read scope/repository, detect username
3. **If not found:** Call `init_config` tool, ask user to verify values
4. **Store context:** Use detected values for all subsequent tool calls

## Schema Changes

### New fields on all elements

```python
# Rename existing embedding field for clarity
"summary_embedding": {
    "type": "dense_vector",
    "dims": 1024,
    "index": True,
    "similarity": "cosine",
}

# New: embedding of raw_code
"code_embedding": {
    "type": "dense_vector",
    "dims": 1024,
    "index": True,
    "similarity": "cosine",
}
```

### New fields on function/method elements

```python
# Calls made within this element
"calls": {
    "type": "nested",
    "properties": {
        "name": {"type": "keyword"},        # Function name: "process"
        "receiver": {"type": "keyword"},    # Self/object: "self", "utils", null
        "line": {"type": "integer"},        # Line number within element
        "resolved_id": {"type": "keyword"}, # Resolved element ID (if found)
    }
}
```

### New fields on file elements

```python
# Imports in this file
"imports": {
    "type": "nested",
    "properties": {
        "name": {"type": "keyword"},        # Imported name: "process"
        "module": {"type": "keyword"},      # Source: "utils", "pathlib"
        "alias": {"type": "keyword"},       # Alias if any: "pd", null
    }
}
```

The `nested` type allows querying inside arrays (e.g., find all elements where `calls.resolved_id = X`).

## Tool Changes

### Modified: `search_code`

Add `search_mode` parameter with `hybrid` as default:

```python
search_code(
    query: str,
    search_mode: "summary" | "code" | "hybrid" = "hybrid",  # NEW
    # ... existing params unchanged
)
```

| Mode | Behavior |
|------|----------|
| `summary` | Search only summary embeddings (current behavior) |
| `code` | Search only code embeddings |
| `hybrid` | Search both, combine scores (default) |

Hybrid scoring combines both embedding similarities. Default weights: `0.5 * summary_score + 0.5 * code_score`.

### New: `pattern_search`

Replaces `grep_code` with ES-native pattern matching:

```python
pattern_search(
    pattern: str,
    mode: "regexp" | "wildcard" | "proximity",  # Required, explicit
    slop: int = 5,              # For proximity mode: max words between terms
    scope: str | None = None,
    repository: str | None = None,
    glob: str | None = None,    # File filter
    limit: int = 50,
    include_tests: bool = True,
)
```

| Mode | ES Query Type | Use Case | Example |
|------|---------------|----------|---------|
| `regexp` | `regexp` query | Complex patterns | `add_column.*Model` |
| `wildcard` | `wildcard` query | Simple glob-like | `*column*Model*` |
| `proximity` | `match_phrase` + slop | Terms near each other | `add_column Model` |

Mode is explicit (not auto-detected) for predictability.

### Modified: `get_call_graph`

No API changes. Internal implementation changes:

**Before:** Regex on raw_code at query time
**After:**
- Callees: Read pre-indexed `calls` field (instant)
- Callers: Query for elements where `calls.resolved_id` matches target (ES query)

### Updated: `find_usages`, `find_implementations`

Switch internal implementation from client-side grep to `pattern_search` with `regexp` mode.

### Deprecated: `grep_code`

Keep for one release cycle with deprecation warning, then remove.

## New Tools

All tools require `scope`, `repository`, and `username` parameters for consistency with the multi-user branching model.

### Call/Dependency Analysis Tools

#### `find_callers`

Find all functions that call a given function.

```python
find_callers(
    scope: str,
    repository: str,
    username: str = "main",
    element_id: str,
    limit: int = 30,
    include_tests: bool = True,
)
```

**Implementation:** Query `WHERE calls.resolved_id = element_id`

**Returns:** List of calling elements with file, line, and call context.

---

#### `find_call_chain`

Trace a chain of calls from A → B → C → D.

```python
find_call_chain(
    scope: str,
    repository: str,
    username: str = "main",
    element_id: str,
    direction: "callers" | "callees" | "both" = "callees",
    max_depth: int = 5,
)
```

**Implementation:** Recursive traversal of `calls.resolved_id` (callees) or reverse query (callers).

**Returns:** Tree structure showing call chain with depth levels.

---

#### `find_dead_code`

Find functions/methods that are never called.

```python
find_dead_code(
    scope: str,
    repository: str,
    username: str = "main",
    include_tests: bool = False,  # Usually want to exclude test code
)
```

**Implementation:**
1. Get all functions/methods
2. Filter to those with zero callers (no element has `calls.resolved_id` pointing to them)
3. Exclude known entry points (decorated with `@app.route`, `@cli.command`, `main`, etc.)

**Returns:** List of potentially dead code elements.

---

#### `find_entry_points`

Find entry points: handlers, CLI commands, main functions.

```python
find_entry_points(
    scope: str,
    repository: str,
    username: str = "main",
)
```

**Implementation:** Find elements that:
- Have decorator patterns: `@app.route`, `@click.command`, `@pytest.fixture`, etc.
- Are named `main`, `__main__`
- Are called externally but have no internal callers

**Returns:** List of entry point elements grouped by type (HTTP, CLI, test, etc.).

---

### Import/Dependency Analysis Tools

#### `find_dependencies`

What does a file or module import?

```python
find_dependencies(
    scope: str,
    repository: str,
    username: str = "main",
    file_path: str | None = None,    # Specific file
    element_id: str | None = None,   # Or by element ID
)
```

**Implementation:** Read file element's `imports` field.

**Returns:** List of imports with module paths and whether they're internal or external.

---

#### `find_dependents`

What files import this module/file?

```python
find_dependents(
    scope: str,
    repository: str,
    username: str = "main",
    module: str,                     # Module name: "utils", "shared.config"
)
```

**Implementation:** Query `WHERE imports.module = module`

**Returns:** List of files that import the specified module.

---

#### `dependency_graph`

Build a module-level dependency graph.

```python
dependency_graph(
    scope: str,
    repository: str,
    username: str = "main",
    internal_only: bool = True,      # Exclude external packages
)
```

**Implementation:** Aggregate all file imports, build directed graph of module → module dependencies.

**Returns:** Graph structure with nodes (modules) and edges (imports), suitable for visualization.

---

### Similarity Tools

#### `find_similar_structure`

Find code that looks structurally similar (same patterns, similar syntax).

```python
find_similar_structure(
    scope: str,
    repository: str,
    username: str = "main",
    element_id: str,
    min_similarity: float = 0.8,
    limit: int = 10,
)
```

**Implementation:** Vector search on `code_embedding` field.

**Returns:** Elements with high structural similarity, useful for finding copy-paste code or similar patterns.

---

#### `find_similar_intent`

Find code that does similar things (same purpose, different implementation).

```python
find_similar_intent(
    scope: str,
    repository: str,
    username: str = "main",
    element_id: str,
    min_similarity: float = 0.7,
    limit: int = 10,
)
```

**Implementation:** Vector search on `summary_embedding` field.

**Returns:** Elements with similar purpose but potentially different structure.

---

#### `find_duplicates`

Find near-duplicate code across the codebase.

```python
find_duplicates(
    scope: str,
    repository: str,
    username: str = "main",
    min_similarity: float = 0.95,    # High threshold for "duplicate"
    min_lines: int = 5,              # Ignore tiny functions
)
```

**Implementation:**
1. Get all function/method elements above min_lines
2. For each, find others with `code_embedding` similarity >= threshold
3. Cluster into duplicate groups

**Returns:** Groups of duplicate/near-duplicate code, sorted by group size.

---

### Meta Tools

#### `explain_element`

Comprehensive overview of an element: what it does, who uses it, what it uses.

```python
explain_element(
    scope: str,
    repository: str,
    username: str = "main",
    element_id: str,
)
```

**Implementation:** Aggregate multiple queries into one response.

**Returns:**
```
{
    "element": { name, type, file, signature, summary },
    "callers": [ ... top 5 callers ... ],
    "callees": [ ... all direct calls ... ],
    "imports": [ ... if file element ... ],
    "similar_code": [ ... top 3 similar ... ],
    "parent": { ... parent class/file ... },
}
```

Saves Claude from making 5+ separate tool calls for common exploration patterns.

---

#### `diff_element`

Compare an element between user branch and main. Shows what changed locally.

```python
diff_element(
    scope: str,
    repository: str,
    username: str,           # The user branch to compare (not "main")
    element_id: str,
)
```

**Implementation:**
1. Fetch element from username's index
2. Fetch element from main's index
3. Compare and generate diff

**Returns:**
```python
{
    "element_id": "...",
    "status": "added" | "modified" | "deleted" | "unchanged",
    "main": {                    # Element from main (null if added)
        "name": "...",
        "signature": "...",
        "raw_code": "...",
        ...
    },
    "user": {                    # Element from username (null if deleted)
        "name": "...",
        "signature": "...",
        "raw_code": "...",
        ...
    },
    "diff": {
        "summary": "Signature changed, 5 lines added",
        "code_diff": "--- main\n+++ user\n@@ -10,5 +10,10 @@\n ...",  # unified diff
        "fields_changed": ["signature", "raw_code", "line_end"],
    }
}
```

**Status values:**
- `added` - Element exists in user but not in main (new code)
- `modified` - Element exists in both but differs
- `deleted` - Element exists in main but not in user (removed code)
- `unchanged` - Element is identical in both

**Use cases:**
- "What did I change in this function?"
- "Review my local changes before committing"
- Code review of local modifications

---

#### `diff_repository`

Compare all changes between user branch and main for a repository.

```python
diff_repository(
    scope: str,
    repository: str,
    username: str,           # The user branch to compare
    include_unchanged: bool = False,
)
```

**Implementation:**
1. Get all elements from username's index
2. Get all elements from main's index
3. Compare and categorize by status

**Returns:**
```python
{
    "summary": {
        "added": 5,
        "modified": 12,
        "deleted": 2,
        "unchanged": 150,
    },
    "changes": [
        {"element_id": "...", "status": "added", "name": "new_function", "file": "..."},
        {"element_id": "...", "status": "modified", "name": "existing_func", "file": "..."},
        {"element_id": "...", "status": "deleted", "name": "old_function", "file": "..."},
    ]
}
```

**Use cases:**
- "What have I changed in this repo?"
- "Show me all my local modifications"
- Pre-commit review of all changes

---

#### `init_config`

Generate `magaldi.yaml` with auto-detected values. Used when config doesn't exist.

```python
init_config(
    directory: str,          # Directory to create magaldi.yaml in
    scope: str | None = None,        # Override auto-detected scope
    repository: str | None = None,   # Override auto-detected repository
)
```

**Auto-detection logic:**

```python
def detect_config(directory: str) -> dict:
    # 1. Try to get repository name from git remote
    #    git remote get-url origin → "git@github.com:mycompany/backend-api.git"
    #    Extract: repository = "backend-api"

    # 2. Try to get scope from git remote org/user
    #    Extract: scope = "mycompany"

    # 3. Fallback: use directory name for repository
    #    /Users/alice/code/my-project → repository = "my-project"

    # 4. Fallback: use parent directory for scope
    #    /Users/alice/code/my-project → scope = "code" (or prompt user)

    return {
        "scope": detected_scope,
        "repository": detected_repository,
        "confidence": "high" | "medium" | "low",
        "source": "git_remote" | "directory_name" | "fallback",
    }
```

**Returns:**
```python
{
    "created": True,
    "path": "/path/to/magaldi.yaml",
    "values": {
        "scope": "mycompany",
        "repository": "backend-api",
    },
    "detection": {
        "confidence": "high",
        "source": "git_remote",
    },
    "message": "Created magaldi.yaml. Please verify these values are correct."
}
```

**Skill integration:**

When `magaldi.yaml` not found:
1. Call `init_config` to generate it
2. Show detected values to user
3. Ask user to verify/edit before continuing
4. Only proceed with tools after confirmation

**Example skill flow:**
```
Claude: I couldn't find magaldi.yaml in this repo. Let me create one.
Claude: [calls init_config]
Claude: I've created magaldi.yaml with:
        - scope: mycompany (detected from git remote)
        - repository: backend-api (detected from git remote)

        Please verify these values are correct before I continue.
        Should I proceed with these values?
```

---

#### `get_context`

Get the current context (scope, repository, username) for this session.

```python
get_context(
    directory: str | None = None,    # Directory to detect from (default: cwd)
)
```

**Returns:**
```python
{
    "scope": "mycompany",
    "repository": "backend-api",
    "username": "alice",
    "config_path": "/path/to/magaldi.yaml",
    "config_exists": True,
}
```

**Use cases:**
- Skill startup to get current context
- Verify context before operations
- Debug context detection issues

## Tree-sitter Extraction

### Python

**Imports** (from file root):

| Code | Extracted |
|------|-----------|
| `import os` | `{"name": "os", "module": "os", "alias": null}` |
| `from utils import process as p` | `{"name": "process", "module": "utils", "alias": "p"}` |

Node types: `import_statement`, `import_from_statement`

**Calls** (from functions/methods):

| Code | Extracted |
|------|-----------|
| `process(x)` | `{"name": "process", "receiver": null, "line": 45}` |
| `self.validate()` | `{"name": "validate", "receiver": "self", "line": 48}` |
| `utils.run()` | `{"name": "run", "receiver": "utils", "line": 52}` |

Node types: `call`

### JavaScript/TypeScript

**Imports**:

| Code | Extracted |
|------|-----------|
| `import { foo } from './utils'` | `{"name": "foo", "module": "./utils", "alias": null}` |
| `const bar = require('lib')` | `{"name": "bar", "module": "lib", "alias": null}` |

Node types: `import_statement`, `call_expression` (for require)

**Calls**: Same structure as Python. Node type: `call_expression`

### Resolution Logic

After all elements are indexed, resolve calls:

1. For each call in a function, look at file's imports
2. If `receiver` matches an import alias/name → resolve to that module
3. If `receiver` is `self`/`this` → resolve to same class
4. Look up resolved path in index → set `resolved_id`

Unresolved calls (builtins, dynamic) get `resolved_id: null`.

## Indexing Pipeline Changes

### Phase 3 (Parsing) - Enhanced

```
Current:  Extract elements (classes, functions, methods)
Addition: Also extract imports (file-level) and calls (per element)
```

Each `CodeElement` gains:
- `imports: list[Import]` (on file elements)
- `calls: list[Call]` (on function/method elements)

### Phase 5 (Summarization) - No changes

### Phase 6 (Embedding) - Dual embedding

```
Current:  Embed summary text → store in "summary_embedding"
Addition: Embed raw_code → store in "code_embedding"
```

Both use `snowflake-arctic-embed2` (8192 token context). Analysis shows median code is ~149 tokens, max ~3367, so no truncation needed.

### New: Call Resolution Phase (after embedding)

```
For each element with calls:
    For each call:
        1. Get file's imports
        2. Resolve receiver to module path
        3. Query index for matching element
        4. Set resolved_id if found
```

### CLI Progress Display

Update thread status display to show both embeddings:

```
Thread 1: [element_name]
  Summary embedding: done
  Code embedding: processing
```

## Implementation Phases

### Phase A: Pattern Search (no schema change)

1. Implement ES-native `pattern_search` tool with regexp/wildcard/proximity modes
2. Update `find_usages`, `find_implementations` to use it internally
3. Deprecate `grep_code`

**Delivers:** Faster, server-side pattern matching immediately.

### Phase B: Code Embeddings

1. Add `code_embedding` field to schema
2. Rename `embedding` → `summary_embedding`
3. Update embedding pipeline to generate both
4. Update CLI progress display to show both embedding statuses
5. Add `search_mode` parameter to `search_code`
6. Implement similarity tools: `find_similar_structure`, `find_similar_intent`, `find_duplicates`
7. Re-index to populate new field

**Delivers:** Semantic search on actual code, similarity detection.

### Phase C: Call Graph & Dependencies

1. Extend tree-sitter extraction for imports and calls
2. Add `imports` and `calls` fields to schema
3. Implement resolution logic
4. Update `get_call_graph` to use indexed data
5. Implement call analysis tools: `find_callers`, `find_call_chain`, `find_dead_code`, `find_entry_points`
6. Implement dependency tools: `find_dependencies`, `find_dependents`, `dependency_graph`
7. Implement `explain_element` meta tool
8. Re-index to populate new fields

**Delivers:** Accurate call graph, dependency analysis, dead code detection.

### Phase D: Web UI Updates

#### Element Detail Page Enhancements

| Section | Content |
|---------|---------|
| **Callers** | List of functions that call this element, with file and line links |
| **Callees** | List of functions this element calls, with resolved links |
| **Imports** | For file elements: list of imports (internal/external badge) |
| **Similar Code** | Tabs: "Similar Structure" / "Similar Intent" with similarity % |
| **Embedding Status** | Visual indicator: summary ✓, code ✓ |
| **Call Chain** | Expandable tree: callers → this → callees (2 levels each) |

#### New Pages

**1. Dependency Graph**
- Interactive graph visualization (D3.js or similar)
- Nodes = files/modules, edges = imports
- Filter: internal only, show external, by directory
- Click node to see file details
- Highlight cycles (circular dependencies)

**2. Dead Code Report**
- List of potentially unused functions/methods
- Grouped by file
- Filter: exclude test code, min lines threshold
- "Mark as intentional" action (e.g., public API)
- Export as CSV/JSON

**3. Duplicate Code Report**
- Groups of similar code (>95% similarity)
- Side-by-side diff view
- Similarity score badge
- Filter by min lines, min similarity
- "Suggest refactor" notes

**4. Entry Points Dashboard**
- Categorized: HTTP handlers, CLI commands, test fixtures, scheduled jobs
- Extracted from decorators and naming patterns
- Link to each entry point's call chain

**5. Call Explorer**
- Start with any function
- Expand callers (upstream) or callees (downstream)
- Tree or graph view toggle
- Depth limit control
- Highlight paths between two functions

**6. Code Similarity Map**
- 2D visualization of code embeddings (UMAP/t-SNE)
- Cluster coloring by feature or directory
- Zoom to see individual elements
- Toggle: summary embeddings vs code embeddings

#### Search Page Enhancements

| Feature | Description |
|---------|-------------|
| **Search mode toggle** | Buttons: Summary / Code / Hybrid (default) |
| **Pattern search tab** | Separate tab with mode selector (regexp/wildcard/proximity) |
| **Slop slider** | For proximity mode, adjust word distance |
| **Preview regex** | Show ES regexp syntax help |

#### Navigation Enhancements

- **Breadcrumb with call context:** When navigating from caller → callee, show the path
- **"Used by" badge:** On element cards, show caller count
- **"Uses" badge:** On element cards, show callee count
- **Quick actions:** From any element: "Find callers", "Find similar", "Show in graph"

**Delivers:** Full visual exploration of code relationships, dependencies, and similarity.

## Migration Notes

- Existing `embedding` field renamed to `summary_embedding` (requires schema update)
- Existing documents get `null` for new fields until re-indexed
- `grep_code` deprecated but functional during transition
- Full re-index required to populate `code_embedding`, `imports`, `calls`

## Open Questions

None at this time.
