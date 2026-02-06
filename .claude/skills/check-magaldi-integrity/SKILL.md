---
name: check-magaldi-integrity
description: >
  Implementation checklist and integrity verification for Magaldi.
  Use BEFORE adding new element types, fields, or modifying parsing/summarization.
  Use AFTER changes to verify nothing is missing.
---

# Magaldi Implementation & Integrity Check

**MANDATORY**: Read this skill before AND after implementing features that add new element types, extract new metadata, or modify parsing/summarization.

## Core Principle

Every piece of data extracted during parsing MUST be surfaced to users through ALL relevant interfaces: Web UI, MCP tools, and AI summarization. No gaps allowed.

---

## 1. Source of Truth

**Never hardcode element types or field lists.** Always reference these source files:

| What | Where |
|------|-------|
| Element types | `CodeElement.element_type` docstring @ `src/magaldi_core/code_parser.py` |
| Element fields/metadata | `CodeElement` dataclass @ `src/magaldi_core/code_parser.py` |
| Extractor types | `ExtractedElement` @ `src/magaldi_core/extractors/types.py` |
| Summarization prompts | `SYSTEM_PROMPTS`, `USER_PROMPTS`, `PROMPTS` @ `src/shared/ai/summarization.py` |
| Prompt sentence ranges | `LINE_THRESHOLDS`, `SENTENCE_RANGES` @ `src/shared/ai/summarization.py` |
| Feature/subfeature prompts | `FEATURE_*_PROMPT`, `SUBFEATURE_*_PROMPT` @ `src/shared/ai/clustering/feature_processor.py` |
| Glossary prompts | `GLOSSARY_*_PROMPT` @ `src/shared/ai/glossary/ai_extractor.py` |
| ES field mappings | `src/shared/db/elasticsearch.py` |
| MCP tool implementations | `src/magaldi_mcp/tools_impl.py` |
| MCP tool schemas | `src/magaldi_mcp/tools/schemas/` |
| MCP formatters | `src/magaldi_mcp/formatters/` |
| Web API routes | `src/magaldi_web/routes/` |
| Web frontend pages | `src/magaldi_web/frontend/src/pages/` |
| TypeScript types | `src/magaldi_web/frontend/src/api.ts` |
| Element icons (Explorer) | `typeConfig` in `src/magaldi_web/frontend/src/pages/Explorer.tsx` |
| Element icons (Element detail) | `typeConfig` in `src/magaldi_web/frontend/src/pages/Element.tsx` |
| Dashboard stats model | `DashboardStats`, `RepoSummary` @ `src/magaldi_web/models.py` |
| Dashboard aggregations | `get_dashboard()` @ `src/magaldi_web/routes/dashboard.py` |
| ES serialization | `index_element()` @ `src/shared/db/repositories/elements.py` |
| Conditional element types | `CONDITIONAL_ELEMENT_TYPES` @ `src/magaldi_web/frontend/src/pages/Explorer.tsx`, `Search.tsx` |

---

## 2. New Element Type Checklist

When adding a new element type:

### Parsing Layer
- [ ] Add to `CodeElement.element_type` docstring in `src/magaldi_core/code_parser.py`
- [ ] Implement extraction in relevant extractors in `src/magaldi_core/extractors/`

### Summarization Prompts (`src/shared/ai/summarization.py`)
- [ ] Add to `LINE_THRESHOLDS` dict
- [ ] Add to `SENTENCE_RANGES` dict
- [ ] Add to `SYSTEM_PROMPTS` dict with anti-verbose instruction
- [ ] Add to `USER_PROMPTS` dict
- [ ] Add to legacy `PROMPTS` dict with anti-verbose instruction
- [ ] Update context building in `build_prompt()` and `build_messages()` if needed

### Web UI
- [ ] Dashboard stats model: Add `{type}_count` to `DashboardStats` and `RepoSummary` in `src/magaldi_web/models.py`
- [ ] Dashboard aggregations: Add aggregation in `src/magaldi_web/routes/dashboard.py`
- [ ] Dashboard TS interface: Add `{type}_count` to `DashboardStats` in `src/magaldi_web/frontend/src/api.ts`
- [ ] **Conditional display**: If type is language-specific (not universal), add to `CONDITIONAL_ELEMENT_TYPES` in `Explorer.tsx` and `Search.tsx`
- [ ] Search filters: `src/magaldi_web/routes/search.py`
- [ ] Explorer icons: Add to `typeConfig` in `src/magaldi_web/frontend/src/pages/Explorer.tsx`
- [ ] Element detail icons: Add to `typeConfig` in `src/magaldi_web/frontend/src/pages/Element.tsx`
- [ ] Element detail rendering: Handle new type in `src/magaldi_web/frontend/src/pages/Element.tsx`
- [ ] TypeScript interfaces: `src/magaldi_web/frontend/src/api.ts`

**Conditional vs Always-Visible Types**:
- **Always visible** (universal across languages): `file`, `class`, `function`, `method`, `variable`, `constant`
- **Conditional** (language-specific, hide when count=0): `interface`, `trait`, `enum`, `type_alias`, `import`
- New language-specific types should be added to `CONDITIONAL_ELEMENT_TYPES` arrays

### MCP Tools
- [ ] Verify `search_code` returns new element type
- [ ] Verify `get_element` returns all fields
- [ ] Update `element_types` description in `src/magaldi_mcp/tools/schemas/search.py` (must list all valid types)
- [ ] Update schemas if needed: `src/magaldi_mcp/tools/schemas/`
- [ ] Update formatters if needed: `src/magaldi_mcp/formatters/`

---

## 3. New Metadata/Field Checklist

When extracting new metadata:

### Parsing Layer
- [ ] Add field to `CodeElement` dataclass in `src/magaldi_core/code_parser.py`
- [ ] Add field to `ExtractedElement` in `src/magaldi_core/extractors/types.py`
- [ ] Implement extraction in relevant extractors

### Storage
- [ ] Add field mapping in `src/shared/db/elasticsearch.py`
- [ ] Update serialization in `src/magaldi_core/storage.py`

### Summarization Prompts
- [ ] Add to relevant prompt context sections (use helper function pattern)
- [ ] Only include if non-empty

### Web UI
- [ ] Display in element detail page
- [ ] Add to search filters if filterable
- [ ] Update TypeScript interfaces

### MCP Tools
- [ ] Include in `get_element` response
- [ ] Include in search results if relevant
- [ ] Create dedicated tool if complex
- [ ] Update formatters

---

## 4. Anti-Verbose Prompt Rules

**ALL summarization prompts MUST include anti-verbose instructions.**

### Pattern
Add at the end of every system/legacy prompt:
```
Start [action] - never start with "This [type]...", "The X [type]...", or similar.
```

### Guidelines by Category

| Category | Instruction Pattern |
|----------|---------------------|
| Containers (file, class) | `Start directly with what it does/models` |
| Contracts (interface, trait) | `Start directly with what contract/capability it defines` |
| Callables (function, method) | `Start with an action verb` |
| Data (enum, type_alias, constant, variable) | `Start directly with what it represents/holds` |
| Dependencies (import) | `Start with what it imports` |

### Examples

| Bad | Good |
|-----|------|
| "This module is responsible for handling authentication" | "Handles user authentication via JWT tokens" |
| "The UserService class is used to manage users" | "Manages user CRUD operations and session state" |
| "This function is used to validate input" | "Validates and sanitizes user input against schema" |

---

## 5. Token-Efficient Context

### Helper Function Pattern
See `_build_*_section()` functions in `src/shared/ai/summarization.py`

```python
def _build_X_section(element: CodeElement) -> str:
    if not element.X_field:
        return ""
    items = element.X_field[:5]  # Limit
    return f"\nX: {', '.join(items)}"
```

### Rules
- Conditional inclusion only
- Compact format: `\nField: value1, value2`
- Limit lists: `[:5]` or `[:3]`
- Never include empty sections

---

## 6. Verification Workflow

### Step 1: Read Source of Truth
```
Read src/magaldi_core/code_parser.py  # CodeElement dataclass
```

### Step 2: Check Summarization
```bash
source .venv/bin/activate
python -c "from shared.ai.summarization import PROMPTS; print(list(PROMPTS.keys()))"
```

### Step 3: Check Web UI
- Read `src/magaldi_web/frontend/src/pages/Element.tsx`
- Read `src/magaldi_web/frontend/src/api.ts`
- Verify all fields rendered and typed

### Step 4: Check Icon Coverage
Verify all element types have icons in both Explorer and Element pages:
```bash
# Extract types from summarization (source of truth for element types)
source .venv/bin/activate
python -c "from shared.ai.summarization import LINE_THRESHOLDS; print(sorted(LINE_THRESHOLDS.keys()))"

# Then grep for typeConfig in both files and verify all types are covered:
grep -A 20 "const typeConfig" src/magaldi_web/frontend/src/pages/Explorer.tsx
grep -A 20 "const typeConfig" src/magaldi_web/frontend/src/pages/Element.tsx
```

Each element type needs an entry with:
- `icon`: Bootstrap icon class (e.g., `bi-box`, `bi-braces`)
- `color`: Badge color (e.g., `primary`, `success`, `warning`)
- Optional flags: `canHaveChildren`, `canHaveCallGraph` (Explorer only)

### Step 5: Check Dashboard Stats
Verify all element types have corresponding `*_count` fields:
```bash
# Get element types from source of truth
source .venv/bin/activate
python -c "from shared.ai.summarization import LINE_THRESHOLDS; print(sorted(LINE_THRESHOLDS.keys()))"

# Check DashboardStats and RepoSummary models
grep -A 20 "class DashboardStats" src/magaldi_web/models.py
grep -A 20 "class RepoSummary" src/magaldi_web/models.py

# Check dashboard route aggregations
grep -A 30 '"aggs":' src/magaldi_web/routes/dashboard.py | head -40
```

Each element type should have:
- `{type}_count: int = 0` field in `DashboardStats`
- `{type}_count: int = 0` field in `RepoSummary`
- Aggregation in dashboard.py: `"{type}_count": {"filter": {"term": {"element_type": "{type}"}}}`

**Note**: `feature` and `subfeature` are AI-generated types (not parsed), so they appear in stats but not in `LINE_THRESHOLDS`.

### Step 6: Check MCP Schemas
Verify `element_types` descriptions include all valid types:
```bash
grep -A 5 "element_types" src/magaldi_mcp/tools/schemas/search.py
```

The description should list ALL element types from the source of truth, not just common ones.

### Step 7: Check MCP Implementation
- Read `src/magaldi_mcp/tools_impl.py`
- Verify `get_element` returns all fields
- Verify search tools filter correctly

### Step 8: Run Tests
```bash
pytest tests/test_summarization.py -v
```

---

## 7. Common Issues

1. **New field in CodeElement but not in**:
   - TypeScript `ElementDetail` interface
   - Element detail page rendering
   - MCP `get_element` response
   - Summarization prompt context

2. **New element type but not in**:
   - Summarization prompts (all 5 dicts!)
   - Dashboard statistics (`DashboardStats`, `RepoSummary` models + route aggregations)
   - Search filter options
   - Explorer `typeConfig` (missing icon/color)
   - Element detail `typeConfig` (missing icon/color)
   - MCP schema `element_types` description

3. **Element links not using hash_id**:
   - Always use `hash_id` (stable) not `element_id` (may change)

4. **Prompt missing anti-verbose instruction**:
   - Check all prompts end with "Start..." instruction

5. **Icon config mismatch between Explorer and Element pages**:
   - Both `typeConfig` objects should have the same element types
   - Icons and colors should be consistent across pages

6. **Dashboard stats showing irrelevant types**:
   - Only "common" types should always display: `file`, `class`, `function`, `method`, `variable`, `constant`
   - "Conditional" types should only appear when count > 0: `interface`, `trait`, `enum`, `type_alias`, `import`
   - See `CONDITIONAL_ELEMENT_TYPES` in `Explorer.tsx` and `Search.tsx`

7. **MCP schema element_types description incomplete**:
   - `search_code` schema must list ALL valid element types, not just common ones
   - Check `src/magaldi_mcp/tools/schemas/search.py`

8. **Variable/constant extraction inconsistency across languages**:
   - All languages MUST extract variables/constants, then apply usefulness filter
   - Check each extractor extracts `variable`/`constant` element types
   - See Variable Usefulness Filter section below

---

## 8. Variable/Constant Usefulness Filter

All language extractors MUST follow the same pattern for variable/constant extraction:

1. **Extract ALL** module-level and class-level variable/constant assignments
2. **Apply usefulness filter** to skip transient/temporary values
3. **Log skipped variables** at DEBUG level for visibility

### Source of Truth Files

| What | Where |
|------|-------|
| Python filter implementation | `_is_useful_assignment()` @ `src/magaldi_core/extractors/python.py` |
| Useful factory functions | `USEFUL_FACTORIES`, `USEFUL_ATTRIBUTE_FACTORIES` @ `src/magaldi_core/extractors/python.py` |
| Skip names list | `SKIP_NAMES` @ `src/magaldi_core/extractors/python.py` |
| Extraction stats | `ExtractionStats`, `SkippedVariable` @ `src/magaldi_core/extractors/python.py` |
| Plan document | `plans/variable_usefulness_filter.md` |

### Usefulness Criteria (apply to ALL languages)

**KEEP (Useful):**
- Constants by naming convention (UPPER_CASE, SCREAMING_SNAKE)
- Literal values (strings, numbers, arrays, objects, booleans)
- Type aliases and type definitions
- Enum definitions
- TypeVar/generic type definitions
- Named tuples / data structures
- Compiled patterns (regex)
- Loggers
- Threading primitives (locks, semaphores)
- Lambda/arrow functions assigned to variables

**SKIP (Not Useful):**
- Instance creations: `client = SomeClient()`, `new HttpClient()`
- Factory method results: `service = Factory.create()`
- Function call results: `data = process_items()`
- Method call results: `response = http.get(url)`
- Short/temp variable names: `i`, `j`, `tmp`, `temp`, `val`, `res`

### Language Implementation Status

| Language | Status | Notes |
|----------|--------|-------|
| Python | ✅ Done | Full filter with logging |
| JavaScript | ❌ TODO | Currently only extracts arrow functions, needs to extract all `const`/`let` |
| PHP | ❌ TODO | Currently only extracts `const` declarations, needs to extract `$var = ...` |
| Rust | ❌ TODO | Currently only extracts `const`/`static`, needs to extract `let` bindings |

### Verification Steps

```bash
# Check Python filter is working
source .venv/bin/activate
python -c "
from magaldi_core.extractors.python import USEFUL_FACTORIES, SKIP_NAMES
print('Useful factories:', len(USEFUL_FACTORIES))
print('Skip names:', len(SKIP_NAMES))
"

# Run extraction with DEBUG logging to see skipped variables
PYTHONPATH=src python -c "
import logging
logging.basicConfig(level=logging.DEBUG)
from magaldi_core.extractors import extract_python_elements
from magaldi_core.tree_sitter_manager import get_manager
code = 'client = HttpClient()\nMAX = 100'
tree = get_manager().parse(code.encode(), 'python')
extract_python_elements(tree, code.split('\n'), 'test.py')
"
```

### Checklist for Adding Filter to New Language

- [ ] Modify extractor to extract ALL variable/constant assignments (not just specific patterns)
- [ ] Import/adapt `USEFUL_FACTORIES`, `SKIP_NAMES` from Python extractor (or create language-specific versions)
- [ ] Implement `_is_useful_assignment()` for language-specific AST node types
- [ ] Add `ExtractionStats` tracking and DEBUG logging
- [ ] Add tests verifying useful variables kept, transient ones skipped
- [ ] Update `plans/variable_usefulness_filter.md` with implementation status

### Report Format

```markdown
### Variable Extraction Gaps

**Language missing filter:**
- [ ] JavaScript: Only extracts arrow functions, not `const x = "value"`
- [ ] PHP: Only extracts `const`, not `$x = "value"`
- [ ] Rust: Only extracts `const`/`static`, not `let x = value`

**Filter too aggressive:**
- [ ] `{language}`: Skipping `{pattern}` which should be kept because `{reason}`

**Filter too permissive:**
- [ ] `{language}`: Keeping `{pattern}` which should be skipped because `{reason}`
```

---

## 9. MCP Analytics Integrity

When modifying MCP analytics (what data is collected, stored, or displayed), verify full data flow:

### Source of Truth Files

| What | Where |
|------|-------|
| Redis storage methods | `RedisMCPAnalyticsRepository` class @ `src/shared/db/redis.py` |
| API response models | `MCP*Response`, `Tool*Info`, `Daily*` @ `src/magaldi_web/models.py` |
| Admin API endpoints | `/admin/mcp-analytics*` @ `src/magaldi_web/routes/admin.py` |
| TypeScript interfaces | `MCP*Response`, `Tool*Info`, `Daily*` @ `src/magaldi_web/frontend/src/api.ts` |
| Admin UI rendering | `Admin.tsx` @ `src/magaldi_web/frontend/src/pages/Admin.tsx` |
| MCP server recording | `server.py` @ `src/magaldi_mcp/server.py` |

### Verification Steps

1. **Check what Redis stores** - Read `RedisMCPAnalyticsRepository` class methods:
   - `record_tool_call()` - what is recorded on tool invocation
   - `record_tool_end()` - what is recorded on completion
   - `get_tool_counts()`, `get_daily_counts()`, `get_tool_transitions()`, `get_tool_durations()` - what can be retrieved

2. **Check what API exposes** - Read admin routes and models:
   - Compare Python response models (e.g., `MCPAnalyticsResponse`) with Redis getter returns
   - Verify all Redis data has corresponding model fields

3. **Check TypeScript types** - Read `api.ts`:
   - Each Python model field must have TypeScript equivalent
   - Compare `MCPAnalyticsResponse` interface with Python model

4. **Check Admin UI rendering** - Read `Admin.tsx`:
   - Every field in `MCPAnalyticsResponse` should be displayed somewhere
   - Charts and tables should cover all data dimensions

### Checklist for New Analytics Data

- [ ] Add storage method in `RedisMCPAnalyticsRepository` (redis.py)
- [ ] Add recording calls in MCP server (server.py)
- [ ] Add getter method in Redis repo if needed
- [ ] Add field to Python response model (models.py)
- [ ] Populate field in admin route handler (admin.py)
- [ ] Add TypeScript interface field (api.ts)
- [ ] Display in Admin.tsx (chart/table/metric)

### Display Guidelines

- Keep Admin page compact - avoid sprawling layouts
- Use **charts** for trends and distributions (pie chart for proportions, line chart for time series)
- Use **tables** for detailed lists with multiple columns
- Combine related data (e.g., add avg_ms column to tool usage table rather than separate table)
- Show numerical summaries (totals, counts) prominently

### Report Format for Analytics

```markdown
### MCP Analytics Gaps

**Redis stores but not exposed:**
- [ ] `X_data` stored by `record_X()` but not returned by any getter

**API exposes but not in model:**
- [ ] `Y_field` returned by admin route but missing from `MCPAnalyticsResponse`

**Model has but TypeScript missing:**
- [ ] `Z_field` in Python `MCPAnalyticsResponse` but not in TS interface

**API returns but not displayed:**
- [ ] `W_field` in TypeScript type but not rendered in Admin.tsx
```

---

## 9. MCP Output Token Efficiency

MCP tool output should be token-efficient since it's consumed by LLMs. Every wasted token costs money and context.

### Source of Truth Files

| What | Where |
|------|-------|
| MCP formatters | `src/magaldi_mcp/formatters/*.py` |
| ES repository getters | `src/shared/db/repositories/*.py` |
| Tool implementations | `src/magaldi_mcp/tools_impl.py` |

### Common Token Waste Patterns

1. **Redundant type prefixes**: Don't repeat `[feature]` on every line when listing features - context already makes it clear
2. **Using `element_id` instead of `hash_id`**: Always return `hash_id` (stable, required for follow-up calls)
3. **Verbose field names**: Prefer compact field names in output
4. **Unnecessary whitespace**: Avoid excessive blank lines between items
5. **Full summaries when `brief=True`**: Respect brief mode and omit summaries

### Checklist for MCP Output

- [ ] ES repository getters include `hash_id` in `_source` list
- [ ] Repository returns include `hash_id` field (not just `element_id` or `feature_id`)
- [ ] Formatters use `hash_id` in output (for follow-up tool calls)
- [ ] Type prefixes only shown when necessary (mixed types or disambiguation)
- [ ] Brief mode omits summaries and verbose fields
- [ ] Output uses compact format (no unnecessary repetition)

### Verification Steps

**IMPORTANT**: Review ALL files listed below, not just ones you think might have issues.

#### Step 1: Review ALL Repository Getters

Read each file and check every `get_*` method:

| File | Check |
|------|-------|
| `src/shared/db/repositories/elements.py` | `_source` includes `hash_id`, return dict has `hash_id` |
| `src/shared/db/repositories/features.py` | `get_features()`, `get_subfeatures()` return `hash_id` |
| `src/shared/db/repositories/search.py` | Search results include `hash_id` |
| `src/shared/db/repositories/glossary.py` | Glossary results include `hash_id` where applicable |

For each getter, verify:
- [ ] `hash_id` is in `_source` list (ES query)
- [ ] `hash_id` is in the returned dict/object

#### Step 2: Review ALL Formatters

Read each formatter file and check every `format()` method:

| File | Formatters to Check |
|------|---------------------|
| `src/magaldi_mcp/formatters/search.py` | `CodeSearchListFormatter`, `FeatureSearchListFormatter`, `GroupedSearchFormatter` |
| `src/magaldi_mcp/formatters/elements.py` | `ElementDetailsFormatter`, `RepoListFormatter`, `FileListFormatter`, `FeatureMembersFormatter`, etc. |
| `src/magaldi_mcp/formatters/analysis.py` | `CallGraphFormatter`, `DeadCodeFormatter`, `EntryPointsFormatter`, `ExplainElementFormatter`, etc. |
| `src/magaldi_mcp/formatters/dependencies.py` | All dependency-related formatters |

For each formatter, verify:
- [ ] Uses `hash_id` in ID suffix (not `element_id`, `feature_id`, `subfeature_id`)
- [ ] Type prefix `[type]` only shown when necessary (mixed types or disambiguation needed)
- [ ] Summaries truncated appropriately (typically 100-200 chars)
- [ ] No excessive blank lines between items
- [ ] Respects `brief` parameter if applicable

#### Step 3: Test Actual Output

Run MCP tools and verify output is token-efficient:

```bash
# Test list_features - should NOT have [feature] prefix on every line
mcp__magaldi__list_features(scope="X", repository="Y", brief=True)

# Test search_code - should use hash_id not element_id
mcp__magaldi__search_code(query="test", limit=3)

# Test get_element - should use hash_id
mcp__magaldi__get_element(hash_id="...")
```

Example of good vs bad output:
```
# Good: compact, uses hash_id, no redundant prefix
feature_name (10 members) | id:abc123def456...

# Bad: wasteful prefix, uses element_id format
[feature] feature_name (10 members) | id:scope:repo:main:feature:name:1
```

### Report Format for Output Issues

After verifying repository getters and formatters, report any gaps found:

```markdown
### MCP Output Token Waste

**Missing hash_id:**
- [ ] `{getter_name}` doesn't include `hash_id` in `_source` or return dict

**Formatter issues:**
- [ ] `{formatter_name}` uses `{wrong_id_field}` instead of `hash_id`
- [ ] `{formatter_name}` repeats type prefix unnecessarily for homogeneous lists

**Verbose output:**
- [ ] `{tool_name}` doesn't respect `brief` parameter
- [ ] `{formatter_name}` has unnecessary blank lines between items
```

---

## 10. MCP Tool Handler Parameter Safety

MCP tool handlers in `server.py` must use `args.get()` for optional parameters to enable auto-detection from `magaldi.yaml`.

### Source of Truth Files

| What | Where |
|------|-------|
| MCP server handlers | `_handle_tool()` @ `src/magaldi_mcp/server.py` |
| Tool implementations | Functions in `src/magaldi_mcp/tools_impl.py` |
| Tool schemas | `src/magaldi_mcp/tools/schemas/*.py` |

### Rules

**Always use `args.get()` for:**
- `scope` - auto-detected from `magaldi.yaml`
- `repository` - auto-detected from `magaldi.yaml`
- `username` - has default value

**Use `args["key"]` only for:**
- Truly required parameters (e.g., `query`, `hash_id`, `pattern`)
- Parameters marked `required: true` in schema with no auto-detection

### Verification

```bash
# Find any args["scope"], args["repository"], or args["username"] usages
grep -n 'args\["scope"\]\|args\["repository"\]\|args\["username"\]' src/magaldi_mcp/server.py
```

**Expected output:** No matches (all should use `args.get()`)

### Bad vs Good

```python
# BAD - crashes when scope/repository not provided
scope=args["scope"],
repository=args["repository"],

# GOOD - allows auto-detection from magaldi.yaml
scope=args.get("scope"),
repository=args.get("repository"),
username=args.get("username", self.default_username),
```

### Report Format

```markdown
### MCP Handler Parameter Issues

**Using args[] instead of args.get():**
- [ ] Line {N}: `args["scope"]` should be `args.get("scope")`
- [ ] Line {N}: `args["repository"]` should be `args.get("repository")`
```

---

## 11. Multi-User Query Merging (MCP Queries)

All MCP/ES queries that filter by `username` MUST query BOTH the current user AND the default `"main"` user, with user results ranked higher.

### Why

Users only have **local changes** indexed (partial parse). The `"main"` branch has the full repository. Queries must merge both to show a complete view, with the user's version taking priority when the same element exists in both.

### Rules

1. **Query both**: Use `should` clause with `[{"term": {"username": current_user}}, {"term": {"username": "main"}}]` instead of a single `term` filter
2. **Boost user results**: Apply a `boost` to the current user's term to rank their results higher
3. **Deduplicate**: When the same element (by `relative_path` + `name` + `element_type`) exists in both user and main, keep only the user's version
4. **Fallback**: If `username == "main"`, skip the merge logic (no duplication needed)

### Pattern

```python
# BAD - only sees user's partial index
filter_clauses.append({"term": {"username": username}})

# GOOD - sees user overlay + main, user ranked higher
if username and username != "main":
    filter_clauses.append({
        "bool": {
            "should": [
                {"term": {"username": {"value": username, "boost": 2.0}}},
                {"term": {"username": "main"}},
            ],
            "minimum_should_match": 1,
        }
    })
else:
    filter_clauses.append({"term": {"username": "main"}})
```

### Deduplication Pattern

After fetching results, deduplicate by keeping user version over main:

```python
def _deduplicate_user_main(results: list[dict], username: str) -> list[dict]:
    """Keep user's version when same element exists in both user and main."""
    if not username or username == "main":
        return results
    seen: dict[str, dict] = {}  # key -> best result
    for r in results:
        key = f"{r.get('relative_path')}:{r.get('name')}:{r.get('element_type')}"
        existing = seen.get(key)
        if existing is None:
            seen[key] = r
        elif r.get("username") == username:
            seen[key] = r  # user version wins
    return list(seen.values())
```

### Verification

```bash
# Check for single-user username filters in MCP-facing queries
grep -n '"term": {"username":' src/shared/db/repositories/search.py
grep -n '"term": {"username":' src/magaldi_core/call_resolution.py
```

### Report Format

```markdown
### Multi-User Query Issues

**Single-user filter (should merge user+main):**
- [ ] `{method_name}` in `{file}` line {N}: Uses single `term` username filter

**Missing deduplication:**
- [ ] `{method_name}` queries both users but doesn't deduplicate results
```

---

## 12. Report Format

After verification, produce:

```markdown
## Integrity Check Results

### Missing in Summarization
- [ ] `new_type` missing from PROMPTS dicts

### Missing in Web UI
- [ ] `new_field` not shown in Element detail
- [ ] `new_type` missing icon in Explorer

### Missing Icon Coverage
- [ ] `new_type` missing from Explorer.tsx `typeConfig`
- [ ] `new_type` missing from Element.tsx `typeConfig`
- [ ] Icon/color mismatch between Explorer and Element for `some_type`

### Missing in Dashboard Stats
- [ ] `{type}_count` missing from `DashboardStats` model
- [ ] `{type}_count` missing from `RepoSummary` model
- [ ] Aggregation for `{type}` missing in dashboard.py
- [ ] `{type}_count` missing from TypeScript `DashboardStats` interface

### Missing Conditional Display Config
- [ ] Language-specific `new_type` not in `CONDITIONAL_ELEMENT_TYPES` (Explorer.tsx)
- [ ] Language-specific `new_type` not in `CONDITIONAL_ELEMENT_TYPES` (Search.tsx)

### Missing in MCP
- [ ] `get_element` doesn't return `new_field`
- [ ] `search_code` schema `element_types` description incomplete

### Anti-verbose Issues
- [ ] `new_type` prompt missing "Start..." instruction

### Recommendations
1. Add `new_type` to all prompt dicts in summarization.py
2. Add `new_field` to Element.tsx
3. Add `new_type` to typeConfig in both Explorer.tsx and Element.tsx
4. Add `{type}_count` to DashboardStats, RepoSummary, and dashboard route
5. If language-specific, add to CONDITIONAL_ELEMENT_TYPES arrays
6. Update MCP schema element_types description
```
