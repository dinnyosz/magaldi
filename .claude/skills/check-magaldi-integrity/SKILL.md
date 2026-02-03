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

---

## 8. Report Format

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
