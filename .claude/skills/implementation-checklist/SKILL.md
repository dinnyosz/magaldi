# Implementation Checklist

**MANDATORY**: Read this skill before implementing any feature that adds new element types, extracts new metadata, or modifies parsing/summarization.

## Core Principle

Every piece of data extracted during parsing MUST be surfaced to users through ALL relevant interfaces: Web UI, MCP tools, and AI summarization. No gaps allowed.

---

## 1. New Element Type Checklist

**Source of truth for element types:** `CodeElement.element_type` field docstring in `src/magaldi_core/code_parser.py:207`

When adding a new element type:

### Parsing Layer
- [ ] Add to `CodeElement.element_type` docstring in `src/magaldi_core/code_parser.py`
- [ ] Implement extraction in relevant language extractors in `src/magaldi_core/extractors/`

### Summarization Prompts
**Source of truth:** `src/shared/ai/summarization.py`

- [ ] Add to `LINE_THRESHOLDS` dict - defines size tiers for the element
- [ ] Add to `SENTENCE_RANGES` dict - defines sentence count per size tier
- [ ] Add to `SYSTEM_PROMPTS` dict - the main prompt (see anti-verbose rules below)
- [ ] Add to `USER_PROMPTS` dict - the variable content template
- [ ] Add to legacy `PROMPTS` dict - single-prompt format for backwards compat
- [ ] Update context building in `build_prompt()` and `build_messages()` if element needs special sections

### Web UI
- [ ] Dashboard stats: `src/magaldi_web/routes/dashboard.py`
- [ ] Search filters: `src/magaldi_web/routes/search.py`
- [ ] Explorer icons: `src/magaldi_web/frontend/src/pages/Explorer.tsx`
- [ ] Element detail page: `src/magaldi_web/frontend/src/pages/Element.tsx`
- [ ] TypeScript interfaces: `src/magaldi_web/frontend/src/api.ts`

### MCP Tools
- [ ] Verify `search_code` returns new element type
- [ ] Verify `get_element` returns all fields
- [ ] Update tool schemas if new filters needed: `src/magaldi_mcp/tools/schemas/`
- [ ] Update formatters if needed: `src/magaldi_mcp/formatters/`

---

## 2. New Metadata Checklist

**Source of truth for element fields:** `CodeElement` dataclass in `src/magaldi_core/code_parser.py`

When extracting new metadata:

### Parsing Layer
- [ ] Add field to `CodeElement` dataclass in `src/magaldi_core/code_parser.py`
- [ ] Add field to `ExtractedElement` in `src/magaldi_core/extractors/types.py`
- [ ] Implement extraction in relevant extractors in `src/magaldi_core/extractors/`

### Elasticsearch
- [ ] Add field mapping in `src/shared/db/elasticsearch.py`
- [ ] Update element serialization in `src/magaldi_core/storage.py`

### Summarization Prompts
- [ ] Add to relevant prompt context sections in `src/shared/ai/summarization.py`
- [ ] Use helper function pattern (see section 4)
- [ ] Only include if non-empty

### Web UI
- [ ] Display in element detail page: `src/magaldi_web/frontend/src/pages/Element.tsx`
- [ ] Add to search filters if filterable: `src/magaldi_web/routes/search.py`
- [ ] Update TypeScript interfaces: `src/magaldi_web/frontend/src/api.ts`

### MCP Tools
- [ ] Include in `get_element` response: `src/magaldi_mcp/tools_impl.py`
- [ ] Include in search results if relevant
- [ ] Create dedicated tool if complex (check existing patterns in `src/magaldi_mcp/tools/schemas/`)
- [ ] Update formatters: `src/magaldi_mcp/formatters/`

---

## 3. Anti-Verbose Prompt Rules

**ALL summarization prompts MUST include anti-verbose instructions.**

### Reference Implementation
See existing prompts in `SYSTEM_PROMPTS` dict in `src/shared/ai/summarization.py` for the exact format used per element type.

### Pattern
Add at the end of every system/legacy prompt:
```
Start [action] - never start with "This [type]...", "The X [type]...", or similar.
```

### Guidelines by Element Category

| Category | Instruction Pattern |
|----------|---------------------|
| Containers (file, class) | `Start directly with what it does/models - never start with "This [type]..."` |
| Contracts (interface, trait) | `Start directly with what contract/capability it defines - never start with "This [type]..."` |
| Callables (function, method) | `Start with an action verb - never start with "This [type]..." or "[type] is used to..."` |
| Data (enum, type_alias, constant, variable) | `Start directly with what it represents/holds - never start with "This [type]..."` |
| Dependencies (import) | `Start with what it imports - never start with "This import..."` |

### Bad vs Good Examples

| Bad | Good |
|-----|------|
| "This module is responsible for handling authentication" | "Handles user authentication via JWT tokens" |
| "The UserService class is used to manage users" | "Manages user CRUD operations and session state" |
| "This function is used to validate input" | "Validates and sanitizes user input against schema" |

---

## 4. Token-Efficient Context in Prompts

### Helper Function Pattern
**Reference:** See `_build_*_section()` functions in `src/shared/ai/summarization.py`

```python
def _build_X_section(element: CodeElement) -> str:
    """Build X section for prompt. Returns empty string if no data."""
    if not element.X_field:
        return ""
    # Limit to reasonable count
    items = element.X_field[:5]
    return f"\nX: {', '.join(items)}"
```

### Rules
- Use conditional inclusion: `{section}` only if non-empty
- Use compact format: `\nField: value1, value2`
- Limit lists: `[:5]` or `[:3]` for long lists
- Never include empty sections
- Never use verbose labels like "The following X may be..."

---

## 5. Source of Truth Reference

| What | Where |
|------|-------|
| Element types | `CodeElement.element_type` docstring @ `src/magaldi_core/code_parser.py:207` |
| Element fields/metadata | `CodeElement` dataclass @ `src/magaldi_core/code_parser.py` |
| Extractor element types | `ExtractedElement` @ `src/magaldi_core/extractors/types.py` |
| Summarization prompts | `SYSTEM_PROMPTS`, `USER_PROMPTS`, `PROMPTS` @ `src/shared/ai/summarization.py` |
| Prompt sentence ranges | `LINE_THRESHOLDS`, `SENTENCE_RANGES` @ `src/shared/ai/summarization.py` |
| Feature/subfeature prompts | `FEATURE_*_PROMPT`, `SUBFEATURE_*_PROMPT` @ `src/shared/ai/clustering/feature_processor.py` |
| Glossary prompts | `GLOSSARY_*_PROMPT` @ `src/shared/ai/glossary/ai_extractor.py` |
| MCP tool implementations | `src/magaldi_mcp/tools_impl.py` |
| MCP tool schemas | `src/magaldi_mcp/tools/schemas/` |
| MCP formatters | `src/magaldi_mcp/formatters/` |
| ES field mappings | `src/shared/db/elasticsearch.py` |
| Web API routes | `src/magaldi_web/routes/` |
| Web frontend pages | `src/magaldi_web/frontend/src/pages/` |
| TypeScript types | `src/magaldi_web/frontend/src/api.ts` |

---

## 6. Verification

### Check element types have prompts
```bash
source .venv/bin/activate
python -c "from shared.ai.summarization import PROMPTS; print('Prompts:', list(PROMPTS.keys()))"
```

### Run tests
```bash
pytest tests/test_summarization.py -v
```

### Use integrity check skill
Run `/check-magaldi-integrity` to verify all element types and fields are properly exposed.
