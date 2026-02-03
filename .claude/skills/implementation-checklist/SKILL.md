# Implementation Checklist

**MANDATORY**: Read this skill before implementing any feature that adds new element types, extracts new metadata, or modifies parsing/summarization.

## Core Principle

Every piece of data extracted during parsing MUST be surfaced to users through ALL relevant interfaces: Web UI, MCP tools, and AI summarization. No gaps allowed.

---

## 1. New Element Type Checklist

When adding a new element type (like `trait`, `enum`, `interface`):

### Parsing Layer
- [ ] Add to `CodeElement.element_type` docstring in `src/magaldi_core/code_parser.py`
- [ ] Implement extraction in relevant language extractors

### Summarization Prompts (`src/shared/ai/summarization.py`)
- [ ] Add to `LINE_THRESHOLDS` dict
- [ ] Add to `SENTENCE_RANGES` dict
- [ ] Add to `SYSTEM_PROMPTS` dict with:
  - Relevant questions for this element type
  - Anti-verbose instruction (see below)
- [ ] Add to `USER_PROMPTS` dict
- [ ] Add to legacy `PROMPTS` dict with anti-verbose instruction
- [ ] Update context building if element needs special sections (e.g., `base_classes_section` for interfaces)

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

When extracting new metadata (like `http_routes`, `security_issues`, `complexity`):

### Parsing Layer
- [ ] Add field to `CodeElement` dataclass in `src/magaldi_core/code_parser.py`
- [ ] Add field to `ExtractedElement` in `src/magaldi_core/extractors/types.py`
- [ ] Implement extraction in relevant extractors

### Elasticsearch
- [ ] Add field mapping in `src/shared/db/elasticsearch.py`
- [ ] Update element serialization in `src/magaldi_core/storage.py`

### Summarization Prompts
- [ ] Add to relevant prompt context sections (token-efficient!)
- [ ] Only include if non-empty: `if element.new_field:`
- [ ] Keep format minimal: `\nNew field: {', '.join(element.new_field)}`

### Web UI
- [ ] Display in element detail page
- [ ] Add to search filters if filterable
- [ ] Update TypeScript interfaces

### MCP Tools
- [ ] Include in `get_element` response
- [ ] Include in search results if relevant
- [ ] Create dedicated tool if complex (e.g., `find_security_issues`)
- [ ] Update formatters to display new field

---

## 3. Anti-Verbose Prompt Rules

**ALL summarization prompts MUST include anti-verbose instructions.**

### Format
Add at the end of every system/legacy prompt:
```
Start [action] - never start with "This [type]...", "The X [type]...", or similar.
```

### Examples by Element Type

| Type | Anti-verbose instruction |
|------|-------------------------|
| file | `Start directly with what it does - never start with "This module...", "This file...", or similar.` |
| class | `Start directly with what it models/does - never start with "This class...", "The X class...", or similar.` |
| function/method | `Start with an action verb - never start with "This function...", "The X function...", or "This function is used to...".` |
| interface/trait | `Start directly with what contract/capability it defines - never start with "This interface...", "This trait...", or similar.` |
| enum | `Start directly with what it represents - never start with "This enum...", "The X enum...", or similar.` |
| constant/variable | `Start directly with what it represents/holds - never start with "This constant...", "This variable...", or similar.` |
| import | `Start with what it imports - never start with "This import...", "The import...", or similar.` |

### Bad vs Good Examples

| Bad | Good |
|-----|------|
| "This module is responsible for handling authentication" | "Handles user authentication via JWT tokens" |
| "The UserService class is used to manage users" | "Manages user CRUD operations and session state" |
| "This function is used to validate input" | "Validates and sanitizes user input against schema" |

---

## 4. Token-Efficient Context in Prompts

When adding metadata to prompts:

### Do
- Use conditional inclusion: `{exceptions_section}` only if non-empty
- Use compact format: `\nRaises: ValueError, TypeError`
- Limit lists: `[:5]` or `[:3]` for long lists
- Use helper functions: `_build_X_section(element)`

### Don't
- Include empty sections
- Use verbose labels: "The following exceptions may be raised:"
- Include full details when names suffice
- Repeat information available in code

### Example Helper Pattern
```python
def _build_exceptions_section(element: CodeElement) -> str:
    if not element.exceptions_raised:
        return ""
    return f"\nRaises: {', '.join(element.exceptions_raised)}"
```

---

## 5. Quick Reference: Key Files

| Area | Files |
|------|-------|
| Element types | `src/magaldi_core/code_parser.py:207` |
| Summarization | `src/shared/ai/summarization.py` |
| Feature/subfeature prompts | `src/shared/ai/clustering/feature_processor.py` |
| Glossary prompts | `src/shared/ai/glossary/ai_extractor.py` |
| MCP tools | `src/magaldi_mcp/tools_impl.py` |
| MCP schemas | `src/magaldi_mcp/tools/schemas/` |
| MCP formatters | `src/magaldi_mcp/formatters/` |
| Web API | `src/magaldi_web/routes/` |
| Web frontend | `src/magaldi_web/frontend/src/` |
| ES mappings | `src/shared/db/elasticsearch.py` |

---

## 6. Verification Commands

```bash
# Check all element types have prompts
python -c "from shared.ai.summarization import PROMPTS; print(list(PROMPTS.keys()))"

# Run summarization tests
pytest tests/test_summarization.py -v

# Check MCP tool outputs
# Use magaldi MCP tools to verify new data is exposed
```
