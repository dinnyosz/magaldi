# Decision: Hybrid SCM Query Architecture for Extractors

**Date:** 2026-01-28
**Author:** dinnyosz
**Context:** Parser Lab implementation and extractor refactoring

---

## Decision: Use hybrid approach - SCM queries for pattern matching, Python for post-processing

**Original plan:** Fully convert extractors to SCM queries, replacing imperative tree-walking entirely.

**Deviation:** Keep Python post-processing; SCM queries only handle pattern matching.

**Why:**
- SCM queries are declarative and easier to maintain for pattern matching
- Complex post-processing (building ExtractedElement objects, parent-child relationships, signatures) is better suited to Python
- Allows incremental migration without rewriting everything at once
- Parser_lab can suggest SCM query changes while Python handles the hard parts

---

## Options Considered

### 1. Full SCM conversion
**Pros:**
- Fully declarative, easier for AI to modify
- Hot-reloadable query files
- Consistent approach across all languages

**Cons:**
- SCM queries can't build complex Python objects
- Would need to rewrite all post-processing logic
- High risk of regressions in a large codebase
- SCM predicates are limited compared to Python logic

### 2. Hybrid approach (chosen)
**Pros:**
- SCM handles what it's good at (pattern matching)
- Python handles what it's good at (complex logic)
- Incremental migration path
- Fallback to imperative if queries unavailable
- Parser_lab can still suggest query improvements

**Cons:**
- Two systems to understand
- Some duplication between query patterns and Python handlers

### 3. Keep imperative only
**Pros:**
- No migration needed
- Single approach to understand

**Cons:**
- Harder for AI to modify (Python tree-walking is verbose)
- No hot-reload capability
- Defeats the purpose of parser_lab self-improvement

---

## Final Decision

**Hybrid approach** - SCM queries for pattern matching with Python post-processing.

**Rationale:** The goal of parser_lab is to let AI agents improve the parser. SCM queries make pattern matching declarative and editable, which is the hard part to get right. Post-processing is mechanical and doesn't need AI modification. This gives us 80% of the benefit with 20% of the risk.

**Implementation details:**
- `extract_python_elements()` checks for SCM queries, uses them if available
- Falls back to `_extract_python_elements_imperative()` if no queries
- Query results processed by existing `_extract_python_function()`, `_extract_python_class()` helpers
- Added `run_query_on_tree()` to avoid re-parsing already-parsed trees

---

## Related Files
- `src/magaldi_core/query_runner.py` - Query infrastructure
- `src/magaldi_core/queries/` - SCM query files
- `src/magaldi_core/extractors/python.py` - Hybrid implementation
- `plans/parser_lab_and_scm_refactor.md` - Original plan
