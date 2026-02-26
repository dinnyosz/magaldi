# Plan: Semantic Passport Embeddings for Call Graph Resolution

## Problem Statement

Currently, `build_code_embedding_text()` embeds **raw source code** for implementation matching. This is wasteful and noisy:

- Raw code includes boilerplate (imports, error handling, logging) that drowns out semantic signal
- Large functions consume excessive tokens without proportionally better matching
- Two implementations of the same algorithm in different styles look dissimilar as raw text
- The code embedding is **not used for call resolution** at all — only `summary_embedding` is used

Meanwhile, `build_summary_embedding_text()` is decent but misses key structural signals:

- No outbound call information (what this function depends on)
- No parameter type shapes (the "data contract" the function operates on)
- No import context (what domain libraries it uses)
- No sibling context (what other functions live in the same class)
- Relies heavily on LLM summary quality (garbage in → garbage out)

## Current State

### What We Embed Today

**summary_embedding** (used for call resolution + semantic relationships):
```
File: src/shared/db/repositories/elements.py
File context: <file LLM summary>
Class context: <class LLM summary>
Function: get_element
Summary: <LLM summary>
Signature: (self, element_id: str) -> dict | None
Docstring: <first 500 chars>
```

**code_embedding** (used for find_similar only):
```
# function: get_element
# File: src/shared/db/repositories/elements.py
# Signature: (self, element_id: str) -> dict | None
<raw source code, up to 8000 tokens>
```

### Call Resolution Pipeline (Phase 5)

1. **Static resolution** (Strategies 1-5): import-based, type-annotated → works well
2. **Embedding resolution** (Strategy 6): cosine similarity ≥ 0.7 on `summary_embedding` → fuzzy, misses calls
3. **Semantic relationships**: top-K similar by `summary_embedding` → noisy neighbors

## Proposed: Semantic Passport Embeddings

Replace the naive raw-code approach in `code_embedding` with a **Semantic Passport** — a structured, token-efficient representation of what a function *is* and *does* without its implementation.

### The Semantic Passport Format

```
# Path: src/shared/db/repositories/elements.py > ElasticsearchRepository > get_element
# Type: method (async: no, visibility: public, test: no)
# Signature: (self, element_id: str) -> dict | None
# Parameters: element_id:str
# Returns: dict | None
# Calls: client.get, INDEX_NAME
# Imports: elasticsearch.Elasticsearch, shared.db.repositories.base.INDEX_NAME
# Siblings: store_element, delete_element, find_elements_by_type
# Decorators: none
# Patterns: repository
# Summary: <LLM summary, max 200 chars>
```

Each line is a **semantic signal**. Together they form a fingerprint that captures:

| Signal | What It Captures | Why It Matters |
|--------|-----------------|----------------|
| **Breadcrumbs** (path) | Architectural placement | Functions in `routes/auth.py` vs `utils/crypto.py` have different intent |
| **Normalized Signature** | Data contract shape | `(user: User, token: str) -> bool` clusters with auth-related code |
| **Parameter Types** | Input domain | Type names carry semantic meaning (`UserRecord` vs `HttpRequest`) |
| **Return Type** | Output shape | Groups functions by what they produce |
| **Outbound Calls** | Dependency fingerprint | Functions calling `bcrypt.hash` + `db.store` = auth storage pattern |
| **Import Context** | Domain libraries used | `from fastapi import ...` = HTTP handler domain |
| **Sibling Names** | Co-located behavior | Methods in same class share semantic intent |
| **Decorators** | Cross-cutting concerns | `@login_required`, `@cache`, `@pytest.fixture` |
| **Patterns** | Design role | `repository`, `factory`, `singleton` |
| **Summary** | Human intent (brief) | Ties structural signals to natural language |

### What We Remove

- **Raw source code** from `code_embedding` — replaced entirely by passport
- **Full docstrings** — too verbose, summary captures intent better
- **File/class summary parroting** — replaced by sibling names and breadcrumbs

## Implementation Steps

### Step 1: Build the Semantic Passport Builder

**File**: `src/shared/ai/embedding.py`

Create `build_semantic_passport()` function that replaces `build_code_embedding_text()`:

```python
def build_semantic_passport(
    element: CodeElement,
    embedding_store: EmbeddingStore,
    sibling_names: list[str] | None = None,
    file_imports: list[Import] | None = None,
    max_tokens: int = 2000,  # Much smaller than 8000!
) -> str:
```

Key design decisions:
- **Max 2000 tokens** — passport is compact by design, no need for 8000
- **Sibling names** passed in (avoids passport builder needing to query ES)
- **File imports** passed in (already available during embedding phase)
- **No raw code** — the whole point

### Step 2: Gather Sibling Context During Embedding Phase

**File**: `src/shared/ai/embedding.py` (in `generate_embeddings()` or equivalent)

When processing elements for a file, collect sibling names:
- For methods: other method names in the same class
- For functions: other function names in the same file
- Limit to 10 sibling names (sorted alphabetically for consistency)

This data is already available — the embedding phase processes elements grouped by file. We just need to pass it through.

### Step 3: Gather File Import Context

**File**: `src/shared/ai/embedding.py`

File imports are already stored on file elements. During embedding:
1. When processing a file's children, load the file element's `imports` list
2. Extract module names (e.g., `elasticsearch`, `fastapi`, `shared.db`)
3. Pass to the passport builder

### Step 4: Switch `code_embedding` to Use Passport

**File**: `src/shared/ai/embedding.py`

Replace the call to `build_code_embedding_text()` with `build_semantic_passport()` in the embedding generation flow. The field name `code_embedding` stays the same in ES — only the content changes.

**Migration**: Since embeddings are regenerated on each parse, no data migration needed. The next `magaldi parse` will produce passport-based embeddings.

### Step 5: Use Both Embeddings for Call Resolution

**File**: `src/magaldi_core/call_resolution.py`

Currently `resolve_calls_by_embedding()` only uses `summary_embedding`. Enhance it to use both:

```python
# Current: single embedding comparison
score = cosine_similarity(caller.summary_embedding, candidate.summary_embedding)

# Enhanced: weighted combination
summary_score = cosine_similarity(caller.summary_embedding, candidate.summary_embedding)
passport_score = cosine_similarity(caller.code_embedding, candidate.code_embedding)
score = (w1 * summary_score) + (w2 * passport_score)
```

Suggested weights: `w1=0.4` (semantic intent), `w2=0.6` (structural fingerprint)

Rationale: The passport embedding captures *structural compatibility* (does this function call things that look like the candidate? do they share imports?), which is a stronger signal for call resolution than semantic similarity alone.

### Step 6: Enhance Semantic Relationships with Dual Scoring

**File**: `src/magaldi_core/call_resolution.py`

For `compute_semantic_relationships()`, use the passport embedding as a re-ranking signal:

1. Retrieve top-K candidates by `summary_embedding` (as today)
2. Re-rank by weighted combination of summary + passport similarity
3. This filters out "topically similar but structurally different" false positives

### Step 7: Tune Thresholds

Current thresholds may need adjustment since passport embeddings are more precise:

| Parameter | Current | Proposed | Rationale |
|-----------|---------|----------|-----------|
| `similarity_threshold` (call resolution) | 0.7 | 0.65 | Passport is more precise, can lower threshold |
| `min_score` (semantic relationships) | 0.5 | 0.5 | Keep, re-ranking handles noise |
| `top_k` (semantic relationships) | 10 | 10 | Keep |

These are starting points — will need empirical tuning.

## What We Do NOT Change

- **Tech stack**: Keep ES, keep current embedding model
- **Static resolution** (Strategies 1-5): Already good, no changes needed
- **`summary_embedding` content**: Keep as-is, it serves semantic search well
- **ES field names**: `code_embedding` field stays, content changes
- **Phase ordering**: Still Phase 4 (embed) → Phase 5 (resolve)

## Expected Impact

### Call Resolution
- **More accurate** matching for untyped calls — structural fingerprint (calls, imports, types) is a stronger signal than raw code text similarity
- **Lower false positives** — two functions that are topically related but structurally different (different calls, different imports) will score lower
- **Better cross-language potential** — passport format is language-agnostic

### Token Efficiency
- **code_embedding text shrinks from ~8000 tokens to ~500-1000** — passport is much more compact
- **Embedding generation is faster** — less text to embed per element
- **Same ES storage** — embedding vector dimensions don't change

### Semantic Relationships
- **Better "similar function" suggestions** — dual scoring filters noise
- **More actionable neighbors** — structurally similar functions are more likely to be relevant for refactoring

## Risk Mitigation

1. **A/B comparison**: Run passport embeddings alongside current code embeddings on a test repo, compare call resolution accuracy before switching
2. **Gradual rollout**: Can introduce passport as a third embedding field first, then deprecate raw code embedding
3. **Fallback**: If passport embedding performs worse for `find_similar` (unlikely but possible), keep raw code embedding for that specific tool

## File Changes Summary

| File | Change |
|------|--------|
| `src/shared/ai/embedding.py` | Add `build_semantic_passport()`, modify embedding generation to gather siblings + imports |
| `src/magaldi_core/call_resolution.py` | Use dual scoring in `resolve_calls_by_embedding()` and `compute_semantic_relationships()` |
| `tests/test_embedding.py` | Tests for passport builder |
| `tests/test_call_resolution.py` | Tests for dual scoring |

## Open Questions

1. **Sibling limit**: How many sibling names to include? Proposed: 10, sorted alphabetically
2. **Import depth**: Include full module path (`shared.db.repositories.elements`) or just top-level (`shared`)? Proposed: top 2 levels (`shared.db`)
3. **Weight tuning**: The `w1`/`w2` weights need empirical tuning. Could make them configurable in `magaldi.yaml`
4. **Call names normalization**: Should we normalize call names (e.g., strip `self.`) before including in passport?
