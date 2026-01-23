# Phase B: Code Embeddings Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add separate code embeddings alongside existing summary embeddings to enable semantic search on actual code structure.

**Architecture:**
- Rename `embedding` → `summary_embedding` in schema
- Add new `code_embedding` field
- Update embedding pipeline to generate BOTH embeddings per element
- Add `search_mode` parameter to `search_code` tool

**Tech Stack:** Elasticsearch 8.11.0, Python, snowflake-arctic-embed2 (1024 dims)

---

## Task 1: Update Elasticsearch Schema

**Files:**
- Modify: `src/shared/db/elasticsearch.py`
- Test: `tests/test_db_elasticsearch.py`

### Step 1: Update INDEX_MAPPING

Change the schema to have two embedding fields:

```python
INDEX_MAPPING = {
    "mappings": {
        "properties": {
            # ... existing fields ...

            # Rename: was "embedding", now explicit
            "summary_embedding": {
                "type": "dense_vector",
                "dims": 1024,
                "index": True,
                "similarity": "cosine",
            },
            # NEW: embedding of raw_code
            "code_embedding": {
                "type": "dense_vector",
                "dims": 1024,
                "index": True,
                "similarity": "cosine",
            },
        }
    }
}
```

### Step 2: Update store_embedding method

Modify to accept an `embedding_type` parameter:

```python
def store_embedding(
    self,
    element_id: str,
    embedding: list[float],
    embedding_type: str = "summary",  # "summary" or "code"
) -> bool:
    field_name = f"{embedding_type}_embedding"
    # ... update document with embedding ...
```

### Step 3: Add migration function

Create a function to handle existing indices:

```python
def migrate_embedding_field(self) -> dict:
    """Migrate 'embedding' field to 'summary_embedding' for existing documents."""
    # Use update_by_query to rename field
```

### Step 4: Update search methods

Add `embedding_type` parameter to `search_similar`:

```python
def search_similar(
    self,
    embedding: list[float],
    embedding_type: str = "summary",  # NEW
    # ... other params
) -> list[dict]:
    field_name = f"{embedding_type}_embedding"
    # ... knn search on field_name ...
```

### Step 5: Commit

```bash
git add src/shared/db/elasticsearch.py tests/test_db_elasticsearch.py
git commit -m "feat(elasticsearch): add dual embedding schema (summary + code)"
```

---

## Task 2: Update Embedding Pipeline

**Files:**
- Modify: `src/shared/ai/embedding.py`
- Modify: `src/magaldi_core/processor.py`
- Test: `tests/test_embedding.py` (if exists)

### Step 1: Update build_embedding_text for summary

Rename to `build_summary_embedding_text` and keep current behavior:

```python
def build_summary_embedding_text(
    element: CodeElement,
    summary_cache: Any,
    max_tokens: int = 8000,
) -> str:
    """Build text for summary embedding (metadata + summary + context)."""
    # Current implementation unchanged
```

### Step 2: Add build_code_embedding_text

New function for code embedding:

```python
def build_code_embedding_text(
    element: CodeElement,
    max_tokens: int = 8000,
) -> str:
    """Build text for code embedding (raw code with minimal context).

    Args:
        element: Code element to embed.
        max_tokens: Maximum context tokens.

    Returns:
        Formatted text for code embedding.
    """
    parts = []

    # Add minimal context (file path, element type)
    parts.append(f"# {element.element_type}: {element.name}")
    parts.append(f"# File: {element.relative_path}")

    # Add signature if available
    if element.signature:
        parts.append(f"# Signature: {element.signature}")

    # Add the raw code
    if element.raw_code:
        parts.append(element.raw_code)

    text = "\n".join(parts)
    return validate_context_length(text, max_tokens)
```

### Step 3: Update processor to generate both embeddings

In `_embed_element`, generate both:

```python
def _embed_element(
    element: CodeElement,
    summary_cache: _SummaryCache,
    embed_client: CodeEmbeddingClient,
    config: ProcessingConfig,
) -> tuple[list[float], list[float]]:
    """Generate both embeddings for an element.

    Returns:
        Tuple of (summary_embedding, code_embedding)
    """
    # Summary embedding (existing logic)
    summary_text = build_summary_embedding_text(element, summary_cache, config.embed_max_context)
    summary_embedding = embed_client.embed_single(summary_text, timeout=config.embed_timeout)
    summary_embedding = normalize_vector(summary_embedding)

    # Code embedding (new)
    code_text = build_code_embedding_text(element, config.embed_max_context)
    code_embedding = embed_client.embed_single(code_text, timeout=config.embed_timeout)
    code_embedding = normalize_vector(code_embedding)

    return summary_embedding, code_embedding
```

### Step 4: Update _index_element to store both

```python
def _index_element(
    element: CodeElement,
    summary: str,
    summary_embedding: list[float] | None,
    code_embedding: list[float] | None,  # NEW
    es_repo: ElasticsearchRepository,
    # ...
) -> bool:
    # ... store both embeddings ...
```

### Step 5: Update TimingStats

Add separate tracking for summary vs code embedding time:

```python
@dataclass
class TimingStats:
    total_summary_embed_by_type: dict[str, float] = field(default_factory=dict)
    total_code_embed_by_type: dict[str, float] = field(default_factory=dict)
    summary_embed_counts_by_type: dict[str, int] = field(default_factory=dict)
    code_embed_counts_by_type: dict[str, int] = field(default_factory=dict)
```

### Step 6: Commit

```bash
git add src/shared/ai/embedding.py src/magaldi_core/processor.py
git commit -m "feat(embedding): generate dual embeddings (summary + code)"
```

---

## Task 3: Update CLI Progress Display

**Files:**
- Modify: `src/shared/cli.py` (or wherever progress is displayed)

### Step 1: Update progress display

Show both embedding statuses:

```
Processing elements...
Thread 1: [process_data] summary: done, code: processing
Thread 2: [validate_input] summary: processing, code: pending
```

### Step 2: Update timing display

Show separate times for summary and code embedding:

```
Timing breakdown:
  Summarization: 1.2s avg
  Summary embedding: 0.3s avg
  Code embedding: 0.4s avg
```

### Step 3: Commit

```bash
git add src/shared/cli.py
git commit -m "feat(cli): show dual embedding progress"
```

---

## Task 4: Update search_code Tool

**Files:**
- Modify: `src/magaldi_mcp/tools.py`
- Modify: `src/magaldi_mcp/server.py`
- Test: `tests/test_mcp_tools.py`

### Step 1: Add search_mode parameter

```python
def search_code(
    es: ElasticsearchRepository,
    query: str,
    search_mode: str = "hybrid",  # NEW: "summary", "code", or "hybrid"
    scope: str | None = None,
    repository: str | None = None,
    # ... existing params
) -> dict[str, Any]:
    """Search code semantically.

    Args:
        query: Natural language query.
        search_mode: Which embeddings to search:
            - "summary": Search summary embeddings only
            - "code": Search code embeddings only
            - "hybrid": Search both, combine scores (default)
        # ... rest
    """
```

### Step 2: Implement hybrid search

```python
if search_mode == "summary":
    results = es.search_similar(query_embedding, embedding_type="summary", ...)
elif search_mode == "code":
    results = es.search_similar(query_embedding, embedding_type="code", ...)
else:  # hybrid
    summary_results = es.search_similar(query_embedding, embedding_type="summary", ...)
    code_results = es.search_similar(query_embedding, embedding_type="code", ...)
    results = merge_and_rank(summary_results, code_results, weights=(0.5, 0.5))
```

### Step 3: Update server.py schema

Add `search_mode` to tool definition:

```python
"search_mode": {
    "type": "string",
    "enum": ["summary", "code", "hybrid"],
    "default": "hybrid",
    "description": "summary: search summaries only, code: search code only, hybrid: search both (default)"
}
```

### Step 4: Commit

```bash
git add src/magaldi_mcp/tools.py src/magaldi_mcp/server.py tests/test_mcp_tools.py
git commit -m "feat(mcp): add search_mode parameter to search_code"
```

---

## Task 5: Implement Similarity Tools

**Files:**
- Modify: `src/magaldi_mcp/tools.py`
- Modify: `src/magaldi_mcp/server.py`
- Test: `tests/test_mcp_tools.py`

### Step 1: Implement find_similar_structure

```python
def find_similar_structure(
    es: ElasticsearchRepository,
    element_id: str,
    scope: str,
    repository: str,
    username: str | None = None,
    min_similarity: float = 0.8,
    limit: int = 10,
) -> dict[str, Any]:
    """Find code that looks structurally similar (using code_embedding)."""
    # Get element's code_embedding
    # Search for similar using code_embedding
```

### Step 2: Implement find_similar_intent

```python
def find_similar_intent(
    es: ElasticsearchRepository,
    element_id: str,
    scope: str,
    repository: str,
    username: str | None = None,
    min_similarity: float = 0.7,
    limit: int = 10,
) -> dict[str, Any]:
    """Find code that does similar things (using summary_embedding)."""
    # Get element's summary_embedding
    # Search for similar using summary_embedding
```

### Step 3: Implement find_duplicates

```python
def find_duplicates(
    es: ElasticsearchRepository,
    scope: str,
    repository: str,
    username: str | None = None,
    min_similarity: float = 0.95,
    min_lines: int = 5,
) -> dict[str, Any]:
    """Find near-duplicate code across the codebase."""
    # Get all function/method elements above min_lines
    # For each, find others with code_embedding similarity >= threshold
    # Cluster into duplicate groups
```

### Step 4: Register tools in server.py

Add tool definitions and handlers.

### Step 5: Commit

```bash
git add src/magaldi_mcp/tools.py src/magaldi_mcp/server.py tests/test_mcp_tools.py
git commit -m "feat(mcp): add similarity tools (find_similar_structure, find_similar_intent, find_duplicates)"
```

---

## Task 6: Update Skill Documentation

**Files:**
- Modify: `src/magaldi_mcp/tools.py` (generate_skill template)
- Modify: `.claude/skills/magaldi/SKILL.md`

### Step 1: Update skill to mention search_mode

Document when to use each mode:

```markdown
### Semantic Search Modes

| Mode | Use When | Example |
|------|----------|---------|
| `summary` | Finding code by what it DOES | "find authentication logic" |
| `code` | Finding code that LOOKS LIKE something | "find code similar to this pattern" |
| `hybrid` | General search (default) | "find user validation" |
```

### Step 2: Document similarity tools

```markdown
### Finding Similar Code

- `find_similar_structure` - Find copy-paste code, similar patterns
- `find_similar_intent` - Find code doing same thing, different implementation
- `find_duplicates` - Find near-duplicate functions (refactoring candidates)
```

### Step 3: Commit

```bash
git add src/magaldi_mcp/tools.py .claude/skills/magaldi/SKILL.md
git commit -m "docs: update skill with dual embedding and similarity tools"
```

---

## Task 7: Run Full Test Suite

### Step 1: Run tests

```bash
pytest tests/ -v --timeout=60
```

### Step 2: Fix any failures

### Step 3: Commit fixes

```bash
git add -A
git commit -m "fix: address test failures after dual embedding implementation"
```

---

## Summary

Phase B implementation creates:

1. **Schema changes:** `summary_embedding` + `code_embedding` fields
2. **Pipeline changes:** Generate both embeddings per element
3. **CLI changes:** Show both embedding statuses and timing
4. **Tool changes:** `search_mode` parameter on `search_code`
5. **New tools:** `find_similar_structure`, `find_similar_intent`, `find_duplicates`

**Re-indexing required:** After schema update, existing documents need re-indexing to populate `code_embedding` field.
