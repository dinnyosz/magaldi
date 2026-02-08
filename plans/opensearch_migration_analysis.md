# OpenSearch Migration Analysis for Magaldi

## Executive Summary

Moving from Elasticsearch 8.11.0 to OpenSearch would be **beneficial** for Magaldi, primarily for licensing freedom (Apache 2.0 vs SSPL), but the real value lies in OpenSearch's **built-in neural search, hybrid search pipelines, and ML Commons** — features that align perfectly with Magaldi's AI-powered code discovery mission.

The migration effort is **moderate** — the Python client swap is near-trivial, but the real work is adapting to OpenSearch's query syntax for vector operations and rearchitecting to exploit native neural capabilities.

---

## Current Elasticsearch Usage in Magaldi

### Client & Infrastructure
- **Client**: `elasticsearch>=8.11.0,<9.0.0` (Python `elasticsearch-py`)
- **Docker image**: `elasticsearch:8.11.0`
- **Single index**: `magaldi-code-elements` (plus `magaldi-relationships`, `magaldi-external-refs`)
- **Visualization**: Kibana 8.11.0

### Features Actively Used
| Feature | Where | OpenSearch Compatible? |
|---------|-------|----------------------|
| `dense_vector` (dims=1024, cosine) | `summary_embedding`, `code_embedding` | Yes (via k-NN plugin) |
| `script_score` + `cosineSimilarity()` | `search.py:search_by_vector()` | Yes, but native `knn` query is better |
| `multi_match` (BM25) | `search.py:search_by_text()`, `search_by_keyword()` | Yes, identical |
| `regexp` on keyword fields | `search.py:search_by_regexp()` | Yes, identical |
| `wildcard` queries | `search.py:search_by_wildcard()` | Yes, identical |
| `nested` field type (18+ nested fields) | Index mapping extensively | Yes, identical |
| `collapse` (field collapsing) | `relationships.py` | Yes, identical |
| `bulk` operations | 24+ bulk calls across repos | Yes, identical |
| `delete_by_query` | Element/file cleanup | Yes, identical |
| `bool` filter/must/should | Everywhere | Yes, identical |

### What's NOT Used (ES-Specific)
- No ES Security features (xpack disabled)
- No ingest pipelines
- No ML features from ES
- No cross-cluster search
- No ILM (index lifecycle management)
- No ES-specific aggregations beyond basic

---

## Migration Effort Assessment

### Trivial Changes (< 1 day)
1. **Python client swap**: `elasticsearch-py` → `opensearch-py` (fork, near-identical API)
   - Change import: `from elasticsearch import Elasticsearch` → `from opensearchpy import OpenSearch`
   - Client constructor is compatible
   - All `client.search()`, `client.index()`, `client.bulk()` calls unchanged
2. **Docker image**: `elasticsearch:8.11.0` → `opensearchproject/opensearch:2.19.0`
3. **Kibana → OpenSearch Dashboards**: Drop-in replacement

### Moderate Changes (1-3 days)
1. **Vector field mapping**: `dense_vector` → `knn_vector` with engine configuration
   ```yaml
   # Current (ES)
   summary_embedding:
     type: dense_vector
     dims: 1024
     index: true
     similarity: cosine

   # OpenSearch
   summary_embedding:
     type: knn_vector
     dimension: 1024
     method:
       name: hnsw
       space_type: cosinesimil
       engine: faiss  # or lucene
   ```
2. **Vector search queries**: Replace `script_score` + `cosineSimilarity()` with native `knn` query
   ```json
   // Current (ES): script_score wrapper
   { "script_score": { "query": {...}, "script": { "source": "cosineSimilarity(...)" } } }

   // OpenSearch: native knn
   { "knn": { "summary_embedding": { "vector": [...], "k": 10, "filter": {...} } } }
   ```
3. **Index settings**: Add `index.knn: true` to index settings
4. **Config abstraction**: Update `MagaldiConfig` for OpenSearch-specific settings

### No Changes Needed
- All `bool`, `term`, `terms`, `multi_match`, `regexp`, `wildcard`, `nested`, `collapse`, `bulk`, `delete_by_query` — these are API-compatible

---

## OpenSearch Features That Unlock New Magaldi Capabilities

### 1. Native Hybrid Search (HIGH IMPACT)

**What**: OpenSearch has a first-class `hybrid` query type that combines BM25 + vector search with automatic score normalization.

**Why it matters for Magaldi**: Currently, `search_code` does BM25 OR vector search. With hybrid search, every query could combine keyword matching (function name, exact code patterns) with semantic similarity (what the code does) in a single query, with tunable weights.

```json
{
  "query": {
    "hybrid": {
      "queries": [
        { "multi_match": { "query": "authentication", "fields": ["name^3", "summary^2"] } },
        { "knn": { "summary_embedding": { "vector": [...], "k": 50 } } }
      ]
    }
  }
}
```

Combined with a **normalization search pipeline** (min-max or L2 normalization + arithmetic_mean or RRF combination), this would dramatically improve search relevance — a user searching "handle authentication" would match both functions literally named `authenticate` AND semantically similar functions like `verify_token`.

**Magaldi use case**: Replace the current "BM25 fallback when no embeddings" pattern with always-on hybrid search. Tune BM25 vs vector weights per query type (e.g., `find_usages` favors exact match, `search_code` favors semantic).

### 2. Neural Sparse Search (HIGH IMPACT)

**What**: Uses learned sparse representations (SPLADE-like) instead of dense vectors. Tokens get weighted scores, stored in an inverted index — as efficient as BM25 but with semantic understanding.

**Why it matters for Magaldi**:
- **No external embedding service needed** for basic semantic search — the model runs inside OpenSearch
- **Memory efficient** — sparse vectors use the existing inverted index, not separate HNSW graphs
- **Better for code**: Code has many exact tokens (function names, keywords) that sparse models handle better than dense embeddings that compress everything into 1024 floats
- **Multilingual code comments**: OpenSearch's sparse models are expanding to multiple languages

**Magaldi use case**: Use neural sparse as a middle ground between BM25 and dense vectors. For example, a `neural_sparse` field could power fast "what does this code do" search without the overhead of maintaining dense embeddings for every element. This could replace or complement the current embedding pipeline for lower-priority elements (variables, imports) where dense embedding is overkill.

### 3. ML Commons — Models Inside the Search Engine (MEDIUM-HIGH IMPACT)

**What**: OpenSearch can host ML models directly (embedding models, cross-encoders, sparse encoders) and apply them at index/query time via search pipelines.

**Why it matters for Magaldi**: Currently, Magaldi runs a separate LLM pipeline (Ollama/LiteLLM) to generate embeddings, then stores them. This is a complex multi-phase process. With ML Commons:
- **Ingest-time embedding**: Index raw text → OpenSearch generates embeddings automatically via an ingest pipeline with a model processor
- **Query-time embedding**: User sends text query → OpenSearch converts to vector automatically via `neural` query
- **No separate embedding service**: Removes the Ollama/embedding dependency for search

**Magaldi use case**: Simplify the Phase 4 pipeline. Instead of `parse → summarize → embed → index`, you could `parse → summarize → index` and let OpenSearch handle embedding at ingest time. The `neural` query type handles the query-side embedding automatically.

### 4. Search Pipelines with Reranking (MEDIUM IMPACT)

**What**: OpenSearch supports search pipelines — a chain of processors that transform search results. This includes normalization, reranking (including cross-encoder reranking via ML Commons), and score explanation.

**Why it matters for Magaldi**:
- **Two-stage retrieval**: Fast first-pass (BM25 + kNN) → expensive reranking (cross-encoder model) for top-N results
- **Score explanation**: The `explanation-processor` shows how hybrid scores were computed — useful for debugging search quality
- **Reciprocal Rank Fusion (RRF)**: Merges results from multiple sources without needing score normalization — simpler and often more effective

**Magaldi use case**: Implement a search pipeline that does:
1. Hybrid query (BM25 + vector) → 100 candidates
2. Cross-encoder rerank using a code-specific model → top 20
3. Return with score explanation

This would significantly improve `search_code` and `find_similar` quality.

### 5. Multiple k-NN Engines: Faiss + Lucene (MEDIUM IMPACT)

**What**: OpenSearch supports 3 vector engines: **Faiss** (Meta's library), **Lucene** (built-in), and NMSLIB (deprecated). Each has different tradeoffs.

**Why it matters for Magaldi**:
- **Faiss IVF**: For large codebases (millions of elements), Faiss IVF with product quantization compresses vectors dramatically while maintaining good recall
- **Faiss HNSW**: Better memory efficiency than Lucene for large vector counts
- **Lucene**: Smart filtering — automatically chooses pre/post filtering strategy based on filter selectivity
- Engine choice can be per-index, allowing different strategies for different data

**Magaldi use case**: Use Lucene engine for small-medium repos (smart filtering is great when queries always filter by scope+repository). Switch to Faiss for multi-tenant hosting with millions of elements across many repos.

### 6. Anomaly Detection on Code Metrics (CREATIVE/OUT-OF-BOX)

**What**: OpenSearch has a built-in anomaly detection plugin using Random Cut Forest (RCF) algorithm.

**Why Magaldi could use this**:
- **Detect complexity spikes**: Monitor `complexity.cyclomatic` over time per repository — alert when a commit introduces unusually complex code
- **Detect code smell patterns**: Track metrics like `param_count`, `nesting_depth`, `line_count` — anomaly detection flags outlier functions
- **Track codebase health**: Time-series of total element count, average complexity, documentation coverage — detect degradation trends

**Magaldi use case**: A `magaldi watch` mode that continuously indexes and uses OpenSearch anomaly detection to flag concerning code changes. "Your last commit introduced 3 functions with abnormally high complexity compared to your codebase average."

### 7. PPL (Piped Processing Language) (LOW-MEDIUM IMPACT)

**What**: SQL-like query language with pipe syntax: `source = index | where type = 'function' | stats count() by language`

**Why it matters for Magaldi**: Could power a natural-language-to-PPL query interface. PPL is more intuitive than DSL for ad-hoc analytics queries like "how many functions per language", "average complexity by module", "files with most TODOs".

**Magaldi use case**: Expose a `magaldi query` CLI command or MCP tool that accepts PPL queries for ad-hoc codebase analytics without building custom ES DSL queries for every possible question.

### 8. Alerting Plugin (LOW-MEDIUM IMPACT)

**What**: Built-in alerting with monitors, triggers, and notification channels.

**Magaldi use case**: For CI/CD integration — set up monitors that alert when:
- A function's complexity exceeds a threshold after re-indexing
- New security issues are detected (hardcoded secrets)
- Documentation coverage drops below a threshold
- Dead code count increases

---

## Risk Assessment

### Risks of Migration
| Risk | Severity | Mitigation |
|------|----------|------------|
| API incompatibility in edge cases | Medium | OS maintains ES 7.10 compat; test thoroughly |
| Performance regression for vector search | Medium | Benchmark on actual data; Faiss engine may actually be faster |
| Smaller ecosystem/community | Low | OS community is growing rapidly; AWS backing |
| Plugin compatibility (Kibana plugins) | Low | Magaldi uses basic visualization only |
| Version lag vs ES features | Medium | OS has diverged with its own features; some ES 8.x features won't be in OS |

### Risks of NOT Migrating
| Risk | Severity | Notes |
|------|----------|-------|
| SSPL licensing concerns for users | High | Users hosting Magaldi as a service face legal uncertainty |
| Missing hybrid search | High | Currently doing BM25 OR vector, never both together |
| Maintaining separate embedding pipeline | Medium | ML Commons could simplify architecture significantly |
| No neural sparse option | Medium | Dense-only vector search is expensive for large codebases |

---

## Docker Setup: OpenSearch Equivalent

### Current (Elasticsearch)
```yaml
elasticsearch:
  image: elasticsearch:8.11.0
  environment:
    - discovery.type=single-node
    - xpack.security.enabled=false
    - xpack.security.enrollment.enabled=false
    - xpack.monitoring.collection.enabled=true
    - xpack.monitoring.elasticsearch.collection.enabled=true
    - "ES_JAVA_OPTS=-Xms${ES_HEAP_SIZE:-1g} -Xmx${ES_HEAP_SIZE:-1g}"

kibana:
  image: kibana:8.11.0
  environment:
    - ELASTICSEARCH_HOSTS=http://elasticsearch:9200
    - xpack.security.enabled=false
```

### OpenSearch Replacement
```yaml
opensearch:
  image: opensearchproject/opensearch:2.19.0
  container_name: magaldi-opensearch
  restart: unless-stopped
  environment:
    - discovery.type=single-node
    - DISABLE_SECURITY_PLUGIN=true          # Equivalent of xpack.security.enabled=false
    - DISABLE_INSTALL_DEMO_CONFIG=true      # Skip demo certs/users
    - "OPENSEARCH_JAVA_OPTS=-Xms${ES_HEAP_SIZE:-1g} -Xmx${ES_HEAP_SIZE:-1g}"
    - plugins.security.disabled=true        # Fully disable security plugin
    - cluster.name=magaldi
  ports:
    - "${ES_PORT:-9200}:9200"
    - "9600:9600"                           # Performance Analyzer port
  volumes:
    - os_data:/usr/share/opensearch/data
  healthcheck:
    test: ["CMD-SHELL", "curl -sf http://localhost:9200/_cluster/health || exit 1"]
    interval: 10s
    timeout: 5s
    retries: 5
    start_period: 30s

opensearch-dashboards:
  image: opensearchproject/opensearch-dashboards:2.19.0
  container_name: magaldi-dashboards
  restart: unless-stopped
  environment:
    - OPENSEARCH_HOSTS=["http://opensearch:9200"]
    - DISABLE_SECURITY_DASHBOARDS_PLUGIN=true
  ports:
    - "${KIBANA_PORT:-5601}:5601"
  depends_on:
    opensearch:
      condition: service_healthy
  healthcheck:
    test: ["CMD-SHELL", "curl -sf http://localhost:5601/api/status || exit 1"]
    interval: 30s
    timeout: 10s
    retries: 5
    start_period: 60s
```

### Key Docker Differences
| Aspect | Elasticsearch | OpenSearch |
|--------|--------------|------------|
| Image | `elasticsearch:8.11.0` | `opensearchproject/opensearch:2.19.0` |
| Dashboard | `kibana:8.11.0` | `opensearchproject/opensearch-dashboards:2.19.0` |
| Disable security | `xpack.security.enabled=false` | `DISABLE_SECURITY_PLUGIN=true` + `plugins.security.disabled=true` |
| Java opts env var | `ES_JAVA_OPTS` | `OPENSEARCH_JAVA_OPTS` |
| Data volume path | `/usr/share/elasticsearch/data` | `/usr/share/opensearch/data` |
| Extra port | none | `9600` (Performance Analyzer) |
| Dashboard hosts env | `ELASTICSEARCH_HOSTS=http://...` | `OPENSEARCH_HOSTS=["http://..."]` (JSON array) |
| Volume name | `es_data` | `os_data` (rename to avoid conflicts) |

### Makefile Updates Needed
```makefile
# Current
services:
	docker compose up -d elasticsearch redis kibana

# OpenSearch
services:
	docker compose up -d opensearch redis opensearch-dashboards
```

### Environment Variables in Web Service
```yaml
# Current
- MAGALDI_ES_HOST=elasticsearch
- MAGALDI_ES_PORT=9200

# OpenSearch (new env var names to match the purge plan)
- MAGALDI_SEARCH_HOST=opensearch
- MAGALDI_SEARCH_PORT=9200
```

The healthcheck endpoint (`/_cluster/health`) and port (`9200`) are identical between ES and OpenSearch, so no changes needed there. The env var names change from `MAGALDI_ES_*` to `MAGALDI_SEARCH_*` as part of the naming purge.

---

## Elasticsearch Purge Plan

Once OpenSearch is battle-tested, all Elasticsearch-specific naming/references will be removed from the codebase. A backend-agnostic compatibility layer will remain so both ES and OpenSearch can be used, but the **default and primary** target will be OpenSearch.

### Naming Renames (Purge "Elasticsearch" from identifiers)

| Current Name | New Name | Location |
|-------------|----------|----------|
| `ElasticsearchBase` | `SearchBackendBase` | `src/shared/db/repositories/base.py` |
| `ElasticsearchRepository` | `SearchRepository` (facade) | `src/shared/db/repositories/__init__.py` |
| `ElasticsearchEmbeddingStore` | `EmbeddingStore` | `src/shared/db/elasticsearch.py` → rename file |
| `ElasticsearchFileStateRepository` | `FileStateRepository` | `src/shared/db/elasticsearch.py` → rename file |
| `ElasticsearchConfig` | `SearchBackendConfig` | `src/shared/config.py` |
| `src/shared/db/elasticsearch.py` | `src/shared/db/store.py` | File rename |
| `MAGALDI_ES_HOST` / `MAGALDI_ES_PORT` | `MAGALDI_SEARCH_HOST` / `MAGALDI_SEARCH_PORT` | Config, docker-compose, env |

### Import Rewrites (43 files)

All `from shared.db.elasticsearch import ElasticsearchRepository` → `from shared.db.store import SearchRepository`

**Source files (34):**
- `src/magaldi_mcp/server.py`
- `src/magaldi_mcp/tools/*.py` (13 tool files: analysis, call_graph, dependencies, design_patterns, elements, files, glossary, patterns, quality, repository, search, trees, usages)
- `src/magaldi_web/dependencies.py`
- `src/magaldi_web/routes/*.py` (9 route files: admin, analysis, browse, dashboard, elements, glossary, repos, search, vectormap)
- `src/magaldi_core/processor/__init__.py`
- `src/magaldi_core/processor/helpers.py`
- `src/magaldi_core/call_resolution.py` (uses `ElasticsearchRepository` as parameter type + docstrings throughout)
- `src/magaldi_core/storage.py` (docstring references to Elasticsearch)
- `src/magaldi_core/extractors/call_categorizer.py` (list of known library names includes "elasticsearch")
- `src/shared/ai/clustering/feature_processor.py`
- `src/shared/ai/clustering/subfeature_processor.py`
- `src/shared/cli/parse.py`, `_runners.py`, `watch.py`, `feature_commands.py`, `glossary_commands.py`
- `src/shared/cli/benchmark_passport.py`

**Direct `elasticsearch` library imports (3 files):**
- `src/shared/db/repositories/base.py` — `from elasticsearch import Elasticsearch`
- `src/shared/db/repositories/metadata.py` — `from elasticsearch import NotFoundError`
- `src/shared/db/repositories/elements.py` — `from elasticsearch import NotFoundError` + `from elasticsearch.helpers import bulk`

These 3 files are the **only** places that touch the actual ES client library. The compatibility layer wraps these.

### Test File Renames

| Current | New |
|---------|-----|
| `tests/test_db_elasticsearch_unit.py` | `tests/test_db_search_unit.py` |
| `tests/test_glossary_elasticsearch.py` | `tests/test_glossary_search.py` |
| Test classes like `TestElasticsearchRepository` | `TestSearchRepository` |
| Test classes like `TestElasticsearchRepositoryInit` | `TestSearchRepositoryInit` |
| Test classes like `TestElasticsearchTextSearch` | `TestTextSearch` |
| Test classes like `TestElasticsearchEmbeddingStore` | `TestEmbeddingStore` |
| Test classes like `TestElasticsearchConfigDefaults` | `TestSearchConfigDefaults` |
| Test classes like `TestElasticsearchIndexing` | `TestIndexing` |

**Test files with `ElasticsearchRepository` imports/references (12 files):**
- `tests/test_db_elasticsearch_unit.py` — imports + class names
- `tests/test_glossary_elasticsearch.py` — imports + instantiation
- `tests/test_mcp_server.py` — mock patches + integration test
- `tests/test_web_dependencies.py` — mock patches
- `tests/test_web_routes_admin.py` — INDEX_NAME imports
- `tests/test_cli.py` — mock patches
- `tests/test_cli_glossary_ai.py` — mock patches
- `tests/test_incomplete_elements.py` — `ElasticsearchFileStateRepository` imports
- `tests/integration/test_cli_e2e.py` — imports + instantiation (9 references)
- `tests/db/conftest.py` — imports + fixture creation
- `tests/db/test_element_repository.py` — imports + class names + integration tests
- `tests/db/test_embedding_repository.py` — imports + `ElasticsearchEmbeddingStore`
- `tests/db/test_search.py` — INDEX_NAME import
- `tests/db/test_import_call_repository.py` — INDEX_NAME import

### Config File Updates

| File | Change |
|------|--------|
| `pyproject.toml` | `elasticsearch>=8.11.0` → `opensearch-py>=2.4.0` (keep ES as optional dep) |
| `config/magaldi.yaml` | `elasticsearch:` section → `search_backend:` |
| `tests/fixtures/config/valid.yaml` | Same |
| `tests/fixtures/config/minimal.yaml` | Same |
| `docker-compose.yml` | Full service replacement (see Docker section above) |
| `Makefile` | Update service targets |

### Documentation Updates

**Source-controlled docs (update references):**
- `README.md`
- `CLAUDE.md`
- `TODO.md`
- `.claude/skills/magaldi/SKILL.md`
- `.claude/skills/check-magaldi-integrity/SKILL.md`
- `src/magaldi_mcp/tools/config.py` (embedded SKILL.md content)

**Plan docs (leave as-is, they're historical):**
- `plans/*.md` — No need to rewrite historical design docs

### Interface-Driven Architecture

The goal is a **pluggable backend** — not just OpenSearch vs Elasticsearch, but any future storage engine (PostgreSQL/pgvector, Qdrant, Meilisearch, etc.). The interface defines *what* Magaldi needs from storage, not *how* any particular engine provides it.

#### Protocol/Interface Definition

```python
# src/shared/db/interfaces.py

from typing import Protocol, Any, runtime_checkable

@runtime_checkable
class SearchBackend(Protocol):
    """Core search backend interface. Any storage engine must implement this.

    Designed from actual usage in Magaldi's repository layer.
    Methods map to operations in elements.py, search.py, metadata.py, etc.
    """

    # --- Lifecycle ---
    def connect(self) -> None: ...
    def ensure_index(self, index_name: str, mapping: dict) -> None: ...
    def health_check(self) -> dict[str, Any]: ...

    # --- Element CRUD ---
    def index_document(self, index: str, doc_id: str, document: dict) -> bool: ...
    def get_document(self, index: str, doc_id: str, source_fields: list[str] | None = None) -> dict | None: ...
    def get_documents_batch(self, index: str, doc_ids: list[str], source_fields: list[str] | None = None) -> list[dict]: ...
    """Maps to current mget() usage — batch lookup with optional _source filtering."""

    def delete_by_query(self, index: str, query: dict, timeout: int | None = None) -> int: ...
    """Maps to current delete_by_query — used for file/repo cleanup."""

    def update_by_query(self, index: str, query: dict, script: str, params: dict, timeout: int | None = None) -> int: ...
    """Maps to current update_by_query with Painless scripts — used for file_hash/element_count updates."""

    def bulk_operations(self, index: str, operations: list[dict]) -> dict: ...
    """Maps to current bulk() — used for batch indexing relationships, external refs."""

    # --- Text Search ---
    def search_multi_match(self, index: str, query: str, fields: list[str],
                           filters: list[dict], size: int, fuzziness: str | None = None) -> list[dict]: ...
    """Maps to multi_match (BM25) in search_by_text() and search_by_keyword()."""

    def search_regexp(self, index: str, field: str, pattern: str,
                      filters: list[dict], size: int) -> list[dict]: ...
    """Maps to regexp query on raw_code.keyword in search_by_regexp()."""

    def search_wildcard(self, index: str, field: str, pattern: str,
                        filters: list[dict], size: int, case_insensitive: bool = True) -> list[dict]: ...
    """Maps to wildcard query on raw_code.keyword in search_by_wildcard()."""

    def search_proximity(self, index: str, field: str, terms: str, slop: int,
                         filters: list[dict], size: int) -> list[dict]: ...
    """Maps to match_phrase with slop in search_by_proximity()."""

    # --- Vector Search ---
    def search_vector(self, index: str, embedding: list[float], field: str,
                      filters: list[dict], size: int, min_score: float) -> list[dict]: ...
    """Maps to script_score+cosineSimilarity (ES) or native knn (OpenSearch)."""

    # --- Nested/Structured Queries ---
    def search_nested(self, index: str, path: str, nested_query: dict,
                      filters: list[dict], size: int) -> list[dict]: ...
    """Maps to nested queries for calls, imports, feature_memberships, etc."""

    # --- Composite Queries ---
    def search(self, index: str, body: dict) -> dict: ...
    """Escape hatch for complex queries not covered by specific methods.
    Returns raw response dict. Use sparingly — prefer specific methods."""

    def collapse_search(self, index: str, query: dict, collapse_field: str,
                        sort: list[dict], size: int) -> list[dict]: ...
    """Maps to field collapsing in relationships.py (multi-user priority)."""

    def count(self, index: str, query: dict) -> int: ...

    # --- Stats ---
    def index_stats(self, index: str) -> dict[str, Any]: ...
    """Maps to _count and index stats for repo statistics."""

    # --- Error Types ---
    @property
    def NotFoundError(self) -> type[Exception]: ...
    """Backend-specific not-found exception. Maps to elasticsearch.NotFoundError
    or opensearchpy.NotFoundError. Used in metadata.py and elements.py."""
```

**Note**: The `search()` escape hatch exists for complex queries that don't fit a clean abstraction (e.g., deeply nested bool queries with multiple should clauses). The goal is to minimize its usage over time as specific methods are added. New backends only need to implement the `search()` raw passthrough — the other methods provide convenience + backend-specific optimizations.

#### Directory Structure

```
src/shared/db/
├── interfaces.py               # Protocol definitions (SearchBackend, etc.)
├── factory.py                  # Backend factory: create_backend(config) → SearchBackend
├── store.py                    # High-level facades (Repository, EmbeddingStore, FileStateRepository)
├── backends/
│   ├── __init__.py             # Re-exports
│   ├── opensearch.py           # OpenSearch implementation (default)
│   └── elasticsearch.py        # ES implementation (optional extra)
├── repositories/
│   ├── __init__.py             # Repository facade (delegates to sub-repos)
│   ├── base.py                 # Shared constants, index mapping definition
│   ├── elements.py             # Element CRUD (uses SearchBackend, not raw client)
│   ├── metadata.py             # Embeddings, summaries, imports/calls
│   ├── search.py               # All search operations
│   ├── features.py             # Feature/subfeature operations
│   ├── glossary.py             # Glossary operations
│   ├── relationships.py        # Knowledge graph edges
│   └── stats.py                # Statistics
```

#### Key Design Decisions

1. **Repositories never import a client library directly.** They receive a `SearchBackend` instance and call interface methods. No `from opensearchpy import ...` anywhere except in `backends/opensearch.py`.

2. **Backend handles dialect differences.** The `search_vector()` interface is the same — OpenSearch backend uses native `knn`, ES backend uses `script_score`. The repository doesn't know or care.

3. **Factory pattern for instantiation:**
   ```python
   # src/shared/db/factory.py
   def create_backend(config: MagaldiConfig) -> SearchBackend:
       backend_type = config.search_backend.type  # "opensearch" | "elasticsearch"
       if backend_type == "opensearch":
           from shared.db.backends.opensearch import OpenSearchBackend
           return OpenSearchBackend(config)
       elif backend_type == "elasticsearch":
           from shared.db.backends.elasticsearch import ElasticsearchBackend
           return ElasticsearchBackend(config)
       else:
           raise ValueError(f"Unknown backend: {backend_type}")
   ```

4. **Config is backend-agnostic:**
   ```yaml
   search_backend:
     type: opensearch          # "opensearch" | "elasticsearch" | future: "pgvector", "qdrant"
     host: localhost
     port: 9200
     timeout: 30
     bulk_timeout: 300
   ```

5. **Optional extras in pyproject.toml:**
   ```toml
   [project.optional-dependencies]
   elasticsearch = ["elasticsearch>=8.11.0,<9.0.0"]
   # Default install gets opensearch
   ```

This way, adding a future backend (e.g., pgvector) means:
- Create `backends/pgvector.py` implementing `SearchBackend`
- Add `"pgvector"` case to factory
- Done. Zero changes to repositories, MCP tools, web routes, or CLI.

### Purge Verification Checklist

After purge, these should return **zero** results (excluding plans/docs):
- [ ] `grep -r "Elasticsearch" src/ --include="*.py"` → 0 hits
- [ ] `grep -r "from elasticsearch" src/ --include="*.py"` → 0 hits (only in `backends/elasticsearch.py`)
- [ ] `grep -r "elasticsearch" src/ --include="*.py" -l` → only `backends/elasticsearch.py`
- [ ] `grep -r "MAGALDI_ES_" src/ --include="*.py"` → 0 hits
- [ ] `grep -r "ElasticsearchRepository\|ElasticsearchBase\|ElasticsearchConfig" src/` → 0 hits
- [ ] No `elasticsearch` in default `docker-compose.yml`
- [ ] `pyproject.toml` lists `opensearch-py` as required, `elasticsearch` as optional extra

---

## Recommended Approach

### Phase 1: Backend Abstraction + OpenSearch Docker (2-3 days)
- Create `backends/` layer with abstract base
- Implement OpenSearch backend (primary)
- Keep Elasticsearch backend (compatibility)
- Update docker-compose with OpenSearch services (see above)
- Update Makefile targets
- Config: `search_backend.type` selector

### Phase 2: OpenSearch-Native Queries (2-3 days)
- Convert `dense_vector` → `knn_vector` with Faiss HNSW engine
- Replace `script_score` vector queries with native `knn` queries
- Validate all existing MCP tools work identically
- Run full test suite against OpenSearch

### Phase 3: Naming Purge (1-2 days)
- Rename all classes/files per table above
- Rewrite imports across 43 files
- Rename test files and test classes
- Update config schema (`elasticsearch:` → `search_backend:`)
- Update all documentation
- Run purge verification checklist

### Phase 4: Hybrid Search (3-5 days)
- Create search pipeline with normalization processor
- Implement hybrid queries combining BM25 + kNN
- Tune weights for different query types
- A/B test search quality

### Phase 5: Neural Features (5-10 days, optional)
- Deploy embedding model via ML Commons
- Set up ingest pipeline for auto-embedding
- Experiment with neural sparse for lightweight semantic search
- Implement cross-encoder reranking pipeline

---

## Conclusion

**Yes, the migration is beneficial.** The licensing alone justifies it for an open-source project. But the real prize is OpenSearch's native neural search stack — hybrid queries, search pipelines, ML Commons — which would transform Magaldi's search from "BM25 or vectors" to "intelligent multi-signal retrieval with automatic score fusion." The migration path is low-risk since Magaldi uses standard ES features that are fully compatible with OpenSearch.

The most exciting opportunity is **eliminating the separate embedding pipeline** by letting OpenSearch handle embedding at both ingest and query time, dramatically simplifying the architecture while improving search quality through native hybrid search.

## Sources

- [OpenSearch vs Elasticsearch: 2025 Comparison (Medium)](https://medium.com/@FrankGoortani/opensearch-vs-elasticsearch-a-comprehensive-comparison-in-2025-aff5a8533422)
- [OpenSearch vs Elasticsearch: What's Changed (Dattell)](https://dattell.com/data-architecture-blog/opensearch-vs-elasticsearch-in-2025-whats-changed-and-what-hasnt/)
- [OpenSearch vs Elasticsearch 2025 (BigData Boutique)](https://bigdataboutique.com/blog/elasticsearch-vs-opensearch-2025-update-5b5c81)
- [OpenSearch Hybrid Search Documentation](https://docs.opensearch.org/latest/vector-search/ai-search/hybrid-search/index/)
- [OpenSearch Neural Sparse Search](https://docs.opensearch.org/latest/vector-search/ai-search/neural-sparse-search/)
- [OpenSearch ML Commons Plugin](https://docs.opensearch.org/latest/ml-commons-plugin/)
- [OpenSearch k-NN Methods & Engines](https://docs.opensearch.org/latest/mappings/supported-field-types/knn-methods-engines/)
- [OpenSearch Normalization Processor](https://docs.opensearch.org/latest/search-plugins/search-pipelines/normalization-processor/)
- [OpenSearch Reranking Documentation](https://docs.opensearch.org/latest/search-plugins/search-relevance/reranking-search-results/)
- [OpenSearch Anomaly Detection](https://docs.opensearch.org/latest/observing-your-data/ad/index/)
- [OpenSearch SQL & PPL](https://docs.opensearch.org/latest/sql-and-ppl/)
- [Semantic Hybrid Search: OpenSearch vs ES (GigaSearch)](https://blog.gigasearch.co/opensearch-vs-elasticsearch-for-semantic-hybrid-search/)
- [opensearch-py Migration Guide](https://dev.to/laysauchoa/how-to-migrate-your-elasticsearch-client-to-using-opensearch-502p)
- [OpenSearch RRF for Hybrid Search](https://opensearch.org/blog/introducing-reciprocal-rank-fusion-hybrid-search/)
- [Vector Search: OpenSearch vs ES (Opster)](https://opster.com/guides/elasticsearch/machine-learning/vector-search-in-opensearch-vs-elasticsearch/)
