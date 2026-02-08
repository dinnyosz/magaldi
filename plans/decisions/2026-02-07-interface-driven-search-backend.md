# Decision: Interface-Driven Search Backend Architecture

**Date:** 2026-02-07

## Decision: Implement SearchBackend Protocol for pluggable storage backends

**Original plan:** The initial migration analysis proposed a simple compatibility layer — swap `elasticsearch-py` for `opensearch-py` and adapt the few query differences.

**Deviation:** Instead of a thin shim, implement a full Protocol-based interface (`SearchBackend`) that all repositories program against. No repository or tool file ever imports a client library directly. Backend selection happens via factory pattern + config.

**Why:** The user indicated that OpenSearch may not be the final destination — the project might move to a different database in the future. A proper interface ensures that adding a new backend (pgvector, Qdrant, Meilisearch, or anything else) requires only:
1. One new file in `backends/`
2. One new case in the factory
3. Zero changes to repositories, MCP tools, web routes, or CLI

**Options considered:**
1. **Direct client swap (no abstraction)** — Replace `elasticsearch-py` with `opensearch-py`, rename classes. Minimal effort but creates the same tight coupling with a different vendor. Future migration would require touching 43+ files again.
2. **Thin compatibility shim** — Wrap both clients behind a minimal adapter that normalizes API differences (e.g., `dense_vector` vs `knn_vector`). Quick to implement but doesn't abstract query construction — repositories still build ES/OS-specific query DSL.
3. **Full Protocol interface (chosen)** — Define `SearchBackend` Protocol with method signatures for all operations Magaldi needs (element CRUD, text search, vector search, nested queries, bulk ops). Repositories call interface methods; backends translate to engine-specific queries. More upfront work but genuinely pluggable.
4. **ORM/query builder layer** — Use something like SQLAlchemy-style query building that compiles to different backends. Over-engineered for current needs, introduces heavy dependency, and search engine query languages don't map well to SQL-style abstractions.

**Final decision:** Full Protocol interface — the upfront cost is moderate (the repositories already go through `ElasticsearchBase._get_client()`, so the abstraction boundary largely exists). The payoff is real decoupling. Only 3 files currently import the `elasticsearch` library directly (`base.py`, `metadata.py`, `elements.py`) — these become the backend implementations. The other 40+ files just need import path changes.

**Architecture:**
- `src/shared/db/interfaces.py` — `SearchBackend` Protocol definition
- `src/shared/db/factory.py` — `create_backend(config)` factory
- `src/shared/db/backends/opensearch.py` — OpenSearch implementation (default)
- `src/shared/db/backends/elasticsearch.py` — ES implementation (optional)
- Config: `search_backend.type: "opensearch" | "elasticsearch" | future values`

**Impact:** All `Elasticsearch*` naming purged from codebase. 43 source files get import rewrites. Test files renamed. Config key changes from `elasticsearch:` to `search_backend:`. Full purge checklist in `plans/opensearch_migration_analysis.md`.
