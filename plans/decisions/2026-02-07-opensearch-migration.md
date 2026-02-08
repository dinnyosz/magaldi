# Decision: Migrate from Elasticsearch to OpenSearch

**Date:** 2026-02-07

## Decision: Replace Elasticsearch with OpenSearch as primary search backend

**Original plan:** Continue using Elasticsearch 8.11.0 as the sole search/storage engine for Magaldi.

**Deviation:** Migrate to OpenSearch 2.19.0 as the primary backend, with Elasticsearch kept as an optional compatibility backend behind an interface.

**Why:** Three converging reasons:
1. **Licensing** — Elasticsearch's SSPL license creates legal uncertainty for users deploying Magaldi as a service. OpenSearch's Apache 2.0 license aligns with Magaldi being an open-source project.
2. **Native hybrid search** — Magaldi currently does BM25 OR vector search, never both together. OpenSearch's native `hybrid` query type with normalization pipelines and RRF (Reciprocal Rank Fusion) enables combining keyword + semantic search in a single query.
3. **ML Commons / Neural features** — OpenSearch can host embedding models directly, potentially eliminating the separate Ollama embedding pipeline. Neural sparse search offers semantic search without dense vector overhead.

**Options considered:**
1. **Stay on Elasticsearch** — No migration effort, but stuck with SSPL licensing, no native hybrid search, and separate embedding pipeline. Performance is ~40-140% faster in raw benchmarks, but irrelevant for Magaldi's scale (<1M documents, single-user).
2. **Move to OpenSearch (chosen)** — Low migration risk (API-compatible for all features Magaldi uses), gains hybrid search + ML Commons + neural sparse + Apache 2.0 licensing. Moderate effort (~2-3 weeks total across all phases).
3. **Move to a dedicated vector DB (Qdrant, Milvus)** — Better pure-vector performance, but would lose the text search, nested queries, regexp/wildcard, and aggregation capabilities Magaldi heavily relies on. Would require maintaining two separate storage systems.
4. **Move to PostgreSQL + pgvector** — Simpler infrastructure (one DB for everything), but significantly weaker text search (no BM25 scoring like ES/OS), no native hybrid search pipelines, and would require rewriting all query logic.

**Final decision:** OpenSearch — it's the lowest-risk migration that unlocks the most new capabilities. The API surface Magaldi uses (bool queries, multi_match, nested, regexp, wildcard, bulk, collapse) is fully compatible. The real value is in the new features: hybrid search, neural sparse, ML Commons, and search pipelines with reranking. The interface-driven architecture (see companion decision) ensures we're not locked in.

**Impact:** Full analysis in `plans/opensearch_migration_analysis.md`. 5-phase implementation plan covering backend abstraction, OpenSearch-native queries, naming purge, hybrid search, and neural features.
