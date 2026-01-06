# Magaldi AI Processing - Phase 6: Embedding

## Overview

The Embedding phase generates vector representations of code elements using Ollama, enabling semantic search in Elasticsearch. Each element is embedded with hierarchical context (parent summaries) to capture architectural relationships, not just individual code semantics.

```
┌─────────────────────────────────────────────────────────────────┐
│                       PHASE 6: EMBEDDING                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  6.1              6.2              6.3              6.4          │
│  ─────            ─────            ─────            ─────        │
│  INITIALIZE   →   BUILD        →   GENERATE    →   STORE        │
│  WORKERS          CONTEXTS         VECTORS         VECTORS       │
│                                                                 │
│  • Worker pool    • File context   • Ollama embed  • ES update  │
│  • Ollama conn    • Class context  • Batch 10-20   • Bulk API   │
│  • Job claiming   • Enrich text    • Normalize     • Verify     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Key Design Decision: Hierarchical Context Enrichment

Unlike simple code embedding, Magaldi enriches each element with parent context:

```
Function embedding includes:
├── File path
├── File summary (from Phase 5)
├── Class summary (if method)
├── Function summary
└── Signature + docstring

This enables:
• Searching "authentication" finds all auth functions, even if individual summaries don't mention it
• Architectural awareness: "database service layer" finds functions in service classes
• Disambiguation: Two validate() functions get different vectors based on context
```

---

## Input

From Phase 5 (Summarization):

```python
# Jobs created in embedding_jobs table
# Elements have summaries populated

@dataclass
class EmbeddingJob:
    id: int
    element_id: str

    status: str                   # 'pending', 'running', 'completed', 'failed'

    worker_id: Optional[str]
    claimed_at: Optional[datetime]
    completed_at: Optional[datetime]
    retry_count: int
    error_message: Optional[str]

# Prerequisites:
# - code_elements.summary populated
# - code_elements.summary_status = 'completed'
```

---

## 6.1 Initialize Workers

### Purpose

Set up worker pool and Ollama embedding client.

### Configuration

```python
@dataclass
class EmbeddingConfig:
    # Ollama settings
    ollama_url: str = "http://ollama:11434"
    model: str = "snowflake-arctic-embed2"

    # Vector settings
    dimensions: int = 1024        # snowflake-arctic-embed2 outputs 1024 dims
    max_context: int = 8192       # Model context limit

    # Worker settings
    num_workers: int = 4
    batch_size: int = 20          # Elements per Ollama request

    # Retry settings
    max_retries: int = 3
    retry_delay: float = 2.0

    # Elasticsearch settings
    es_batch_size: int = 100      # Vectors per ES bulk request


class EmbeddingWorkerPool:
    """Manages workers for processing embedding jobs"""

    def __init__(self, config: EmbeddingConfig, db: MySQLConnection,
                 es: ElasticsearchClient):
        self.config = config
        self.db = db
        self.es = es
        self.ollama = OllamaEmbedClient(config.ollama_url, config.model)
        self.workers: List[EmbeddingWorker] = []
        self.shutdown_event = threading.Event()

    def start(self, num_workers: Optional[int] = None):
        """Start worker pool"""

        num = num_workers or self.config.num_workers

        # Verify Ollama embedding model
        if not self.ollama.verify_model():
            raise RuntimeError(f"Embedding model {self.config.model} not available")

        # Verify ES index has correct vector dimensions
        self.verify_es_mapping()

        for i in range(num):
            worker = EmbeddingWorker(
                worker_id=f"embed-{i}",
                config=self.config,
                db=self.db,
                es=self.es,
                ollama=self.ollama,
                shutdown_event=self.shutdown_event
            )
            worker.start()
            self.workers.append(worker)

        log.info(f"Started {num} embedding workers with model {self.config.model}")

    def verify_es_mapping(self):
        """Verify ES index has correct vector dimensions"""

        mapping = self.es.indices.get_mapping(index="magaldi_code_elements")
        props = mapping["magaldi_code_elements"]["mappings"]["properties"]

        if "embedding" not in props:
            raise RuntimeError("ES index missing 'embedding' field")

        dims = props["embedding"].get("dims")
        if dims != self.config.dimensions:
            raise RuntimeError(
                f"ES embedding dims ({dims}) != model dims ({self.config.dimensions})"
            )
```

### Ollama Embedding Client

```python
class OllamaEmbedClient:
    """Client for Ollama embedding API"""

    def __init__(self, url: str, model: str):
        self.url = url.rstrip('/')
        self.model = model
        self.session = requests.Session()

    def verify_model(self) -> bool:
        """Check if embedding model is available"""
        try:
            response = self.session.get(f"{self.url}/api/tags")
            models = response.json().get('models', [])
            return any(m['name'] == self.model for m in models)
        except Exception as e:
            log.error(f"Failed to connect to Ollama: {e}")
            return False

    def embed_single(self, text: str) -> List[float]:
        """Generate embedding for single text"""

        response = self.session.post(
            f"{self.url}/api/embed",
            json={"model": self.model, "input": text},
            timeout=30
        )
        response.raise_for_status()

        return response.json()['embeddings'][0]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for batch of texts"""

        response = self.session.post(
            f"{self.url}/api/embed",
            json={"model": self.model, "input": texts},
            timeout=60
        )
        response.raise_for_status()

        return response.json()['embeddings']
```

---

## 6.2 Build Contexts

### Purpose

Construct embedding text with hierarchical context enrichment.

### Context Building Strategy

```python
def build_embedding_text(element: CodeElement, db: MySQLConnection) -> str:
    """Build enriched text for embedding"""

    if element.element_type == 'file':
        return build_file_embedding_text(element)

    elif element.element_type == 'class':
        file_summary = get_file_summary(element, db)
        return build_class_embedding_text(element, file_summary)

    elif element.element_type in ('function', 'method'):
        file_summary = get_file_summary(element, db)
        class_summary = get_class_summary(element, db) if element.parent_id else None
        return build_function_embedding_text(element, file_summary, class_summary)

    else:
        return build_default_embedding_text(element)
```

### File Embedding Text

```python
def build_file_embedding_text(element: CodeElement) -> str:
    """Build embedding text for file elements"""

    parts = [
        f"File: {element.relative_path}",
        f"Language: {element.language}",
        f"Summary: {element.summary or 'No summary available'}",
    ]

    return "\n".join(parts)

# Example output:
# File: src/auth/login.py
# Language: python
# Summary: Handles user authentication flows including login, logout,
# and session management. Provides the primary entry points for credential
# validation and integrates with the token service for JWT generation.
```

### Class Embedding Text

```python
def build_class_embedding_text(element: CodeElement, file_summary: str) -> str:
    """Build embedding text for class elements"""

    parts = [
        f"File: {element.relative_path}",
        f"File context: {file_summary}",
        f"Class: {element.name}",
        f"Summary: {element.summary or 'No summary available'}",
    ]

    if element.docstring:
        parts.append(f"Docstring: {element.docstring}")

    return "\n".join(parts)

# Example output:
# File: src/auth/login.py
# File context: Handles user authentication flows including login...
# Class: AuthService
# Summary: Core service class responsible for user authentication operations.
# Manages credential verification against the user repository...
```

### Function/Method Embedding Text

```python
def build_function_embedding_text(element: CodeElement,
                                  file_summary: str,
                                  class_summary: Optional[str] = None) -> str:
    """Build embedding text for function/method elements"""

    parts = [
        f"File: {element.relative_path}",
        f"File context: {file_summary}",
    ]

    if class_summary:
        parts.append(f"Class context: {class_summary}")

    parts.append(f"Function: {element.name}")
    parts.append(f"Summary: {element.summary or 'No summary available'}")

    if element.signature:
        parts.append(f"Signature: {element.signature}")

    if element.docstring:
        # Truncate long docstrings
        docstring = element.docstring[:500]
        if len(element.docstring) > 500:
            docstring += "..."
        parts.append(f"Docstring: {docstring}")

    return "\n".join(parts)

# Example output (~400 tokens):
# File: src/auth/login.py
# File context: Handles user authentication flows including login, logout,
# and session management. Provides the primary entry points for credential
# validation and integrates with the token service for JWT generation.
# Implements rate limiting and failed attempt tracking for security.
#
# Class context: Core service class responsible for user authentication
# operations. Manages credential verification against the user repository,
# coordinates with the token service for session creation, and maintains
# audit logs for security compliance.
#
# Function: authenticate_user
# Summary: Validates provided username and password against stored credentials.
# On successful validation, generates a new JWT token with appropriate claims
# and expiration. Returns None if authentication fails.
# Signature: def authenticate_user(self, username: str, password: str) -> Optional[Token]
```

### Context Loading Helpers

```python
def get_file_summary(element: CodeElement, db: MySQLConnection) -> str:
    """Get file summary for context enrichment"""

    result = db.query_one("""
        SELECT summary FROM code_elements
        WHERE scope = %s
          AND repository = %s
          AND username = %s
          AND relative_path = %s
          AND element_type = 'file'
    """, (element.scope, element.repository, element.username, element.relative_path))

    return result.summary if result and result.summary else "No file summary available."


def get_class_summary(element: CodeElement, db: MySQLConnection) -> Optional[str]:
    """Get parent class summary for method context"""

    if not element.parent_id:
        return None

    result = db.query_one("""
        SELECT summary, element_type FROM code_elements
        WHERE element_id = %s
    """, (element.parent_id,))

    if result and result.element_type == 'class' and result.summary:
        return result.summary

    return None
```

### Token Estimation

```python
def estimate_tokens(text: str) -> int:
    """Estimate token count (rough approximation)"""
    # Average: 1 token ~= 4 characters for English text
    # Code tends to be slightly more tokens per character
    return len(text) // 3


def validate_context_length(text: str, max_tokens: int = 8000) -> str:
    """Ensure text fits within model context"""

    estimated = estimate_tokens(text)

    if estimated <= max_tokens:
        return text

    # Truncate from the middle sections (preserve file context and summary)
    lines = text.split('\n')
    truncated = []
    token_count = 0

    for line in lines:
        line_tokens = estimate_tokens(line)
        if token_count + line_tokens <= max_tokens * 0.9:  # 10% buffer
            truncated.append(line)
            token_count += line_tokens
        else:
            truncated.append("... (context truncated)")
            break

    return '\n'.join(truncated)
```

---

## 6.3 Generate Vectors

### Purpose

Generate embedding vectors using Ollama in batches.

### Batch Processing

```python
class EmbeddingWorker(threading.Thread):
    """Worker thread for processing embedding jobs"""

    def __init__(self, worker_id: str, config: EmbeddingConfig,
                 db: MySQLConnection, es: ElasticsearchClient,
                 ollama: OllamaEmbedClient, shutdown_event: threading.Event):
        super().__init__(daemon=True)
        self.worker_id = worker_id
        self.config = config
        self.db = db
        self.es = es
        self.ollama = ollama
        self.shutdown_event = shutdown_event
        self.pending_vectors = []  # Buffer for ES bulk insert

    def run(self):
        """Main worker loop"""

        while not self.shutdown_event.is_set():
            jobs = self.claim_jobs()

            if not jobs:
                # Flush any pending vectors before sleeping
                self.flush_pending_vectors()
                time.sleep(1.0)
                continue

            self.process_batch(jobs)

        # Final flush on shutdown
        self.flush_pending_vectors()

    def claim_jobs(self) -> List[EmbeddingJob]:
        """Claim batch of pending jobs"""

        self.db.begin_transaction()

        try:
            jobs = self.db.query("""
                SELECT * FROM embedding_jobs
                WHERE status = 'pending'
                ORDER BY id
                LIMIT %s
                FOR UPDATE SKIP LOCKED
            """, (self.config.batch_size,))

            if not jobs:
                self.db.rollback()
                return []

            job_ids = [j.id for j in jobs]
            self.db.execute("""
                UPDATE embedding_jobs
                SET status = 'running',
                    worker_id = %s,
                    claimed_at = NOW()
                WHERE id IN (%s)
            """, (self.worker_id, ','.join(map(str, job_ids))))

            self.db.commit()
            return jobs

        except Exception as e:
            self.db.rollback()
            log.error(f"Failed to claim jobs: {e}")
            return []

    def process_batch(self, jobs: List[EmbeddingJob]):
        """Process batch of embedding jobs"""

        # 1. Load elements and build contexts
        elements_and_texts = []
        for job in jobs:
            try:
                element = load_element(job.element_id, self.db)
                if element:
                    text = build_embedding_text(element, self.db)
                    text = validate_context_length(text, self.config.max_context)
                    elements_and_texts.append((job, element, text))
                else:
                    self.mark_job_failed(job.id, "Element not found")
            except Exception as e:
                self.mark_job_failed(job.id, str(e))

        if not elements_and_texts:
            return

        # 2. Generate embeddings in batch
        texts = [t[2] for t in elements_and_texts]

        try:
            embeddings = self.generate_embeddings_with_retry(texts)
        except Exception as e:
            # Mark all jobs as failed
            for job, _, _ in elements_and_texts:
                self.mark_job_failed(job.id, str(e))
            return

        # 3. Store results
        for (job, element, _), embedding in zip(elements_and_texts, embeddings):
            self.store_embedding(job, element, embedding)


def generate_embeddings_with_retry(self, texts: List[str]) -> List[List[float]]:
    """Generate embeddings with retry logic"""

    last_error = None

    for attempt in range(self.config.max_retries):
        try:
            embeddings = self.ollama.embed_batch(texts)

            # Validate dimensions
            for emb in embeddings:
                if len(emb) != self.config.dimensions:
                    raise ValueError(
                        f"Wrong dimensions: expected {self.config.dimensions}, "
                        f"got {len(emb)}"
                    )

            return embeddings

        except Exception as e:
            last_error = e
            log.warning(f"Embedding error (attempt {attempt + 1}): {e}")

            if attempt < self.config.max_retries - 1:
                time.sleep(self.config.retry_delay * (2 ** attempt))

    raise last_error
```

### Vector Normalization

```python
def normalize_vector(vector: List[float]) -> List[float]:
    """Normalize vector to unit length for cosine similarity"""

    import math

    magnitude = math.sqrt(sum(x * x for x in vector))

    if magnitude == 0:
        return vector

    return [x / magnitude for x in vector]


def validate_vector(vector: List[float], expected_dims: int) -> bool:
    """Validate vector dimensions and values"""

    if len(vector) != expected_dims:
        return False

    # Check for NaN or Inf values
    for v in vector:
        if math.isnan(v) or math.isinf(v):
            return False

    return True
```

---

## 6.4 Store Vectors

### Purpose

Persist embedding vectors to MySQL and Elasticsearch.

### Store to MySQL

```python
def store_embedding(self, job: EmbeddingJob, element: CodeElement,
                   embedding: List[float]):
    """Store embedding and update statuses"""

    # 1. Update MySQL element status
    self.db.execute("""
        UPDATE code_elements
        SET embedding_status = 'completed'
        WHERE element_id = %s
    """, (element.element_id,))

    # 2. Mark job completed
    self.db.execute("""
        UPDATE embedding_jobs
        SET status = 'completed',
            completed_at = NOW()
        WHERE id = %s
    """, (job.id,))

    # 3. Add to pending ES updates
    self.pending_vectors.append((element.element_id, embedding))

    # 4. Flush if buffer is full
    if len(self.pending_vectors) >= self.config.es_batch_size:
        self.flush_pending_vectors()
```

### Bulk Update Elasticsearch

```python
def flush_pending_vectors(self):
    """Bulk update pending vectors to Elasticsearch"""

    if not self.pending_vectors:
        return

    actions = []
    for element_id, embedding in self.pending_vectors:
        actions.append({
            "_op_type": "update",
            "_index": "magaldi_code_elements",
            "_id": element_id,
            "doc": {
                "embedding": embedding
            }
        })

    try:
        success, errors = helpers.bulk(
            self.es,
            actions,
            raise_on_error=False,
            stats_only=False
        )

        if errors:
            log.warning(f"ES bulk errors: {len(errors)}")
            for error in errors[:5]:
                log.warning(f"  {error}")

        log.debug(f"Flushed {success} vectors to ES")

    except Exception as e:
        log.error(f"ES bulk update failed: {e}")
        # Vectors are in MySQL status; ES can be rebuilt

    finally:
        self.pending_vectors = []


def mark_job_failed(self, job_id: int, error_message: str):
    """Mark embedding job as failed"""

    self.db.execute("""
        UPDATE embedding_jobs
        SET status = 'failed',
            error_message = %s,
            completed_at = NOW()
        WHERE id = %s
    """, (error_message, job_id))

    # Update element status
    self.db.execute("""
        UPDATE code_elements ce
        JOIN embedding_jobs ej ON ce.element_id = ej.element_id
        SET ce.embedding_status = 'failed'
        WHERE ej.id = %s
    """, (job_id,))
```

### Verify Embeddings

```python
def verify_embeddings(scope: str, repository: str, username: str,
                     db: MySQLConnection, es: ElasticsearchClient) -> Dict:
    """Verify embedding completeness and consistency"""

    # Check MySQL status
    mysql_stats = db.query_one("""
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN embedding_status = 'completed' THEN 1 ELSE 0 END) as completed,
            SUM(CASE WHEN embedding_status = 'failed' THEN 1 ELSE 0 END) as failed
        FROM code_elements
        WHERE scope = %s AND repository = %s AND username = %s
          AND element_type IN ('file', 'class', 'function', 'method')
    """, (scope, repository, username))

    # Check ES has embeddings
    es_result = es.count(
        index="magaldi_code_elements",
        body={
            "query": {
                "bool": {
                    "must": [
                        {"term": {"scope": scope}},
                        {"term": {"repository": repository}},
                        {"term": {"username": username}},
                        {"exists": {"field": "embedding"}}
                    ]
                }
            }
        }
    )

    return {
        'mysql_total': mysql_stats.total,
        'mysql_completed': mysql_stats.completed,
        'mysql_failed': mysql_stats.failed,
        'es_with_embedding': es_result['count'],
        'sync_ok': mysql_stats.completed == es_result['count']
    }
```

---

## Output

Phase 6 produces:

```python
# MySQL: code_elements.embedding_status = 'completed'
# MySQL: embedding_jobs.status = 'completed'
# Elasticsearch: embedding field populated (1024-dim vector)

@dataclass
class EmbeddingResult:
    scope: str
    repository: str
    username: str

    # Counts
    total_elements: int
    embedded_elements: int
    failed_elements: int

    # Performance
    start_time: datetime
    end_time: datetime
    avg_embedding_time_ms: float
    vectors_per_second: float

    # Verification
    es_sync_verified: bool

    # Errors
    errors: List[EmbeddingError]
```

---

## Semantic Search

With embeddings in place, Elasticsearch supports semantic search:

### Search Query

```python
def semantic_search(query: str, scope: str, repository: str,
                   username: str, ollama: OllamaEmbedClient,
                   es: ElasticsearchClient, limit: int = 10) -> List[SearchResult]:
    """Perform semantic search using query embedding"""

    # 1. Embed the search query
    query_embedding = ollama.embed_single(query)

    # 2. Search ES with kNN
    results = es.search(
        index="magaldi_code_elements",
        body={
            "size": limit,
            "query": {
                "bool": {
                    "must": [
                        {
                            "script_score": {
                                "query": {
                                    "bool": {
                                        "filter": [
                                            {"term": {"scope": scope}},
                                            {"term": {"repository": repository}},
                                            # Include main + user's overlays
                                            {"terms": {"username": ["main", username]}}
                                        ]
                                    }
                                },
                                "script": {
                                    "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                                    "params": {"query_vector": query_embedding}
                                }
                            }
                        }
                    ]
                }
            },
            "_source": ["element_id", "name", "element_type", "relative_path",
                       "summary", "signature", "line_start"]
        }
    )

    return [
        SearchResult(
            element_id=hit['_source']['element_id'],
            name=hit['_source']['name'],
            element_type=hit['_source']['element_type'],
            relative_path=hit['_source']['relative_path'],
            summary=hit['_source'].get('summary'),
            signature=hit['_source'].get('signature'),
            line_start=hit['_source'].get('line_start'),
            score=hit['_score']
        )
        for hit in results['hits']['hits']
    ]
```

### Hybrid Search (Semantic + Keyword)

```python
def hybrid_search(query: str, scope: str, repository: str,
                 username: str, ollama: OllamaEmbedClient,
                 es: ElasticsearchClient, limit: int = 10) -> List[SearchResult]:
    """Combine semantic and keyword search"""

    query_embedding = ollama.embed_single(query)

    results = es.search(
        index="magaldi_code_elements",
        body={
            "size": limit,
            "query": {
                "bool": {
                    "must": [
                        {"terms": {"username": ["main", username]}},
                        {"term": {"scope": scope}},
                        {"term": {"repository": repository}}
                    ],
                    "should": [
                        # Semantic similarity (boosted)
                        {
                            "script_score": {
                                "query": {"match_all": {}},
                                "script": {
                                    "source": "(cosineSimilarity(params.query_vector, 'embedding') + 1.0) * 2",
                                    "params": {"query_vector": query_embedding}
                                }
                            }
                        },
                        # Keyword match on name
                        {"match": {"name": {"query": query, "boost": 1.5}}},
                        # Keyword match on summary
                        {"match": {"summary": {"query": query, "boost": 1.0}}},
                        # Keyword match on docstring
                        {"match": {"docstring": {"query": query, "boost": 0.5}}}
                    ]
                }
            }
        }
    )

    return parse_search_results(results)
```

---

## Progress Reporting

```
[Embedding]
Initializing workers...               4 workers, model: snowflake-arctic-embed2
Verifying ES mapping...               ✓ 1024 dimensions

Processing repository...              backend:auth-service:main
Building contexts...                  59 elements
Generating embeddings...              59/59 (100%)

Batch 1:                              20/20 vectors (0.8s)
Batch 2:                              20/20 vectors (0.7s)
Batch 3:                              19/19 vectors (0.6s)

Flushing to Elasticsearch...          59 vectors indexed

Summary:
  Total elements:     59
  Embedded:           59
  Failed:             0
  Avg time:           35ms/element
  Throughput:         28 vectors/sec
  ES sync:            ✓ verified
```

---

## Error Handling

| Error | Action |
|-------|--------|
| Ollama unavailable | Retry 3x with backoff, then fail job |
| Model not loaded | Attempt model pull, retry |
| Context too long | Truncate, regenerate |
| Wrong dimensions | Verify model, fail if mismatch |
| ES bulk partial failure | Log failures, continue |
| ES unavailable | Log warning, mark for reindex |
| NaN/Inf in vector | Skip element, log error |
| Worker crash | Reclaim stale jobs |

### Reindex Failed Elements

```python
def reindex_failed_embeddings(scope: str, repository: str, username: str,
                             db: MySQLConnection):
    """Reset failed embedding jobs for retry"""

    db.execute("""
        UPDATE embedding_jobs ej
        JOIN code_elements ce ON ej.element_id = ce.element_id
        SET ej.status = 'pending',
            ej.retry_count = 0,
            ej.error_message = NULL,
            ce.embedding_status = 'pending'
        WHERE ej.status = 'failed'
          AND ce.scope = %s
          AND ce.repository = %s
          AND ce.username = %s
    """, (scope, repository, username))

    log.info(f"Reset {db.rowcount} failed embedding jobs")
```

### Rebuild ES Embeddings

```python
def rebuild_es_embeddings(scope: str, repository: str, username: str,
                         db: MySQLConnection, es: ElasticsearchClient):
    """Rebuild ES embeddings from MySQL (disaster recovery)"""

    # Note: MySQL stores embedding_status, not the actual vectors
    # Vectors only live in ES, so this regenerates them

    log.warning("ES embedding rebuild requires re-running embedding phase")

    # Reset all embedding jobs to pending
    db.execute("""
        UPDATE embedding_jobs ej
        JOIN code_elements ce ON ej.element_id = ce.element_id
        SET ej.status = 'pending',
            ej.retry_count = 0,
            ce.embedding_status = 'pending'
        WHERE ce.scope = %s
          AND ce.repository = %s
          AND ce.username = %s
          AND ce.summary_status = 'completed'
    """, (scope, repository, username))

    log.info(f"Reset {db.rowcount} embedding jobs for regeneration")
```

---

## Performance Considerations

| Operation | Bottleneck | Optimization |
|-----------|------------|--------------|
| Ollama embedding | CPU/GPU | Batch 10-20 texts per request |
| Context building | Database | Cache file/class summaries |
| ES bulk insert | Network | Batch 100 vectors |
| Memory | Vector size | 1024 dims * 4 bytes = 4KB/vector |

### Throughput Estimates

| Hardware | Model | Vectors/sec | Notes |
|----------|-------|-------------|-------|
| CPU only | snowflake-arctic-embed2 | ~10-20 | Development |
| RTX 3060 | snowflake-arctic-embed2 | ~30-50 | 4 workers |
| RTX 3080 | snowflake-arctic-embed2 | ~50-80 | 4 workers |

### Memory Requirements

```
Embedding model:     ~1.2 GB VRAM/RAM
Vector buffer:       100 vectors * 4KB = 400KB (negligible)
Context cache:       ~10MB for file/class summaries
Total overhead:      ~1.5 GB
```

### Storage Estimates

```
Per element (ES):    ~4KB (1024 dims * 4 bytes)
10,000 elements:     ~40 MB
100,000 elements:    ~400 MB
1,000,000 elements:  ~4 GB
```

---

## CLI Interface

```bash
# Start embedding for a repository
magaldi embed --scope backend --repo auth-service --user main

# Resume failed jobs
magaldi embed --scope backend --repo auth-service --user main --retry-failed

# Verify embeddings
magaldi embed --scope backend --repo auth-service --user main --verify

# Monitor progress
magaldi embed --scope backend --repo auth-service --user main --watch

# Force regeneration
magaldi embed --scope backend --repo auth-service --user main --force
```

---

## Summary of Decisions

| Decision | Value |
|----------|-------|
| Embedding model | snowflake-arctic-embed2 (Ollama) |
| Vector dimensions | 1024 |
| Max context | 8192 tokens |
| Context strategy | Hierarchical enrichment (file + class + element) |
| Batch size (Ollama) | 20 texts per request |
| Batch size (ES) | 100 vectors per bulk request |
| Worker model | Thread pool (4 workers default) |
| Job claiming | SKIP LOCKED (non-blocking) |
| Retry strategy | 3 attempts, exponential backoff |
| Vector normalization | Yes (for cosine similarity) |
| Storage | ES only (MySQL stores status) |
| Search strategy | Hybrid (semantic + keyword) |
| User overlay | Search main + user vectors together |
