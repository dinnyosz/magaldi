# Magaldi Parser - Phase 4: Storage

## Overview

The Storage phase persists parsed elements to MySQL and Elasticsearch, handles deletions, and creates jobs for AI processing (summarization and embedding).

```
┌─────────────────────────────────────────────────────────────────┐
│                       PHASE 4: STORAGE                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  4.1              4.2              4.3              4.4          │
│  ─────            ─────            ─────            ─────        │
│  HANDLE       →   STORE        →   INDEX       →   CREATE       │
│  DELETIONS        MYSQL            ELASTICSEARCH   JOBS          │
│                                                                 │
│  • Remove old     • Upsert files   • Bulk index    • Summarize  │
│  • Clean ES       • Upsert elems   • No embeddings • Embed      │
│  • Update states  • Transactions     yet           • By level   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Input

From Phase 3 (Parsing):

```python
@dataclass
class ParsingResult:
    scope: str
    repository: str
    username: str

    parsed_files: List[ParsedFile]
    failed_files: List[FailedFile]

# Plus from Phase 2:
@dataclass
class ChangeManifest:
    deleted_files: List[FileInfo]
```

---

## 4.1 Handle Deletions

### Purpose

Remove elements from files that were deleted or modified (before inserting new versions).

### Deletion Scenarios

| Scenario | Action |
|----------|--------|
| File deleted | Remove all elements for that file |
| File modified | Remove old elements, then insert new |
| User branch file same as main | No action (skip) |

### Process

```python
def handle_deletions(
    manifest: ChangeManifest,
    db: MySQLConnection,
    es: ElasticsearchClient
):
    """Handle all deletion scenarios"""

    # 1. Files explicitly deleted
    for file_info in manifest.deleted_files:
        delete_file_elements(file_info, db, es)
        mark_file_deleted(file_info, db)

    # 2. Modified files (remove old elements before inserting new)
    for file_info in manifest.modified_files:
        delete_file_elements(file_info, db, es)
        # File state will be updated when new elements are stored
```

### Delete File Elements

```python
def delete_file_elements(file_info: FileInfo, db: MySQLConnection, es: ElasticsearchClient):
    """Remove all elements for a file"""

    # Build element ID prefix for this file
    id_prefix = f"{file_info.scope}:{file_info.repository}:{file_info.username}:{file_info.relative_path}:"

    # 1. Delete from Elasticsearch
    es.delete_by_query(
        index="magaldi_code_elements",
        body={
            "query": {
                "prefix": {
                    "element_id": id_prefix
                }
            }
        }
    )

    # 2. Delete from MySQL code_elements
    db.execute("""
        DELETE FROM code_elements
        WHERE scope = %s
          AND repository = %s
          AND username = %s
          AND relative_path = %s
    """, (file_info.scope, file_info.repository, file_info.username, file_info.relative_path))

    # 3. Delete any pending jobs for these elements
    db.execute("""
        DELETE FROM summarization_jobs
        WHERE element_id LIKE %s
    """, (id_prefix + '%',))

    db.execute("""
        DELETE FROM embedding_jobs
        WHERE element_id LIKE %s
    """, (id_prefix + '%',))
```

### Mark File Deleted (User Branches)

```python
def mark_file_deleted(file_info: FileInfo, db: MySQLConnection):
    """Mark file as deleted in file_states (for user branches)"""

    if file_info.username == 'main':
        # Main branch: actually delete the record
        db.execute("""
            DELETE FROM file_states
            WHERE scope = %s
              AND repository = %s
              AND username = 'main'
              AND relative_path = %s
        """, (file_info.scope, file_info.repository, file_info.relative_path))
    else:
        # User branch: mark as deleted (overlay on main)
        db.execute("""
            INSERT INTO file_states (scope, repository, username, relative_path, is_deleted, parsed_at, expires_at)
            VALUES (%s, %s, %s, %s, TRUE, NOW(), DATE_ADD(NOW(), INTERVAL 30 DAY))
            ON DUPLICATE KEY UPDATE
                is_deleted = TRUE,
                file_hash = NULL,
                parsed_at = NOW(),
                expires_at = DATE_ADD(NOW(), INTERVAL 30 DAY)
        """, (file_info.scope, file_info.repository, file_info.username, file_info.relative_path))
```

---

## 4.2 Store to MySQL

### Purpose

Persist file states and code elements to MySQL for metadata storage and job tracking.

### Transaction Strategy

```python
def store_parsed_files(
    parsed_files: List[ParsedFile],
    username: str,
    db: MySQLConnection
):
    """Store all parsed files in a transaction"""

    try:
        db.begin_transaction()

        for parsed_file in parsed_files:
            # 1. Update file state
            store_file_state(parsed_file, username, db)

            # 2. Store elements
            store_elements(parsed_file.elements, db)

        db.commit()

    except Exception as e:
        db.rollback()
        raise StorageError(f"Failed to store parsed files: {e}")
```

### Store File State

```python
def store_file_state(parsed_file: ParsedFile, username: str, db: MySQLConnection):
    """Insert or update file state"""

    file_info = parsed_file.file_info
    expires_at = None if username == 'main' else 'DATE_ADD(NOW(), INTERVAL 30 DAY)'

    db.execute("""
        INSERT INTO file_states (
            scope, repository, username, relative_path,
            file_hash, language, file_size, line_count,
            is_deleted, parsed_at, expires_at
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, FALSE, NOW(), """ + (expires_at or 'NULL') + """)
        ON DUPLICATE KEY UPDATE
            file_hash = VALUES(file_hash),
            language = VALUES(language),
            file_size = VALUES(file_size),
            line_count = VALUES(line_count),
            is_deleted = FALSE,
            parsed_at = NOW(),
            expires_at = """ + (expires_at or 'NULL') + """
    """, (
        file_info.scope,
        file_info.repository,
        username,
        file_info.relative_path,
        parsed_file.file_hash,
        file_info.language,
        get_file_size(file_info.absolute_path),
        parsed_file.line_count,
    ))
```

### Store Elements

```python
def store_elements(elements: List[RawElement], db: MySQLConnection):
    """Batch insert elements"""

    if not elements:
        return

    # Prepare batch insert
    values = []
    for elem in elements:
        values.append((
            elem.element_id,
            elem.scope,
            elem.repository,
            elem.username,
            elem.relative_path,
            elem.element_type,
            elem.name,
            elem.line_start,
            elem.line_end,
            elem.raw_code,
            elem.signature,
            elem.docstring,
            elem.level,
            elem.parent_id,
            elem.language,
            json.dumps(elem.decorators),
            elem.visibility,
            elem.is_async,
            'pending',  # summary_status
            'pending',  # embedding_status
            None if elem.username == 'main' else 'DATE_ADD(NOW(), INTERVAL 30 DAY)',
        ))

    # Batch insert with ON DUPLICATE KEY UPDATE
    db.executemany("""
        INSERT INTO code_elements (
            element_id, scope, repository, username, relative_path,
            element_type, name, line_start, line_end,
            raw_code, signature, docstring,
            level, parent_id, language, decorators, visibility, is_async,
            summary_status, embedding_status, expires_at
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            raw_code = VALUES(raw_code),
            signature = VALUES(signature),
            docstring = VALUES(docstring),
            level = VALUES(level),
            parent_id = VALUES(parent_id),
            decorators = VALUES(decorators),
            visibility = VALUES(visibility),
            is_async = VALUES(is_async),
            summary_status = 'pending',
            embedding_status = 'pending',
            summary = NULL
    """, values)
```

### Update Repository Metadata

```python
def update_repository_metadata(
    scope: str,
    repository: str,
    parsed_files: List[ParsedFile],
    db: MySQLConnection
):
    """Update repository-level statistics"""

    # Aggregate stats
    languages = {}
    total_files = 0
    total_lines = 0
    total_elements = 0

    for pf in parsed_files:
        lang = pf.file_info.language
        languages[lang] = languages.get(lang, {'files': 0, 'lines': 0})
        languages[lang]['files'] += 1
        languages[lang]['lines'] += pf.line_count
        total_files += 1
        total_lines += pf.line_count
        total_elements += len(pf.elements)

    # Upsert repository
    db.execute("""
        INSERT INTO repositories (scope, name, created_at)
        VALUES (%s, %s, NOW())
        ON DUPLICATE KEY UPDATE
            scope = VALUES(scope)
    """, (scope, repository))

    # Update language stats
    db.execute("""
        DELETE FROM repository_languages
        WHERE scope = %s AND repository = %s
    """, (scope, repository))

    for lang, stats in languages.items():
        db.execute("""
            INSERT INTO repository_languages (scope, repository, language, file_count, line_count)
            VALUES (%s, %s, %s, %s, %s)
        """, (scope, repository, lang, stats['files'], stats['lines']))
```

---

## 4.3 Index to Elasticsearch

### Purpose

Index elements to Elasticsearch for semantic search. Embeddings are added later by the embedding phase.

### Index Mapping

```python
MAGALDI_INDEX_MAPPING = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
        "analysis": {
            "analyzer": {
                "code_analyzer": {
                    "type": "custom",
                    "tokenizer": "standard",
                    "filter": ["lowercase", "code_synonyms"]
                }
            },
            "filter": {
                "code_synonyms": {
                    "type": "synonym",
                    "synonyms": [
                        "func,function,method,def",
                        "cls,class,type",
                        "var,variable,const,constant",
                        "auth,authentication,login",
                        "db,database,storage"
                    ]
                }
            }
        }
    },
    "mappings": {
        "properties": {
            # Identity
            "element_id": {"type": "keyword"},
            "scope": {"type": "keyword"},
            "repository": {"type": "keyword"},
            "username": {"type": "keyword"},
            "relative_path": {"type": "keyword"},

            # Element info
            "element_type": {"type": "keyword"},
            "name": {"type": "text", "analyzer": "code_analyzer", "fields": {"raw": {"type": "keyword"}}},
            "language": {"type": "keyword"},

            # Position
            "line_start": {"type": "integer"},
            "line_end": {"type": "integer"},

            # Content (searchable)
            "raw_code": {"type": "text", "analyzer": "code_analyzer"},
            "signature": {"type": "text", "analyzer": "code_analyzer"},
            "docstring": {"type": "text"},
            "summary": {"type": "text"},  # Added by summarization phase

            # Hierarchy
            "level": {"type": "integer"},
            "parent_id": {"type": "keyword"},

            # Metadata
            "decorators": {"type": "keyword"},
            "visibility": {"type": "keyword"},
            "is_async": {"type": "boolean"},

            # Vector (added by embedding phase)
            "embedding": {
                "type": "dense_vector",
                "dims": 1024,  # snowflake-arctic-embed2
                "index": True,
                "similarity": "cosine"
            },

            # Timestamps
            "indexed_at": {"type": "date"},
            "expires_at": {"type": "date"}
        }
    }
}
```

### Create Index

```python
def ensure_index_exists(es: ElasticsearchClient):
    """Create index if it doesn't exist"""

    index_name = "magaldi_code_elements"

    if not es.indices.exists(index=index_name):
        es.indices.create(index=index_name, body=MAGALDI_INDEX_MAPPING)
        log.info(f"Created index: {index_name}")
```

### Bulk Index Elements

```python
def index_elements(elements: List[RawElement], es: ElasticsearchClient):
    """Bulk index elements to Elasticsearch"""

    if not elements:
        return

    actions = []
    for elem in elements:
        action = {
            "_index": "magaldi_code_elements",
            "_id": elem.element_id,
            "_source": {
                "element_id": elem.element_id,
                "scope": elem.scope,
                "repository": elem.repository,
                "username": elem.username,
                "relative_path": elem.relative_path,
                "element_type": elem.element_type,
                "name": elem.name,
                "language": elem.language,
                "line_start": elem.line_start,
                "line_end": elem.line_end,
                "raw_code": elem.raw_code,
                "signature": elem.signature,
                "docstring": elem.docstring,
                "summary": None,  # Set by summarization phase
                "level": elem.level,
                "parent_id": elem.parent_id,
                "decorators": elem.decorators,
                "visibility": elem.visibility,
                "is_async": elem.is_async,
                # embedding: set by embedding phase
                "indexed_at": datetime.utcnow().isoformat(),
                "expires_at": elem.expires_at,
            }
        }
        actions.append(action)

    # Bulk index
    success, errors = helpers.bulk(es, actions, raise_on_error=False)

    if errors:
        log.warning(f"Elasticsearch indexing errors: {len(errors)}")
        for error in errors[:5]:  # Log first 5
            log.warning(f"  {error}")

    return success, errors
```

### Refresh Strategy

```python
def index_elements_with_refresh(
    elements: List[RawElement],
    es: ElasticsearchClient,
    batch_size: int = 500
):
    """Index in batches with periodic refresh"""

    total_indexed = 0

    for i in range(0, len(elements), batch_size):
        batch = elements[i:i + batch_size]
        success, _ = index_elements(batch, es)
        total_indexed += success

        # Refresh every 2000 documents
        if total_indexed % 2000 == 0:
            es.indices.refresh(index="magaldi_code_elements")

    # Final refresh
    es.indices.refresh(index="magaldi_code_elements")

    return total_indexed
```

---

## 4.4 Create Jobs

### Purpose

Create summarization and embedding jobs for the AI processing phases.

### Job Tables Schema

```sql
-- Summarization jobs (hierarchical, level-dependent)
CREATE TABLE summarization_jobs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    element_id VARCHAR(512) NOT NULL,
    level INT NOT NULL,                    -- 0=file, 1=class, 2=function, 3=variable
    parent_element_id VARCHAR(512),        -- Dependency

    status ENUM('pending', 'running', 'completed', 'failed') DEFAULT 'pending',
    dependencies_met BOOLEAN DEFAULT FALSE,
    priority INT DEFAULT 0,                -- Higher = process first

    worker_id VARCHAR(100),
    claimed_at DATETIME,
    completed_at DATETIME,
    retry_count INT DEFAULT 0,
    error_message TEXT,

    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,

    UNIQUE KEY unique_element (element_id),
    INDEX idx_status_level (status, level, dependencies_met),
    INDEX idx_worker (worker_id)
);

-- Embedding jobs (flat, no dependencies)
CREATE TABLE embedding_jobs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    element_id VARCHAR(512) NOT NULL,

    status ENUM('pending', 'running', 'completed', 'failed') DEFAULT 'pending',

    worker_id VARCHAR(100),
    claimed_at DATETIME,
    completed_at DATETIME,
    retry_count INT DEFAULT 0,
    error_message TEXT,

    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,

    UNIQUE KEY unique_element (element_id),
    INDEX idx_status (status)
);
```

### Create Summarization Jobs

```python
def create_summarization_jobs(elements: List[RawElement], db: MySQLConnection):
    """Create hierarchical summarization jobs"""

    # Group by level
    by_level = {}
    for elem in elements:
        level = elem.level
        if level not in by_level:
            by_level[level] = []
        by_level[level].append(elem)

    # Create jobs level by level
    for level in sorted(by_level.keys()):
        level_elements = by_level[level]

        values = []
        for elem in level_elements:
            # Level 0 (files) have no dependencies
            # Other levels depend on parent
            dependencies_met = (level == 0)

            values.append((
                elem.element_id,
                level,
                elem.parent_id,
                'pending',
                dependencies_met,
                100 - level,  # Priority: files first, then classes, then functions
            ))

        db.executemany("""
            INSERT INTO summarization_jobs (
                element_id, level, parent_element_id,
                status, dependencies_met, priority
            )
            VALUES (%s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                status = 'pending',
                dependencies_met = VALUES(dependencies_met),
                retry_count = 0,
                error_message = NULL
        """, values)


def update_dependencies_after_completion(element_id: str, db: MySQLConnection):
    """Mark dependent jobs as ready when parent completes"""

    db.execute("""
        UPDATE summarization_jobs
        SET dependencies_met = TRUE
        WHERE parent_element_id = %s
          AND status = 'pending'
    """, (element_id,))
```

### Create Embedding Jobs

```python
def create_embedding_jobs(elements: List[RawElement], db: MySQLConnection):
    """Create embedding jobs (created after summarization completes)"""

    # Only create embedding jobs for elements that should be embedded
    # Skip variables unless they're significant
    embeddable = [e for e in elements if should_embed(e)]

    values = [(elem.element_id, 'pending') for elem in embeddable]

    db.executemany("""
        INSERT INTO embedding_jobs (element_id, status)
        VALUES (%s, %s)
        ON DUPLICATE KEY UPDATE
            status = 'pending',
            retry_count = 0,
            error_message = NULL
    """, values)


def should_embed(element: RawElement) -> bool:
    """Determine if element should be embedded"""

    # Always embed files, classes, functions, methods
    if element.element_type in ('file', 'class', 'function', 'method'):
        return True

    # Only embed significant variables (constants, config, exports)
    if element.element_type == 'variable':
        name = element.name
        # Embed if: ALL_CAPS (constant), exported, or has docstring
        if name.isupper():
            return True
        if element.docstring:
            return True
        # Skip private variables
        if name.startswith('_'):
            return False

    return False
```

### Job Creation Strategy

```
┌─────────────────────────────────────────────────────────────────┐
│                    JOB CREATION STRATEGY                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Phase 4 (Storage):                                             │
│  ├── Create summarization_jobs for ALL elements                 │
│  ├── Level 0 jobs: dependencies_met = TRUE                      │
│  └── Level 1+ jobs: dependencies_met = FALSE                    │
│                                                                 │
│  Phase 5 (Summarization - separate process):                    │
│  ├── Workers claim Level 0 jobs (no dependencies)               │
│  ├── On completion: update_dependencies_after_completion()      │
│  ├── Level 1 jobs become available                              │
│  └── Continue until all levels complete                         │
│                                                                 │
│  Phase 6 (Embedding - after summarization):                     │
│  ├── Create embedding_jobs for summarized elements              │
│  ├── No dependencies (all parallel)                             │
│  └── Workers process in any order                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Output

Phase 4 produces:

```python
@dataclass
class StorageResult:
    scope: str
    repository: str
    username: str

    # Counts
    files_stored: int
    elements_stored: int
    elements_indexed: int
    elements_deleted: int

    # Jobs created
    summarization_jobs_created: int
    embedding_jobs_created: int

    # Errors
    storage_errors: List[str]
    indexing_errors: List[str]
```

---

## Progress Reporting

```
[Storage]
Handling deletions...                 1 file, 12 elements removed
Storing to MySQL...                   8 files, 67 elements
  Updating file states...             done
  Inserting elements...               67/67 (100%)
  Updating repository stats...        done
Indexing to Elasticsearch...          67/67 (100%)
Creating jobs...
  Summarization jobs...               67 created
  Embedding jobs...                   59 created (8 variables skipped)

Summary:
  Files stored:     8
  Elements stored:  67
  Elements indexed: 67
  Jobs created:     126
```

---

## Error Handling

| Error | Action |
|-------|--------|
| MySQL connection failed | Retry 3x, then abort |
| MySQL transaction failed | Rollback, report error |
| Elasticsearch unavailable | Log warning, continue (MySQL is source of truth) |
| ES bulk insert partial failure | Log failures, continue with successes |
| Job creation failed | Log error, elements still stored |

### Retry Logic

```python
def with_retry(func, max_retries: int = 3, delay: float = 1.0):
    """Execute function with exponential backoff retry"""

    for attempt in range(max_retries):
        try:
            return func()
        except (ConnectionError, TimeoutError) as e:
            if attempt == max_retries - 1:
                raise
            sleep_time = delay * (2 ** attempt)
            log.warning(f"Retry {attempt + 1}/{max_retries} after {sleep_time}s: {e}")
            time.sleep(sleep_time)
```

### Consistency Guarantee

```
MySQL is the source of truth.
Elasticsearch is a search index that can be rebuilt.

On ES failure:
1. Log warning
2. Continue with MySQL storage
3. Mark elements as "needs_reindex" in MySQL
4. Background job rebuilds ES index from MySQL
```

---

## Performance Considerations

| Operation | Bottleneck | Optimization |
|-----------|------------|--------------|
| MySQL inserts | Network, locks | Batch inserts, single transaction |
| ES indexing | Network | Bulk API, batch 500 docs |
| Deletion queries | Full scan | Use element_id prefix index |
| Job creation | Inserts | Batch inserts |

### Batch Sizes

```python
BATCH_SIZES = {
    'mysql_insert': 100,      # Elements per INSERT statement
    'es_bulk': 500,           # Documents per bulk request
    'es_refresh_interval': 2000,  # Docs between refresh
    'job_insert': 200,        # Jobs per INSERT statement
}
```

---

## Cleanup Operations

### Expire Old User Data

```python
def expire_old_data(db: MySQLConnection, es: ElasticsearchClient):
    """Remove expired user branch data"""

    # Find expired files
    expired = db.query("""
        SELECT scope, repository, username, relative_path
        FROM file_states
        WHERE expires_at < NOW()
          AND username != 'main'
    """)

    for row in expired:
        # Delete elements
        delete_file_elements(row, db, es)

        # Delete file state
        db.execute("""
            DELETE FROM file_states
            WHERE scope = %s AND repository = %s
              AND username = %s AND relative_path = %s
        """, (row.scope, row.repository, row.username, row.relative_path))

    # Delete expired elements directly
    db.execute("""
        DELETE FROM code_elements
        WHERE expires_at < NOW()
    """)

    # Delete from ES
    es.delete_by_query(
        index="magaldi_code_elements",
        body={
            "query": {
                "range": {
                    "expires_at": {"lt": "now"}
                }
            }
        }
    )
```

### Rebuild Elasticsearch Index

```python
def rebuild_es_index(db: MySQLConnection, es: ElasticsearchClient):
    """Rebuild ES index from MySQL (disaster recovery)"""

    # Delete and recreate index
    es.indices.delete(index="magaldi_code_elements", ignore=[404])
    ensure_index_exists(es)

    # Stream elements from MySQL
    offset = 0
    page_size = 1000

    while True:
        elements = db.query("""
            SELECT * FROM code_elements
            ORDER BY element_id
            LIMIT %s OFFSET %s
        """, (page_size, offset))

        if not elements:
            break

        # Convert to RawElement and index
        raw_elements = [row_to_element(row) for row in elements]
        index_elements(raw_elements, es)

        offset += page_size
        log.info(f"Reindexed {offset} elements...")

    es.indices.refresh(index="magaldi_code_elements")
    log.info("ES index rebuild complete")
```

---

## Summary of Decisions

| Decision | Value |
|----------|-------|
| Source of truth | MySQL (ES is derived) |
| Transaction scope | Per-file batch |
| ES bulk size | 500 documents |
| Deletion strategy | Remove old before insert new |
| User deletion | Soft delete (is_deleted marker) |
| Main deletion | Hard delete |
| Job creation | After successful storage |
| Embedding jobs | Created after summarization (not in Phase 4) |
| Expiration | 30 days for user branches |
| Retry strategy | 3 attempts, exponential backoff |
