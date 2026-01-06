# Magaldi AI Processing - Phase 5: Summarization

## Overview

The Summarization phase uses Ollama to generate natural language summaries of code elements. Processing is hierarchical (files first, then classes, then functions) because child summaries require parent context for higher quality.

```
┌─────────────────────────────────────────────────────────────────┐
│                     PHASE 5: SUMMARIZATION                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  5.1              5.2              5.3              5.4          │
│  ─────            ─────            ─────            ─────        │
│  INITIALIZE   →   PROCESS      →   GENERATE    →   STORE        │
│  WORKERS          LEVELS           SUMMARIES       RESULTS       │
│                                                                 │
│  • Worker pool    • Level 0 first  • Ollama API    • MySQL      │
│  • Ollama conn    • Unlock deps    • Prompts       • Update ES  │
│  • Job claiming   • Batch by file  • Retry logic   • Mark done  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Hierarchical Processing Order

```
Level 0: FILES          (no dependencies - process first)
    ↓
Level 1: CLASSES        (depend on file summary)
    ↓
Level 2: FUNCTIONS      (depend on file + class summary)
         METHODS
    ↓
Level 3: VARIABLES      (depend on file summary, optional)
```

---

## Input

From Phase 4 (Storage):

```python
# Jobs created in summarization_jobs table
# Schema from Phase 4:

@dataclass
class SummarizationJob:
    id: int
    element_id: str
    level: int                    # 0=file, 1=class, 2=function, 3=variable
    parent_element_id: Optional[str]

    status: str                   # 'pending', 'running', 'completed', 'failed'
    dependencies_met: bool        # True when parent is summarized
    priority: int                 # Higher = process first

    worker_id: Optional[str]
    claimed_at: Optional[datetime]
    completed_at: Optional[datetime]
    retry_count: int
    error_message: Optional[str]
```

---

## 5.1 Initialize Workers

### Purpose

Set up worker pool and Ollama connection for processing summarization jobs.

### Worker Pool Configuration

```python
@dataclass
class SummarizationConfig:
    # Ollama settings
    ollama_url: str = "http://ollama:11434"
    model: str = "qwen2.5-coder:7b"

    # Worker settings
    num_workers: int = 4
    batch_size: int = 10              # Jobs per batch
    claim_timeout: int = 300          # 5 minutes before job is reclaimed

    # Retry settings
    max_retries: int = 3
    retry_delay: float = 5.0          # Seconds between retries

    # Model settings
    temperature: float = 0.3          # Low for consistent summaries
    max_tokens: int = 256             # Summary length limit
    context_window: int = 8192        # Safe limit for qwen2.5-coder:7b


class SummarizationWorkerPool:
    """Manages workers for processing summarization jobs"""

    def __init__(self, config: SummarizationConfig, db: MySQLConnection):
        self.config = config
        self.db = db
        self.ollama = OllamaClient(config.ollama_url, config.model)
        self.workers: List[SummarizationWorker] = []
        self.shutdown_event = threading.Event()

    def start(self, num_workers: Optional[int] = None):
        """Start worker pool"""

        num = num_workers or self.config.num_workers

        # Verify Ollama connection and model
        if not self.ollama.verify_model():
            raise RuntimeError(f"Model {self.config.model} not available in Ollama")

        for i in range(num):
            worker = SummarizationWorker(
                worker_id=f"summarize-{i}",
                config=self.config,
                db=self.db,
                ollama=self.ollama,
                shutdown_event=self.shutdown_event
            )
            worker.start()
            self.workers.append(worker)

        log.info(f"Started {num} summarization workers")

    def shutdown(self, wait: bool = True):
        """Gracefully shutdown workers"""
        self.shutdown_event.set()
        if wait:
            for worker in self.workers:
                worker.join(timeout=30)
```

### Ollama Client

```python
class OllamaClient:
    """Client for Ollama API interactions"""

    def __init__(self, url: str, model: str):
        self.url = url.rstrip('/')
        self.model = model
        self.session = requests.Session()
        self.session.headers['Content-Type'] = 'application/json'

    def verify_model(self) -> bool:
        """Check if model is available"""
        try:
            response = self.session.get(f"{self.url}/api/tags")
            models = response.json().get('models', [])
            return any(m['name'] == self.model for m in models)
        except Exception as e:
            log.error(f"Failed to connect to Ollama: {e}")
            return False

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate completion from Ollama"""

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": kwargs.get('temperature', 0.3),
                "num_predict": kwargs.get('max_tokens', 256),
            }
        }

        response = self.session.post(
            f"{self.url}/api/generate",
            json=payload,
            timeout=60
        )
        response.raise_for_status()

        return response.json()['response'].strip()
```

### Worker ID Generation

```python
def generate_worker_id() -> str:
    """Generate unique worker ID for job claiming"""

    hostname = socket.gethostname()
    pid = os.getpid()
    timestamp = int(time.time())

    return f"summarize-{hostname}-{pid}-{timestamp}"
```

---

## 5.2 Process Levels

### Purpose

Process summarization jobs in hierarchical order, respecting dependencies between levels.

### Level Processing Strategy

```python
class SummarizationWorker(threading.Thread):
    """Worker thread for processing summarization jobs"""

    def __init__(self, worker_id: str, config: SummarizationConfig,
                 db: MySQLConnection, ollama: OllamaClient,
                 shutdown_event: threading.Event):
        super().__init__(daemon=True)
        self.worker_id = worker_id
        self.config = config
        self.db = db
        self.ollama = ollama
        self.shutdown_event = shutdown_event

    def run(self):
        """Main worker loop"""

        while not self.shutdown_event.is_set():
            # Try to claim and process jobs
            jobs = self.claim_jobs()

            if not jobs:
                # No work available, sleep and retry
                time.sleep(1.0)
                continue

            for job in jobs:
                if self.shutdown_event.is_set():
                    break
                self.process_job(job)

    def claim_jobs(self) -> List[SummarizationJob]:
        """Claim available jobs with dependency checking"""

        # Atomic claim: select and update in transaction
        self.db.begin_transaction()

        try:
            # Find jobs where dependencies are met
            # Process lower levels first (files before classes)
            jobs = self.db.query("""
                SELECT * FROM summarization_jobs
                WHERE status = 'pending'
                  AND dependencies_met = TRUE
                ORDER BY level ASC, priority DESC
                LIMIT %s
                FOR UPDATE SKIP LOCKED
            """, (self.config.batch_size,))

            if not jobs:
                self.db.rollback()
                return []

            # Mark as running
            job_ids = [j.id for j in jobs]
            self.db.execute("""
                UPDATE summarization_jobs
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


def reclaim_stale_jobs(db: MySQLConnection, timeout_minutes: int = 5):
    """Reclaim jobs from dead workers"""

    db.execute("""
        UPDATE summarization_jobs
        SET status = 'pending',
            worker_id = NULL,
            claimed_at = NULL,
            retry_count = retry_count + 1
        WHERE status = 'running'
          AND claimed_at < DATE_SUB(NOW(), INTERVAL %s MINUTE)
          AND retry_count < 3
    """, (timeout_minutes,))
```

### Dependency Resolution

```python
def update_dependencies_after_completion(element_id: str, db: MySQLConnection):
    """Mark child jobs as ready when parent completes"""

    # Find all jobs that depend on this element
    db.execute("""
        UPDATE summarization_jobs
        SET dependencies_met = TRUE
        WHERE parent_element_id = %s
          AND status = 'pending'
          AND dependencies_met = FALSE
    """, (element_id,))

    affected = db.rowcount
    if affected > 0:
        log.debug(f"Unlocked {affected} dependent jobs for {element_id}")


def check_level_completion(level: int, scope: str, repository: str,
                          username: str, db: MySQLConnection) -> bool:
    """Check if all jobs at a level are complete"""

    result = db.query_one("""
        SELECT COUNT(*) as pending
        FROM summarization_jobs sj
        JOIN code_elements ce ON sj.element_id = ce.element_id
        WHERE sj.level = %s
          AND sj.status IN ('pending', 'running')
          AND ce.scope = %s
          AND ce.repository = %s
          AND ce.username = %s
    """, (level, scope, repository, username))

    return result.pending == 0
```

### Batch Processing by File

```python
def process_jobs_by_file(jobs: List[SummarizationJob],
                        db: MySQLConnection) -> Dict[str, List[SummarizationJob]]:
    """Group jobs by file for efficient context loading"""

    by_file = {}

    for job in jobs:
        # Extract file path from element_id
        # Format: scope:repo:user:path:type:name:line
        parts = job.element_id.split(':')
        file_path = parts[3]

        if file_path not in by_file:
            by_file[file_path] = []
        by_file[file_path].append(job)

    return by_file
```

---

## 5.3 Generate Summaries

### Purpose

Generate natural language summaries using Ollama with level-appropriate prompts.

### Prompt Templates

```python
PROMPTS = {
    'file': """Summarize this {language} file in 2-3 sentences. Focus on:
- Primary purpose and responsibility
- Key classes/functions it provides
- Notable patterns or design choices

File: {file_path}

Code:
{code}

Summary:""",

    'class': """Summarize this {language} class in 2-3 sentences. Include its purpose,
key responsibilities, and notable patterns.

File context: {file_summary}

Class: {class_name}
{decorators}

Code:
{code}

Summary:""",

    'function': """Summarize what this function does in 1-2 concise sentences.

File context: {file_summary}
{class_context}

Function: {function_name}
Signature: {signature}
{docstring_section}

Code:
{code}

Summary:""",

    'method': """Summarize what this method does in 1-2 concise sentences.

File context: {file_summary}
Class context: {class_summary}

Method: {method_name}
Signature: {signature}
{docstring_section}

Code:
{code}

Summary:""",

    'variable': """Describe this {language} variable/constant in one sentence.

File: {file_path}
Name: {name}
Value:
{code}

Description:"""
}


def build_prompt(element: CodeElement, parent_summaries: Dict[str, str]) -> str:
    """Build prompt with parent context"""

    element_type = element.element_type

    # Get appropriate template
    if element_type == 'method':
        template = PROMPTS['method']
    elif element_type in PROMPTS:
        template = PROMPTS[element_type]
    else:
        template = PROMPTS['function']  # Default

    # Build context sections
    file_summary = parent_summaries.get('file', 'No file context available.')
    class_summary = parent_summaries.get('class', '')

    class_context = ""
    if class_summary and element_type in ('method', 'function'):
        class_context = f"Class context: {class_summary}"

    docstring_section = ""
    if element.docstring:
        docstring_section = f"Docstring: {element.docstring}"

    decorators = ""
    if element.decorators:
        decorators = f"Decorators: {', '.join(element.decorators)}"

    # Format prompt
    return template.format(
        language=element.language,
        file_path=element.relative_path,
        file_summary=file_summary,
        class_summary=class_summary,
        class_context=class_context,
        class_name=element.name if element_type == 'class' else '',
        function_name=element.name,
        method_name=element.name,
        name=element.name,
        signature=element.signature or '',
        docstring_section=docstring_section,
        decorators=decorators,
        code=truncate_code(element.raw_code, max_tokens=4000)
    )
```

### Code Truncation

```python
def truncate_code(code: str, max_tokens: int = 4000) -> str:
    """Truncate code to fit context window"""

    # Rough estimate: 1 token ~= 4 characters for code
    max_chars = max_tokens * 4

    if len(code) <= max_chars:
        return code

    # Try to truncate at a logical boundary
    truncated = code[:max_chars]

    # Find last complete line
    last_newline = truncated.rfind('\n')
    if last_newline > max_chars * 0.8:  # Don't truncate too much
        truncated = truncated[:last_newline]

    return truncated + "\n\n# ... (truncated)"
```

### Summary Generation

```python
def generate_summary(job: SummarizationJob, db: MySQLConnection,
                    ollama: OllamaClient, config: SummarizationConfig) -> str:
    """Generate summary for a single element"""

    # 1. Load element
    element = load_element(job.element_id, db)
    if not element:
        raise ValueError(f"Element not found: {job.element_id}")

    # 2. Load parent summaries
    parent_summaries = load_parent_summaries(element, db)

    # 3. Build prompt
    prompt = build_prompt(element, parent_summaries)

    # 4. Generate with Ollama
    summary = ollama.generate(
        prompt=prompt,
        temperature=config.temperature,
        max_tokens=config.max_tokens
    )

    # 5. Clean up summary
    summary = clean_summary(summary)

    return summary


def load_parent_summaries(element: CodeElement, db: MySQLConnection) -> Dict[str, str]:
    """Load summaries from parent elements"""

    summaries = {}

    # Load file summary (always available after level 0)
    if element.level > 0:
        file_element = db.query_one("""
            SELECT summary FROM code_elements
            WHERE scope = %s AND repository = %s AND username = %s
              AND relative_path = %s AND element_type = 'file'
        """, (element.scope, element.repository, element.username, element.relative_path))

        if file_element and file_element.summary:
            summaries['file'] = file_element.summary

    # Load class summary for methods
    if element.parent_id and element.element_type in ('method', 'function'):
        parent = db.query_one("""
            SELECT summary, element_type FROM code_elements
            WHERE element_id = %s
        """, (element.parent_id,))

        if parent and parent.element_type == 'class' and parent.summary:
            summaries['class'] = parent.summary

    return summaries


def clean_summary(summary: str) -> str:
    """Clean and normalize generated summary"""

    # Remove common prefixes from LLM responses
    prefixes_to_remove = [
        "Summary:",
        "This function",
        "This method",
        "This class",
        "This file",
    ]

    summary = summary.strip()

    for prefix in prefixes_to_remove:
        if summary.startswith(prefix):
            summary = summary[len(prefix):].strip()

    # Ensure it ends with a period
    if summary and not summary.endswith('.'):
        summary += '.'

    # Capitalize first letter
    if summary:
        summary = summary[0].upper() + summary[1:]

    return summary
```

### Retry Logic

```python
def process_job_with_retry(job: SummarizationJob, db: MySQLConnection,
                          ollama: OllamaClient, config: SummarizationConfig) -> bool:
    """Process job with retry handling"""

    last_error = None

    for attempt in range(config.max_retries):
        try:
            summary = generate_summary(job, db, ollama, config)

            # Success - store result
            store_summary(job.element_id, summary, db)
            mark_job_completed(job.id, db)
            update_dependencies_after_completion(job.element_id, db)

            return True

        except OllamaError as e:
            last_error = e
            log.warning(f"Ollama error (attempt {attempt + 1}): {e}")

            if attempt < config.max_retries - 1:
                time.sleep(config.retry_delay * (2 ** attempt))  # Exponential backoff

        except Exception as e:
            last_error = e
            log.error(f"Unexpected error processing {job.element_id}: {e}")
            break

    # All retries failed
    mark_job_failed(job.id, str(last_error), db)
    return False
```

---

## 5.4 Store Results

### Purpose

Persist generated summaries to MySQL and update Elasticsearch.

### Store Summary

```python
def store_summary(element_id: str, summary: str, db: MySQLConnection):
    """Store summary in MySQL"""

    db.execute("""
        UPDATE code_elements
        SET summary = %s,
            summary_status = 'completed'
        WHERE element_id = %s
    """, (summary, element_id))


def mark_job_completed(job_id: int, db: MySQLConnection):
    """Mark summarization job as completed"""

    db.execute("""
        UPDATE summarization_jobs
        SET status = 'completed',
            completed_at = NOW()
        WHERE id = %s
    """, (job_id,))


def mark_job_failed(job_id: int, error_message: str, db: MySQLConnection):
    """Mark job as failed with error"""

    db.execute("""
        UPDATE summarization_jobs
        SET status = 'failed',
            error_message = %s,
            completed_at = NOW()
        WHERE id = %s
    """, (error_message, job_id))

    # Also update element status
    db.execute("""
        UPDATE code_elements ce
        JOIN summarization_jobs sj ON ce.element_id = sj.element_id
        SET ce.summary_status = 'failed'
        WHERE sj.id = %s
    """, (job_id,))
```

### Update Elasticsearch

```python
def update_es_summary(element_id: str, summary: str, es: ElasticsearchClient):
    """Update summary in Elasticsearch"""

    es.update(
        index="magaldi_code_elements",
        id=element_id,
        body={
            "doc": {
                "summary": summary
            }
        }
    )


def batch_update_es_summaries(summaries: List[Tuple[str, str]],
                              es: ElasticsearchClient):
    """Batch update summaries in Elasticsearch"""

    actions = []
    for element_id, summary in summaries:
        actions.append({
            "_op_type": "update",
            "_index": "magaldi_code_elements",
            "_id": element_id,
            "doc": {"summary": summary}
        })

    if actions:
        helpers.bulk(es, actions, raise_on_error=False)
```

### Post-Processing: Create Embedding Jobs

```python
def create_embedding_jobs_for_completed(scope: str, repository: str,
                                       username: str, db: MySQLConnection):
    """Create embedding jobs for elements with completed summaries"""

    # Find elements with summaries but no embedding jobs
    elements = db.query("""
        SELECT ce.element_id, ce.element_type
        FROM code_elements ce
        LEFT JOIN embedding_jobs ej ON ce.element_id = ej.element_id
        WHERE ce.scope = %s
          AND ce.repository = %s
          AND ce.username = %s
          AND ce.summary_status = 'completed'
          AND ej.id IS NULL
    """, (scope, repository, username))

    # Filter by embeddable types
    embeddable = [
        (e.element_id,)
        for e in elements
        if should_embed(e.element_type)
    ]

    if embeddable:
        db.executemany("""
            INSERT INTO embedding_jobs (element_id, status)
            VALUES (%s, 'pending')
            ON DUPLICATE KEY UPDATE status = status
        """, embeddable)

        log.info(f"Created {len(embeddable)} embedding jobs")


def should_embed(element_type: str) -> bool:
    """Determine if element type should be embedded"""
    return element_type in ('file', 'class', 'function', 'method')
```

---

## Output

Phase 5 produces summaries stored in:

```python
# MySQL: code_elements.summary field populated
# MySQL: code_elements.summary_status = 'completed'
# MySQL: summarization_jobs.status = 'completed'
# Elasticsearch: summary field updated

@dataclass
class SummarizationResult:
    scope: str
    repository: str
    username: str

    # Counts by level
    files_summarized: int
    classes_summarized: int
    functions_summarized: int
    variables_summarized: int

    # Status
    total_jobs: int
    completed_jobs: int
    failed_jobs: int

    # Timing
    start_time: datetime
    end_time: datetime
    avg_summary_time_ms: float

    # Errors
    errors: List[SummarizationError]
```

---

## Progress Reporting

```
[Summarization]
Initializing workers...               4 workers, model: qwen2.5-coder:7b
Processing repository...              backend:auth-service:main

Level 0 (Files):                      8/8 (100%)
  src/auth/login.py                   ✓ (145ms)
  src/auth/session.py                 ✓ (132ms)
  src/utils/helpers.js                ✓ (98ms)
  ...

Level 1 (Classes):                    5/5 (100%)
  AuthService                         ✓ (187ms)
  SessionManager                      ✓ (156ms)
  ...

Level 2 (Functions/Methods):          31/31 (100%)
  authenticate_user                   ✓ (89ms)
  validate_credentials                ✓ (76ms)
  ...

Summary:
  Total elements:     67
  Summarized:         67
  Failed:             0
  Avg time:           112ms
  Total time:         7.5s

Creating embedding jobs...            59 jobs created
```

---

## Error Handling

| Error | Action |
|-------|--------|
| Ollama unavailable | Retry 3x with backoff, then fail job |
| Model not loaded | Attempt model pull, retry |
| Context too long | Truncate code, regenerate |
| Empty response | Retry with higher temperature |
| Timeout | Retry with extended timeout |
| Invalid element | Skip, mark failed |
| Worker crash | Reclaim stale jobs after timeout |
| Database error | Rollback, retry transaction |

### Health Monitoring

```python
class SummarizationHealthCheck:
    """Monitor summarization pipeline health"""

    def __init__(self, db: MySQLConnection, ollama: OllamaClient):
        self.db = db
        self.ollama = ollama

    def check_ollama(self) -> bool:
        """Verify Ollama is responsive"""
        try:
            return self.ollama.verify_model()
        except Exception:
            return False

    def check_queue_depth(self) -> Dict[str, int]:
        """Get pending jobs by level"""
        result = self.db.query("""
            SELECT level, COUNT(*) as count
            FROM summarization_jobs
            WHERE status = 'pending'
            GROUP BY level
        """)
        return {row.level: row.count for row in result}

    def check_failure_rate(self, window_minutes: int = 60) -> float:
        """Calculate recent failure rate"""
        result = self.db.query_one("""
            SELECT
                SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed,
                COUNT(*) as total
            FROM summarization_jobs
            WHERE completed_at > DATE_SUB(NOW(), INTERVAL %s MINUTE)
        """, (window_minutes,))

        if result.total == 0:
            return 0.0
        return result.failed / result.total

    def check_stale_jobs(self) -> int:
        """Count jobs stuck in running state"""
        result = self.db.query_one("""
            SELECT COUNT(*) as count
            FROM summarization_jobs
            WHERE status = 'running'
              AND claimed_at < DATE_SUB(NOW(), INTERVAL 10 MINUTE)
        """)
        return result.count
```

---

## Performance Considerations

| Operation | Bottleneck | Optimization |
|-----------|------------|--------------|
| Ollama inference | GPU/CPU | Batch prompts where possible |
| Context loading | Database | Cache parent summaries in memory |
| Job claiming | Lock contention | SKIP LOCKED, batch claims |
| ES updates | Network | Batch updates after level completion |
| Large files | Context window | Smart truncation |

### Throughput Estimates

| Hardware | Model | Elements/min | Notes |
|----------|-------|--------------|-------|
| RTX 3060 | qwen2.5-coder:7b | ~40-60 | 4 workers |
| RTX 3080 | qwen2.5-coder:7b | ~60-80 | 4 workers |
| RTX 4090 | qwen2.5-coder:14b | ~30-50 | Higher quality |
| CPU only | qwen2.5-coder:3b | ~10-20 | Development |

### Memory Optimization

```python
# Cache parent summaries per file to avoid repeated queries
class ParentSummaryCache:
    """LRU cache for parent summaries"""

    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.max_size = max_size
        self.access_order = []

    def get(self, element_id: str) -> Optional[str]:
        if element_id in self.cache:
            self.access_order.remove(element_id)
            self.access_order.append(element_id)
            return self.cache[element_id]
        return None

    def put(self, element_id: str, summary: str):
        if len(self.cache) >= self.max_size:
            oldest = self.access_order.pop(0)
            del self.cache[oldest]

        self.cache[element_id] = summary
        self.access_order.append(element_id)
```

---

## CLI Interface

```bash
# Start summarization for a repository
magaldi summarize --scope backend --repo auth-service --user main

# Resume failed jobs
magaldi summarize --scope backend --repo auth-service --user main --retry-failed

# Monitor progress
magaldi summarize --scope backend --repo auth-service --user main --watch

# Run with specific worker count
magaldi summarize --workers 8 --scope backend --repo auth-service --user main

# Dry run (show what would be processed)
magaldi summarize --scope backend --repo auth-service --user main --dry-run
```

---

## Summary of Decisions

| Decision | Value |
|----------|-------|
| Processing model | Ollama (local) |
| Summarization model | qwen2.5-coder:7b |
| Processing order | Hierarchical (files → classes → functions) |
| Worker model | Thread pool (configurable count) |
| Job claiming | SKIP LOCKED (non-blocking) |
| Batch size | 10 jobs per claim |
| Retry strategy | 3 attempts, exponential backoff |
| Stale job timeout | 5 minutes |
| Summary storage | MySQL (source of truth) + ES (search) |
| Context handling | Truncate at 4000 tokens |
| Temperature | 0.3 (consistent output) |
| Max summary tokens | 256 |
| Parent context | Loaded from completed parent summaries |
| Embedding job creation | After summarization completes |
