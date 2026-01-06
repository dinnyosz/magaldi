# Magaldi Project - Complete Implementation Plan

## Project Overview

**Magaldi** is an open-source code discovery engine named after Agustín Magaldi, who helped launch Eva Perón's career. The project helps AI agents and developers navigate and understand codebases through intelligent indexing and semantic search.

### Goals
- Parse codebases and extract structured elements (classes, functions, variables, etc.)
- Summarize code elements using local Ollama models
- Create a vector database for semantic search
- Build an MCP server for Claude Code and other coding agents
- Provide a visual web interface for exploration
- Detect feature clusters automatically

---

## Technology Stack

### Core Technologies
- **Parser**: Tree-sitter (multi-language support)
- **Database**: Percona MySQL 8.0 (persistent storage)
- **Vector Store**: Elasticsearch (embeddings and search)
- **AI Models**: Ollama (local, with GPU support)
- **MCP**: Python MCP SDK
- **Web**: Flask/FastAPI backend, HTML/JS/CSS frontend
- **Visualization**: D3.js, three.js, or sigma.js
- **Deployment**: Docker Compose

### Supported Languages
- Python
- JavaScript
- TypeScript
- Node.js
- PHP
- Rust

---

## Architecture Overview

### Docker Services

```yaml
services:
  - mysql (Percona 8.0)
  - elasticsearch (8.11.0)
  - ollama (with GPU support)
  - parser (Python service)
  - mcp_server (MCP interface)
  - web (Web UI)
```

### Storage Volumes
- `./storage/mysql_data` - Percona MySQL data
- `./storage/elasticsearch_data` - Elasticsearch indices
- `./storage/ollama_models` - Ollama model cache
- `./storage/repos` - Source repositories (read-only)

---

## Project Structure

```
magaldi/
├── docker-compose.yml
├── Dockerfile.parser
├── Dockerfile.ollama
├── Dockerfile.elasticsearch
├── Dockerfile.web
├── .env.example
├── config/
│   ├── mysql/
│   │   └── my.cnf
│   ├── elasticsearch.yml
│   └── ollama-models.txt
├── schema/
│   └── init.sql
├── parsers/
│   ├── __init__.py
│   ├── base_parser.py
│   ├── tree_sitter_parser.py
│   ├── language_parsers/
│   │   ├── python_parser.py
│   │   ├── javascript_parser.py
│   │   ├── typescript_parser.py
│   │   ├── php_parser.py
│   │   ├── rust_parser.py
│   │   └── nodejs_parser.py
│   └── queries/
│       ├── python.scm
│       ├── javascript.scm
│       ├── typescript.scm
│       ├── php.scm
│       └── rust.scm
├── core/
│   ├── __init__.py
│   ├── database.py              # MySQL connection pool
│   ├── repository_manager.py    # Repo scanning and scoping
│   ├── file_tracker.py          # SHA256-based change detection
│   ├── parser_orchestrator.py   # Main parsing coordinator
│   ├── summarizer.py            # Ollama summarization
│   ├── embedder.py              # Vector generation
│   ├── indexer.py               # Elasticsearch operations
│   ├── clusterer.py             # Feature detection
│   └── job_manager.py           # Hierarchical job processing
├── mcp_server/
│   ├── __init__.py
│   ├── server.py
│   ├── tools/
│   │   ├── search.py
│   │   ├── explore.py
│   │   └── analyze.py
│   └── schemas/
├── web/
│   ├── backend/
│   │   ├── app.py
│   │   └── api/
│   │       ├── repositories.py
│   │       ├── search.py
│   │       └── visualize.py
│   └── frontend/
│       ├── index.html
│       ├── static/
│       │   ├── css/
│       │   ├── js/
│       │   └── assets/
│       └── components/
├── storage/
│   ├── repos/                   # Mounted repositories
│   ├── mysql_data/              # Percona data
│   ├── elasticsearch_data/      # ES data
│   └── ollama_models/           # Ollama models
└── tests/
    ├── unit/
    └── integration/
```

---

## Key Design Decisions

### 1. Repository & Scoping System

**Repository Auto-Detection:**
- Scan `/repos/` directory
- Each subdirectory = one repository
- Repository name = directory name
- Automatically extract from path structure

**Scoping:**
- Multiple repositories can share a scope (logical grouping)
- Example: `backend-services` scope contains `project-a` and `project-b`
- Enables cross-repository semantic search within a scope

**Repository Metadata:**
- Git remote, branch, last commit (extracted from `.git/`)
- Primary language (auto-detected from files)
- Description (from README.md)
- File count, line count
- Tags for categorization

### 2. Change Detection & Deduplication

**File Tracking:**
- SHA256 hash of each file stored in MySQL
- Before parsing: check if file hash changed
- If unchanged: skip parsing entirely
- If changed: remove old elements, parse fresh

**Element IDs:**
```
Format: {scope}:{repository}:{relative_path}:{element_type}:{element_name}:{line}
Example: backend-services:project-a:src/auth/login.py:function:authenticate_user:45
```

**Deduplication Strategy:**
- Each element ID is unique
- On file change: delete old element IDs from Elasticsearch
- Re-run parser: no duplicate entries created
- File tracker maintains file → element IDs mapping

### 3. Percona MySQL as Primary Storage

**Why Percona MySQL:**
- Multiple concurrent writers (parallel processing)
- Better for distributed systems
- ACID at scale
- Network access for all services
- Familiar tooling and monitoring
- Future-proof for replication/HA

**What's Stored:**
- File states (hash, last parsed, metadata)
- Repository metadata
- Code elements (before and after AI processing)
- Processing jobs (summarization, embedding)
- Language statistics
- Audit logs

### 4. Tree-sitter for Parsing

**Advantages:**
- Unified API across all languages
- Fast incremental parsing
- Handles broken/incomplete code gracefully
- Rich query system (S-expressions)
- Performance (C library with Python bindings)

**What We Extract:**
- Classes (name, docstring, decorators, base classes)
- Functions/methods (signature, docstring, decorators, async)
- Variables/constants (module-level, class attributes)
- Imports (from/import statements)
- Type hints and annotations
- Comments and documentation

**Metadata from Tree-sitter:**
- Byte ranges (exact location)
- Parent-child relationships (nesting)
- Syntax context (async, decorated, exported)
- Comment associations
- Type information

### 5. Hierarchical AI Processing

**Key Insight:** Parsing is fast, AI processing is slow. Parse everything first, then parallelize AI operations hierarchically.

**Processing Phases:**

**Phase 1: PARSE (Fast, Sequential)**
- Tree-sitter parses all files
- Extract all elements (classes, functions, variables)
- Build dependency graph
- Store raw elements in MySQL + Elasticsearch (no summaries yet)

**Phase 2: SUMMARIZATION (Slow, Parallel, Hierarchical)**

```
Level 0: File summaries
  ├─ Job per file
  ├─ Context: just the file content
  └─ Output: file-level summary

Level 1: Class summaries
  ├─ Job per class
  ├─ Context: file summary + class code
  └─ Output: class-level summary

Level 2: Function/Method summaries
  ├─ Job per function
  ├─ Context: file summary + class summary (if in class) + function code
  └─ Output: function-level summary

Level 3: Variable summaries (meaningful ones only)
  ├─ Job per significant variable
  ├─ Context: file + class + function summaries + variable context
  └─ Output: variable-level summary
```

**Phase 3: EMBEDDING (Slow, Parallel, No Dependencies)**
- Job per element (all levels)
- Context: element summary + metadata
- Output: vector embedding

**Phase 4: CLUSTERING (Medium, Sequential or Parallel)**
- Process all embeddings
- Detect feature clusters
- Store cluster metadata

**Why Hierarchical:**
- Better quality summaries (context from upper levels)
- Natural code structure (file → class → function → variable)
- Parallel processing within each level
- Dependencies between levels prevent race conditions
- More efficient GPU utilization

---

## Database Schema

### MySQL Tables

**code_elements** - All parsed code elements
- Primary key: element_id (string)
- Location: scope, repository, file_path, relative_path
- Element info: type, name, language, level, parent_id
- Content: raw_code, signature, docstring
- Metadata: line numbers, complexity, decorators
- Processing status: parse_status, summary_status, embedding_status
- AI results: summary, embedding_vector (or ES only)
- Hierarchy: parent_id, level (0=file, 1=class, 2=function, 3=variable)

**file_states** - Change detection
- File hash (SHA256), last parsed timestamp
- Scope, repository, relative_path (unique key)
- Language, file size, last modified

**file_elements** - Mapping: file → element IDs
- For cleanup when file changes
- Foreign key to file_states

**repositories** - Repository metadata
- Name, path, scope
- Git metadata (remote, branch, commit, author)
- Auto-detected info (languages, file count, lines)
- Description, tags (JSON)
- Tracking timestamps

**repository_languages** - Language stats per repo
- Language, file count, line count, percentage

**summarization_jobs** - Hierarchical summarization tasks
- Element ID, level (hierarchy level)
- Dependencies (must complete before this runs)
- Status, worker ID, retry count
- Priority based on level

**embedding_jobs** - Flat embedding tasks
- Element ID (no dependencies)
- Status, worker ID

**parsing_jobs** - Repository-level tracking (optional)
- Repository ID, status, progress
- Files processed, elements created

**audit_log** - Event tracking
- Event type, repository, scope, details (JSON)

### Elasticsearch Indices

**magaldi_code_elements** - Main index
- All element data
- Vector embeddings (dense_vector field)
- Metadata for filtering
- Hybrid search: vector similarity + keyword/metadata

**Mapping Fields:**
- Element identification: id, scope, repository, file_path, name, type, language
- Content: raw_code, summary, docstring
- Vector: embedding (dense_vector)
- Metadata: line numbers, complexity, parent_id, level
- Searchable: scope, repository, language, type filters

---

## Hierarchical Processing Workflow

### Phase 1: Fast Sequential Parse

**Process:**
1. Repository Manager scans `/repos/` directory
2. Auto-detect repositories and languages
3. Extract git metadata from `.git/` directories
4. For each repository in each scope:
   - File Tracker checks SHA256 hashes
   - Get list of changed files only
   - Tree-sitter parses changed files
   - Extract all elements (classes, functions, variables)
   - Build parent-child relationships
   - Insert raw elements into MySQL (summary_status='pending')
   - Insert raw elements into Elasticsearch (no embeddings yet)
   - Update file_states with new hashes
   - Update file_elements mapping

**Output:** All code elements in database, ready for AI processing

### Phase 2: Hierarchical Summarization

**Level 0 - Files:**
1. Create summarization_jobs for all file-level elements
2. Workers claim jobs from MySQL (no dependencies)
3. For each file:
   - Load file content
   - Send to Ollama for summarization
   - Update code_elements.summary
   - Mark job completed
4. Wait for all Level 0 jobs to complete

**Level 1 - Classes:**
1. Create summarization_jobs for all class-level elements
2. Mark dependencies: parent file must be summarized
3. Workers claim jobs where dependencies_met=true
4. For each class:
   - Load parent file summary
   - Load class code
   - Build context prompt with file summary
   - Send to Ollama for summarization
   - Update code_elements.summary
   - Mark job completed
5. Wait for all Level 1 jobs to complete

**Level 2 - Functions/Methods:**
1. Create summarization_jobs for functions
2. Dependencies: parent file + parent class (if in class)
3. Workers claim jobs where dependencies_met=true
4. For each function:
   - Load file summary
   - Load class summary (if applicable)
   - Load function code
   - Build context prompt with parent summaries
   - Send to Ollama for summarization
   - Update code_elements.summary
   - Mark job completed
5. Wait for all Level 2 jobs to complete

**Level 3 - Variables (meaningful only):**
1. Filter: only variables with significant scope/usage
2. Create summarization_jobs for selected variables
3. Dependencies: all parent contexts (file + class + function)
4. Workers process with full context hierarchy
5. Update summaries

**Job Claiming (Atomic):**
```sql
-- Worker claims next job atomically
START TRANSACTION;
SELECT * FROM summarization_jobs
WHERE status = 'pending' 
  AND dependencies_met = TRUE
  AND level = {current_level}
ORDER BY priority DESC, created_at ASC
LIMIT 1
FOR UPDATE;

UPDATE summarization_jobs
SET status = 'running', worker_id = 'worker-X', claimed_at = NOW()
WHERE id = {job_id};

COMMIT;
```

**Dependency Resolution:**
- After each job completes, check dependent jobs
- Mark dependencies_met=TRUE if all parents done
- Dependent jobs become available for claiming

### Phase 3: Parallel Embedding Generation

**Process:**
1. Create embedding_jobs for ALL elements (no hierarchy needed)
2. Workers claim jobs (simple queue, no dependencies)
3. For each element:
   - Load element summary + metadata
   - Build embedding text: `name + summary + docstring`
   - Send to Ollama embedding model
   - Store vector in Elasticsearch
   - Update code_elements.embedding_status
   - Mark job completed

**Batching Strategy:**
- Workers collect 10-20 elements
- Send batch to Ollama for efficiency
- GPU processes batch in parallel
- Bulk update Elasticsearch

### Phase 4: Feature Clustering

**Process:**
1. Fetch all embeddings from Elasticsearch
2. Apply clustering algorithm (HDBSCAN, K-means, or hierarchical)
3. Identify clusters representing "features"
4. Label clusters (authentication, database, API, etc.)
5. Store cluster metadata in Elasticsearch
6. Generate cluster summaries

**Clustering Strategy Options:**
- **HDBSCAN**: Variable number of clusters, handles noise
- **K-means**: Fixed K clusters, fast
- **Hierarchical**: Tree structure of features
- **Semantic grouping**: Use Ollama to label clusters

---

## Parallel Processing Architecture

### Worker Pool Design

**Configuration:**
- Worker count: Environment variable `PARSER_WORKERS` (default: 4)
- Each worker: Separate process (multiprocessing)
- Worker types: Generic (handles any job type)

**Worker Process:**
```python
while running:
    job = claim_next_job()  # Atomic MySQL query
    
    if job:
        process_job(job)  # Summarization or embedding
        mark_completed(job)
    else:
        sleep(5)  # No jobs available
```

**Job Types:**
- Summarization jobs (hierarchical, level-dependent)
- Embedding jobs (flat, no dependencies)

**GPU Optimization:**
- Workers batch requests to Ollama
- Batch size: 10-20 elements
- Asynchronous: workers continue parsing while waiting for GPU
- Single Ollama instance shared by all workers

### Fault Tolerance

**Heartbeat Monitoring:**
- Workers update heartbeat timestamp every 30 seconds
- Coordinator releases jobs if heartbeat stale (>5 min)
- Automatic retry for failed jobs (max 3 attempts)

**Job Recovery:**
- If worker dies mid-job, job returns to 'pending'
- No data corruption (MySQL transactions)
- Progress tracked at job level

---

## Ollama Integration

### Model Selection (To Be Decided)

**Summarization Model Options:**
- `deepseek-coder:6.7b` - Good for code understanding
- `codellama:13b` - Larger, better context
- `qwen2.5-coder:7b` - Fast and accurate

**Embedding Model Options:**
- `nomic-embed-text` - Good for semantic search
- `mxbai-embed-large` - Higher quality embeddings
- `all-minilm` - Fast and lightweight

### Prompt Engineering

**File Summary Prompt:**
```
Summarize this {language} file in 2-3 sentences. Focus on purpose and main functionality.

File: {file_path}
Code:
{file_content}

Summary:
```

**Class Summary Prompt:**
```
Summarize this {language} class in 1-2 sentences.

File context: {file_summary}

Class: {class_name}
Code:
{class_code}

Summary:
```

**Function Summary Prompt:**
```
Summarize what this function does in one concise sentence.

File context: {file_summary}
Class context: {class_summary}  # if applicable

Function: {function_name}
Signature: {signature}
Code:
{function_code}

Summary:
```

### GPU Utilization

**Docker GPU Access:**
```yaml
ollama:
  image: ollama/ollama:latest
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: all
            capabilities: [gpu]
```

**Performance Expectations:**
- GPU: 10-50ms per element (summarization)
- GPU: 1-5ms per element (embedding)
- CPU-only: 10-20x slower

---

## MCP Server Design

### Tools to Expose

**search_codebase:**
- Semantic search across scope/repository
- Parameters: query, scope, repository, language, element_type
- Returns: Top N matching elements with summaries

**find_related_code:**
- Find similar elements via vector similarity
- Parameters: element_id, threshold, limit
- Returns: Related elements with similarity scores

**get_feature_cluster:**
- Retrieve elements in a feature cluster
- Parameters: cluster_id or cluster_name
- Returns: All elements in cluster with relationships

**explain_code_element:**
- Get detailed summary and context
- Parameters: element_id
- Returns: Element with parent context chain

**find_dependencies:**
- Trace dependencies and relationships
- Parameters: element_id, depth
- Returns: Dependency graph

**search_by_scope:**
- Search within a specific scope
- Parameters: scope_name, query, filters
- Returns: Elements filtered by scope

**get_repository_info:**
- Get repository metadata
- Parameters: repository_name or scope
- Returns: Repo metadata, stats, languages

### Subagent/Skills for Claude Code

**Autonomous exploration capabilities:**
- Discover related codebases automatically
- Suggest relevant code when user describes task
- Find similar implementations across repositories
- Identify patterns and best practices

---

## Web UI Design

### Pages & Features

**Dashboard:**
- Overview of all scopes and repositories
- Statistics (total repos, files, elements, clusters)
- Recent parsing activity
- System health (Elasticsearch, Ollama, MySQL)

**Scopes View:**
- List all scopes with repository counts
- Drill down into scope → see all repositories
- Cross-repository search within scope

**Repository Detail:**
- Repository metadata (git info, description, tags)
- Language distribution chart
- File tree browser
- Element statistics
- Feature clusters discovered in repo
- Recent changes feed
- Search within repository

**Search Page:**
- Global semantic search
- Filters: scope, repository, language, element type
- Results with summaries and links
- "Find Similar" for each result

**Visualization:**
- Vector space visualization (3D or 2D)
- Color by: repository, language, element type, cluster
- Interactive: click element → show details
- Cluster view: highlight related elements
- Export visualization as image/data

**File Browser:**
- Navigate repository file structure
- View file with parsed elements highlighted
- Show element summaries on hover
- Jump to related elements

**Feature Clusters:**
- List all detected features
- Drill down into cluster → see all elements
- Cluster graph visualization
- Cross-repository cluster view

### Technology

**Backend:**
- Flask or FastAPI
- REST API for frontend
- WebSocket for real-time updates

**Frontend:**
- Vanilla JS or React (lightweight)
- D3.js for visualizations
- Three.js for 3D vector space
- Sigma.js for graph views

---

## Configuration

### Repository Configuration

**repositories.yaml:**
```yaml
scopes:
  backend-services:
    description: "Backend microservices"
    repositories:
      - name: project-a
        path: /repos/project-a
        tags: [api, auth, microservice]
      - name: project-b
        path: /repos/project-b
        tags: [api, database, microservice]
  
  frontend:
    description: "Frontend applications"
    repositories:
      - name: web-app
        path: /repos/web-app
        tags: [react, typescript, ui]
```

**Environment Variables:**
```bash
# MySQL
MYSQL_HOST=mysql
MYSQL_PORT=3306
MYSQL_DATABASE=magaldi
MYSQL_USER=magaldi_user
MYSQL_PASSWORD=your_secure_password
MYSQL_ROOT_PASSWORD=root_secure_password

# Elasticsearch
ELASTICSEARCH_URL=http://elasticsearch:9200

# Ollama
OLLAMA_URL=http://ollama:11434
OLLAMA_SUMMARIZE_MODEL=deepseek-coder:6.7b
OLLAMA_EMBED_MODEL=nomic-embed-text

# Workers
PARSER_WORKERS=4

# Processing
BATCH_SIZE=10
MAX_RETRIES=3
```

---

## Implementation Phases

### Phase 1: Core Parsing (Week 1-2)
- Docker environment setup
- Percona MySQL + Elasticsearch + Ollama
- Tree-sitter parser for Python + JavaScript
- Repository manager with auto-detection
- File tracker with SHA256 hashing
- Basic MySQL schema
- Parse and store raw elements

### Phase 2: Hierarchical AI Processing (Week 3-4)
- Job system in MySQL
- Worker pool implementation
- Ollama integration (summarization)
- Hierarchical processing (Level 0-3)
- Dependency resolution
- Embedding generation

### Phase 3: Search & Indexing (Week 5)
- Elasticsearch schema
- Bulk indexing
- Semantic search implementation
- Filtering and aggregations

### Phase 4: MCP Server (Week 6)
- MCP server setup
- Core tools implementation
- Testing with Claude Code
- Subagent configuration

### Phase 5: Web UI (Week 7-8)
- Backend API (Flask/FastAPI)
- Frontend structure
- Dashboard and repository views
- Search interface
- Basic visualization

### Phase 6: Clustering & Features (Week 9)
- Clustering algorithm implementation
- Feature detection
- Cluster labeling
- Cluster visualization

### Phase 7: Additional Languages (Week 10)
- TypeScript, PHP, Rust parsers
- Language-specific query files
- Testing across all languages

### Phase 8: Polish & Optimization (Week 11-12)
- Performance tuning
- Error handling improvements
- Documentation
- Testing and bug fixes

---

## Testing Strategy

### Unit Tests
- Parser tests for each language
- File tracker hash verification
- Job claiming atomicity
- Dependency resolution logic

### Integration Tests
- End-to-end parse → summarize → embed → search
- Multi-repository scenarios
- Change detection and re-parsing
- Worker fault tolerance

### Performance Tests
- Large repository parsing (10K+ files)
- Concurrent worker load testing
- GPU utilization measurement
- Elasticsearch query performance

---

## Monitoring & Observability

### Metrics to Track
- Parsing throughput (files/sec)
- Summarization latency (ms/element)
- Embedding generation rate
- GPU utilization percentage
- Worker health and heartbeats
- Job queue depth by level
- Elasticsearch query latency
- MySQL connection pool usage

### Dashboard
- Real-time worker status
- Job queue visualization (pending/running/completed)
- Processing progress by level
- Error rates and failed jobs
- System resource usage

### Alerts
- Worker unresponsive (heartbeat timeout)
- Job failures exceeding threshold
- GPU unavailable
- Elasticsearch/MySQL connection issues

---

## Open Questions & Decisions Needed

### Ollama Models
- **Q:** Which specific models for summarization vs embedding?
- **Options:** deepseek-coder, codellama, qwen2.5-coder for summarization; nomic-embed-text, mxbai-embed-large for embedding
- **Decision:** Test and benchmark before finalizing

### Clustering Algorithm
- **Q:** Which clustering approach?
- **Options:** HDBSCAN (variable clusters), K-means (fixed K), hierarchical (tree structure)
- **Decision:** Start with HDBSCAN, allow configuration

### MCP Tool Granularity
- **Q:** How many tools? One generic search or multiple specialized?
- **Options:** Generic (flexible, complex params) vs Specialized (simple, many tools)
- **Decision:** Start with 5-7 core tools, expand based on usage

### Web UI Framework
- **Q:** Vanilla JS or React?
- **Options:** Vanilla (lightweight, simple) vs React (component-based, more deps)
- **Decision:** Start vanilla, migrate to React if complexity grows

### Elasticsearch vs Hybrid Storage
- **Q:** Store embeddings in ES only or also in MySQL?
- **Options:** ES only (simpler), MySQL backup (redundancy)
- **Decision:** ES primary, MySQL metadata only

### Real-time vs Batch Processing
- **Q:** Watch repositories for changes (daemon) or manual trigger?
- **Options:** Daemon (real-time, complex) vs Manual (simpler, scheduled)
- **Decision:** Manual for MVP, add watch mode later

---

## Success Criteria

### MVP Success (Phase 1-4)
- ✅ Parse 5 repositories across 3 languages
- ✅ Generate summaries for all elements hierarchically
- ✅ Create embeddings and enable semantic search
- ✅ MCP server functional with Claude Code
- ✅ Basic web UI for search and browsing

### Full Success (All Phases)
- ✅ Support all 6 languages (Python, JS, TS, PHP, Rust, Node.js)
- ✅ Process 10+ repositories with 50K+ files
- ✅ Sub-second search response times
- ✅ Feature clustering with 80%+ accuracy
- ✅ Rich web UI with visualizations
- ✅ Documentation and examples

### Performance Targets
- Parse 10K files in < 30 minutes
- Summarize 10K elements in < 2 hours (with GPU)
- Search latency < 500ms for semantic queries
- Support 4-8 concurrent workers
- 95% uptime for all services

---

## Future Enhancements

### Phase 2 Features
- Real-time file watching (daemon mode)
- Incremental updates (parse only changed functions)
- Multi-language code search
- Code similarity detection (plagiarism, duplicates)
- Cross-repository refactoring suggestions

### Advanced Features
- Code quality metrics and scoring
- Security vulnerability detection
- License compliance checking
- API documentation generation
- Code review assistant (via MCP)

### Scaling
- Multi-node Elasticsearch cluster
- MySQL replication for HA
- Redis caching layer
- Horizontal worker scaling
- Cloud deployment (AWS, GCP, Azure)

---

## Resources & Dependencies

### Python Libraries
- `tree-sitter` - Parser framework
- `tree-sitter-languages` - Pre-built grammars
- `mysql-connector-python` - MySQL client
- `elasticsearch` - ES client
- `requests` - HTTP client for Ollama
- `gitpython` - Git metadata extraction
- `flask` or `fastapi` - Web framework
- `mcp` - MCP SDK

### Docker Images
- `percona:8.0` - MySQL database
- `docker.elastic.co/elasticsearch/elasticsearch:8.11.0` - Elasticsearch
- `ollama/ollama:latest` - Ollama with GPU support
- Custom Python images for services

### External Tools
- Tree-sitter CLI (for query testing)
- Percona Monitoring and Management (optional)
- Kibana (optional, for ES visualization)

---

## Contact & Contribution

**Project Lead:** [Your Name]  
**Repository:** [GitHub URL - TBD]  
**License:** [Open Source License - TBD]  
**Documentation:** [Docs URL - TBD]

---

*This plan is a living document and will be updated as the project evolves.*
