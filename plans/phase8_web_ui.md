# Magaldi Web UI - Phase 8: Web UI

## Overview

The Web UI provides a visual interface for exploring indexed codebases, searching code semantically, browsing repository structure, and monitoring system health. It complements the MCP Server by providing a human-friendly exploration experience.

```
┌─────────────────────────────────────────────────────────────────┐
│                       PHASE 8: WEB UI                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  8.1              8.2              8.3              8.4          │
│  ─────            ─────            ─────            ─────        │
│  DASHBOARD    →   SEARCH       →   BROWSER     →   ADMIN        │
│                   INTERFACE        & EXPLORER      & MONITOR    │
│                                                                 │
│  • Overview       • Semantic       • File tree     • Parse jobs │
│  • Stats          • Filters        • Code view     • AI status  │
│  • Recent         • Results        • Summaries     • Index stats│
│                   • Preview        • Navigation                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        WEB BROWSER                               │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                  React Frontend                          │   │
│  │                                                         │   │
│  │  Pages:                                                 │   │
│  │  • Dashboard        → /                                 │   │
│  │  • Search           → /search                           │   │
│  │  • Repository       → /repo/:scope/:name                │   │
│  │  • File Browser     → /repo/:scope/:name/file/*         │   │
│  │  • Element Detail   → /element/:id                      │   │
│  │  • Admin            → /admin                            │   │
│  │                                                         │   │
│  └──────────────────────────┬──────────────────────────────┘   │
│                             │                                   │
└─────────────────────────────┼───────────────────────────────────┘
                              │ HTTP/REST
┌─────────────────────────────┼───────────────────────────────────┐
│                             ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                  FastAPI Backend                         │   │
│  │                                                         │   │
│  │  /api/v1/                                               │   │
│  │  ├── search/          → semantic + keyword search       │   │
│  │  ├── repos/           → repository listing & stats      │   │
│  │  ├── files/           → file tree & content            │   │
│  │  ├── elements/        → code element details           │   │
│  │  └── admin/           → jobs, stats, health            │   │
│  │                                                         │   │
│  └──────────────────────────┬──────────────────────────────┘   │
│                             │                                   │
│                   ┌─────────┴─────────┐                        │
│                   │                   │                        │
│                   ▼                   ▼                        │
│            ┌───────────┐       ┌───────────┐                   │
│            │   MySQL   │       │ Elastic   │                   │
│            │           │       │ search    │                   │
│            └───────────┘       └───────────┘                   │
│                                                                 │
│                     MAGALDI BACKEND                             │
└─────────────────────────────────────────────────────────────────┘
```

---

## Input

Prerequisites from earlier phases:

```python
# MySQL with parsed elements and summaries (Phases 4-5)
# Elasticsearch with embeddings (Phase 6)
# Ollama for query embedding (Phase 6)

# Same backend services as MCP Server
```

---

## 8.1 Dashboard

### Purpose

Provide an overview of indexed repositories, system health, and quick access to search.

### Dashboard Layout

```
┌─────────────────────────────────────────────────────────────────────────┐
│  🔍 Magaldi                                     [Search...]    [Admin]  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    Quick Search                                  │   │
│  │  ┌───────────────────────────────────────────────────────────┐  │   │
│  │  │ Search code semantically...                               │  │   │
│  │  └───────────────────────────────────────────────────────────┘  │   │
│  │  Examples: "authentication handler" | "database connection"     │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────┐ │
│  │    Repositories     │  │      Elements       │  │     Search      │ │
│  │        12           │  │       45,231        │  │    Queries      │ │
│  │    ↑ 2 this week    │  │   ↑ 1,234 today     │  │      847        │ │
│  └─────────────────────┘  └─────────────────────┘  └─────────────────┘ │
│                                                                         │
│  ┌─────────────────────────────────┐  ┌─────────────────────────────┐  │
│  │       Repositories             │  │      Recent Searches         │  │
│  │  ┌──────────────────────────┐  │  │                              │  │
│  │  │ backend/auth-service     │  │  │  • "user authentication"    │  │
│  │  │ 156 files • 3,421 elems  │  │  │  • "database models"        │  │
│  │  │ Python, TypeScript       │  │  │  • "API endpoints"          │  │
│  │  └──────────────────────────┘  │  │  • "error handling"         │  │
│  │  ┌──────────────────────────┐  │  │  • "rate limiting"          │  │
│  │  │ frontend/web-app         │  │  │                              │  │
│  │  │ 89 files • 1,892 elems   │  │  │                              │  │
│  │  │ TypeScript, JavaScript   │  │  │                              │  │
│  │  └──────────────────────────┘  │  │                              │  │
│  │  [View All Repositories →]     │  │                              │  │
│  └─────────────────────────────────┘  └─────────────────────────────┘  │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    System Status                                 │   │
│  │  MySQL: ✓ Connected    ES: ✓ Connected    Ollama: ✓ Ready       │   │
│  │  Parse Queue: 0        Summarize Queue: 0    Embed Queue: 0     │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### API Endpoints

```python
# GET /api/v1/dashboard
@router.get("/dashboard")
async def get_dashboard(db: MySQLConnection, es: ElasticsearchClient) -> DashboardResponse:
    """Get dashboard data"""

    # Repository count
    repo_count = await db.query_one_async(
        "SELECT COUNT(*) as count FROM repositories"
    )

    # Element count
    element_count = await db.query_one_async(
        "SELECT COUNT(*) as count FROM code_elements WHERE username = 'main'"
    )

    # Recent repos
    recent_repos = await db.query_async("""
        SELECT r.scope, r.name, r.description,
               COUNT(DISTINCT fs.relative_path) as file_count,
               COUNT(ce.element_id) as element_count,
               GROUP_CONCAT(DISTINCT rl.language) as languages
        FROM repositories r
        LEFT JOIN file_states fs ON r.scope = fs.scope AND r.name = fs.repository
        LEFT JOIN code_elements ce ON r.scope = ce.scope AND r.name = ce.repository
        LEFT JOIN repository_languages rl ON r.scope = rl.scope AND r.name = rl.repository
        WHERE fs.username = 'main'
        GROUP BY r.id
        ORDER BY r.created_at DESC
        LIMIT 5
    """)

    # Queue depths
    summarize_queue = await db.query_one_async(
        "SELECT COUNT(*) as count FROM summarization_jobs WHERE status = 'pending'"
    )
    embed_queue = await db.query_one_async(
        "SELECT COUNT(*) as count FROM embedding_jobs WHERE status = 'pending'"
    )

    # Service health
    es_health = await check_es_health(es)
    ollama_health = await check_ollama_health()

    return DashboardResponse(
        stats=DashboardStats(
            repository_count=repo_count.count,
            element_count=element_count.count,
            search_count=0,  # From analytics table if implemented
        ),
        recent_repos=[
            RepoSummary(
                scope=r.scope,
                name=r.name,
                description=r.description,
                file_count=r.file_count,
                element_count=r.element_count,
                languages=r.languages.split(",") if r.languages else []
            )
            for r in recent_repos
        ],
        queue_status=QueueStatus(
            parse_queue=0,
            summarize_queue=summarize_queue.count,
            embed_queue=embed_queue.count,
        ),
        health=HealthStatus(
            mysql=True,
            elasticsearch=es_health,
            ollama=ollama_health,
        )
    )
```

### React Component

```typescript
// src/pages/Dashboard.tsx
import { useQuery } from '@tanstack/react-query';
import { SearchBar } from '../components/SearchBar';
import { StatCard } from '../components/StatCard';
import { RepoCard } from '../components/RepoCard';
import { StatusBadge } from '../components/StatusBadge';

export function Dashboard() {
  const { data, isLoading } = useQuery({
    queryKey: ['dashboard'],
    queryFn: () => fetch('/api/v1/dashboard').then(r => r.json()),
    refetchInterval: 30000, // Refresh every 30s
  });

  if (isLoading) return <LoadingSpinner />;

  return (
    <div className="dashboard">
      <header className="dashboard-header">
        <h1>Magaldi</h1>
        <SearchBar placeholder="Search code semantically..." />
      </header>

      <section className="stats-row">
        <StatCard
          title="Repositories"
          value={data.stats.repository_count}
          icon="folder"
        />
        <StatCard
          title="Elements"
          value={data.stats.element_count}
          icon="code"
        />
        <StatCard
          title="Searches"
          value={data.stats.search_count}
          icon="search"
        />
      </section>

      <div className="dashboard-grid">
        <section className="repos-section">
          <h2>Repositories</h2>
          {data.recent_repos.map(repo => (
            <RepoCard key={`${repo.scope}/${repo.name}`} repo={repo} />
          ))}
          <Link to="/repos">View All →</Link>
        </section>

        <section className="recent-section">
          <h2>Recent Searches</h2>
          <RecentSearchList />
        </section>
      </div>

      <footer className="status-bar">
        <StatusBadge service="MySQL" healthy={data.health.mysql} />
        <StatusBadge service="Elasticsearch" healthy={data.health.elasticsearch} />
        <StatusBadge service="Ollama" healthy={data.health.ollama} />
        <span className="queue-status">
          Queues: {data.queue_status.summarize_queue + data.queue_status.embed_queue} pending
        </span>
      </footer>
    </div>
  );
}
```

---

## 8.2 Search Interface

### Purpose

Provide semantic code search with filters, preview, and navigation to results.

### Search Page Layout

```
┌─────────────────────────────────────────────────────────────────────────┐
│  🔍 Magaldi    [════════════════════════════════════════]    [Search]   │
│                 authentication handler                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Filters:  [All Scopes ▼]  [All Repos ▼]  [All Types ▼]  [All Lang ▼]  │
│                                                                         │
│  Found 23 results (0.34s)                              [Sort: Relevance]│
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ ⚡ authenticate_user                                    98.2%    │   │
│  │ method in AuthService                                           │   │
│  │ backend/auth-service • src/auth/login.py:45                     │   │
│  │                                                                 │   │
│  │ Validates provided username and password against stored         │   │
│  │ credentials. On successful validation, generates a new JWT...   │   │
│  │                                                                 │   │
│  │ def authenticate_user(self, username: str, password: str) -> .. │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ ⚡ AuthService                                          94.1%    │   │
│  │ class                                                           │   │
│  │ backend/auth-service • src/auth/login.py:15                     │   │
│  │                                                                 │   │
│  │ Core service class responsible for user authentication          │   │
│  │ operations. Manages credential verification...                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ ⚡ validate_credentials                                 91.8%    │   │
│  │ function                                                        │   │
│  │ backend/auth-service • src/auth/validators.py:23                │   │
│  │                                                                 │   │
│  │ Validates user credentials against security policies...         │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  [Load More Results...]                                                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Search API

```python
# POST /api/v1/search
@router.post("/search")
async def search(
    request: SearchRequest,
    db: MySQLConnection,
    es: ElasticsearchClient,
    ollama: OllamaEmbedClient
) -> SearchResponse:
    """Perform semantic code search"""

    # Generate query embedding
    query_embedding = await ollama.embed_async(request.query)

    # Build filters
    filters = [{"terms": {"username": ["main", request.username or "main"]}}]

    if request.scope:
        filters.append({"term": {"scope": request.scope}})
    if request.repository:
        filters.append({"term": {"repository": request.repository}})
    if request.element_types:
        filters.append({"terms": {"element_type": request.element_types}})
    if request.language:
        filters.append({"term": {"language": request.language}})

    # Execute hybrid search
    results = await es.search_async(
        index="magaldi_code_elements",
        body={
            "size": request.limit,
            "from": request.offset,
            "query": {
                "bool": {
                    "filter": filters,
                    "should": [
                        {
                            "script_score": {
                                "query": {"match_all": {}},
                                "script": {
                                    "source": "(cosineSimilarity(params.qv, 'embedding') + 1.0) * 2",
                                    "params": {"qv": query_embedding}
                                }
                            }
                        },
                        {"match": {"name": {"query": request.query, "boost": 1.5}}},
                        {"match": {"summary": {"query": request.query, "boost": 1.0}}},
                        {"match": {"docstring": {"query": request.query, "boost": 0.5}}},
                    ],
                    "minimum_should_match": 1
                }
            },
            "highlight": {
                "fields": {
                    "summary": {},
                    "docstring": {},
                    "name": {}
                },
                "pre_tags": ["<mark>"],
                "post_tags": ["</mark>"]
            },
            "_source": [
                "element_id", "name", "element_type", "relative_path",
                "line_start", "language", "summary", "signature",
                "repository", "scope", "parent_id"
            ]
        }
    )

    # Calculate max score for percentage
    max_score = results["hits"]["max_score"] or 1

    return SearchResponse(
        query=request.query,
        total=results["hits"]["total"]["value"],
        took_ms=results["took"],
        results=[
            SearchResult(
                element_id=hit["_source"]["element_id"],
                name=hit["_source"]["name"],
                element_type=hit["_source"]["element_type"],
                file_path=hit["_source"]["relative_path"],
                line=hit["_source"]["line_start"],
                language=hit["_source"]["language"],
                summary=hit["_source"].get("summary"),
                signature=hit["_source"].get("signature"),
                repository=hit["_source"]["repository"],
                scope=hit["_source"]["scope"],
                score=hit["_score"],
                relevance_pct=round((hit["_score"] / max_score) * 100, 1),
                highlights=hit.get("highlight", {})
            )
            for hit in results["hits"]["hits"]
        ]
    )


# Request/Response models
@dataclass
class SearchRequest:
    query: str
    scope: Optional[str] = None
    repository: Optional[str] = None
    username: Optional[str] = None
    element_types: Optional[List[str]] = None
    language: Optional[str] = None
    limit: int = 20
    offset: int = 0


@dataclass
class SearchResult:
    element_id: str
    name: str
    element_type: str
    file_path: str
    line: int
    language: str
    summary: Optional[str]
    signature: Optional[str]
    repository: str
    scope: str
    score: float
    relevance_pct: float
    highlights: Dict[str, List[str]]
```

### React Component

```typescript
// src/pages/Search.tsx
import { useState } from 'react';
import { useSearchParams } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';

export function SearchPage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const query = searchParams.get('q') || '';

  const [filters, setFilters] = useState({
    scope: null,
    repository: null,
    elementTypes: [],
    language: null,
  });

  const { data, isLoading, error } = useQuery({
    queryKey: ['search', query, filters],
    queryFn: () => searchCode({ query, ...filters }),
    enabled: query.length > 0,
  });

  return (
    <div className="search-page">
      <header className="search-header">
        <SearchInput
          value={query}
          onChange={(q) => setSearchParams({ q })}
          placeholder="Search code semantically..."
        />
      </header>

      <div className="search-filters">
        <ScopeFilter
          value={filters.scope}
          onChange={(scope) => setFilters(f => ({ ...f, scope }))}
        />
        <RepoFilter
          value={filters.repository}
          scope={filters.scope}
          onChange={(repository) => setFilters(f => ({ ...f, repository }))}
        />
        <TypeFilter
          value={filters.elementTypes}
          onChange={(types) => setFilters(f => ({ ...f, elementTypes: types }))}
        />
        <LanguageFilter
          value={filters.language}
          onChange={(lang) => setFilters(f => ({ ...f, language: lang }))}
        />
      </div>

      {isLoading && <LoadingSpinner />}

      {data && (
        <>
          <div className="search-meta">
            Found {data.total} results ({data.took_ms}ms)
          </div>

          <div className="search-results">
            {data.results.map(result => (
              <SearchResultCard key={result.element_id} result={result} />
            ))}
          </div>

          {data.total > data.results.length && (
            <LoadMoreButton onClick={() => loadMore()} />
          )}
        </>
      )}
    </div>
  );
}

function SearchResultCard({ result }: { result: SearchResult }) {
  return (
    <Link
      to={`/element/${encodeURIComponent(result.element_id)}`}
      className="result-card"
    >
      <div className="result-header">
        <ElementIcon type={result.element_type} />
        <span className="result-name">{result.name}</span>
        <span className="result-score">{result.relevance_pct}%</span>
      </div>

      <div className="result-meta">
        <span className="result-type">{result.element_type}</span>
        <span className="result-location">
          {result.scope}/{result.repository} • {result.file_path}:{result.line}
        </span>
      </div>

      {result.summary && (
        <p className="result-summary">{result.summary}</p>
      )}

      {result.signature && (
        <code className="result-signature">{result.signature}</code>
      )}
    </Link>
  );
}
```

---

## 8.3 Browser & Explorer

### Purpose

Browse repository structure, view file contents with summaries, and navigate code hierarchy.

### Repository Browser Layout

```
┌─────────────────────────────────────────────────────────────────────────┐
│  🔍 Magaldi  >  backend  >  auth-service                    [Search]   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Active Users: alice (3 files), bob (1 file)              [View All]   │
│                                                                         │
│  ┌─────────────────────┐  ┌─────────────────────────────────────────┐  │
│  │ 📁 File Tree        │  │  src/auth/login.py                      │  │
│  │                     │  │                                         │  │
│  │ ▼ src/              │  │  Language: Python  |  Lines: 245        │  │
│  │   ▼ auth/           │  │                                         │  │
│  │     ● login.py 👤2  │  │  ┌─────────────────────────────────┐   │  │
│  │       session.py    │  │  │ Currently working on this file: │   │  │
│  │       validators.py │  │  │ • alice (modified, 2h ago)      │   │  │
│  │   ▼ utils/          │  │  │ • bob (same as main, 1d ago)    │   │  │
│  │       helpers.py    │  │  └─────────────────────────────────┘   │  │
│  │       crypto.py 👤1 │  │                                         │  │
│  │   ▶ models/         │  │  Summary:                               │  │
│  │   ▶ api/            │  │  Handles user authentication flows      │  │
│  │ ▶ tests/            │  │  including login, logout, and session   │  │
│  │ ▶ config/           │  │  management. Provides the primary entry │  │
│  │                     │  │  points for credential validation...    │  │
│  │                     │  │                                         │  │
│  │                     │  │  ─────────────────────────────────────  │  │
│  │                     │  │                                         │  │
│  │                     │  │  Structure:                             │  │
│  │                     │  │                                         │  │
│  │                     │  │  ▼ class AuthService (line 15)          │  │
│  │                     │  │      Core service class responsible...  │  │
│  │                     │  │                                         │  │
│  │                     │  │      • authenticate_user (line 45)      │  │
│  │                     │  │        Validates username/password...   │  │
│  │                     │  │                                         │  │
│  │                     │  │      • logout (line 89)                 │  │
│  │                     │  │        Invalidates the current...       │  │
│  │                     │  │                                         │  │
│  │                     │  │      • refresh_token (line 112)         │  │
│  │                     │  │        Generates a new access token...  │  │
│  │                     │  │                                         │  │
│  │                     │  │  ▶ function hash_password (line 180)    │  │
│  │                     │  │                                         │  │
│  │                     │  │  ▶ function verify_password (line 195)  │  │
│  │                     │  │                                         │  │
│  │                     │  │  [View Source Code →]                   │  │
│  │                     │  │                                         │  │
│  └─────────────────────┘  └─────────────────────────────────────────┘  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Key UI elements for contributors:**
- **File tree badges**: `👤2` shows 2 users have active versions of this file
- **Active users bar**: Shows who's actively working in this repository
- **File detail panel**: Shows who's working on the selected file, whether their version differs from main

### File Tree API

```python
# GET /api/v1/repos/{scope}/{repository}/tree
@router.get("/repos/{scope}/{repository}/tree")
async def get_file_tree(
    scope: str,
    repository: str,
    username: str = "main",
    db: MySQLConnection = Depends()
) -> FileTreeResponse:
    """Get file tree for repository"""

    files = await db.query_async("""
        SELECT relative_path, language, summary
        FROM code_elements
        WHERE scope = %s AND repository = %s AND username = %s
          AND element_type = 'file'
        ORDER BY relative_path
    """, (scope, repository, username))

    # Build tree structure
    tree = build_tree_from_paths([f.relative_path for f in files])

    # Attach metadata
    file_map = {f.relative_path: f for f in files}
    annotate_tree(tree, file_map)

    return FileTreeResponse(
        scope=scope,
        repository=repository,
        tree=tree
    )


def build_tree_from_paths(paths: List[str]) -> TreeNode:
    """Build hierarchical tree from flat path list"""

    root = TreeNode(name="", type="directory", children={})

    for path in paths:
        parts = path.split("/")
        current = root

        for i, part in enumerate(parts):
            is_file = (i == len(parts) - 1)

            if part not in current.children:
                current.children[part] = TreeNode(
                    name=part,
                    type="file" if is_file else "directory",
                    path=path if is_file else None,
                    children={} if not is_file else None
                )

            current = current.children[part]

    return root
```

### File Contributors API

```python
# GET /api/v1/repos/{scope}/{repository}/files/{path:path}/contributors
@router.get("/repos/{scope}/{repository}/files/{path:path}/contributors")
async def get_file_contributors(
    scope: str,
    repository: str,
    path: str,
    db: MySQLConnection = Depends()
) -> FileContributorsResponse:
    """Get users with active indexed versions of this file"""

    # Get all user versions (excluding main)
    contributors = await db.query_async("""
        SELECT
            fs.username,
            fs.file_hash,
            fs.parsed_at,
            fs.expires_at,
            (SELECT file_hash FROM file_states
             WHERE scope = fs.scope AND repository = fs.repository
               AND relative_path = fs.relative_path AND username = 'main') as main_hash
        FROM file_states fs
        WHERE fs.scope = %s
          AND fs.repository = %s
          AND fs.relative_path = %s
          AND fs.username != 'main'
          AND fs.is_deleted = FALSE
          AND (fs.expires_at IS NULL OR fs.expires_at > NOW())
        ORDER BY fs.parsed_at DESC
    """, (scope, repository, path))

    return FileContributorsResponse(
        file_path=path,
        contributors=[
            Contributor(
                username=c.username,
                has_changes=c.file_hash != c.main_hash,
                last_indexed=c.parsed_at,
                expires_at=c.expires_at,
            )
            for c in contributors
        ]
    )


# GET /api/v1/repos/{scope}/{repository}/active-users
@router.get("/repos/{scope}/{repository}/active-users")
async def get_active_users(
    scope: str,
    repository: str,
    db: MySQLConnection = Depends()
) -> ActiveUsersResponse:
    """Get all users with active branches in this repository"""

    users = await db.query_async("""
        SELECT
            username,
            COUNT(DISTINCT relative_path) as file_count,
            MAX(parsed_at) as last_activity
        FROM file_states
        WHERE scope = %s AND repository = %s
          AND username != 'main'
          AND is_deleted = FALSE
          AND (expires_at IS NULL OR expires_at > NOW())
        GROUP BY username
        ORDER BY last_activity DESC
    """, (scope, repository))

    return ActiveUsersResponse(
        repository=f"{scope}/{repository}",
        users=[
            ActiveUser(
                username=u.username,
                files_modified=u.file_count,
                last_activity=u.last_activity,
            )
            for u in users
        ]
    )
```

### File Detail API

```python
# GET /api/v1/repos/{scope}/{repository}/files/{path:path}
@router.get("/repos/{scope}/{repository}/files/{path:path}")
async def get_file_detail(
    scope: str,
    repository: str,
    path: str,
    username: str = "main",
    db: MySQLConnection = Depends()
) -> FileDetailResponse:
    """Get file details with element structure"""

    # Get file element
    file_elem = await db.query_one_async("""
        SELECT * FROM code_elements
        WHERE scope = %s AND repository = %s AND username = %s
          AND relative_path = %s AND element_type = 'file'
    """, (scope, repository, username, path))

    if not file_elem:
        raise HTTPException(404, "File not found")

    # Get all elements in file
    elements = await db.query_async("""
        SELECT element_id, name, element_type, line_start, line_end,
               summary, signature, level, parent_id, docstring
        FROM code_elements
        WHERE scope = %s AND repository = %s AND username = %s
          AND relative_path = %s AND element_type != 'file'
        ORDER BY line_start
    """, (scope, repository, username, path))

    # Build nested structure
    structure = build_element_tree(elements, file_elem.element_id)

    return FileDetailResponse(
        file=FileInfo(
            path=path,
            language=file_elem.language,
            summary=file_elem.summary,
            line_count=file_elem.line_end,
        ),
        structure=structure,
        stats=ElementStats(
            classes=sum(1 for e in elements if e.element_type == "class"),
            functions=sum(1 for e in elements if e.element_type == "function"),
            methods=sum(1 for e in elements if e.element_type == "method"),
        )
    )
```

### Element Detail Page

```python
# GET /api/v1/elements/{element_id}
@router.get("/elements/{element_id}")
async def get_element_detail(
    element_id: str,
    db: MySQLConnection = Depends()
) -> ElementDetailResponse:
    """Get full element details with context"""

    element = await db.query_one_async("""
        SELECT * FROM code_elements WHERE element_id = %s
    """, (element_id,))

    if not element:
        raise HTTPException(404, "Element not found")

    # Get file context
    file_elem = await db.query_one_async("""
        SELECT element_id, name, summary FROM code_elements
        WHERE scope = %s AND repository = %s AND username = %s
          AND relative_path = %s AND element_type = 'file'
    """, (element.scope, element.repository, element.username, element.relative_path))

    # Get parent context
    parent = None
    if element.parent_id:
        parent = await db.query_one_async("""
            SELECT element_id, name, element_type, summary, signature
            FROM code_elements WHERE element_id = %s
        """, (element.parent_id,))

    # Get children
    children = await db.query_async("""
        SELECT element_id, name, element_type, line_start, summary, signature
        FROM code_elements WHERE parent_id = %s ORDER BY line_start
    """, (element_id,))

    # Get siblings
    siblings = []
    if element.parent_id:
        siblings = await db.query_async("""
            SELECT element_id, name, element_type, line_start, summary
            FROM code_elements
            WHERE parent_id = %s AND element_id != %s
            ORDER BY line_start
        """, (element.parent_id, element_id))

    return ElementDetailResponse(
        element=ElementInfo(
            id=element.element_id,
            name=element.name,
            type=element.element_type,
            file=element.relative_path,
            line_start=element.line_start,
            line_end=element.line_end,
            language=element.language,
            summary=element.summary,
            signature=element.signature,
            docstring=element.docstring,
            raw_code=element.raw_code,
            decorators=json.loads(element.decorators) if element.decorators else [],
            visibility=element.visibility,
            is_async=element.is_async,
        ),
        context=ElementContext(
            file=FileContext(
                id=file_elem.element_id,
                name=file_elem.name,
                summary=file_elem.summary,
            ) if file_elem else None,
            parent=ParentContext(
                id=parent.element_id,
                name=parent.name,
                type=parent.element_type,
                summary=parent.summary,
            ) if parent else None,
            children=[
                ChildInfo(
                    id=c.element_id,
                    name=c.name,
                    type=c.element_type,
                    line=c.line_start,
                    summary=c.summary,
                    signature=c.signature,
                )
                for c in children
            ],
            siblings=[
                SiblingInfo(
                    id=s.element_id,
                    name=s.name,
                    type=s.element_type,
                    line=s.line_start,
                    summary=s.summary,
                )
                for s in siblings
            ]
        ),
        repository=RepoRef(
            scope=element.scope,
            name=element.repository,
        )
    )
```

### React Components

```typescript
// src/pages/FileBrowser.tsx
export function FileBrowser() {
  const { scope, repository } = useParams();
  const [selectedPath, setSelectedPath] = useState<string | null>(null);

  const { data: tree } = useQuery({
    queryKey: ['fileTree', scope, repository],
    queryFn: () => getFileTree(scope!, repository!),
  });

  const { data: fileDetail } = useQuery({
    queryKey: ['fileDetail', scope, repository, selectedPath],
    queryFn: () => getFileDetail(scope!, repository!, selectedPath!),
    enabled: !!selectedPath,
  });

  return (
    <div className="file-browser">
      <Breadcrumb items={[scope, repository, selectedPath]} />

      <div className="browser-layout">
        <aside className="file-tree-panel">
          <FileTree
            tree={tree}
            selectedPath={selectedPath}
            onSelect={setSelectedPath}
          />
        </aside>

        <main className="file-detail-panel">
          {fileDetail ? (
            <FileDetailView file={fileDetail} />
          ) : (
            <RepoOverview scope={scope!} repository={repository!} />
          )}
        </main>
      </div>
    </div>
  );
}

// src/components/FileDetailView.tsx
function FileDetailView({ file }: { file: FileDetail }) {
  return (
    <div className="file-detail">
      <header className="file-header">
        <h2>{file.file.path}</h2>
        <div className="file-meta">
          <LanguageBadge language={file.file.language} />
          <span>{file.file.line_count} lines</span>
        </div>
      </header>

      {file.file.summary && (
        <section className="file-summary">
          <h3>Summary</h3>
          <p>{file.file.summary}</p>
        </section>
      )}

      <section className="file-structure">
        <h3>Structure</h3>
        <ElementTree elements={file.structure} />
      </section>

      <footer className="file-actions">
        <Link to={`/source/${encodeURIComponent(file.file.path)}`}>
          View Source Code →
        </Link>
      </footer>
    </div>
  );
}
```

---

## 8.4 Vector Space Visualization

### Purpose

Visualize the semantic embedding space to understand code relationships, find clusters, and explore how elements relate to each other.

### Visualization Approach

```
┌─────────────────────────────────────────────────────────────────┐
│                 DIMENSIONALITY REDUCTION                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1024-dim embeddings                                            │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────────┐                                           │
│  │  UMAP / t-SNE   │  (computed on backend, cached)            │
│  │  reduction      │                                           │
│  └────────┬────────┘                                           │
│           │                                                     │
│           ▼                                                     │
│      2D or 3D coordinates                                       │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────┐                                           │
│  │    D3.js        │  (interactive scatter/force graph)        │
│  │  visualization  │                                           │
│  └─────────────────┘                                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Vector Map Page Layout

```
┌─────────────────────────────────────────────────────────────────────────┐
│  🔍 Magaldi  >  Vector Map                                   [Search]   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Scope: [backend ▼]  Repo: [auth-service ▼]  Color by: [type ▼]        │
│  Show: [✓] Classes  [✓] Functions  [✓] Methods  [ ] Files              │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                                                                 │   │
│  │           ○ ○                                                   │   │
│  │        ○ ○ ○ ○      Authentication                              │   │
│  │          ○ ○        cluster                                     │   │
│  │                                     ○                           │   │
│  │                                   ○ ○ ○    Database              │   │
│  │    ○                               ○ ○     cluster               │   │
│  │   ○ ○  Validation                    ○                          │   │
│  │    ○   cluster                                                  │   │
│  │                           ○ ○                                   │   │
│  │                          ○ ○ ○ ○   API handlers                 │   │
│  │                           ○ ○                                   │   │
│  │        ●←─ Selected: authenticate_user                          │   │
│  │                                                                 │   │
│  │  [Zoom +] [Zoom -] [Reset] [3D Toggle]          Legend:        │   │
│  │                                            ○ class  ○ function │   │
│  │                                            ○ method ○ file     │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Selected: authenticate_user (method)                           │   │
│  │  File: src/auth/login.py:45                                     │   │
│  │  Summary: Validates provided username and password...           │   │
│  │                                                                 │   │
│  │  Nearest neighbors:                                             │   │
│  │  • validate_credentials (0.92 similarity)                       │   │
│  │  • check_password (0.89 similarity)                             │   │
│  │  • login_user (0.87 similarity)                                 │   │
│  │                                                     [View Code] │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Backend API for Vector Coordinates

```python
# GET /api/v1/repos/{scope}/{repository}/vector-map
@router.get("/repos/{scope}/{repository}/vector-map")
async def get_vector_map(
    scope: str,
    repository: str,
    username: str = "main",
    element_types: List[str] = Query(default=["class", "function", "method"]),
    dimensions: int = 2,  # 2 or 3
    algorithm: str = "umap",  # "umap" or "tsne"
    db: MySQLConnection = Depends(),
    es: ElasticsearchClient = Depends(),
    cache: Cache = Depends()
) -> VectorMapResponse:
    """
    Get 2D/3D coordinates for visualizing embedding space.

    Coordinates are cached and recomputed when embeddings change.
    """

    cache_key = f"vector_map:{scope}:{repository}:{username}:{'-'.join(sorted(element_types))}:{dimensions}:{algorithm}"

    # Check cache
    cached = await cache.get(cache_key)
    if cached:
        return VectorMapResponse(**cached)

    # Fetch embeddings from Elasticsearch
    elements = await es.search_async(
        index="magaldi_code_elements",
        body={
            "size": 5000,  # Limit for performance
            "query": {
                "bool": {
                    "filter": [
                        {"term": {"scope": scope}},
                        {"term": {"repository": repository}},
                        {"terms": {"username": ["main", username]}},
                        {"terms": {"element_type": element_types}},
                        {"exists": {"field": "embedding"}}
                    ]
                }
            },
            "_source": ["element_id", "name", "element_type", "relative_path",
                       "line_start", "summary", "embedding"]
        },
        scroll="2m"
    )

    # Extract embeddings and metadata
    embeddings = []
    metadata = []

    for hit in elements["hits"]["hits"]:
        src = hit["_source"]
        embeddings.append(src["embedding"])
        metadata.append({
            "id": src["element_id"],
            "name": src["name"],
            "type": src["element_type"],
            "file": src["relative_path"],
            "line": src["line_start"],
            "summary": src.get("summary", "")[:100],
        })

    if not embeddings:
        return VectorMapResponse(points=[], metadata=[])

    # Reduce dimensions
    coords = reduce_dimensions(
        embeddings,
        n_components=dimensions,
        algorithm=algorithm
    )

    # Build response
    points = [
        VectorPoint(
            x=float(coords[i][0]),
            y=float(coords[i][1]),
            z=float(coords[i][2]) if dimensions == 3 else None,
            **metadata[i]
        )
        for i in range(len(coords))
    ]

    response = VectorMapResponse(
        points=points,
        bounds={
            "x": [float(coords[:, 0].min()), float(coords[:, 0].max())],
            "y": [float(coords[:, 1].min()), float(coords[:, 1].max())],
            "z": [float(coords[:, 2].min()), float(coords[:, 2].max())] if dimensions == 3 else None,
        },
        algorithm=algorithm,
        dimensions=dimensions,
        element_count=len(points)
    )

    # Cache for 1 hour
    await cache.set(cache_key, response.dict(), ttl=3600)

    return response


def reduce_dimensions(
    embeddings: List[List[float]],
    n_components: int = 2,
    algorithm: str = "umap"
) -> np.ndarray:
    """Reduce high-dimensional embeddings to 2D/3D for visualization."""

    import numpy as np

    X = np.array(embeddings)

    if algorithm == "umap":
        import umap
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=15,
            min_dist=0.1,
            metric="cosine",
            random_state=42
        )
    elif algorithm == "tsne":
        from sklearn.manifold import TSNE
        reducer = TSNE(
            n_components=n_components,
            perplexity=30,
            metric="cosine",
            random_state=42
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    return reducer.fit_transform(X)


@dataclass
class VectorPoint:
    x: float
    y: float
    z: Optional[float]
    id: str
    name: str
    type: str
    file: str
    line: int
    summary: str


@dataclass
class VectorMapResponse:
    points: List[VectorPoint]
    bounds: dict
    algorithm: str
    dimensions: int
    element_count: int
```

### D3.js Visualization Component

```html
<!-- templates/vector-map.html -->
<div id="vector-map-container">
    <svg id="vector-map"></svg>
    <div id="vector-tooltip" class="tooltip"></div>
</div>

<script src="https://d3js.org/d3.v7.min.js"></script>
<script>
class VectorMapVisualization {
    constructor(containerId, options = {}) {
        this.container = d3.select(containerId);
        this.svg = this.container.select('svg');
        this.tooltip = this.container.select('.tooltip');

        this.width = options.width || 800;
        this.height = options.height || 600;
        this.colorScale = d3.scaleOrdinal()
            .domain(['class', 'function', 'method', 'file'])
            .range(['#4e79a7', '#f28e2c', '#e15759', '#76b7b2']);

        this.zoom = d3.zoom()
            .scaleExtent([0.5, 10])
            .on('zoom', (event) => this.handleZoom(event));

        this.svg
            .attr('width', this.width)
            .attr('height', this.height)
            .call(this.zoom);

        this.g = this.svg.append('g');
        this.selectedPoint = null;
        this.onSelect = options.onSelect || (() => {});
    }

    async loadData(scope, repository, options = {}) {
        const params = new URLSearchParams({
            element_types: options.types?.join(',') || 'class,function,method',
            dimensions: options.dimensions || 2,
            algorithm: options.algorithm || 'umap'
        });

        const response = await fetch(
            `/api/v1/repos/${scope}/${repository}/vector-map?${params}`
        );
        this.data = await response.json();
        this.render();
    }

    render() {
        const { points, bounds } = this.data;

        // Create scales
        this.xScale = d3.scaleLinear()
            .domain(bounds.x)
            .range([50, this.width - 50]);

        this.yScale = d3.scaleLinear()
            .domain(bounds.y)
            .range([this.height - 50, 50]);

        // Clear previous
        this.g.selectAll('*').remove();

        // Draw points
        const circles = this.g.selectAll('circle')
            .data(points)
            .enter()
            .append('circle')
            .attr('cx', d => this.xScale(d.x))
            .attr('cy', d => this.yScale(d.y))
            .attr('r', 6)
            .attr('fill', d => this.colorScale(d.type))
            .attr('stroke', '#fff')
            .attr('stroke-width', 1)
            .attr('opacity', 0.7)
            .attr('cursor', 'pointer')
            .on('mouseover', (event, d) => this.showTooltip(event, d))
            .on('mouseout', () => this.hideTooltip())
            .on('click', (event, d) => this.selectPoint(d));

        // Animate entrance
        circles
            .attr('r', 0)
            .transition()
            .duration(500)
            .attr('r', 6);
    }

    showTooltip(event, d) {
        this.tooltip
            .style('display', 'block')
            .style('left', (event.pageX + 10) + 'px')
            .style('top', (event.pageY - 10) + 'px')
            .html(`
                <strong>${d.name}</strong><br>
                <span class="badge bg-secondary">${d.type}</span><br>
                <small>${d.file}:${d.line}</small><br>
                <em>${d.summary || 'No summary'}</em>
            `);
    }

    hideTooltip() {
        this.tooltip.style('display', 'none');
    }

    selectPoint(point) {
        // Highlight selected
        this.g.selectAll('circle')
            .attr('stroke', d => d.id === point.id ? '#000' : '#fff')
            .attr('stroke-width', d => d.id === point.id ? 3 : 1)
            .attr('r', d => d.id === point.id ? 10 : 6);

        this.selectedPoint = point;
        this.onSelect(point);
    }

    handleZoom(event) {
        this.g.attr('transform', event.transform);
    }

    highlightNeighbors(elementId, neighborIds) {
        this.g.selectAll('circle')
            .attr('opacity', d => {
                if (d.id === elementId) return 1;
                if (neighborIds.includes(d.id)) return 0.9;
                return 0.2;
            });
    }

    resetHighlight() {
        this.g.selectAll('circle').attr('opacity', 0.7);
    }

    filterByType(types) {
        this.g.selectAll('circle')
            .attr('display', d => types.includes(d.type) ? 'block' : 'none');
    }
}

// Usage
const vectorMap = new VectorMapVisualization('#vector-map-container', {
    width: 900,
    height: 600,
    onSelect: async (point) => {
        // Load neighbors and show in panel
        const neighbors = await fetch(`/api/v1/elements/${point.id}/similar?limit=5`);
        const data = await neighbors.json();
        showNeighborPanel(point, data);
        vectorMap.highlightNeighbors(point.id, data.map(n => n.element_id));
    }
});

// Load data
vectorMap.loadData('backend', 'auth-service', {
    types: ['class', 'function', 'method'],
    algorithm: 'umap'
});
</script>

<style>
#vector-map-container {
    position: relative;
    border: 1px solid #ddd;
    border-radius: 8px;
    overflow: hidden;
}

#vector-map {
    background: #fafafa;
}

.tooltip {
    position: absolute;
    display: none;
    background: white;
    border: 1px solid #ccc;
    border-radius: 4px;
    padding: 8px 12px;
    font-size: 12px;
    max-width: 300px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    z-index: 1000;
}

.tooltip strong {
    font-size: 14px;
}

.tooltip em {
    color: #666;
    display: block;
    margin-top: 4px;
}
</style>
```

### 3D Visualization Option (Three.js)

```javascript
// Optional: 3D visualization using Three.js
class VectorMap3D {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.scene = new THREE.Scene();
        this.camera = new THREE.PerspectiveCamera(75, 800/600, 0.1, 1000);
        this.renderer = new THREE.WebGLRenderer({ antialias: true });

        this.renderer.setSize(800, 600);
        this.container.appendChild(this.renderer.domElement);

        this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
        this.camera.position.z = 50;

        this.raycaster = new THREE.Raycaster();
        this.mouse = new THREE.Vector2();

        this.animate();
    }

    loadData(points) {
        const geometry = new THREE.BufferGeometry();
        const positions = [];
        const colors = [];

        const colorMap = {
            'class': new THREE.Color(0x4e79a7),
            'function': new THREE.Color(0xf28e2c),
            'method': new THREE.Color(0xe15759),
            'file': new THREE.Color(0x76b7b2)
        };

        points.forEach(p => {
            positions.push(p.x * 10, p.y * 10, p.z * 10);
            const color = colorMap[p.type] || new THREE.Color(0x999999);
            colors.push(color.r, color.g, color.b);
        });

        geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
        geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));

        const material = new THREE.PointsMaterial({
            size: 0.5,
            vertexColors: true,
            transparent: true,
            opacity: 0.8
        });

        this.pointCloud = new THREE.Points(geometry, material);
        this.scene.add(this.pointCloud);
        this.pointsData = points;
    }

    animate() {
        requestAnimationFrame(() => this.animate());
        this.controls.update();
        this.renderer.render(this.scene, this.camera);
    }
}
```

### Cluster Detection API

```python
# GET /api/v1/repos/{scope}/{repository}/clusters
@router.get("/repos/{scope}/{repository}/clusters")
async def get_clusters(
    scope: str,
    repository: str,
    n_clusters: int = 10,
    db: MySQLConnection = Depends(),
    es: ElasticsearchClient = Depends()
) -> ClustersResponse:
    """
    Identify semantic clusters in the codebase.

    Uses K-means on embeddings to find groups of related code.
    """
    from sklearn.cluster import KMeans

    # Fetch embeddings
    elements = await fetch_embeddings(scope, repository, es)

    if len(elements) < n_clusters:
        n_clusters = max(2, len(elements) // 2)

    embeddings = np.array([e["embedding"] for e in elements])

    # Cluster
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)

    # Group elements by cluster
    clusters = {}
    for i, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(elements[i])

    # Find representative element for each cluster (closest to centroid)
    result = []
    for label, members in clusters.items():
        centroid = kmeans.cluster_centers_[label]
        member_embeddings = np.array([m["embedding"] for m in members])
        distances = np.linalg.norm(member_embeddings - centroid, axis=1)
        representative_idx = np.argmin(distances)
        representative = members[representative_idx]

        # Infer cluster theme from representative
        result.append({
            "cluster_id": int(label),
            "size": len(members),
            "representative": {
                "name": representative["name"],
                "type": representative["type"],
                "file": representative["file"],
                "summary": representative.get("summary", ""),
            },
            "members": [
                {"id": m["id"], "name": m["name"], "type": m["type"]}
                for m in members[:20]  # Limit to 20 per cluster
            ]
        })

    return ClustersResponse(
        clusters=sorted(result, key=lambda x: -x["size"]),
        total_elements=len(elements)
    )
```

---

## 8.5 Admin & Monitor

### Purpose

Monitor system health, view job queues, manage indexes, and troubleshoot issues.

### Admin Dashboard Layout

```
┌─────────────────────────────────────────────────────────────────────────┐
│  🔍 Magaldi Admin                                            [Logout]   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  [Overview]  [Jobs]  [Indexes]  [Settings]                              │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    System Health                                 │   │
│  │                                                                 │   │
│  │  MySQL          Elasticsearch      Ollama                       │   │
│  │  ● Connected    ● Connected        ● Ready                      │   │
│  │  23ms latency   45ms latency       qwen2.5-coder:7b loaded      │   │
│  │                                    snowflake-arctic-embed2       │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    Job Queues                                    │   │
│  │                                                                 │   │
│  │  Summarization                     Embedding                    │   │
│  │  ┌─────────────────────────┐      ┌─────────────────────────┐  │   │
│  │  │ Pending:     45         │      │ Pending:     0          │  │   │
│  │  │ Running:     4          │      │ Running:     0          │  │   │
│  │  │ Completed:   12,456     │      │ Completed:   12,411     │  │   │
│  │  │ Failed:      12         │      │ Failed:      3          │  │   │
│  │  │                         │      │                         │  │   │
│  │  │ [Retry Failed] [Pause]  │      │ [Retry Failed]          │  │   │
│  │  └─────────────────────────┘      └─────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    Recent Activity                               │   │
│  │                                                                 │   │
│  │  12:45:23  Completed embedding batch (20 elements)              │   │
│  │  12:45:21  Summarized AuthService.authenticate_user             │   │
│  │  12:45:18  Parsed backend/auth-service (12 files changed)       │   │
│  │  12:44:55  Started summarization for backend/auth-service       │   │
│  │  12:44:30  User 'alice' triggered parse                         │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    Index Statistics                              │   │
│  │                                                                 │   │
│  │  Elasticsearch Index: magaldi_code_elements                     │   │
│  │  • Documents:   45,231                                          │   │
│  │  • Size:        892 MB                                          │   │
│  │  • With vectors: 44,892 (99.2%)                                 │   │
│  │                                                                 │   │
│  │  [Refresh Index]  [Rebuild Vectors]  [Clear Cache]              │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Admin API Endpoints

```python
# GET /api/v1/admin/health
@router.get("/admin/health")
async def get_health(
    db: MySQLConnection,
    es: ElasticsearchClient
) -> HealthResponse:
    """Get detailed health status"""

    # MySQL health
    mysql_start = time.time()
    await db.query_one_async("SELECT 1")
    mysql_latency = (time.time() - mysql_start) * 1000

    # ES health
    es_start = time.time()
    es_health = await es.cluster.health_async()
    es_latency = (time.time() - es_start) * 1000

    # Ollama health
    ollama_status = await check_ollama_status()

    return HealthResponse(
        mysql=ServiceHealth(
            status="healthy",
            latency_ms=mysql_latency,
        ),
        elasticsearch=ServiceHealth(
            status=es_health["status"],
            latency_ms=es_latency,
            details={
                "cluster_name": es_health["cluster_name"],
                "number_of_nodes": es_health["number_of_nodes"],
            }
        ),
        ollama=ServiceHealth(
            status="healthy" if ollama_status["running"] else "unhealthy",
            details={
                "models_loaded": ollama_status.get("models", [])
            }
        )
    )


# GET /api/v1/admin/jobs
@router.get("/admin/jobs")
async def get_job_stats(db: MySQLConnection) -> JobStatsResponse:
    """Get job queue statistics"""

    summarize_stats = await db.query_one_async("""
        SELECT
            SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) as pending,
            SUM(CASE WHEN status = 'running' THEN 1 ELSE 0 END) as running,
            SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed,
            SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed
        FROM summarization_jobs
    """)

    embed_stats = await db.query_one_async("""
        SELECT
            SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) as pending,
            SUM(CASE WHEN status = 'running' THEN 1 ELSE 0 END) as running,
            SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed,
            SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed
        FROM embedding_jobs
    """)

    return JobStatsResponse(
        summarization=QueueStats(
            pending=summarize_stats.pending,
            running=summarize_stats.running,
            completed=summarize_stats.completed,
            failed=summarize_stats.failed,
        ),
        embedding=QueueStats(
            pending=embed_stats.pending,
            running=embed_stats.running,
            completed=embed_stats.completed,
            failed=embed_stats.failed,
        )
    )


# POST /api/v1/admin/jobs/retry
@router.post("/admin/jobs/retry")
async def retry_failed_jobs(
    job_type: Literal["summarization", "embedding"],
    db: MySQLConnection
) -> RetryResponse:
    """Retry failed jobs"""

    table = f"{job_type}_jobs"

    result = await db.execute_async(f"""
        UPDATE {table}
        SET status = 'pending',
            retry_count = 0,
            error_message = NULL,
            worker_id = NULL
        WHERE status = 'failed'
    """)

    return RetryResponse(jobs_reset=result.rowcount)


# GET /api/v1/admin/index-stats
@router.get("/admin/index-stats")
async def get_index_stats(es: ElasticsearchClient) -> IndexStatsResponse:
    """Get Elasticsearch index statistics"""

    stats = await es.indices.stats_async(index="magaldi_code_elements")
    index_stats = stats["indices"]["magaldi_code_elements"]

    # Count documents with embeddings
    with_vectors = await es.count_async(
        index="magaldi_code_elements",
        body={"query": {"exists": {"field": "embedding"}}}
    )

    total_docs = index_stats["primaries"]["docs"]["count"]

    return IndexStatsResponse(
        index_name="magaldi_code_elements",
        document_count=total_docs,
        size_bytes=index_stats["primaries"]["store"]["size_in_bytes"],
        with_vectors=with_vectors["count"],
        vector_coverage_pct=round((with_vectors["count"] / total_docs) * 100, 1) if total_docs > 0 else 0,
    )
```

---

## Output

Phase 8 produces:

```python
# Web application:
# - Frontend: React SPA
# - Backend: FastAPI REST API
# - Serves on configurable port (default 8080)

# Pages:
# - / (Dashboard)
# - /search
# - /repos
# - /repo/:scope/:repository
# - /repo/:scope/:repository/file/*
# - /element/:id
# - /admin

# API endpoints:
# - GET  /api/v1/dashboard
# - POST /api/v1/search
# - GET  /api/v1/repos
# - GET  /api/v1/repos/:scope/:repository/tree
# - GET  /api/v1/repos/:scope/:repository/files/:path
# - GET  /api/v1/elements/:id
# - GET  /api/v1/admin/health
# - GET  /api/v1/admin/jobs
# - POST /api/v1/admin/jobs/retry
# - GET  /api/v1/admin/index-stats
```

---

## Progress Reporting

```
[Web UI]
Starting Magaldi Web UI...

Backend:
  FastAPI server starting...          ✓
  Connecting to MySQL...              ✓
  Connecting to Elasticsearch...      ✓
  Connecting to Ollama...             ✓

Frontend:
  Building React app...               ✓
  Serving static files...             ✓

Server ready at http://localhost:8080

Routes:
  • /                   Dashboard
  • /search             Semantic search
  • /repos              Repository list
  • /repo/:s/:r         Repository browser
  • /element/:id        Element detail
  • /admin              Admin panel
```

---

## Error Handling

| Error | Action |
|-------|--------|
| Database unavailable | Show error banner, retry connection |
| Elasticsearch unavailable | Disable search, show warning |
| Ollama unavailable | Fall back to keyword-only search |
| Element not found | 404 page with suggestions |
| Search timeout | Return partial results with warning |
| Invalid filters | Clear filters, show validation error |

---

## Performance Considerations

| Operation | Bottleneck | Optimization |
|-----------|------------|--------------|
| Search | Query embedding | Cache recent queries |
| File tree | Large repos | Lazy load subdirectories |
| Element detail | Multiple queries | Eager load context |
| Dashboard | Aggregate queries | Cache stats (30s TTL) |

### Caching Strategy

```python
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from fastapi_cache.decorator import cache

# Dashboard stats (30 second cache)
@router.get("/dashboard")
@cache(expire=30)
async def get_dashboard(...):
    ...

# File tree (5 minute cache per repo)
@router.get("/repos/{scope}/{repository}/tree")
@cache(expire=300)
async def get_file_tree(...):
    ...

# Search (no cache - always fresh)
@router.post("/search")
async def search(...):
    ...
```

---

## CLI Interface

```bash
# Start web server
magaldi web serve --port 8080

# Start with hot reload (development)
magaldi web serve --reload

# Build frontend only
magaldi web build

# Health check
magaldi web health
```

---

## Docker Deployment

```yaml
# docker-compose.yml
services:
  magaldi-web:
    image: magaldi/web:latest
    ports:
      - "8080:8080"
    environment:
      - MYSQL_HOST=mysql
      - MYSQL_PASSWORD=${MYSQL_PASSWORD}
      - ELASTICSEARCH_URL=http://elasticsearch:9200
      - OLLAMA_URL=http://ollama:11434
    depends_on:
      - mysql
      - elasticsearch
      - ollama
```

---

## Summary of Decisions

| Decision | Value |
|----------|-------|
| Frontend framework | Vanilla JS + Bootstrap 5 |
| Backend framework | FastAPI (Python) |
| Styling | Bootstrap CSS |
| API format | REST (JSON) |
| Caching | Redis (optional) |
| Authentication | None (internal tool) |
| Default port | 8080 |
| Search | Hybrid (semantic + keyword) |
| File tree | Lazy-loaded |
| Stats caching | 30 seconds |
