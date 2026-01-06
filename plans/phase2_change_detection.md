# Magaldi Parser - Phase 2: Change Detection

## Overview

The Change Detection phase identifies which files have changed since the last parse, avoiding redundant processing. It handles both `main` branch (full parse) and user branches (diff from main).

```
┌─────────────────────────────────────────────────────────────────┐
│                   PHASE 2: CHANGE DETECTION                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  2.1              2.2              2.3              2.4          │
│  ─────            ─────            ─────            ─────        │
│  FILE         →   HASH         →   COMPARE      →   MANIFEST    │
│  ENUMERATION      COMPUTATION      LOGIC            GENERATION   │
│                                                                 │
│  • Walk dirs      • SHA256         • Main vs DB     • New       │
│  • Apply filters  • Stream large   • User vs Main   • Modified  │
│  • Skip excluded  • Parallel       • Detect deletes • Deleted   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2.1 File Enumeration

### Purpose
Build list of all parseable files in repository.

### Input
```python
{
    "repo_path": "/path/to/project-a",
    "exclude_directories": ["node_modules", "__pycache__", ...],
    "exclude_files": ["*.min.js", "*.lock", ...]
}
```

### Process

```
1. Walk directory tree recursively starting from repo_path
2. For each entry:
   a. Skip if symlink
   b. Skip if hidden (starts with .)
   c. Skip if directory matches exclude_directories
   d. Skip if file matches exclude_files patterns
   e. Skip if extension not in SUPPORTED_EXTENSIONS
   f. Add to file list
```

### Supported Extensions

```python
SUPPORTED_EXTENSIONS = {
    '.py': 'python',
    '.js': 'javascript',
    '.jsx': 'javascript',
    '.ts': 'typescript',
    '.tsx': 'typescript',
    '.php': 'php',
    '.rs': 'rust',
    '.mjs': 'javascript',
    '.cjs': 'javascript'
}
```

### Output

```python
[
    {
        "relative_path": "src/auth/login.py",
        "absolute_path": "/path/to/project-a/src/auth/login.py",
        "language": "python"
    },
    {
        "relative_path": "src/utils/helpers.js",
        "absolute_path": "/path/to/project-a/src/utils/helpers.js",
        "language": "javascript"
    },
    ...
]
```

### Pattern Matching

| Pattern | Matches |
|---------|---------|
| `node_modules` | Any directory named `node_modules` at any depth |
| `*.min.js` | Any file ending in `.min.js` |
| `*.lock` | `package-lock.json`, `yarn.lock`, etc. |
| `test_*.py` | Files starting with `test_` (if configured) |

---

## 2.2 Hash Computation

### Purpose
Generate SHA256 hash for each file to detect changes.

### Implementation

```python
def compute_hash(file_path: str) -> str:
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        # Stream in chunks for large files
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()
```

### Parallel Processing
- Use thread pool (I/O bound, not CPU bound)
- Default: 4-8 threads (configurable)
- Collect results as they complete

### Output

```python
[
    {
        "relative_path": "src/auth/login.py",
        "absolute_path": "/path/to/project-a/src/auth/login.py",
        "language": "python",
        "hash": "a1b2c3d4e5f6..."
    },
    ...
]
```

### Edge Cases

| Case | Handling |
|------|----------|
| File unreadable | Log error, skip file, continue |
| File deleted mid-scan | Log warning, skip file |
| Binary file | Still hash it, will fail at parse phase |
| Empty file | Valid hash, will produce no elements |

---

## 2.3 Compare Logic

### 2.3.1 Main Branch Comparison

```
┌─────────────────────────────────────────────────────────────────┐
│                 MAIN BRANCH COMPARISON                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  For each local file:                                           │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Query: SELECT file_hash FROM file_states                │   │
│  │        WHERE scope=? AND repository=? AND username='main'│   │
│  │        AND relative_path=?                              │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  Result:                                                        │
│  • No record         → NEW                                      │
│  • Hash matches      → UNCHANGED                                │
│  • Hash differs      → MODIFIED                                 │
│                                                                 │
│  For deletions:                                                 │
│  • Files in DB but not on disk → DELETED                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Status Matrix:**

| Local File | DB Record | Status |
|------------|-----------|--------|
| Exists | None | NEW |
| Exists | Hash matches | UNCHANGED |
| Exists | Hash differs | MODIFIED |
| Missing | Exists | DELETED |

### 2.3.2 User Branch Comparison

```
┌─────────────────────────────────────────────────────────────────┐
│                 USER BRANCH COMPARISON                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Prerequisite: main branch must exist                           │
│  → If not: ERROR "Run 'magaldi parse --user main' first"        │
│                                                                 │
│  For each local file:                                           │
│                                                                 │
│  Step 1: Compare to main                                        │
│  • Hash matches main → SKIP (no diff, use main's data)          │
│  • Hash differs from main → Continue to step 2                  │
│  • Not in main → NEW file (user added)                          │
│                                                                 │
│  Step 2: Compare to previous user parse (if differs from main)  │
│  • No record         → NEW (for this user)                      │
│  • Hash matches      → UNCHANGED (already parsed by user)       │
│  • Hash differs      → MODIFIED (user changed again)            │
│                                                                 │
│  For deletions (files in main but not on user's disk):          │
│  • Check if deletion marker already exists                      │
│  • If not → mark as DELETED                                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Status Matrix:**

| Local File | Main Record | User Record | Status |
|------------|-------------|-------------|--------|
| Exists, hash = main | Any | Any | SKIP (use main) |
| Exists, hash ≠ main | Exists | None | NEW (for user) |
| Exists, hash ≠ main | Exists | Hash matches | UNCHANGED |
| Exists, hash ≠ main | Exists | Hash differs | MODIFIED |
| Exists | None | None | NEW (user added file) |
| Missing | Exists | None | DELETED (new marker) |
| Missing | Exists | is_deleted=true | UNCHANGED (already marked) |

### Batch Query Strategy

Fetch all DB records at once instead of per-file queries:

```python
# GOOD: 1 query
db_files = query("""
    SELECT relative_path, file_hash 
    FROM file_states 
    WHERE scope = ? AND repository = ? AND username = ?
""")
db_map = {row.relative_path: row.file_hash for row in db_files}

for file in files:
    db_hash = db_map.get(file.relative_path)
    # Compare locally
```

---

## 2.4 Deletion Detection

### Pagination for Large Repos

```sql
-- Paginated query for DB files
SELECT relative_path, file_hash
FROM file_states
WHERE scope = ?
  AND repository = ?
  AND username = ?
ORDER BY relative_path
LIMIT 1000 OFFSET ?
```

### Algorithm

```python
local_files = Set(all local file paths)  # Keep in memory (just paths)
deleted_files = []

offset = 0
page_size = 1000

while True:
    db_page = query(... LIMIT page_size OFFSET offset)
    
    if not db_page:
        break
    
    for db_file in db_page:
        if db_file.relative_path not in local_files:
            deleted_files.append(db_file)
    
    offset += page_size

return deleted_files
```

### User Branch Deletion Query

```sql
-- Find files in main that don't exist locally AND aren't already marked deleted
SELECT m.relative_path, m.file_hash
FROM file_states m
LEFT JOIN file_states u 
    ON u.scope = m.scope 
    AND u.repository = m.repository 
    AND u.username = ?
    AND u.relative_path = m.relative_path
WHERE m.scope = ?
  AND m.repository = ?
  AND m.username = 'main'
  AND m.relative_path NOT IN (... local files ...)
  AND (u.id IS NULL OR u.is_deleted = FALSE)
```

---

## 2.5 Manifest Generation

### Output Structure

```python
@dataclass
class ChangeManifest:
    # Context
    scope: str
    repository: str
    username: str
    timestamp: datetime
    
    # Counts
    total_files_scanned: int
    
    # File lists
    new_files: List[FileInfo]
    modified_files: List[FileInfo]
    deleted_files: List[FileInfo]
    unchanged_files: int           # Count only, not full list
    skipped_files: int             # User branch: same as main
    
    # Summary
    files_to_parse: int            # new + modified
    files_to_remove: int           # deleted
    
@dataclass
class FileInfo:
    relative_path: str
    absolute_path: str
    language: str
    hash: str
    previous_hash: Optional[str]   # For modified files
```

### Example Manifest

```python
{
    "scope": "backend-services",
    "repository": "project-a",
    "username": "alice",
    "timestamp": "2024-01-15T10:30:00Z",
    
    "total_files_scanned": 150,
    
    "new_files": [
        {
            "relative_path": "src/new_feature.py",
            "absolute_path": "/path/to/project-a/src/new_feature.py",
            "language": "python",
            "hash": "abc123...",
            "previous_hash": None
        }
    ],
    
    "modified_files": [
        {
            "relative_path": "src/auth/login.py",
            "absolute_path": "/path/to/project-a/src/auth/login.py",
            "language": "python",
            "hash": "def456...",
            "previous_hash": "old789..."
        }
    ],
    
    "deleted_files": [
        {
            "relative_path": "src/deprecated/old_auth.py",
            "absolute_path": None,
            "language": "python",
            "hash": None,
            "previous_hash": "xyz000..."
        }
    ],
    
    "unchanged_files": 140,
    "skipped_files": 8,
    
    "files_to_parse": 2,
    "files_to_remove": 1
}
```

---

## Progress Reporting

### Output Format

```
┌─────────────────────────────────────────────────────────────────┐
│                   PROGRESS REPORTING                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  $ magaldi parse /path/to/project-a --user alice                │
│                                                                 │
│  [Discovery]                                                    │
│  Loading config...                    done                      │
│  Validating repository...             done                      │
│                                                                 │
│  [Change Detection]                                             │
│  Scanning files...                    243 found                 │
│  Computing hashes...                  243/243 (100%)            │
│  Comparing to main...                 done                      │
│  Comparing to user branch...          done                      │
│  Detecting deletions...               done                      │
│                                                                 │
│  Summary:                                                       │
│    New:        5 files                                          │
│    Modified:   3 files                                          │
│    Deleted:    1 file                                           │
│    Unchanged:  234 files                                        │
│    Skipped:    12 files (same as main)                          │
│                                                                 │
│  [Parsing]                                                      │
│  Parsing files...                     8/8 (100%)                │
│                                                                 │
│  [Storage]                                                      │
│  Storing elements...                  145/145 (100%)            │
│  Updating file states...              done                      │
│                                                                 │
│  Complete: 145 elements from 8 files                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Verbosity Levels

```bash
# Quiet: errors only
magaldi parse /path/to/repo --user main -q

# Default: phase summaries
magaldi parse /path/to/repo --user main

# Verbose: progress bars, counts
magaldi parse /path/to/repo --user main -v

# Very verbose: individual files
magaldi parse /path/to/repo --user main -vv
```

| Level | Shows |
|-------|-------|
| `-q` (quiet) | Errors only |
| (default) | Phase summaries |
| `-v` (verbose) | Progress bars, counts |
| `-vv` (very verbose) | Individual files |

### Implementation

```python
class ProgressReporter:
    """Simple CLI progress reporting"""
    
    def phase(self, name: str):
        """Start a new phase"""
        print(f"\n[{name}]")
    
    def step(self, message: str, status: str = ""):
        """Report a step"""
        print(f"  {message:<35} {status}")
    
    def progress(self, message: str, current: int, total: int):
        """Report progress with percentage"""
        pct = int(current / total * 100) if total > 0 else 100
        print(f"\r  {message:<35} {current}/{total} ({pct}%)", end="", flush=True)
        if current == total:
            print()  # newline when complete
```

---

## Performance Considerations

| Operation | Bottleneck | Optimization |
|-----------|------------|--------------|
| File enumeration | Filesystem I/O | Single pass, filter early |
| Hash computation | Disk I/O | Thread pool (4-8 threads) |
| DB comparison | Network + queries | Batch queries, fetch all at once |
| Deletion detection | Memory / queries | Paginated (1000 per page) |

---

## Error Handling

| Error | Action |
|-------|--------|
| DB connection failed | Retry 3x, then abort |
| Main branch doesn't exist (user parse) | Error with clear message |
| File read error during hash | Log, skip file, continue |
| Inconsistent DB state | Log warning, continue |

---

## Summary of Decisions

| Decision | Value |
|----------|-------|
| Hash algorithm | SHA256, streamed for large files |
| Hash parallelism | Thread pool (4-8 threads) |
| Deletion detection | Paginated queries (1000 per page) |
| Main branch comparison | Local hash vs DB hash |
| User branch comparison | Compare to main first, then to previous user |
| User branch storage | Diff only (files that differ from main) |
| Deletion handling | Store marker (is_deleted: true) |
| Progress reporting | Yes, with verbosity levels |
| Dry-run mode | Not implemented |
