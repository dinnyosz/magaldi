# Magaldi Parser - Phase 1: Discovery

## Overview

The Discovery phase is the entry point of the parser pipeline. It validates the repository path, loads configuration, and prepares metadata for subsequent phases.

```
┌─────────────────────────────────────────────────────────────────┐
│                         PHASE 1: DISCOVERY                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1.1              1.2              1.3              1.4          │
│  ─────            ─────            ─────            ─────        │
│  VALIDATE     →   LOAD         →   RESOLVE      →   ENUMERATE   │
│  PATH             CONFIG           USERNAME         LANGUAGES    │
│                                                                 │
│  • Path exists    • magaldi.yaml   • CLI arg        • Walk dirs │
│  • Is directory   • Scope required • Env var        • Count files│
│  • Is readable    • Exclusions     • Config file    • Per lang  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Architecture Decisions

### Repository Structure
- **Flat structure only**: `/path/to/repo/`
- No nested organization support
- Each repository parsed independently via CLI

### Invocation Model
```bash
# Parser called per-repo
magaldi parse /path/to/project-a --user main
magaldi parse /path/to/project-b --user alice
```

### Multi-User Support
```
┌─────────────────────────────────────────────────────────────────┐
│                     DATA IDENTITY                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Unique key: (scope, repository, username, relative_path)       │
│                                                                 │
│  "main" branch:                                                 │
│  • Central/CI parsed                                            │
│  • Complete parse of all files                                  │
│  • Source of truth                                              │
│                                                                 │
│  User branches (alice, bob, etc):                               │
│  • Only files differing from main                               │
│  • Requires main to exist first                                 │
│  • Auto-expires after N days                                    │
│  • Manual cleanup available                                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 1.1 Path Validation

### Input
```bash
magaldi parse /path/to/project-a
```

### Validation Checks

| Check | Required | On Failure |
|-------|----------|------------|
| Path exists | Yes | Error: "Path does not exist" |
| Is directory | Yes | Error: "Path is not a directory" |
| Is readable | Yes | Error: "Permission denied" |

### Symlinks
- **Decision**: Ignore symlinks entirely
- Avoids infinite loops and complexity

---

## 1.2 Configuration Loading

### Config File
- **Filename**: `magaldi.yaml`
- **Location**: Repository root (`/path/to/project-a/magaldi.yaml`)

### Config Schema

```yaml
# /path/to/project-a/magaldi.yaml

# Required
scope: backend-services

# Optional - overrides dirname
name: project-a

# Optional - used if not provided via CLI/env
user: alice

# Optional - metadata
description: "Authentication microservice"
tags:
  - api
  - auth

# Exclusions (merged with defaults)
exclude_directories:
  - node_modules
  - __pycache__
  - .venv
  - dist
  - build
  - target
  - .git

exclude_files:
  - "*.min.js"
  - "*.min.css"
  - "*.map"
  - "*.lock"
  - "package-lock.json"
```

### Missing Config Behavior
- If `magaldi.yaml` does not exist:
  - Prompt user for scope
  - Generate config file
  - Continue with parse

### Config Validation

| Field | Required | Default |
|-------|----------|---------|
| scope | Yes | None (must prompt) |
| name | No | Directory name |
| user | No | None (must be from CLI/env) |
| description | No | Empty |
| tags | No | Empty list |
| exclude_directories | No | Default list |
| exclude_files | No | Default list |

---

## 1.3 Username Resolution

### Precedence (highest to lowest)

```
1. CLI argument        --user alice
2. Environment var     MAGALDI_USER=alice
3. Config file         user: alice (in magaldi.yaml)
4. Default             ERROR: username required
```

### Special Username: "main"
- Represents the central/CI parsed version
- Full parse of all files
- Source of truth for user branch comparisons

### User Branch Rules
- Requires `main` to be parsed first
- Only stores files that differ from main
- Auto-expires after configurable days
- Can be manually cleaned up

---

## 1.4 Language Enumeration

### Purpose
Count files per supported language (respecting exclusions).

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
{
    "languages": {
        "python": {"files": 145, "lines": 12500},
        "javascript": {"files": 32, "lines": 4200},
        "typescript": {"files": 15, "lines": 1550}
    },
    "total_files": 192,
    "total_lines": 18250
}
```

### Design Decisions
- **No primary language**: List all languages found
- **Generated code**: Parse everything, user adds exclusions if needed
- **Line counting**: Actual line count (not estimated)

---

## Default Exclusions

### Directories
```
node_modules/
vendor/
__pycache__/
.git/
dist/
build/
.venv/
venv/
target/
.next/
```

### Files
```
*.min.js
*.min.css
*.map
*.lock
package-lock.json
yarn.lock
composer.lock
Cargo.lock
```

---

## Database Schema

### repositories Table

```sql
CREATE TABLE repositories (
    id INT AUTO_INCREMENT PRIMARY KEY,
    scope VARCHAR(100) NOT NULL,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    tags JSON,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE KEY unique_scope_name (scope, name)
);
```

### file_states Table

```sql
CREATE TABLE file_states (
    id INT AUTO_INCREMENT PRIMARY KEY,
    scope VARCHAR(100) NOT NULL,
    repository VARCHAR(255) NOT NULL,
    username VARCHAR(100) NOT NULL,        -- "main" or user
    relative_path VARCHAR(1024) NOT NULL,
    
    file_hash VARCHAR(64),                 -- NULL if deleted
    language VARCHAR(50),
    file_size INT,
    line_count INT,
    
    is_deleted BOOLEAN DEFAULT FALSE,      -- Deletion marker
    
    parsed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    expires_at DATETIME,                   -- NULL for main, set for users
    
    UNIQUE KEY unique_file (scope, repository, username, relative_path),
    INDEX idx_expiry (expires_at),
    INDEX idx_user (username)
);
```

---

## CLI Commands

```bash
# Parse main branch (CI/central)
magaldi parse /path/to/project-a --user main

# Parse user branch (local changes)
magaldi parse /path/to/project-a --user alice

# With environment variable
export MAGALDI_USER=alice
magaldi parse /path/to/project-a

# Cleanup user data
magaldi cleanup --user alice
magaldi cleanup --user alice --scope backend-services
magaldi cleanup --user alice --repo project-a

# Auto-expire old user data
magaldi expire --days 30
```

---

## Element ID Format

All paths are relative to repository root:

```
{scope}:{repository}:{username}:{relative_path}:{type}:{name}:{line}

Examples:
backend-services:project-a:main:src/auth/login.py:function:authenticate:45
backend-services:project-a:alice:src/auth/login.py:function:authenticate:45
```

---

## Output

Discovery phase produces:

```python
@dataclass
class DiscoveryResult:
    # Identity
    scope: str
    repository: str
    username: str
    repo_path: str
    
    # Config
    description: Optional[str]
    tags: List[str]
    exclude_directories: List[str]
    exclude_files: List[str]
    
    # Languages
    languages: Dict[str, LanguageStats]
    total_files: int
    total_lines: int
```

This is passed to Phase 2: Change Detection.

---

## Summary of Decisions

| Decision | Value |
|----------|-------|
| Directory structure | Flat only |
| Symlinks | Ignore |
| Config filename | `magaldi.yaml` |
| Scope source | Required in config |
| Repo name | Derive from dirname, overridable |
| Missing config | Prompt user, create config |
| Username precedence | CLI > env > config |
| Main branch name | `main` |
| Primary language | No, list all |
| Generated code | Parse all, user excludes if needed |
| Git tracking | None |
