# Magaldi Parser - Architecture Overview

## Project Summary

Magaldi is an open-source code discovery engine that helps AI agents and developers navigate and understand codebases through intelligent indexing and semantic search.

---

## Parser Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PARSER PIPELINE                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Phase 1          Phase 2           Phase 3          Phase 4               │
│  ────────         ────────          ────────         ────────              │
│  DISCOVERY   →    CHANGE      →     PARSING    →     STORAGE               │
│                   DETECTION                                                 │
│                                                                             │
│  • Validate path  • Hash files      • Tree-sitter    • MySQL insert        │
│  • Load config    • Compare DB      • Extract AST    • ES index            │
│  • Resolve user   • Filter changed  • Build elements • Update tracking     │
│  • Enum languages • Mark deletions  • Link parents   • Create jobs         │
│                                                                             │
│  Status: ✅       Status: ✅        Status: ✅       Status: ✅            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                         AI PROCESSING PIPELINE                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Phase 5          Phase 6           Phase 7          Phase 8               │
│  ────────         ────────          ────────         ────────              │
│  SUMMARIZE   →    EMBEDDING    →    MCP SERVER  →    WEB UI                │
│                                                                             │
│  • Ollama LLM     • Ollama embed    • Search tools   • Dashboard           │
│  • Hierarchical   • Batch process   • Claude Code    • Visualization       │
│  • File→Class→Fn  • Store in ES     • Subagents      • Search UI           │
│  • Job workers    • Vector search   • Skills         • Repository view     │
│                                                                             │
│  Status: ✅       Status: ✅        Status: ✅       Status: ✅            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Core Architecture Decisions

### Invocation Model
| Decision | Value |
|----------|-------|
| Parse trigger | CLI command per repository |
| Structure | Flat repos only (no nesting) |
| Config file | `magaldi.yaml` in repo root |

```bash
magaldi parse /path/to/project-a --user main
magaldi parse /path/to/project-b --user alice
```

### Multi-User Model

```
┌─────────────────────────────────────────────────────────────────┐
│                     BRANCHING MODEL                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  "main" branch:                                                 │
│  • Parsed by CI/central system                                  │
│  • Full parse of all files                                      │
│  • Source of truth for team                                     │
│                                                                 │
│  User branches (alice, bob, etc):                               │
│  • Only files differing from main                               │
│  • Requires main to exist first                                 │
│  • Auto-expires after N days                                    │
│  • Manual cleanup available                                     │
│                                                                 │
│  Cross-repo search:                                             │
│  • Repos with same scope searchable together                    │
│  • User can overlay their changes on main                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Identity Keys

| Data Type | Unique Key |
|-----------|------------|
| File | (scope, repository, username, relative_path) |
| Element | (scope, repository, username, relative_path, type, name, line) |

**Element ID Format:**
```
{scope}:{repository}:{username}:{relative_path}:{type}:{name}:{line}

Example:
backend-services:project-a:main:src/auth/login.py:function:authenticate:45
backend-services:project-a:alice:src/auth/login.py:function:authenticate:45
```

### Path Handling
- All paths stored as **relative to repository root**
- Enables multi-user, multi-machine scenarios
- Absolute paths only used at runtime

---

## Configuration

### Repository Config (`magaldi.yaml`)

```yaml
# Required
scope: backend-services

# Optional
name: project-a              # Default: dirname
user: alice                  # Default: from CLI/env
description: "Auth service"
tags:
  - api
  - auth

# Exclusions
exclude_directories:
  - node_modules
  - __pycache__
  - .venv
  - dist

exclude_files:
  - "*.min.js"
  - "*.lock"
```

### Username Resolution (Precedence)
```
1. CLI argument        --user alice     (highest)
2. Environment var     MAGALDI_USER=alice
3. Config file         user: alice
4. Default             ERROR            (lowest)
```

---

## Supported Languages

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

### repositories

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

### file_states

```sql
CREATE TABLE file_states (
    id INT AUTO_INCREMENT PRIMARY KEY,
    scope VARCHAR(100) NOT NULL,
    repository VARCHAR(255) NOT NULL,
    username VARCHAR(100) NOT NULL,
    relative_path VARCHAR(1024) NOT NULL,
    
    file_hash VARCHAR(64),
    language VARCHAR(50),
    file_size INT,
    line_count INT,
    
    is_deleted BOOLEAN DEFAULT FALSE,
    
    parsed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    expires_at DATETIME,
    
    UNIQUE KEY unique_file (scope, repository, username, relative_path),
    INDEX idx_expiry (expires_at),
    INDEX idx_user (username)
);
```

### code_elements

```sql
CREATE TABLE code_elements (
    element_id VARCHAR(512) PRIMARY KEY,
    scope VARCHAR(100) NOT NULL,
    repository VARCHAR(255) NOT NULL,
    username VARCHAR(100) NOT NULL,
    relative_path VARCHAR(1024) NOT NULL,
    
    element_type VARCHAR(50) NOT NULL,
    name VARCHAR(255) NOT NULL,
    line_start INT NOT NULL,
    line_end INT,
    
    raw_code TEXT,
    signature VARCHAR(1024),
    docstring TEXT,
    
    level INT NOT NULL,
    parent_id VARCHAR(512),
    
    language VARCHAR(50),
    decorators JSON,
    visibility VARCHAR(20),
    is_async BOOLEAN DEFAULT FALSE,
    
    summary_status ENUM('pending', 'running', 'completed', 'failed') DEFAULT 'pending',
    embedding_status ENUM('pending', 'running', 'completed', 'failed') DEFAULT 'pending',
    summary TEXT,
    
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    expires_at DATETIME,
    
    INDEX idx_scope_repo (scope, repository),
    INDEX idx_username (username),
    INDEX idx_expiry (expires_at)
);
```

---

## CLI Commands

### Parsing
```bash
# Main branch (CI/central)
magaldi parse /path/to/project-a --user main

# User branch
magaldi parse /path/to/project-a --user alice

# Verbosity
magaldi parse /path/to/repo -q      # Quiet: errors only
magaldi parse /path/to/repo         # Default: summaries
magaldi parse /path/to/repo -v      # Verbose: progress
magaldi parse /path/to/repo -vv     # Very verbose: files
```

### Cleanup
```bash
# All user data
magaldi cleanup --user alice

# Specific scope
magaldi cleanup --user alice --scope backend-services

# Specific repo
magaldi cleanup --user alice --repo project-a

# Auto-expire old data
magaldi expire --days 30
```

---

## Phase Status

| Phase | Document | Status |
|-------|----------|--------|
| Phase 1: Discovery | `phase1_discovery.md` | ✅ Complete |
| Phase 2: Change Detection | `phase2_change_detection.md` | ✅ Complete |
| Phase 3: Parsing | `phase3_parsing.md` | ✅ Complete |
| Phase 4: Storage | `phase4_storage.md` | ✅ Complete |
| Phase 5: Summarization | `phase5_summarization.md` | ✅ Complete |
| Phase 6: Embedding | `phase6_embedding.md` | ✅ Complete |
| Phase 7: MCP Server | `phase7_mcp_server.md` | ✅ Complete |
| Phase 8: Web UI | `phase8_web_ui.md` | ✅ Complete |

---

## Key Design Decisions Summary

| Area | Decision |
|------|----------|
| Invocation | CLI per repository |
| Structure | Flat repos only |
| Config | `magaldi.yaml` in repo root |
| Missing config | Prompt user, create file |
| Paths | Always relative to repo root |
| Username | Required (CLI > env > config) |
| Main branch | Named "main", full parse |
| User branches | Diff from main only |
| Deletions | Store marker (is_deleted: true) |
| User cleanup | Manual + auto-expire |
| Symlinks | Ignore |
| Git tracking | None |
| Primary language | No, list all languages |
| Generated code | Parse all, user excludes if needed |
| Hash algorithm | SHA256, streamed |
| Hash parallelism | Thread pool (4-8) |
| Deletion detection | Paginated (1000/page) |
| Progress reporting | Yes, verbosity levels |
| Dry-run | Not implemented |
