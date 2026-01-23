# Phase C: Call Graph & Dependencies Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add accurate call graph and dependency analysis through tree-sitter extraction.

**Architecture:**
- Extract imports and calls at parse time via tree-sitter
- Add `imports` and `calls` fields to ES schema
- Implement resolution logic to link calls to element IDs
- New MCP tools for call/dependency analysis

**Tech Stack:** Tree-sitter, Elasticsearch 8.11.0, Python

---

## Task 1: Update Elasticsearch Schema for Calls/Imports

**Files:**
- Modify: `src/shared/db/elasticsearch.py`
- Test: `tests/test_db_elasticsearch.py`

### Step 1: Add nested fields to INDEX_MAPPING

```python
# On file elements
"imports": {
    "type": "nested",
    "properties": {
        "name": {"type": "keyword"},        # Imported name
        "module": {"type": "keyword"},      # Source module
        "alias": {"type": "keyword"},       # Alias if any
    }
}

# On function/method elements
"calls": {
    "type": "nested",
    "properties": {
        "name": {"type": "keyword"},        # Function name
        "receiver": {"type": "keyword"},    # self, utils, null
        "line": {"type": "integer"},        # Line number
        "resolved_id": {"type": "keyword"}, # Resolved element ID
    }
}
```

### Step 2: Add methods to store/retrieve imports and calls

```python
def store_imports(self, element_id: str, imports: list[dict]) -> bool
def store_calls(self, element_id: str, calls: list[dict]) -> bool
def get_imports(self, element_id: str) -> list[dict]
def get_calls(self, element_id: str) -> list[dict]
def find_elements_calling(self, target_id: str, ...) -> list[dict]  # Query calls.resolved_id
def find_elements_importing(self, module: str, ...) -> list[dict]  # Query imports.module
```

### Step 3: Add tests

### Step 4: Commit
```bash
git commit -m "feat(elasticsearch): add calls and imports nested fields"
```

---

## Task 2: Extend Tree-sitter Parser for Imports

**Files:**
- Modify: `src/magaldi_core/code_parser.py`
- Test: `tests/test_code_parser.py`

### Step 1: Add Import dataclass

```python
@dataclass
class Import:
    name: str           # Imported name
    module: str         # Source module
    alias: str | None   # Alias if any
    line: int           # Line number
```

### Step 2: Add import extraction for Python

Extract from:
- `import_statement`: `import os` → `{"name": "os", "module": "os", "alias": null}`
- `import_from_statement`: `from utils import process as p` → `{"name": "process", "module": "utils", "alias": "p"}`

### Step 3: Add import extraction for JS/TS

Extract from:
- `import_statement`: `import { foo } from './utils'`
- `call_expression` (require): `const bar = require('lib')`

### Step 4: Store imports on file elements

```python
@dataclass
class CodeElement:
    # ... existing fields ...
    imports: list[Import] = field(default_factory=list)  # Only on file elements
```

### Step 5: Add tests

### Step 6: Commit
```bash
git commit -m "feat(parser): extract imports from Python/JS/TS files"
```

---

## Task 3: Extend Tree-sitter Parser for Calls

**Files:**
- Modify: `src/magaldi_core/code_parser.py`
- Test: `tests/test_code_parser.py`

### Step 1: Add Call dataclass

```python
@dataclass
class Call:
    name: str               # Function name
    receiver: str | None    # self, utils, object name, null
    line: int               # Line number
    resolved_id: str | None = None  # Filled in resolution phase
```

### Step 2: Add call extraction for Python

From `call` nodes:
- `process(x)` → `{"name": "process", "receiver": null, "line": 45}`
- `self.validate()` → `{"name": "validate", "receiver": "self", "line": 48}`
- `utils.run()` → `{"name": "run", "receiver": "utils", "line": 52}`

### Step 3: Add call extraction for JS/TS

From `call_expression` nodes.

### Step 4: Store calls on function/method elements

```python
@dataclass
class CodeElement:
    # ... existing fields ...
    calls: list[Call] = field(default_factory=list)  # Only on function/method elements
```

### Step 5: Add tests

### Step 6: Commit
```bash
git commit -m "feat(parser): extract function calls from Python/JS/TS"
```

---

## Task 4: Update Processor for Imports/Calls Indexing

**Files:**
- Modify: `src/magaldi_core/processor.py`
- Test: `tests/test_processor.py`

### Step 1: Update _index_element to store imports and calls

```python
def _index_element(
    element: CodeElement,
    ...
) -> bool:
    # ... existing indexing ...

    # Store imports for file elements
    if element.element_type == "file" and element.imports:
        es_repo.store_imports(element.element_id, [asdict(i) for i in element.imports])

    # Store calls for function/method elements
    if element.element_type in ("function", "method") and element.calls:
        es_repo.store_calls(element.element_id, [asdict(c) for c in element.calls])
```

### Step 2: Add call resolution phase

After all elements indexed, resolve call targets:

```python
def resolve_calls(es_repo: ElasticsearchRepository, scope: str, repo: str, username: str):
    # Get all elements with calls
    # For each call, try to resolve to target element_id
    # Update calls.resolved_id
```

### Step 3: Add tests

### Step 4: Commit
```bash
git commit -m "feat(processor): index imports and calls with resolution"
```

---

## Task 5: Implement Call Analysis MCP Tools

**Files:**
- Modify: `src/magaldi_mcp/tools.py`
- Modify: `src/magaldi_mcp/server.py`
- Test: `tests/test_mcp_tools.py`

### Tools to implement:

1. **find_callers** - Find all functions that call a given function
2. **find_call_chain** - Trace call chains (A → B → C → D)
3. **find_dead_code** - Find functions never called
4. **find_entry_points** - Find handlers, CLI commands, main functions

### Step 1: Implement find_callers

```python
def find_callers(
    es: ElasticsearchRepository,
    element_id: str,
    scope: str | None = None,
    repository: str | None = None,
    username: str | None = None,
    limit: int = 30,
    include_tests: bool = True,
) -> dict[str, Any]:
    # Query: WHERE calls.resolved_id = element_id
```

### Step 2: Implement find_call_chain

```python
def find_call_chain(
    es: ElasticsearchRepository,
    element_id: str,
    direction: str = "callees",  # callers, callees, both
    max_depth: int = 5,
    ...
) -> dict[str, Any]:
    # Recursive traversal
```

### Step 3: Implement find_dead_code

```python
def find_dead_code(
    es: ElasticsearchRepository,
    scope: str,
    repository: str,
    username: str | None = None,
    include_tests: bool = False,
) -> dict[str, Any]:
    # Find functions with zero callers, exclude entry points
```

### Step 4: Implement find_entry_points

```python
def find_entry_points(
    es: ElasticsearchRepository,
    scope: str,
    repository: str,
    username: str | None = None,
) -> dict[str, Any]:
    # Find decorators: @app.route, @click.command, etc.
```

### Step 5: Register tools in server.py

### Step 6: Add tests

### Step 7: Commit
```bash
git commit -m "feat(mcp): add call analysis tools (find_callers, find_call_chain, find_dead_code, find_entry_points)"
```

---

## Task 6: Implement Dependency Analysis MCP Tools

**Files:**
- Modify: `src/magaldi_mcp/tools.py`
- Modify: `src/magaldi_mcp/server.py`
- Test: `tests/test_mcp_tools.py`

### Tools to implement:

1. **find_dependencies** - What does a file import?
2. **find_dependents** - What files import this module?
3. **dependency_graph** - Module-level dependency graph

### Step 1: Implement find_dependencies

```python
def find_dependencies(
    es: ElasticsearchRepository,
    file_path: str | None = None,
    element_id: str | None = None,
    ...
) -> dict[str, Any]:
    # Read file's imports field
```

### Step 2: Implement find_dependents

```python
def find_dependents(
    es: ElasticsearchRepository,
    module: str,
    scope: str,
    repository: str,
    ...
) -> dict[str, Any]:
    # Query: WHERE imports.module = module
```

### Step 3: Implement dependency_graph

```python
def dependency_graph(
    es: ElasticsearchRepository,
    scope: str,
    repository: str,
    internal_only: bool = True,
    ...
) -> dict[str, Any]:
    # Build graph of module → module dependencies
```

### Step 4: Register tools in server.py

### Step 5: Add tests

### Step 6: Commit
```bash
git commit -m "feat(mcp): add dependency analysis tools (find_dependencies, find_dependents, dependency_graph)"
```

---

## Task 7: Update get_call_graph to Use Indexed Data

**Files:**
- Modify: `src/magaldi_mcp/tools.py`
- Test: `tests/test_mcp_tools.py`

### Step 1: Replace regex-based implementation

Current: Uses regex on raw_code at query time
New:
- Callees: Read pre-indexed `calls` field
- Callers: Query `calls.resolved_id`

### Step 2: Update tests

### Step 3: Commit
```bash
git commit -m "feat(mcp): update get_call_graph to use indexed calls data"
```

---

## Task 8: Implement explain_element Meta Tool

**Files:**
- Modify: `src/magaldi_mcp/tools.py`
- Modify: `src/magaldi_mcp/server.py`
- Test: `tests/test_mcp_tools.py`

### Step 1: Implement explain_element

```python
def explain_element(
    es: ElasticsearchRepository,
    element_id: str,
    scope: str | None = None,
    repository: str | None = None,
    username: str | None = None,
) -> dict[str, Any]:
    """Comprehensive overview of an element."""
    return {
        "element": { name, type, file, signature, summary },
        "callers": [ ... top 5 callers ... ],
        "callees": [ ... all direct calls ... ],
        "imports": [ ... if file element ... ],
        "similar_code": [ ... top 3 similar ... ],
        "parent": { ... parent class/file ... },
    }
```

### Step 2: Register in server.py

### Step 3: Add tests

### Step 4: Commit
```bash
git commit -m "feat(mcp): add explain_element meta tool"
```

---

## Task 9: Update Skill Documentation for Phase C

**Files:**
- Modify: `.claude/skills/magaldi/SKILL.md`

### Step 1: Document new call analysis tools
### Step 2: Document new dependency tools
### Step 3: Add workflow examples

### Step 4: Commit (local only, .claude is gitignored)

---

## Task 10: Run Full Test Suite

### Step 1: Run tests
```bash
pytest tests/ -v
```

### Step 2: Fix any failures

### Step 3: Commit fixes
```bash
git commit -m "fix: address test failures after Phase C implementation"
```

---

## Summary

Phase C implementation creates:

1. **Schema changes:** `imports` and `calls` nested fields
2. **Parser changes:** Extract imports and calls via tree-sitter
3. **Processor changes:** Index imports/calls, resolve call targets
4. **New tools:** `find_callers`, `find_call_chain`, `find_dead_code`, `find_entry_points`, `find_dependencies`, `find_dependents`, `dependency_graph`, `explain_element`
5. **Updated tools:** `get_call_graph` uses indexed data

**Re-indexing required:** After implementation, existing codebases need re-indexing to populate import/call data.
