# Phase D: Web UI Updates Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add visual exploration of call graphs, dependencies, and dual embeddings to the web interface.

**Architecture:**
- Backend: FastAPI routes for new endpoints
- Frontend: React/TypeScript pages using existing patterns
- Visualization: D3.js for interactive graphs

**Tech Stack:** FastAPI, React, TypeScript, D3.js

---

## Task 1: Add Backend Routes for Call Analysis

**Files:**
- Modify: `src/magaldi_web/routes/elements.py`
- Add: `src/magaldi_web/routes/analysis.py`
- Modify: `src/magaldi_web/models.py`
- Test: `tests/test_web_routes_elements.py`

### Step 1: Add element detail endpoint extensions

Update `/api/elements/{element_id}` to include:
- callers (top 5)
- callees (all)
- imports (if file element)
- similar_code (top 3)
- embedding_status (has_summary, has_code)

### Step 2: Create new analysis routes

```python
# /api/analysis/callers/{element_id}
# /api/analysis/call-chain/{element_id}?direction=both&max_depth=5
# /api/analysis/dead-code?scope=...&repository=...
# /api/analysis/entry-points?scope=...&repository=...
```

### Step 3: Add Pydantic models

```python
class CallerResponse(BaseModel):
    element_id: str
    name: str
    type: str
    file: str
    line: int
    summary: str

class CallChainNode(BaseModel):
    element_id: str | None
    name: str
    type: str | None
    file: str | None
    line: int | None
    callers: list["CallChainNode"] = []
    callees: list["CallChainNode"] = []
    cycle: bool = False
    unresolved: bool = False

class DeadCodeItem(BaseModel):
    element_id: str
    name: str
    type: str
    file: str
    line: int
    summary: str

class EntryPoint(BaseModel):
    element_id: str
    name: str
    type: str
    file: str
    line: int
    decorators: list[str]
    category: str  # http, cli, test, main, async
```

### Step 4: Commit
```bash
git commit -m "feat(web): add call analysis API routes"
```

---

## Task 2: Add Backend Routes for Dependency Analysis

**Files:**
- Modify: `src/magaldi_web/routes/analysis.py`
- Modify: `src/magaldi_web/models.py`

### Step 1: Create dependency routes

```python
# /api/analysis/dependencies/{element_id}  (file imports)
# /api/analysis/dependents?module=...&scope=...&repository=...
# /api/analysis/dependency-graph?scope=...&repository=...
```

### Step 2: Add Pydantic models

```python
class ImportInfo(BaseModel):
    name: str
    module: str
    alias: str | None
    line: int
    is_internal: bool

class DependencyGraphResponse(BaseModel):
    nodes: list[str]
    edges: list[dict]  # {from, to}
    cycles: list[list[str]]
    stats: dict
```

### Step 3: Commit
```bash
git commit -m "feat(web): add dependency analysis API routes"
```

---

## Task 3: Update Element Detail Page

**Files:**
- Modify: `src/magaldi_web/frontend/src/pages/Element.tsx`
- Modify: `src/magaldi_web/frontend/src/api.ts`

### Step 1: Add API calls for element details

```typescript
// api.ts
export async function getElementCallers(elementId: string): Promise<Caller[]>
export async function getElementCallees(elementId: string): Promise<Callee[]>
export async function getElementSimilar(elementId: string): Promise<Similar[]>
```

### Step 2: Update Element.tsx with new sections

- **Callers section**: List with file:line links
- **Callees section**: List with resolved links, unresolved marked
- **Imports section**: For file elements, internal/external badge
- **Similar Code section**: Tabs for "Similar Structure" / "Similar Intent"
- **Embedding Status**: Visual indicator icons
- **Call Chain**: Expandable tree (2 levels)

### Step 3: Commit
```bash
git commit -m "feat(web): enhance element detail page with call analysis"
```

---

## Task 4: Create Dead Code Report Page

**Files:**
- Add: `src/magaldi_web/frontend/src/pages/DeadCode.tsx`
- Modify: `src/magaldi_web/frontend/src/App.tsx`
- Modify: `src/magaldi_web/frontend/src/api.ts`

### Step 1: Create DeadCode.tsx component

- Table listing potentially dead functions
- Grouped by file
- Filters: exclude tests, min lines
- Stats summary at top

### Step 2: Add route in App.tsx

```typescript
<Route path="/:scope/:repo/dead-code" element={<DeadCode />} />
```

### Step 3: Commit
```bash
git commit -m "feat(web): add dead code report page"
```

---

## Task 5: Create Entry Points Dashboard

**Files:**
- Add: `src/magaldi_web/frontend/src/pages/EntryPoints.tsx`
- Modify: `src/magaldi_web/frontend/src/App.tsx`
- Modify: `src/magaldi_web/frontend/src/api.ts`

### Step 1: Create EntryPoints.tsx component

- Tabbed view: HTTP | CLI | Test | Main | Async
- Each tab shows list of entry points
- Link to entry point's call chain

### Step 2: Add route in App.tsx

```typescript
<Route path="/:scope/:repo/entry-points" element={<EntryPoints />} />
```

### Step 3: Commit
```bash
git commit -m "feat(web): add entry points dashboard"
```

---

## Task 6: Create Dependency Graph Page

**Files:**
- Add: `src/magaldi_web/frontend/src/pages/DependencyGraph.tsx`
- Modify: `src/magaldi_web/frontend/src/App.tsx`
- Modify: `src/magaldi_web/frontend/src/api.ts`

### Step 1: Create DependencyGraph.tsx component

- D3.js force-directed graph
- Nodes = files/modules
- Edges = imports
- Highlight cycles in red
- Filter controls: internal only, by directory

### Step 2: Add route in App.tsx

```typescript
<Route path="/:scope/:repo/dependency-graph" element={<DependencyGraph />} />
```

### Step 3: Commit
```bash
git commit -m "feat(web): add dependency graph visualization"
```

---

## Task 7: Create Call Explorer Page

**Files:**
- Add: `src/magaldi_web/frontend/src/pages/CallExplorer.tsx`
- Modify: `src/magaldi_web/frontend/src/App.tsx`
- Modify: `src/magaldi_web/frontend/src/api.ts`

### Step 1: Create CallExplorer.tsx component

- Start with search for any function
- Expand callers (upstream) or callees (downstream)
- Tree view with collapse/expand
- Depth limit control (slider)

### Step 2: Add route in App.tsx

```typescript
<Route path="/:scope/:repo/call-explorer" element={<CallExplorer />} />
```

### Step 3: Commit
```bash
git commit -m "feat(web): add call explorer page"
```

---

## Task 8: Update Search Page

**Files:**
- Modify: `src/magaldi_web/frontend/src/pages/Search.tsx`
- Modify: `src/magaldi_web/routes/search.py`
- Modify: `src/magaldi_web/frontend/src/api.ts`

### Step 1: Add search mode toggle

- Buttons: Summary | Code | Hybrid (default)
- Pass `search_mode` parameter to API

### Step 2: Add pattern search tab

- Mode selector: regexp | wildcard | proximity
- For proximity: slop slider (1-10)
- ES regexp syntax help tooltip

### Step 3: Update backend route

Add `search_mode` parameter to `/api/search` endpoint.

### Step 4: Commit
```bash
git commit -m "feat(web): add search mode toggle and pattern search"
```

---

## Task 9: Add Duplicate Code Report Page

**Files:**
- Add: `src/magaldi_web/frontend/src/pages/Duplicates.tsx`
- Modify: `src/magaldi_web/frontend/src/App.tsx`
- Modify: `src/magaldi_web/frontend/src/api.ts`
- Modify: `src/magaldi_web/routes/analysis.py`

### Step 1: Add backend route

```python
# /api/analysis/duplicates?scope=...&repository=...&min_similarity=0.95
```

### Step 2: Create Duplicates.tsx component

- Groups of similar code (>95% similarity)
- Side-by-side code preview
- Similarity score badge
- Filters: min lines, min similarity

### Step 3: Add route in App.tsx

```typescript
<Route path="/:scope/:repo/duplicates" element={<Duplicates />} />
```

### Step 4: Commit
```bash
git commit -m "feat(web): add duplicate code report page"
```

---

## Task 10: Add Navigation Enhancements

**Files:**
- Modify: `src/magaldi_web/frontend/src/pages/Element.tsx`
- Modify: `src/magaldi_web/frontend/src/pages/Explorer.tsx`
- Modify: `src/magaldi_web/frontend/src/pages/Repository.tsx`

### Step 1: Add caller/callee badges

- On element cards: show caller count badge
- On element cards: show callee count badge

### Step 2: Add quick action buttons

- "Find callers" button
- "Find similar" button
- "Show in graph" button

### Step 3: Add breadcrumb with call context

When navigating from caller → callee, show the path.

### Step 4: Commit
```bash
git commit -m "feat(web): add navigation enhancements with badges and quick actions"
```

---

## Task 11: Run Full Test Suite

### Step 1: Run tests
```bash
pytest tests/ -v
```

### Step 2: Fix any failures

### Step 3: Commit fixes
```bash
git commit -m "fix: address test failures after Phase D implementation"
```

---

## Summary

Phase D implementation creates:

1. **Backend routes:** Call analysis, dependency analysis endpoints
2. **Element page enhancements:** Callers, callees, imports, similar, embedding status
3. **New pages:** Dead Code, Entry Points, Dependency Graph, Call Explorer, Duplicates
4. **Search enhancements:** Search mode toggle, pattern search tab
5. **Navigation:** Badges, quick actions, breadcrumbs

**Note:** Some advanced visualizations (Code Similarity Map with UMAP/t-SNE) may be deferred to a future phase due to complexity.
