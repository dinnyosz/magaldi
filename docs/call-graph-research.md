# Call Graph & Code Intelligence Research

Research survey of tree-sitter based call graph exploration tools and techniques,
conducted 2026-02-28. Ideas prioritized by applicability to Magaldi's call resolution pipeline.

## Implemented

### Strategy 5.6: Constructor-Based Type Inference
**Inspired by:** GitHub's stack-graph / `locals.scm` scope analysis approach.

Traces `var = ClassName()` patterns to infer variable types, then resolves method
calls on those variables. Pure regex on `raw_code`, reuses existing `_lookup_method_by_type`.

**Commit:** `feat: add constructor-based type inference for call resolution (Strategy 5.6)`

### RRF Scoring for Strategy 6
**Inspired by:** [Axon](https://github.com/harshkedia177/axon)'s Reciprocal Rank Fusion across BM25 + vector + fuzzy search.

Replaced pure cosine similarity (0.7 threshold, ~0.1% accuracy) with RRF across three
signals: receiver-class name affinity, embedding cosine similarity, and path context.
Works even without embeddings via name/path signals.

**Commit:** `feat: replace pure cosine similarity with RRF scoring in Strategy 6`

---

## Future Ideas (prioritized)

### 1. Stack Graph-Inspired Scope Analysis
**Source:** [tree-sitter-stack-graphs](https://github.com/github/stack-graphs/tree/main/tree-sitter-stack-graphs) | [Blog](https://github.blog/open-source/introducing-stack-graphs/)

GitHub's name resolution system builds file-incremental graphs where paths represent
valid name bindings. Sub-100ms symbol navigation without full-program analysis.

**Application:** Go beyond constructor patterns to full assignment tracking:
- Chained assignments: `a = get_foo(); b = a.bar; b.method()`
- Conditional flows: `x = Foo() if cond else Bar()`
- Context manager: `with open(f) as handle: handle.read()`

**Effort:** Medium (2-3 weeks). Would require tree-sitter query-based scope tracking
per language rather than regex.

**Expected impact:** +5-10pp on `untyped` category (~2,000-4,000 calls).

### 2. Transitive Impact Analysis Tool
**Source:** [Axon](https://github.com/harshkedia177/axon) | [Roam-Code](https://github.com/Cranot/roam-code)

Axon's `axon_impact("validate")` returns all affected symbols grouped by depth with
confidence scores. Our `find_callers`/`get_call_graph` only go one level.

**Application:** New MCP tool `impact_analysis(element_id, depth=3)` that:
- Walks caller chains transitively up to N levels
- Scores confidence by depth (direct caller = 1.0, 2-hop = 0.7, etc.)
- Groups results by depth level
- Includes both callers (who's affected?) and callees (what does it depend on?)

**Effort:** Small (2-3 days). Recursive traversal over existing `find_callers`.

### 3. Improved Dead Code Detection
**Source:** [Axon](https://github.com/harshkedia177/axon)

Current `find_dead_code` just checks for zero callers. Axon's exemption list is more
sophisticated:
- Entry points (main, CLI commands, route handlers)
- Exports (`__all__`, public API)
- Decorators (`@app.route`, `@pytest.fixture`)
- Test code (test functions, setup/teardown)
- Protocol conformance / ABC implementations
- Constructor/dunder methods

**Effort:** Small (1-2 days). Mostly filter logic on existing dead code query.

### 4. Git-Diff Impact Analysis
**Source:** [GitNexus](https://github.com/abhigyanpatwari/GitNexus)

Combines graph structure with git diffs to answer "what's affected by this change."

**Application:** New MCP tool `diff_impact(base_branch="main")` that:
1. Gets changed files from `git diff`
2. Finds all elements in changed files
3. Walks call graph outward to find downstream consumers
4. Reports: "You changed X, which is called by Y and Z"

**Effort:** Medium (1 week). Needs git integration + graph traversal.

### 5. Token Budget Optimization
**Source:** [Roam-Code](https://github.com/Cranot/roam-code)

Roam-Code reduced MCP context from ~36K to <3K tokens (92% reduction) via:
- Compact output formats
- Smart truncation
- Schema optimization

**Application:** Audit our MCP tool descriptions with `mcp-tool-optimizer` skill.
Already have the tooling, just need to run it.

**Effort:** Small (1 day).

### 6. Declarative Graph Construction via tree-sitter-graph DSL
**Source:** [tree-sitter-graph](https://github.com/tree-sitter/tree-sitter-graph)

A DSL for constructing arbitrary graph structures from parsed source code. Stanzas
match tree-sitter query patterns and emit graph nodes/edges.

**Application:** Replace some imperative tree-sitter extraction code with declarative
stanzas. More maintainable when adding new languages.

**Effort:** Large (2-4 weeks). Requires rewriting extractors in Rust DSL.
**Trade-off:** Cleaner but adds a Rust dependency.

### 7. Datalog Queries for Graph Traversal
**Source:** [CIE](https://github.com/kraklabs/cie) (uses CozoDB)

CIE uses Datalog queries for recursive graph traversal — elegant for call-chain
operations that are awkward in OpenSearch.

**Application:** For complex queries like "find all paths from A to B" or
"find all functions that transitively call X," Datalog is natural.

**Trade-off:** OpenSearch isn't ideal for recursive graphs but adding CozoDB/Datalog
is a new dependency. Consider only if graph query performance becomes a bottleneck.

**Effort:** Large (3+ weeks).

---

## Reference Projects

| Project | Tech | Key Feature | URL |
|---------|------|-------------|-----|
| tree-sitter-graph | Rust DSL | Declarative graph construction from AST | [GitHub](https://github.com/tree-sitter/tree-sitter-graph) |
| tree-sitter-stack-graphs | Rust | File-incremental name resolution | [GitHub](https://github.com/github/stack-graphs) |
| Roam-Code | SQLite + tree-sitter | 137 commands, 92% token reduction | [GitHub](https://github.com/Cranot/roam-code) |
| CIE | CozoDB + tree-sitter | 25+ MCP tools, Datalog queries | [GitHub](https://github.com/kraklabs/cie) |
| Axon | Knowledge graph | 12-phase pipeline, RRF, impact analysis | [GitHub](https://github.com/harshkedia177/axon) |
| GitNexus | KuzuDB WASM | Browser-based, git-diff impact | [GitHub](https://github.com/abhigyanpatwari/GitNexus) |
| Code-Graph-RAG | Memgraph | Tree-sitter + knowledge graph + 10 MCP tools | [GitHub](https://github.com/vitali87/code-graph-rag) |
| ACER | Tree-sitter | 3-method framework for call graph generators | [GitHub](https://github.com/WM-SEMERU/ACER) |
| IBM tree-sitter-codeviews | Tree-sitter | Multi-view graphs (call, control, data flow) | [GitHub](https://github.com/IBM/tree-sitter-codeviews) |
| CocoIndex | Rust + tree-sitter | Real-time incremental indexing | [GitHub](https://github.com/cocoindex-io/realtime-codebase-indexing) |

## Academic Papers

- **ACER** (2023): AST-based Call Graph Generator Framework — [arXiv](https://arxiv.org/pdf/2308.15669)
- **Stack Graphs** (2022): Incremental name resolution — [arXiv](https://arxiv.org/pdf/2211.01224)
- **GitHub Static Analysis at Scale** (2021): Naive name-binding via `locals.scm` — [ACM](https://dl.acm.org/doi/fullHtml/10.1145/3487019.3487022)
- **Codified Context** (2026): Infrastructure for AI agents in complex codebases — [arXiv](https://arxiv.org/abs/2602.20478)
- **CodeRAG** (2025): Requirement graphs + DS-code graphs — [arXiv](https://arxiv.org/html/2504.10046v1)
