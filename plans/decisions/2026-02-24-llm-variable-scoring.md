## Decision: Replace heuristic variable filter with LLM-based scoring

**Original plan:** Variables are filtered during extraction (Phase 3) using hand-curated heuristics — SKIP_NAMES list, AST node type checks, USEFUL_FACTORIES/USEFUL_CONSTRUCTORS frozensets, PascalCase detection for instance creation, and module-level vs local scope rules. Two separate implementations: `should_skip_variable()` (shared, used by JS/PHP/Rust) and `is_useful_assignment()` (Python-specific).

**Deviation:** Replace the entire heuristic filter with an LLM-based scoring phase. New Phase 4 sends batches of variables to the local LLM for multi-dimensional scoring (config_value, architectural_role, data_definition, general_usefulness). Variables scoring below threshold on ALL dimensions are dropped before Phase 5 (summarize → embed → index).

**Why:** The heuristic filter answers a semantic question ("is this variable meaningful for code discovery?") with syntactic tools. It gets the easy cases right (skip `i`, `j`, `tmp`) but fails on the interesting cases:
- `app = Flask(__name__)` → skipped as "instance_creation" (PascalCase heuristic)
- `router = express.Router()` → skipped as "method_call_result"
- `db = create_engine(...)` → skipped as "function_call_result"
These are often the MOST important variables in a file. The USEFUL_FACTORIES allowlist requires constant maintenance for every new framework. We have a local LLM — use it for what it's good at.

**Options considered:**
1. **Keep heuristic filter, fix gaps** - Fix the 6 identified bugs (dead code paths, JS/PHP scope leak, Bash missing filter, Python drift). Pros: no LLM cost, fast. Cons: doesn't fix the fundamental problem — still guessing intent from syntax, still needs USEFUL_FACTORIES maintenance.
2. **Simple scope-only filter** (module-level = keep, local = skip) - Eliminate all factory/naming heuristics, keep only scope as signal. Pros: simple, no maintenance. Cons: would still miss useful locals, doesn't help with the `app = Flask()` problem at module level.
3. **LLM summarizer decides** (no preflight, let Phase 5 handle it) - Extract everything, summarize everything, let low-quality summaries rank low in search. Pros: no separate scoring step. Cons: much more expensive — full summarization + 3 embedding passes per variable.
4. **LLM-based preflight scoring** (chosen) - Cheap batch scoring with small model, then threshold. Pros: semantically correct decisions, dynamic batching keeps cost low, reuses existing tiered infrastructure. Cons: adds a new phase, LLM can be flaky on structured output.

**Final decision:** Option 4 — LLM-based preflight scoring as new Phase 4. Additionally:
- Extract variables at ALL depths in ALL languages (user explicitly chose this over fixing extractors to top-level only)
- Drop low-scorers entirely (user chose over handcrafted summary fallback or metadata-only indexing)
- Dynamic batch sizing based on token estimation, not fixed batch count
