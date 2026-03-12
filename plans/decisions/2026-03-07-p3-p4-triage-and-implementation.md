## Decision: Implement P4 items (super resolution, nested functions) instead of deferring

**Original plan:** Both items were deferred from P2/P3 as "complex":
- super() resolution: "Medium complexity, requires cross-file resolution changes"
- JS nested function hierarchy: "Larger feature requiring recursive traversal and parent tracking"

**Deviation:** Investigated both and found they were straightforward. Implemented both in the same session.

**Why:** Investigation revealed the complexity was overestimated:
- super() calls were already extracted as `receiver="super"` by existing extractors. `base_classes` was already stored on class elements. All repo lookup methods (get_document_by_name_only, get_method_by_class) already existed.
- Nested functions were already found by extractors (Python/JS both use walk_tree). The only issue was `_set_hierarchy()` defaulting orphan elements to the file parent instead of using line-range containment.

**Options considered:**
1. Defer to P4 batch with separate planning — adds overhead, items are simpler than assumed
2. Implement now after investigation confirms low complexity — faster, no context switching cost

**Final decision:** Implemented both. Super resolution: ~100 lines as Strategy 5.8. Nested functions: ~30 lines enhancing `_set_hierarchy()` with line-range containment. Both work across all languages automatically.

---

## Decision: Use line-range containment in _set_hierarchy() for nested function parents

**Original plan:** Two approaches possible:
1. Track parent_node explicitly in extractors and convert to parent_id in parsers
2. Use line-range containment in _set_hierarchy() to detect nesting automatically

**Deviation:** Chose option 2 (line-range containment) instead of per-extractor parent_node tracking.

**Why:** Line-range containment works automatically for ALL languages without modifying any extractor or parser. It finds the tightest enclosing container (smallest line span that fully contains the element). Python extractor already had partial parent_node tracking but it was never converted to parent_id — this approach obsoletes that need.

**Options considered:**
1. Explicit parent_node tracking — requires changes in each extractor (Python already has it partially, JS/Rust/PHP don't). More accurate but more work.
2. Line-range containment — single ~30-line change in base.py, works for all languages. Slightly less precise if elements share exact same line ranges (handled by strict inequality check).

**Final decision:** Line-range containment. The approach correctly handles Python `def outer: def inner`, JS `function outer() { function inner() {} }`, methods inside classes, and even 3+ levels of nesting. All 3209 existing tests pass without modification.
