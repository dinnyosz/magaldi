# Parser Lab & SCM Refactor Plan

Created: 2025-01-28

## Overview

This plan covers two related initiatives:
1. **Consolidate extractors** - Merge useful code from `languages/` into `extractors/`, then delete `languages/`
2. **Refactor to SCM queries** - Convert Python tree-walking to declarative `.scm` query files
3. **Implement parser_lab** - Self-improvement MCP tools for Magaldi's parser

---

## Phase 1: Consolidate `languages/` into `extractors/` - COMPLETED

### Findings: Initial agent report was INCORRECT

Upon detailed analysis, `extractors/` was found to be the MORE COMPLETE version:

| Language | `languages/` lines | `extractors/` lines | Unique functions |
|----------|-------------------|--------------------|--------------------|
| Python | 853 | 1042 | `extractors/` has `_extract_nested_functions`, `_extract_python_parameters` |
| JavaScript | 772 | 933 | `extractors/` has `_extract_js_parameters`, `_extract_js_return_type` |
| PHP | 514 | 660 | `extractors/` has `_extract_php_parameters`, `_extract_php_return_type` |
| Rust | 655 | 763 | `extractors/` has `_extract_rust_parameters`, `_extract_rust_return_type` |

**Conclusion:** `languages/` had NO unique code - it was stale duplicate code.

### Completed Tasks

1. [x] Compared all function definitions between directories
2. [x] Verified `extractors/` has MORE features than `languages/`
3. [x] Confirmed no external imports from `languages/`
4. [x] Deleted `src/magaldi_core/languages/` directory
5. [x] Verified all extractors still work

---

## Phase 2: Refactor to SCM Queries

### Why SCM Queries?

| Aspect | Current (Python) | SCM Queries |
|--------|------------------|-------------|
| Pattern matching | Imperative loops | Declarative patterns |
| Maintainability | Edit Python code | Edit `.scm` files |
| Hot-reload | Requires restart | Can reload at runtime |
| parser_lab fit | Claude edits Python | Claude edits simpler `.scm` |
| Testing | Test Python functions | Test query patterns directly |

### Proposed Architecture

```
src/magaldi_core/
├── queries/                    # NEW: SCM query files
│   ├── python/
│   │   ├── elements.scm       # Classes, functions, methods
│   │   ├── imports.scm        # Import statements
│   │   ├── calls.scm          # Function/method calls
│   │   └── references.scm     # Type hints, instantiations
│   ├── javascript/
│   │   └── ...
│   ├── php/
│   │   └── ...
│   └── rust/
│       └── ...
├── extractors/
│   ├── base.py                # BaseExtractor + query loader
│   ├── python.py              # PythonExtractor (uses queries + post-processing)
│   └── ...
└── tree_sitter_manager.py     # Query compilation & caching
```

### Hybrid Approach

1. **SCM queries handle pattern matching:**
   ```scheme
   ; queries/python/elements.scm
   (function_definition
     name: (identifier) @function.name
     parameters: (parameters) @function.params
     return_type: (type)? @function.return_type
     body: (block) @function.body) @function.def

   (class_definition
     name: (identifier) @class.name
     superclasses: (argument_list)? @class.bases
     body: (block) @class.body) @class.def
   ```

2. **Python handles post-processing:**
   ```python
   def extract_elements(self, tree: Tree, lines: list[str]) -> list[ExtractedElement]:
       # Load and run query
       matches = self.query_runner.run("elements", tree)

       # Post-process matches into rich data structures
       elements = []
       for match in matches:
           if match.pattern == "function.def":
               elements.append(self._build_function_element(match, lines))
           elif match.pattern == "class.def":
               elements.append(self._build_class_element(match, lines))
       return elements
   ```

### Tasks - Infrastructure COMPLETED

1. [x] Create `src/magaldi_core/queries/` directory structure
2. [x] Implement query loader in `query_runner.py`
3. [x] Create Python element queries (`queries/python/elements.scm`)
4. [x] Create Python import queries (`queries/python/imports.scm`)
5. [x] Create Python call queries (`queries/python/calls.scm`)
6. [x] Integrate QueryRunner into `tree_sitter_manager.py`
7. [ ] Refactor `PythonExtractor` to use queries (optional - can use hybrid)
8. [ ] Add tests for query-based extraction
9. [ ] Repeat for JavaScript, PHP, Rust
10. [ ] Update documentation

### Created Files

- `src/magaldi_core/query_runner.py` - QueryRunner and QueryMatch classes
- `src/magaldi_core/queries/python/elements.scm` - Element extraction patterns
- `src/magaldi_core/queries/python/imports.scm` - Import extraction patterns
- `src/magaldi_core/queries/python/calls.scm` - Call extraction patterns

### Usage

```python
from magaldi_core.tree_sitter_manager import get_manager

manager = get_manager()
result = manager.run_query(code, 'python', 'elements')

for match in result.filter_by_capture('function.def'):
    name = match.get_text('function.name')
    print(f'Found function: {name}')
```

### Example SCM Queries

#### Python Functions
```scheme
; Function definitions (including async)
(function_definition
  name: (identifier) @name
  parameters: (parameters) @params
  return_type: (type)? @return_type) @definition

; Decorated functions
(decorated_definition
  (decorator)+ @decorators
  definition: (function_definition
    name: (identifier) @name)) @decorated_function
```

#### Python Classes
```scheme
; Class definitions
(class_definition
  name: (identifier) @name
  superclasses: (argument_list
    (identifier) @base_class)*
  body: (block) @body) @definition

; Decorated classes
(decorated_definition
  (decorator)+ @decorators
  definition: (class_definition
    name: (identifier) @name)) @decorated_class
```

#### Python Imports
```scheme
; import X
(import_statement
  name: (dotted_name) @module) @import

; from X import Y
(import_from_statement
  module_name: (dotted_name) @module
  name: (dotted_name) @name) @from_import

; from X import Y as Z
(import_from_statement
  module_name: (dotted_name) @module
  name: (aliased_import
    name: (dotted_name) @name
    alias: (identifier) @alias)) @aliased_import
```

---

## Phase 3: Parser Lab Implementation

### Tools (after SCM refactor)

| Tool | Purpose |
|------|---------|
| `parser_lab_analyze` | Parse code, show elements + gaps |
| `parser_lab_create_test` | Generate pytest test for expected behavior |
| `parser_lab_run_tests` | Execute parser tests |
| `parser_lab_suggest_fix` | Suggest `.scm` query modifications |

### Key Benefit of SCM for parser_lab

With SCM queries, `parser_lab_suggest_fix` can output:
```
Gap: Django @api_view decorator not detected as HTTP route

Suggested addition to queries/python/routes.scm:

; DRF api_view decorator
(decorated_definition
  (decorator
    (call
      function: (identifier) @decorator_name
      arguments: (argument_list
        (list) @methods)))
  (#eq? @decorator_name "api_view")) @drf_route
```

This is much easier for Claude to suggest than Python code modifications.

### Workflow

```
1. ANALYZE
   parser_lab_analyze(file_path="/external/django/views.py")
   → Shows: elements extracted, AST nodes NOT captured

2. CREATE TEST (TDD)
   parser_lab_create_test(
     name="drf_api_view",
     code="@api_view(['GET'])\ndef view(req): pass",
     expected={routes: [{method: "GET"}]}
   )
   → Creates: tests/extractors/test_python_drf_api_view.py

3. RUN TEST (should fail)
   parser_lab_run_tests(filter="test_drf_api_view")
   → FAILED: no routes detected

4. GET SUGGESTION
   parser_lab_suggest_fix(gap="api_view not detected", language="python")
   → Suggests: addition to queries/python/routes.scm

5. APPLY (Claude uses Edit tool on .scm file)

6. RUN TEST (should pass)
   parser_lab_run_tests(filter="test_drf_api_view")
   → PASSED
```

---

## Implementation Order

1. **Phase 1: Consolidate** (prerequisite cleanup)
   - Port useful code from `languages/` to `extractors/`
   - Delete `languages/`
   - ~2-3 hours

2. **Phase 2: SCM Refactor** (foundation)
   - Create query infrastructure
   - Convert Python extractor first (most complete)
   - ~4-6 hours for Python, then parallelize others

3. **Phase 3: Parser Lab** (builds on SCM)
   - Implement 4 MCP tools
   - Much simpler with SCM queries
   - ~3-4 hours

---

## Success Criteria

- [ ] `languages/` directory deleted, no functionality lost
- [ ] All existing tests pass after SCM refactor
- [ ] `.scm` query files for all 4 languages
- [ ] parser_lab tools working end-to-end
- [ ] Can use parser_lab to add a new pattern (e.g., FastAPI routes) via TDD

---

## Notes

- Tree-sitter query docs: https://tree-sitter.github.io/tree-sitter/using-parsers/queries/1-syntax.html
- py-tree-sitter Query class: https://tree-sitter.github.io/py-tree-sitter/classes/tree_sitter.Query.html
- Existing queries for reference: https://github.com/tree-sitter/tree-sitter-python/blob/master/queries/highlights.scm
