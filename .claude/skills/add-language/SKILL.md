---
name: add-language
description: >
  Orchestrator for adding a new programming language to Magaldi's parser pipeline.
  Dispatches focused subagents for each phase: research, core implementation,
  analysis integration, framework support, testing, and real-world validation.
---

# Add Language to Magaldi

Adds a new programming language to Magaldi's code discovery engine. Uses subagent dispatch to keep context small and survive compaction.

## Orchestrator Workflow

When the user says "add <language> support" or invokes `/add-language`:

### 1. Initialize tracking

```
TodoWrite([
  {content: "Phase 0: Research <lang>", status: "pending"},
  {content: "Phase 1: Core implementation (extractor + parser + registration)", status: "pending"},
  {content: "Phase 2: Analysis integration (metrics, concurrency, categorizer, doc comments, module resolver, scope bindings)", status: "pending"},
  {content: "Phase 3: Framework support (web routes + CLI commands)", status: "pending"},
  {content: "Phase 4: Testing (fixtures + unit tests + Parser Lab + integrity check)", status: "pending"},
  {content: "Phase 5: Test repos (find, register, clone, parse, validate)", status: "pending"},
])
```

### 2. Execute phases sequentially

For each phase:
1. Mark phase `in_progress` in TodoWrite
2. Launch subagent using the Task tool with `subagent_type: "general-purpose"` and the corresponding prompt template below
3. Wait for subagent completion
4. Review output, verify files were created/modified correctly
5. Mark phase `completed` in TodoWrite
6. Commit using `/commit-message`

**Phases 2 & 3 can run in parallel** (both depend on Phase 1 but not on each other).

### 3. Language Profile artifact

Phase 0 produces `plans/languages/<lang>-profile.md` — a structured research document that all subsequent subagents read. This avoids re-researching the same things.

---

## Phase 0: Research Subagent

```
You are researching the {language} programming language to prepare for adding it to Magaldi's
code discovery engine. Produce a structured "Language Profile" document.

## Tools Available

Use these tools for research:
- **WebSearch**: Language syntax, frameworks, built-in types, stdlib modules
- **Context7**: Tree-sitter grammar docs, framework routing APIs
  - `mcp__plugin_context7_context7__resolve-library-id(libraryName="...", query="...")`
  - `mcp__plugin_context7_context7__query-docs(libraryId="...", query="...")`
- **Parser Lab**: AST exploration (only if tree-sitter-{lang} is already installed)
  - `mcp__magaldi__parser_lab_analyze(language="{lang}", code="...")`

## Research Tasks

### 1. Language Fundamentals (WebSearch)
Search for each of these:
- Syntax: how {language} defines classes/structs, functions/methods, interfaces/traits, enums, type aliases
- Self keyword: `self`, `this`, `$this`, `Self`, etc.
- Visibility model: `pub`, `public/private/protected`, uppercase-for-public, `export`, etc.
- Doc comment format: `///`, `/** */`, `#`, `//!`, `"""`, etc.
- Async model: async/await, goroutines, coroutines, channels
- Module/import system: how imports work (dotted paths, `use`, `require`, `import`, namespaces)
- Test conventions: naming patterns, test frameworks, attributes/annotations
- Standard library module prefixes (for call categorization)
- Built-in types and their methods (Array, String, Map, Vec, etc.)

Example searches:
- "{language} syntax cheat sheet"
- "{language} import system explained"
- "{language} visibility access modifiers"
- "{language} doc comment format conventions"
- "{language} async await concurrency model"
- "{language} standard library modules list"
- "{language} built-in types methods"

### 2. Tree-sitter Grammar (Context7 + WebSearch)
Use Context7: `resolve-library-id(libraryName="tree-sitter-{lang}")` then `query-docs` for node types.
If Context7 lacks coverage, search: "tree-sitter-{lang} node types", "tree-sitter-{lang} grammar.js"

Identify AST node types for:
- Function/method definitions
- Class/struct/interface definitions
- Import/use/require statements
- Variable/constant declarations
- Decorator/attribute/annotation syntax
- Async function variants
- Call expressions
- Parameter lists and type annotations

### 3. Top Framework Research (WebSearch + Context7)
Search: "most popular {language} web framework", "{language} CLI framework"

For each framework, use Context7 to get routing API details:
- Route/handler definition patterns (decorators, annotations, method calls, macros)
- HTTP method mapping
- Path parameter syntax (`:id`, `{{id}}`, `<id>`, etc.)
- CLI command/option definition patterns
- Entry point patterns

### 4. Concurrency Patterns (WebSearch)
Search: "{language} concurrency patterns", "{language} environment variable access"
- Threading/async primitives and their names
- Env var access patterns (e.g., `os.getenv()`, `std::env::var()`, `process.env.X`)

## Output

Write the profile to: /Users/dinnyosz/code/magaldi/plans/languages/{lang}-profile.md

Use this EXACT structure:

```markdown
# {Language} Language Profile

## Syntax Fundamentals
- **Self keyword**: ...
- **Visibility**: ...
- **Doc comments**: ...
- **Async model**: ...
- **Module system**: ...
- **File extensions**: ...

## Tree-sitter Grammar
- **Package**: tree-sitter-{lang}
- **Import**: `import tree_sitter_{lang} as ts_{lang}`
- **Language function**: `ts_{lang}.language()` (verify exact name)

### Key Node Types
| Construct | Node Type |
|-----------|-----------|
| Function definition | `...` |
| Method definition | `...` |
| Class/struct | `...` |
| Interface/trait | `...` |
| Enum | `...` |
| Import | `...` |
| Variable declaration | `...` |
| Constant declaration | `...` |
| Call expression | `...` |
| Decorator/attribute | `...` |
| Async function | `...` |
| Parameter list | `...` |
| Type annotation | `...` |

## Top Web Framework: {name}
- **Pattern type**: decorator-based / annotation-based / method-call-based / macro-based
- **Route patterns**:
  | Decorator/Call | HTTP Method | Framework |
  |---------------|-------------|-----------|
  | `...` | GET | {framework} |
  | `...` | POST | {framework} |
- **Path param syntax**: `{id}` / `:id` / `<id>`
- **Entry point patterns**: ...

## Top CLI Framework: {name}
- **Command patterns**: ...
- **Option/argument patterns**: ...

## Analysis Data
### DECISION_NODES (control flow AST types for cyclomatic complexity)
- `if_statement`, `for_statement`, ...

### BUILTIN_METHODS (type → [methods])
- String: [...]
- Array/List/Vec: [...]
- Map/Dict/Hash: [...]

### STDLIB_MODULES (module prefixes)
- [...]

### ENV_VAR_PATTERNS (regex)
- `...`

### CONCURRENCY_PATTERNS
- `...`

## Test Conventions
- **Test file patterns**: `test_*.{ext}`, `*_test.{ext}`, `tests/**`
- **Test function patterns**: `test_*`, `#[test]`, `@Test`, etc.
- **Test element detection**: ...
```

IMPORTANT: Be thorough. Every subsequent phase reads this profile instead of re-researching.
```

---

## Phase 1: Core Implementation Subagent

```
You are implementing core {language} support for Magaldi's code discovery engine.

Read the language profile first: /Users/dinnyosz/code/magaldi/plans/languages/{lang}-profile.md

Use magaldi MCP tools (search_code, find_usages, pattern_search, find_files,
get_file_structure, etc.) instead of built-in Grep/Glob for code search.

## Your Tasks (in order)

### Task 1: Install tree-sitter grammar
- Add `tree-sitter-{lang}` to `pyproject.toml` under `[project.dependencies]`
- Run: `pip install -e .`

### Task 2: Register in TreeSitterManager
**File**: `src/magaldi_core/tree_sitter_manager.py`
1. Add import: `import tree_sitter_{lang} as ts_{lang}`
2. Add to `LANGUAGE_CONFIG` dict: `"{lang}": (ts_{lang}, "language")`
   (Check the grammar package for the exact function name — some use `language_{lang}`)
3. Add to `get_extractor()`: elif branch returning `{Lang}Extractor()`
4. After Task 3: add extract function imports and `__all__` entries

### Task 3: Create the Extractor
**Create**: `src/magaldi_core/extractors/{lang}.py` (or `{lang}/` package for complex languages)

Must implement `BaseExtractor` from `src/magaldi_core/extractors/base.py`:

Required abstract methods:
| Method | Returns | Purpose |
|--------|---------|---------|
| `extract_elements` | `list[ExtractedElement]` | Top-level classes, functions, variables, imports |
| `extract_class_members` | `tuple[methods, variables]` | Methods + class-level variables from a class node |
| `extract_imports` | `list[ExtractedImport]` | Import/use/require statements |
| `extract_references` | `list[ExtractedReference]` | Cross-file references |
| `extract_calls` | `list[ExtractedCall]` | Function/method calls within a function body |

Optional overrides (default returns empty):
- `extract_class_attributes` — instance attributes from constructors
- `extract_base_classes` — superclass names
- `extract_raised_exceptions` — exception types thrown
- `extract_modified_attributes` — self/this attributes assigned in methods

Key data types (from `src/magaldi_core/extractors/types.py`):
- `ExtractedElement`: set `element_type` to: class|interface|trait|enum|type_alias|function|method|constant|variable|import
- `ExtractedImport`: name, module, alias, line
- `ExtractedCall`: name, receiver, line
- `ExtractedReference`: ref_type (instantiation/function_call/method_call/type_hint), target_name, line
- `DecoratorInfo`: name, args, full
- `ParameterInfo`: name, type, default

Reference implementations (read these first):
- Simple: `src/magaldi_core/extractors/bash.py`
- Medium: `src/magaldi_core/extractors/rust.py`
- Complex: `src/magaldi_core/extractors/python/`

Also create standalone functions that delegate to the extractor:
```python
def extract_{lang}_elements(tree, lines) -> list[ExtractedElement]: ...
def extract_{lang}_imports(tree, lines) -> list[ExtractedImport]: ...
def extract_{lang}_calls(function_node) -> list[ExtractedCall]: ...
def extract_top_level_{lang}_calls(tree) -> list[ExtractedCall]: ...
```

Use Context7 if unsure about AST node types:
```
mcp__plugin_context7_context7__query-docs(libraryId="<tree-sitter-id>", query="function definition fields")
```

### Task 4: Create the Parser
**Create**: `src/magaldi_core/parsers/{lang}.py`

Inherits `TreeSitterParser` from `src/magaldi_core/parsers/base.py`.
Converts `ExtractedElement` → `CodeElement`.

Key patterns:
- `self._create_file_element(content, file_info, scope, repository, username, "{lang}")` for file element
- `extract_preceding_doc_comment(lines, ext.line_start, "{lang}")` for docstrings
- `generate_element_id(scope, repo, user, path, type, name, ext.get_byte_offset())` for IDs
- Level hierarchy: file=0, class/trait/enum/interface=1, function/method=2, variable/constant=3
- `self._set_hierarchy(elements, file_element)` for parent-child relationships
- `self._resolve_calls_in_file(elements, self_keyword="<from profile>")` for same-file call resolution

Read an existing parser for the full pattern: `src/magaldi_core/parsers/rust.py`

### Task 5: Register the Parser
**5a. `src/magaldi_core/parsers/__init__.py`**: Import + `__all__`
**5b. `src/magaldi_core/code_parser.py`**:
  - Import `{Lang}Parser`
  - Add to `PARSERS` dict: `"{lang}": {Lang}Parser()`
  - Add cross-file references case in `parse_file()` if applicable
  - Add test path patterns to `TEST_PATH_PATTERNS`
  - Add test detection to `is_test_element()`
**5c. `src/magaldi_core/tree_sitter_manager.py`**: Add extract function imports + `__all__`

### Task 6: Register file extensions
**File**: `src/magaldi_core/discovery.py`
- Add to `SUPPORTED_EXTENSIONS`: `".{ext}": "{lang}"`
- Add to `SUPPORTED_FILENAMES` if special filenames exist
- Add to `SHEBANG_PATTERNS` if extensionless files use shebangs

### Validation
After all tasks, use Parser Lab to verify extraction works:
```
mcp__magaldi__parser_lab_analyze(language="{lang}", code="<sample from profile>")
```
Check the gap analysis. Fix any missing elements before finishing.

## Common Pitfalls
- Wrong `self_keyword` — must match language (check profile)
- Use `ext.get_byte_offset()` not `ext.line_start` for element IDs
- Level hierarchy: file=0, class=1, function=2, variable=3
- Don't forget `__all__` exports in tree_sitter_manager.py
```

---

## Phase 2: Analysis Integration Subagent

```
You are adding {language}-specific analysis data to Magaldi's code discovery engine.

Read the language profile: /Users/dinnyosz/code/magaldi/plans/languages/{lang}-profile.md
It contains the exact values to add for each dict.

Use magaldi MCP tools (search_code, find_usages, pattern_search, find_files,
get_file_structure, etc.) instead of built-in Grep/Glob for code search.

## Tasks

### 1. Cyclomatic complexity nodes
**File**: `src/magaldi_core/analysis/metrics.py` → `DECISION_NODES` dict
Add entry: `"{lang}": {{...}}` with control flow AST node types from the profile.
Read existing entries (python, javascript, etc.) for the pattern.

### 2. Concurrency patterns
**File**: `src/magaldi_core/analysis/concurrency.py`
- Add `"{lang}"` entry to `ENV_VAR_PATTERNS` — regex for env var access
- Add `"{lang}"` entry to `CONCURRENCY_PATTERNS` — async/threading/locking patterns
Read existing entries for the pattern.

### 3. Call categorization
**File**: `src/magaldi_core/extractors/call_categorizer.py`
- Add `"{lang}"` entry to `BUILTIN_METHODS` — built-in type methods from profile
- Add `"{lang}"` entry to `BUILTINS` if the language has built-in functions
- Add `"{lang}"` entry to `STDLIB_MODULES` — stdlib module prefixes from profile

### 4. Doc comment support
**File**: `src/magaldi_core/parsers/base.py`
Based on the language's doc comment format (from profile):
- Block doc comments (`/** */`): Add to `_BLOCK_DOC_LANGUAGES` frozenset
- Attribute lines (`#[...]`): Add to `_ATTRIBUTE_LANGUAGES` frozenset
- Line doc comments (`///`): Add new elif branch in `extract_preceding_doc_comment()`
- Hash comments (`#`): Add language to existing Python/Bash condition

Also update `determine_visibility()` if the language has naming-based visibility (e.g., Go uppercase = public).

### 5. Module resolution
**File**: `src/magaldi_core/module_resolver.py`
Create `{Lang}ModuleResolver(ModuleResolver)` with:
- `is_external_module(self, module: str) -> bool`
- `module_to_file_paths(self, module, caller_path=None) -> list[str]`

Read existing resolvers (PythonModuleResolver, etc.) for the pattern.
Use the profile's module system description to implement correctly.

### 6. Scope bindings
**File**: `src/magaldi_core/scope_bindings.py`
Add AST-based variable binding extraction patterns for {language}.
This enables Strategy 5.7 type resolution in call resolution.

Read existing patterns for Python/JS/PHP/Rust as templates.
Use Parser Lab to identify correct AST node types for variable bindings:
```
mcp__magaldi__parser_lab_analyze(language="{lang}", code="val result = getUser()\nval repo = Repository()")
```

## Validation
After all tasks, verify by reading each modified file and confirming the new entries exist.
Run: `make lint` to check for import/formatting issues.
```

---

## Phase 3: Framework Support Subagent

```
You are implementing web framework route detection and CLI command detection for {language}
in Magaldi's code discovery engine.

Read the language profile: /Users/dinnyosz/code/magaldi/plans/languages/{lang}-profile.md
It contains the framework name, route patterns, and path parameter syntax.

Use magaldi MCP tools (search_code, find_usages, pattern_search, find_files,
get_file_structure, etc.) instead of built-in Grep/Glob for code search.

Also use Context7 for detailed framework API docs:
```
mcp__plugin_context7_context7__resolve-library-id(libraryName="<framework>", query="HTTP route handler")
mcp__plugin_context7_context7__query-docs(libraryId="<id>", query="define HTTP routes GET POST PUT DELETE")
```

## Tasks

### 1. Create web route detector
**Create**: `src/magaldi_core/extractors/patterns/{lang}/web_routes.py`

Determine the pattern type from the profile:
- **Decorator-based** (like Python FastAPI, JS NestJS, Rust Actix): Maps decorator name → (HTTP method, framework)
  - Reference: `src/magaldi_core/extractors/patterns/python/web_routes.py`
- **Method-call-based** (like Express `app.get()`, Laravel `Route::get()`, Gin `r.GET()`): Walks AST for call patterns
  - Reference: `src/magaldi_core/extractors/patterns/php/laravel.py`

Structure:
```python
from magaldi_core.extractors.types import DecoratorInfo, HttpRoute

_{LANG}_HTTP_ROUTE_PATTERNS: dict[str, tuple[str, str]] = {{
    "<decorator>": ("GET", "<framework>"),
    ...
}}

def detect_{lang}_http_routes(decorators: list[DecoratorInfo]) -> list[HttpRoute]:
    ...
```

Path parameter extraction — use framework-specific syntax:
- `{{param}}` — FastAPI, Actix-web, Spring, ASP.NET
- `:param` — Express, NestJS, Gin, Echo
- `<param>` — Flask, Rocket

### 2. Create package init
**Create**: `src/magaldi_core/extractors/patterns/{lang}/__init__.py`
```python
from magaldi_core.extractors.patterns.{lang}.web_routes import detect_{lang}_http_routes
__all__ = ["detect_{lang}_http_routes"]
```

### 3. Wire into api_detection.py
**File**: `src/magaldi_core/analysis/api_detection.py`
- Add import of `detect_{lang}_http_routes`
- Add elif branch in `detect_http_routes()`
- Add framework decorators to `_PUBLIC_API_DECORATORS` set

### 4. Register in patterns __init__.py
**File**: `src/magaldi_core/extractors/patterns/__init__.py`
- Add import and `__all__` entry

### 5. CLI command detection (if applicable)
If the profile lists a CLI framework:
**Create**: `src/magaldi_core/extractors/patterns/{lang}/cli_commands.py`
Reference: `src/magaldi_core/extractors/patterns/python/cli_commands.py`
Wire into `detect_cli_commands()` in `api_detection.py`.

### 6. AST-level route extraction (if needed)
If the framework uses method calls (not decorators):
- The detection function receives the AST `tree` instead of decorator info
- Walk tree for specific call patterns
- Wire into the parser's `parse()` method in `src/magaldi_core/parsers/{lang}.py`

## Validation
Run `make lint` to verify imports/formatting.
```

---

## Phase 4: Testing Subagent

```
You are creating test fixtures and unit tests for {language} support in Magaldi.

Read the language profile: /Users/dinnyosz/code/magaldi/plans/languages/{lang}-profile.md

Use magaldi MCP tools (search_code, find_usages, pattern_search, find_files,
get_file_structure, etc.) instead of built-in Grep/Glob for code search.

## Tasks

### 1. Create test fixture
**Create**: `tests/fixtures/languages/teatro_<name>.{ext}`

Follow the Teatro theme (theater management system). See existing fixtures:
- `tests/fixtures/languages/teatro_performers.py`
- `tests/fixtures/languages/teatro_orchestra.rs`

Must cover ALL element types the extractor supports:
- Class/struct with constructor and methods
- Free functions (sync and async)
- Constants and variables
- Import statements
- Decorators/attributes/annotations
- Doc comments
- Interface/trait (if applicable)
- Enum (if applicable)
- Type alias (if applicable)
- Framework route handlers (from top web framework)
- CLI command definitions (if CLI framework was implemented)

### 2. Write extractor tests
**Create**: `tests/test_{lang}_extractor.py`
Cover: element extraction, import extraction, call extraction, class members,
decorators, async detection, parameters, return types, doc comments.
Read `tests/test_rust_extractor.py` or `tests/test_php_extractor.py` for patterns.

### 3. Write parser tests
**Create**: `tests/test_{lang}_parser.py`
Cover: full parse pipeline, element ID generation, hierarchy (parent_id),
same-file call resolution, test detection (is_test flag).

### 4. Write framework pattern tests
**Create**: `tests/test_{lang}_patterns.py`
Cover: web route detection, CLI command detection, path parameter extraction,
public API detection for framework handlers.

### 5. Parser Lab regression tests
Create regression tests for each major construct:
```
mcp__magaldi__parser_lab_create_test(name="{lang}_basic_function", language="{lang}",
    code="...", expected={{"elements": [{{"type": "function", "name": "..."}}]}})
mcp__magaldi__parser_lab_create_test(name="{lang}_class_with_methods", ...)
mcp__magaldi__parser_lab_create_test(name="{lang}_async_function", ...)
mcp__magaldi__parser_lab_create_test(name="{lang}_framework_route", ...)
```
Run all: `mcp__magaldi__parser_lab_run_tests(filter="{lang}")`

### 6. Create tree-sitter query files (optional)
**Directory**: `src/magaldi_core/queries/{lang}/`
Create `.scm` files: `elements.scm`, `imports.scm`, `calls.scm`
Reference existing files in `queries/python/`, `queries/rust/`.

### 7. Run all tests
```bash
pytest tests/test_{lang}_extractor.py tests/test_{lang}_parser.py tests/test_{lang}_patterns.py -v
make lint
```

Fix any failures before finishing.

### 8. Run integrity check
Invoke `/check-magaldi-integrity` to verify:
- Summarization prompts handle the new language
- MCP tools surface the new language correctly
- Web UI displays elements properly
- No missing integration points
```

---

## Phase 5: Test Repos Subagent

```
You are finding and registering real-world test repositories for {language} in Magaldi's
test infrastructure.

Read the language profile: /Users/dinnyosz/code/magaldi/plans/languages/{lang}-profile.md

## Tasks

### 1. Find test repos (WebSearch)
Search: "most popular {language} open source projects github", "best {language} libraries github stars"

Select 3 repos following this tier convention:
| Tier | Purpose | Size |
|------|---------|------|
| 1 | Smoke test — small, well-structured | < 50 files |
| 2 | Pattern coverage — uses top framework | 50-500 files |
| 2 | Variety — different patterns, edge cases | 50-500 files |

Good traits: popular (many stars), uses top web/CLI framework, has tests, has doc comments,
has async code, has classes with methods, has interfaces/traits/enums.

### 2. Register in clone script
**File**: `tools/clone-test-repos.sh` → `REPOS` array
Add entries:
```bash
# {Language}
"<github-org>/<repo-name>|<tier>|{lang}"
```
Also update the `--help` text to include {lang} in the language list.

### 3. Register in parse script
**File**: `tools/parse-test-repos.sh` → `REPOS` array
Add matching entries (dirname only):
```bash
"<repo-dirname>|<tier>|{lang}"
```

### 4. Update repo-tester skill
**File**: `.claude/skills/magaldi-repo-tester/SKILL.md`
Add entries to the `Repo-to-Language Mapping` table.

### 5. Clone and parse
```bash
./tools/clone-test-repos.sh --lang {lang}
./tools/parse-test-repos.sh --lang {lang} --skip-ai
```

### 6. Validate (if parsing succeeds)
Use MCP tools to spot-check:
```
mcp__magaldi__get_repo_stats(scope="test-repo", repository="<repo-name>")
mcp__magaldi__search_code(query="main function", scope="test-repo", repository="<repo-name>")
```

If issues found, report them — the `/magaldi-repo-tester` skill handles systematic fixing.

## Output
Report: repos selected, URLs, tier assignments, element counts after parsing.
```

---

## Quick Reference: All Files to Touch

```
Modified files:
  pyproject.toml                                      # tree-sitter dependency
  src/magaldi_core/tree_sitter_manager.py              # grammar import, LANGUAGE_CONFIG, get_extractor, __all__
  src/magaldi_core/parsers/__init__.py                 # parser import + __all__
  src/magaldi_core/code_parser.py                      # PARSERS dict, parse_file(), TEST_PATH_PATTERNS, is_test_element()
  src/magaldi_core/discovery.py                        # SUPPORTED_EXTENSIONS, SUPPORTED_FILENAMES, SHEBANG_PATTERNS
  src/magaldi_core/parsers/base.py                     # doc comment support (if needed)
  src/magaldi_core/analysis/metrics.py                 # DECISION_NODES
  src/magaldi_core/analysis/concurrency.py             # ENV_VAR_PATTERNS, CONCURRENCY_PATTERNS
  src/magaldi_core/analysis/api_detection.py           # detect_http_routes(), detect_cli_commands(), _PUBLIC_API_DECORATORS
  src/magaldi_core/extractors/call_categorizer.py      # BUILTIN_METHODS, STDLIB_MODULES
  src/magaldi_core/extractors/patterns/__init__.py     # new pattern module imports
  src/magaldi_core/module_resolver.py                  # <Lang>ModuleResolver
  src/magaldi_core/scope_bindings.py                   # binding patterns
  tools/clone-test-repos.sh                            # add new repos to REPOS array
  tools/parse-test-repos.sh                            # add new repos to REPOS array
  .claude/skills/magaldi-repo-tester/SKILL.md          # add repos to mapping table

New files:
  plans/languages/<lang>-profile.md                    # language research output
  src/magaldi_core/extractors/<lang>.py                # or <lang>/ package
  src/magaldi_core/parsers/<lang>.py                   # parser
  src/magaldi_core/queries/<lang>/*.scm                # query files (optional)
  src/magaldi_core/extractors/patterns/<lang>/         # framework pattern detection
    __init__.py
    web_routes.py                                      # HTTP route detection
    cli_commands.py                                    # CLI command detection (if applicable)
  tests/fixtures/languages/teatro_<name>.<ext>         # test fixture
  tests/test_<lang>_extractor.py                       # extractor tests
  tests/test_<lang>_parser.py                          # parser tests
  tests/test_<lang>_patterns.py                        # framework pattern tests
```
