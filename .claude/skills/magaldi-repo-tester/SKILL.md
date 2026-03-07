---
name: magaldi-repo-tester
description: >
  Validate Magaldi's parser quality against external test repos, then fix issues.
  Two-phase workflow: detect gaps (per-repo), then fix them (per-language) using
  Parser Lab TDD and Prompt Lab. Use after running parse-test-repos.sh.
---

# Magaldi Repo Tester

Validates parsing quality by comparing what Magaldi indexed against what the source code actually contains. When issues are found, launches per-language fixer subagents that use Parser Lab's TDD workflow to fix them.

## Prerequisites

1. Test repos cloned: `./tools/clone-test-repos.sh`
2. Test repos parsed: `./tools/parse-test-repos.sh` (or specific repos)
3. OpenSearch running with indexed data

All test repos use `scope: test-repo` and `repository: <dirname>`.

## How It Works

### Two-Phase Architecture

**Phase A — Detect (per-repo):** Launch subagents per test repo. Each samples elements, compares indexed data vs actual source, and produces a structured issue list categorized by fix type.

**Phase B — Fix (per-language + per-category):** Aggregate issues by language. Launch one fixer subagent per language that had issues. Also launch dedicated fixers for cross-cutting concerns (summarization prompts, call resolution). Each fixer uses Parser Lab's TDD cycle: reproduce → write test → fix → verify.

Why per-language for fixes? A missing pattern in `queries/python/elements.scm` affects ALL Python repos. One subagent per language prevents conflicting edits to the same files.

## Modes

The user can request:
- **detect only** ("just detect", "report only", "audit") → Run Phase A only, produce reports
- **full** (default) → Run Phase A, then Phase B, then validate

---

## Step 1: Detect

### 1a. List Parsed Test Repos

```
mcp__magaldi__list_repos()
```

Filter for repos with `scope: test-repo`. If none found, tell the user to run `./tools/parse-test-repos.sh` first.

### 1b. Launch Detect Subagents

Use the Task tool with `subagent_type: "general-purpose"` for each repo. Launch 2-3 in parallel.

Each detect subagent gets this prompt:

```
You are validating Magaldi's parser quality for the "{repo_name}" repository ({language}).
The repo is indexed with scope="test-repo", repository="{repo_name}".
The source code is at: /Users/dinnyosz/code/magaldi/test_repos/{repo_name}/

Use magaldi MCP tools (search_code, find_usages, pattern_search, find_files,
get_element, get_call_graph, get_repo_stats, etc.) instead of built-in Grep/Glob.

## Your Task

### Phase 1: Overview
1. Call `get_repo_stats` with scope="test-repo", repository="{repo_name}" to understand what was indexed
2. Note element counts per type (function, class, method, etc.)
3. Flag any element types with 0 count that should exist for {language}

### Phase 2: Sample and Validate (5-10 elements)
For each sample:

1. Use `search_code` with a generic query (e.g., "handle", "process", "create", "validate")
   to get random functions. Use `brief=false` and `include_code=true`.
2. Use `get_element` with `brief=false, include_code=true` to get full indexed data
3. Use `get_call_graph` to get callers and callees
4. Read the actual source file with the Read tool at the element's line range
5. Compare:
   - **Calls made**: Does the source code call functions that aren't in the indexed callees?
   - **Callers**: Are there phantom callers (indexed but don't exist in source)?
   - **Summary quality**: Is the summary accurate? Does it start with an action verb (not "This function...")?
   - **Parameters**: Are all parameters captured? Types correct?
   - **Return type**: Is it captured?
   - **Element type**: Is function vs method classification correct?

### Phase 3: Structural Checks
1. Pick 2-3 source files at random (Read them), look for:
   - Functions/classes that exist in source but are NOT indexed
   - Imports that aren't captured
   - Constants/variables that should have been extracted
2. Check class hierarchy: pick a class, verify its methods are indexed as children

### Phase 4: Produce Issue List

Return your findings as a structured issue list. Use this EXACT format — one YAML block per issue.
ONLY include real issues you confirmed. Do not fabricate issues.

```yaml
ISSUES:
- category: missing_element
  language: {language}
  file: "{relative_path_within_repo}"
  element: "{element_name}"
  type: "{expected_type: function|class|method|constant|variable|import|interface|trait|enum|type_alias}"
  line: {line_number}
  code_snippet: |
    {the actual code of the missing element, 5-15 lines}
  details: "{why it should be indexed}"

- category: wrong_element_type
  language: {language}
  element_id: "{element_id or hash_id}"
  actual_type: "{what magaldi says}"
  expected_type: "{what it should be}"
  details: "{explanation}"

- category: bad_summary
  language: {language}
  element_id: "{element_id or hash_id}"
  current_summary: "{the bad summary}"
  details: "{what's wrong — e.g. starts with 'This function...', inaccurate, too vague}"

- category: missing_calls
  language: {language}
  element_id: "{element_id or hash_id}"
  element_name: "{name}"
  file: "{relative_path}"
  missing: ["{call1}", "{call2}"]
  code_snippet: |
    {the code showing the calls that should be indexed}
  details: "{context on what calls are missing}"

- category: phantom_calls
  language: {language}
  element_id: "{element_id or hash_id}"
  phantom: ["{call1}", "{call2}"]
  details: "{these callers/callees are indexed but don't exist in source}"

- category: missing_params
  language: {language}
  element_id: "{element_id or hash_id}"
  element_name: "{name}"
  missing_params: ["{param1}", "{param2}"]
  missing_return_type: true|false
  code_snippet: |
    {the function signature showing the missing params/types}
  details: "{what's missing}"

- category: missing_element_type_for_language
  language: {language}
  expected_types: ["{type1}", "{type2}"]
  details: "{language should have these types but 0 were indexed}"
```

Also write a human-readable report to: /Users/dinnyosz/code/magaldi/test_repos/{repo_name}/_quality_report.md

The report should include:
- Stats summary (elements indexed, type breakdown)
- Tables for each issue category (same as the issues above, but in markdown table format)
- Recommendations section

IMPORTANT: The structured YAML issue list must be the LAST thing in your response,
after all analysis is complete. This is parsed by the orchestrator.
```

---

## Step 2: Triage

After all detect subagents complete, aggregate their issues:

1. Parse the YAML issue lists from each subagent's response
2. Group issues by language and category
3. Determine which fixers to launch:

| Condition | Fixer to launch |
|-----------|-----------------|
| Any `missing_element`, `wrong_element_type`, `missing_params` for language X | Language X fixer |
| Any `bad_summary` issues | Prompts fixer |
| Any `missing_calls`, `phantom_calls` issues | Call resolution fixer |
| `missing_element_type_for_language` | Language X fixer |

4. If no issues found → report clean bill of health and stop

Skip fixers for categories with fewer than 1 issue (not worth spawning a subagent for).

---

## Step 3: Fix

Launch fixer subagents in parallel (up to 3 at a time). Each fixer type has its own prompt template.

### 3a. Language Fixer Subagent

One per language that had parser-level issues (`missing_element`, `wrong_element_type`, `missing_params`, `missing_element_type_for_language`).

```
You are fixing Magaldi's {language} parser based on issues found in test repos.

Use magaldi MCP tools (search_code, find_usages, pattern_search, find_files,
get_element, get_call_graph, get_repo_stats, etc.) instead of built-in Grep/Glob.

## Files You May Modify
- Tree-sitter queries: src/magaldi_core/queries/{language}/*.scm
- Extractors: src/magaldi_core/extractors/{language}/*.py
- Language parser: src/magaldi_core/parsers/{language}.py
- Test fixtures: tests/fixtures/languages/example_{language}.*

Do NOT modify files outside your language scope.

## Issues to Fix

{paste the aggregated issues for this language here, YAML format}

## Fix Workflow (TDD with Parser Lab)

For EACH issue, follow this cycle:

### 1. Reproduce
Use `parser_lab_analyze` to confirm the gap:
- Pass the `code_snippet` from the issue as the `code` parameter
- Set `language` to "{language}"
- Check the gap analysis output — does it confirm the element is missing or wrong?

### 2. Write a Failing Test
Use `parser_lab_create_test` to create a test that captures expected behavior:
- `name`: descriptive slug, e.g., "python_async_generator_function"
- `language`: "{language}"
- `code`: the code snippet from the issue
- `expected`: what SHOULD be extracted (elements list with type+name, calls, etc.)

### 3. Verify the Test Fails
Run `parser_lab_run_tests` with `filter` set to the test name.
Confirm it fails for the right reason (missing element, wrong type, etc.).
If it passes → the issue may already be fixed or was a false positive. Skip it.

### 4. Get a Fix Suggestion
Use `parser_lab_suggest_fix`:
- `gap_description`: describe what's not extracted correctly
- `language`: "{language}"
- `failing_test`: path to the failing test file (from step 2 output)

Review the suggestion. It will point to specific .scm queries or extractor code.

### 5. Apply the Fix
Read the suggested file(s), then use Edit to make the changes.
Common fix patterns:
- **Missing element**: Add a new S-expression pattern to `elements.scm`
- **Wrong type**: Fix classification logic in `element_extractor.py`
- **Missing params**: Fix parameter extraction in `element_extractor.py` or `utils.py`
- **Missing return type**: Fix return type extraction in the extractor

### 6. Verify the Test Passes
Run `parser_lab_run_tests` with `filter` set to the test name.
If it still fails, iterate on the fix.

### 7. Regression Check
Run `parser_lab_run_tests` WITHOUT a filter to ensure no existing tests broke.
If regressions found, fix them before moving on.

## Output

After fixing all issues, return a summary:

```yaml
FIXES:
- issue: "{brief description}"
  status: fixed|skipped|partial
  test_name: "{test name created}"
  files_modified: ["{file1}", "{file2}"]
  details: "{what was changed}"

- issue: "..."
  status: skipped
  reason: "{why — false positive, already fixed, too complex, etc.}"
```
```

### 3b. Prompts Fixer Subagent

Launched when `bad_summary` issues are found.

```
You are fixing Magaldi's summarization prompt quality based on issues found in test repos.

Use magaldi MCP tools (search_code, find_usages, pattern_search, find_files,
get_element, etc.) instead of built-in Grep/Glob.

## Files You May Modify
- Summarization prompts: src/shared/ai/prompts.py
- Summarization client: src/shared/ai/summarization.py
- Processor helpers: src/magaldi_core/processor/helpers.py

Do NOT modify parser or extractor files.

## Issues to Fix

{paste all bad_summary issues here, YAML format}

## Fix Workflow

### For Anti-Verbose Violations ("This function...", "This class...")
1. Read `src/shared/ai/prompts.py`
2. Find the SYSTEM_PROMPTS and PROMPTS entries for the affected element type
3. Check if the anti-verbose instruction is present and clear
4. If missing or weak, strengthen it. Pattern:
   "Start with an action verb — never start with 'This [type]...', 'The [type]...', or similar."
5. Run `make test-fast` to verify no test breaks

### For Inaccurate/Vague Summaries
1. Use `prompt_lab_improve` with the element_id of the element that has a bad summary
   - Set `max_iterations` to 5
   - Set `target_score` to 8
2. Review the optimized prompt — it shows what changes improved the score
3. If the improvement suggests a systemic prompt change (not element-specific),
   apply the learning to the relevant prompt template in `prompts.py`
4. If the issue is element-specific (just needs re-processing), note it as "needs re-index"

## Output

```yaml
FIXES:
- issue: "{element_name}: {problem}"
  status: fixed|skipped|needs_reindex
  files_modified: ["{file1}"]
  details: "{what was changed in the prompt}"
  prompt_lab_score: {before} -> {after}

- issue: "..."
  status: needs_reindex
  reason: "Prompt is fine, element just needs re-processing"
```
```

### 3c. Call Resolution Fixer Subagent

Launched when `missing_calls` or `phantom_calls` issues are found.

```
You are fixing Magaldi's call resolution logic based on issues found in test repos.

Use magaldi MCP tools (search_code, find_usages, pattern_search, find_files,
get_element, get_call_graph, etc.) instead of built-in Grep/Glob.

## Files You May Modify
- Call resolution: src/magaldi_core/call_resolution.py
- Module resolver: src/magaldi_core/module_resolver.py
- Scope bindings: src/magaldi_core/scope_bindings.py
- Call extractors: src/magaldi_core/extractors/{language}/call_extractor.py

Do NOT modify summarization or unrelated parser files.

## Issues to Fix

{paste all missing_calls and phantom_calls issues here, YAML format}

## Analysis Workflow

For each issue:

1. **Understand the call pattern** — Read the source code at the element's location.
   Identify what kind of call is being missed:
   - Bare function call: `foo()`
   - Method call on self: `self.foo()`
   - Method call on typed var: `repo.get()`
   - Module-qualified call: `utils.foo()`
   - Import-aliased call: `from x import y as z; z()`
   - Chained call: `obj.method().another()`
   - Call within comprehension/lambda

2. **Check which strategy should resolve it** — The 6 strategies:
   - Strategy 1-2: Same-file bare calls + self-method calls (at parse time)
   - Strategy 3: Import-based (`from utils import foo; foo()`)
   - Strategy 4: Module method (`import utils; utils.foo()`)
   - Strategy 5: Type-annotated (`repo: Repository; repo.get_document()`)
   - Strategy 6: Embedding-based RRF (semantic similarity fallback)

3. **Check call extraction** — Use `parser_lab_analyze` on the source code.
   Does the call even appear in the extracted calls list? If not, the issue is
   in the call extractor (`extractors/{language}/call_extractor.py`), not resolution.

4. **Diagnose the resolution failure** — If the call IS extracted but not resolved:
   - Check import map: is the import captured?
   - Check type annotations: is the variable typed?
   - Check if the target element exists in the index

5. **Apply fix** — Common patterns:
   - **Missing call extraction**: Fix the call extractor's AST traversal
   - **Import not mapped**: Fix `_build_import_map()` in call_resolution.py
   - **Type annotation not followed**: Fix Strategy 5 logic
   - **Phantom caller**: Fix over-eager matching in resolution strategies

6. **Test** — Use `parser_lab_run_tests` for call-related tests.
   Also use `parser_lab_analyze` on the code snippet to verify calls are now extracted.

## Output

```yaml
FIXES:
- issue: "{element_name}: missing calls [{calls}]"
  status: fixed|skipped|partial
  root_cause: "call_extraction|import_mapping|type_resolution|strategy_gap"
  strategy_affected: "{1-6}"
  files_modified: ["{file1}"]
  details: "{what was changed}"

- issue: "..."
  status: skipped
  reason: "{why — e.g., requires embedding re-index, semantic resolution, etc.}"
```
```

---

## Step 4: Validate

After all fixers complete, re-validate to confirm improvements:

1. Run `parser_lab_run_tests` to ensure all tests pass (including new ones from fixers)
2. If the user wants full validation: re-parse affected test repos and re-run detect subagents
3. Compare before/after issue counts

---

## Step 5: Report

Produce a final summary combining detect and fix results:

```markdown
## Repo Tester Results

### Detection Summary

| Repo | Lang | Elements | Issues Found |
|------|------|----------|--------------|
| click | python | 150 | 5 |
| express | javascript | 200 | 8 |

### Issues by Category

| Category | Count | Fixed | Skipped |
|----------|-------|-------|---------|
| missing_element | 4 | 3 | 1 |
| bad_summary | 6 | 4 | 2 |
| missing_calls | 3 | 2 | 1 |
| missing_params | 2 | 2 | 0 |

### Fixes Applied

| Fix | Files Modified | Test Created |
|-----|---------------|--------------|
| Python: added async generator pattern to elements.scm | queries/python/elements.scm | test_python_async_generator |
| Prompts: strengthened anti-verbose for class type | src/shared/ai/prompts.py | - |

### Remaining Issues (Skipped)
1. {issue}: {reason skipped}

### Integrity Check Reminder
If any structural changes were made (new element types, new fields), run:
`/check-magaldi-integrity`
```

Write report to: `/Users/dinnyosz/code/magaldi/test_repos/_repo_tester_report.md`

---

## Targeting Specific Repos

The user can specify which repos to test:
- "test click and express" → only those two
- "test all rust repos" → fd, ripgrep, bat
- "test tier 1" → smoke test repos only
- "just detect python repos" → detect-only mode for Python repos

Use the repo list from `tools/clone-test-repos.sh --list` to map language/tier filters.

## Repo-to-Language Mapping

| Repo | Language | Tier |
|------|----------|------|
| click | python | 1 |
| requests | python | 2 |
| httpx | python | 2 |
| got | javascript | 1 |
| express | javascript | 2 |
| lodash | javascript | 2 |
| zod | typescript | 1 |
| trpc | typescript | 2 |
| drizzle-orm | typescript | 2 |
| guzzle | php | 1 |
| composer | php | 2 |
| PHPMailer | php | 2 |
| fd | rust | 1 |
| ripgrep | rust | 2 |
| bat | rust | 2 |
| gson | java | 1 |
| spring-petclinic | java | 2 |
| okhttp | java | 2 |
| neofetch | bash | 1 |
| rbenv | bash | 2 |
| nvm | bash | 2 |
| nickel | polyglot | 2 |

## Key Things to Catch

1. **Call resolution gaps**: Function calls `foo()` in source but `foo` not in indexed callees
2. **Phantom callers**: Indexed caller relationships that don't exist in source
3. **Missing elements**: Functions/classes in source code that weren't indexed at all
4. **Wrong element type**: Function classified as method or vice versa
5. **Bad summaries**: Starting with "This function/class/module..." instead of action verb
6. **Missing parameters**: Parameters present in source but not in indexed element
7. **Missing return types**: Return type annotation present but not captured
8. **Import gaps**: Import statements not captured
9. **Constant/variable gaps**: Module-level constants not extracted
10. **Class hierarchy**: Methods not properly linked to parent class

## Integrity Check

**IMPORTANT**: If fixer subagents make structural changes (new element types, new fields in extractors), you MUST run `/check-magaldi-integrity` after all fixes are applied. This ensures changes propagate to Web UI, MCP tools, and summarization.
