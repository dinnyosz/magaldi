---
name: magaldi-repo-tester
description: >
  Validate Magaldi's parser quality against external test repos, then fix issues.
  Session-based workflow: each run gets a date-based session, repos run in parallel
  subagents, findings tracked in structured session files, fixes coordinated from
  a main session log. Use after running parse-test-repos.sh.
---

# Magaldi Repo Tester

Validates parsing quality by comparing what Magaldi indexed against what the source code actually contains. When issues are found, launches per-language fixer subagents that use Parser Lab's TDD workflow to fix them.

Every run is a **session** — a coordinated set of per-repo analyses tracked by a main session log. Sessions are date-based and resumable.

## Prerequisites

1. Test repos cloned: `./tools/clone-test-repos.sh`
2. Test repos parsed: `./tools/parse-test-repos.sh` (or specific repos)
3. OpenSearch running with indexed data

All test repos use `scope: test-repo` and `repository: <dirname>`.

## Session Architecture

### File Structure

```
test_repos/
├── _sessions/
│   ├── 2026-03-10_001.md          # Main session log (coordinates full run)
│   ├── 2026-03-10_002.md          # Second run same day
│   └── ...
├── click/
│   ├── _sessions/
│   │   ├── 2026-03-10_001.md      # Per-repo findings for this session
│   │   └── ...
│   └── ... source code ...
├── got/
│   ├── _sessions/
│   │   └── 2026-03-10_001.md
│   └── ...
└── _results/                       # Existing parse results (untouched)
```

### Session Name Format

`YYYY-MM-DD_NNN` — date plus zero-padded sequence number. First run of the day = `_001`.

---

## How It Works

### Orchestrator vs Subagent Responsibilities

The orchestrator (you) does ONLY lightweight coordination:
- Generate session name, create initial session log (Step 0)
- Launch subagents and collect their responses
- Parse subagent return values to decide what to launch next
- Post progress updates to the user

ALL heavy work runs in subagents via the Task tool:
- **Detect subagents** (Step 1) — one per repo, does all MCP calls + source reading + Parser Lab
- **Triage subagent** (Step 2) — reads findings files, aggregates, updates session log
- **Fixer subagents** (Step 3) — one per language/category, does TDD fix cycle
- **Finalize subagent** (Step 4) — runs final tests, updates session log

This keeps the orchestrator's context window small and focused on coordination.

### Subagent Model & Budget

| Subagent | Model | max_turns | Rationale |
|----------|-------|-----------|-----------|
| Detect (per-repo) | `sonnet` | 30 | MCP calls + file reads + comparisons. Structured output, no invention. |
| Triage | `haiku` | 15 | Read files, aggregate counts, update markdown tables. Pure data wrangling. |
| Language Fixer | (default/opus) | — | TDD cycle: diagnose → write test → fix parser. Needs deep reasoning. |
| Prompts Fixer | `sonnet` | 20 | Editing prompt strings. Straightforward text changes. |
| Call Resolution Fixer | (default/opus) | — | Complex call resolution logic, strategy analysis. |
| Finalize | `haiku` | 10 | Run tests, tally counts, update markdown. Pure bookkeeping. |

### Two-Phase Architecture

**Phase A — Detect (per-repo):** Launch one subagent per test repo. Each subagent samples indexed elements, compares against source, runs Parser Lab spot-checks, and writes findings to `{repo}/_sessions/{session}.md`.

**Phase B — Fix (per-language + per-category):** Triage subagent aggregates issues from all repo findings. Fixer subagents launched per language/category. Each fixer works from the session log and updates it as tasks complete.

### Modes

The user can request:
- **detect only** ("just detect", "report only", "audit") → Phase A + triage only
- **full** (default) → Phase A → triage → Phase B → finalize

---

## Step 0: Generate Session (orchestrator — lightweight)

1. Determine today's date in `YYYY-MM-DD` format
2. List existing files in `test_repos/_sessions/` matching today's date
3. Increment the sequence: if `2026-03-10_001.md` exists, next is `2026-03-10_002`
4. Create directories if needed:
   ```bash
   mkdir -p test_repos/_sessions
   ```
5. Create the main session log file with initial content:

```markdown
# Session: {session_name}

**Started:** {ISO timestamp}
**Status:** detecting
**Mode:** {detect_only|full}
**Repos:** {comma-separated repo list, or "tier N" / "all"}

## Repos

| Repo | Language | Status | Issues | Findings File |
|------|----------|--------|--------|---------------|
| {repo1} | {lang1} | pending | - | {repo1}/_sessions/{session_name}.md |
| {repo2} | {lang2} | pending | - | {repo2}/_sessions/{session_name}.md |

## Triage Summary

(Populated after all repos complete)

## Fix Log

(Populated during fix phase)
```

---

## Step 1: Detect

### 1a. List Parsed Test Repos (orchestrator — one MCP call)

```
mcp__magaldi__list_repos()
```

Filter for repos with `scope: test-repo`. If none found, tell the user to run `./tools/parse-test-repos.sh` first. Use the result to build the repo list for the session log and subagent launches.

### 1b. Launch Detect Subagents (one per repo)

Use the Task tool with `subagent_type: "general-purpose"`, `model: "sonnet"`, `max_turns: 30` for each repo. Launch 2-3 in parallel.

Each detect subagent gets this prompt:

```
You are validating Magaldi's parser quality for the "{repo_name}" repository ({language}).
The repo is indexed with scope="test-repo", repository="{repo_name}".
The source code is at: /Users/dinnyosz/code/magaldi/test_repos/{repo_name}/

Session: {session_name}

Use magaldi MCP tools (search_code, find_usages, pattern_search, find_files,
get_element, get_call_graph, get_repo_stats, etc.) instead of built-in Grep/Glob.

## Your Task

### Phase 1: Overview
1. Call `get_repo_stats` with scope="test-repo", repository="{repo_name}" to understand what was indexed
2. Note element counts per type (function, class, method, etc.)
3. Flag any element types with 0 count that should exist for {language}

### Phase 2: Sample and Validate (10-20 elements)
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

### Phase 3: Parser Lab Spot-Checks
1. Use the Glob tool to list source files matching the language extension
   (e.g., `**/*.py`, `**/*.js`, `**/*.rs`) in the repo directory
2. Exclude test files (paths containing `test`, `spec`, `__test__`, `fixtures`),
   vendored code (`vendor/`, `node_modules/`, `third_party/`), and generated files
3. Pick 6-10 files at random, spread across different directories
4. For each file:
   a. Read the file with the Read tool
   b. Run `parser_lab_analyze` with the file contents and `language="{language}"`
   c. Compare Parser Lab results against source — any missing elements are `parser_lab_gap`

### Phase 4: Structural Checks
1. Pick 4-6 source files at random (Read them), look for:
   - Functions/classes that exist in source but are NOT indexed
   - Imports that aren't captured
   - Constants/variables that should have been extracted
2. Check class hierarchy: pick a class, verify its methods are indexed as children

### Phase 5: Write Findings File

Create the per-repo findings file at:
`/Users/dinnyosz/code/magaldi/test_repos/{repo_name}/_sessions/{session_name}.md`

Create the `_sessions` directory first if it doesn't exist:
```bash
mkdir -p /Users/dinnyosz/code/magaldi/test_repos/{repo_name}/_sessions
```

Use this format:

```markdown
# Findings: {repo_name} — Session {session_name}

**Repo:** {repo_name}
**Language:** {language}
**Scope:** test-repo
**Analyzed:** {ISO timestamp}
**Elements indexed:** {total}
**Issues found:** {count}

## Stats

| Type | Count |
|------|-------|
| function | N |
| class | N |
| method | N |
| ... | ... |

## Issues

ISSUES:
- category: {category}
  language: {language}
  ...
(use the same YAML issue format documented below)

## Samples Checked

- `{element_id_or_name}` — {brief result}
- ...

## Parser Lab Spot-Checks

- `{file_path}` — {N elements found, M gaps}
- ...
```

### Phase 6: Return Summary

Return a brief summary at the end of your response:
- Total issues found
- Issue breakdown by category
- The path to the findings file you wrote

Also include the YAML issue list as the LAST thing in your response (parsed by orchestrator).

## Issue Categories (YAML format)

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

- category: parser_lab_gap
  language: {language}
  file: "{relative_path_within_repo}"
  element: "{element_name}"
  type: "{expected_type}"
  line: {line_number}
  code_snippet: |
    {the code that parser_lab_analyze failed to extract}
  details: "{element visible in source but parser_lab_analyze did not extract it — parser gap}"
```

IMPORTANT: The structured YAML issue list must be the LAST thing in the subagent response.
```

### 1c. Update Main Session Log (orchestrator — lightweight)

As each detect subagent returns:
1. Extract the issue count from the subagent's response (it returns this in its summary)
2. Update the repo's row in the main session log file:
   - Status: `completed`
   - Issues: count
3. If a subagent fails, mark status as `failed` with error note

This is a quick file edit — the subagent already wrote the detailed findings file.

---

## Step 2: Triage (subagent)

After ALL detect subagents complete, launch a **triage subagent** to aggregate and prioritize findings. This keeps the orchestrator's context clean.

Use the Task tool with `subagent_type: "general-purpose"`, `model: "haiku"`, `max_turns: 15`:

```
You are triaging parser quality findings from a Magaldi repo tester session.
Session: {session_name}
Main session log: /Users/dinnyosz/code/magaldi/test_repos/_sessions/{session_name}.md

## Your Task

1. Read the main session log to get the list of repos and their findings files
2. Read each per-repo findings file from `{repo}/_sessions/{session_name}.md`
3. Parse the YAML issue blocks from each file
4. Aggregate issues by language and category
5. Deduplicate: same missing element found by multiple checks → keep one with most detail
6. Assign priorities:
   - HIGH: issues affecting many elements or blocking core functionality
   - MEDIUM: systematic gaps in a language
   - LOW: cosmetic, single-instance, or edge-case issues
7. Determine which fixers are needed:

| Condition | Fixer to launch |
|-----------|-----------------|
| Any `missing_element`, `wrong_element_type`, `missing_params`, `parser_lab_gap` for language X | Language X fixer |
| Any `bad_summary` issues | Prompts fixer |
| Any `missing_calls`, `phantom_calls` issues | Call resolution fixer |
| `missing_element_type_for_language` | Language X fixer |

8. Update the main session log:
   - Set **Status** to `triage`
   - Populate the **Triage Summary** table:

| # | Category | Language | Count | Priority | Fix Status |
|---|----------|----------|-------|----------|------------|
| 1 | {category description} | {lang} | {count} | {HIGH|MEDIUM|LOW} | pending |

   - Initialize the **Fix Log** section with empty table headers

9. Return a structured summary listing which fixers to launch and their issue payloads (YAML).
   Format:

```yaml
TRIAGE:
  fixers_needed:
    - type: language_fixer
      language: {language}
      issue_count: N
      issues: |
        {YAML issues block for this language}
    - type: prompts_fixer
      issue_count: N
      issues: |
        {YAML issues block}
    - type: call_resolution_fixer
      issue_count: N
      issues: |
        {YAML issues block}
  total_issues: N
  clean_repos: [{repo names with 0 issues}]
```

If no issues found, return `total_issues: 0` and set session status to `completed`.
```

After the triage subagent completes:
- If `total_issues: 0` → report clean bill of health, stop
- If mode is `detect_only` → mark session `completed`, stop here
- Otherwise → proceed to Step 3 using the fixer payloads from the triage response

---

## Step 3: Fix

Update main session log status to `fixing`.

Launch fixer subagents in parallel (up to 3 at a time). Each fixer type has its own prompt template.

### 3a. Language Fixer Subagent

One per language that had parser-level issues.

```
You are fixing Magaldi's {language} parser based on issues found in test repos.
Session: {session_name}
Main session log: /Users/dinnyosz/code/magaldi/test_repos/_sessions/{session_name}.md

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
Confirm it fails for the right reason. If it passes → skip (false positive or already fixed).

### 4. Get a Fix Suggestion
Use `parser_lab_suggest_fix`:
- `gap_description`: describe what's not extracted correctly
- `language`: "{language}"
- `failing_test`: path to the failing test file (from step 2 output)

### 5. Apply the Fix
Read the suggested file(s), then use Edit to make the changes.

### 6. Verify the Test Passes
Run `parser_lab_run_tests` with `filter` set to the test name.

### 7. Regression Check
Run `parser_lab_run_tests` WITHOUT a filter to ensure no existing tests broke.

## Updating the Session Log

After each issue is fixed (or skipped), update the main session log's Fix Log table:

Read the current session log, find the relevant triage row, and update its Fix Status.
Add a row to the Fix Log section:

| # | Task | Fixer | Status | Files Modified | Details |
|---|------|-------|--------|----------------|---------|
| N | {brief description} | {language}-parser | {fixed|skipped|partial} | {files} | {what changed} |

## Output

Return a summary:

```yaml
FIXES:
- issue: "{brief description}"
  status: fixed|skipped|partial
  test_name: "{test name created}"
  files_modified: ["{file1}", "{file2}"]
  details: "{what was changed}"
```
```

### 3b. Prompts Fixer Subagent

Launched when `bad_summary` issues are found. Use `model: "sonnet"`, `max_turns: 20`.

```
You are fixing Magaldi's summarization prompt quality based on issues found in test repos.
Session: {session_name}
Main session log: /Users/dinnyosz/code/magaldi/test_repos/_sessions/{session_name}.md

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
3. Strengthen the anti-verbose instruction if needed
4. Run `make test-fast` to verify no test breaks

### For Inaccurate/Vague Summaries
1. Use `prompt_lab_improve` with the element_id
2. Review the optimized prompt
3. Apply systemic changes to `prompts.py` if applicable

## Updating the Session Log

After each fix, update the main session log Fix Log table (same pattern as language fixer).

## Output

```yaml
FIXES:
- issue: "{element_name}: {problem}"
  status: fixed|skipped|needs_reindex
  files_modified: ["{file1}"]
  details: "{what was changed}"
```
```

### 3c. Call Resolution Fixer Subagent

Launched when `missing_calls` or `phantom_calls` issues are found.

```
You are fixing Magaldi's call resolution logic based on issues found in test repos.
Session: {session_name}
Main session log: /Users/dinnyosz/code/magaldi/test_repos/_sessions/{session_name}.md

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
1. Read the source code at the element's location
2. Identify the call pattern (bare, method, module-qualified, chained, etc.)
3. Check which resolution strategy should handle it (Strategies 1-6)
4. Use `parser_lab_analyze` to check if the call is even extracted
5. Diagnose and fix the gap
6. Test with `parser_lab_run_tests`

## Updating the Session Log

After each fix, update the main session log Fix Log table.

## Output

```yaml
FIXES:
- issue: "{element_name}: missing calls [{calls}]"
  status: fixed|skipped|partial
  root_cause: "call_extraction|import_mapping|type_resolution|strategy_gap"
  files_modified: ["{file1}"]
  details: "{what was changed}"
```
```

---

## Step 4: Finalize (subagent)

After all fixer subagents complete, launch a **finalize subagent** to wrap up.

Use the Task tool with `subagent_type: "general-purpose"`, `model: "haiku"`, `max_turns: 10`:

```
You are finalizing a Magaldi repo tester session.
Session: {session_name}
Main session log: /Users/dinnyosz/code/magaldi/test_repos/_sessions/{session_name}.md

## Your Task

1. Read the main session log
2. Run `parser_lab_run_tests` to ensure all tests pass (including new ones from fixers)
3. Tally final counts: issues found, fixed, skipped, partial
4. Update the main session log:
   - Set **Status** to `completed`
   - Add **Completed:** timestamp at the bottom
   - Ensure all Fix Log rows have final statuses
5. Return a brief summary for the user:
   - Total repos analyzed
   - Total issues found
   - Issues fixed / skipped / partial
   - Any failing tests
   - Whether structural changes were made (triggers integrity check reminder)
```

After the finalize subagent completes, report the summary to the user.

If structural changes were made (new element types, new fields), remind the user to run `/check-magaldi-integrity`.

---

## Targeting Specific Repos

The user can specify which repos to test:
- "test click and express" → only those two
- "test all rust repos" → fd, ripgrep, bat, tokio, serde, ruff, zellij
- "test tier 1" → smoke test repos only
- "just detect python repos" → detect-only mode for Python repos

Use the repo list below to map language/tier filters.

## Repo-to-Language Mapping

| Repo | Language | Tier | Type |
|------|----------|------|------|
| click | python | 1 | library |
| requests | python | 2 | library |
| httpx | python | 2 | library |
| fastapi | python | 3 | library |
| pydantic | python | 3 | library |
| full-stack-fastapi-template | python | 3 | app |
| core (home-assistant) | python | 3 | app |
| got | javascript | 1 | library |
| express | javascript | 2 | library |
| lodash | javascript | 2 | library |
| axios | javascript | 3 | library |
| date-fns | javascript | 3 | library |
| Ghost | javascript | 3 | app |
| nodebestpractices | javascript | 3 | app |
| zod | typescript | 1 | library |
| trpc | typescript | 2 | library |
| drizzle-orm | typescript | 2 | library |
| prisma | typescript | 3 | library |
| typeorm | typescript | 3 | library |
| cal.com | typescript | 3 | app |
| immich | typescript | 3 | app |
| guzzle | php | 1 | library |
| composer | php | 2 | app |
| PHPMailer | php | 2 | library |
| framework (laravel) | php | 3 | library |
| console (symfony) | php | 3 | library |
| firefly-iii | php | 3 | app |
| matomo | php | 3 | app |
| fd | rust | 1 | app |
| ripgrep | rust | 2 | app |
| bat | rust | 2 | app |
| tokio | rust | 3 | library |
| serde | rust | 3 | library |
| ruff | rust | 3 | app |
| zellij | rust | 3 | app |
| gson | java | 1 | library |
| spring-petclinic | java | 2 | app |
| okhttp | java | 2 | library |
| commons-lang | java | 3 | library |
| junit5 | java | 3 | library |
| java-design-patterns | java | 3 | app |
| kafka | java | 3 | app |
| neofetch | bash | 1 | app |
| rbenv | bash | 2 | app |
| nvm | bash | 2 | app |
| ohmyzsh | bash | 3 | library |
| asdf | bash | 3 | library |
| pi-hole | bash | 3 | app |
| dokku | bash | 3 | app |
| nickel | polyglot | 2 | app |

## Key Things to Catch

1. **Missing elements**: Functions/classes in source that weren't indexed
2. **Wrong element type**: Function classified as method or vice versa
3. **Bad summaries**: Starting with "This function/class/module..."
4. **Missing parameters**: Parameters present in source but not indexed
5. **Missing return types**: Return type annotation present but not captured
6. **Call resolution gaps**: Function calls in source but not in indexed callees
7. **Phantom callers**: Indexed caller relationships that don't exist in source
8. **Parser Lab gaps**: Elements visible in source that `parser_lab_analyze` fails to extract
9. **Import gaps**: Import statements not captured
10. **Class hierarchy**: Methods not properly linked to parent class

## Integrity Check

**IMPORTANT**: If fixer subagents make structural changes (new element types, new fields in extractors), you MUST run `/check-magaldi-integrity` after all fixes are applied.
