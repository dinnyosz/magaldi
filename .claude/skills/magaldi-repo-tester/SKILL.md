---
name: magaldi-repo-tester
description: >
  Validate Magaldi's parser quality against external test repos.
  After repos are parsed, picks random functions, compares indexed callers/callees
  against actual source code, and reports gaps. Use after running parse-test-repos.sh.
---

# Magaldi Repo Tester

Validates parsing quality by comparing what Magaldi indexed against what the source code actually contains. Designed to catch parser gaps, missing call resolution, bad summaries, and missing element types.

## Prerequisites

1. Test repos cloned: `./tools/clone-test-repos.sh`
2. Test repos parsed: `./tools/parse-test-repos.sh` (or specific repos)
3. OpenSearch running with indexed data

All test repos use `scope: test-repo` and `repository: <dirname>`.

## How It Works

For each parsed test repo, this skill launches a subagent that:

1. **Discovers what's indexed** — queries Magaldi for the repo's element counts by type
2. **Samples random functions/methods** — picks 5-10 elements at random
3. **For each sampled element:**
   a. Gets the indexed data (summary, callers, callees, parameters, return type)
   b. Reads the actual source file to see the real code
   c. Compares indexed callers/callees against what the code actually calls
   d. Checks if the summary is accurate and non-generic
   e. Checks if parameters and return types are captured correctly
4. **Reports gaps** — missing calls, phantom callers, bad summaries, missing elements

## Invocation

When the user asks to validate test repos, run the procedure below.

### Step 1: List Parsed Test Repos

```
mcp__magaldi__list_repos()
```

Filter for repos with `scope: test-repo`. If none found, tell the user to run `./tools/parse-test-repos.sh` first.

### Step 2: For Each Repo, Launch a Subagent

Use the Task tool with `subagent_type: "general-purpose"` for each repo. Launch repos in parallel where possible (2-3 at a time).

Each subagent gets this prompt template:

```
You are validating Magaldi's parser quality for the "{repo_name}" repository ({language}).
The repo is indexed with scope="test-repo", repository="{repo_name}".
The source code is at: /Users/dinnyosz/code/magaldi/test_repos/{repo_name}/

Use magaldi MCP tools (search_code, find_usages, pattern_search, find_files,
get_element, get_call_graph, get_repo_stats, etc.) instead of built-in Grep/Glob.

## Your Task

### Phase 1: Overview
1. Get repo stats to understand what was indexed
2. Note element counts per type (function, class, method, etc.)
3. Flag any element types with 0 count that should exist for this language

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

### Phase 4: Report
Produce a structured report in this exact format:

```markdown
## {repo_name} ({language}) - Parser Quality Report

### Stats
- Elements indexed: {count}
- Types: {breakdown}
- Missing types: {list of expected but 0-count types}

### Call Resolution Quality
| Element | Expected Calls | Indexed Calls | Missing | Phantom |
|---------|---------------|---------------|---------|---------|
| func_name | 5 | 3 | foo, bar | - |

### Summary Quality
| Element | Issue |
|---------|-------|
| func_name | Starts with "This function..." (anti-pattern) |
| func_name | Summary doesn't mention key behavior X |

### Missing Elements
| File | Element | Type | Why Missing |
|------|---------|------|-------------|
| path/to/file.py | some_func | function | Not indexed at all |

### Parameter/Type Gaps
| Element | Issue |
|---------|-------|
| func_name | Missing param: `timeout` |
| func_name | Return type not captured |

### Recommendations
1. [Actionable improvement for the parser]
2. [Actionable improvement for call resolution]
```

Write the report to: /Users/dinnyosz/code/magaldi/test_repos/{repo_name}/_quality_report.md
```

### Step 3: Aggregate Results

After all subagents complete, produce a summary:

```markdown
## Test Repo Quality Summary

| Repo | Lang | Elements | Call Accuracy | Summary Quality | Missing Elements |
|------|------|----------|--------------|-----------------|------------------|
| click | python | 150 | 85% | 90% | 3 |
| express | javascript | 200 | 70% | 85% | 8 |

### Top Issues Across All Repos
1. [Most common gap]
2. [Second most common]
3. [Third]

### Parser Improvement Priorities
1. [Highest impact fix]
2. [Second]
```

Write aggregate report to: `/Users/dinnyosz/code/magaldi/test_repos/_aggregate_report.md`

## Targeting Specific Repos

The user can specify which repos to test:
- "test click and express" → only those two
- "test all rust repos" → fd, ripgrep, bat
- "test tier 1" → smoke test repos only

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
