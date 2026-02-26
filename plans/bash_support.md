# Bash Script Support for Magaldi

## Research Summary

### Tree-sitter-bash AST Analysis

Installed `tree-sitter-bash` (v0.25.1) and analyzed representative bash scripts. The grammar parses cleanly with well-structured nodes.

### Key AST Node Types

| Node Type | Maps To | Notes |
|-----------|---------|-------|
| `function_definition` | `function` element | First `word` child = name, `compound_statement` = body |
| `variable_assignment` (top-level) | `constant` / `variable` | `variable_name` child = name |
| `declaration_command` with `readonly` | `constant` | Contains `variable_assignment` children |
| `declaration_command` with `export` | `constant` | Contains `variable_assignment` children |
| `declaration_command` with `declare` | `variable` | `-r` flag → constant, `-A`/-a → array |
| `declaration_command` with `local` | (skip) | Local variables inside functions, not useful |
| `command` | calls | `command_name` → `word` child = command/function name |
| `command` with `source`/`.` | `import` | Source file imports |
| `comment` | comments | Including `#!` shebang line |
| `pipeline` | (within calls) | Piped commands |
| `if_statement`, `while_statement`, `for_statement`, `case_statement` | (control flow) | Inside function bodies |
| `heredoc_redirect` / `heredoc_body` | (within function code) | Heredoc support |
| `process_substitution` | (within calls) | `<()` and `>()` |

### Function Definition Structure
```
function_definition
  word                    → function name (e.g., "deploy_to_k8s")
  (                       → literal
  )                       → literal
  compound_statement      → function body { ... }
    {
    declaration_command   → local variables
    command               → function calls / external commands
      command_name
        word              → command name
      string / word       → arguments
    pipeline              → piped commands
    if_statement          → conditionals
    while_statement       → loops
    }
```

### Variable Assignment Structure
```
variable_assignment
  variable_name           → "DEPLOY_ENV"
  =
  string / number / array → value

declaration_command
  readonly / export / declare / local
  variable_assignment     → same structure as above
```

### Import (source) Structure
```
command
  command_name
    word                  → "source" or "."
  word / string           → "./lib/common.sh"
```

## Implementation Plan

### Step 1: Add dependency
**File:** `pyproject.toml`
- Add `"tree-sitter-bash>=0.23.0"` to dependencies

### Step 2: Register file extensions
**File:** `src/magaldi_core/discovery.py`
- Add to `SUPPORTED_EXTENSIONS`:
  - `".sh": "bash"`
  - `".bash": "bash"`
- Add to `SUPPORTED_FILENAMES`:
  - Files with no extension but `#!/bin/bash` or `#!/usr/bin/env bash` shebang → handled by existing shebang detection or new entry

### Step 3: Register tree-sitter grammar
**File:** `src/magaldi_core/tree_sitter_manager.py`
- Add `import tree_sitter_bash as ts_bash`
- Add to `LANGUAGE_CONFIG`: `"bash": (ts_bash, "language")`
- Add to `get_extractor()`: bash → BashExtractor

### Step 4: Create BashExtractor (simple extractor, like Dockerfile)
**File:** `src/magaldi_core/extractors/bash.py`

Extract:
- **Functions**: `function_definition` nodes → element_type `"function"`
  - Name: first `word` child
  - Line range from node start/end
  - Raw code from lines
- **Top-level variables/constants**:
  - `variable_assignment` at root level → `"constant"` (UPPER_CASE convention in bash)
  - `declaration_command` with `readonly`/`export` → `"constant"`
  - `declaration_command` with `declare` → `"variable"` or `"constant"` (if `-r` flag)
- **Calls within functions**: `command` nodes inside `compound_statement`
  - `command_name` → first `word` child = call name
  - Exclude builtins like `echo`, `cd`, `local`, `return`, `exit`, `shift`, `set`
- **Imports**: `command` nodes where command_name is `source` or `.`
  - Second child (word/string) = source file path

### Step 5: Create BashParser
**File:** `src/magaldi_core/parsers/bash.py`
- Follow `DockerfileParser` pattern exactly
- Create file element + function elements + constant elements
- Wire up BashExtractor

### Step 6: Register in all registries
**Files:**
- `src/magaldi_core/extractors/__init__.py` - Add BashExtractor to `_EXTRACTORS`
- `src/magaldi_core/parsers/__init__.py` - Add BashParser
- `src/magaldi_core/code_parser.py` - Add to `PARSERS` dict

### Step 7: Update Parser Lab support
**Files:**
- `src/magaldi_mcp/tools/parser_lab.py` - Add `.sh`/`.bash` to `EXTENSION_TO_LANGUAGE`
- `src/magaldi_mcp/tools/schemas/parser_lab.py` - Add `"bash"` to language enums (3 places)

### Step 8: Tests
**File:** `tests/test_bash_extractor.py`
Test cases:
1. Extract functions from bash script
2. Extract top-level constants (readonly, export, plain assignment)
3. Extract calls within functions
4. Extract source imports
5. Handle shebang line
6. Handle complex scripts (case statements, heredocs, pipelines)
7. Handle edge cases (no functions, only variables, empty script)

## What NOT to Extract (Scope Limitation)
- Local variables inside functions (noise)
- Inline comments (handled by semantic analysis layer)
- Subshell contents
- Alias definitions (too simple, like variable assignments to commands)
- Trap definitions (could add later)

## Files Changed (Summary)
| File | Change |
|------|--------|
| `pyproject.toml` | Add `tree-sitter-bash` dependency |
| `src/magaldi_core/discovery.py` | Add `.sh`, `.bash` extensions |
| `src/magaldi_core/tree_sitter_manager.py` | Register bash grammar + extractor |
| `src/magaldi_core/extractors/bash.py` | **NEW** - BashExtractor |
| `src/magaldi_core/extractors/__init__.py` | Register BashExtractor |
| `src/magaldi_core/parsers/bash.py` | **NEW** - BashParser |
| `src/magaldi_core/parsers/__init__.py` | Register BashParser |
| `src/magaldi_core/code_parser.py` | Add to PARSERS registry |
| `src/magaldi_mcp/tools/parser_lab.py` | Add bash to extension map |
| `src/magaldi_mcp/tools/schemas/parser_lab.py` | Add bash to language enums |
| `tests/test_bash_extractor.py` | **NEW** - Test suite |
