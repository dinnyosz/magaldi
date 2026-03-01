# Call Resolution Improvements

**Date**: 2026-03-01
**Phase**: 6 (Call Resolution)
**Status**: Implementation

## Context

Phase 6 static call resolution resolved only 4,577/23,040 calls (19.9%):
- via imports: 3,051
- via types: 400
- via constructors: 1,125
- via scope: 1

Investigation revealed 50-70% of "unresolved" calls are correctly unresolvable (builtins, stdlib, external libraries). The remaining 30-50% are internal calls that should be resolvable but fail due to specific bugs and limitations.

## Root Causes Identified

### 1. JavaScript `or True` Bug (Critical)
**File**: `src/magaldi_core/module_resolver.py` line 240
**Impact**: All non-relative JS/TS imports marked as external

The `is_external_module()` method for JavaScript has:
```python
return (
    base in self._NODE_BUILTINS
    or base in self._NPM_PACKAGES
    or True  # <-- BUG: marks ALL bare specifiers as external
)
```

This was intentional for the "bare specifiers are npm packages" convention, but it prevents resolution of project-internal path aliases and workspace packages.

**Fix**: Remove `or True`, add configurable internal module prefixes.

### 2. Wildcard Imports Not Handled
**File**: `src/magaldi_core/call_resolution.py` `_build_import_map()`
**Impact**: 600-2,000 unresolved calls estimated

`from utils import *` creates import entry `name="*"`, but bare calls like `process()` don't match `"*"` in the import map.

**Fix**: When building import map, if a wildcard import is found, query the index for all elements in the imported module's file and add them to the import map.

### 3. False-Positive External Module Detection
**File**: `src/magaldi_core/module_resolver.py` line 117-121
**Impact**: 500-2,000 unresolved calls

Python module prefix matching is too aggressive. Uses first component matching against stdlib/third-party sets. A custom module like `logging_handlers` gets classified as stdlib because `logging` is in the set.

However, looking more closely: this is actually correct — `first_component = module.split(".")[0]` extracts `logging_handlers` from `logging_handlers`, NOT `logging`. The prefix check is on the full first component, not a substring. So `logging_handlers` would NOT match `logging`. This is fine.

**Actual issue**: The categorizer in `call_categorizer.py` checks `receiver in STDLIB_MODULES` which IS an exact match. No false positives here either.

**Revised assessment**: False-positive external detection is NOT a real issue. The first-component check works correctly.

### 4. No Diagnostic Logging
**File**: `src/magaldi_core/call_resolution.py` `_lookup_element_by_import()`
**Impact**: Makes debugging impossible

Returns `None` at 4 different failure points with zero logging. Can't diagnose why specific calls fail.

**Fix**: Add debug-level logging at each failure point.

### 5. `__init__.py` Re-exports Not Tracked
**File**: `src/magaldi_core/call_resolution.py` `_find_element_in_file()`
**Impact**: 1,000-2,000 unresolved calls

`from mypackage import User` where User is re-exported via `mypackage/__init__.py` fails because `_find_element_in_file()` only searches for elements defined IN the file, not re-exported through it.

**Fix**: When `_find_element_in_file()` returns None for an `__init__.py` file, check the file's imports for re-exports and follow them.

### 6. Path Resolution Too Narrow
**File**: `src/magaldi_core/module_resolver.py` `module_to_file_paths()`
**Impact**: 500-1,000 unresolved calls

Only tries `src/` and root for path prefixes. Misses non-standard layouts.

**Fix**: Add fallback to search by element name when path-based lookup fails.

## Implementation Plan

### Fix 1: Diagnostic Logging
Add debug logging to `_lookup_element_by_import()` at each failure/success point.

### Fix 2: JavaScript Bare Specifier Bug
Replace `or True` with a smarter check that allows known internal patterns.

### Fix 3: Wildcard Import Resolution
Expand wildcard imports in `_build_import_map()` by querying index for all elements in the wildcard-imported file.

### Fix 4: `__init__.py` Re-export Following
When element not found in `__init__.py`, check its imports for the element name and follow the chain.

### Fix 5: Fallback Name-based Lookup
When path-based lookup fails, fall back to `get_document_by_name_only()` as a last resort.

### Fix 6: Tests
Cover all new behaviors with unit tests.

## Estimated Impact

| Fix | Est. Additional Resolutions |
|-----|---------------------------|
| JS bare specifier | 2,000-5,000 (for JS/TS codebases) |
| Wildcard imports | 600-2,000 |
| __init__.py re-exports | 1,000-2,000 |
| Fallback name lookup | 500-1,000 |
| Diagnostic logging | Enables debugging |
| **Total** | **4,100-10,000** |

## Files Modified

- `src/magaldi_core/call_resolution.py` — logging, wildcard imports, re-export following, fallback lookup
- `src/magaldi_core/module_resolver.py` — JS bare specifier fix
- `tests/test_call_resolution.py` — new tests
- `tests/test_module_resolver.py` — new tests
