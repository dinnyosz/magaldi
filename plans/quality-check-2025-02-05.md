# Quality Check Plan - 2025-02-05

## Overview
Comprehensive quality audit of the Magaldi codebase focusing on large files, complexity, security, and maintainability.

## Issues Found

### CRITICAL: High Complexity Functions

| Function | File | Cyclomatic | Lines | Action |
|----------|------|------------|-------|--------|
| `run_feature_extraction` | `src/shared/cli/extract.py:76` | **154** | 822 | Split into 5+ smaller functions |
| `process_elements` | `src/magaldi_core/processor.py:1730` | **122** | 532 | Extract setup, processing, cleanup phases |
| `get_element_detail` | `src/magaldi_web/routes/elements.py:135` | **92** | 646 | Split by section (callers, children, siblings) |
| `benchmark_models` | `src/shared/cli/benchmark.py:1001` | **91** | 303 | Extract model comparison logic |
| `mcp_self_review` | `src/magaldi_mcp/tools_impl.py:5145` | **87** | 359 | Split analysis phases |

### HIGH: Silent Exception Swallowing (38 instances)

Files with most issues:
- `src/magaldi_web/routes/admin.py` - 7 instances
- `src/magaldi_core/processor.py` - 7 instances
- `src/magaldi_mcp/tools_impl.py` - 3 instances
- `src/shared/ai/llm_client.py` - 4 instances

### MEDIUM: Security Issues

| Kind | Severity | File:Line | Issue |
|------|----------|-----------|-------|
| SSRF | medium | `src/shared/cli/_shared.py:129` | HTTP request with f-string URL |
| SSRF | medium | `src/magaldi_web/dependencies.py:48` | HTTP request with f-string URL |
| Path Traversal | medium | `src/magaldi_mcp/tools/parser_lab.py:600` | Path with f-string |

### MEDIUM: Undocumented Public Functions (30+)

- `src/magaldi_core/change_detection.py` - 8 public functions
- `src/shared/ai/clustering/feature_processor.py` - 12+ helpers
- `src/magaldi_web/routes/analysis.py` - API endpoints

### LOW: Active TODOs (6)

- `src/magaldi_mcp/tools_impl.py:1230` - detect language from element
- `src/magaldi_mcp/tools_impl.py:1346` - support other languages
- `src/magaldi_web/routes/admin.py:105,176,194` - job stats, activity logging, retry logic
- `src/magaldi_mcp/tools/parser_lab.py:30` - check scope/repository

## Actionable Items

### Immediate (This Sprint)
- [ ] 1. Refactor `run_feature_extraction` - Split into smaller functions
- [ ] 2. Add logging to bare exceptions (38 instances)
- [ ] 3. Fix SSRF patterns - Validate URLs

### Short-term (Next 2 Sprints)
- [ ] 4. Split `tools_impl.py` into category-based modules
- [ ] 5. Add docstrings to change_detection.py
- [ ] 6. Complete TODOs in admin.py

### Medium-term (Next Quarter)
- [ ] 7. Reduce `process_elements` complexity
- [ ] 8. Add enums for job status
- [ ] 9. Split large parser functions

### Documentation
- [ ] 10. Add docstrings to undocumented public functions
