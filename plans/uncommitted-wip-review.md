# Uncommitted WIP Review

Three separate pieces of uncommitted work need review, testing, and committing.

## 1. Compact Formatter Refactor

**Files:** `src/magaldi_mcp/formatters/analysis.py`, `elements.py`, `search.py`
**New file:** `src/magaldi_mcp/formatters/_compact.py`
**Test file:** `tests/mcp_tools/test_compact.py`

**What it does:** Introduces a shared `_compact` module with helper functions (`compact_element`, `compact_ref`, `file_group`, etc.) and migrates `CallGraphFormatter`, `DeadCodeFormatter`, `EntryPointsFormatter`, element formatters, and search formatters to use them.

**Review checklist:**
- [ ] Verify all formatter outputs are equivalent or intentionally improved
- [ ] Run `tests/mcp_tools/test_compact.py` — should all pass
- [ ] Fix 3 pre-existing formatter test failures in `test_mcp_server.py` (TestFormatResultImplementations, TestFormatResultElement, TestFormatResultCallGraph) — these broke because output format changed
- [ ] Run `make test-fast` and confirm no regressions
- [ ] Commit: `refactor: extract shared compact formatter helpers`

## 2. Variable Scoring Module Updates

**Files:** `src/magaldi_core/variable_scoring/__init__.py`, `src/magaldi_core/variable_scoring/models.py` (new)

**What it does:** Expands the LLM-based variable scoring module — likely adds Pydantic models and refines batch processing logic.

**Review checklist:**
- [ ] Read the diff and verify models.py aligns with the scoring logic in `__init__.py`
- [ ] Run `tests/variable_scoring/test_scoring.py`
- [ ] Commit: `refactor: extract variable scoring models and improve batch logic`

## 3. CLI Pipeline Updates

**Files:** `src/shared/cli/_runners.py`, `src/shared/cli/parse.py`

**What it does:** Updates phase references (Phase 5 → Phase 6 for call resolution) and expands `_runners.py` with ~140 new lines — likely new runner logic for the variable scoring phase.

**Review checklist:**
- [ ] Verify phase numbering is consistent across all references
- [ ] Run `tests/test_cli.py` and `tests/integration/test_cli_e2e.py`
- [ ] Commit: `feat: add variable scoring runner to CLI pipeline`

## Order of operations

1. Formatter refactor first (standalone, no deps on others)
2. Variable scoring models (standalone module change)
3. CLI pipeline last (depends on variable scoring being correct)
