# Test Element Marking Design

## Overview

Mark code elements as test code (`is_test: true`) during parsing so MCP search results can distinguish between production code and test code. This enables LLMs to see related tests naturally in search results and make informed decisions about what needs updating.

## Use Case

When searching for code, test elements that are semantically similar (and thus appear in vector search results) will be flagged as tests. The LLM can then:
- Know which results are tests vs production code
- Advise that related tests may need changes when modifying code
- Filter out tests when they're noise

## Data Model

### CodeElement Changes

```python
class CodeElement:
    # ... existing fields ...
    is_test: bool = False  # New field
```

### Elasticsearch Mapping

```json
{
  "is_test": { "type": "boolean" }
}
```

Element IDs remain unchanged - `is_test` is metadata, not identity.

## Detection Logic

Detection happens during Phase 3 (Parsing) with two checks.

### Check 1: Path Patterns (File Level)

If file path matches a test pattern, the file element and all children inherit `is_test: true`.

```python
TEST_PATH_PATTERNS = {
    "python": [r"test_.*\.py$", r".*_test\.py$", r"tests/", r"conftest\.py$"],
    "javascript": [r".*\.test\.[jt]sx?$", r".*\.spec\.[jt]sx?$", r"__tests__/", r"test/"],
    "typescript": [r".*\.test\.tsx?$", r".*\.spec\.tsx?$", r"__tests__/", r"test/"],
    "php": [r".*Test\.php$", r"tests/"],
    "rust": [r"tests/"],  # Integration tests only; unit tests need AST check
}
```

### Check 2: AST Markers (Element Level)

For elements not caught by path patterns, check AST markers:

| Language | AST Pattern |
|----------|-------------|
| Python | Function decorated with `@pytest.*`, `@unittest.*`, or name starts with `test_` |
| JavaScript/TypeScript | Calls to `describe()`, `it()`, `test()`, `beforeEach()`, etc. |
| PHP | Method with `@test` docblock, name starts with `test`, or class extends `*TestCase` |
| Rust | Function with `#[test]` attribute, or inside `#[cfg(test)]` module |

### Logic Flow

1. If file path matches test pattern -> mark file element and all children as `is_test: true`
2. Else, check each element's AST -> mark individually if it matches test markers
3. Children inherit parent's `is_test` unless they have their own marker

## MCP Response Changes

### New Parameter

Add `include_tests` parameter to search tools:

```python
include_tests: bool = True  # Default: include tests
```

### Grouped Response Structure

Results are grouped into code and test sections:

```python
{
    "code_results": [
        {"element_id": "...", "name": "UserService", "is_test": False, ...},
        {"element_id": "...", "name": "authenticate", "is_test": False, ...},
    ],
    "test_results": [
        {"element_id": "...", "name": "test_authenticate", "is_test": True, ...},
        {"element_id": "...", "name": "TestUserService", "is_test": True, ...},
    ],
    "total_code": 2,
    "total_tests": 2
}
```

### Behavior

- `include_tests=True` (default): Return both groups
- `include_tests=False`: Return only `code_results`, `test_results` is empty

### Affected MCP Tools

- `search_code`
- `search_features`
- `find_similar`
- `grep_code` (for consistency)

## Implementation Locations

### Parser (`src/parser/`)

| File | Change |
|------|--------|
| `code_parser.py` | Add `is_test` detection logic after parsing each element |
| `queries/` | May need new S-expression queries for detecting test decorators/attributes |

### Models (`src/shared/`)

| File | Change |
|------|--------|
| `models.py` | Add `is_test: bool = False` to `CodeElement` |

### Storage (`src/shared/db/`)

| File | Change |
|------|--------|
| `elasticsearch.py` | Update ES mapping to include `is_test` boolean field |

### MCP (`src/mcp/`)

| File | Change |
|------|--------|
| `tools.py` | Add `include_tests` parameter to relevant tools |
| `tools.py` | Modify response formatting to group code vs test results |

## Migration

Existing indexed elements won't have `is_test` field. Strategy:
- Treat missing `is_test` as `False` (graceful fallback)
- Re-indexing populates the field correctly over time
- No forced migration required

## Out of Scope

- Explicit test-to-code mapping (relying on vector similarity instead)
- Test coverage metrics
- Test execution integration
