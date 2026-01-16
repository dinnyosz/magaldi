# Test Element Marking Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Mark code elements as test code (`is_test: true`) during parsing so MCP can group and filter test results.

**Architecture:** Add `is_test` field to CodeElement, detect via path patterns + AST markers during parsing, store in ES, and update MCP tools to return grouped results with optional filtering.

**Tech Stack:** Python, Tree-sitter, Elasticsearch, pytest

---

## Task 1: Add is_test Field to CodeElement

**Files:**
- Modify: `src/magaldi_core/code_parser.py:42-95` (CodeElement dataclass)
- Test: `tests/test_code_parser.py`

**Step 1: Write the failing test**

Add to `tests/test_code_parser.py` after the existing `TestCodeElement` class tests (~line 540):

```python
def test_is_test_default_false(self):
    element = CodeElement()
    assert element.is_test is False

def test_is_test_can_be_set(self):
    element = CodeElement(is_test=True)
    assert element.is_test is True
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_code_parser.py::TestCodeElement::test_is_test_default_false -v`
Expected: FAIL with "unexpected keyword argument 'is_test'"

**Step 3: Write minimal implementation**

Add to `CodeElement` dataclass in `src/magaldi_core/code_parser.py` after line 73 (`is_async: bool = False`):

```python
    is_test: bool = False  # Whether this element is test code
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_code_parser.py::TestCodeElement -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/magaldi_core/code_parser.py tests/test_code_parser.py
git commit -m "feat: add is_test field to CodeElement"
```

---

## Task 2: Add Test Path Detection Utility

**Files:**
- Modify: `src/magaldi_core/code_parser.py` (add new function)
- Test: `tests/test_code_parser.py`

**Step 1: Write the failing test**

Add new test class to `tests/test_code_parser.py`:

```python
class TestIsTestPath:
    """Tests for is_test_path utility function."""

    @pytest.mark.parametrize("path,language,expected", [
        # Python test paths
        ("test_foo.py", "python", True),
        ("foo_test.py", "python", True),
        ("tests/test_module.py", "python", True),
        ("tests/unit/test_foo.py", "python", True),
        ("conftest.py", "python", True),
        ("src/conftest.py", "python", True),
        # Python non-test paths
        ("foo.py", "python", False),
        ("testing.py", "python", False),
        ("src/app.py", "python", False),
        # JavaScript/TypeScript test paths
        ("foo.test.js", "javascript", True),
        ("foo.spec.js", "javascript", True),
        ("foo.test.ts", "typescript", True),
        ("foo.spec.tsx", "typescript", True),
        ("__tests__/foo.js", "javascript", True),
        ("test/foo.js", "javascript", True),
        # JavaScript non-test paths
        ("foo.js", "javascript", False),
        ("testing.js", "javascript", False),
        # PHP test paths
        ("FooTest.php", "php", True),
        ("tests/FooTest.php", "php", True),
        # PHP non-test paths
        ("Foo.php", "php", False),
        # Rust test paths
        ("tests/integration.rs", "rust", True),
        # Rust non-test paths (unit tests are in-file)
        ("src/lib.rs", "rust", False),
    ])
    def test_is_test_path(self, path: str, language: str, expected: bool):
        from magaldi_core.code_parser import is_test_path
        assert is_test_path(path, language) == expected
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_code_parser.py::TestIsTestPath -v`
Expected: FAIL with "cannot import name 'is_test_path'"

**Step 3: Write minimal implementation**

Add after the imports section in `src/magaldi_core/code_parser.py` (~line 35):

```python
import re

# Test path patterns by language
TEST_PATH_PATTERNS: dict[str, list[str]] = {
    "python": [
        r"(^|/)test_[^/]+\.py$",      # test_*.py
        r"(^|/)[^/]+_test\.py$",       # *_test.py
        r"(^|/)tests/",                # tests/ directory
        r"(^|/)conftest\.py$",         # conftest.py
    ],
    "javascript": [
        r"\.(test|spec)\.[jt]sx?$",    # *.test.js, *.spec.ts, etc.
        r"(^|/)__tests__/",            # __tests__/ directory
        r"(^|/)test/",                 # test/ directory
    ],
    "typescript": [
        r"\.(test|spec)\.[jt]sx?$",    # *.test.ts, *.spec.tsx, etc.
        r"(^|/)__tests__/",            # __tests__/ directory
        r"(^|/)test/",                 # test/ directory
    ],
    "php": [
        r"Test\.php$",                 # *Test.php
        r"(^|/)tests/",                # tests/ directory
    ],
    "rust": [
        r"(^|/)tests/",                # tests/ directory (integration tests)
    ],
}


def is_test_path(relative_path: str, language: str) -> bool:
    """Check if a file path indicates test code.

    Args:
        relative_path: File path relative to repository root.
        language: Programming language of the file.

    Returns:
        True if the path matches test file patterns.
    """
    patterns = TEST_PATH_PATTERNS.get(language, [])
    for pattern in patterns:
        if re.search(pattern, relative_path):
            return True
    return False
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_code_parser.py::TestIsTestPath -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/magaldi_core/code_parser.py tests/test_code_parser.py
git commit -m "feat: add is_test_path utility for test file detection"
```

---

## Task 3: Add AST-Based Test Detection

**Files:**
- Modify: `src/magaldi_core/code_parser.py`
- Test: `tests/test_code_parser.py`

**Step 1: Write the failing test**

Add new test class to `tests/test_code_parser.py`:

```python
class TestIsTestElement:
    """Tests for is_test_element utility function."""

    @pytest.mark.parametrize("name,decorators,language,expected", [
        # Python test elements
        ("test_foo", [], "python", True),
        ("test_something_complex", [], "python", True),
        ("foo", ["pytest.mark.parametrize"], "python", True),
        ("foo", ["pytest.fixture"], "python", True),
        ("foo", ["unittest.skip"], "python", True),
        # Python non-test elements
        ("foo", [], "python", False),
        ("testing_helper", [], "python", False),
        ("my_test", [], "python", False),  # doesn't start with test_
        # Rust test elements
        ("test_foo", ["test"], "rust", True),
        ("foo", ["test"], "rust", True),
        ("foo", ["cfg(test)"], "rust", True),
        # Rust non-test elements
        ("foo", [], "rust", False),
    ])
    def test_is_test_element(self, name: str, decorators: list[str], language: str, expected: bool):
        from magaldi_core.code_parser import is_test_element
        assert is_test_element(name, decorators, language) == expected
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_code_parser.py::TestIsTestElement -v`
Expected: FAIL with "cannot import name 'is_test_element'"

**Step 3: Write minimal implementation**

Add after `is_test_path` function in `src/magaldi_core/code_parser.py`:

```python
def is_test_element(name: str, decorators: list[str], language: str) -> bool:
    """Check if an element is test code based on name/decorators.

    Args:
        name: Element name (function/method/class name).
        decorators: List of decorator/attribute names.
        language: Programming language.

    Returns:
        True if the element appears to be test code.
    """
    # Python: test_ prefix or pytest/unittest decorators
    if language == "python":
        if name.startswith("test_"):
            return True
        test_decorators = {"pytest", "unittest", "pytest.mark", "pytest.fixture"}
        for dec in decorators:
            for test_dec in test_decorators:
                if dec.startswith(test_dec):
                    return True
        return False

    # Rust: #[test] or #[cfg(test)] attributes
    if language == "rust":
        test_attrs = {"test", "cfg(test)"}
        return any(dec in test_attrs for dec in decorators)

    # JavaScript/TypeScript: detected via call patterns (describe/it/test)
    # These are handled separately during parsing
    # PHP: @test annotation or Test suffix handled via path/class name
    return False
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_code_parser.py::TestIsTestElement -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/magaldi_core/code_parser.py tests/test_code_parser.py
git commit -m "feat: add is_test_element for AST-based test detection"
```

---

## Task 4: Apply is_test During Parsing

**Files:**
- Modify: `src/magaldi_core/code_parser.py` (parse_file and _convert_* methods)
- Test: `tests/test_code_parser.py`

**Step 1: Write the failing test**

Add to `tests/test_code_parser.py`:

```python
class TestParseFileTestDetection:
    """Tests for is_test detection during parsing."""

    def test_marks_test_file_elements(self, tmp_path: Path):
        """Test that elements in test files are marked is_test=True."""
        test_file = tmp_path / "test_example.py"
        test_file.write_text('''
def test_something():
    pass

def helper():
    pass
''')
        file_info = FileInfo(
            relative_path="test_example.py",
            absolute_path=test_file,
            language="python",
        )

        result = parse_file(file_info, "scope", "repo", "main")

        # All elements should be marked as test (file-level detection)
        for elem in result.elements:
            assert elem.is_test is True, f"{elem.name} should be is_test=True"

    def test_marks_test_functions_by_name(self, tmp_path: Path):
        """Test that test_ functions in non-test files are marked."""
        src_file = tmp_path / "example.py"
        src_file.write_text('''
def test_inline():
    """An inline test."""
    pass

def regular_function():
    pass
''')
        file_info = FileInfo(
            relative_path="src/example.py",
            absolute_path=src_file,
            language="python",
        )

        result = parse_file(file_info, "scope", "repo", "main")

        # Find elements by name
        elements = {e.name: e for e in result.elements}

        # File element should not be test
        assert elements["example.py"].is_test is False
        # test_ function should be test
        assert elements["test_inline"].is_test is True
        # regular function should not be test
        assert elements["regular_function"].is_test is False

    def test_non_test_file_not_marked(self, tmp_path: Path):
        """Test that regular files are not marked as test."""
        src_file = tmp_path / "app.py"
        src_file.write_text('''
def main():
    pass
''')
        file_info = FileInfo(
            relative_path="src/app.py",
            absolute_path=src_file,
            language="python",
        )

        result = parse_file(file_info, "scope", "repo", "main")

        for elem in result.elements:
            assert elem.is_test is False, f"{elem.name} should be is_test=False"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_code_parser.py::TestParseFileTestDetection -v`
Expected: FAIL (elements won't have is_test set correctly)

**Step 3: Write minimal implementation**

Modify the `parse_file` function in `src/magaldi_core/code_parser.py` (around line 700+). After creating the ParsedFile and before returning, add test detection:

Find the `parse_file` function and add this logic after elements are collected:

```python
    # Detect if this is a test file
    file_is_test = is_test_path(file_info.relative_path, file_info.language)

    # Apply is_test to all elements
    for elem in parsed.elements:
        if file_is_test:
            # All elements in test files are test code
            elem.is_test = True
        else:
            # Check individual elements for test markers
            elem.is_test = is_test_element(elem.name, elem.decorators, file_info.language)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_code_parser.py::TestParseFileTestDetection -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/magaldi_core/code_parser.py tests/test_code_parser.py
git commit -m "feat: apply is_test detection during file parsing"
```

---

## Task 5: Add is_test to Elasticsearch Mapping

**Files:**
- Modify: `src/shared/db/elasticsearch.py:37-82` (INDEX_MAPPING)
- Modify: `src/shared/db/elasticsearch.py:140-175` (index_element doc creation)
- Test: `tests/test_db_elasticsearch.py`

**Step 1: Write the failing test**

Add to `tests/test_db_elasticsearch.py`:

```python
class TestIsTestIndexing:
    """Tests for is_test field indexing."""

    def test_indexes_is_test_field(self, es_repo, sample_element):
        """Test that is_test field is indexed."""
        sample_element.is_test = True
        es_repo.index_element(sample_element)

        doc = es_repo.get_document(sample_element.element_id)
        assert doc is not None
        assert doc.get("is_test") is True

    def test_is_test_defaults_to_false(self, es_repo, sample_element):
        """Test that is_test defaults to False."""
        sample_element.is_test = False
        es_repo.index_element(sample_element)

        doc = es_repo.get_document(sample_element.element_id)
        assert doc is not None
        assert doc.get("is_test") is False
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_db_elasticsearch.py::TestIsTestIndexing -v`
Expected: FAIL (is_test field not in document)

**Step 3: Write minimal implementation**

1. Add to INDEX_MAPPING properties in `src/shared/db/elasticsearch.py` (after line 59, `is_async`):

```python
            "is_test": {"type": "boolean"},  # Whether element is test code
```

2. Add to the `doc` dict in `index_element` method (after `is_async` around line 165):

```python
            "is_test": element.is_test,
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_db_elasticsearch.py::TestIsTestIndexing -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/shared/db/elasticsearch.py tests/test_db_elasticsearch.py
git commit -m "feat: add is_test field to Elasticsearch mapping"
```

---

## Task 6: Update search_code to Group Results

**Files:**
- Modify: `src/magaldi_mcp/tools.py:20-120` (search_code function)
- Test: `tests/test_mcp_tools.py`

**Step 1: Write the failing test**

Add to `tests/test_mcp_tools.py`:

```python
class TestSearchCodeTestGrouping:
    """Tests for search_code test result grouping."""

    def test_groups_test_and_code_results(self, mock_es_repo, mock_embed_client):
        """Test that results are grouped by is_test."""
        mock_es_repo.search_by_vector.return_value = [
            {"element_id": "id1", "name": "UserService", "element_type": "class", "is_test": False},
            {"element_id": "id2", "name": "test_user_service", "element_type": "function", "is_test": True},
        ]

        result = search_code(
            es=mock_es_repo,
            embed_client=mock_embed_client,
            query="user service",
        )

        assert "code_results" in result
        assert "test_results" in result
        assert len(result["code_results"]) == 1
        assert len(result["test_results"]) == 1
        assert result["code_results"][0]["name"] == "UserService"
        assert result["test_results"][0]["name"] == "test_user_service"

    def test_include_tests_false_excludes_tests(self, mock_es_repo, mock_embed_client):
        """Test that include_tests=False excludes test results."""
        mock_es_repo.search_by_vector.return_value = [
            {"element_id": "id1", "name": "UserService", "element_type": "class", "is_test": False},
            {"element_id": "id2", "name": "test_user_service", "element_type": "function", "is_test": True},
        ]

        result = search_code(
            es=mock_es_repo,
            embed_client=mock_embed_client,
            query="user service",
            include_tests=False,
        )

        assert len(result["code_results"]) == 1
        assert len(result["test_results"]) == 0

    def test_results_include_is_test_field(self, mock_es_repo, mock_embed_client):
        """Test that individual results include is_test field."""
        mock_es_repo.search_by_vector.return_value = [
            {"element_id": "id1", "name": "foo", "element_type": "function", "is_test": True},
        ]

        result = search_code(
            es=mock_es_repo,
            embed_client=mock_embed_client,
            query="foo",
        )

        assert result["test_results"][0]["is_test"] is True
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_mcp_tools.py::TestSearchCodeTestGrouping -v`
Expected: FAIL (no code_results/test_results keys)

**Step 3: Write minimal implementation**

Modify `search_code` in `src/magaldi_mcp/tools.py`:

1. Add `include_tests` parameter:
```python
def search_code(
    es: ElasticsearchRepository,
    embed_client: CodeEmbeddingClient | None,
    query: str,
    scope: str | None = None,
    repository: str | None = None,
    username: str = "main",
    element_types: list[str] | None = None,
    language: str | None = None,
    limit: int = 20,
    include_code: bool = False,
    brief: bool = False,
    include_tests: bool = True,  # New parameter
) -> dict[str, Any]:  # Changed return type
```

2. Change return type and restructure the return logic:
```python
    # Group results by is_test
    code_results = []
    test_results = []

    for result in results:
        # ... existing filtering by language ...

        is_test = result.get("is_test", False)

        # Skip tests if not included
        if is_test and not include_tests:
            continue

        entry: dict[str, Any] = {
            "name": name,
            "type": result.get("element_type"),
            "file": result.get("relative_path"),
            "line": result.get("line_start"),
            "element_id": result.get("element_id"),
            "is_test": is_test,
        }

        if not brief:
            entry["summary"] = result.get("summary", "")
            sig = result.get("signature")
            if sig:
                entry["signature"] = sig
            if include_code and result.get("raw_code"):
                entry["code"] = result["raw_code"]

        if is_test:
            test_results.append(entry)
        else:
            code_results.append(entry)

    return {
        "code_results": code_results[:limit],
        "test_results": test_results[:limit] if include_tests else [],
        "total_code": len(code_results),
        "total_tests": len(test_results) if include_tests else 0,
    }
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_mcp_tools.py::TestSearchCodeTestGrouping -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/magaldi_mcp/tools.py tests/test_mcp_tools.py
git commit -m "feat: group search_code results by is_test with filtering"
```

---

## Task 7: Update find_similar to Group Results

**Files:**
- Modify: `src/magaldi_mcp/tools.py:194-260` (find_similar function)
- Test: `tests/test_mcp_tools.py`

**Step 1: Write the failing test**

Add to `tests/test_mcp_tools.py`:

```python
class TestFindSimilarTestGrouping:
    """Tests for find_similar test result grouping."""

    def test_groups_similar_by_is_test(self, mock_es_repo):
        """Test that similar results are grouped by is_test."""
        mock_es_repo.get_document.return_value = {
            "element_id": "id1",
            "embedding": [0.1] * 1024,
        }
        mock_es_repo.search_by_vector.return_value = [
            {"element_id": "id2", "name": "similar_func", "is_test": False},
            {"element_id": "id3", "name": "test_similar", "is_test": True},
        ]

        result = find_similar(es=mock_es_repo, element_id="id1")

        assert "code_results" in result
        assert "test_results" in result

    def test_include_tests_false(self, mock_es_repo):
        """Test include_tests parameter."""
        mock_es_repo.get_document.return_value = {
            "element_id": "id1",
            "embedding": [0.1] * 1024,
        }
        mock_es_repo.search_by_vector.return_value = [
            {"element_id": "id2", "name": "test_func", "is_test": True},
        ]

        result = find_similar(es=mock_es_repo, element_id="id1", include_tests=False)

        assert len(result["test_results"]) == 0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_mcp_tools.py::TestFindSimilarTestGrouping -v`
Expected: FAIL

**Step 3: Write minimal implementation**

Update `find_similar` similarly to `search_code` - add `include_tests` parameter and group results.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_mcp_tools.py::TestFindSimilarTestGrouping -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/magaldi_mcp/tools.py tests/test_mcp_tools.py
git commit -m "feat: group find_similar results by is_test"
```

---

## Task 8: Update grep_code to Group Results

**Files:**
- Modify: `src/magaldi_mcp/tools.py:820-916` (grep_code function)
- Test: `tests/test_mcp_tools.py`

**Step 1: Write the failing test**

Add to `tests/test_mcp_tools.py`:

```python
class TestGrepCodeTestGrouping:
    """Tests for grep_code test result grouping."""

    def test_groups_grep_results_by_is_test(self, mock_es_repo):
        """Test that grep results are grouped by is_test."""
        # Mock ES search to return elements with raw_code
        mock_es_repo._get_client().search.return_value = {
            "hits": {
                "hits": [
                    {
                        "_source": {
                            "element_id": "id1",
                            "name": "func",
                            "element_type": "function",
                            "relative_path": "src/app.py",
                            "line_start": 1,
                            "raw_code": "def func():\n    pattern_match\n",
                            "is_test": False,
                        }
                    },
                    {
                        "_source": {
                            "element_id": "id2",
                            "name": "test_func",
                            "element_type": "function",
                            "relative_path": "tests/test_app.py",
                            "line_start": 1,
                            "raw_code": "def test_func():\n    pattern_match\n",
                            "is_test": True,
                        }
                    },
                ]
            }
        }

        result = grep_code(es=mock_es_repo, pattern="pattern_match")

        assert "code_results" in result
        assert "test_results" in result
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_mcp_tools.py::TestGrepCodeTestGrouping -v`
Expected: FAIL

**Step 3: Write minimal implementation**

Update `grep_code` to include `include_tests` parameter and return grouped results structure.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_mcp_tools.py::TestGrepCodeTestGrouping -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/magaldi_mcp/tools.py tests/test_mcp_tools.py
git commit -m "feat: group grep_code results by is_test"
```

---

## Task 9: Update MCP Server Tool Definitions

**Files:**
- Modify: `src/magaldi_mcp/server.py` (if tool schemas are defined there)
- Or check where MCP tool schemas are registered

**Step 1: Find and update tool schemas**

Search for where the MCP tool parameters are defined (likely in server.py or a schema file). Add `include_tests` parameter with description:

```python
{
    "name": "include_tests",
    "type": "boolean",
    "default": True,
    "description": "Include test elements in results. Default: true."
}
```

**Step 2: Test the MCP server responds correctly**

Manual test: Start the MCP server and verify the new parameter is exposed.

**Step 3: Commit**

```bash
git add src/magaldi_mcp/server.py
git commit -m "feat: add include_tests parameter to MCP tool schemas"
```

---

## Task 10: Run Full Test Suite and Fix Any Issues

**Step 1: Run all tests**

```bash
pytest tests/ -v --tb=short
```

**Step 2: Fix any failing tests**

Update test fixtures and mocks as needed to account for the new `is_test` field.

**Step 3: Commit fixes**

```bash
git add -A
git commit -m "fix: update tests for is_test field changes"
```

---

## Task 11: Final Integration Test

**Step 1: Manual integration test**

1. Parse a repository with test files
2. Verify `is_test` is set correctly in ES
3. Test MCP `search_code` returns grouped results
4. Test `include_tests=false` filters correctly

**Step 2: Commit any final fixes**

```bash
git add -A
git commit -m "fix: integration test fixes for test element marking"
```
