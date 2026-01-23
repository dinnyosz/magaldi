# Phase A: Pattern Search Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace inefficient client-side `grep_code` with ES-native `pattern_search` tool

**Architecture:** Add three Elasticsearch query methods (regexp, wildcard, match_phrase with slop) and expose them through a single `pattern_search` MCP tool with explicit mode selection.

**Tech Stack:** Elasticsearch 8.11.0, Python MCP SDK, pytest

---

## Task 1: Add Elasticsearch Pattern Search Methods

**Files:**
- Modify: `src/shared/db/elasticsearch.py`
- Test: `tests/test_db_elasticsearch.py`

### Step 1: Write failing tests for regexp search

Add to `tests/test_db_elasticsearch.py`:

```python
class TestPatternSearch:
    """Tests for pattern search methods."""

    @pytest.fixture
    def sample_elements(self, es_repo):
        """Create sample elements with raw_code for pattern testing."""
        elements = [
            {
                "element_id": "test:repo:main:file.py:function:add_column:10",
                "scope": "test",
                "repository": "repo",
                "username": "main",
                "relative_path": "file.py",
                "element_type": "function",
                "name": "add_column",
                "line_start": 10,
                "raw_code": "def add_column(table, Model):\n    table.add_column('name', String)",
                "is_test": False,
            },
            {
                "element_id": "test:repo:main:utils.py:function:process:20",
                "scope": "test",
                "repository": "repo",
                "username": "main",
                "relative_path": "utils.py",
                "element_type": "function",
                "name": "process",
                "line_start": 20,
                "raw_code": "def process(data):\n    return data.strip()",
                "is_test": False,
            },
            {
                "element_id": "test:repo:main:test_file.py:function:test_add:30",
                "scope": "test",
                "repository": "repo",
                "username": "main",
                "relative_path": "test_file.py",
                "element_type": "function",
                "name": "test_add",
                "line_start": 30,
                "raw_code": "def test_add():\n    add_column(t, Model)",
                "is_test": True,
            },
        ]
        for elem in elements:
            es_repo.index_element(elem)
        es_repo._get_client().indices.refresh(index="magaldi-code-elements")
        return elements

    def test_search_by_regexp(self, es_repo, sample_elements):
        """Test regexp pattern search."""
        results = es_repo.search_by_regexp(
            pattern="add_column.*Model",
            scope="test",
            repository="repo",
        )
        assert len(results) >= 1
        assert any("add_column" in r.get("raw_code", "") for r in results)

    def test_search_by_regexp_no_match(self, es_repo, sample_elements):
        """Test regexp with no matches."""
        results = es_repo.search_by_regexp(
            pattern="nonexistent_function",
            scope="test",
            repository="repo",
        )
        assert len(results) == 0
```

### Step 2: Run test to verify it fails

```bash
pytest tests/test_db_elasticsearch.py::TestPatternSearch::test_search_by_regexp -v
```

Expected: FAIL with "AttributeError: 'ElasticsearchRepository' object has no attribute 'search_by_regexp'"

### Step 3: Implement search_by_regexp method

Add to `src/shared/db/elasticsearch.py` after `search_by_keyword` method (~line 730):

```python
def search_by_regexp(
    self,
    pattern: str,
    scope: str | None = None,
    repository: str | None = None,
    username: str = "main",
    glob: str | None = None,
    size: int = 50,
    include_tests: bool = True,
) -> list[dict[str, Any]]:
    """Search raw_code field using Elasticsearch regexp query.

    Uses Lucene regexp syntax (not Python re). Key differences:
    - . matches any character (no need to escape)
    - * is zero or more of previous
    - .* matches any string
    - No lookahead/lookbehind support

    Args:
        pattern: Lucene regexp pattern.
        scope: Filter by scope.
        repository: Filter by repository.
        username: Filter by username.
        glob: File path glob filter (e.g., '*.py').
        size: Maximum results.
        include_tests: Include test elements.

    Returns:
        List of matching documents.
    """
    must_clauses: list[dict[str, Any]] = [
        {"regexp": {"raw_code": {"value": pattern, "flags": "ALL"}}},
    ]

    filter_clauses: list[dict[str, Any]] = [
        {"term": {"username": username}},
    ]

    if scope:
        filter_clauses.append({"term": {"scope": scope}})
    if repository:
        filter_clauses.append({"term": {"repository": repository}})
    if glob:
        filter_clauses.append({"wildcard": {"relative_path": glob}})
    if not include_tests:
        filter_clauses.append({"term": {"is_test": False}})

    query = {
        "bool": {
            "must": must_clauses,
            "filter": filter_clauses,
        }
    }

    client = self._get_client()
    result = client.search(
        index=INDEX_NAME,
        body={"query": query, "size": size},
    )

    return [hit["_source"] for hit in result["hits"]["hits"]]
```

### Step 4: Run test to verify it passes

```bash
pytest tests/test_db_elasticsearch.py::TestPatternSearch::test_search_by_regexp -v
```

Expected: PASS

### Step 5: Commit

```bash
git add src/shared/db/elasticsearch.py tests/test_db_elasticsearch.py
git commit -m "feat(elasticsearch): add search_by_regexp method"
```

---

## Task 2: Add Wildcard Search Method

**Files:**
- Modify: `src/shared/db/elasticsearch.py`
- Test: `tests/test_db_elasticsearch.py`

### Step 1: Write failing test for wildcard search

Add to `tests/test_db_elasticsearch.py` in `TestPatternSearch`:

```python
def test_search_by_wildcard(self, es_repo, sample_elements):
    """Test wildcard pattern search."""
    results = es_repo.search_by_wildcard(
        pattern="*column*Model*",
        scope="test",
        repository="repo",
    )
    assert len(results) >= 1
    assert any("add_column" in r.get("name", "") for r in results)

def test_search_by_wildcard_question_mark(self, es_repo, sample_elements):
    """Test wildcard with ? for single character."""
    results = es_repo.search_by_wildcard(
        pattern="*proce??*",
        scope="test",
        repository="repo",
    )
    assert len(results) >= 1
```

### Step 2: Run test to verify it fails

```bash
pytest tests/test_db_elasticsearch.py::TestPatternSearch::test_search_by_wildcard -v
```

Expected: FAIL with "AttributeError: 'ElasticsearchRepository' object has no attribute 'search_by_wildcard'"

### Step 3: Implement search_by_wildcard method

Add to `src/shared/db/elasticsearch.py` after `search_by_regexp`:

```python
def search_by_wildcard(
    self,
    pattern: str,
    scope: str | None = None,
    repository: str | None = None,
    username: str = "main",
    glob: str | None = None,
    size: int = 50,
    include_tests: bool = True,
) -> list[dict[str, Any]]:
    """Search raw_code field using Elasticsearch wildcard query.

    Wildcard syntax:
    - * matches zero or more characters
    - ? matches exactly one character

    Args:
        pattern: Wildcard pattern.
        scope: Filter by scope.
        repository: Filter by repository.
        username: Filter by username.
        glob: File path glob filter.
        size: Maximum results.
        include_tests: Include test elements.

    Returns:
        List of matching documents.
    """
    must_clauses: list[dict[str, Any]] = [
        {"wildcard": {"raw_code": {"value": pattern, "case_insensitive": True}}},
    ]

    filter_clauses: list[dict[str, Any]] = [
        {"term": {"username": username}},
    ]

    if scope:
        filter_clauses.append({"term": {"scope": scope}})
    if repository:
        filter_clauses.append({"term": {"repository": repository}})
    if glob:
        filter_clauses.append({"wildcard": {"relative_path": glob}})
    if not include_tests:
        filter_clauses.append({"term": {"is_test": False}})

    query = {
        "bool": {
            "must": must_clauses,
            "filter": filter_clauses,
        }
    }

    client = self._get_client()
    result = client.search(
        index=INDEX_NAME,
        body={"query": query, "size": size},
    )

    return [hit["_source"] for hit in result["hits"]["hits"]]
```

### Step 4: Run test to verify it passes

```bash
pytest tests/test_db_elasticsearch.py::TestPatternSearch::test_search_by_wildcard -v
```

Expected: PASS

### Step 5: Commit

```bash
git add src/shared/db/elasticsearch.py tests/test_db_elasticsearch.py
git commit -m "feat(elasticsearch): add search_by_wildcard method"
```

---

## Task 3: Add Proximity Search Method

**Files:**
- Modify: `src/shared/db/elasticsearch.py`
- Test: `tests/test_db_elasticsearch.py`

### Step 1: Write failing test for proximity search

Add to `tests/test_db_elasticsearch.py` in `TestPatternSearch`:

```python
def test_search_by_proximity(self, es_repo, sample_elements):
    """Test proximity search with slop."""
    results = es_repo.search_by_proximity(
        terms="add column Model",
        slop=5,
        scope="test",
        repository="repo",
    )
    assert len(results) >= 1

def test_search_by_proximity_exact_phrase(self, es_repo, sample_elements):
    """Test proximity search with slop=0 for exact phrase."""
    results = es_repo.search_by_proximity(
        terms="def process",
        slop=0,
        scope="test",
        repository="repo",
    )
    assert len(results) >= 1
```

### Step 2: Run test to verify it fails

```bash
pytest tests/test_db_elasticsearch.py::TestPatternSearch::test_search_by_proximity -v
```

Expected: FAIL with "AttributeError: 'ElasticsearchRepository' object has no attribute 'search_by_proximity'"

### Step 3: Implement search_by_proximity method

Add to `src/shared/db/elasticsearch.py` after `search_by_wildcard`:

```python
def search_by_proximity(
    self,
    terms: str,
    slop: int = 5,
    scope: str | None = None,
    repository: str | None = None,
    username: str = "main",
    glob: str | None = None,
    size: int = 50,
    include_tests: bool = True,
) -> list[dict[str, Any]]:
    """Search raw_code for terms within proximity of each other.

    Uses match_phrase with slop to find terms that appear near each other.

    Args:
        terms: Space-separated terms to find near each other.
        slop: Maximum number of positions between terms.
        scope: Filter by scope.
        repository: Filter by repository.
        username: Filter by username.
        glob: File path glob filter.
        size: Maximum results.
        include_tests: Include test elements.

    Returns:
        List of matching documents.
    """
    must_clauses: list[dict[str, Any]] = [
        {"match_phrase": {"raw_code": {"query": terms, "slop": slop}}},
    ]

    filter_clauses: list[dict[str, Any]] = [
        {"term": {"username": username}},
    ]

    if scope:
        filter_clauses.append({"term": {"scope": scope}})
    if repository:
        filter_clauses.append({"term": {"repository": repository}})
    if glob:
        filter_clauses.append({"wildcard": {"relative_path": glob}})
    if not include_tests:
        filter_clauses.append({"term": {"is_test": False}})

    query = {
        "bool": {
            "must": must_clauses,
            "filter": filter_clauses,
        }
    }

    client = self._get_client()
    result = client.search(
        index=INDEX_NAME,
        body={"query": query, "size": size},
    )

    return [hit["_source"] for hit in result["hits"]["hits"]]
```

### Step 4: Run test to verify it passes

```bash
pytest tests/test_db_elasticsearch.py::TestPatternSearch::test_search_by_proximity -v
```

Expected: PASS

### Step 5: Commit

```bash
git add src/shared/db/elasticsearch.py tests/test_db_elasticsearch.py
git commit -m "feat(elasticsearch): add search_by_proximity method"
```

---

## Task 4: Create pattern_search MCP Tool

**Files:**
- Modify: `src/magaldi_mcp/tools.py`
- Test: `tests/test_mcp_tools.py`

### Step 1: Write failing test for pattern_search tool

Add to `tests/test_mcp_tools.py`:

```python
# Add to imports at top of file
from magaldi_mcp.tools import pattern_search


class TestPatternSearch:
    """Tests for pattern_search function."""

    def test_pattern_search_regexp_mode(self, mock_es_repo):
        """Test pattern_search with regexp mode."""
        mock_es_repo.search_by_regexp.return_value = [
            {
                "element_id": "test:repo:main:file.py:function:add_column:10",
                "name": "add_column",
                "element_type": "function",
                "relative_path": "file.py",
                "line_start": 10,
                "raw_code": "def add_column(table, Model):\n    pass",
                "is_test": False,
            }
        ]

        result = pattern_search(
            es=mock_es_repo,
            pattern="add_column.*Model",
            mode="regexp",
            scope="test",
            repository="repo",
        )

        assert "code_results" in result
        assert len(result["code_results"]) == 1
        mock_es_repo.search_by_regexp.assert_called_once()

    def test_pattern_search_wildcard_mode(self, mock_es_repo):
        """Test pattern_search with wildcard mode."""
        mock_es_repo.search_by_wildcard.return_value = []

        result = pattern_search(
            es=mock_es_repo,
            pattern="*column*",
            mode="wildcard",
            scope="test",
            repository="repo",
        )

        assert "code_results" in result
        mock_es_repo.search_by_wildcard.assert_called_once()

    def test_pattern_search_proximity_mode(self, mock_es_repo):
        """Test pattern_search with proximity mode."""
        mock_es_repo.search_by_proximity.return_value = []

        result = pattern_search(
            es=mock_es_repo,
            pattern="add column Model",
            mode="proximity",
            slop=5,
            scope="test",
            repository="repo",
        )

        assert "code_results" in result
        mock_es_repo.search_by_proximity.assert_called_once_with(
            terms="add column Model",
            slop=5,
            scope="test",
            repository="repo",
            username="main",
            glob=None,
            size=50,
            include_tests=True,
        )

    def test_pattern_search_invalid_mode(self, mock_es_repo):
        """Test pattern_search with invalid mode raises error."""
        with pytest.raises(ValueError, match="Invalid mode"):
            pattern_search(
                es=mock_es_repo,
                pattern="test",
                mode="invalid",
                scope="test",
                repository="repo",
            )
```

### Step 2: Run test to verify it fails

```bash
pytest tests/test_mcp_tools.py::TestPatternSearch -v
```

Expected: FAIL with "cannot import name 'pattern_search'"

### Step 3: Implement pattern_search tool

Add to `src/magaldi_mcp/tools.py` after `grep_code` function:

```python
def pattern_search(
    es: ElasticsearchRepository,
    pattern: str,
    mode: str,
    scope: str,
    repository: str,
    username: str = "main",
    slop: int = 5,
    glob: str | None = None,
    limit: int = 50,
    include_tests: bool = True,
) -> dict[str, Any]:
    """Search code using ES-native pattern matching.

    Three modes available:
    - regexp: Lucene regexp syntax (e.g., "add_column.*Model")
    - wildcard: Simple wildcards (e.g., "*column*Model*")
    - proximity: Terms near each other (e.g., "add column Model")

    Args:
        es: Elasticsearch repository.
        pattern: Search pattern (syntax depends on mode).
        mode: One of "regexp", "wildcard", "proximity".
        scope: Filter by scope (required).
        repository: Filter by repository (required).
        username: User branch to search.
        slop: For proximity mode: max positions between terms.
        glob: File path glob filter (e.g., '*.py').
        limit: Maximum results to return.
        include_tests: Whether to include test results.

    Returns:
        Dict with code_results, test_results, and totals.

    Raises:
        ValueError: If mode is not one of the valid options.
    """
    valid_modes = ("regexp", "wildcard", "proximity")
    if mode not in valid_modes:
        raise ValueError(f"Invalid mode: {mode}. Must be one of {valid_modes}")

    # Call the appropriate ES method
    if mode == "regexp":
        results = es.search_by_regexp(
            pattern=pattern,
            scope=scope,
            repository=repository,
            username=username,
            glob=glob,
            size=limit,
            include_tests=include_tests,
        )
    elif mode == "wildcard":
        results = es.search_by_wildcard(
            pattern=pattern,
            scope=scope,
            repository=repository,
            username=username,
            glob=glob,
            size=limit,
            include_tests=include_tests,
        )
    else:  # proximity
        results = es.search_by_proximity(
            terms=pattern,
            slop=slop,
            scope=scope,
            repository=repository,
            username=username,
            glob=glob,
            size=limit,
            include_tests=include_tests,
        )

    # Format results (similar to grep_code output)
    code_results: list[dict[str, Any]] = []
    test_results: list[dict[str, Any]] = []

    for result in results:
        is_test = result.get("is_test", False)

        entry = {
            "element_id": result.get("element_id"),
            "file": result.get("relative_path"),
            "name": result.get("name"),
            "element_type": result.get("element_type"),
            "line_start": result.get("line_start"),
            "raw_code": result.get("raw_code"),
            "is_test": is_test,
        }

        if is_test:
            test_results.append(entry)
        else:
            code_results.append(entry)

    return {
        "code_results": code_results,
        "test_results": test_results,
        "totals": {
            "code": len(code_results),
            "tests": len(test_results),
        },
        "mode": mode,
        "pattern": pattern,
    }
```

### Step 4: Run test to verify it passes

```bash
pytest tests/test_mcp_tools.py::TestPatternSearch -v
```

Expected: PASS

### Step 5: Commit

```bash
git add src/magaldi_mcp/tools.py tests/test_mcp_tools.py
git commit -m "feat(mcp): add pattern_search tool"
```

---

## Task 5: Register pattern_search in MCP Server

**Files:**
- Modify: `src/magaldi_mcp/server.py`
- Test: `tests/test_mcp_server.py`

### Step 1: Write failing test for pattern_search registration

Add to `tests/test_mcp_server.py`:

```python
@pytest.mark.asyncio
async def test_pattern_search_tool_registered(mcp_server):
    """Test that pattern_search tool is registered."""
    tools = await mcp_server.server.list_tools()
    tool_names = [t.name for t in tools]
    assert "pattern_search" in tool_names


@pytest.mark.asyncio
async def test_pattern_search_tool_schema(mcp_server):
    """Test pattern_search tool has correct schema."""
    tools = await mcp_server.server.list_tools()
    pattern_tool = next(t for t in tools if t.name == "pattern_search")

    schema = pattern_tool.inputSchema
    assert "pattern" in schema["properties"]
    assert "mode" in schema["properties"]
    assert schema["properties"]["mode"]["enum"] == ["regexp", "wildcard", "proximity"]
    assert "scope" in schema["required"]
    assert "repository" in schema["required"]
```

### Step 2: Run test to verify it fails

```bash
pytest tests/test_mcp_server.py::test_pattern_search_tool_registered -v
```

Expected: FAIL (tool not found)

### Step 3: Register pattern_search tool

Add to `src/magaldi_mcp/server.py` in the `list_tools` function, after the `grep_code` Tool definition:

```python
Tool(
    name="pattern_search",
    description="PATTERN SEARCH: ES-native pattern matching on code. "
    "Faster than grep_code - query runs server-side. "
    "Three modes: regexp (Lucene syntax), wildcard (* and ?), proximity (terms near each other).",
    inputSchema={
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Search pattern. Syntax depends on mode.",
            },
            "mode": {
                "type": "string",
                "enum": ["regexp", "wildcard", "proximity"],
                "description": "regexp: Lucene regex (e.g., 'add_column.*Model'). "
                "wildcard: Simple wildcards (e.g., '*column*'). "
                "proximity: Terms near each other (e.g., 'add column Model').",
            },
            "scope": {"type": "string", "description": "Filter by scope (required)"},
            "repository": {"type": "string", "description": "Filter by repo (required)"},
            "username": {"type": "string", "default": "main"},
            "slop": {
                "type": "integer",
                "default": 5,
                "description": "For proximity mode: max word distance",
            },
            "glob": {"type": "string", "description": "File filter (e.g., '*.py')"},
            "limit": {"type": "integer", "default": 50},
            "include_tests": {"type": "boolean", "default": True},
        },
        "required": ["pattern", "mode", "scope", "repository"],
    },
),
```

Add handler in `_handle_tool` method:

```python
elif name == "pattern_search":
    from magaldi_mcp.tools import pattern_search
    return await asyncio.to_thread(
        pattern_search,
        es,
        pattern=args["pattern"],
        mode=args["mode"],
        scope=args["scope"],
        repository=args["repository"],
        username=args.get("username", self.default_username),
        slop=args.get("slop", 5),
        glob=args.get("glob"),
        limit=args.get("limit", 50),
        include_tests=args.get("include_tests", True),
    )
```

### Step 4: Run test to verify it passes

```bash
pytest tests/test_mcp_server.py::test_pattern_search_tool_registered -v
```

Expected: PASS

### Step 5: Commit

```bash
git add src/magaldi_mcp/server.py tests/test_mcp_server.py
git commit -m "feat(mcp): register pattern_search tool in server"
```

---

## Task 6: Add Deprecation Warning to grep_code

**Files:**
- Modify: `src/magaldi_mcp/tools.py`
- Modify: `src/magaldi_mcp/server.py`
- Test: `tests/test_mcp_tools.py`

### Step 1: Write test for deprecation warning

Add to `tests/test_mcp_tools.py`:

```python
import warnings


class TestGrepCodeDeprecation:
    """Tests for grep_code deprecation."""

    def test_grep_code_emits_deprecation_warning(self, mock_es_repo):
        """Test that grep_code emits a deprecation warning."""
        mock_es_repo._get_client.return_value.search.return_value = {
            "hits": {"hits": []}
        }

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            grep_code(
                es=mock_es_repo,
                pattern="test",
                scope="test",
                repository="repo",
            )

            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "pattern_search" in str(w[0].message)
```

### Step 2: Run test to verify it fails

```bash
pytest tests/test_mcp_tools.py::TestGrepCodeDeprecation -v
```

Expected: FAIL (no warning emitted)

### Step 3: Add deprecation warning to grep_code

Modify `grep_code` function in `src/magaldi_mcp/tools.py` to add warning at the start:

```python
def grep_code(
    es: ElasticsearchRepository,
    pattern: str,
    scope: str | None = None,
    repository: str | None = None,
    username: str = "main",
    glob: str | None = None,
    context_lines: int = 0,
    limit: int = 50,
    include_tests: bool = True,
) -> dict[str, Any]:
    """Search indexed code with regex pattern.

    .. deprecated::
        Use `pattern_search` instead for better performance.
        grep_code will be removed in a future release.

    ...existing docstring...
    """
    import warnings

    warnings.warn(
        "grep_code is deprecated. Use pattern_search with mode='regexp' instead. "
        "pattern_search runs queries server-side for better performance.",
        DeprecationWarning,
        stacklevel=2,
    )

    # ... rest of function unchanged ...
```

Also update the tool description in `server.py`:

```python
Tool(
    name="grep_code",
    description="[DEPRECATED: Use pattern_search instead] "
    "GREP CODE: Search with regex pattern (like ripgrep). "
    "USE pattern_search for better performance - queries run server-side.",
    # ... rest unchanged ...
),
```

### Step 4: Run test to verify it passes

```bash
pytest tests/test_mcp_tools.py::TestGrepCodeDeprecation -v
```

Expected: PASS

### Step 5: Commit

```bash
git add src/magaldi_mcp/tools.py src/magaldi_mcp/server.py tests/test_mcp_tools.py
git commit -m "chore(mcp): deprecate grep_code in favor of pattern_search"
```

---

## Task 7: Update find_usages to Use pattern_search

**Files:**
- Modify: `src/magaldi_mcp/tools.py`
- Test: `tests/test_mcp_tools.py`

### Step 1: Write test verifying find_usages uses pattern_search

Add to `tests/test_mcp_tools.py`:

```python
class TestFindUsagesWithPatternSearch:
    """Tests for find_usages using pattern_search."""

    def test_find_usages_uses_regexp_search(self, mock_es_repo):
        """Test that find_usages uses search_by_regexp internally."""
        mock_es_repo.get_document.return_value = {
            "element_id": "test:repo:main:file.py:function:my_func:10",
            "name": "my_func",
            "element_type": "function",
            "relative_path": "file.py",
            "line_start": 10,
            "scope": "test",
            "repository": "repo",
            "username": "main",
        }
        mock_es_repo.search_by_regexp.return_value = [
            {
                "element_id": "test:repo:main:other.py:function:caller:20",
                "name": "caller",
                "element_type": "function",
                "relative_path": "other.py",
                "line_start": 20,
                "raw_code": "def caller():\n    my_func()",
                "is_test": False,
            }
        ]

        result = find_usages(
            es=mock_es_repo,
            element_id="test:repo:main:file.py:function:my_func:10",
        )

        # Verify search_by_regexp was called (not the old client.search)
        mock_es_repo.search_by_regexp.assert_called()
        assert len(result) >= 0  # May filter out definition
```

### Step 2: Run test to verify it fails

```bash
pytest tests/test_mcp_tools.py::TestFindUsagesWithPatternSearch -v
```

Expected: FAIL (search_by_regexp not called)

### Step 3: Update find_usages to use search_by_regexp

Replace the grep_code call in `find_usages` with direct use of `search_by_regexp`:

```python
def find_usages(
    es: ElasticsearchRepository,
    element_id: str,
    limit: int = 30,
) -> list[dict[str, Any]]:
    """Find usages of a function/class/method.

    Searches indexed code in Elasticsearch - no filesystem access needed.

    Args:
        es: Elasticsearch repository.
        element_id: Element to find usages of.
        limit: Maximum usages to return.

    Returns:
        List of usage locations with context.
    """
    import re

    # Get the element to find its name
    doc = es.get_document(element_id)
    if not doc:
        raise ValueError(f"Element not found: {element_id}")

    name = doc.get("name")
    element_type = doc.get("element_type")
    defining_file = doc.get("relative_path")
    defining_line = doc.get("line_start")
    scope = doc.get("scope")
    repository = doc.get("repository")
    username = doc.get("username", "main")

    # Build search pattern based on element type (Lucene regexp syntax)
    if element_type == "function":
        # Function calls: name(
        pattern = rf"{re.escape(name)}\s*\("
    elif element_type == "method":
        # Method calls: .name(
        pattern = rf"\.{re.escape(name)}\s*\("
    elif element_type == "class":
        # Class references: word boundary simulation
        pattern = rf"(^|[^a-zA-Z0-9_]){re.escape(name)}([^a-zA-Z0-9_]|$)"
    else:
        # Generic: name with boundaries
        pattern = rf"(^|[^a-zA-Z0-9_]){re.escape(name)}([^a-zA-Z0-9_]|$)"

    # Use ES-native regexp search
    results = es.search_by_regexp(
        pattern=pattern,
        scope=scope,
        repository=repository,
        username=username,
        size=limit + 10,  # Get extra to filter out definition
        include_tests=True,
    )

    # Filter out the definition itself and format results
    usages = []
    for result in results:
        raw_code = result.get("raw_code", "")
        rel_path = result.get("relative_path", "")
        line_start = result.get("line_start", 1)

        # Find matching lines in the raw_code
        lines = raw_code.splitlines()
        for i, line in enumerate(lines):
            if re.search(pattern, line):
                actual_line = line_start + i

                # Skip if it's the definition line
                if rel_path == defining_file and actual_line == defining_line:
                    continue

                # Skip if it looks like a definition
                content = line.strip()
                if element_type == "function" and content.startswith("def "):
                    continue
                if element_type == "class" and content.startswith("class "):
                    continue
                if element_type == "method" and content.startswith("def "):
                    continue

                usages.append({
                    "file": rel_path,
                    "line": actual_line,
                    "content": line,
                    "element_id": result.get("element_id"),
                    "element_name": result.get("name"),
                })

                if len(usages) >= limit:
                    return usages

    return usages
```

### Step 4: Run test to verify it passes

```bash
pytest tests/test_mcp_tools.py::TestFindUsagesWithPatternSearch -v
```

Expected: PASS

### Step 5: Commit

```bash
git add src/magaldi_mcp/tools.py tests/test_mcp_tools.py
git commit -m "refactor(mcp): update find_usages to use ES regexp search"
```

---

## Task 8: Update find_implementations to Use pattern_search

**Files:**
- Modify: `src/magaldi_mcp/tools.py`
- Test: `tests/test_mcp_tools.py`

### Step 1: Write test verifying find_implementations uses regexp search

Add to `tests/test_mcp_tools.py`:

```python
class TestFindImplementationsWithPatternSearch:
    """Tests for find_implementations using pattern_search."""

    def test_find_implementations_uses_regexp_search(self, mock_es_repo):
        """Test that find_implementations uses search_by_regexp."""
        mock_es_repo.get_document.return_value = {
            "element_id": "test:repo:main:base.py:class:BaseClass:1",
            "name": "BaseClass",
            "element_type": "class",
            "scope": "test",
            "repository": "repo",
        }
        mock_es_repo.search_by_regexp.return_value = [
            {
                "element_id": "test:repo:main:impl.py:class:MyImpl:10",
                "name": "MyImpl",
                "element_type": "class",
                "relative_path": "impl.py",
                "line_start": 10,
                "raw_code": "class MyImpl(BaseClass):\n    pass",
                "is_test": False,
            }
        ]

        result = find_implementations(
            es=mock_es_repo,
            element_id="test:repo:main:base.py:class:BaseClass:1",
        )

        mock_es_repo.search_by_regexp.assert_called()
        assert len(result) >= 1
```

### Step 2: Run test to verify it fails

```bash
pytest tests/test_mcp_tools.py::TestFindImplementationsWithPatternSearch -v
```

Expected: FAIL

### Step 3: Update find_implementations to use search_by_regexp

Update `find_implementations` in `src/magaldi_mcp/tools.py`:

```python
def find_implementations(
    es: ElasticsearchRepository,
    element_id: str | None = None,
    class_name: str | None = None,
    scope: str | None = None,
    repository: str | None = None,
    username: str = "main",
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Find classes that implement/inherit from a protocol or base class.

    Searches indexed code in Elasticsearch - no filesystem access needed.

    Args:
        es: Elasticsearch repository.
        element_id: Element ID of the protocol/base class.
        class_name: Or just the class name to search for.
        scope: Filter by scope.
        repository: Filter by repository.
        username: User branch to search.
        limit: Maximum implementations to return.

    Returns:
        List of implementing classes with their info.
    """
    import re

    # Get the name and scope/repo to search for
    if element_id:
        doc = es.get_document(element_id)
        if not doc:
            raise ValueError(f"Element not found: {element_id}")
        name = doc.get("name")
        scope = scope or doc.get("scope")
        repository = repository or doc.get("repository")
    elif class_name:
        name = class_name
    else:
        raise ValueError("Either element_id or class_name must be provided")

    if not scope or not repository:
        raise ValueError("scope and repository are required")

    # Search for class definitions that inherit from this class
    # Pattern: class SomeClass(BaseClass) or class SomeClass(module.BaseClass)
    pattern = rf"class\s+\w+\s*\([^)]*{re.escape(name)}[^)]*\)"

    results = es.search_by_regexp(
        pattern=pattern,
        scope=scope,
        repository=repository,
        username=username,
        size=limit,
        include_tests=True,
    )

    implementations = []
    for result in results:
        # Skip the base class itself
        if result.get("name") == name:
            continue

        implementations.append({
            "element_id": result.get("element_id"),
            "name": result.get("name"),
            "file": result.get("relative_path"),
            "line": result.get("line_start"),
            "raw_code": result.get("raw_code"),
            "is_test": result.get("is_test", False),
        })

    return implementations
```

### Step 4: Run test to verify it passes

```bash
pytest tests/test_mcp_tools.py::TestFindImplementationsWithPatternSearch -v
```

Expected: PASS

### Step 5: Commit

```bash
git add src/magaldi_mcp/tools.py tests/test_mcp_tools.py
git commit -m "refactor(mcp): update find_implementations to use ES regexp search"
```

---

## Task 9: Run Full Test Suite and Fix Any Issues

**Files:**
- All modified files

### Step 1: Run full test suite

```bash
pytest tests/test_mcp_tools.py tests/test_db_elasticsearch.py tests/test_mcp_server.py -v
```

### Step 2: Fix any failing tests

Review failures and fix. Common issues:
- Mock setup might need updating for new method signatures
- Import statements might be missing

### Step 3: Run integration tests if available

```bash
pytest tests/integration/ -v --timeout=60
```

### Step 4: Commit any fixes

```bash
git add -A
git commit -m "fix: address test failures after pattern_search implementation"
```

---

## Task 10: Update Magaldi Skill Documentation

**Files:**
- Modify: `.claude/skills/magaldi/SKILL.md` (if exists)

### Step 1: Check if skill file exists

```bash
ls -la .claude/skills/magaldi/
```

### Step 2: If exists, update to mention pattern_search

Add documentation about the new tool and when to use each mode.

### Step 3: Commit

```bash
git add .claude/skills/
git commit -m "docs: update magaldi skill with pattern_search documentation"
```

---

## Summary

Phase A implementation creates:

1. **Elasticsearch methods:** `search_by_regexp`, `search_by_wildcard`, `search_by_proximity`
2. **MCP tool:** `pattern_search` with mode selection
3. **Deprecation:** `grep_code` marked deprecated with warning
4. **Refactored:** `find_usages` and `find_implementations` to use ES-native search

**Performance improvement:** Queries now run server-side in Elasticsearch instead of fetching thousands of documents and filtering in Python.
