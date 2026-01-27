# Pattern Detection MCP Exposure & Improvements

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Expose detected design patterns via MCP tools and improve pattern detection heuristics.

**Architecture:** Add two MCP tools: `list_patterns` (list all patterns in a repo) and `find_by_pattern` (find classes by pattern type). Improve detection heuristics for singleton, builder, factory to catch more real-world cases.

**Tech Stack:** Python, Elasticsearch, MCP SDK, pytest

---

## Task 1: Add `list_patterns` MCP Tool

**Files:**
- Create: `src/magaldi_mcp/tools/schemas/patterns.py`
- Modify: `src/magaldi_mcp/tools/schemas/__init__.py`
- Modify: `src/magaldi_mcp/tools/__init__.py`
- Modify: `src/magaldi_mcp/tools_impl.py`
- Modify: `src/magaldi_mcp/server.py`
- Test: `tests/test_mcp_tools.py`

**Step 1: Write the failing test**

Add to `tests/test_mcp_tools.py`:

```python
class TestListPatterns:
    """Tests for list_patterns tool."""

    def test_returns_pattern_summary(self, es_repo_with_patterns):
        """Test list_patterns returns pattern distribution."""
        result = list_patterns(
            es_repo_with_patterns,
            scope="test",
            repository="test-repo",
        )
        assert "patterns" in result
        assert isinstance(result["patterns"], list)

    def test_includes_pattern_counts(self, es_repo_with_patterns):
        """Test each pattern includes count and examples."""
        result = list_patterns(
            es_repo_with_patterns,
            scope="test",
            repository="test-repo",
        )
        for pattern in result["patterns"]:
            assert "name" in pattern
            assert "count" in pattern
            assert "examples" in pattern
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_mcp_tools.py::TestListPatterns -v`
Expected: FAIL with "ImportError: cannot import name 'list_patterns'"

**Step 3: Create patterns schema file**

Create `src/magaldi_mcp/tools/schemas/patterns.py`:

```python
"""Pattern detection tool schemas."""

from mcp.types import Tool

PATTERN_TOOLS = [
    Tool(
        name="list_patterns",
        description="LIST PATTERNS: Show all detected design patterns in a repository. "
        "Returns pattern types (singleton, builder, factory, repository) with counts and example classes.",
        inputSchema={
            "type": "object",
            "properties": {
                "scope": {"type": "string", "description": "Repository scope (required)"},
                "repository": {"type": "string", "description": "Repository name (required)"},
                "username": {"type": "string", "description": "User branch"},
            },
            "required": ["scope", "repository"],
        },
    ),
    Tool(
        name="find_by_pattern",
        description="FIND BY PATTERN: Find all classes implementing a specific design pattern. "
        "Supports: singleton, builder, factory, repository.",
        inputSchema={
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "enum": ["singleton", "builder", "factory", "repository"],
                    "description": "Pattern type to search for",
                },
                "scope": {"type": "string", "description": "Repository scope (required)"},
                "repository": {"type": "string", "description": "Repository name (required)"},
                "username": {"type": "string", "description": "User branch"},
                "min_confidence": {
                    "type": "number",
                    "default": 0.6,
                    "description": "Minimum confidence score (0.0-1.0)",
                },
                "limit": {"type": "integer", "default": 20},
            },
            "required": ["pattern", "scope", "repository"],
        },
    ),
]
```

**Step 4: Update schemas __init__.py**

Modify `src/magaldi_mcp/tools/schemas/__init__.py`:

```python
"""Tool schemas for MCP server."""

from magaldi_mcp.tools.schemas.analysis import ANALYSIS_TOOLS
from magaldi_mcp.tools.schemas.dependencies import DEPENDENCY_TOOLS
from magaldi_mcp.tools.schemas.features import FEATURE_TOOLS
from magaldi_mcp.tools.schemas.files import FILE_TOOLS
from magaldi_mcp.tools.schemas.glossary import GLOSSARY_TOOLS
from magaldi_mcp.tools.schemas.inspect import INSPECT_TOOLS
from magaldi_mcp.tools.schemas.meta import META_TOOLS
from magaldi_mcp.tools.schemas.patterns import PATTERN_TOOLS
from magaldi_mcp.tools.schemas.search import SEARCH_TOOLS

# Combine all tool schemas
ALL_TOOL_SCHEMAS = (
    SEARCH_TOOLS +
    INSPECT_TOOLS +
    FILE_TOOLS +
    FEATURE_TOOLS +
    ANALYSIS_TOOLS +
    GLOSSARY_TOOLS +
    DEPENDENCY_TOOLS +
    PATTERN_TOOLS +
    META_TOOLS
)

__all__ = [
    "ALL_TOOL_SCHEMAS",
    "SEARCH_TOOLS",
    "INSPECT_TOOLS",
    "FILE_TOOLS",
    "FEATURE_TOOLS",
    "ANALYSIS_TOOLS",
    "GLOSSARY_TOOLS",
    "DEPENDENCY_TOOLS",
    "PATTERN_TOOLS",
    "META_TOOLS",
]
```

**Step 5: Implement list_patterns in tools_impl.py**

Add to `src/magaldi_mcp/tools_impl.py`:

```python
def list_patterns(
    es: ElasticsearchRepository,
    scope: str,
    repository: str,
    username: str = "main",
) -> dict[str, Any]:
    """List all detected design patterns in a repository.

    Args:
        es: Elasticsearch repository.
        scope: Repository scope.
        repository: Repository name.
        username: User branch.

    Returns:
        Dict with patterns list containing name, count, and examples.
    """
    client = es._get_client()

    # Query for classes with detected_patterns
    result = client.search(
        index=INDEX_NAME,
        body={
            "query": {
                "bool": {
                    "filter": [
                        {"term": {"scope": scope}},
                        {"term": {"repository": repository}},
                        {"term": {"username": username}},
                        {"term": {"element_type": "class"}},
                        {"exists": {"field": "detected_patterns"}},
                    ]
                }
            },
            "size": 1000,
            "_source": ["element_id", "name", "relative_path", "detected_patterns", "pattern_confidence"],
        },
    )

    # Group by pattern type
    pattern_map: dict[str, list[dict]] = {}
    for hit in result.get("hits", {}).get("hits", []):
        source = hit["_source"]
        patterns = source.get("detected_patterns", [])
        confidence = source.get("pattern_confidence", {})

        for pattern in patterns:
            if pattern not in pattern_map:
                pattern_map[pattern] = []
            pattern_map[pattern].append({
                "element_id": source["element_id"],
                "name": source["name"],
                "file": source.get("relative_path", ""),
                "confidence": confidence.get(pattern, 0.0),
            })

    # Format output
    patterns_list = []
    for pattern_name, classes in sorted(pattern_map.items()):
        # Sort by confidence descending
        classes.sort(key=lambda x: x["confidence"], reverse=True)
        patterns_list.append({
            "name": pattern_name,
            "count": len(classes),
            "examples": classes[:5],  # Top 5 examples
        })

    return {
        "patterns": patterns_list,
        "total_classes_with_patterns": sum(len(c) for c in pattern_map.values()),
    }
```

**Step 6: Implement find_by_pattern in tools_impl.py**

Add to `src/magaldi_mcp/tools_impl.py`:

```python
def find_by_pattern(
    es: ElasticsearchRepository,
    pattern: str,
    scope: str,
    repository: str,
    username: str = "main",
    min_confidence: float = 0.6,
    limit: int = 20,
) -> dict[str, Any]:
    """Find all classes implementing a specific design pattern.

    Args:
        es: Elasticsearch repository.
        pattern: Pattern type (singleton, builder, factory, repository).
        scope: Repository scope.
        repository: Repository name.
        username: User branch.
        min_confidence: Minimum confidence threshold.
        limit: Maximum results.

    Returns:
        Dict with matching classes.
    """
    client = es._get_client()

    result = client.search(
        index=INDEX_NAME,
        body={
            "query": {
                "bool": {
                    "filter": [
                        {"term": {"scope": scope}},
                        {"term": {"repository": repository}},
                        {"term": {"username": username}},
                        {"term": {"element_type": "class"}},
                        {"term": {"detected_patterns": pattern}},
                    ]
                }
            },
            "size": limit,
            "_source": [
                "element_id", "name", "relative_path", "line_start",
                "detected_patterns", "pattern_confidence", "summary",
                "class_attributes", "base_classes",
            ],
        },
    )

    # Filter by confidence and format
    classes = []
    for hit in result.get("hits", {}).get("hits", []):
        source = hit["_source"]
        confidence = source.get("pattern_confidence", {}).get(pattern, 0.0)

        if confidence >= min_confidence:
            classes.append({
                "element_id": source["element_id"],
                "name": source["name"],
                "file": source.get("relative_path", ""),
                "line": source.get("line_start"),
                "confidence": confidence,
                "summary": source.get("summary", ""),
                "all_patterns": source.get("detected_patterns", []),
            })

    # Sort by confidence
    classes.sort(key=lambda x: x["confidence"], reverse=True)

    return {
        "pattern": pattern,
        "count": len(classes),
        "classes": classes,
    }
```

**Step 7: Export from tools/__init__.py**

Add to exports in `src/magaldi_mcp/tools/__init__.py`:

```python
from magaldi_mcp.tools_impl import (
    # ... existing imports ...
    find_by_pattern,
    list_patterns,
)

__all__ = [
    # ... existing exports ...
    "find_by_pattern",
    "list_patterns",
]
```

**Step 8: Wire up in server.py**

Add to `_handle_tool` method in `src/magaldi_mcp/server.py`:

```python
elif name == "list_patterns":
    return await asyncio.to_thread(
        list_patterns,
        es,
        scope=args["scope"],
        repository=args["repository"],
        username=args.get("username", self.default_username),
    )
elif name == "find_by_pattern":
    return await asyncio.to_thread(
        find_by_pattern,
        es,
        pattern=args["pattern"],
        scope=args["scope"],
        repository=args["repository"],
        username=args.get("username", self.default_username),
        min_confidence=args.get("min_confidence", 0.6),
        limit=args.get("limit", 20),
    )
```

**Step 9: Run tests to verify they pass**

Run: `pytest tests/test_mcp_tools.py::TestListPatterns -v`
Expected: PASS

**Step 10: Commit**

```bash
git add src/magaldi_mcp/tools/schemas/patterns.py src/magaldi_mcp/tools/schemas/__init__.py src/magaldi_mcp/tools/__init__.py src/magaldi_mcp/tools_impl.py src/magaldi_mcp/server.py tests/test_mcp_tools.py
git commit -m "feat(mcp): add list_patterns and find_by_pattern tools"
```

---

## Task 2: Improve Singleton Detection

**Files:**
- Modify: `src/magaldi_core/analysis/api_detection.py`
- Modify: `tests/test_code_intelligence.py`

**Step 1: Write failing test for improved singleton detection**

Add to `tests/test_code_intelligence.py` in `TestPatternDetection`:

```python
def test_detect_singleton_with_class_variable(self):
    """Detect singleton with class-level _instance variable."""
    class_info = {
        "name": "Logger",
        "attributes": [],  # No instance attributes
        "methods": ["__new__", "log", "error"],
        "class_variables": ["_instance"],
    }
    patterns, confidence = detect_patterns(class_info, [], "python")
    assert "singleton" in patterns

def test_detect_singleton_with_instance_method(self):
    """Detect singleton with instance() class method."""
    class_info = {
        "name": "Configuration",
        "attributes": ["_settings"],
        "methods": ["instance", "get", "set"],
        "decorators": ["classmethod"],
    }
    patterns, confidence = detect_patterns(class_info, [], "python")
    assert "singleton" in patterns
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_code_intelligence.py::TestPatternDetection::test_detect_singleton_with_class_variable -v`
Expected: FAIL

**Step 3: Improve _detect_singleton in api_detection.py**

Modify `_detect_singleton` in `src/magaldi_core/analysis/api_detection.py`:

```python
def _detect_singleton(class_info: dict[str, Any]) -> float:
    """Detect singleton pattern.

    Looks for:
    - _instance attribute (instance or class level)
    - get_instance/getInstance/instance methods
    - __new__ method override
    - @classmethod or @staticmethod on instance getter
    """
    score = 0.0
    attributes = class_info.get("attributes", [])
    methods = class_info.get("methods", [])
    class_variables = class_info.get("class_variables", [])
    decorators = class_info.get("decorators", [])

    # Has _instance attribute (instance or class level)
    instance_attrs = ["_instance", "instance", "_singleton", "_shared_instance"]
    if any(attr in attributes for attr in instance_attrs):
        score += 0.3
    if any(attr in class_variables for attr in instance_attrs):
        score += 0.3

    # Has get_instance/getInstance/instance method
    instance_methods = ["get_instance", "getInstance", "instance", "shared", "default"]
    if any(m in methods for m in instance_methods):
        score += 0.3

    # Has __new__ method (Python singleton pattern)
    if "__new__" in methods:
        score += 0.2

    # Uses classmethod decorator (common for singletons)
    if "classmethod" in decorators:
        score += 0.1

    # Returns self/instance from get_instance
    if class_info.get("method_returns_self"):
        score += 0.1

    return min(score, 1.0)  # Cap at 1.0
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_code_intelligence.py::TestPatternDetection::test_detect_singleton -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/magaldi_core/analysis/api_detection.py tests/test_code_intelligence.py
git commit -m "feat(patterns): improve singleton detection heuristics"
```

---

## Task 3: Improve Builder Detection

**Files:**
- Modify: `src/magaldi_core/analysis/api_detection.py`
- Modify: `tests/test_code_intelligence.py`

**Step 1: Write failing test for improved builder detection**

Add to `tests/test_code_intelligence.py`:

```python
def test_detect_builder_with_fluent_methods(self):
    """Detect builder with with_* methods."""
    class_info = {
        "name": "RequestBuilder",
        "attributes": ["_url", "_headers", "_body"],
        "methods": ["with_url", "with_header", "with_body", "send"],
    }
    patterns, confidence = detect_patterns(class_info, [], "python")
    assert "builder" in patterns

def test_detect_builder_with_set_methods(self):
    """Detect builder with set_* chained methods."""
    class_info = {
        "name": "ConfigBuilder",
        "attributes": ["_config"],
        "methods": ["set_timeout", "set_retries", "set_base_url", "build"],
        "methods_return_self": ["set_timeout", "set_retries", "set_base_url"],
    }
    patterns, confidence = detect_patterns(class_info, [], "python")
    assert "builder" in patterns
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_code_intelligence.py::TestPatternDetection::test_detect_builder_with_fluent_methods -v`
Expected: FAIL

**Step 3: Improve _detect_builder in api_detection.py**

```python
def _detect_builder(class_info: dict[str, Any]) -> float:
    """Detect builder pattern.

    Looks for:
    - Multiple methods returning self (method chaining)
    - build() method
    - with_*/set_*/add_* method naming
    - *Builder class name suffix
    """
    score = 0.0
    methods = class_info.get("methods", [])
    returns_self = class_info.get("methods_return_self", [])
    name = class_info.get("name", "")

    # Multiple methods that return self (method chaining)
    if len(returns_self) >= 2:
        score += 0.4

    # Has a build() method
    if "build" in methods:
        score += 0.3

    # Name ends with Builder
    if name.endswith("Builder"):
        score += 0.3

    # Has with_* methods (fluent interface)
    with_methods = [m for m in methods if m.startswith("with_")]
    if len(with_methods) >= 2:
        score += 0.3

    # Has set_* methods (common builder pattern)
    set_methods = [m for m in methods if m.startswith("set_")]
    if len(set_methods) >= 2:
        score += 0.2

    # Has add_* methods (collection builders)
    add_methods = [m for m in methods if m.startswith("add_")]
    if len(add_methods) >= 2:
        score += 0.2

    return min(score, 1.0)  # Cap at 1.0
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_code_intelligence.py::TestPatternDetection::test_detect_builder -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/magaldi_core/analysis/api_detection.py tests/test_code_intelligence.py
git commit -m "feat(patterns): improve builder detection heuristics"
```

---

## Task 4: Improve Factory Detection

**Files:**
- Modify: `src/magaldi_core/analysis/api_detection.py`
- Modify: `tests/test_code_intelligence.py`

**Step 1: Write failing test for improved factory detection**

```python
def test_detect_factory_with_from_methods(self):
    """Detect factory with from_* class methods."""
    class_info = {
        "name": "Parser",
        "attributes": [],
        "methods": ["from_string", "from_file", "from_dict", "parse"],
        "decorators": ["classmethod"],
    }
    patterns, confidence = detect_patterns(class_info, [], "python")
    assert "factory" in patterns

def test_detect_factory_with_build_methods(self):
    """Detect factory with build_* methods."""
    class_info = {
        "name": "ConnectionManager",
        "attributes": ["_pool"],
        "methods": ["build_connection", "build_pool", "close"],
    }
    patterns, confidence = detect_patterns(class_info, [], "python")
    assert "factory" in patterns
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_code_intelligence.py::TestPatternDetection::test_detect_factory_with_from_methods -v`
Expected: FAIL

**Step 3: Improve _detect_factory in api_detection.py**

```python
def _detect_factory(class_info: dict[str, Any], calls: list[ExtractedCall]) -> float:
    """Detect factory pattern.

    Looks for:
    - *Factory class name
    - create_*/make_*/build_*/from_*/new_* methods
    - Methods that instantiate other classes
    - @classmethod/@staticmethod decorators
    """
    score = 0.0
    methods = class_info.get("methods", [])
    name = class_info.get("name", "")
    decorators = class_info.get("decorators", [])

    # Name contains Factory
    if "Factory" in name or "factory" in name.lower():
        score += 0.3

    # Has create*/make*/build*/from*/new* methods
    factory_prefixes = ("create", "make", "build", "from_", "new_")
    factory_methods = [m for m in methods if any(m.startswith(p) or m.startswith(p.title()) for p in factory_prefixes)]
    if factory_methods:
        score += 0.3
    if len(factory_methods) >= 2:
        score += 0.1

    # Methods instantiate other classes (uppercase call = class constructor)
    instantiation_calls = [c for c in calls if c.receiver is None and c.name and c.name[0].isupper()]
    if instantiation_calls:
        score += 0.3

    # Uses classmethod/staticmethod (common for factories)
    if "classmethod" in decorators or "staticmethod" in decorators:
        score += 0.1

    return min(score, 1.0)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_code_intelligence.py::TestPatternDetection -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/magaldi_core/analysis/api_detection.py tests/test_code_intelligence.py
git commit -m "feat(patterns): improve factory detection heuristics"
```

---

## Task 5: Extract class_variables During Parsing

**Files:**
- Modify: `src/magaldi_core/code_parser.py`
- Modify: `src/magaldi_core/extractors/python.py` (if needed)
- Test: `tests/test_code_parser.py`

**Step 1: Write failing test**

Add to `tests/test_code_parser.py`:

```python
def test_extracts_class_variables_for_patterns(self, parser):
    """Test that class variables are extracted for pattern detection."""
    code = '''
class Singleton:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
'''
    elements = parser.parse_content(code, "test.py", FileInfo("test.py", ""))
    class_elem = next(e for e in elements if e.element_type == "class")

    # Should detect singleton pattern
    assert "singleton" in (class_elem.detected_patterns or [])
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_code_parser.py::test_extracts_class_variables_for_patterns -v`
Expected: FAIL

**Step 3: Modify code_parser.py to extract class variables**

In the pattern detection block around line 828, add class variable extraction:

```python
# Pattern detection (for classes)
if elem.element_type == "class":
    # Collect method names from child elements
    class_methods = [
        e.name for e in elements
        if e.parent_id == elem.element_id and e.element_type == "method"
    ]

    # Extract class-level variables from class_attributes
    class_variables = []
    if elem.class_attributes:
        class_variables = [a.get("name", "") for a in elem.class_attributes if a.get("line", 0) < 10]  # Class-level vars near top

    class_info = {
        "name": elem.name,
        "attributes": [a.get("name", "") for a in (elem.class_attributes or [])],
        "methods": class_methods,
        "class_variables": class_variables,
        "decorators": elem.decorators or [],
    }
    patterns, confidence = detect_patterns(class_info, [], "python")
    elem.detected_patterns = patterns
    elem.pattern_confidence = confidence
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_code_parser.py::test_extracts_class_variables_for_patterns -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/magaldi_core/code_parser.py tests/test_code_parser.py
git commit -m "feat(parser): extract class variables for pattern detection"
```

---

## Task 6: Add Integration Test for Pattern MCP Tools

**Files:**
- Test: `tests/test_mcp_server.py`

**Step 1: Write integration test**

Add to `tests/test_mcp_server.py`:

```python
class TestPatternTools:
    """Integration tests for pattern detection MCP tools."""

    @pytest.fixture
    def es_with_patterns(self, es_repo):
        """Set up ES with classes having patterns."""
        from shared.models import CodeElement

        # Index a repository class
        elem = CodeElement(
            scope="test",
            repository="test-repo",
            username="main",
            relative_path="src/repos.py",
            element_type="class",
            name="UserRepository",
            language="python",
            line_start=1,
            line_end=50,
            raw_code="class UserRepository: ...",
        )
        elem.detected_patterns = ["repository"]
        elem.pattern_confidence = {"repository": 0.8}
        es_repo.index_element(elem)

        # Refresh index
        es_repo._get_client().indices.refresh(index="magaldi-code-elements")
        return es_repo

    def test_list_patterns_returns_data(self, es_with_patterns):
        """Test list_patterns returns pattern data."""
        from magaldi_mcp.tools import list_patterns

        result = list_patterns(es_with_patterns, "test", "test-repo")

        assert result["patterns"]
        assert result["patterns"][0]["name"] == "repository"
        assert result["patterns"][0]["count"] == 1

    def test_find_by_pattern_returns_classes(self, es_with_patterns):
        """Test find_by_pattern returns matching classes."""
        from magaldi_mcp.tools import find_by_pattern

        result = find_by_pattern(es_with_patterns, "repository", "test", "test-repo")

        assert result["count"] == 1
        assert result["classes"][0]["name"] == "UserRepository"
```

**Step 2: Run test**

Run: `pytest tests/test_mcp_server.py::TestPatternTools -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/test_mcp_server.py
git commit -m "test(mcp): add integration tests for pattern tools"
```

---

## Summary

| Task | Description | Files |
|------|-------------|-------|
| 1 | Add `list_patterns` and `find_by_pattern` MCP tools | 6 files |
| 2 | Improve singleton detection | 2 files |
| 3 | Improve builder detection | 2 files |
| 4 | Improve factory detection | 2 files |
| 5 | Extract class_variables for pattern detection | 2 files |
| 6 | Integration tests | 1 file |

After completing these tasks:
- Run `make test` to verify all tests pass
- Re-index the Magaldi codebase: `magaldi parse . --user main`
- Test the new MCP tools via Claude Code
