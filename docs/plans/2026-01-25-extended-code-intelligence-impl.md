# Extended Code Intelligence Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement 5 code intelligence features: Type Flow, Pattern Detection, Documentation Linkage, API Surface Analysis, and Purity/Mutation Tracking.

**Architecture:** Extend existing `CodeElement` dataclass with new fields, add extraction functions in `tree_sitter_manager.py`, update ES mapping, and add ~18 new MCP tools. Extractors run during parsing phase, data flows to ES, MCP tools query it.

**Tech Stack:** Python dataclasses, Tree-sitter AST, Elasticsearch nested fields, MCP SDK tools.

---

## Phase 1: Data Model & Storage Foundation

### Task 1.1: Add Supporting Dataclasses to tree_sitter_manager.py

**Files:**
- Modify: `src/magaldi_core/tree_sitter_manager.py` (after line 100, before TreeSitterManager)

**Step 1: Add the dataclasses**

Add these dataclasses after `ExtractedCall` (around line 100):

```python
# =============================================================================
# EXTENDED CODE INTELLIGENCE DATA STRUCTURES
# =============================================================================


@dataclass
class TypeAnnotation:
    """Type annotation extracted from source code."""

    name: str  # "User", "List[str]", "Optional[int]"
    kind: str  # "parameter", "return", "variable", "attribute"
    location: str  # "param:user_id", "return", "var:result"
    line: int
    generic_args: list[str] | None = None  # ["str"] for List[str]


@dataclass
class TodoItem:
    """A TODO/FIXME comment extracted from source code."""

    kind: str  # "TODO", "FIXME", "HACK", "XXX", "BUG", "NOTE"
    text: str
    line: int
    assignee: str | None = None  # "alice" from TODO(alice)
    priority: str | None = None  # "high", "low" (from ! markers)
    issue_ref: str | None = None  # "GH-123", "#456"


@dataclass
class SectionMarker:
    """A section marker comment (e.g., # === HELPERS ===)."""

    label: str  # "HELPERS", "PRIVATE METHODS"
    line: int
    style: str  # "equals", "dashes", "hash"


@dataclass
class Comment:
    """A comment associated with a code element."""

    text: str
    line: int
    kind: str  # "inline", "block", "docstring"
    position: str  # "above", "inline", "below"


@dataclass
class HttpRoute:
    """An HTTP route extracted from a web framework."""

    method: str  # "GET", "POST", "PUT", "DELETE"
    path: str  # "/users/{id}"
    path_params: list[str]  # ["id"]
    framework: str  # "fastapi", "flask", "express"


@dataclass
class CliCommand:
    """A CLI command extracted from a CLI framework."""

    name: str  # "parse", "index"
    options: list[dict[str, Any]]  # CliOption as dicts
    framework: str  # "click", "typer", "argparse"


@dataclass
class PurityInfo:
    """Purity analysis result for a function."""

    level: str  # "pure", "read_only", "mutates_self", "mutates_external"
    confidence: str  # "high", "medium", "low"
    reasons: list[str]  # ["calls print()", "modifies self.cache"]


@dataclass
class SideEffect:
    """A side effect detected in a function."""

    kind: str  # "state_mutation", "io_file", "io_network", "console", "subprocess"
    target: str | None  # "self.cache", "/tmp/file.txt"
    line: int
```

**Step 2: Verify no syntax errors**

Run: `python -c "from magaldi_core.tree_sitter_manager import TypeAnnotation, TodoItem, PurityInfo"`
Expected: No output (success)

**Step 3: Commit**

```bash
git add src/magaldi_core/tree_sitter_manager.py
git commit -m "feat: add dataclasses for extended code intelligence"
```

---

### Task 1.2: Extend CodeElement with New Fields

**Files:**
- Modify: `src/magaldi_core/code_parser.py` (CodeElement dataclass around line 179)

**Step 1: Add new fields to CodeElement**

Add these fields after `content_hash` (around line 240):

```python
    # === EXTENDED CODE INTELLIGENCE FIELDS ===

    # Type Flow
    type_annotations: list[dict[str, Any]] = field(default_factory=list)

    # Pattern Detection
    detected_patterns: list[str] = field(default_factory=list)  # ["singleton", "factory"]
    pattern_confidence: dict[str, float] = field(default_factory=dict)  # {"singleton": 0.95}

    # Documentation
    todos: list[dict[str, Any]] = field(default_factory=list)
    section_markers: list[dict[str, Any]] = field(default_factory=list)
    associated_comments: list[dict[str, Any]] = field(default_factory=list)

    # API Surface
    is_public_api: bool = False
    http_routes: list[dict[str, Any]] = field(default_factory=list)
    cli_commands: list[dict[str, Any]] = field(default_factory=list)

    # Purity/Mutation
    purity: dict[str, Any] | None = None
    side_effects: list[dict[str, Any]] = field(default_factory=list)
    mutated_state: list[str] = field(default_factory=list)
```

**Step 2: Verify import and usage**

Run: `python -c "from magaldi_core.code_parser import CodeElement; e = CodeElement(); print(e.detected_patterns, e.purity)"`
Expected: `[] None`

**Step 3: Commit**

```bash
git add src/magaldi_core/code_parser.py
git commit -m "feat: extend CodeElement with code intelligence fields"
```

---

### Task 1.3: Update Elasticsearch Mapping

**Files:**
- Modify: `src/shared/db/elasticsearch.py` (INDEX_MAPPING around line 37)

**Step 1: Add new mapping properties**

Add these properties inside `INDEX_MAPPING["mappings"]["properties"]` (after line 153, before the closing brace):

```python
            # === EXTENDED CODE INTELLIGENCE MAPPINGS ===
            # Type Flow
            "type_annotations": {
                "type": "nested",
                "properties": {
                    "name": {"type": "keyword"},
                    "kind": {"type": "keyword"},
                    "location": {"type": "keyword"},
                    "line": {"type": "integer"},
                    "generic_args": {"type": "keyword"},
                },
            },
            # Pattern Detection
            "detected_patterns": {"type": "keyword"},
            "pattern_confidence": {"type": "object"},
            # Documentation
            "todos": {
                "type": "nested",
                "properties": {
                    "kind": {"type": "keyword"},
                    "text": {"type": "text"},
                    "line": {"type": "integer"},
                    "assignee": {"type": "keyword"},
                    "priority": {"type": "keyword"},
                    "issue_ref": {"type": "keyword"},
                },
            },
            "section_markers": {
                "type": "nested",
                "properties": {
                    "label": {"type": "keyword"},
                    "line": {"type": "integer"},
                    "style": {"type": "keyword"},
                },
            },
            "associated_comments": {
                "type": "nested",
                "properties": {
                    "text": {"type": "text"},
                    "line": {"type": "integer"},
                    "kind": {"type": "keyword"},
                    "position": {"type": "keyword"},
                },
            },
            # API Surface
            "is_public_api": {"type": "boolean"},
            "http_routes": {
                "type": "nested",
                "properties": {
                    "method": {"type": "keyword"},
                    "path": {"type": "keyword"},
                    "path_params": {"type": "keyword"},
                    "framework": {"type": "keyword"},
                },
            },
            "cli_commands": {
                "type": "nested",
                "properties": {
                    "name": {"type": "keyword"},
                    "options": {"type": "nested"},
                    "framework": {"type": "keyword"},
                },
            },
            # Purity/Mutation
            "purity": {
                "type": "object",
                "properties": {
                    "level": {"type": "keyword"},
                    "confidence": {"type": "keyword"},
                    "reasons": {"type": "keyword"},
                },
            },
            "side_effects": {
                "type": "nested",
                "properties": {
                    "kind": {"type": "keyword"},
                    "target": {"type": "keyword"},
                    "line": {"type": "integer"},
                },
            },
            "mutated_state": {"type": "keyword"},
```

**Step 2: Verify mapping is valid Python**

Run: `python -c "from shared.db.elasticsearch import INDEX_MAPPING; print(len(INDEX_MAPPING['mappings']['properties']))"`
Expected: Number > 50 (approx 55-60 properties)

**Step 3: Commit**

```bash
git add src/shared/db/elasticsearch.py
git commit -m "feat: add ES mapping for code intelligence fields"
```

---

### Task 1.4: Update index_element() to Store New Fields

**Files:**
- Modify: `src/shared/db/elasticsearch.py` (index_element method around line 197)

**Step 1: Add new fields to the doc dict**

In `index_element()`, after the existing enhanced context fields (around line 265), add:

```python
        # Extended code intelligence fields
        if element.type_annotations:
            doc["type_annotations"] = element.type_annotations
        if element.detected_patterns:
            doc["detected_patterns"] = element.detected_patterns
        if element.pattern_confidence:
            doc["pattern_confidence"] = element.pattern_confidence
        if element.todos:
            doc["todos"] = element.todos
        if element.section_markers:
            doc["section_markers"] = element.section_markers
        if element.associated_comments:
            doc["associated_comments"] = element.associated_comments
        if element.is_public_api:
            doc["is_public_api"] = element.is_public_api
        if element.http_routes:
            doc["http_routes"] = element.http_routes
        if element.cli_commands:
            doc["cli_commands"] = element.cli_commands
        if element.purity:
            doc["purity"] = element.purity
        if element.side_effects:
            doc["side_effects"] = element.side_effects
        if element.mutated_state:
            doc["mutated_state"] = element.mutated_state
```

**Step 2: Verify no syntax errors**

Run: `python -c "from shared.db.elasticsearch import ElasticsearchRepository"`
Expected: No output (success)

**Step 3: Commit**

```bash
git add src/shared/db/elasticsearch.py
git commit -m "feat: store code intelligence fields in ES"
```

---

## Phase 2: Comment/TODO Extraction

### Task 2.1: Implement TODO Extraction

**Files:**
- Modify: `src/magaldi_core/tree_sitter_manager.py`
- Test: `tests/test_code_intelligence.py` (create new file)

**Step 1: Write the test**

Create `tests/test_code_intelligence.py`:

```python
"""Tests for extended code intelligence extraction."""

import pytest

from magaldi_core.tree_sitter_manager import (
    TodoItem,
    extract_todos,
)


class TestTodoExtraction:
    """Tests for TODO comment extraction."""

    def test_extract_simple_todo(self):
        source = "# TODO: fix this bug"
        todos = extract_todos(source)
        assert len(todos) == 1
        assert todos[0].kind == "TODO"
        assert todos[0].text == "fix this bug"
        assert todos[0].line == 1

    def test_extract_todo_with_assignee(self):
        source = "# TODO(alice): review this code"
        todos = extract_todos(source)
        assert len(todos) == 1
        assert todos[0].assignee == "alice"
        assert todos[0].text == "review this code"

    def test_extract_todo_with_issue_ref(self):
        source = "# TODO #123: implement feature"
        todos = extract_todos(source)
        assert len(todos) == 1
        assert todos[0].issue_ref == "#123"

    def test_extract_fixme(self):
        source = "# FIXME: memory leak here"
        todos = extract_todos(source)
        assert len(todos) == 1
        assert todos[0].kind == "FIXME"

    def test_extract_multiple_todos(self):
        source = """# TODO: first thing
def foo():
    # FIXME: second thing
    pass
"""
        todos = extract_todos(source)
        assert len(todos) == 2
        assert todos[0].line == 1
        assert todos[1].line == 3

    def test_extract_todo_with_priority(self):
        source = "# TODO!!: urgent fix needed"
        todos = extract_todos(source)
        assert len(todos) == 1
        assert todos[0].priority == "high"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_code_intelligence.py::TestTodoExtraction -v`
Expected: FAIL with "cannot import name 'extract_todos'"

**Step 3: Implement extract_todos**

Add to `src/magaldi_core/tree_sitter_manager.py` after the new dataclasses:

```python
import re

# TODO extraction pattern
_TODO_PATTERN = re.compile(
    r"(?P<kind>TODO|FIXME|HACK|XXX|BUG|NOTE|OPTIMIZE)"
    r"(?:\((?P<assignee>\w+)\))?"  # TODO(alice)
    r"(?:\s*(?P<priority>!+))?"  # TODO!!
    r"(?:\s*(?P<issue>[#]?\w+-?\d+))?"  # TODO #123 or GH-456
    r"\s*:?\s*"
    r"(?P<text>.+)",
    re.IGNORECASE,
)


def extract_todos(source: str) -> list[TodoItem]:
    """Extract TODO/FIXME comments from source code.

    Args:
        source: Source code string.

    Returns:
        List of TodoItem objects.
    """
    todos = []
    lines = source.split("\n")

    for line_num, line in enumerate(lines, start=1):
        # Look for comment markers
        comment_start = -1
        for marker in ("#", "//", "/*", "*"):
            idx = line.find(marker)
            if idx != -1 and (comment_start == -1 or idx < comment_start):
                comment_start = idx

        if comment_start == -1:
            continue

        comment_text = line[comment_start:]
        match = _TODO_PATTERN.search(comment_text)
        if match:
            priority = None
            if match.group("priority"):
                priority = "high" if len(match.group("priority")) >= 2 else "low"

            todos.append(
                TodoItem(
                    kind=match.group("kind").upper(),
                    text=match.group("text").strip(),
                    line=line_num,
                    assignee=match.group("assignee"),
                    priority=priority,
                    issue_ref=match.group("issue"),
                )
            )

    return todos
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_code_intelligence.py::TestTodoExtraction -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add src/magaldi_core/tree_sitter_manager.py tests/test_code_intelligence.py
git commit -m "feat: implement TODO/FIXME extraction"
```

---

### Task 2.2: Implement Section Marker Extraction

**Files:**
- Modify: `src/magaldi_core/tree_sitter_manager.py`
- Modify: `tests/test_code_intelligence.py`

**Step 1: Write the test**

Add to `tests/test_code_intelligence.py`:

```python
from magaldi_core.tree_sitter_manager import (
    SectionMarker,
    extract_section_markers,
)


class TestSectionMarkerExtraction:
    """Tests for section marker extraction."""

    def test_extract_equals_style(self):
        source = "# === HELPERS ==="
        markers = extract_section_markers(source)
        assert len(markers) == 1
        assert markers[0].label == "HELPERS"
        assert markers[0].style == "equals"

    def test_extract_dashes_style(self):
        source = "# --- PRIVATE METHODS ---"
        markers = extract_section_markers(source)
        assert len(markers) == 1
        assert markers[0].label == "PRIVATE METHODS"
        assert markers[0].style == "dashes"

    def test_extract_multiple_markers(self):
        source = """# === IMPORTS ===
import os

# === HELPERS ===
def helper():
    pass
"""
        markers = extract_section_markers(source)
        assert len(markers) == 2
        assert markers[0].label == "IMPORTS"
        assert markers[1].label == "HELPERS"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_code_intelligence.py::TestSectionMarkerExtraction -v`
Expected: FAIL with "cannot import name 'extract_section_markers'"

**Step 3: Implement extract_section_markers**

Add to `src/magaldi_core/tree_sitter_manager.py`:

```python
# Section marker pattern
_SECTION_PATTERN = re.compile(
    r"^\s*[#/]+\s*"
    r"(?P<style_start>={3,}|-{3,})?\s*"
    r"(?P<label>[A-Z][A-Z0-9 _]+)"
    r"\s*(?P<style_end>={3,}|-{3,})?\s*$",
)


def extract_section_markers(source: str) -> list[SectionMarker]:
    """Extract section marker comments from source code.

    Args:
        source: Source code string.

    Returns:
        List of SectionMarker objects.
    """
    markers = []
    lines = source.split("\n")

    for line_num, line in enumerate(lines, start=1):
        match = _SECTION_PATTERN.match(line)
        if match:
            style_start = match.group("style_start") or ""
            style_end = match.group("style_end") or ""

            if "=" in style_start or "=" in style_end:
                style = "equals"
            elif "-" in style_start or "-" in style_end:
                style = "dashes"
            else:
                style = "hash"

            markers.append(
                SectionMarker(
                    label=match.group("label").strip(),
                    line=line_num,
                    style=style,
                )
            )

    return markers
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_code_intelligence.py::TestSectionMarkerExtraction -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add src/magaldi_core/tree_sitter_manager.py tests/test_code_intelligence.py
git commit -m "feat: implement section marker extraction"
```

---

### Task 2.3: Implement Comment Association

**Files:**
- Modify: `src/magaldi_core/tree_sitter_manager.py`
- Modify: `tests/test_code_intelligence.py`

**Step 1: Write the test**

Add to `tests/test_code_intelligence.py`:

```python
from magaldi_core.tree_sitter_manager import (
    Comment,
    extract_comments,
    associate_comments,
)


class TestCommentExtraction:
    """Tests for comment extraction and association."""

    def test_extract_inline_comment(self):
        source = "x = 1  # set x to one"
        comments = extract_comments(source)
        assert len(comments) == 1
        assert comments[0].kind == "inline"
        assert "set x to one" in comments[0].text

    def test_extract_block_comment(self):
        source = "# This is a block comment\ndef foo(): pass"
        comments = extract_comments(source)
        assert len(comments) == 1
        assert comments[0].kind == "block"

    def test_associate_comment_above(self):
        comments = [Comment(text="Helper function", line=5, kind="block", position="above")]
        element_line = 6
        associated = associate_comments(element_line, comments, max_distance=3)
        assert len(associated) == 1

    def test_no_associate_distant_comment(self):
        comments = [Comment(text="Far away", line=1, kind="block", position="above")]
        element_line = 10
        associated = associate_comments(element_line, comments, max_distance=3)
        assert len(associated) == 0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_code_intelligence.py::TestCommentExtraction -v`
Expected: FAIL with import error

**Step 3: Implement extract_comments and associate_comments**

Add to `src/magaldi_core/tree_sitter_manager.py`:

```python
def extract_comments(source: str) -> list[Comment]:
    """Extract all comments from source code.

    Args:
        source: Source code string.

    Returns:
        List of Comment objects.
    """
    comments = []
    lines = source.split("\n")

    for line_num, line in enumerate(lines, start=1):
        stripped = line.strip()

        # Skip empty lines
        if not stripped:
            continue

        # Python/shell style comments
        if stripped.startswith("#"):
            # Check if it's inline (code before the #)
            hash_pos = line.find("#")
            code_before = line[:hash_pos].strip()
            kind = "inline" if code_before else "block"
            position = "inline" if kind == "inline" else "above"

            comments.append(
                Comment(
                    text=stripped[1:].strip(),
                    line=line_num,
                    kind=kind,
                    position=position,
                )
            )
        # C-style single line comments
        elif "//" in line:
            slash_pos = line.find("//")
            code_before = line[:slash_pos].strip()
            kind = "inline" if code_before else "block"
            position = "inline" if kind == "inline" else "above"

            comments.append(
                Comment(
                    text=line[slash_pos + 2 :].strip(),
                    line=line_num,
                    kind=kind,
                    position=position,
                )
            )

    return comments


def associate_comments(
    element_line: int,
    all_comments: list[Comment],
    max_distance: int = 3,
) -> list[Comment]:
    """Associate comments with an element based on proximity.

    Args:
        element_line: Line number of the element (1-indexed).
        all_comments: All extracted comments.
        max_distance: Maximum line distance for association.

    Returns:
        List of comments associated with this element.
    """
    associated = []

    for comment in all_comments:
        # Comments above the element (within max_distance)
        if element_line - max_distance <= comment.line < element_line:
            comment.position = "above"
            associated.append(comment)
        # Inline comments on the same line
        elif comment.line == element_line and comment.kind == "inline":
            comment.position = "inline"
            associated.append(comment)

    return associated
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_code_intelligence.py::TestCommentExtraction -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add src/magaldi_core/tree_sitter_manager.py tests/test_code_intelligence.py
git commit -m "feat: implement comment extraction and association"
```

---

## Phase 3: Purity Analysis

### Task 3.1: Define Impure Call Patterns

**Files:**
- Modify: `src/magaldi_core/tree_sitter_manager.py`

**Step 1: Add impure call patterns**

Add after the section marker pattern:

```python
# Impure function call patterns by language
IMPURE_CALLS: dict[str, dict[str, list[str]]] = {
    "python": {
        "io_file": [
            "open", "read", "write", "Path.write_text", "Path.read_text",
            "Path.mkdir", "os.remove", "shutil.copy", "shutil.move",
        ],
        "io_network": [
            "requests.get", "requests.post", "requests.put", "requests.delete",
            "httpx.get", "httpx.post", "urllib.request.urlopen", "socket.connect",
        ],
        "console": [
            "print", "logging.info", "logging.debug", "logging.warning",
            "logging.error", "logger.info", "logger.debug", "logger.warning",
        ],
        "subprocess": [
            "subprocess.run", "subprocess.call", "subprocess.Popen",
            "os.system", "os.popen",
        ],
        "database": [
            "cursor.execute", "session.commit", "session.add", "session.delete",
        ],
    },
    "javascript": {
        "io_file": [
            "fs.readFile", "fs.writeFile", "fs.readFileSync", "fs.writeFileSync",
            "fs.appendFile", "fs.unlink", "fs.mkdir",
        ],
        "io_network": [
            "fetch", "axios.get", "axios.post", "http.get", "http.request",
        ],
        "console": [
            "console.log", "console.error", "console.warn", "console.info",
        ],
        "subprocess": [
            "child_process.exec", "child_process.spawn", "child_process.execSync",
        ],
    },
    "typescript": {},  # Same as javascript, will fall back
    "php": {
        "io_file": [
            "file_get_contents", "file_put_contents", "fopen", "fwrite", "fread",
        ],
        "io_network": [
            "curl_exec", "file_get_contents",  # when used with URL
        ],
        "console": [
            "echo", "print", "var_dump", "print_r",
        ],
    },
    "rust": {
        "io_file": [
            "File::open", "File::create", "fs::read", "fs::write",
        ],
        "io_network": [
            "TcpStream::connect", "reqwest::get",
        ],
        "console": [
            "println!", "eprintln!", "print!", "dbg!",
        ],
        "subprocess": [
            "Command::new", "process::Command",
        ],
    },
}
```

**Step 2: Verify syntax**

Run: `python -c "from magaldi_core.tree_sitter_manager import IMPURE_CALLS; print(len(IMPURE_CALLS['python']['io_file']))"`
Expected: `8` or similar number

**Step 3: Commit**

```bash
git add src/magaldi_core/tree_sitter_manager.py
git commit -m "feat: add impure call patterns for purity analysis"
```

---

### Task 3.2: Implement Purity Analysis

**Files:**
- Modify: `src/magaldi_core/tree_sitter_manager.py`
- Modify: `tests/test_code_intelligence.py`

**Step 1: Write the test**

Add to `tests/test_code_intelligence.py`:

```python
from magaldi_core.tree_sitter_manager import (
    PurityInfo,
    SideEffect,
    analyze_purity,
    ExtractedCall,
)


class TestPurityAnalysis:
    """Tests for function purity analysis."""

    def test_pure_function(self):
        calls = []
        mutations = []
        purity = analyze_purity(calls, mutations, "python")
        assert purity.level == "pure"
        assert purity.confidence == "high"

    def test_console_impure(self):
        calls = [ExtractedCall(name="print", receiver=None, line=5)]
        mutations = []
        purity = analyze_purity(calls, mutations, "python")
        assert purity.level == "mutates_external"
        assert "print" in purity.reasons[0]

    def test_self_mutation(self):
        calls = []
        mutations = ["self.cache"]
        purity = analyze_purity(calls, mutations, "python")
        assert purity.level == "mutates_self"

    def test_file_io_impure(self):
        calls = [ExtractedCall(name="open", receiver=None, line=10)]
        mutations = []
        purity = analyze_purity(calls, mutations, "python")
        assert purity.level == "mutates_external"
        assert "io_file" in str(purity.reasons)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_code_intelligence.py::TestPurityAnalysis -v`
Expected: FAIL with import error

**Step 3: Implement analyze_purity**

Add to `src/magaldi_core/tree_sitter_manager.py`:

```python
def analyze_purity(
    calls: list[ExtractedCall],
    mutations: list[str],
    language: str,
) -> PurityInfo:
    """Analyze function purity based on calls and mutations.

    Args:
        calls: List of function calls within the function.
        mutations: List of mutated attributes (e.g., ["self.cache"]).
        language: Programming language.

    Returns:
        PurityInfo with level, confidence, and reasons.
    """
    reasons = []
    level = "pure"

    # Get impure patterns for this language (fall back to python for unknown)
    lang_patterns = IMPURE_CALLS.get(language, IMPURE_CALLS.get("python", {}))

    # If language not found and no fallback, use python patterns
    if not lang_patterns and language == "typescript":
        lang_patterns = IMPURE_CALLS.get("javascript", {})

    # Check for impure calls
    for call in calls:
        call_name = call.name
        if call.receiver:
            call_name = f"{call.receiver}.{call.name}"

        for effect_kind, patterns in lang_patterns.items():
            for pattern in patterns:
                if call_name == pattern or call.name == pattern:
                    reasons.append(f"calls {call_name} ({effect_kind})")
                    level = "mutates_external"
                    break

    # Check for self mutations
    for mutation in mutations:
        if mutation.startswith("self.") or mutation.startswith("this."):
            reasons.append(f"modifies {mutation}")
            if level == "pure":
                level = "mutates_self"

    # Determine confidence
    if level == "pure" and not calls:
        confidence = "high"
    elif level == "pure" and calls:
        # Has calls but none matched impure patterns - lower confidence
        confidence = "medium"
    else:
        confidence = "high"

    return PurityInfo(level=level, confidence=confidence, reasons=reasons)


def extract_side_effects(
    calls: list[ExtractedCall],
    mutations: list[str],
    language: str,
) -> list[SideEffect]:
    """Extract individual side effects from calls and mutations.

    Args:
        calls: List of function calls.
        mutations: List of mutations.
        language: Programming language.

    Returns:
        List of SideEffect objects.
    """
    effects = []
    lang_patterns = IMPURE_CALLS.get(language, IMPURE_CALLS.get("python", {}))

    if not lang_patterns and language == "typescript":
        lang_patterns = IMPURE_CALLS.get("javascript", {})

    for call in calls:
        call_name = call.name
        if call.receiver:
            call_name = f"{call.receiver}.{call.name}"

        for effect_kind, patterns in lang_patterns.items():
            for pattern in patterns:
                if call_name == pattern or call.name == pattern:
                    effects.append(
                        SideEffect(kind=effect_kind, target=call_name, line=call.line)
                    )
                    break

    for mutation in mutations:
        effects.append(
            SideEffect(kind="state_mutation", target=mutation, line=0)
        )

    return effects
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_code_intelligence.py::TestPurityAnalysis -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add src/magaldi_core/tree_sitter_manager.py tests/test_code_intelligence.py
git commit -m "feat: implement purity analysis"
```

---

## Phase 4: Type Flow Extraction

### Task 4.1: Implement Python Type Annotation Extraction

**Files:**
- Modify: `src/magaldi_core/tree_sitter_manager.py`
- Modify: `tests/test_code_intelligence.py`

**Step 1: Write the test**

Add to `tests/test_code_intelligence.py`:

```python
from magaldi_core.tree_sitter_manager import (
    TypeAnnotation,
    extract_type_annotations,
    get_manager,
)


class TestTypeAnnotationExtraction:
    """Tests for type annotation extraction."""

    def test_extract_parameter_types(self):
        source = "def foo(x: int, y: str) -> bool: pass"
        manager = get_manager()
        tree = manager.parse(source.encode(), "python")
        annotations = extract_type_annotations(tree.root_node, "python")

        param_types = [a for a in annotations if a.kind == "parameter"]
        assert len(param_types) == 2
        assert any(a.name == "int" and a.location == "param:x" for a in param_types)

    def test_extract_return_type(self):
        source = "def foo() -> str: pass"
        manager = get_manager()
        tree = manager.parse(source.encode(), "python")
        annotations = extract_type_annotations(tree.root_node, "python")

        return_types = [a for a in annotations if a.kind == "return"]
        assert len(return_types) == 1
        assert return_types[0].name == "str"

    def test_extract_generic_types(self):
        source = "def foo(items: List[str]) -> Dict[str, int]: pass"
        manager = get_manager()
        tree = manager.parse(source.encode(), "python")
        annotations = extract_type_annotations(tree.root_node, "python")

        list_type = next((a for a in annotations if "List" in a.name), None)
        assert list_type is not None
        assert list_type.generic_args == ["str"]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_code_intelligence.py::TestTypeAnnotationExtraction -v`
Expected: FAIL with import error

**Step 3: Implement extract_type_annotations**

Add to `src/magaldi_core/tree_sitter_manager.py`:

```python
def _parse_generic_type(type_str: str) -> tuple[str, list[str] | None]:
    """Parse a generic type like List[str] into base and args.

    Args:
        type_str: Type string like "List[str]" or "Dict[str, int]"

    Returns:
        Tuple of (full_type, generic_args or None)
    """
    if "[" not in type_str:
        return type_str, None

    bracket_pos = type_str.index("[")
    # Extract the content between brackets
    inner = type_str[bracket_pos + 1 : -1]  # Remove outer brackets
    # Simple split by comma (doesn't handle nested generics perfectly)
    args = [arg.strip() for arg in inner.split(",")]
    return type_str, args


def extract_type_annotations(node: Node, language: str) -> list[TypeAnnotation]:
    """Extract type annotations from an AST node.

    Args:
        node: Root AST node to search.
        language: Programming language.

    Returns:
        List of TypeAnnotation objects.
    """
    annotations = []

    if language == "python":
        annotations.extend(_extract_python_type_annotations(node))
    elif language in ("typescript", "javascript"):
        annotations.extend(_extract_typescript_type_annotations(node))

    return annotations


def _extract_python_type_annotations(node: Node) -> list[TypeAnnotation]:
    """Extract type annotations from Python AST."""
    annotations = []

    for func_node in find_nodes(node, "function_definition"):
        func_line = func_node.start_point[0] + 1

        # Get parameters
        params = get_child_by_field(func_node, "parameters")
        if params:
            for param in params.children:
                if param.type == "typed_parameter":
                    name_node = param.children[0] if param.children else None
                    type_node = get_child_by_field(param, "type")

                    if name_node and type_node:
                        param_name = get_node_text(name_node)
                        type_str = get_node_text(type_node)
                        full_type, generic_args = _parse_generic_type(type_str)

                        annotations.append(
                            TypeAnnotation(
                                name=full_type,
                                kind="parameter",
                                location=f"param:{param_name}",
                                line=func_line,
                                generic_args=generic_args,
                            )
                        )

        # Get return type
        return_type = get_child_by_field(func_node, "return_type")
        if return_type:
            type_str = get_node_text(return_type)
            full_type, generic_args = _parse_generic_type(type_str)

            annotations.append(
                TypeAnnotation(
                    name=full_type,
                    kind="return",
                    location="return",
                    line=func_line,
                    generic_args=generic_args,
                )
            )

    return annotations


def _extract_typescript_type_annotations(node: Node) -> list[TypeAnnotation]:
    """Extract type annotations from TypeScript AST."""
    annotations = []

    # Find function declarations and arrow functions
    for func_type in ("function_declaration", "arrow_function", "method_definition"):
        for func_node in find_nodes(node, func_type):
            func_line = func_node.start_point[0] + 1

            # Look for type annotations in parameters
            params = get_child_by_field(func_node, "parameters")
            if params:
                for param in params.children:
                    if param.type == "required_parameter" or param.type == "optional_parameter":
                        # Look for type_annotation child
                        for child in param.children:
                            if child.type == "type_annotation":
                                type_node = child.children[-1] if child.children else None
                                name_node = param.children[0] if param.children else None

                                if type_node and name_node:
                                    param_name = get_node_text(name_node)
                                    type_str = get_node_text(type_node)
                                    full_type, generic_args = _parse_generic_type(type_str)

                                    annotations.append(
                                        TypeAnnotation(
                                            name=full_type,
                                            kind="parameter",
                                            location=f"param:{param_name}",
                                            line=func_line,
                                            generic_args=generic_args,
                                        )
                                    )

            # Look for return type annotation
            for child in func_node.children:
                if child.type == "type_annotation":
                    type_node = child.children[-1] if child.children else None
                    if type_node:
                        type_str = get_node_text(type_node)
                        full_type, generic_args = _parse_generic_type(type_str)

                        annotations.append(
                            TypeAnnotation(
                                name=full_type,
                                kind="return",
                                location="return",
                                line=func_line,
                                generic_args=generic_args,
                            )
                        )

    return annotations
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_code_intelligence.py::TestTypeAnnotationExtraction -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add src/magaldi_core/tree_sitter_manager.py tests/test_code_intelligence.py
git commit -m "feat: implement type annotation extraction"
```

---

## Phase 5: API Surface Detection

### Task 5.1: Implement HTTP Route Detection

**Files:**
- Modify: `src/magaldi_core/tree_sitter_manager.py`
- Modify: `tests/test_code_intelligence.py`

**Step 1: Write the test**

Add to `tests/test_code_intelligence.py`:

```python
from magaldi_core.tree_sitter_manager import (
    HttpRoute,
    detect_http_routes,
    DecoratorInfo,
)


class TestHttpRouteDetection:
    """Tests for HTTP route detection."""

    def test_detect_fastapi_route(self):
        decorators = [
            DecoratorInfo(name="router.get", args='"/users/{id}"', full='router.get("/users/{id}")')
        ]
        routes = detect_http_routes(decorators, "python")
        assert len(routes) == 1
        assert routes[0].method == "GET"
        assert routes[0].path == "/users/{id}"
        assert routes[0].path_params == ["id"]
        assert routes[0].framework == "fastapi"

    def test_detect_flask_route(self):
        decorators = [
            DecoratorInfo(name="app.route", args='"/api/items", methods=["POST"]', full='app.route("/api/items", methods=["POST"])')
        ]
        routes = detect_http_routes(decorators, "python")
        assert len(routes) == 1
        assert routes[0].method == "POST"
        assert routes[0].path == "/api/items"

    def test_no_route_decorators(self):
        decorators = [
            DecoratorInfo(name="staticmethod", args=None, full="staticmethod")
        ]
        routes = detect_http_routes(decorators, "python")
        assert len(routes) == 0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_code_intelligence.py::TestHttpRouteDetection -v`
Expected: FAIL with import error

**Step 3: Implement detect_http_routes**

Add to `src/magaldi_core/tree_sitter_manager.py`:

```python
# HTTP route decorator patterns
_HTTP_ROUTE_PATTERNS: dict[str, dict[str, str]] = {
    # FastAPI patterns
    "router.get": ("GET", "fastapi"),
    "router.post": ("POST", "fastapi"),
    "router.put": ("PUT", "fastapi"),
    "router.delete": ("DELETE", "fastapi"),
    "router.patch": ("PATCH", "fastapi"),
    "app.get": ("GET", "fastapi"),
    "app.post": ("POST", "fastapi"),
    "app.put": ("PUT", "fastapi"),
    "app.delete": ("DELETE", "fastapi"),
    # Flask patterns
    "app.route": ("*", "flask"),
    "blueprint.route": ("*", "flask"),
}


def _extract_path_params(path: str) -> list[str]:
    """Extract path parameters from a route path.

    Args:
        path: Route path like "/users/{id}" or "/items/<item_id>"

    Returns:
        List of parameter names.
    """
    params = []
    # FastAPI/OpenAPI style: {param}
    import re
    for match in re.finditer(r"\{(\w+)\}", path):
        params.append(match.group(1))
    # Flask style: <param>
    for match in re.finditer(r"<(\w+)>", path):
        params.append(match.group(1))
    return params


def _extract_path_from_args(args: str | None) -> str | None:
    """Extract path from decorator arguments.

    Args:
        args: Decorator arguments string like '"/users/{id}"'

    Returns:
        Extracted path or None.
    """
    if not args:
        return None

    # Look for quoted string at the start
    import re
    match = re.match(r'["\']([^"\']+)["\']', args.strip())
    if match:
        return match.group(1)
    return None


def _extract_method_from_flask_args(args: str | None) -> str:
    """Extract HTTP method from Flask route arguments.

    Args:
        args: Flask route arguments like '"/path", methods=["POST"]'

    Returns:
        HTTP method or "GET" as default.
    """
    if not args:
        return "GET"

    import re
    match = re.search(r'methods\s*=\s*\[([^\]]+)\]', args)
    if match:
        methods_str = match.group(1)
        # Extract first method
        method_match = re.search(r'["\'](\w+)["\']', methods_str)
        if method_match:
            return method_match.group(1).upper()

    return "GET"


def detect_http_routes(
    decorators: list[DecoratorInfo],
    language: str,
) -> list[HttpRoute]:
    """Detect HTTP routes from decorator information.

    Args:
        decorators: List of decorator info from the element.
        language: Programming language.

    Returns:
        List of detected HTTP routes.
    """
    routes = []

    for dec in decorators:
        if dec.name in _HTTP_ROUTE_PATTERNS:
            method, framework = _HTTP_ROUTE_PATTERNS[dec.name]
            path = _extract_path_from_args(dec.args)

            if path:
                # For Flask, method comes from args
                if method == "*":
                    method = _extract_method_from_flask_args(dec.args)

                routes.append(
                    HttpRoute(
                        method=method,
                        path=path,
                        path_params=_extract_path_params(path),
                        framework=framework,
                    )
                )

    return routes
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_code_intelligence.py::TestHttpRouteDetection -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add src/magaldi_core/tree_sitter_manager.py tests/test_code_intelligence.py
git commit -m "feat: implement HTTP route detection"
```

---

### Task 5.2: Implement CLI Command Detection

**Files:**
- Modify: `src/magaldi_core/tree_sitter_manager.py`
- Modify: `tests/test_code_intelligence.py`

**Step 1: Write the test**

Add to `tests/test_code_intelligence.py`:

```python
from magaldi_core.tree_sitter_manager import (
    CliCommand,
    detect_cli_commands,
)


class TestCliCommandDetection:
    """Tests for CLI command detection."""

    def test_detect_click_command(self):
        decorators = [
            DecoratorInfo(name="click.command", args=None, full="click.command()")
        ]
        commands = detect_cli_commands(decorators, "parse", "python")
        assert len(commands) == 1
        assert commands[0].name == "parse"
        assert commands[0].framework == "click"

    def test_detect_typer_command(self):
        decorators = [
            DecoratorInfo(name="app.command", args=None, full="app.command()")
        ]
        commands = detect_cli_commands(decorators, "run", "python")
        assert len(commands) == 1
        assert commands[0].framework == "typer"

    def test_no_cli_decorators(self):
        decorators = [
            DecoratorInfo(name="property", args=None, full="property")
        ]
        commands = detect_cli_commands(decorators, "getter", "python")
        assert len(commands) == 0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_code_intelligence.py::TestCliCommandDetection -v`
Expected: FAIL with import error

**Step 3: Implement detect_cli_commands**

Add to `src/magaldi_core/tree_sitter_manager.py`:

```python
# CLI command decorator patterns
_CLI_COMMAND_PATTERNS: dict[str, str] = {
    "click.command": "click",
    "click.group": "click",
    "app.command": "typer",
    "typer.command": "typer",
}


def detect_cli_commands(
    decorators: list[DecoratorInfo],
    function_name: str,
    language: str,
) -> list[CliCommand]:
    """Detect CLI commands from decorator information.

    Args:
        decorators: List of decorator info from the element.
        function_name: Name of the decorated function.
        language: Programming language.

    Returns:
        List of detected CLI commands.
    """
    commands = []

    for dec in decorators:
        if dec.name in _CLI_COMMAND_PATTERNS:
            framework = _CLI_COMMAND_PATTERNS[dec.name]

            # Extract options from sibling decorators
            options = []
            for other_dec in decorators:
                if other_dec.name in ("click.option", "click.argument", "typer.Option", "typer.Argument"):
                    options.append({
                        "name": other_dec.args.split(",")[0].strip('"\'') if other_dec.args else "",
                        "type": None,
                        "required": "required=True" in (other_dec.full or ""),
                    })

            commands.append(
                CliCommand(
                    name=function_name,
                    options=options,
                    framework=framework,
                )
            )

    return commands
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_code_intelligence.py::TestCliCommandDetection -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add src/magaldi_core/tree_sitter_manager.py tests/test_code_intelligence.py
git commit -m "feat: implement CLI command detection"
```

---

### Task 5.3: Implement Public API Detection

**Files:**
- Modify: `src/magaldi_core/tree_sitter_manager.py`
- Modify: `tests/test_code_intelligence.py`

**Step 1: Write the test**

Add to `tests/test_code_intelligence.py`:

```python
from magaldi_core.tree_sitter_manager import detect_public_api


class TestPublicApiDetection:
    """Tests for public API detection."""

    def test_public_function(self):
        assert detect_public_api("process_data", [], "public", "python") is True

    def test_private_function(self):
        assert detect_public_api("_helper", [], "private", "python") is False

    def test_dunder_method(self):
        assert detect_public_api("__init__", [], "public", "python") is False

    def test_api_decorator(self):
        decorators = [DecoratorInfo(name="api_endpoint", args=None, full="api_endpoint")]
        assert detect_public_api("handler", decorators, "public", "python") is True

    def test_route_is_public_api(self):
        decorators = [DecoratorInfo(name="router.get", args='"/users"', full='router.get("/users")')]
        assert detect_public_api("get_users", decorators, "public", "python") is True
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_code_intelligence.py::TestPublicApiDetection -v`
Expected: FAIL with import error

**Step 3: Implement detect_public_api**

Add to `src/magaldi_core/tree_sitter_manager.py`:

```python
# Decorators that indicate public API
_PUBLIC_API_DECORATORS = {
    "api_endpoint", "public", "export", "exposed",
    "router.get", "router.post", "router.put", "router.delete", "router.patch",
    "app.get", "app.post", "app.put", "app.delete",
    "app.route", "blueprint.route",
    "click.command", "click.group", "app.command",
}


def detect_public_api(
    name: str,
    decorators: list[DecoratorInfo],
    visibility: str,
    language: str,
) -> bool:
    """Detect if an element is a public API.

    Args:
        name: Element name.
        decorators: List of decorators.
        visibility: Visibility level ("public", "private", "protected").
        language: Programming language.

    Returns:
        True if the element is a public API.
    """
    # Private/protected are not public API
    if visibility != "public":
        return False

    # Dunder methods are not public API (except __init__ in some cases)
    if name.startswith("__") and name.endswith("__"):
        return False

    # Check for public API decorators
    for dec in decorators:
        if dec.name in _PUBLIC_API_DECORATORS:
            return True

    # Default: public visibility and not private naming = public API
    return True
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_code_intelligence.py::TestPublicApiDetection -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add src/magaldi_core/tree_sitter_manager.py tests/test_code_intelligence.py
git commit -m "feat: implement public API detection"
```

---

## Phase 6: Pattern Detection

### Task 6.1: Implement Singleton Pattern Detection

**Files:**
- Modify: `src/magaldi_core/tree_sitter_manager.py`
- Modify: `tests/test_code_intelligence.py`

**Step 1: Write the test**

Add to `tests/test_code_intelligence.py`:

```python
from magaldi_core.tree_sitter_manager import detect_patterns


class TestPatternDetection:
    """Tests for design pattern detection."""

    def test_detect_singleton(self):
        class_info = {
            "name": "DatabaseConnection",
            "attributes": ["_instance"],
            "methods": ["get_instance", "__new__"],
            "method_returns_self": True,
        }
        patterns, confidence = detect_patterns(class_info, [], "python")
        assert "singleton" in patterns
        assert confidence.get("singleton", 0) >= 0.6

    def test_detect_builder(self):
        class_info = {
            "name": "QueryBuilder",
            "attributes": ["_query"],
            "methods": ["select", "where", "order_by", "build"],
            "methods_return_self": ["select", "where", "order_by"],
        }
        patterns, confidence = detect_patterns(class_info, [], "python")
        assert "builder" in patterns

    def test_no_pattern(self):
        class_info = {
            "name": "SimpleClass",
            "attributes": ["value"],
            "methods": ["get_value", "set_value"],
        }
        patterns, confidence = detect_patterns(class_info, [], "python")
        assert len(patterns) == 0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_code_intelligence.py::TestPatternDetection -v`
Expected: FAIL with import error

**Step 3: Implement detect_patterns**

Add to `src/magaldi_core/tree_sitter_manager.py`:

```python
def detect_patterns(
    class_info: dict[str, Any],
    calls: list[ExtractedCall],
    language: str,
) -> tuple[list[str], dict[str, float]]:
    """Detect design patterns in a class.

    Args:
        class_info: Dict with class name, attributes, methods, etc.
        calls: Calls made within the class methods.
        language: Programming language.

    Returns:
        Tuple of (detected pattern names, confidence scores).
    """
    patterns = []
    confidence = {}

    # Singleton detection
    singleton_score = _detect_singleton(class_info)
    if singleton_score >= 0.6:
        patterns.append("singleton")
        confidence["singleton"] = singleton_score

    # Builder detection
    builder_score = _detect_builder(class_info)
    if builder_score >= 0.6:
        patterns.append("builder")
        confidence["builder"] = builder_score

    # Factory detection
    factory_score = _detect_factory(class_info, calls)
    if factory_score >= 0.6:
        patterns.append("factory")
        confidence["factory"] = factory_score

    # Repository detection
    repository_score = _detect_repository(class_info)
    if repository_score >= 0.6:
        patterns.append("repository")
        confidence["repository"] = repository_score

    return patterns, confidence


def _detect_singleton(class_info: dict[str, Any]) -> float:
    """Detect singleton pattern."""
    score = 0.0
    attributes = class_info.get("attributes", [])
    methods = class_info.get("methods", [])

    # Has _instance attribute
    if "_instance" in attributes or "instance" in attributes:
        score += 0.3

    # Has get_instance method
    if "get_instance" in methods or "getInstance" in methods:
        score += 0.3

    # Has __new__ method (Python singleton pattern)
    if "__new__" in methods:
        score += 0.2

    # Returns self/instance from get_instance
    if class_info.get("method_returns_self"):
        score += 0.2

    return score


def _detect_builder(class_info: dict[str, Any]) -> float:
    """Detect builder pattern."""
    score = 0.0
    methods = class_info.get("methods", [])
    returns_self = class_info.get("methods_return_self", [])

    # Multiple methods that return self (method chaining)
    if len(returns_self) >= 2:
        score += 0.4

    # Has a build() method
    if "build" in methods:
        score += 0.3

    # Name ends with Builder
    if class_info.get("name", "").endswith("Builder"):
        score += 0.3

    return score


def _detect_factory(class_info: dict[str, Any], calls: list[ExtractedCall]) -> float:
    """Detect factory pattern."""
    score = 0.0
    methods = class_info.get("methods", [])
    name = class_info.get("name", "")

    # Name contains Factory
    if "Factory" in name or "factory" in name.lower():
        score += 0.3

    # Has create* methods
    create_methods = [m for m in methods if m.startswith("create") or m.startswith("make")]
    if create_methods:
        score += 0.3

    # Methods instantiate other classes
    instantiation_calls = [c for c in calls if c.receiver is None and c.name[0].isupper()]
    if instantiation_calls:
        score += 0.4

    return score


def _detect_repository(class_info: dict[str, Any]) -> float:
    """Detect repository pattern."""
    score = 0.0
    methods = class_info.get("methods", [])
    name = class_info.get("name", "")

    # Name contains Repository
    if "Repository" in name or "Repo" in name:
        score += 0.3

    # Has CRUD-like methods
    crud_methods = {"get", "find", "save", "update", "delete", "create", "add", "remove"}
    found_crud = [m for m in methods if any(crud in m.lower() for crud in crud_methods)]
    if len(found_crud) >= 2:
        score += 0.4

    # Has find_by_* methods
    find_by_methods = [m for m in methods if m.startswith("find_by") or m.startswith("get_by")]
    if find_by_methods:
        score += 0.3

    return score
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_code_intelligence.py::TestPatternDetection -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add src/magaldi_core/tree_sitter_manager.py tests/test_code_intelligence.py
git commit -m "feat: implement design pattern detection"
```

---

## Phase 7: Integrate Extractors into Parsing Pipeline

### Task 7.1: Update PythonParser to Use New Extractors

**Files:**
- Modify: `src/magaldi_core/code_parser.py`

**Step 1: Import new functions**

Add to the imports at the top of `code_parser.py`:

```python
from magaldi_core.tree_sitter_manager import (
    # ... existing imports ...
    extract_todos,
    extract_section_markers,
    extract_comments,
    associate_comments,
    extract_type_annotations,
    analyze_purity,
    extract_side_effects,
    detect_http_routes,
    detect_cli_commands,
    detect_public_api,
    detect_patterns,
)
```

**Step 2: Update PythonParser.parse() method**

In `PythonParser.parse()`, after extracting elements and before `_set_hierarchy`, add:

```python
        # Extract file-level documentation
        todos = extract_todos(content)
        section_markers = extract_section_markers(content)
        all_comments = extract_comments(content)

        # Populate file element with documentation
        file_element.todos = [
            {"kind": t.kind, "text": t.text, "line": t.line,
             "assignee": t.assignee, "priority": t.priority, "issue_ref": t.issue_ref}
            for t in todos
        ]
        file_element.section_markers = [
            {"label": m.label, "line": m.line, "style": m.style}
            for m in section_markers
        ]

        # Process each element for extended intelligence
        for elem in elements:
            if elem.element_type == "file":
                continue

            # Associate comments
            assoc_comments = associate_comments(elem.line_start, all_comments)
            elem.associated_comments = [
                {"text": c.text, "line": c.line, "kind": c.kind, "position": c.position}
                for c in assoc_comments
            ]

            # Type annotations (for functions/methods)
            if elem.element_type in ("function", "method") and elem.raw_code:
                # Re-parse element code for type annotations
                elem_tree = self.manager.parse(elem.raw_code.encode("utf-8"), "python")
                type_annots = extract_type_annotations(elem_tree.root_node, "python")
                elem.type_annotations = [
                    {"name": a.name, "kind": a.kind, "location": a.location,
                     "line": a.line, "generic_args": a.generic_args}
                    for a in type_annots
                ]

            # Purity analysis (for functions/methods)
            if elem.element_type in ("function", "method"):
                mutations = elem.attributes_modified or []
                purity = analyze_purity(elem.calls, mutations, "python")
                elem.purity = {
                    "level": purity.level,
                    "confidence": purity.confidence,
                    "reasons": purity.reasons,
                }
                effects = extract_side_effects(elem.calls, mutations, "python")
                elem.side_effects = [
                    {"kind": e.kind, "target": e.target, "line": e.line}
                    for e in effects
                ]
                elem.mutated_state = mutations

            # API surface detection
            if elem.decorator_details:
                dec_infos = [
                    DecoratorInfo(name=d["name"], args=d.get("args"), full=d.get("full"))
                    for d in elem.decorator_details
                ]
                routes = detect_http_routes(dec_infos, "python")
                elem.http_routes = [
                    {"method": r.method, "path": r.path,
                     "path_params": r.path_params, "framework": r.framework}
                    for r in routes
                ]
                commands = detect_cli_commands(dec_infos, elem.name, "python")
                elem.cli_commands = [
                    {"name": c.name, "options": c.options, "framework": c.framework}
                    for c in commands
                ]
                elem.is_public_api = detect_public_api(
                    elem.name, dec_infos, elem.visibility, "python"
                )
            else:
                elem.is_public_api = detect_public_api(
                    elem.name, [], elem.visibility, "python"
                )

            # Pattern detection (for classes)
            if elem.element_type == "class":
                class_info = {
                    "name": elem.name,
                    "attributes": [a["name"] for a in (elem.class_attributes or [])],
                    "methods": [],  # Would need to collect from child elements
                }
                patterns, confidence = detect_patterns(class_info, [], "python")
                elem.detected_patterns = patterns
                elem.pattern_confidence = confidence
```

**Step 3: Verify no syntax errors**

Run: `python -c "from magaldi_core.code_parser import PythonParser"`
Expected: No output (success)

**Step 4: Run existing parser tests**

Run: `pytest tests/test_code_parser.py -v -x`
Expected: All tests PASS (no regressions)

**Step 5: Commit**

```bash
git add src/magaldi_core/code_parser.py
git commit -m "feat: integrate code intelligence extractors into Python parser"
```

---

## Phase 8: Add MCP Tools

### Task 8.1: Add TODO/Documentation MCP Tools

**Files:**
- Modify: `src/magaldi_mcp/tools.py`

**Step 1: Add find_todos tool**

Add to `tools.py`:

```python
def find_todos(
    es: ElasticsearchRepository,
    scope: str | None = None,
    repository: str | None = None,
    username: str = "main",
    kind: str | None = None,
    assignee: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Find TODO/FIXME comments in the codebase.

    Args:
        es: Elasticsearch repository.
        scope: Filter by scope.
        repository: Filter by repository.
        username: User branch.
        kind: Filter by kind (TODO, FIXME, HACK, etc.).
        assignee: Filter by assignee.
        limit: Maximum results.

    Returns:
        List of TODO items with file and line info.
    """
    # Build query for nested todos
    must_clauses = []
    if scope:
        must_clauses.append({"term": {"scope": scope}})
    if repository:
        must_clauses.append({"term": {"repository": repository}})
    must_clauses.append({"term": {"username": username}})

    # Nested query for todos
    nested_must = []
    if kind:
        nested_must.append({"term": {"todos.kind": kind.upper()}})
    if assignee:
        nested_must.append({"term": {"todos.assignee": assignee}})

    query = {
        "bool": {
            "must": must_clauses,
            "filter": {
                "nested": {
                    "path": "todos",
                    "query": {
                        "bool": {
                            "must": nested_must if nested_must else [{"exists": {"field": "todos.kind"}}]
                        }
                    },
                }
            }
        }
    }

    client = es._get_client()
    result = client.search(
        index=es.INDEX_NAME if hasattr(es, 'INDEX_NAME') else "magaldi-code-elements",
        body={
            "query": query,
            "size": limit,
            "_source": ["relative_path", "todos", "name"],
        },
    )

    todos = []
    for hit in result.get("hits", {}).get("hits", []):
        source = hit["_source"]
        file_path = source.get("relative_path", "")
        for todo in source.get("todos", []):
            if kind and todo.get("kind") != kind.upper():
                continue
            if assignee and todo.get("assignee") != assignee:
                continue
            todos.append({
                "file": file_path,
                "line": todo.get("line"),
                "kind": todo.get("kind"),
                "text": todo.get("text"),
                "assignee": todo.get("assignee"),
                "priority": todo.get("priority"),
                "issue_ref": todo.get("issue_ref"),
            })

    return todos[:limit]
```

**Step 2: Verify no syntax errors**

Run: `python -c "from magaldi_mcp.tools import find_todos"`
Expected: No output (success)

**Step 3: Commit**

```bash
git add src/magaldi_mcp/tools.py
git commit -m "feat: add find_todos MCP tool"
```

---

### Task 8.2: Add Purity Analysis MCP Tools

**Files:**
- Modify: `src/magaldi_mcp/tools.py`

**Step 1: Add purity tools**

Add to `tools.py`:

```python
def get_purity(
    es: ElasticsearchRepository,
    element_id: str,
) -> dict[str, Any]:
    """Get purity analysis for a function/method.

    Args:
        es: Elasticsearch repository.
        element_id: Element ID.

    Returns:
        Purity info dict.
    """
    doc = es.get_document_by_id_or_hash(element_id)
    if not doc:
        raise ValueError(f"Element not found: {element_id}")

    return {
        "element_id": doc.get("element_id"),
        "name": doc.get("name"),
        "purity": doc.get("purity"),
        "side_effects": doc.get("side_effects", []),
        "mutated_state": doc.get("mutated_state", []),
    }


def find_pure_functions(
    es: ElasticsearchRepository,
    scope: str | None = None,
    repository: str | None = None,
    username: str = "main",
    confidence: str = "high",
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Find pure functions in the codebase.

    Args:
        es: Elasticsearch repository.
        scope: Filter by scope.
        repository: Filter by repository.
        username: User branch.
        confidence: Minimum confidence level.
        limit: Maximum results.

    Returns:
        List of pure functions.
    """
    must_clauses = [
        {"term": {"purity.level": "pure"}},
        {"term": {"username": username}},
    ]
    if scope:
        must_clauses.append({"term": {"scope": scope}})
    if repository:
        must_clauses.append({"term": {"repository": repository}})
    if confidence:
        must_clauses.append({"term": {"purity.confidence": confidence}})

    client = es._get_client()
    result = client.search(
        index="magaldi-code-elements",
        body={
            "query": {"bool": {"must": must_clauses}},
            "size": limit,
            "_source": ["element_id", "name", "relative_path", "line_start", "purity"],
        },
    )

    return [
        {
            "element_id": hit["_source"].get("element_id"),
            "name": hit["_source"].get("name"),
            "file": hit["_source"].get("relative_path"),
            "line": hit["_source"].get("line_start"),
            "confidence": hit["_source"].get("purity", {}).get("confidence"),
        }
        for hit in result.get("hits", {}).get("hits", [])
    ]


def find_side_effects(
    es: ElasticsearchRepository,
    effect_kind: str | None = None,
    scope: str | None = None,
    repository: str | None = None,
    username: str = "main",
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Find functions with specific side effects.

    Args:
        es: Elasticsearch repository.
        effect_kind: Type of side effect (io_file, io_network, console, etc.).
        scope: Filter by scope.
        repository: Filter by repository.
        username: User branch.
        limit: Maximum results.

    Returns:
        List of functions with side effects.
    """
    must_clauses = [{"term": {"username": username}}]
    if scope:
        must_clauses.append({"term": {"scope": scope}})
    if repository:
        must_clauses.append({"term": {"repository": repository}})

    nested_filter = {"exists": {"field": "side_effects.kind"}}
    if effect_kind:
        nested_filter = {"term": {"side_effects.kind": effect_kind}}

    must_clauses.append({
        "nested": {
            "path": "side_effects",
            "query": nested_filter,
        }
    })

    client = es._get_client()
    result = client.search(
        index="magaldi-code-elements",
        body={
            "query": {"bool": {"must": must_clauses}},
            "size": limit,
            "_source": ["element_id", "name", "relative_path", "line_start", "side_effects"],
        },
    )

    return [
        {
            "element_id": hit["_source"].get("element_id"),
            "name": hit["_source"].get("name"),
            "file": hit["_source"].get("relative_path"),
            "line": hit["_source"].get("line_start"),
            "side_effects": hit["_source"].get("side_effects", []),
        }
        for hit in result.get("hits", {}).get("hits", [])
    ]
```

**Step 2: Verify no syntax errors**

Run: `python -c "from magaldi_mcp.tools import get_purity, find_pure_functions, find_side_effects"`
Expected: No output (success)

**Step 3: Commit**

```bash
git add src/magaldi_mcp/tools.py
git commit -m "feat: add purity analysis MCP tools"
```

---

### Task 8.3: Add Type Flow MCP Tools

**Files:**
- Modify: `src/magaldi_mcp/tools.py`

**Step 1: Add type flow tools**

Add to `tools.py`:

```python
def trace_type(
    es: ElasticsearchRepository,
    type_name: str,
    scope: str | None = None,
    repository: str | None = None,
    username: str = "main",
    limit: int = 50,
) -> dict[str, Any]:
    """Track where a type is used in the codebase.

    Args:
        es: Elasticsearch repository.
        type_name: Type name to search for.
        scope: Filter by scope.
        repository: Filter by repository.
        username: User branch.
        limit: Maximum results.

    Returns:
        Dict with producers, consumers, and other usages.
    """
    must_clauses = [{"term": {"username": username}}]
    if scope:
        must_clauses.append({"term": {"scope": scope}})
    if repository:
        must_clauses.append({"term": {"repository": repository}})

    must_clauses.append({
        "nested": {
            "path": "type_annotations",
            "query": {"term": {"type_annotations.name": type_name}},
        }
    })

    client = es._get_client()
    result = client.search(
        index="magaldi-code-elements",
        body={
            "query": {"bool": {"must": must_clauses}},
            "size": limit,
            "_source": ["element_id", "name", "relative_path", "line_start", "type_annotations"],
        },
    )

    producers = []  # Functions that return this type
    consumers = []  # Functions that take this type as parameter

    for hit in result.get("hits", {}).get("hits", []):
        source = hit["_source"]
        for annot in source.get("type_annotations", []):
            if type_name not in annot.get("name", ""):
                continue

            entry = {
                "element_id": source.get("element_id"),
                "name": source.get("name"),
                "file": source.get("relative_path"),
                "line": source.get("line_start"),
                "location": annot.get("location"),
            }

            if annot.get("kind") == "return":
                producers.append(entry)
            elif annot.get("kind") == "parameter":
                consumers.append(entry)

    return {
        "type": type_name,
        "producers": producers[:limit],
        "consumers": consumers[:limit],
        "total_producers": len(producers),
        "total_consumers": len(consumers),
    }


def find_type_producers(
    es: ElasticsearchRepository,
    type_name: str,
    scope: str | None = None,
    repository: str | None = None,
    username: str = "main",
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Find functions that return a specific type.

    Args:
        es: Elasticsearch repository.
        type_name: Type name to search for.
        scope: Filter by scope.
        repository: Filter by repository.
        username: User branch.
        limit: Maximum results.

    Returns:
        List of functions returning this type.
    """
    result = trace_type(es, type_name, scope, repository, username, limit)
    return result["producers"]


def find_type_consumers(
    es: ElasticsearchRepository,
    type_name: str,
    scope: str | None = None,
    repository: str | None = None,
    username: str = "main",
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Find functions that take a specific type as parameter.

    Args:
        es: Elasticsearch repository.
        type_name: Type name to search for.
        scope: Filter by scope.
        repository: Filter by repository.
        username: User branch.
        limit: Maximum results.

    Returns:
        List of functions consuming this type.
    """
    result = trace_type(es, type_name, scope, repository, username, limit)
    return result["consumers"]
```

**Step 2: Verify no syntax errors**

Run: `python -c "from magaldi_mcp.tools import trace_type, find_type_producers, find_type_consumers"`
Expected: No output (success)

**Step 3: Commit**

```bash
git add src/magaldi_mcp/tools.py
git commit -m "feat: add type flow MCP tools"
```

---

### Task 8.4: Add API Surface MCP Tools

**Files:**
- Modify: `src/magaldi_mcp/tools.py`

**Step 1: Add API surface tools**

Add to `tools.py`:

```python
def find_http_routes(
    es: ElasticsearchRepository,
    method: str | None = None,
    path_pattern: str | None = None,
    framework: str | None = None,
    scope: str | None = None,
    repository: str | None = None,
    username: str = "main",
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Find HTTP route handlers.

    Args:
        es: Elasticsearch repository.
        method: HTTP method filter (GET, POST, etc.).
        path_pattern: Path pattern to match.
        framework: Framework filter (fastapi, flask, etc.).
        scope: Filter by scope.
        repository: Filter by repository.
        username: User branch.
        limit: Maximum results.

    Returns:
        List of HTTP route handlers.
    """
    must_clauses = [{"term": {"username": username}}]
    if scope:
        must_clauses.append({"term": {"scope": scope}})
    if repository:
        must_clauses.append({"term": {"repository": repository}})

    nested_must = [{"exists": {"field": "http_routes.method"}}]
    if method:
        nested_must.append({"term": {"http_routes.method": method.upper()}})
    if framework:
        nested_must.append({"term": {"http_routes.framework": framework}})
    if path_pattern:
        nested_must.append({"wildcard": {"http_routes.path": f"*{path_pattern}*"}})

    must_clauses.append({
        "nested": {
            "path": "http_routes",
            "query": {"bool": {"must": nested_must}},
        }
    })

    client = es._get_client()
    result = client.search(
        index="magaldi-code-elements",
        body={
            "query": {"bool": {"must": must_clauses}},
            "size": limit,
            "_source": ["element_id", "name", "relative_path", "line_start", "http_routes"],
        },
    )

    routes = []
    for hit in result.get("hits", {}).get("hits", []):
        source = hit["_source"]
        for route in source.get("http_routes", []):
            if method and route.get("method") != method.upper():
                continue
            if framework and route.get("framework") != framework:
                continue

            routes.append({
                "element_id": source.get("element_id"),
                "handler": source.get("name"),
                "file": source.get("relative_path"),
                "line": source.get("line_start"),
                "method": route.get("method"),
                "path": route.get("path"),
                "path_params": route.get("path_params", []),
                "framework": route.get("framework"),
            })

    return routes[:limit]


def find_cli_commands(
    es: ElasticsearchRepository,
    command_name: str | None = None,
    framework: str | None = None,
    scope: str | None = None,
    repository: str | None = None,
    username: str = "main",
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Find CLI command handlers.

    Args:
        es: Elasticsearch repository.
        command_name: Command name filter.
        framework: Framework filter (click, typer, etc.).
        scope: Filter by scope.
        repository: Filter by repository.
        username: User branch.
        limit: Maximum results.

    Returns:
        List of CLI command handlers.
    """
    must_clauses = [{"term": {"username": username}}]
    if scope:
        must_clauses.append({"term": {"scope": scope}})
    if repository:
        must_clauses.append({"term": {"repository": repository}})

    nested_must = [{"exists": {"field": "cli_commands.name"}}]
    if command_name:
        nested_must.append({"term": {"cli_commands.name": command_name}})
    if framework:
        nested_must.append({"term": {"cli_commands.framework": framework}})

    must_clauses.append({
        "nested": {
            "path": "cli_commands",
            "query": {"bool": {"must": nested_must}},
        }
    })

    client = es._get_client()
    result = client.search(
        index="magaldi-code-elements",
        body={
            "query": {"bool": {"must": must_clauses}},
            "size": limit,
            "_source": ["element_id", "name", "relative_path", "line_start", "cli_commands"],
        },
    )

    commands = []
    for hit in result.get("hits", {}).get("hits", []):
        source = hit["_source"]
        for cmd in source.get("cli_commands", []):
            commands.append({
                "element_id": source.get("element_id"),
                "handler": source.get("name"),
                "file": source.get("relative_path"),
                "line": source.get("line_start"),
                "command": cmd.get("name"),
                "options": cmd.get("options", []),
                "framework": cmd.get("framework"),
            })

    return commands[:limit]


def get_public_api(
    es: ElasticsearchRepository,
    scope: str | None = None,
    repository: str | None = None,
    username: str = "main",
    limit: int = 100,
) -> dict[str, Any]:
    """Get public API surface for a repository.

    Args:
        es: Elasticsearch repository.
        scope: Filter by scope.
        repository: Filter by repository.
        username: User branch.
        limit: Maximum results per category.

    Returns:
        Dict with http_routes, cli_commands, and public_functions.
    """
    return {
        "http_routes": find_http_routes(es, scope=scope, repository=repository, username=username, limit=limit),
        "cli_commands": find_cli_commands(es, scope=scope, repository=repository, username=username, limit=limit),
    }
```

**Step 2: Verify no syntax errors**

Run: `python -c "from magaldi_mcp.tools import find_http_routes, find_cli_commands, get_public_api"`
Expected: No output (success)

**Step 3: Commit**

```bash
git add src/magaldi_mcp/tools.py
git commit -m "feat: add API surface MCP tools"
```

---

### Task 8.5: Add Pattern Detection MCP Tools

**Files:**
- Modify: `src/magaldi_mcp/tools.py`

**Step 1: Add pattern detection tools**

Add to `tools.py`:

```python
def find_patterns(
    es: ElasticsearchRepository,
    pattern: str | None = None,
    min_confidence: float = 0.6,
    scope: str | None = None,
    repository: str | None = None,
    username: str = "main",
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Find elements matching a design pattern.

    Args:
        es: Elasticsearch repository.
        pattern: Pattern name (singleton, factory, builder, repository).
        min_confidence: Minimum confidence score.
        scope: Filter by scope.
        repository: Filter by repository.
        username: User branch.
        limit: Maximum results.

    Returns:
        List of elements matching the pattern.
    """
    must_clauses = [{"term": {"username": username}}]
    if scope:
        must_clauses.append({"term": {"scope": scope}})
    if repository:
        must_clauses.append({"term": {"repository": repository}})
    if pattern:
        must_clauses.append({"term": {"detected_patterns": pattern}})
    else:
        must_clauses.append({"exists": {"field": "detected_patterns"}})

    client = es._get_client()
    result = client.search(
        index="magaldi-code-elements",
        body={
            "query": {"bool": {"must": must_clauses}},
            "size": limit,
            "_source": ["element_id", "name", "relative_path", "line_start",
                       "detected_patterns", "pattern_confidence"],
        },
    )

    elements = []
    for hit in result.get("hits", {}).get("hits", []):
        source = hit["_source"]
        confidence = source.get("pattern_confidence", {})

        # Filter by minimum confidence
        if pattern and confidence.get(pattern, 0) < min_confidence:
            continue

        elements.append({
            "element_id": source.get("element_id"),
            "name": source.get("name"),
            "file": source.get("relative_path"),
            "line": source.get("line_start"),
            "patterns": source.get("detected_patterns", []),
            "confidence": confidence,
        })

    return elements[:limit]


def list_patterns(
    es: ElasticsearchRepository,
    scope: str | None = None,
    repository: str | None = None,
    username: str = "main",
) -> dict[str, int]:
    """List all detected patterns and their counts.

    Args:
        es: Elasticsearch repository.
        scope: Filter by scope.
        repository: Filter by repository.
        username: User branch.

    Returns:
        Dict mapping pattern names to counts.
    """
    must_clauses = [{"term": {"username": username}}]
    if scope:
        must_clauses.append({"term": {"scope": scope}})
    if repository:
        must_clauses.append({"term": {"repository": repository}})
    must_clauses.append({"exists": {"field": "detected_patterns"}})

    client = es._get_client()
    result = client.search(
        index="magaldi-code-elements",
        body={
            "query": {"bool": {"must": must_clauses}},
            "size": 0,
            "aggs": {
                "patterns": {
                    "terms": {
                        "field": "detected_patterns",
                        "size": 100,
                    }
                }
            },
        },
    )

    return {
        bucket["key"]: bucket["doc_count"]
        for bucket in result.get("aggregations", {}).get("patterns", {}).get("buckets", [])
    }
```

**Step 2: Verify no syntax errors**

Run: `python -c "from magaldi_mcp.tools import find_patterns, list_patterns"`
Expected: No output (success)

**Step 3: Commit**

```bash
git add src/magaldi_mcp/tools.py
git commit -m "feat: add pattern detection MCP tools"
```

---

### Task 8.6: Register New Tools in MCP Server

**Files:**
- Modify: `src/magaldi_mcp/server.py`

**Step 1: Add tool registrations**

In `server.py`, add the new tools to the tool registry. Find where other tools are registered and add:

```python
# Extended Code Intelligence Tools
@server.tool()
async def find_todos_tool(
    scope: str | None = None,
    repository: str | None = None,
    kind: str | None = None,
    assignee: str | None = None,
    limit: int = 50,
) -> list[dict]:
    """Find TODO/FIXME comments in the codebase."""
    from magaldi_mcp.tools import find_todos
    return find_todos(es_repo, scope, repository, "main", kind, assignee, limit)


@server.tool()
async def get_purity_tool(element_id: str) -> dict:
    """Get purity analysis for a function/method."""
    from magaldi_mcp.tools import get_purity
    return get_purity(es_repo, element_id)


@server.tool()
async def find_pure_functions_tool(
    scope: str | None = None,
    repository: str | None = None,
    confidence: str = "high",
    limit: int = 50,
) -> list[dict]:
    """Find pure functions in the codebase."""
    from magaldi_mcp.tools import find_pure_functions
    return find_pure_functions(es_repo, scope, repository, "main", confidence, limit)


@server.tool()
async def trace_type_tool(
    type_name: str,
    scope: str | None = None,
    repository: str | None = None,
    limit: int = 50,
) -> dict:
    """Track where a type is used in the codebase."""
    from magaldi_mcp.tools import trace_type
    return trace_type(es_repo, type_name, scope, repository, "main", limit)


@server.tool()
async def find_http_routes_tool(
    method: str | None = None,
    path_pattern: str | None = None,
    framework: str | None = None,
    scope: str | None = None,
    repository: str | None = None,
    limit: int = 50,
) -> list[dict]:
    """Find HTTP route handlers."""
    from magaldi_mcp.tools import find_http_routes
    return find_http_routes(es_repo, method, path_pattern, framework, scope, repository, "main", limit)


@server.tool()
async def find_cli_commands_tool(
    command_name: str | None = None,
    framework: str | None = None,
    scope: str | None = None,
    repository: str | None = None,
    limit: int = 50,
) -> list[dict]:
    """Find CLI command handlers."""
    from magaldi_mcp.tools import find_cli_commands
    return find_cli_commands(es_repo, command_name, framework, scope, repository, "main", limit)


@server.tool()
async def find_patterns_tool(
    pattern: str | None = None,
    min_confidence: float = 0.6,
    scope: str | None = None,
    repository: str | None = None,
    limit: int = 50,
) -> list[dict]:
    """Find elements matching a design pattern."""
    from magaldi_mcp.tools import find_patterns
    return find_patterns(es_repo, pattern, min_confidence, scope, repository, "main", limit)


@server.tool()
async def list_patterns_tool(
    scope: str | None = None,
    repository: str | None = None,
) -> dict:
    """List all detected patterns and their counts."""
    from magaldi_mcp.tools import list_patterns
    return list_patterns(es_repo, scope, repository, "main")
```

**Step 2: Verify no syntax errors**

Run: `python -c "from magaldi_mcp.server import server"`
Expected: No output (success)

**Step 3: Commit**

```bash
git add src/magaldi_mcp/server.py
git commit -m "feat: register code intelligence tools in MCP server"
```

---

## Phase 9: Final Integration Tests

### Task 9.1: Run Full Test Suite

**Step 1: Run all tests**

Run: `make test`
Expected: All tests PASS

**Step 2: Run linting**

Run: `make lint`
Expected: No errors

**Step 3: Run type checking**

Run: `make typecheck`
Expected: No errors (or only pre-existing ones)

**Step 4: Final commit if needed**

```bash
git add -A
git commit -m "fix: address any test/lint issues from code intelligence implementation"
```

---

## Summary

This implementation plan covers:

1. **Data Model (Phase 1)**: Added 10 new dataclasses and extended CodeElement with 13 new fields
2. **ES Mapping (Phase 1)**: Added nested mappings for all new fields
3. **TODO/Comment Extraction (Phase 2)**: `extract_todos`, `extract_section_markers`, `extract_comments`
4. **Purity Analysis (Phase 3)**: `analyze_purity`, `extract_side_effects` with impure call patterns
5. **Type Flow (Phase 4)**: `extract_type_annotations` for Python and TypeScript
6. **API Surface (Phase 5)**: `detect_http_routes`, `detect_cli_commands`, `detect_public_api`
7. **Pattern Detection (Phase 6)**: `detect_patterns` for singleton, builder, factory, repository
8. **Parser Integration (Phase 7)**: Integrated all extractors into PythonParser
9. **MCP Tools (Phase 8)**: Added 12 new MCP tools
10. **Final Testing (Phase 9)**: Verify all tests pass

Total new MCP tools: 12
- `find_todos`
- `get_purity`, `find_pure_functions`, `find_side_effects`
- `trace_type`, `find_type_producers`, `find_type_consumers`
- `find_http_routes`, `find_cli_commands`, `get_public_api`
- `find_patterns`, `list_patterns`
