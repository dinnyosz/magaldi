# Extended Code Intelligence Features

**Date:** 2026-01-25
**Status:** Design Complete

## Overview

This document describes 5 new code intelligence features that extend Magaldi's tree-sitter extraction capabilities:

1. **Type Flow Analysis** - Track how types flow through the codebase
2. **Pattern Detection** - Detect design patterns (singleton, factory, observer, etc.)
3. **Documentation Linkage** - Extract TODOs, comments, and section markers
4. **API Surface Analysis** - Identify public APIs, HTTP routes, CLI commands
5. **Purity/Mutation Tracking** - Analyze side effects and function purity

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Storage approach | Extend existing CodeElement | Simple, unified, everything in one place |
| Language scope | All languages with language-appropriate depth | Consistent baseline, deeper where language enables it |
| MCP tool granularity | Many specialized tools | Clear purpose, easy discovery, focused parameters |
| Pattern detection scope | Full suite (10+ patterns) | Comprehensive analysis capability |
| TODO extraction | Full comment intelligence | TODOs + metadata + section markers + association |
| Framework detection | Extensible registry with full initial coverage | Future-proof + comprehensive |
| Purity analysis | Conservative with confidence levels | Safe for refactoring decisions, transparent reasoning |

---

## Data Model

### New Fields on CodeElement

```python
@dataclass
class CodeElement:
    # ... existing fields ...

    # === TYPE FLOW ===
    type_annotations: list[dict]  # TypeAnnotation as dicts for ES storage

    # === PATTERNS ===
    detected_patterns: list[str]  # ["singleton", "factory", "builder"]
    pattern_confidence: dict[str, float]  # {"singleton": 0.95}

    # === DOCUMENTATION ===
    todos: list[dict]  # TodoItem as dicts
    section_markers: list[dict]  # SectionMarker as dicts
    associated_comments: list[dict]  # Comment as dicts

    # === API SURFACE ===
    is_public_api: bool
    http_routes: list[dict]  # HttpRoute as dicts
    cli_commands: list[dict]  # CliCommand as dicts

    # === PURITY/MUTATION ===
    purity: dict | None  # PurityInfo as dict
    side_effects: list[dict]  # SideEffect as dicts
    mutated_state: list[str]
```

### Supporting Data Structures

#### Type Flow

```python
@dataclass
class TypeAnnotation:
    name: str              # "User", "List[str]", "Optional[int]"
    kind: str              # "parameter", "return", "variable", "attribute"
    location: str          # "param:user_id", "return", "var:result"
    line: int
    generic_args: list[str] | None  # ["str"] for List[str]
```

#### Documentation

```python
@dataclass
class TodoItem:
    kind: str              # "TODO", "FIXME", "HACK", "XXX", "BUG", "NOTE"
    text: str
    line: int
    assignee: str | None   # "alice" from TODO(alice)
    priority: str | None   # "high", "low"
    issue_ref: str | None  # "GH-123", "#456"

@dataclass
class SectionMarker:
    label: str             # "HELPERS", "PRIVATE METHODS"
    line: int
    style: str             # "equals", "dashes", "hash"

@dataclass
class Comment:
    text: str
    line: int
    kind: str              # "inline", "block", "docstring"
    position: str          # "above", "inline", "below"
```

#### API Surface

```python
@dataclass
class HttpRoute:
    method: str            # "GET", "POST", "PUT", "DELETE"
    path: str              # "/users/{id}"
    path_params: list[str] # ["id"]
    framework: str         # "fastapi", "flask", "express"

@dataclass
class CliCommand:
    name: str              # "parse", "index"
    options: list[dict]    # CliOption as dicts
    framework: str         # "click", "typer", "argparse"

@dataclass
class CliOption:
    name: str              # "--verbose", "-v"
    type: str | None       # "bool", "str", "int"
    required: bool
    default: str | None
```

#### Purity/Mutation

```python
@dataclass
class PurityInfo:
    level: str             # "pure", "read_only", "mutates_self", "mutates_external"
    confidence: str        # "high", "medium", "low"
    reasons: list[str]     # ["calls print()", "modifies self.cache"]

@dataclass
class SideEffect:
    kind: str              # "state_mutation", "io_file", "io_network", "console", "subprocess"
    target: str | None     # "self.cache", "/tmp/file.txt"
    line: int
```

---

## Elasticsearch Mapping Additions

```python
# Add to INDEX_MAPPING["mappings"]["properties"]
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
"detected_patterns": {"type": "keyword"},
"pattern_confidence": {"type": "object"},
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

---

## Extraction Architecture

### Extraction Flow

```
Source File
    ↓
tree-sitter parse → AST
    ↓
┌─────────────────────────────────────────────────┐
│  Existing Extractors                            │
│  - extract_{lang}_elements()                    │
│  - extract_{lang}_calls()                       │
│  - extract_{lang}_imports()                     │
└─────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────┐
│  New Extractors (run on same AST)               │
│  - extract_type_annotations(node)               │
│  - extract_todos_and_comments(source, lines)    │
│  - detect_patterns(element, calls, class_info)  │
│  - detect_api_surface(element, decorators)      │
│  - analyze_purity(element, calls)               │
└─────────────────────────────────────────────────┘
    ↓
CodeElement (enriched)
    ↓
Elasticsearch
```

### Extraction Timing

| Extractor | Input | When |
|-----------|-------|------|
| Type annotations | AST node | During element extraction |
| TODOs/Comments | Raw source lines | After parsing, before storage |
| Pattern detection | Element + calls + class info | Post-extraction pass |
| API surface | Decorators + framework detection | During element extraction |
| Purity analysis | Element + calls + AST | Post-extraction pass |

### Language Dispatching

```python
def extract_type_annotations(node: Node, language: str) -> list[TypeAnnotation]:
    match language:
        case "python": return _extract_python_type_annotations(node)
        case "typescript" | "javascript": return _extract_ts_type_annotations(node)
        case "php": return _extract_php_type_annotations(node)
        case "rust": return _extract_rust_type_annotations(node)
        case _: return []
```

---

## Pattern Detection

### Patterns to Detect

#### Easy (Structural Analysis)

| Pattern | Detection Heuristics |
|---------|---------------------|
| **Singleton** | Private constructor + static `_instance` attribute + `get_instance()` or `__new__` |
| **Factory** | Function returning instances of different classes |
| **Builder** | Methods returning `self`/`this` + used in method chains |
| **Decorator** | Function taking callable, returning callable, with inner wrapper |

#### Medium (Call Graph Required)

| Pattern | Detection Heuristics |
|---------|---------------------|
| **Observer** | `subscribe`/`add_listener` + `emit`/`notify` methods + callback storage |
| **Repository** | `get`/`find`/`save`/`delete`/`update` methods on consistent type |
| **Strategy** | Interface/Protocol + multiple implementations with same signatures |

#### Hard (Deep Analysis)

| Pattern | Detection Heuristics |
|---------|---------------------|
| **Dependency Injection** | Constructor params typed as abstract + no concrete instantiation |
| **State Machine** | Enum states + transition methods checking/changing state |
| **Command** | `execute()` method + optional `undo()` + command history pattern |

### Confidence Scoring

```python
def detect_singleton(element: CodeElement) -> tuple[bool, float]:
    score = 0.0

    if has_private_constructor(element): score += 0.3
    if has_instance_attribute(element): score += 0.3
    if has_get_instance_method(element): score += 0.2
    if returns_cached_instance(element): score += 0.2

    return (score >= 0.6, score)
```

---

## Comment/TODO Extraction

### Regex Patterns

```python
TODO_PATTERN = re.compile(r"""
    (?P<kind>TODO|FIXME|HACK|XXX|BUG|NOTE|OPTIMIZE)
    (?:\((?P<assignee>\w+)\))?      # TODO(alice)
    (?:\s*@(?P<mention>\w+))?       # TODO @alice
    (?:\s*(?P<priority>!+))?        # TODO!!
    (?:\s*(?P<issue>\#?\w+-?\d+))?  # TODO #123 or GH-456
    \s*:?\s*
    (?P<text>.+)
""", re.VERBOSE | re.IGNORECASE)

SECTION_PATTERN = re.compile(r"""
    ^\s*[#/]+\s*
    (?:={3,}|-{3,})\s*
    (?P<label>[A-Z][A-Z0-9 _]+)
    \s*(?:={3,}|-{3,})?
""", re.VERBOSE)
```

### Comment Association

```python
def associate_comments(element: CodeElement, all_comments: list[Comment]) -> list[Comment]:
    associated = []

    # Comments directly above element (within 3 lines, no code between)
    for comment in all_comments:
        if element.line_start - 3 <= comment.line < element.line_start:
            if no_code_between(comment.line, element.line_start):
                associated.append(comment)

    # Inline comments on same line
    for comment in all_comments:
        if comment.line == element.line_start and comment.kind == "inline":
            associated.append(comment)

    return associated
```

---

## API Surface Detection

### Framework Registry

```python
@dataclass
class FrameworkPattern:
    name: str
    language: str
    decorator_patterns: list[str]
    route_extractor: Callable

FRAMEWORK_REGISTRY = [
    # Python HTTP
    FrameworkPattern("fastapi", "python",
        ["router.get", "router.post", "router.put", "router.delete", "app.get", "app.post"],
        extract_fastapi_route),
    FrameworkPattern("flask", "python",
        ["app.route", "blueprint.route"],
        extract_flask_route),
    FrameworkPattern("django", "python", [], extract_django_route),

    # Python CLI
    FrameworkPattern("click", "python",
        ["click.command", "click.group", "click.option", "click.argument"],
        extract_click_command),
    FrameworkPattern("typer", "python",
        ["app.command", "typer.Option", "typer.Argument"],
        extract_typer_command),

    # JavaScript/TypeScript HTTP
    FrameworkPattern("express", "javascript", [], extract_express_route),
    FrameworkPattern("fastify", "javascript", [], extract_fastify_route),
    FrameworkPattern("nestjs", "typescript",
        ["Get", "Post", "Put", "Delete", "Controller"],
        extract_nestjs_route),

    # JavaScript CLI
    FrameworkPattern("commander", "javascript", [], extract_commander_command),

    # PHP
    FrameworkPattern("laravel", "php", [], extract_laravel_route),
    FrameworkPattern("symfony", "php", ["Route"], extract_symfony_route),

    # Rust
    FrameworkPattern("actix", "rust", ["get", "post", "put", "delete"], extract_actix_route),
    FrameworkPattern("axum", "rust", [], extract_axum_route),
    FrameworkPattern("clap", "rust", ["command", "arg"], extract_clap_command),
]
```

---

## Purity Analysis

### Impure Call Patterns

```python
IMPURE_CALLS = {
    "python": {
        "io_file": ["open", "read", "write", "Path.write_text", "Path.read_text",
                    "Path.mkdir", "os.remove", "shutil.copy"],
        "io_network": ["requests.get", "requests.post", "httpx.get", "httpx.post",
                       "urllib.request.urlopen", "socket.connect"],
        "console": ["print", "logging.info", "logging.debug", "logging.warning",
                    "logging.error", "logger.info", "logger.debug"],
        "subprocess": ["subprocess.run", "subprocess.call", "subprocess.Popen",
                       "os.system", "os.popen"],
        "database": ["cursor.execute", "session.commit", "session.add"],
    },
    "javascript": {
        "io_file": ["fs.readFile", "fs.writeFile", "fs.readFileSync", "fs.writeFileSync"],
        "io_network": ["fetch", "axios.get", "axios.post", "http.get"],
        "console": ["console.log", "console.error", "console.warn"],
        "subprocess": ["child_process.exec", "child_process.spawn"],
    },
    # ... similar for PHP, Rust
}
```

### Purity Levels

| Level | Description | Criteria |
|-------|-------------|----------|
| `pure` | No side effects | No impure calls, no state mutation, no globals |
| `read_only` | Reads but doesn't modify | May read globals/files, doesn't write |
| `mutates_self` | Modifies instance state | `self.x = ...` but no external effects |
| `mutates_external` | Modifies external state | Writes globals, files, network, etc. |

### Conservative Approach

Mark as **impure** when:
- Any call to known impure functions
- Any `self.x = ...` assignments → `mutates_self`
- Any global variable write → `mutates_external`
- Any `yield` (generators have implicit state)
- Calls to unknown functions (can't prove pure)

Mark as **pure** with high confidence only when:
- No external function calls, OR all calls are to known pure functions
- No attribute mutations
- No global access
- Parameters only read, not modified

---

## MCP Tools

### Type Flow Tools

| Tool | Description | Parameters |
|------|-------------|------------|
| `trace_type` | Track where a type is used | `type_name`, `scope`, `repository` |
| `find_type_producers` | Functions returning this type | `type_name`, `scope`, `repository` |
| `find_type_consumers` | Functions taking this type | `type_name`, `scope`, `repository` |

### Pattern Tools

| Tool | Description | Parameters |
|------|-------------|------------|
| `find_patterns` | Find elements matching pattern | `pattern`, `min_confidence`, `scope`, `repository` |
| `get_element_patterns` | Get patterns for element | `element_id` |
| `list_patterns` | List all patterns in repo | `scope`, `repository` |

### Documentation Tools

| Tool | Description | Parameters |
|------|-------------|------------|
| `find_todos` | Find TODO comments | `kind`, `assignee`, `scope`, `repository` |
| `get_section_structure` | Get section markers | `file_path`, `scope`, `repository` |
| `get_element_comments` | Get associated comments | `element_id` |

### API Surface Tools

| Tool | Description | Parameters |
|------|-------------|------------|
| `get_public_api` | List public elements | `scope`, `repository` |
| `find_http_routes` | Find HTTP handlers | `method`, `path_pattern`, `framework` |
| `find_cli_commands` | Find CLI handlers | `command_name`, `framework` |
| `get_api_surface` | Combined API view | `scope`, `repository` |

### Purity Tools

| Tool | Description | Parameters |
|------|-------------|------------|
| `get_purity` | Get purity info | `element_id` |
| `find_pure_functions` | Find pure functions | `scope`, `repository`, `confidence` |
| `find_side_effects` | Find functions with effect | `effect_kind`, `scope`, `repository` |
| `get_mutation_graph` | What mutates what | `scope`, `repository` |

---

## Implementation Phases

### Phase 1: Data Model & Storage (Foundation)
**Files:** `tree_sitter_manager.py`, `code_parser.py`, `elasticsearch.py`

- Add new dataclasses for all structures
- Extend CodeElement with new fields
- Update ES mapping with new nested fields
- Update `index_element()` to store new fields

### Phase 2: Comment/TODO Extraction
**Files:** `tree_sitter_manager.py`, `tools.py`

- Implement `extract_todos(source: str) -> list[TodoItem]`
- Implement `extract_section_markers(source: str) -> list[SectionMarker]`
- Implement `extract_comments(tree: Tree, lines: list[str]) -> list[Comment]`
- Implement `associate_comments(element, comments) -> list[Comment]`
- Add MCP tools: `find_todos`, `get_section_structure`, `get_element_comments`

### Phase 3: Purity Analysis
**Files:** `tree_sitter_manager.py`, `tools.py`

- Define `IMPURE_CALLS` patterns for all languages
- Implement `analyze_purity(element, language) -> PurityInfo`
- Implement `extract_side_effects(element, calls) -> list[SideEffect]`
- Add MCP tools: `get_purity`, `find_pure_functions`, `find_side_effects`

### Phase 4: Type Flow
**Files:** `tree_sitter_manager.py`, `tools.py`

- Implement `extract_type_annotations(node, language) -> list[TypeAnnotation]`
- Handle generics parsing (List[T], Optional[T], etc.)
- Add MCP tools: `trace_type`, `find_type_producers`, `find_type_consumers`

### Phase 5: API Surface
**Files:** `tree_sitter_manager.py`, `tools.py`

- Implement `FrameworkPattern` registry
- Implement framework-specific route extractors
- Implement `detect_public_api(element) -> bool`
- Add MCP tools: `get_public_api`, `find_http_routes`, `find_cli_commands`

### Phase 6: Pattern Detection
**Files:** `tree_sitter_manager.py`, `tools.py`

- Implement easy pattern detectors (singleton, factory, builder, decorator)
- Implement medium pattern detectors (observer, repository, strategy)
- Implement hard pattern detectors (DI, state machine, command)
- Add MCP tools: `find_patterns`, `get_element_patterns`, `list_patterns`

---

## File Changes Summary

| File | Changes |
|------|---------|
| `src/magaldi_core/tree_sitter_manager.py` | Add dataclasses, extraction functions for all 5 features |
| `src/magaldi_core/code_parser.py` | Extend CodeElement, integrate new extractors in parsing |
| `src/shared/db/elasticsearch.py` | Add ES mapping, update index_element() |
| `src/magaldi_mcp/tools.py` | Add ~18 new MCP tools |
| `src/magaldi_web/models.py` | Add Pydantic models for new data |
| `src/magaldi_web/routes/elements.py` | Expose new fields in element detail |

---

## Success Criteria

1. **Type Flow**: Can answer "Where is type X used?" across the codebase
2. **Patterns**: Detects 10+ patterns with >80% precision
3. **TODOs**: Extracts all TODO variants with metadata (assignee, priority, issue ref)
4. **API Surface**: Identifies HTTP routes and CLI commands for all supported frameworks
5. **Purity**: Correctly classifies >90% of functions with high confidence

---

## Open Questions

1. Should pattern detection run on every parse, or be a separate "analyze" command?
2. How to handle purity analysis for functions calling other project functions? (transitive purity)
3. Should we track type flow across file boundaries during indexing, or compute at query time?
