# Code Knowledge Graph Design

## Overview

Extend Magaldi from "code search" to "code understanding" by extracting and storing relationships between code elements as a queryable knowledge graph.

## Goals

1. **Trace execution paths** - "What happens when a user signs up?"
2. **Impact analysis** - "What breaks if I change this?"
3. **Data lineage** - "Where does this value come from/go to?"
4. **Architecture visualization** - Auto-generate diagrams from code
5. **Natural language queries** - "Show me the authentication flow"

## Data Model

### Nodes (existing CodeElement)

Already indexed:
- Functions, methods, classes, files
- Features, subfeatures
- Glossary terms

### Edges (new: CodeRelationship)

```python
class RelationshipType(Enum):
    # Inheritance & Types
    INHERITS = "inherits"              # class Foo(Bar)
    IMPLEMENTS = "implements"          # class Foo(Protocol)

    # Composition & Dependencies
    INJECTS = "injects"                # @inject, constructor params
    IMPORTS = "imports"                # from x import y
    INSTANTIATES = "instantiates"      # Foo()

    # Registration & Hierarchy
    REGISTERS_COMMAND = "registers_command"    # @group.command
    REGISTERS_ROUTE = "registers_route"        # @blueprint.route
    REGISTERS_HANDLER = "registers_handler"    # @emitter.on
    REGISTERS_PLUGIN = "registers_plugin"      # @app.register
    BELONGS_TO_GROUP = "belongs_to_group"      # CLI group hierarchy
    MOUNTS_ROUTER = "mounts_router"            # app.include_router()

    # Data Flow
    READS_CONFIG = "reads_config"      # config.get("KEY")
    READS_ENV = "reads_env"            # os.environ["KEY"]
    QUERIES_MODEL = "queries_model"    # Model.query.filter()
    MUTATES_MODEL = "mutates_model"    # model.save()

    # API Contracts
    ACCEPTS_TYPE = "accepts_type"      # Request body type
    RETURNS_TYPE = "returns_type"      # Response type
    RAISES = "raises"                  # Exception types

    # Control Flow
    CALLS = "calls"                    # Already have this
    CALLED_BY = "called_by"            # Inverse of calls
    TRIGGERS_EVENT = "triggers_event"  # emit("user.created")
    HANDLES_EVENT = "handles_event"    # @on("user.created")

    # Testing
    TESTS = "tests"                    # test_foo tests foo
    MOCKS = "mocks"                    # @patch("module.foo")
    FIXTURES = "fixtures"              # @pytest.fixture used by

class CodeRelationship:
    """Edge in the knowledge graph."""

    relationship_id: str          # Unique ID
    source_id: str                # Source element_id
    target_id: str                # Target element_id (or external ref)
    relationship_type: RelationshipType

    # Metadata
    confidence: float             # 0-1, how certain is this relationship
    line: int | None              # Where in source this relationship is defined

    # Type-specific data
    details: dict                 # Relationship-specific metadata
    # Examples:
    # - REGISTERS_COMMAND: {"command_name": "serve", "group_path": ["main", "web"]}
    # - REGISTERS_ROUTE: {"method": "GET", "path": "/users/{id}", "full_path": "/api/v1/users/{id}"}
    # - READS_CONFIG: {"key": "database.url", "default": None}
    # - HANDLES_EVENT: {"event_pattern": "user.*", "async": True}
```

### External References

Some relationships point outside the codebase:

```python
class ExternalReference:
    """Reference to something outside the indexed code."""

    ref_type: str          # "config_key", "env_var", "event", "table", "url"
    ref_id: str            # Unique identifier
    name: str              # Human-readable name

    # Examples:
    # - {"ref_type": "env_var", "ref_id": "env:DATABASE_URL", "name": "DATABASE_URL"}
    # - {"ref_type": "event", "ref_id": "event:user.created", "name": "user.created"}
    # - {"ref_type": "table", "ref_id": "table:users", "name": "users"}
```

## Extraction Pipeline

### Phase 1: Registration Hierarchies (CLI, Routes, Events)

**CLI Command Hierarchy:**

```python
class CliHierarchyExtractor:
    """Extract Click/Typer command hierarchies."""

    def extract(self, elements: list[CodeElement]) -> list[CodeRelationship]:
        relationships = []

        # 1. Find all @group and @command decorators
        groups = {}  # name -> element_id
        commands = []

        for el in elements:
            for dec in el.decorator_details:
                if dec.name.endswith(".group"):
                    # @main.group -> this defines a group
                    groups[el.name] = el.element_id
                elif dec.name.endswith(".command"):
                    # @web.command("serve") -> command under "web" group
                    parent_name = dec.name.split(".")[0]  # "web"
                    commands.append((el, parent_name, dec.args))

        # 2. Resolve parent references
        for el, parent_name, args in commands:
            if parent_name in groups:
                relationships.append(CodeRelationship(
                    source_id=el.element_id,
                    target_id=groups[parent_name],
                    relationship_type=RelationshipType.BELONGS_TO_GROUP,
                    details={
                        "command_name": extract_command_name(args),
                        "parent_group": parent_name,
                    }
                ))

        return relationships
```

**HTTP Route Hierarchy:**

```python
class RouteHierarchyExtractor:
    """Extract Flask blueprint / FastAPI router hierarchies."""

    def extract(self, elements: list[CodeElement]) -> list[CodeRelationship]:
        # 1. Find router/blueprint definitions
        # 2. Find app.include_router() / app.register_blueprint() calls
        # 3. Build prefix chains: /api + /v1 + /users = /api/v1/users
        # 4. Create MOUNTS_ROUTER and REGISTERS_ROUTE relationships
        pass
```

### Phase 2: Data Flow (Config, Env, Models)

**Config/Env Extractor:**

```python
class ConfigExtractor:
    """Extract config and environment variable usage."""

    PATTERNS = [
        # Python
        (r'os\.environ\[(["\'])(.+?)\1\]', "env_var"),
        (r'os\.getenv\((["\'])(.+?)\1', "env_var"),
        (r'config\.get\((["\'])(.+?)\1', "config_key"),
        (r'settings\.(\w+)', "config_key"),
        # JavaScript
        (r'process\.env\.(\w+)', "env_var"),
    ]

    def extract(self, element: CodeElement) -> list[CodeRelationship]:
        relationships = []
        for pattern, ref_type in self.PATTERNS:
            for match in re.finditer(pattern, element.raw_code):
                key = match.group(2) if match.lastindex >= 2 else match.group(1)
                relationships.append(CodeRelationship(
                    source_id=element.element_id,
                    target_id=f"{ref_type}:{key}",
                    relationship_type=RelationshipType.READS_CONFIG if ref_type == "config_key"
                                     else RelationshipType.READS_ENV,
                    details={"key": key, "line": get_line(match)}
                ))
        return relationships
```

**ORM Model Extractor:**

```python
class ORMExtractor:
    """Extract database model relationships and queries."""

    def extract(self, elements: list[CodeElement]) -> list[CodeRelationship]:
        # 1. Find model definitions (class User(Base), class User(models.Model))
        # 2. Extract field relationships (ForeignKey, relationship())
        # 3. Find query patterns (Model.query, session.query(Model))
        # 4. Create QUERIES_MODEL and MUTATES_MODEL relationships
        pass
```

### Phase 3: Event & Message Flow

**Event Extractor:**

```python
class EventExtractor:
    """Extract event emission and handling."""

    EMIT_PATTERNS = [
        (r'emit\((["\'])(.+?)\1', "emit"),
        (r'publish\((["\'])(.+?)\1', "publish"),
        (r'send\((["\'])(.+?)\1', "send"),
    ]

    HANDLER_DECORATORS = [
        "on", "subscribe", "handler", "listener", "consumer"
    ]

    def extract(self, element: CodeElement) -> list[CodeRelationship]:
        # Match emitters to handlers via event names
        pass
```

### Phase 4: Test Coverage Mapping

```python
class TestMappingExtractor:
    """Map tests to code under test."""

    def extract(self, test_element: CodeElement) -> list[CodeRelationship]:
        relationships = []

        # 1. Parse test name: test_user_signup -> user_signup
        # 2. Find imports in test file
        # 3. Analyze calls within test
        # 4. Check @patch decorators for mocked dependencies

        return relationships
```

## Storage

### Elasticsearch Index: `magaldi-relationships`

```json
{
  "mappings": {
    "properties": {
      "relationship_id": { "type": "keyword" },
      "source_id": { "type": "keyword" },
      "target_id": { "type": "keyword" },
      "relationship_type": { "type": "keyword" },
      "confidence": { "type": "float" },
      "line": { "type": "integer" },
      "details": { "type": "object", "enabled": true },

      "scope": { "type": "keyword" },
      "repository": { "type": "keyword" },
      "username": { "type": "keyword" },
      "indexed_at": { "type": "date" }
    }
  }
}
```

### Elasticsearch Index: `magaldi-external-refs`

```json
{
  "mappings": {
    "properties": {
      "ref_id": { "type": "keyword" },
      "ref_type": { "type": "keyword" },
      "name": { "type": "keyword" },
      "description": { "type": "text" },

      "scope": { "type": "keyword" },
      "repository": { "type": "keyword" }
    }
  }
}
```

## Query API

### MCP Tools

```python
# Find all relationships from/to an element
def get_relationships(
    element_id: str,
    direction: Literal["outgoing", "incoming", "both"] = "both",
    relationship_types: list[str] | None = None,
) -> list[CodeRelationship]:
    pass

# Trace a path between two elements
def find_path(
    source_id: str,
    target_id: str,
    max_depth: int = 10,
    relationship_types: list[str] | None = None,
) -> list[list[CodeRelationship]]:
    pass

# Get CLI command tree
def get_command_tree(
    scope: str,
    repository: str,
    root_command: str | None = None,  # e.g., "magaldi"
) -> CommandTree:
    pass

# Get route tree
def get_route_tree(
    scope: str,
    repository: str,
    base_path: str | None = None,  # e.g., "/api/v1"
) -> RouteTree:
    pass

# Find all usages of a config key or env var
def find_config_usages(
    key: str,
    ref_type: Literal["config_key", "env_var"],
    scope: str,
    repository: str,
) -> list[CodeElement]:
    pass

# Trace data flow
def trace_data_flow(
    element_id: str,
    direction: Literal["forward", "backward", "both"] = "both",
    max_depth: int = 5,
) -> DataFlowGraph:
    pass

# Impact analysis
def analyze_impact(
    element_id: str,
    change_type: Literal["modify", "delete", "rename"],
) -> ImpactReport:
    """What would be affected if this element changes?"""
    pass
```

### Web UI Pages

1. **CLI Command Explorer** - Tree view of all CLI commands with full paths
2. **Route Explorer** - Tree view of all HTTP routes with full URLs
3. **Config/Env Catalog** - All config keys and env vars with their usages
4. **Event Catalog** - All events with emitters and handlers
5. **Dependency Graph** - Interactive graph visualization
6. **Data Flow Tracer** - Visual trace of data through the system
7. **Impact Analyzer** - "What if I change this?" tool

## Implementation Phases

### Phase 1: CLI & Route Hierarchies (Foundation)
- [ ] Implement `CliHierarchyExtractor`
- [ ] Implement `RouteHierarchyExtractor`
- [ ] Create `magaldi-relationships` index
- [ ] Add MCP tools: `get_command_tree`, `get_route_tree`
- [ ] Add Web UI: CLI Command Explorer, Route Explorer
- [ ] Update Entry Points page to show full command paths

**Deliverable:** `magaldi web serve` displays correctly, route tree shows `/api/v1/users/{id}`

### Phase 2: Config & Environment
- [ ] Implement `ConfigExtractor`
- [ ] Create `magaldi-external-refs` index
- [ ] Add MCP tool: `find_config_usages`
- [ ] Add Web UI: Config/Env Catalog

**Deliverable:** "What code uses DATABASE_URL?" works

### Phase 3: Data Flow & Models
- [ ] Implement `ORMExtractor`
- [ ] Implement basic data flow tracing
- [ ] Add MCP tools: `trace_data_flow`
- [ ] Add Web UI: Data Flow Tracer

**Deliverable:** Can trace how user input flows through the system

### Phase 4: Events & Messages
- [ ] Implement `EventExtractor`
- [ ] Add MCP tool: event queries
- [ ] Add Web UI: Event Catalog

**Deliverable:** "What happens when user.created is emitted?" works

### Phase 5: Impact Analysis
- [ ] Implement `ImpactAnalyzer` combining all relationships
- [ ] Add MCP tool: `analyze_impact`
- [ ] Add Web UI: Impact Analyzer

**Deliverable:** "What breaks if I delete this function?" works

### Phase 6: Natural Language Queries
- [ ] Train/prompt LLM to translate questions to graph queries
- [ ] Add conversational interface

**Deliverable:** "Show me the authentication flow" returns relevant subgraph

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         Parsing Pipeline                         │
├─────────────────────────────────────────────────────────────────┤
│  Tree-sitter  →  Element Extractor  →  Relationship Extractors  │
│                                              │                   │
│                                              ▼                   │
│                                    ┌─────────────────────┐       │
│                                    │  CLI Hierarchy      │       │
│                                    │  Route Hierarchy    │       │
│                                    │  Config/Env         │       │
│                                    │  ORM Models         │       │
│                                    │  Events             │       │
│                                    │  Test Mapping       │       │
│                                    └─────────────────────┘       │
└─────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Elasticsearch                             │
├─────────────────────────────────────────────────────────────────┤
│  magaldi-elements  │  magaldi-relationships  │  magaldi-ext-refs │
└─────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                         Query Layer                              │
├─────────────────────────────────────────────────────────────────┤
│  Graph Traversal  │  Path Finding  │  Impact Analysis           │
└─────────────────────────────────────────────────────────────────┘
                                     │
                          ┌──────────┴──────────┐
                          ▼                     ▼
                    ┌──────────┐          ┌──────────┐
                    │ MCP Tools │          │  Web UI  │
                    └──────────┘          └──────────┘
```

## Open Questions

1. **Graph database?** Should we use Neo4j/DGraph instead of ES for relationships?
   - Pro: Native graph queries, better path finding
   - Con: Another dependency, ES can handle it with proper indices

2. **Real-time vs batch?** Extract relationships during parsing or as post-process?
   - Recommendation: During parsing for speed, with background enrichment

3. **Cross-repo relationships?** Handle imports from other indexed repos?
   - Recommendation: Phase 2+, start with single-repo

4. **Confidence scoring?** How to handle uncertain relationships?
   - Recommendation: Store confidence, filter in queries

5. **Visualization library?** D3.js, Cytoscape.js, or something else?
   - Recommendation: Cytoscape.js for graph viz, D3 for trees
