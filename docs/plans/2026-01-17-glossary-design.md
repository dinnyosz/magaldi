# Glossary Feature Design

## Overview

Introduce a glossary system that extracts domain concepts from element names and links them to features. This enables discovery of codebases by domain terminology rather than just code structure.

**Example:** A codebase with `UserController`, `UserService`, `EmailValidator`, `sendRegistrationEmail` would generate glossary entries for "user", "registration", "email" - and features containing these elements would be linked to those glossary terms.

---

## Data Model

### Glossary Entry Structure

```
glossary_entry:
  term: string              # e.g., "email", "user", "registration"
  scope: string             # same scoping as elements
  repository: string
  username: string          # typically "main"

  # Occurrence tracking
  total_count: int          # total appearances across all elements
  element_ids: list[str]    # elements containing this term
  file_paths: list[str]     # files where term appears

  # Feature associations (populated after feature generation)
  feature_associations: [
    {
      feature_id: string,
      feature_label: string,
      frequency: int,       # how many members contain this term
      total_members: int,   # total elements in feature
      percentage: float     # frequency / total_members * 100
    }
  ]

  created_at: datetime
  updated_at: datetime
```

### Glossary ID Format

```
{scope}:{repository}:{username}:glossary:{term}

Examples:
backend-services:project-a:main:glossary:email
backend-services:project-a:main:glossary:user
```

### Storage

Stored in `magaldi-code-elements` index with `doc_type: "glossary"`, consistent with features and subfeatures.

---

## Term Extraction Logic

### Source

All element names: classes, functions, and methods.

### Splitting Algorithm

```python
def extract_terms(name: str) -> list[str]:
    # 1. Split CamelCase: "UserService" → ["User", "Service"]
    # 2. Split snake_case: "user_service" → ["user", "service"]
    # 3. Handle mixed: "getUserById" → ["get", "User", "By", "Id"]
    # 4. Lowercase all terms
    # 5. Filter out common terms
    # 6. Filter out single-character terms
    # 7. Return unique terms
```

### Common Terms Filter

Terms excluded from glossary (too generic to be useful):

```python
COMMON_TERMS = {
    # Verbs
    "get", "set", "add", "remove", "delete", "update", "create",
    "find", "fetch", "load", "save", "init", "handle", "process",
    "validate", "check", "is", "has", "can", "should",

    # Architectural suffixes
    "service", "controller", "handler", "manager", "factory",
    "repository", "provider", "helper", "util", "utils",
    "impl", "interface", "abstract", "base",

    # Common patterns
    "by", "for", "with", "from", "to", "and", "or", "the",
    "id", "ids", "name", "type", "data", "info", "item", "items",
    "list", "array", "map", "dict", "config", "options", "params",
    "request", "response", "result", "error", "exception",
    "test", "spec", "mock",
}
```

This list is configurable per-repository.

---

## Pipeline & Worker Job

### Job Trigger

Glossary extraction runs as a separate worker job after parsing completes:

```
Parse Complete → Queue Jobs:
  - summarization jobs (existing)
  - embedding jobs (existing)
  - glossary_extraction job (NEW)
```

### Job Flow

```
1. EXTRACT PHASE (runs once after parse)
   ├── Fetch all elements for (scope, repository, username)
   ├── For each element:
   │   └── extract_terms(element.name) → terms[]
   ├── Aggregate: term → {count, element_ids[], file_paths[]}
   └── Upsert glossary entries to Elasticsearch

2. LINK PHASE (runs after feature generation completes)
   ├── Fetch all features for (scope, repository, username)
   ├── Fetch all glossary entries
   ├── For each feature:
   │   ├── Get member element names
   │   ├── Extract terms from each member
   │   ├── Count term frequencies
   │   └── Update glossary entries with feature associations
   └── Bulk update glossary documents
```

### Job Dependencies

```
parse_complete
    ├──→ summarization_jobs ──→ feature_clustering ──┐
    ├──→ embedding_jobs                              │
    └──→ glossary_extraction (extract phase) ────────┴──→ glossary_linking (link phase)
```

The link phase waits for both glossary extraction AND feature clustering to complete.

---

## MCP Tool Exposure

### New Tools

#### `list_glossary`

List all glossary terms for a repository.

- **Params:** scope, repository, min_count (filter low-frequency terms)
- **Returns:** terms sorted by occurrence count
- **Use case:** "What domain concepts exist in this codebase?"

#### `get_glossary_term`

Get details for a specific term.

- **Params:** scope, repository, term
- **Returns:** full glossary entry with element IDs, files, feature associations
- **Use case:** "Where is 'email' used and which features involve it?"

#### `search_glossary`

Search glossary terms by partial match.

- **Params:** scope, repository, query (e.g., "user" matches "user", "username")
- **Returns:** matching terms with counts
- **Use case:** "Find all terms related to 'auth'"

### Enhanced Existing Tools

#### `get_feature_members` (enhance)

Add `glossary_terms` field to response showing `[{term, frequency, percentage}]` for the feature.

- **Use case:** "What domain concepts does this feature deal with?"

#### `search_features` (enhance)

Add optional `glossary_term` filter parameter.

- **Params:** ..., glossary_term="email"
- **Returns:** features where term appears in >N% of members
- **Use case:** "Find all features involving email"

---

## Implementation Details

### Elasticsearch Mapping Addition

```python
"glossary": {
    "properties": {
        "doc_type": {"type": "keyword"},  # "glossary"
        "term": {"type": "keyword"},
        "scope": {"type": "keyword"},
        "repository": {"type": "keyword"},
        "username": {"type": "keyword"},
        "total_count": {"type": "integer"},
        "element_ids": {"type": "keyword"},  # array
        "file_paths": {"type": "keyword"},   # array
        "feature_associations": {
            "type": "nested",
            "properties": {
                "feature_id": {"type": "keyword"},
                "feature_label": {"type": "keyword"},
                "frequency": {"type": "integer"},
                "total_members": {"type": "integer"},
                "percentage": {"type": "float"}
            }
        },
        "created_at": {"type": "date"},
        "updated_at": {"type": "date"}
    }
}
```

### File Structure

```
src/
  shared/
    ai/
      glossary/
        __init__.py
        extractor.py      # Term extraction logic
        linker.py         # Feature-glossary linking
  workers/
    glossary_worker.py    # Job processing
  mcp/
    tools/
      glossary_tools.py   # MCP tool implementations
```

### Configuration

Optional additions to `magaldi.yaml`:

```yaml
glossary:
  extra_common_terms: ["foo", "bar"]     # add to filter
  preserve_terms: ["id", "api"]          # remove from filter
  min_term_length: 2                     # default: 2
```

---

## Summary

| Aspect | Decision |
|--------|----------|
| Source | All element names (classes, functions, methods) |
| Splitting | CamelCase + snake_case with common term filtering |
| Timing | Separate worker job after parsing |
| Association | Term-based matching with frequency tracking |
| Weight | Frequency count + percentage of feature members |
| Storage | `doc_type: "glossary"` in main index |
| MCP | 3 new tools + 2 enhanced existing tools |
