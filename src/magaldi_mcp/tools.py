"""MCP Tool implementations for Magaldi.

Each tool function takes an ElasticsearchRepository and optional CodeEmbeddingClient,
plus tool-specific parameters, and returns a dict or list result.
"""

from __future__ import annotations

from typing import Any

from shared.db.elasticsearch import ElasticsearchRepository
from shared.ai.embedding import CodeEmbeddingClient


# =============================================================================
# SEARCH TOOLS
# =============================================================================


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
    include_tests: bool = True,
) -> dict[str, Any]:
    """Semantic search for code elements.

    Tries vector search first, falls back to keyword search if Ollama unavailable.

    Args:
        es: Elasticsearch repository.
        embed_client: Embedding client for query (optional, falls back to keyword).
        query: Natural language search query.
        scope: Filter by scope.
        repository: Filter by repository.
        username: User branch to search.
        element_types: Filter by element types.
        language: Filter by programming language.
        limit: Maximum results.
        include_code: Include source code in results (for detailed inspection).
        brief: Minimal output - just name, type, file, line (for exploration).
        include_tests: Include test results (default True).

    Returns:
        Dict with code_results and test_results lists, grouped by is_test field.
    """
    # Validate limit
    limit = max(1, min(limit, 50))

    # Try vector search first, fall back to keyword search
    results = []
    if embed_client is not None:
        try:
            query_embedding = embed_client.embed_single(query)
            results = es.search_by_vector(
                embedding=query_embedding,
                scope=scope,
                repository=repository,
                username=username,
                element_types=element_types,
                size=limit,
            )
        except Exception:
            pass  # Fall through to keyword search

    # Fallback to keyword search if vector search failed or unavailable
    if not results:
        results = es.search_by_keyword(
            query=query,
            scope=scope,
            repository=repository,
            username=username,
            element_types=element_types,
            size=limit,
        )

    # Group results by is_test
    code_results = []
    test_results = []

    for result in results:
        # Filter by language if specified
        if language and result.get("language") != language:
            continue

        is_test = result.get("is_test", False)

        # Skip tests if not included
        if is_test and not include_tests:
            continue

        # Build qualified name: ClassName.method_name for methods
        name = result.get("name")
        if result.get("element_type") == "method" and result.get("parent_id"):
            # Try to get parent class name from parent_id
            parent_doc = es.get_document(result["parent_id"])
            if parent_doc and parent_doc.get("element_type") == "class":
                name = f"{parent_doc.get('name')}.{name}"

        entry: dict[str, Any] = {
            "name": name,
            "type": result.get("element_type"),
            "file": result.get("relative_path"),
            "line": result.get("line_start"),
            "element_id": result.get("element_id"),
            "is_test": is_test,
        }

        # Brief mode: just the basics for exploration
        if not brief:
            entry["summary"] = result.get("summary", "")
            # Only include signature if present and non-empty
            sig = result.get("signature")
            if sig:
                entry["signature"] = sig
            # Include code if requested
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


def search_features(
    es: ElasticsearchRepository,
    embed_client: CodeEmbeddingClient | None,
    query: str,
    scope: str | None = None,
    repository: str | None = None,
    username: str = "main",
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Search for features/capabilities.

    Tries vector search first, falls back to keyword search if Ollama unavailable.

    Args:
        es: Elasticsearch repository.
        embed_client: Embedding client for query (optional, falls back to keyword).
        query: Search query for features.
        scope: Filter by scope.
        repository: Filter by repository.
        username: User branch.
        limit: Maximum results.

    Returns:
        List of matching features.
    """
    limit = max(1, min(limit, 50))

    # Try vector search first, fall back to keyword search
    results = []
    if embed_client is not None:
        try:
            query_embedding = embed_client.embed_single(query)
            results = es.search_by_vector(
                embedding=query_embedding,
                scope=scope,
                repository=repository,
                username=username,
                element_types=["feature", "subfeature"],
                size=limit,
            )
        except Exception:
            pass  # Fall through to keyword search

    # Fallback to keyword search if vector search failed or unavailable
    if not results:
        results = es.search_by_keyword(
            query=query,
            scope=scope,
            repository=repository,
            username=username,
            element_types=["feature", "subfeature"],
            size=limit,
        )

    formatted = []
    for result in results:
        entry = {
            "label": result.get("cluster_label", result.get("name")),
            "summary": result.get("summary", ""),
            "member_count": result.get("member_count", 0),
            "feature_id": result.get("element_id"),  # For follow-up queries
            "type": result.get("element_type", "feature"),  # "feature" or "subfeature"
        }
        # Include parent feature info for subfeatures
        if result.get("element_type") == "subfeature":
            entry["parent_feature_label"] = result.get("parent_feature_label", "")
            entry["parent_feature_summary"] = result.get("parent_feature_summary", "")
        formatted.append(entry)

    return formatted


def find_similar(
    es: ElasticsearchRepository,
    element_id: str,
    limit: int = 10,
    same_repo_only: bool = False,
) -> list[dict[str, Any]]:
    """Find code elements similar to a given element.

    Args:
        es: Elasticsearch repository.
        element_id: Source element ID.
        limit: Maximum results.
        same_repo_only: Only search within same repository.

    Returns:
        List of similar elements with similarity scores.
    """
    limit = max(1, min(limit, 50))

    # Get source element
    doc = es.get_document(element_id)
    if not doc:
        raise ValueError(f"Element not found: {element_id}")

    embedding = doc.get("embedding")
    if not embedding:
        raise ValueError(f"Element has no embedding: {element_id}")

    # Build search filters
    scope = doc.get("scope") if same_repo_only else None
    repository = doc.get("repository") if same_repo_only else None

    # Search excluding self
    results = es.search_by_vector(
        embedding=embedding,
        scope=scope,
        repository=repository,
        username=doc.get("username", "main"),
        size=limit + 1,  # Get one extra to filter out self
    )

    formatted = []
    for result in results:
        # Skip self
        if result.get("element_id") == element_id:
            continue

        entry: dict[str, Any] = {
            "name": result.get("name"),
            "type": result.get("element_type"),
            "file": result.get("relative_path"),
            "line": result.get("line_start"),
            "summary": result.get("summary", ""),
            "element_id": result.get("element_id"),
        }
        formatted.append(entry)

        if len(formatted) >= limit:
            break

    return formatted


# =============================================================================
# CONTEXT TOOLS
# =============================================================================


def get_element(
    es: ElasticsearchRepository,
    element_id: str,
    include_code: bool = False,
) -> dict[str, Any]:
    """Get full details of a code element.

    Args:
        es: Elasticsearch repository.
        element_id: Element ID.
        include_code: Include raw source code.

    Returns:
        Element details.
    """
    doc = es.get_document(element_id)
    if not doc:
        raise ValueError(f"Element not found: {element_id}")

    result: dict[str, Any] = {
        "name": doc.get("name"),
        "type": doc.get("element_type"),
        "file": doc.get("relative_path"),
        "line_start": doc.get("line_start"),
        "line_end": doc.get("line_end"),
        "summary": doc.get("summary", ""),
    }

    # Only include if present and meaningful
    if doc.get("signature"):
        result["signature"] = doc["signature"]
    if doc.get("docstring"):
        result["docstring"] = doc["docstring"]
    if doc.get("decorators"):
        result["decorators"] = doc["decorators"]
    if doc.get("is_async"):
        result["is_async"] = True
    if doc.get("parent_id"):
        result["parent_id"] = doc["parent_id"]

    if include_code:
        result["code"] = doc.get("raw_code", "")

    return result


def get_context(
    es: ElasticsearchRepository,
    element_id: str,
    include_siblings: bool = False,
    include_children: bool = True,
) -> dict[str, Any]:
    """Get hierarchical context for an element.

    Args:
        es: Elasticsearch repository.
        element_id: Element ID.
        include_siblings: Include sibling elements.
        include_children: Include child elements.

    Returns:
        Hierarchical context.
    """
    doc = es.get_document(element_id)
    if not doc:
        raise ValueError(f"Element not found: {element_id}")

    context: dict[str, Any] = {
        "element": {
            "id": doc.get("element_id"),
            "name": doc.get("name"),
            "type": doc.get("element_type"),
            "file": doc.get("relative_path"),
            "line_start": doc.get("line_start"),
            "line_end": doc.get("line_end"),
            "summary": doc.get("summary", ""),
            "signature": doc.get("signature", ""),
        },
        "file": None,
        "parent": None,
        "siblings": [],
        "children": [],
    }

    scope = doc.get("scope")
    repository = doc.get("repository")
    username = doc.get("username", "main")
    relative_path = doc.get("relative_path")
    parent_id = doc.get("parent_id")

    # Get file context
    file_doc = _find_file_element(es, scope, repository, username, relative_path)
    if file_doc:
        context["file"] = {
            "id": file_doc.get("element_id"),
            "name": file_doc.get("name"),
            "summary": file_doc.get("summary", ""),
        }

    # Get parent context
    if parent_id:
        parent_doc = es.get_document(parent_id)
        if parent_doc:
            context["parent"] = {
                "id": parent_doc.get("element_id"),
                "name": parent_doc.get("name"),
                "type": parent_doc.get("element_type"),
                "summary": parent_doc.get("summary", ""),
                "signature": parent_doc.get("signature", ""),
            }

    # Get siblings
    if include_siblings and parent_id:
        siblings = _find_children(es, parent_id)
        context["siblings"] = [
            {
                "id": s.get("element_id"),
                "name": s.get("name"),
                "type": s.get("element_type"),
                "line": s.get("line_start"),
                "summary": s.get("summary", ""),
            }
            for s in siblings
            if s.get("element_id") != element_id
        ]

    # Get children
    if include_children:
        children = _find_children(es, element_id)
        context["children"] = [
            {
                "id": c.get("element_id"),
                "name": c.get("name"),
                "type": c.get("element_type"),
                "line": c.get("line_start"),
                "summary": c.get("summary", ""),
                "signature": c.get("signature", ""),
            }
            for c in children
        ]

    return context


def get_file_structure(
    es: ElasticsearchRepository,
    scope: str,
    repository: str,
    file_path: str,
    username: str = "main",
) -> dict[str, Any]:
    """Get full structure of a file.

    Args:
        es: Elasticsearch repository.
        scope: Repository scope.
        repository: Repository name.
        file_path: Relative file path.
        username: User branch.

    Returns:
        File structure with nested elements.
    """
    # Get file element
    file_doc = _find_file_element(es, scope, repository, username, file_path)
    if not file_doc:
        raise ValueError(f"File not found: {file_path}")

    # Get all elements in file
    elements = _find_elements_in_file(es, scope, repository, username, file_path)

    # Build tree structure
    def build_tree(parent_id: str | None) -> list[dict]:
        children = []
        for elem in elements:
            if elem.get("parent_id") == parent_id:
                node = {
                    "id": elem.get("element_id"),
                    "name": elem.get("name"),
                    "type": elem.get("element_type"),
                    "line_start": elem.get("line_start"),
                    "line_end": elem.get("line_end"),
                    "summary": elem.get("summary", ""),
                    "signature": elem.get("signature", ""),
                    "children": build_tree(elem.get("element_id")),
                }
                children.append(node)
        return sorted(children, key=lambda x: x.get("line_start", 0))

    file_id = file_doc.get("element_id")

    return {
        "file": {
            "path": file_path,
            "language": file_doc.get("language"),
            "summary": file_doc.get("summary", ""),
            "line_count": file_doc.get("line_end", 0),
        },
        "structure": build_tree(file_id),
        "stats": {
            "classes": sum(1 for e in elements if e.get("element_type") == "class"),
            "functions": sum(1 for e in elements if e.get("element_type") == "function"),
            "methods": sum(1 for e in elements if e.get("element_type") == "method"),
            "total": len(elements),
        },
    }


# =============================================================================
# DISCOVERY TOOLS
# =============================================================================


def list_repos(
    es: ElasticsearchRepository,
    scope: str | None = None,
) -> list[dict[str, Any]]:
    """List all indexed repositories.

    Args:
        es: Elasticsearch repository.
        scope: Filter by scope.

    Returns:
        List of repositories with statistics.
    """
    return es.get_indexed_repositories(scope=scope)


def list_features(
    es: ElasticsearchRepository,
    scope: str,
    repository: str,
    username: str = "main",
) -> list[dict[str, Any]]:
    """List all features and subfeatures for a repository.

    Args:
        es: Elasticsearch repository.
        scope: Repository scope.
        repository: Repository name.
        username: User branch.

    Returns:
        List of features and subfeatures with parent info.
    """
    # Get features
    features = es.get_features(scope, repository, username)
    for f in features:
        f["type"] = "feature"

    # Get subfeatures
    subfeatures = es.get_subfeatures(scope, repository, username)
    for sf in subfeatures:
        sf["type"] = "subfeature"

    # Combine and sort by member count
    all_features = features + subfeatures
    all_features.sort(key=lambda x: x.get("member_count", 0), reverse=True)

    return all_features


def get_repo_stats(
    es: ElasticsearchRepository,
    scope: str,
    repository: str,
    username: str = "main",
) -> dict[str, Any]:
    """Get statistics for a repository.

    Args:
        es: Elasticsearch repository.
        scope: Repository scope.
        repository: Repository name.
        username: User branch.

    Returns:
        Repository statistics.
    """
    return es.get_repository_stats(scope, repository, username)


# =============================================================================
# NAVIGATION TOOLS
# =============================================================================


def get_children(
    es: ElasticsearchRepository,
    element_id: str,
) -> list[dict[str, Any]]:
    """Get child elements of a parent.

    Args:
        es: Elasticsearch repository.
        element_id: Parent element ID.

    Returns:
        List of child elements.
    """
    children = _find_children(es, element_id)
    return [
        {
            "element_id": c.get("element_id"),
            "name": c.get("name"),
            "type": c.get("element_type"),
            "line_start": c.get("line_start"),
            "line_end": c.get("line_end"),
            "summary": c.get("summary", ""),
            "signature": c.get("signature", ""),
        }
        for c in children
    ]


def get_feature_members(
    es: ElasticsearchRepository,
    feature_id: str,
) -> list[dict[str, Any]]:
    """Get all members of a feature or subfeature cluster.

    Args:
        es: Elasticsearch repository.
        feature_id: Feature or subfeature ID.

    Returns:
        List of member elements.
    """
    # Get feature/subfeature document
    feature = es.get_document(feature_id)
    if not feature:
        raise ValueError(f"Feature/subfeature not found: {feature_id}")

    member_ids = feature.get("member_ids", [])
    if not member_ids:
        return []

    # Fetch member documents
    members = []
    for member_id in member_ids:
        doc = es.get_document(member_id)
        if doc:
            members.append({
                "element_id": doc.get("element_id"),
                "name": doc.get("name"),
                "type": doc.get("element_type"),
                "file": doc.get("relative_path"),
                "line": doc.get("line_start"),
                "summary": doc.get("summary", ""),
                "signature": doc.get("signature", ""),
            })

    return members


# =============================================================================
# FILE TOOLS
# =============================================================================


def find_files(
    es: ElasticsearchRepository,
    pattern: str,
    scope: str | None = None,
    repository: str | None = None,
    username: str = "main",
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Find indexed files by glob pattern.

    Searches file elements in Elasticsearch - no filesystem access needed.

    Args:
        es: Elasticsearch repository.
        pattern: Glob pattern (e.g., '**/*.py', 'src/**/*.ts').
        scope: Filter by scope.
        repository: Filter by repository.
        username: User branch to search.
        limit: Maximum files to return.

    Returns:
        List of matching files with basic info.
    """
    import fnmatch

    client = es._get_client()

    # Build ES query for file elements
    filters = [
        {"term": {"element_type": "file"}},
        {"term": {"username": username}},
    ]
    if scope:
        filters.append({"term": {"scope": scope}})
    if repository:
        filters.append({"term": {"repository": repository}})

    # Fetch file elements
    es_result = client.search(
        index="magaldi-code-elements",
        body={
            "query": {"bool": {"filter": filters}},
            "_source": ["element_id", "relative_path", "language", "line_end"],
            "size": min(limit * 5, 2000),  # Get extra to filter by pattern
            "sort": [{"relative_path": "asc"}],
        },
    )

    hits = es_result.get("hits", {}).get("hits", [])
    matches = []

    for hit in hits:
        source = hit["_source"]
        rel_path = source.get("relative_path", "")

        # Apply glob pattern filter
        if fnmatch.fnmatch(rel_path, pattern):
            matches.append({
                "path": rel_path,
                "language": source.get("language"),
                "lines": source.get("line_end", 0),
            })
            if len(matches) >= limit:
                break

    return matches


def batch_get_elements(
    es: ElasticsearchRepository,
    element_ids: list[str],
    include_code: bool = False,
) -> list[dict[str, Any]]:
    """Get multiple elements by ID in one call.

    Args:
        es: Elasticsearch repository.
        element_ids: List of element IDs.
        include_code: Include source code.

    Returns:
        List of elements (preserves order, skips missing).
    """
    results = []
    for eid in element_ids:
        doc = es.get_document(eid)
        if doc:
            entry: dict[str, Any] = {
                "name": doc.get("name"),
                "type": doc.get("element_type"),
                "file": doc.get("relative_path"),
                "line": doc.get("line_start"),
                "summary": doc.get("summary", ""),
                "element_id": eid,
            }
            if doc.get("signature"):
                entry["signature"] = doc["signature"]
            if include_code and doc.get("raw_code"):
                entry["code"] = doc["raw_code"]
            results.append(entry)
    return results


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _find_file_element(
    es: ElasticsearchRepository,
    scope: str,
    repository: str,
    username: str,
    relative_path: str,
) -> dict[str, Any] | None:
    """Find the file element for a given path."""
    client = es._get_client()
    result = client.search(
        index="magaldi-code-elements",
        body={
            "query": {
                "bool": {
                    "filter": [
                        {"term": {"scope": scope}},
                        {"term": {"repository": repository}},
                        {"term": {"username": username}},
                        {"term": {"relative_path": relative_path}},
                        {"term": {"element_type": "file"}},
                    ]
                }
            },
            "size": 1,
        },
    )

    hits = result.get("hits", {}).get("hits", [])
    return hits[0]["_source"] if hits else None


def _find_children(
    es: ElasticsearchRepository,
    parent_id: str,
) -> list[dict[str, Any]]:
    """Find all children of an element."""
    client = es._get_client()
    result = client.search(
        index="magaldi-code-elements",
        body={
            "query": {
                "term": {"parent_id": parent_id}
            },
            "size": 100,
            "sort": [{"line_start": "asc"}],
        },
    )

    return [hit["_source"] for hit in result.get("hits", {}).get("hits", [])]


def _find_elements_in_file(
    es: ElasticsearchRepository,
    scope: str,
    repository: str,
    username: str,
    relative_path: str,
) -> list[dict[str, Any]]:
    """Find all elements in a file."""
    client = es._get_client()
    result = client.search(
        index="magaldi-code-elements",
        body={
            "query": {
                "bool": {
                    "filter": [
                        {"term": {"scope": scope}},
                        {"term": {"repository": repository}},
                        {"term": {"username": username}},
                        {"term": {"relative_path": relative_path}},
                    ],
                    "must_not": [
                        {"term": {"element_type": "file"}},
                    ],
                }
            },
            "size": 500,
            "sort": [{"line_start": "asc"}],
        },
    )

    return [hit["_source"] for hit in result.get("hits", {}).get("hits", [])]


# =============================================================================
# CODE SEARCH TOOLS (regex/grep-based)
# =============================================================================


def grep_code(
    es: ElasticsearchRepository,
    pattern: str,
    scope: str | None = None,
    repository: str | None = None,
    username: str = "main",
    glob: str | None = None,
    context_lines: int = 0,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Search indexed code with regex pattern.

    Searches the raw_code field in Elasticsearch - no filesystem access needed.

    Args:
        es: Elasticsearch repository.
        pattern: Regex pattern to search.
        scope: Filter by scope.
        repository: Filter by repository.
        username: User branch to search.
        glob: File glob filter (e.g., '*.py', '*.ts').
        context_lines: Lines of context before/after match.
        limit: Maximum matches to return.

    Returns:
        List of matches with file, line, content, and context.
    """
    import fnmatch
    import re

    client = es._get_client()
    results: list[dict[str, Any]] = []

    # Build ES query - fetch elements with raw_code
    filters = [
        {"exists": {"field": "raw_code"}},
        {"term": {"username": username}},
    ]
    if scope:
        filters.append({"term": {"scope": scope}})
    if repository:
        filters.append({"term": {"repository": repository}})

    # Fetch candidates from ES (get more than limit since we'll filter by regex)
    es_result = client.search(
        index="magaldi-code-elements",
        body={
            "query": {"bool": {"filter": filters}},
            "_source": ["element_id", "name", "element_type", "relative_path", "line_start", "raw_code"],
            "size": min(limit * 10, 5000),  # Fetch extra to filter
        },
    )

    hits = es_result.get("hits", {}).get("hits", [])
    compiled = re.compile(pattern)

    for hit in hits:
        source = hit["_source"]
        raw_code = source.get("raw_code", "")
        rel_path = source.get("relative_path", "")

        # Apply glob filter if specified
        if glob and not fnmatch.fnmatch(rel_path, glob):
            continue

        # Search for pattern in raw_code
        lines = raw_code.splitlines()
        element_start_line = source.get("line_start", 1)

        for i, line in enumerate(lines):
            match = compiled.search(line)
            if match:
                actual_line = element_start_line + i

                entry: dict[str, Any] = {
                    "file": rel_path,
                    "line": actual_line,
                    "content": line,
                    "match": match.group(0),
                    "element_name": source.get("name"),
                    "element_type": source.get("element_type"),
                }

                # Add context if requested
                if context_lines > 0:
                    start = max(0, i - context_lines)
                    end = min(len(lines), i + context_lines + 1)
                    entry["context_before"] = lines[start:i]
                    entry["context_after"] = lines[i + 1:end]

                results.append(entry)

                if len(results) >= limit:
                    return results

    return results


def find_usages(
    es: ElasticsearchRepository,
    element_id: str,
    limit: int = 30,
) -> list[dict[str, Any]]:
    """Find where an element is used/called/referenced.

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

    # Build search pattern based on element type
    if element_type == "function":
        # Function calls: name(
        pattern = rf"\b{re.escape(name)}\s*\("
    elif element_type == "method":
        # Method calls: .name( or self.name(
        pattern = rf"\.{re.escape(name)}\s*\("
    elif element_type == "class":
        # Class references: inheritance, instantiation, type hints
        pattern = rf"\b{re.escape(name)}\b"
    else:
        # Generic: just the name as word boundary
        pattern = rf"\b{re.escape(name)}\b"

    # Search with grep_code (now uses ES)
    matches = grep_code(
        es=es,
        pattern=pattern,
        scope=scope,
        repository=repository,
        username=username,
        glob="*.py",  # TODO: detect language from element
        context_lines=1,
        limit=limit + 10,  # Get extra to filter out definition
    )

    # Filter out the definition itself
    usages = []
    for match in matches:
        # Skip if it's the definition line
        if match["file"] == defining_file and match["line"] == defining_line:
            continue

        # Skip if it looks like a definition (def/class keyword)
        content = match["content"].strip()
        if element_type == "function" and content.startswith("def "):
            continue
        if element_type == "class" and content.startswith("class "):
            continue
        if element_type == "method" and content.startswith("def "):
            continue

        usages.append({
            "file": match["file"],
            "line": match["line"],
            "content": match["content"],
            "context_before": match.get("context_before", []),
            "context_after": match.get("context_after", []),
        })

        if len(usages) >= limit:
            break

    return usages


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
        username = doc.get("username", "main")
    elif class_name:
        name = class_name
    else:
        raise ValueError("Either element_id or class_name required")

    # Search for class definitions that inherit from this name
    # Pattern: class SomeClass(Name) or class SomeClass(Name, Other)
    pattern = rf"class\s+\w+\s*\([^)]*\b{re.escape(name)}\b"

    matches = grep_code(
        es=es,
        pattern=pattern,
        scope=scope,
        repository=repository,
        username=username,
        glob="*.py",
        context_lines=2,
        limit=limit,
    )

    results = []
    for match in matches:
        # Extract class name from the match
        class_match = re.search(r"class\s+(\w+)", match["content"])
        impl_name = class_match.group(1) if class_match else "Unknown"

        results.append({
            "class_name": impl_name,
            "file": match["file"],
            "line": match["line"],
            "definition": match["content"].strip(),
            "context_after": match.get("context_after", []),
        })

    return results


def generate_skill(
    project_root: str | None = None,
    skill_name: str = "magaldi",
    scope: str = "project",
) -> dict[str, Any]:
    """Generate a SKILL.md file that teaches LLMs how to use this MCP effectively.

    Args:
        project_root: Project root directory (required for scope="project").
        skill_name: Name of the skill (default: "magaldi").
        scope: Where to install - "project" (.claude/skills in project) or "global" (~/.claude/skills).

    Returns:
        Dict with skill content and metadata.
    """
    from pathlib import Path

    skill_content = '''---
name: magaldi
description: >
  ALWAYS use for: grep, find usages, search patterns, find implementations,
  call graphs, find where X is used/called, search code by meaning.
  These tools use the PRE-INDEXED codebase for faster, richer results than raw file search.
  Invoke BEFORE using built-in Grep/Glob/Read tools.
---

# Magaldi Code Discovery

**CRITICAL: Use magaldi tools INSTEAD OF built-in Grep/Glob for code search.**

The codebase is pre-indexed with:
- Semantic embeddings (search by meaning)
- Pre-computed summaries (understand without reading)
- Call graphs (who calls what)
- Feature clustering (related functions grouped)

## When to Use Magaldi vs Built-in Tools

| User Request | USE THIS | NOT THIS |
|--------------|----------|----------|
| "grep for X" / "find pattern X" | `mcp__magaldi__grep_code` | Built-in Grep |
| "find where X is used/called" | `mcp__magaldi__find_usages` | Built-in Grep |
| "search for functions that do X" | `mcp__magaldi__search_code` | Built-in Grep |
| "find files matching *.py" | `mcp__magaldi__find_files` | Built-in Glob |
| "what implements Interface X" | `mcp__magaldi__find_implementations` | Built-in Grep |
| "who calls this function" | `mcp__magaldi__get_call_graph` | Built-in Grep |
| "find similar code to X" | `mcp__magaldi__find_similar` | N/A |
| "what does the codebase do" | `mcp__magaldi__search_features` | N/A |

## Why Magaldi Tools Are Better

| Feature | Magaldi | Built-in Grep/Glob |
|---------|---------|-------------------|
| Pre-indexed | Yes - instant results | No - scans every file |
| Summaries | Every function has AI summary | None |
| Semantic search | "authentication" finds login, auth, verify | Only literal matches |
| Call graphs | Built-in | Must grep manually |
| Context | Parent class, siblings, children | Just file/line |

## Tool Priority (Use in This Order)

### 1. SEMANTIC SEARCH (Start Here for "what does X do")
```
mcp__magaldi__search_code(query="authentication logic", brief=true)
```
- Natural language: "function that validates tokens"
- Returns summaries, not just file:line
- Use `brief=true` for exploration

### 2. PATTERN SEARCH (For literal patterns, regex)
```
mcp__magaldi__grep_code(pattern="\\\\.add_job\\\\(", context_lines=2)
```
- Regex patterns
- Exact string matches
- When you need literal occurrences

### 3. USAGE TRACKING (For "where is X called")
```
mcp__magaldi__find_usages(element_id="...")
```
- After search_code found the element
- Shows all call sites with context
- Filters out definitions automatically

### 4. RELATIONSHIPS (For refactoring, impact analysis)
```
mcp__magaldi__get_call_graph(element_id="...")
mcp__magaldi__find_implementations(class_name="BaseClass")
```
- Before modifying shared code
- Understanding dependencies

## Workflow Examples

### "Grep for X" / "Find pattern X"
```
1. mcp__magaldi__grep_code(pattern="X", context_lines=2)
   - NOT: built-in Grep tool
```

### "Find where function X is called"
```
1. mcp__magaldi__search_code(query="X", element_types=["function"])
2. mcp__magaldi__find_usages(element_id=result.element_id)
   - NOT: grep for "X("
```

### "What implements interface Y"
```
1. mcp__magaldi__find_implementations(class_name="Y")
   - NOT: grep for "class.*Y"
```

### "How does X work"
```
1. mcp__magaldi__search_code(query="X functionality", brief=true)
2. mcp__magaldi__get_element(element_id=best_match, include_code=true)
   - NOT: grep then read file
```

### "Find all authentication code"
```
1. mcp__magaldi__search_features(query="authentication")
2. mcp__magaldi__get_feature_members(feature_id=result.feature_id)
   - Returns grouped, related functions
```

### "Refactor function Z"
```
1. mcp__magaldi__search_code(query="Z")
2. mcp__magaldi__find_usages(element_id)  # Impact analysis
3. mcp__magaldi__get_call_graph(element_id)  # Dependencies
4. THEN make changes
```

## Anti-Patterns (NEVER Do These)

1. **Using built-in Grep instead of magaldi__grep_code**
   - Magaldi grep has indexed context, built-in doesn't

2. **Using built-in Glob instead of magaldi__find_files**
   - Magaldi knows which files are indexed

3. **Grepping for function calls instead of find_usages**
   - find_usages filters definitions, has context

4. **Reading whole files to understand them**
   - Use search_code -> get_element with summaries

5. **Skipping semantic search**
   - Summaries save tokens, embeddings find related code

## Available Tools Quick Reference

| Tool | Purpose |
|------|---------|
| `search_code` | Semantic search by meaning |
| `search_features` | Find high-level capabilities |
| `grep_code` | Regex pattern search (USE THIS not built-in Grep) |
| `find_usages` | Where is this called/used |
| `find_implementations` | What implements this interface |
| `get_call_graph` | Callers and callees |
| `find_similar` | Similar code patterns |
| `get_element` | Full element details |
| `get_context` | Parent, siblings, children |
| `find_files` | Glob pattern search (USE THIS not built-in Glob) |
| `list_features` | All features in repo |
| `get_feature_members` | Functions in a feature |
| `list_repos` | All indexed repos |
| `get_repo_stats` | Repository statistics |

## Remember

The index has already done the hard work:
- Code is parsed and structured
- Summaries explain what code does
- Embeddings enable semantic search
- Call graphs are pre-computed

**Use magaldi tools. Don't re-grep what's already indexed.**
'''

    result = {
        "skill_name": skill_name,
        "content": skill_content,
        "version": "1.0.0",
        "scope": scope,
    }

    # Determine target path based on scope
    if scope == "global":
        skill_dir = Path.home() / ".claude" / "skills" / skill_name
        skill_path = skill_dir / "SKILL.md"
    elif scope == "project":
        if not project_root:
            result["error"] = "project_root is required for scope='project'"
            return result
        skill_dir = Path(project_root) / ".claude" / "skills" / skill_name
        skill_path = skill_dir / "SKILL.md"
    else:
        result["error"] = f"Invalid scope '{scope}'. Use 'project' or 'global'."
        return result

    # Check for existing skill in both locations to avoid duplication
    global_path = Path.home() / ".claude" / "skills" / skill_name / "SKILL.md"
    project_path = Path(project_root) / ".claude" / "skills" / skill_name / "SKILL.md" if project_root else None

    if skill_path.exists():
        result["skipped"] = True
        result["reason"] = f"Skill already exists at: {skill_path}"
        result["path"] = str(skill_path)
        return result

    # Warn if exists in the other location
    if scope == "project" and global_path.exists():
        result["warning"] = f"Note: Skill also exists globally at {global_path}"
    elif scope == "global" and project_path and project_path.exists():
        result["warning"] = f"Note: Skill also exists in project at {project_path}"

    # Write the skill file
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_path.write_text(skill_content)
    result["written_to"] = str(skill_path)

    return result


def get_call_graph(
    es: ElasticsearchRepository,
    element_id: str,
    direction: str = "both",
) -> dict[str, Any]:
    """Get callers and/or callees of a function/method.

    Analyzes indexed code in Elasticsearch - no filesystem access needed.

    Args:
        es: Elasticsearch repository.
        element_id: Function/method element ID.
        direction: 'callers', 'callees', or 'both'.

    Returns:
        Call graph with callers and callees lists.
    """
    doc = es.get_document(element_id)
    if not doc:
        raise ValueError(f"Element not found: {element_id}")

    name = doc.get("name")
    element_type = doc.get("element_type")
    raw_code = doc.get("raw_code", "")
    defining_file = doc.get("relative_path")

    result: dict[str, Any] = {
        "element": {
            "name": name,
            "type": element_type,
            "file": defining_file,
        },
        "callers": [],
        "callees": [],
    }

    # Find callers (who calls this function)
    if direction in ("callers", "both"):
        usages = find_usages(es, element_id, limit=20)
        for usage in usages:
            result["callers"].append({
                "file": usage["file"],
                "line": usage["line"],
                "content": usage["content"],
            })

    # Find callees (what this function calls)
    if direction in ("callees", "both") and raw_code:
        # Extract function/method calls from the code
        import re

        # Match function calls: name( but not def name(
        call_pattern = r"(?<!def\s)(?<!class\s)\b(\w+)\s*\("
        calls = re.findall(call_pattern, raw_code)

        # Deduplicate and filter builtins
        builtins = {"print", "len", "str", "int", "float", "list", "dict", "set",
                    "range", "enumerate", "zip", "map", "filter", "sorted", "type",
                    "isinstance", "hasattr", "getattr", "setattr", "super", "open"}

        seen = set()
        for call_name in calls:
            if call_name in seen or call_name in builtins:
                continue
            seen.add(call_name)

            result["callees"].append({
                "name": call_name,
                "type": "call",
            })

    return result
