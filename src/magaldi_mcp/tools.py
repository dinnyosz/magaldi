"""MCP Tool implementations for Magaldi.

Each tool function takes an ElasticsearchRepository and optional CodeEmbeddingClient,
plus tool-specific parameters, and returns a dict or list result.
"""

from __future__ import annotations

import warnings
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
    glossary_term: str | None = None,
    min_percentage: float = 0.0,
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
        glossary_term: Optional term to filter by. Only returns features where this
            term appears in the feature's members.
        min_percentage: Minimum percentage of feature members that must contain the
            glossary_term (0-100). Default 0 means any occurrence.

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

    # Apply glossary term filter if provided
    if glossary_term is not None:
        glossary_entry = es.get_glossary_term(scope, repository, glossary_term, username)
        if glossary_entry is None:
            # Term not found, return empty results
            return []

        # Build set of feature_ids that meet the min_percentage threshold
        feature_associations = glossary_entry.get("feature_associations", [])
        valid_feature_ids = {
            assoc["feature_id"]
            for assoc in feature_associations
            if assoc.get("percentage", 0) >= min_percentage
        }

        # Filter results to only include features in the valid set
        results = [r for r in results if r.get("element_id") in valid_feature_ids]

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
    include_tests: bool = True,
) -> dict[str, Any]:
    """Find code elements similar to a given element.

    Args:
        es: Elasticsearch repository.
        element_id: Source element ID.
        limit: Maximum results.
        same_repo_only: Only search within same repository.
        include_tests: Include test results (default True).

    Returns:
        Dict with code_results and test_results lists, grouped by is_test field.
    """
    limit = max(1, min(limit, 50))

    # Get source element
    doc = es.get_document(element_id)
    if not doc:
        raise ValueError(f"Element not found: {element_id}")

    # Try summary_embedding first, fall back to code_embedding
    embedding = doc.get("summary_embedding") or doc.get("code_embedding")
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

    # Group results by is_test
    code_results = []
    test_results = []

    for result in results:
        # Skip self
        if result.get("element_id") == element_id:
            continue

        is_test = result.get("is_test", False)

        # Skip tests if not included
        if is_test and not include_tests:
            continue

        entry: dict[str, Any] = {
            "name": result.get("name"),
            "type": result.get("element_type"),
            "file": result.get("relative_path"),
            "line": result.get("line_start"),
            "summary": result.get("summary", ""),
            "element_id": result.get("element_id"),
            "is_test": is_test,
        }

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
) -> dict[str, Any]:
    """Get all members of a feature or subfeature cluster.

    Args:
        es: Elasticsearch repository.
        feature_id: Feature or subfeature ID.

    Returns:
        Dict with 'members' list and 'glossary_terms' list.
    """
    # Get feature/subfeature document
    feature = es.get_document(feature_id)
    if not feature:
        raise ValueError(f"Feature/subfeature not found: {feature_id}")

    member_ids = feature.get("member_ids", [])
    if not member_ids:
        return {"members": [], "glossary_terms": []}

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

    # Parse feature_id to get scope, repository, username
    # Format: scope:repo:username:feature:N or scope:repo:username:subfeature:N
    parts = feature_id.split(":")
    glossary_terms = []
    if len(parts) >= 3:
        scope = parts[0]
        repository = parts[1]
        username = parts[2]

        # Get all glossary terms for this repo
        all_terms = es.get_glossary_terms(scope, repository, username)

        # Filter to terms that have associations with this feature
        for term_entry in all_terms:
            for assoc in term_entry.get("feature_associations", []):
                if assoc.get("feature_id") == feature_id:
                    glossary_terms.append({
                        "term": term_entry.get("term"),
                        "frequency": assoc.get("frequency"),
                        "percentage": assoc.get("percentage"),
                    })
                    break  # Found association for this feature

    return {
        "members": members,
        "glossary_terms": sorted(
            glossary_terms,
            key=lambda x: x.get("percentage", 0),
            reverse=True,
        ),
    }


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
    include_tests: bool = True,
) -> dict[str, Any]:
    """Search indexed code with regex pattern.

    .. deprecated::
        Use `pattern_search` with mode='regexp' instead for better performance.
        pattern_search runs queries server-side. grep_code will be removed in a
        future release.

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
        include_tests: Whether to include test results.

    Returns:
        Dict with code_results, test_results, and totals.
    """
    warnings.warn(
        "grep_code is deprecated. Use pattern_search with mode='regexp' instead. "
        "pattern_search runs queries server-side for better performance.",
        DeprecationWarning,
        stacklevel=2,
    )

    import fnmatch
    import re

    client = es._get_client()
    code_results: list[dict[str, Any]] = []
    test_results: list[dict[str, Any]] = []

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
            "_source": ["element_id", "name", "element_type", "relative_path", "line_start", "raw_code", "is_test"],
            "size": min(limit * 10, 5000),  # Fetch extra to filter
        },
    )

    hits = es_result.get("hits", {}).get("hits", [])
    compiled = re.compile(pattern)
    total_matches = 0

    for hit in hits:
        source = hit["_source"]
        raw_code = source.get("raw_code", "")
        rel_path = source.get("relative_path", "")
        is_test = source.get("is_test", False)

        # Skip tests if include_tests is False
        if not include_tests and is_test:
            continue

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
                    "is_test": is_test,
                }

                # Add context if requested
                if context_lines > 0:
                    start = max(0, i - context_lines)
                    end = min(len(lines), i + context_lines + 1)
                    entry["context_before"] = lines[start:i]
                    entry["context_after"] = lines[i + 1:end]

                # Group by is_test
                if is_test:
                    test_results.append(entry)
                else:
                    code_results.append(entry)

                total_matches += 1
                if total_matches >= limit:
                    return {
                        "code_results": code_results,
                        "test_results": test_results,
                        "total_code": len(code_results),
                        "total_tests": len(test_results),
                    }

    return {
        "code_results": code_results,
        "test_results": test_results,
        "total_code": len(code_results),
        "total_tests": len(test_results),
    }


def _escape_for_lucene_regexp(name: str) -> str:
    """Escape special characters for Lucene regexp syntax.

    Lucene regexp has different special chars than Python regex.
    Key chars to escape: . + * ? ^ $ { } [ ] | ( ) \

    Args:
        name: The name to escape.

    Returns:
        Escaped string safe for Lucene regexp.
    """
    # Lucene regexp special chars that need escaping
    special_chars = r'.\+*?^${}[]|()'
    result = []
    for char in name:
        if char in special_chars:
            result.append('\\')
        result.append(char)
    return ''.join(result)


def find_usages(
    es: ElasticsearchRepository,
    element_id: str,
    limit: int = 30,
) -> list[dict[str, Any]]:
    """Find where an element is used/called/referenced.

    Searches indexed code in Elasticsearch using regexp search - no filesystem access needed.

    Args:
        es: Elasticsearch repository.
        element_id: Element to find usages of.
        limit: Maximum usages to return.

    Returns:
        List of usage locations with context.
    """
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

    # Escape name for Lucene regexp
    escaped_name = _escape_for_lucene_regexp(name)

    # Build Lucene regexp pattern based on element type
    # Note: .* in Lucene matches any string, \\( is literal paren
    if element_type == "function":
        # Function calls: name followed by optional whitespace then paren
        # Lucene: name.*\( (dots match any char, .* matches any string)
        pattern = f"{escaped_name}.*\\("
    elif element_type == "method":
        # Method calls: .name( (dot then name then paren)
        pattern = f"\\.{escaped_name}.*\\("
    elif element_type == "class":
        # Class references: just the name (word boundaries are tricky in Lucene)
        pattern = escaped_name
    else:
        # Generic: just the name
        pattern = escaped_name

    # Search using ES regexp search (server-side)
    results = es.search_by_regexp(
        pattern=pattern,
        scope=scope,
        repository=repository,
        username=username,
        glob="*.py",  # TODO: detect language from element
        size=limit + 10,  # Get extra to filter out definition
        include_tests=True,
    )

    # Filter out the definition itself and process results
    usages = []
    for result in results:
        result_file = result.get("relative_path")
        result_line = result.get("line_start")
        raw_code = result.get("raw_code", "")

        # Skip if it's the defining element itself (same file and line)
        if result_file == defining_file and result_line == defining_line:
            continue

        # Get the first line of raw_code as content
        lines = raw_code.splitlines() if raw_code else []
        content = lines[0] if lines else ""

        # For functions/methods, skip if this element IS a definition of the target
        # (i.e., if the first line defines a function/method with the same name)
        # This catches cases where a function has the same name in different files
        content_stripped = content.strip()
        if element_type == "function":
            # Skip only if this is defining the SAME function
            if content_stripped.startswith(f"def {name}(") or content_stripped.startswith(f"def {name} ("):
                continue
        elif element_type == "class":
            if content_stripped.startswith(f"class {name}(") or content_stripped.startswith(f"class {name}:"):
                continue
        elif element_type == "method":
            if content_stripped.startswith(f"def {name}(") or content_stripped.startswith(f"def {name} ("):
                continue

        # Build context from raw_code lines
        context_before = []  # Empty for now (we only have the element, not surrounding code)
        context_after = lines[1:2] if len(lines) > 1 else []  # Second line if exists

        usages.append({
            "file": result_file,
            "line": result_line,
            "content": content,
            "context_before": context_before,
            "context_after": context_after,
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

    Searches indexed code in Elasticsearch using regexp search - no filesystem access needed.

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

    # Escape name for Lucene regexp
    escaped_name = _escape_for_lucene_regexp(name)

    # Build Lucene regexp pattern for class inheritance
    # Pattern: class.*\(.*{name}.*\) - matches class definitions with name in parents
    # Note: Lucene regexp uses .* for any string, \\( for literal paren
    pattern = f"class.*\\(.*{escaped_name}.*\\)"

    # Search using ES regexp search (server-side)
    results = es.search_by_regexp(
        pattern=pattern,
        scope=scope,
        repository=repository,
        username=username,
        glob="*.py",  # TODO: support other languages
        size=limit,
        include_tests=True,
    )

    # Process results and extract implementing class names
    implementations = []
    for result in results:
        # Skip the base class itself
        if result.get("name") == name:
            continue

        raw_code = result.get("raw_code", "")
        lines = raw_code.splitlines() if raw_code else []
        first_line = lines[0] if lines else ""

        # Extract class name from the first line
        class_match = re.search(r"class\s+(\w+)", first_line)
        impl_name = class_match.group(1) if class_match else "Unknown"

        # Get context (second line if exists)
        context_after = lines[1:3] if len(lines) > 1 else []

        implementations.append({
            "class_name": impl_name,
            "file": result.get("relative_path"),
            "line": result.get("line_start"),
            "definition": first_line.strip(),
            "context_after": context_after,
        })

    return implementations


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
| "grep for X" / "find pattern X" | `mcp__magaldi__pattern_search` (mode="regexp") | Built-in Grep |
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

### 2. PATTERN SEARCH (For literal patterns, regex, wildcards)
```
mcp__magaldi__pattern_search(pattern="add_job.*\\\\(", mode="regexp", scope="...", repository="...")
```
- **Three modes:**
  - `regexp`: Lucene regex (e.g., `"add_column.*Model"`)
  - `wildcard`: Simple wildcards (e.g., `"*column*Model*"`)
  - `proximity`: Terms near each other (e.g., `"add column Model"` with slop=5)
- ES-native - queries run server-side for better performance
- Requires `scope` and `repository` parameters

**Note:** `grep_code` is deprecated - use `pattern_search` with `mode="regexp"` instead.

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
1. mcp__magaldi__pattern_search(pattern="X", mode="regexp", scope="...", repository="...")
   - NOT: built-in Grep tool
   - For wildcards: mode="wildcard" with patterns like "*X*"
   - For proximity: mode="proximity" with slop parameter
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

1. **Using built-in Grep instead of magaldi__pattern_search**
   - Magaldi pattern_search runs queries server-side in Elasticsearch
   - Built-in Grep scans files one by one

2. **Using deprecated grep_code**
   - Use `pattern_search` with `mode="regexp"` instead
   - grep_code is deprecated and will be removed

3. **Using built-in Glob instead of magaldi__find_files**
   - Magaldi knows which files are indexed

4. **Grepping for function calls instead of find_usages**
   - find_usages filters definitions, has context

5. **Reading whole files to understand them**
   - Use search_code -> get_element with summaries

6. **Skipping semantic search**
   - Summaries save tokens, embeddings find related code

## Available Tools Quick Reference

| Tool | Purpose |
|------|---------|
| `search_code` | Semantic search by meaning |
| `search_features` | Find high-level capabilities |
| `pattern_search` | **ES-native pattern search** - regexp, wildcard, or proximity mode |
| `grep_code` | ~~Deprecated~~ - use `pattern_search` with mode="regexp" |
| `find_usages` | Where is this called/used (uses ES regexp internally) |
| `find_implementations` | What implements this interface (uses ES regexp internally) |
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
    defining_file = doc.get("relative_path")
    line_start = doc.get("line_start")
    scope = doc.get("scope")
    repository = doc.get("repository")
    username = doc.get("username", "main")

    result: dict[str, Any] = {
        "element": {
            "name": name,
            "type": element_type,
            "file": defining_file,
            "line": line_start,
            "element_id": element_id,
        },
        "callers": [],
        "callees": [],
    }

    # Find callers (who calls this function) using indexed call data
    if direction in ("callers", "both"):
        callers = es.find_elements_calling(
            target_id=element_id,
            scope=scope,
            repository=repository,
            username=username,
            limit=30,
        )
        for caller in callers:
            result["callers"].append({
                "element_id": caller.get("element_id"),
                "name": caller.get("name"),
                "type": caller.get("element_type"),
                "file": caller.get("relative_path"),
                "line": caller.get("line_start"),
                "summary": caller.get("summary", ""),
            })

    # Find callees (what this function calls) using indexed call data
    if direction in ("callees", "both"):
        calls = es.get_calls(element_id)
        for call in calls:
            callee_entry: dict[str, Any] = {
                "name": call.get("name"),
                "receiver": call.get("receiver"),
                "line": call.get("line"),
            }
            # Include resolved target info if available
            resolved_id = call.get("resolved_id")
            if resolved_id:
                callee_entry["element_id"] = resolved_id
                # Get resolved element details
                resolved_doc = es.get_document(resolved_id)
                if resolved_doc:
                    callee_entry["type"] = resolved_doc.get("element_type")
                    callee_entry["file"] = resolved_doc.get("relative_path")
                    callee_entry["target_line"] = resolved_doc.get("line_start")
                    callee_entry["summary"] = resolved_doc.get("summary", "")
            result["callees"].append(callee_entry)

    return result


def find_callers(
    es: ElasticsearchRepository,
    element_id: str,
    scope: str | None = None,
    repository: str | None = None,
    username: str | None = None,
    limit: int = 30,
    include_tests: bool = True,
) -> dict[str, Any]:
    """Find all functions that call the specified element.

    Uses indexed call data to find callers via calls.resolved_id.

    Args:
        es: Elasticsearch repository.
        element_id: Target element ID to find callers of.
        scope: Filter by scope.
        repository: Filter by repository.
        username: Filter by username branch.
        limit: Maximum results to return.
        include_tests: Whether to include test functions as callers.

    Returns:
        Dict with target info and lists of callers grouped by code/tests.
    """
    # Get target element info
    doc = es.get_document(element_id)
    if not doc:
        raise ValueError(f"Element not found: {element_id}")

    # Use target's scope/repo if not specified
    scope = scope or doc.get("scope")
    repository = repository or doc.get("repository")
    username = username or doc.get("username", "main")

    # Find elements calling the target
    callers = es.find_elements_calling(
        target_id=element_id,
        scope=scope,
        repository=repository,
        username=username,
        limit=limit * 2 if include_tests else limit,  # Get extra to filter
    )

    # Group by is_test
    code_results: list[dict[str, Any]] = []
    test_results: list[dict[str, Any]] = []

    for caller in callers:
        is_test = caller.get("is_test", False)

        # Skip tests if not included
        if is_test and not include_tests:
            continue

        entry = {
            "element_id": caller.get("element_id"),
            "name": caller.get("name"),
            "type": caller.get("element_type"),
            "file": caller.get("relative_path"),
            "line": caller.get("line_start"),
            "summary": caller.get("summary", ""),
            "is_test": is_test,
        }

        if is_test:
            test_results.append(entry)
        else:
            code_results.append(entry)

        # Respect limit
        if len(code_results) + len(test_results) >= limit:
            break

    return {
        "target": {
            "element_id": element_id,
            "name": doc.get("name"),
            "type": doc.get("element_type"),
            "file": doc.get("relative_path"),
            "line": doc.get("line_start"),
        },
        "code_results": code_results[:limit],
        "test_results": test_results[:limit] if include_tests else [],
        "total_code": len(code_results),
        "total_tests": len(test_results) if include_tests else 0,
    }


def find_call_chain(
    es: ElasticsearchRepository,
    element_id: str,
    direction: str = "callees",
    max_depth: int = 5,
    scope: str | None = None,
    repository: str | None = None,
    username: str | None = None,
) -> dict[str, Any]:
    """Trace call chains from an element.

    Args:
        es: Elasticsearch repository.
        element_id: Starting element ID.
        direction: "callees" (what does it call), "callers" (what calls it),
                   or "both".
        max_depth: Maximum depth to traverse (default 5).
        scope: Filter by scope.
        repository: Filter by repository.
        username: Filter by username branch.

    Returns:
        Tree structure with depth levels showing call relationships.
    """
    # Validate max_depth
    max_depth = max(1, min(max_depth, 10))

    # Get root element info
    doc = es.get_document(element_id)
    if not doc:
        raise ValueError(f"Element not found: {element_id}")

    # Use element's scope/repo if not specified
    scope = scope or doc.get("scope")
    repository = repository or doc.get("repository")
    username = username or doc.get("username", "main")

    root_node = {
        "element_id": element_id,
        "name": doc.get("name"),
        "type": doc.get("element_type"),
        "file": doc.get("relative_path"),
        "line": doc.get("line_start"),
        "depth": 0,
    }

    result: dict[str, Any] = {
        "root": root_node,
        "direction": direction,
        "max_depth": max_depth,
    }

    visited: set[str] = {element_id}

    def build_callers_tree(node_id: str, depth: int) -> list[dict[str, Any]]:
        """Recursively build tree of callers."""
        if depth >= max_depth:
            return []

        callers = es.find_elements_calling(
            target_id=node_id,
            scope=scope,
            repository=repository,
            username=username,
            limit=10,  # Limit branching
        )

        children = []
        for caller in callers:
            caller_id = caller.get("element_id")
            if caller_id in visited:
                # Cycle detected, mark but don't recurse
                children.append({
                    "element_id": caller_id,
                    "name": caller.get("name"),
                    "type": caller.get("element_type"),
                    "file": caller.get("relative_path"),
                    "line": caller.get("line_start"),
                    "depth": depth + 1,
                    "cycle": True,
                })
                continue

            visited.add(caller_id)
            node = {
                "element_id": caller_id,
                "name": caller.get("name"),
                "type": caller.get("element_type"),
                "file": caller.get("relative_path"),
                "line": caller.get("line_start"),
                "depth": depth + 1,
                "callers": build_callers_tree(caller_id, depth + 1),
            }
            children.append(node)

        return children

    def build_callees_tree(node_id: str, depth: int) -> list[dict[str, Any]]:
        """Recursively build tree of callees."""
        if depth >= max_depth:
            return []

        calls = es.get_calls(node_id)
        children = []

        for call in calls:
            resolved_id = call.get("resolved_id")
            if not resolved_id:
                # Unresolved call - just record the name
                children.append({
                    "name": call.get("name"),
                    "receiver": call.get("receiver"),
                    "line": call.get("line"),
                    "depth": depth + 1,
                    "unresolved": True,
                })
                continue

            if resolved_id in visited:
                # Cycle detected
                children.append({
                    "element_id": resolved_id,
                    "name": call.get("name"),
                    "line": call.get("line"),
                    "depth": depth + 1,
                    "cycle": True,
                })
                continue

            visited.add(resolved_id)

            # Get resolved element info
            resolved_doc = es.get_document(resolved_id)
            if resolved_doc:
                node = {
                    "element_id": resolved_id,
                    "name": resolved_doc.get("name"),
                    "type": resolved_doc.get("element_type"),
                    "file": resolved_doc.get("relative_path"),
                    "line": resolved_doc.get("line_start"),
                    "depth": depth + 1,
                    "callees": build_callees_tree(resolved_id, depth + 1),
                }
                children.append(node)
            else:
                # Resolved ID but no document found
                children.append({
                    "element_id": resolved_id,
                    "name": call.get("name"),
                    "line": call.get("line"),
                    "depth": depth + 1,
                    "missing": True,
                })

        return children

    if direction in ("callers", "both"):
        result["callers"] = build_callers_tree(element_id, 0)
    if direction in ("callees", "both"):
        # Reset visited for callees direction if doing both
        if direction == "both":
            visited = {element_id}
        result["callees"] = build_callees_tree(element_id, 0)

    return result


def find_dead_code(
    es: ElasticsearchRepository,
    scope: str,
    repository: str,
    username: str | None = None,
    include_tests: bool = False,
) -> dict[str, Any]:
    """Find functions/methods that are never called.

    Excludes:
    - Entry points (decorated with @app.route, @click.command, etc.)
    - Functions named 'main', '__main__'
    - Test functions (if include_tests=False)
    - Magic methods (__init__, __str__, etc.)

    Args:
        es: Elasticsearch repository.
        scope: Repository scope (required).
        repository: Repository name (required).
        username: User branch (defaults to "main").
        include_tests: Whether to include test functions in dead code check.

    Returns:
        Dict with potentially_dead list and statistics.
    """
    username = username or "main"

    # Entry point decorators to exclude
    entry_point_decorators = {
        "app.route", "route", "get", "post", "put", "delete", "patch",
        "click.command", "command", "click.group", "group",
        "pytest.fixture", "fixture",
        "property", "staticmethod", "classmethod",
        "abstractmethod", "abstractproperty",
        "celery.task", "task",
        "api_view", "action",
    }

    # Names to exclude (entry points, magic methods)
    excluded_names = {
        "main", "__main__", "__init__", "__new__", "__del__",
        "__str__", "__repr__", "__hash__", "__eq__", "__ne__",
        "__lt__", "__le__", "__gt__", "__ge__",
        "__add__", "__sub__", "__mul__", "__truediv__",
        "__iter__", "__next__", "__getitem__", "__setitem__",
        "__len__", "__call__", "__enter__", "__exit__",
        "__contains__", "__bool__", "__getattr__", "__setattr__",
        "setUp", "tearDown", "setUpClass", "tearDownClass",
    }

    client = es._get_client()

    # Get all functions and methods in the repository
    filters = [
        {"term": {"scope": scope}},
        {"term": {"repository": repository}},
        {"term": {"username": username}},
        {"terms": {"element_type": ["function", "method"]}},
    ]

    if not include_tests:
        filters.append({"term": {"is_test": False}})

    es_result = client.search(
        index="magaldi-code-elements",
        body={
            "query": {"bool": {"filter": filters}},
            "_source": ["element_id", "hash_id", "name", "element_type", "relative_path",
                        "line_start", "decorators", "is_test", "summary"],
            "size": 2000,
        },
    )

    hits = es_result.get("hits", {}).get("hits", [])

    # Filter out entry points and find uncalled functions
    potentially_dead: list[dict[str, Any]] = []
    excluded_count = 0
    called_count = 0

    for hit in hits:
        source = hit["_source"]
        element_id = source.get("element_id")
        name = source.get("name")
        decorators = source.get("decorators", []) or []

        # Skip excluded names
        if name in excluded_names:
            excluded_count += 1
            continue

        # Skip if has entry point decorator
        is_entry_point = False
        for dec in decorators:
            # Decorator might be "app.route" or just the decorator name in the list
            dec_lower = dec.lower() if isinstance(dec, str) else ""
            for ep in entry_point_decorators:
                if ep in dec_lower or dec_lower.endswith(ep):
                    is_entry_point = True
                    break
            if is_entry_point:
                break

        if is_entry_point:
            excluded_count += 1
            continue

        # Check if anything calls this element
        callers = es.find_elements_calling(
            target_id=element_id,
            scope=scope,
            repository=repository,
            username=username,
            limit=1,  # Just need to know if there's at least one
        )

        if not callers:
            potentially_dead.append({
                "element_id": element_id,
                "hash_id": source.get("hash_id"),
                "name": name,
                "type": source.get("element_type"),
                "file": source.get("relative_path"),
                "line": source.get("line_start"),
                "summary": source.get("summary", ""),
                "is_test": source.get("is_test", False),
            })
        else:
            called_count += 1

    return {
        "potentially_dead": potentially_dead,
        "stats": {
            "total_functions": len(hits),
            "excluded_entry_points": excluded_count,
            "called": called_count,
            "potentially_dead": len(potentially_dead),
        },
    }


def find_entry_points(
    es: ElasticsearchRepository,
    scope: str,
    repository: str,
    username: str | None = None,
) -> dict[str, Any]:
    """Find entry points: HTTP handlers, CLI commands, test fixtures, main functions.

    Detection patterns:
    - Decorator patterns: @app.route, @click.command, @pytest.fixture, etc.
    - Named: main, __main__
    - Called externally with no internal callers

    Args:
        es: Elasticsearch repository.
        scope: Repository scope (required).
        repository: Repository name (required).
        username: User branch (defaults to "main").

    Returns:
        Entry points grouped by type (http, cli, test, main, other).
    """
    username = username or "main"

    # Decorator categories
    http_decorators = {"route", "app.route", "get", "post", "put", "delete", "patch",
                       "api_view", "action", "api.route", "blueprint.route"}
    cli_decorators = {"click.command", "command", "click.group", "group",
                      "click.option", "argument"}
    test_decorators = {"pytest.fixture", "fixture", "pytest.mark"}
    async_decorators = {"celery.task", "task", "dramatiq.actor", "actor"}

    client = es._get_client()

    # Get all functions and methods
    filters = [
        {"term": {"scope": scope}},
        {"term": {"repository": repository}},
        {"term": {"username": username}},
        {"terms": {"element_type": ["function", "method"]}},
    ]

    es_result = client.search(
        index="magaldi-code-elements",
        body={
            "query": {"bool": {"filter": filters}},
            "_source": ["element_id", "hash_id", "name", "element_type", "relative_path",
                        "line_start", "decorators", "is_test", "summary"],
            "size": 2000,
        },
    )

    hits = es_result.get("hits", {}).get("hits", [])

    # Categorize entry points
    http_handlers: list[dict[str, Any]] = []
    cli_commands: list[dict[str, Any]] = []
    test_fixtures: list[dict[str, Any]] = []
    main_functions: list[dict[str, Any]] = []
    async_tasks: list[dict[str, Any]] = []
    other_entry_points: list[dict[str, Any]] = []

    for hit in hits:
        source = hit["_source"]
        name = source.get("name")
        decorators = source.get("decorators", []) or []

        entry = {
            "element_id": source.get("element_id"),
            "hash_id": source.get("hash_id"),
            "name": name,
            "type": source.get("element_type"),
            "file": source.get("relative_path"),
            "line": source.get("line_start"),
            "summary": source.get("summary", ""),
            "decorators": decorators,
        }

        # Check for main function
        if name in ("main", "__main__"):
            main_functions.append(entry)
            continue

        # Check decorators
        decorator_matched = False
        for dec in decorators:
            dec_lower = dec.lower() if isinstance(dec, str) else ""

            # HTTP handlers
            for pattern in http_decorators:
                if pattern in dec_lower:
                    http_handlers.append(entry)
                    decorator_matched = True
                    break

            if decorator_matched:
                break

            # CLI commands
            for pattern in cli_decorators:
                if pattern in dec_lower:
                    cli_commands.append(entry)
                    decorator_matched = True
                    break

            if decorator_matched:
                break

            # Test fixtures
            for pattern in test_decorators:
                if pattern in dec_lower:
                    test_fixtures.append(entry)
                    decorator_matched = True
                    break

            if decorator_matched:
                break

            # Async tasks
            for pattern in async_decorators:
                if pattern in dec_lower:
                    async_tasks.append(entry)
                    decorator_matched = True
                    break

            if decorator_matched:
                break

    return {
        "http": http_handlers,
        "cli": cli_commands,
        "test": test_fixtures,
        "main": main_functions,
        "async_tasks": async_tasks,
        "other": other_entry_points,
        "stats": {
            "total_http": len(http_handlers),
            "total_cli": len(cli_commands),
            "total_test": len(test_fixtures),
            "total_main": len(main_functions),
            "total_async": len(async_tasks),
            "total": len(http_handlers) + len(cli_commands) + len(test_fixtures) +
                    len(main_functions) + len(async_tasks),
        },
    }


# =============================================================================
# PATTERN SEARCH TOOLS
# =============================================================================


def pattern_search(
    es: ElasticsearchRepository,
    pattern: str,
    mode: str,
    scope: str,
    repository: str,
    username: str | None = None,
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
        username: User branch to search (optional).
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


# =============================================================================
# GLOSSARY TOOLS
# =============================================================================


def list_glossary(
    es: ElasticsearchRepository,
    scope: str,
    repository: str,
    username: str = "main",
    min_count: int = 1,
) -> list[dict[str, Any]]:
    """List all glossary terms for a repository.

    Glossary terms are domain concepts extracted from element names
    (e.g., 'user', 'email', 'order') that appear across the codebase.

    Args:
        es: Elasticsearch repository.
        scope: Repository scope.
        repository: Repository name.
        username: User branch (default: main).
        min_count: Minimum occurrence count to include (default: 1).

    Returns:
        List of glossary terms sorted by count, each with:
        - term: The domain term
        - total_count: Number of elements containing this term
        - file_paths: Files where term appears
        - feature_associations: Linked features (if glossary was linked)
    """
    return es.get_glossary_terms(
        scope=scope,
        repository=repository,
        username=username,
        min_count=min_count,
    )


def get_glossary_term(
    es: ElasticsearchRepository,
    scope: str,
    repository: str,
    term: str,
    username: str = "main",
) -> dict[str, Any] | None:
    """Get full details for a specific glossary term.

    Args:
        es: Elasticsearch repository.
        scope: Repository scope.
        repository: Repository name.
        term: The glossary term to retrieve.
        username: User branch (default: main).

    Returns:
        Glossary entry with:
        - term: The domain term
        - total_count: Number of elements containing this term
        - element_ids: List of element IDs containing this term
        - file_paths: Files where term appears
        - feature_associations: Linked features with frequency/percentage
        Or None if term not found.
    """
    return es.get_glossary_term(
        scope=scope,
        repository=repository,
        term=term,
        username=username,
    )


def search_glossary(
    es: ElasticsearchRepository,
    scope: str,
    repository: str,
    query: str,
    username: str = "main",
) -> list[dict[str, Any]]:
    """Search glossary terms by partial match.

    Finds terms that contain the query string. For example,
    searching 'user' might return 'user', 'username', 'userid'.

    Args:
        es: Elasticsearch repository.
        scope: Repository scope.
        repository: Repository name.
        query: Partial term to search for.
        username: User branch (default: main).

    Returns:
        List of matching glossary terms with term and total_count.
    """
    return es.search_glossary(
        scope=scope,
        repository=repository,
        query=query,
        username=username,
    )


# =============================================================================
# DEPENDENCY ANALYSIS TOOLS
# =============================================================================


def find_dependencies(
    es: ElasticsearchRepository,
    file_path: str | None = None,
    element_id: str | None = None,
    scope: str | None = None,
    repository: str | None = None,
    username: str | None = None,
) -> dict[str, Any]:
    """Get imports for a file.

    Shows what a file depends on - both internal (from within the repo)
    and external (third-party packages) imports.

    Args:
        es: Elasticsearch repository.
        file_path: Relative file path (e.g., "src/utils.py").
        element_id: Or provide file element ID directly.
        scope: Filter by scope (required if using file_path).
        repository: Filter by repository (required if using file_path).
        username: User branch (defaults to "main").

    Returns:
        Dict with:
        - internal_imports: Imports from within the repo
        - external_imports: Imports from external packages
        - all_imports: Combined list
        - file_info: File element details

    Raises:
        ValueError: If neither file_path nor element_id provided, or file not found.
    """
    username = username or "main"

    # Get the file element
    if element_id:
        file_doc = es.get_document(element_id)
        if not file_doc:
            raise ValueError(f"Element not found: {element_id}")
        if file_doc.get("element_type") != "file":
            raise ValueError(f"Element is not a file: {element_id}")
        scope = file_doc.get("scope")
        repository = file_doc.get("repository")
    elif file_path:
        if not scope or not repository:
            raise ValueError("scope and repository required when using file_path")
        file_doc = _find_file_element(es, scope, repository, username, file_path)
        if not file_doc:
            raise ValueError(f"File not found: {file_path}")
        element_id = file_doc.get("element_id")
    else:
        raise ValueError("Either file_path or element_id required")

    # Get imports from the file element
    imports = es.get_imports(element_id)

    # Classify imports as internal vs external
    internal_imports: list[dict[str, Any]] = []
    external_imports: list[dict[str, Any]] = []

    # Get all file paths in the repo to detect internal imports
    client = es._get_client()
    file_result = client.search(
        index="magaldi-code-elements",
        body={
            "query": {
                "bool": {
                    "filter": [
                        {"term": {"scope": scope}},
                        {"term": {"repository": repository}},
                        {"term": {"username": username}},
                        {"term": {"element_type": "file"}},
                    ]
                }
            },
            "_source": ["relative_path"],
            "size": 2000,
        },
    )
    repo_files = {
        hit["_source"]["relative_path"] for hit in file_result.get("hits", {}).get("hits", [])
    }

    # Also create a set of potential module names from file paths
    # e.g., "src/utils.py" -> "src.utils", "utils"
    repo_modules: set[str] = set()
    for path in repo_files:
        # Remove .py extension and convert to module name
        if path.endswith(".py"):
            module_path = path[:-3].replace("/", ".").replace("\\", ".")
            repo_modules.add(module_path)
            # Also add the last component
            parts = module_path.split(".")
            if parts:
                repo_modules.add(parts[-1])

    for imp in imports:
        module = imp.get("module", "")
        name = imp.get("name", "")

        import_entry = {
            "name": name,
            "module": module,
            "alias": imp.get("alias"),
            "line": imp.get("line"),
        }

        # Check if this is an internal import
        is_internal = False

        # Check if module starts with "." (relative import)
        if module and module.startswith("."):
            is_internal = True
        # Check if module matches a file in the repo
        elif module in repo_modules:
            is_internal = True
        # Check common patterns for internal imports
        elif module:
            # Check if any part of the module path matches a repo module
            parts = module.split(".")
            for i in range(len(parts)):
                partial = ".".join(parts[:i + 1])
                if partial in repo_modules:
                    is_internal = True
                    break

        if is_internal:
            internal_imports.append(import_entry)
        else:
            external_imports.append(import_entry)

    return {
        "file_info": {
            "path": file_doc.get("relative_path"),
            "element_id": element_id,
        },
        "internal_imports": internal_imports,
        "external_imports": external_imports,
        "all_imports": imports,
        "stats": {
            "total": len(imports),
            "internal": len(internal_imports),
            "external": len(external_imports),
        },
    }


def find_dependents(
    es: ElasticsearchRepository,
    module: str,
    scope: str,
    repository: str,
    username: str | None = None,
    limit: int = 50,
) -> dict[str, Any]:
    """Find files that import a module.

    Shows what depends on a given module - useful for impact analysis
    before making changes to a module.

    Args:
        es: Elasticsearch repository.
        module: Module name to search for (e.g., "utils", "shared.config", "./utils").
        scope: Repository scope (required).
        repository: Repository name (required).
        username: User branch (defaults to "main").
        limit: Maximum files to return.

    Returns:
        Dict with:
        - module: The searched module
        - dependents: List of files that import the module
        - total: Number of dependents found
    """
    username = username or "main"

    # Find files that import this module
    dependents = es.find_elements_importing(
        module=module,
        scope=scope,
        repository=repository,
        username=username,
        limit=limit,
    )

    formatted_dependents = []
    for dep in dependents:
        formatted_dependents.append({
            "file": dep.get("relative_path"),
            "element_id": dep.get("element_id"),
            "language": dep.get("language"),
        })

    return {
        "module": module,
        "dependents": formatted_dependents,
        "total": len(formatted_dependents),
    }


def dependency_graph(
    es: ElasticsearchRepository,
    scope: str,
    repository: str,
    username: str | None = None,
    internal_only: bool = True,
) -> dict[str, Any]:
    """Build a module-level dependency graph.

    Creates a directed graph of module dependencies within the repository.
    Useful for understanding code architecture and detecting circular dependencies.

    Args:
        es: Elasticsearch repository.
        scope: Repository scope (required).
        repository: Repository name (required).
        username: User branch (defaults to "main").
        internal_only: Only include internal imports (default True).

    Returns:
        Dict with:
        - nodes: List of module names (files)
        - edges: List of {from, to} dependency links
        - cycles: List of circular dependency chains (if any)
        - stats: Graph statistics
    """
    username = username or "main"

    client = es._get_client()

    # Get all file elements with their imports
    result = client.search(
        index="magaldi-code-elements",
        body={
            "query": {
                "bool": {
                    "filter": [
                        {"term": {"scope": scope}},
                        {"term": {"repository": repository}},
                        {"term": {"username": username}},
                        {"term": {"element_type": "file"}},
                    ]
                }
            },
            "_source": ["element_id", "relative_path", "imports"],
            "size": 2000,
        },
    )

    hits = result.get("hits", {}).get("hits", [])

    # Build set of all file paths for internal import detection
    file_paths = {hit["_source"]["relative_path"] for hit in hits}

    # Build module name mapping from file paths
    path_to_module: dict[str, str] = {}
    module_to_path: dict[str, str] = {}

    for path in file_paths:
        if path.endswith(".py"):
            # Convert path to module name
            module_name = path[:-3].replace("/", ".").replace("\\", ".")
            path_to_module[path] = module_name
            module_to_path[module_name] = path
            # Also map just the filename without extension
            parts = module_name.split(".")
            if parts:
                module_to_path[parts[-1]] = path

    # Build nodes and edges
    nodes: list[str] = list(file_paths)
    edges: list[dict[str, str]] = []
    edge_set: set[tuple[str, str]] = set()

    for hit in hits:
        source = hit["_source"]
        from_path = source.get("relative_path")
        imports = source.get("imports", []) or []

        for imp in imports:
            module = imp.get("module", "")
            if not module:
                continue

            # Determine the target path
            to_path = None

            # Handle relative imports
            if module.startswith("."):
                # Relative import - resolve relative to current file
                from_dir = "/".join(from_path.split("/")[:-1])
                rel_module = module.lstrip(".")
                if rel_module:
                    # e.g., ".utils" from "src/main.py" -> "src/utils.py"
                    if from_dir:
                        to_path = f"{from_dir}/{rel_module.replace('.', '/')}.py"
                    else:
                        to_path = f"{rel_module.replace('.', '/')}.py"
                else:
                    # Just "." means current package's __init__.py
                    if from_dir:
                        to_path = f"{from_dir}/__init__.py"
            else:
                # Absolute import - try to find matching file
                # Try direct module path
                possible_paths = [
                    f"{module.replace('.', '/')}.py",
                    f"{module.replace('.', '/')}/__init__.py",
                ]
                for pp in possible_paths:
                    if pp in file_paths:
                        to_path = pp
                        break

                # Also check if module name directly maps to a known module
                if not to_path and module in module_to_path:
                    to_path = module_to_path[module]

            # Skip if target not found or if filtering external imports
            if not to_path or to_path not in file_paths:
                if internal_only:
                    continue
                # For external imports when internal_only=False, we'd need to handle differently
                # For now, skip external imports entirely
                continue

            # Add edge if not already present
            edge_key = (from_path, to_path)
            if edge_key not in edge_set:
                edge_set.add(edge_key)
                edges.append({"from": from_path, "to": to_path})

    # Detect cycles using DFS
    cycles = _detect_cycles(nodes, edges)

    return {
        "nodes": nodes,
        "edges": edges,
        "cycles": cycles,
        "stats": {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "cycle_count": len(cycles),
            "has_cycles": len(cycles) > 0,
        },
    }


def _detect_cycles(nodes: list[str], edges: list[dict[str, str]]) -> list[list[str]]:
    """Detect circular dependencies in a directed graph.

    Uses DFS to find cycles.

    Args:
        nodes: List of node names.
        edges: List of {from, to} edges.

    Returns:
        List of cycles, where each cycle is a list of nodes forming the cycle.
    """
    # Build adjacency list
    graph: dict[str, list[str]] = {node: [] for node in nodes}
    for edge in edges:
        from_node = edge["from"]
        to_node = edge["to"]
        if from_node in graph:
            graph[from_node].append(to_node)

    cycles: list[list[str]] = []
    visited: set[str] = set()
    rec_stack: set[str] = set()
    path: list[str] = []

    def dfs(node: str) -> None:
        visited.add(node)
        rec_stack.add(node)
        path.append(node)

        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                dfs(neighbor)
            elif neighbor in rec_stack:
                # Found a cycle
                cycle_start = path.index(neighbor)
                cycle = path[cycle_start:] + [neighbor]
                # Avoid duplicate cycles (by normalizing)
                normalized = _normalize_cycle(cycle)
                if normalized not in [_normalize_cycle(c) for c in cycles]:
                    cycles.append(cycle)

        path.pop()
        rec_stack.remove(node)

    for node in nodes:
        if node not in visited:
            dfs(node)

    return cycles


def _normalize_cycle(cycle: list[str]) -> tuple[str, ...]:
    """Normalize a cycle for comparison by starting from the lexicographically smallest element."""
    if not cycle:
        return ()
    # Remove the duplicate end element if present
    if len(cycle) > 1 and cycle[0] == cycle[-1]:
        cycle = cycle[:-1]
    # Find the smallest element's index
    min_idx = cycle.index(min(cycle))
    # Rotate to start from smallest
    normalized = cycle[min_idx:] + cycle[:min_idx]
    return tuple(normalized)


# =============================================================================
# META TOOLS
# =============================================================================


def explain_element(
    es: ElasticsearchRepository,
    element_id: str,
) -> dict[str, Any]:
    """Comprehensive overview of a code element.

    Provides a complete picture of an element including:
    - Basic info (name, type, file, signature, summary)
    - Who calls it (callers)
    - What it calls (callees)
    - Imports (if file element)
    - Similar code (top 3 similar)
    - Parent context (class or file)

    Args:
        es: Elasticsearch repository.
        element_id: Element ID to explain.

    Returns:
        Dict with element, callers, callees, imports, similar_code, parent.

    Raises:
        ValueError: If element not found.
    """
    doc = es.get_document(element_id)
    if not doc:
        raise ValueError(f"Element not found: {element_id}")

    name = doc.get("name")
    element_type = doc.get("element_type")
    relative_path = doc.get("relative_path")
    line_start = doc.get("line_start")
    scope = doc.get("scope")
    repository = doc.get("repository")
    username = doc.get("username", "main")

    result: dict[str, Any] = {
        "element": {
            "element_id": element_id,
            "name": name,
            "type": element_type,
            "file": relative_path,
            "line": line_start,
            "signature": doc.get("signature"),
            "summary": doc.get("summary", ""),
            "docstring": doc.get("docstring"),
            "decorators": doc.get("decorators", []),
            "is_test": doc.get("is_test", False),
        },
        "callers": [],
        "callees": [],
        "imports": [],
        "similar_code": [],
        "parent": None,
    }

    # Get callers (who calls this function/method)
    if element_type in ("function", "method"):
        callers = es.find_elements_calling(
            target_id=element_id,
            scope=scope,
            repository=repository,
            username=username,
            limit=5,  # Top 5 callers
        )
        for caller in callers:
            result["callers"].append({
                "element_id": caller.get("element_id"),
                "name": caller.get("name"),
                "type": caller.get("element_type"),
                "file": caller.get("relative_path"),
                "line": caller.get("line_start"),
                "summary": caller.get("summary", ""),
            })

        # Get callees (what this function/method calls)
        calls = es.get_calls(element_id)
        for call in calls:
            callee_entry: dict[str, Any] = {
                "name": call.get("name"),
                "receiver": call.get("receiver"),
                "line": call.get("line"),
            }
            resolved_id = call.get("resolved_id")
            if resolved_id:
                callee_entry["element_id"] = resolved_id
                resolved_doc = es.get_document(resolved_id)
                if resolved_doc:
                    callee_entry["type"] = resolved_doc.get("element_type")
                    callee_entry["file"] = resolved_doc.get("relative_path")
                    callee_entry["target_line"] = resolved_doc.get("line_start")
                    callee_entry["summary"] = resolved_doc.get("summary", "")
            result["callees"].append(callee_entry)

    # Get imports (if file element)
    if element_type == "file":
        imports = es.get_imports(element_id)
        result["imports"] = imports

    # Get similar code (top 3)
    summary_embedding = doc.get("summary_embedding")
    if summary_embedding:
        similar_results = es.search_similar(
            embedding=summary_embedding,
            embedding_type="summary",
            scope=scope,
            repository=repository,
            username=username,
            size=4,  # Get 4 to filter out self
            include_tests=True,
        )
        for sim in similar_results:
            if sim.get("element_id") == element_id:
                continue  # Skip self
            result["similar_code"].append({
                "element_id": sim.get("element_id"),
                "name": sim.get("name"),
                "type": sim.get("element_type"),
                "file": sim.get("relative_path"),
                "line": sim.get("line_start"),
                "summary": sim.get("summary", ""),
                "similarity": sim.get("score", 0),
            })
            if len(result["similar_code"]) >= 3:
                break

    # Get parent context
    parent_id = doc.get("parent_id")
    if parent_id:
        parent_doc = es.get_document(parent_id)
        if parent_doc:
            result["parent"] = {
                "element_id": parent_id,
                "name": parent_doc.get("name"),
                "type": parent_doc.get("element_type"),
                "file": parent_doc.get("relative_path"),
                "line": parent_doc.get("line_start"),
                "summary": parent_doc.get("summary", ""),
            }

    return result
