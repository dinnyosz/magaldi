"""Browse API routes for code element exploration."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from magaldi_web.dependencies import get_es_repository
from shared.db.elasticsearch import ElasticsearchRepository, INDEX_NAME

router = APIRouter()


@router.get("/browse/elements")
async def browse_elements(
    scope: str | None = None,
    repository: str | None = None,
    username: str = "main",
    element_type: str | None = None,
    parent_id: str | None = None,
    language: str | None = None,
    page: int = Query(default=1, ge=1),
    limit: int = Query(default=50, ge=1, le=200),
    es_repo: ElasticsearchRepository = Depends(get_es_repository),
) -> dict:
    """Browse code elements with filtering.

    Args:
        scope: Filter by repository scope.
        repository: Filter by repository name.
        username: User branch (default: main).
        element_type: Filter by element type (file, class, function, method, variable, constant).
        parent_id: Filter by parent element ID (for hierarchy).
        language: Filter by programming language.
        page: Page number (1-indexed).
        limit: Results per page.

    Returns:
        Paginated list of elements with metadata.
    """
    client = es_repo._get_client()

    # Build filters
    filters = []

    # Username filter - include both main and user branch
    usernames = ["main"]
    if username and username != "main":
        usernames.append(username)
    filters.append({"terms": {"username": usernames}})

    if scope:
        filters.append({"term": {"scope": scope}})
    if repository:
        filters.append({"term": {"repository": repository}})
    if element_type:
        filters.append({"term": {"element_type": element_type}})
    if parent_id:
        filters.append({"term": {"parent_id": parent_id}})
    if language:
        filters.append({"term": {"language": language}})

    # Calculate offset
    offset = (page - 1) * limit

    # Sort by file path and line number for consistent ordering
    sort = [
        {"relative_path": {"order": "asc"}},
        {"line_start": {"order": "asc"}},
    ]

    # For files, sort alphabetically by path
    if element_type == "file":
        sort = [{"relative_path": {"order": "asc"}}]

    # Execute search
    result = client.search(
        index=INDEX_NAME,
        body={
            "size": limit,
            "from": offset,
            "query": {
                "bool": {
                    "filter": filters,
                },
            },
            "sort": sort,
            "_source": [
                "element_id",
                "name",
                "element_type",
                "relative_path",
                "line_start",
                "line_end",
                "language",
                "summary",
                "signature",
                "repository",
                "scope",
                "parent_id",
                "visibility",
                "is_async",
                "docstring",
            ],
        },
    )

    # Process results
    hits = result.get("hits", {})
    total = hits.get("total", {}).get("value", 0)

    elements = []
    for hit in hits.get("hits", []):
        source = hit["_source"]
        elements.append({
            "element_id": source["element_id"],
            "name": source["name"],
            "element_type": source["element_type"],
            "file_path": source.get("relative_path", ""),
            "line_start": source.get("line_start", 0),
            "line_end": source.get("line_end"),
            "language": source.get("language", ""),
            "summary": source.get("summary"),
            "signature": source.get("signature"),
            "repository": source["repository"],
            "scope": source["scope"],
            "parent_id": source.get("parent_id"),
            "visibility": source.get("visibility"),
            "is_async": source.get("is_async", False),
            "has_docstring": bool(source.get("docstring")),
        })

    return {
        "elements": elements,
        "total": total,
        "page": page,
        "limit": limit,
        "total_pages": (total + limit - 1) // limit if total > 0 else 0,
    }


@router.get("/browse/element/{element_id}/children")
async def get_element_children(
    element_id: str,
    username: str = "main",
    es_repo: ElasticsearchRepository = Depends(get_es_repository),
) -> dict:
    """Get child elements of a parent element.

    Args:
        element_id: Parent element ID.
        username: User branch.

    Returns:
        List of child elements grouped by type.
    """
    from urllib.parse import unquote
    element_id = unquote(element_id)

    client = es_repo._get_client()

    # Build filters
    usernames = ["main"]
    if username and username != "main":
        usernames.append(username)

    # Search for children of this element
    result = client.search(
        index=INDEX_NAME,
        body={
            "size": 200,
            "query": {
                "bool": {
                    "filter": [
                        {"terms": {"username": usernames}},
                        {"term": {"parent_id": element_id}},
                    ],
                },
            },
            "sort": [
                {"element_type": {"order": "asc"}},
                {"line_start": {"order": "asc"}},
            ],
            "_source": [
                "element_id",
                "name",
                "element_type",
                "line_start",
                "line_end",
                "summary",
                "signature",
                "visibility",
                "is_async",
            ],
        },
    )

    # Group children by type
    children_by_type: dict[str, list] = {}
    for hit in result.get("hits", {}).get("hits", []):
        source = hit["_source"]
        elem_type = source["element_type"]

        if elem_type not in children_by_type:
            children_by_type[elem_type] = []

        children_by_type[elem_type].append({
            "element_id": source["element_id"],
            "name": source["name"],
            "element_type": elem_type,
            "line_start": source.get("line_start", 0),
            "line_end": source.get("line_end"),
            "summary": source.get("summary"),
            "signature": source.get("signature"),
            "visibility": source.get("visibility"),
            "is_async": source.get("is_async", False),
        })

    # Order type keys in a logical order
    type_order = ["class", "function", "method", "variable", "constant"]
    ordered_children = {}
    for t in type_order:
        if t in children_by_type:
            ordered_children[t] = children_by_type[t]
    # Add any remaining types
    for t, children in children_by_type.items():
        if t not in ordered_children:
            ordered_children[t] = children

    return {
        "element_id": element_id,
        "children": ordered_children,
        "total_children": sum(len(c) for c in children_by_type.values()),
    }


@router.get("/browse/filters")
async def get_browse_filters(
    es_repo: ElasticsearchRepository = Depends(get_es_repository),
) -> dict:
    """Get available filter options for browsing.

    Returns:
        Available scopes, repositories, element types, languages, and usernames.
    """
    client = es_repo._get_client()

    result = client.search(
        index=INDEX_NAME,
        body={
            "size": 0,
            "aggs": {
                "scopes": {"terms": {"field": "scope", "size": 50}},
                "repositories": {
                    "composite": {
                        "size": 100,
                        "sources": [
                            {"scope": {"terms": {"field": "scope"}}},
                            {"repository": {"terms": {"field": "repository"}}},
                        ],
                    },
                },
                "element_types": {"terms": {"field": "element_type", "size": 20}},
                "languages": {"terms": {"field": "language", "size": 30}},
                "usernames": {"terms": {"field": "username", "size": 50}},
            },
        },
    )

    aggs = result.get("aggregations", {})

    # Build repo list with scope prefix
    repos = []
    for bucket in aggs.get("repositories", {}).get("buckets", []):
        repos.append({
            "scope": bucket["key"]["scope"],
            "repository": bucket["key"]["repository"],
        })

    return {
        "scopes": [b["key"] for b in aggs.get("scopes", {}).get("buckets", [])],
        "repositories": repos,
        "element_types": [b["key"] for b in aggs.get("element_types", {}).get("buckets", [])],
        "languages": [b["key"] for b in aggs.get("languages", {}).get("buckets", [])],
        "usernames": [b["key"] for b in aggs.get("usernames", {}).get("buckets", [])],
    }


@router.get("/browse/stats")
async def get_browse_stats(
    scope: str | None = None,
    repository: str | None = None,
    username: str = "main",
    es_repo: ElasticsearchRepository = Depends(get_es_repository),
) -> dict:
    """Get element counts by type for the current filters.

    Args:
        scope: Filter by repository scope.
        repository: Filter by repository name.
        username: User branch.

    Returns:
        Counts for each element type.
    """
    client = es_repo._get_client()

    # Build filters
    filters = []
    usernames = ["main"]
    if username and username != "main":
        usernames.append(username)
    filters.append({"terms": {"username": usernames}})

    if scope:
        filters.append({"term": {"scope": scope}})
    if repository:
        filters.append({"term": {"repository": repository}})

    result = client.search(
        index=INDEX_NAME,
        body={
            "size": 0,
            "query": {"bool": {"filter": filters}},
            "aggs": {
                "by_type": {"terms": {"field": "element_type", "size": 20}},
                "by_language": {"terms": {"field": "language", "size": 20}},
            },
        },
    )

    aggs = result.get("aggregations", {})

    type_counts = {
        bucket["key"]: bucket["doc_count"]
        for bucket in aggs.get("by_type", {}).get("buckets", [])
    }

    language_counts = {
        bucket["key"]: bucket["doc_count"]
        for bucket in aggs.get("by_language", {}).get("buckets", [])
    }

    return {
        "type_counts": type_counts,
        "language_counts": language_counts,
        "total": sum(type_counts.values()),
    }
