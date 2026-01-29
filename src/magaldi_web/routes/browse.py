"""Browse API routes for code element exploration."""

from __future__ import annotations

import re
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
                "hash_id",
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
                "decorators",
                "level",
                "member_count",  # For features/subfeatures
                "cluster_label",  # For features/subfeatures
                "description",  # For glossary
                "total_count",  # For glossary
                "feature_count",  # For glossary
            ],
        },
    )

    # Process results
    hits = result.get("hits", {})
    total = hits.get("total", {}).get("value", 0)

    # Collect parent_ids to resolve container names
    parent_ids = set()
    for hit in hits.get("hits", []):
        pid = hit["_source"].get("parent_id")
        if pid:
            parent_ids.add(pid)

    # Batch fetch parent info
    parent_info: dict[str, dict] = {}
    if parent_ids:
        parent_result = client.mget(
            index=INDEX_NAME,
            ids=list(parent_ids),
            _source=["name", "element_type", "relative_path", "hash_id"],
        )
        for doc in parent_result.get("docs", []):
            if doc.get("found") and doc.get("_source"):
                parent_info[doc["_id"]] = doc["_source"]

    # Collect feature labels to fetch subfeatures
    feature_labels: dict[str, str] = {}  # element_id -> cluster_label
    for hit in hits.get("hits", []):
        source = hit["_source"]
        if source.get("element_type") == "feature":
            feature_labels[source["element_id"]] = source.get("cluster_label") or source.get("name")

    # Batch fetch subfeatures for features
    subfeatures_by_feature: dict[str, list[dict]] = {}
    if feature_labels:
        # Get a sample feature to know scope/repo/username
        sample_hit = next((h for h in hits.get("hits", []) if h["_source"].get("element_type") == "feature"), None)
        if sample_hit:
            sample_src = sample_hit["_source"]
            # Query all subfeatures for these features
            subfeature_result = client.search(
                index=INDEX_NAME,
                body={
                    "size": 1000,
                    "query": {
                        "bool": {
                            "filter": [
                                {"term": {"scope": sample_src.get("scope")}},
                                {"term": {"repository": sample_src.get("repository")}},
                                {"term": {"username": sample_src.get("username", "main")}},
                                {"term": {"element_type": "subfeature"}},
                                {"terms": {"parent_feature_label": list(feature_labels.values())}},
                            ],
                        },
                    },
                    "_source": ["element_id", "hash_id", "cluster_label", "member_count", "parent_feature_label"],
                },
            )
            for sf_hit in subfeature_result.get("hits", {}).get("hits", []):
                sf_src = sf_hit["_source"]
                parent_label = sf_src.get("parent_feature_label")
                # Find which feature this belongs to
                for fid, flabel in feature_labels.items():
                    if flabel == parent_label:
                        if fid not in subfeatures_by_feature:
                            subfeatures_by_feature[fid] = []
                        subfeatures_by_feature[fid].append({
                            "element_id": sf_src["element_id"],
                            "hash_id": sf_src.get("hash_id"),
                            "label": sf_src.get("cluster_label"),
                            "member_count": sf_src.get("member_count", 0),
                        })
                        break

    elements = []
    for hit in hits.get("hits", []):
        source = hit["_source"]
        parent_id = source.get("parent_id")

        # Build container info from parent
        container = None
        if parent_id and parent_id in parent_info:
            pinfo = parent_info[parent_id]
            container = {
                "element_id": parent_id,
                "hash_id": pinfo.get("hash_id"),
                "name": pinfo.get("name"),
                "element_type": pinfo.get("element_type"),
            }

        # For glossary elements, use description as summary
        summary = source.get("summary")
        if source.get("element_type") == "glossary":
            summary = source.get("description", "")

        elem = {
            "element_id": source["element_id"],
            "hash_id": source.get("hash_id"),
            "name": source["name"],
            "element_type": source["element_type"],
            "file_path": source.get("relative_path", ""),
            "line_start": source.get("line_start", 0),
            "line_end": source.get("line_end"),
            "language": source.get("language", ""),
            "summary": summary,
            "signature": source.get("signature"),
            "repository": source["repository"],
            "scope": source["scope"],
            "parent_id": parent_id,
            "container": container,
            "visibility": source.get("visibility"),
            "is_async": source.get("is_async", False),
            "has_docstring": bool(source.get("docstring")),
            "decorators": source.get("decorators", []),
            "level": source.get("level", 0),
        }

        # Add subfeature info for features
        if source.get("element_type") == "feature":
            elem["member_count"] = source.get("member_count", 0)
            elem["subfeatures"] = subfeatures_by_feature.get(source["element_id"], [])

        # Add glossary-specific fields
        if source.get("element_type") == "glossary":
            elem["total_count"] = source.get("total_count", 0)
            elem["feature_count"] = source.get("feature_count", 0)

        elements.append(elem)

    return {
        "elements": elements,
        "total": total,
        "page": page,
        "limit": limit,
        "total_pages": (total + limit - 1) // limit if total > 0 else 0,
    }


@router.get("/browse/element/{hash_id}/children")
async def get_element_children(
    hash_id: str,
    username: str = "main",
    es_repo: ElasticsearchRepository = Depends(get_es_repository),
) -> dict:
    """Get child elements of a parent element.

    Args:
        hash_id: Parent element hash ID.
        username: User branch.

    Returns:
        List of child elements grouped by type.
    """
    # Look up element by hash_id to get the element_id
    source = es_repo.get_document_by_hash_id(hash_id)
    if not source:
        return {"error": "Element not found", "children": {}, "total_children": 0}

    element_id = source["element_id"]
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
                "hash_id",
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
        src = hit["_source"]
        elem_type = src["element_type"]

        if elem_type not in children_by_type:
            children_by_type[elem_type] = []

        children_by_type[elem_type].append({
            "element_id": src["element_id"],
            "hash_id": src.get("hash_id"),
            "name": src["name"],
            "element_type": elem_type,
            "line_start": src.get("line_start", 0),
            "line_end": src.get("line_end"),
            "summary": src.get("summary"),
            "signature": src.get("signature"),
            "visibility": src.get("visibility"),
            "is_async": src.get("is_async", False),
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
        "hash_id": hash_id,
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


@router.get("/browse/element/{hash_id}/details")
async def get_element_details(
    hash_id: str,
    include_call_graph: bool = False,
    username: str = "main",
    es_repo: ElasticsearchRepository = Depends(get_es_repository),
) -> dict:
    """Get detailed information about an element.

    Args:
        hash_id: Element hash ID (64-char SHA256).
        include_call_graph: Whether to compute callers/callees (expensive).
        username: User branch.

    Returns:
        Detailed element info including container hierarchy and optionally call graph.
    """
    # Look up element by hash_id
    source = es_repo.get_document_by_hash_id(hash_id)
    if not source:
        return {"error": "Element not found"}

    element_id = source["element_id"]
    client = es_repo._get_client()

    # Build container chain (parent -> grandparent -> ...)
    containers = []
    current_parent = source.get("parent_id")
    seen = {element_id}  # Prevent infinite loops

    while current_parent and current_parent not in seen:
        seen.add(current_parent)
        try:
            parent_doc = client.get(
                index=INDEX_NAME,
                id=current_parent,
                _source=["element_id", "hash_id", "name", "element_type", "relative_path", "line_start"],
            )
            psource = parent_doc["_source"]
            containers.append({
                "element_id": psource.get("element_id"),
                "hash_id": psource.get("hash_id"),
                "name": psource.get("name"),
                "element_type": psource.get("element_type"),
                "file_path": psource.get("relative_path"),
                "line_start": psource.get("line_start"),
            })
            current_parent = psource.get("parent_id") if "parent_id" in psource else None
        except Exception:
            break

    # Get child count
    children_result = client.count(
        index=INDEX_NAME,
        body={"query": {"term": {"parent_id": element_id}}},
    )
    child_count = children_result.get("count", 0)

    result = {
        "element_id": source.get("element_id"),
        "hash_id": source.get("hash_id"),
        "name": source.get("name"),
        "element_type": source.get("element_type"),
        "file_path": source.get("relative_path"),
        "line_start": source.get("line_start"),
        "line_end": source.get("line_end"),
        "language": source.get("language"),
        "summary": source.get("summary"),
        "signature": source.get("signature"),
        "docstring": source.get("docstring"),
        "visibility": source.get("visibility"),
        "is_async": source.get("is_async", False),
        "decorators": source.get("decorators", []),
        "level": source.get("level", 0),
        "repository": source.get("repository"),
        "scope": source.get("scope"),
        "containers": containers,  # Parent chain from immediate to file
        "child_count": child_count,
    }

    # Compute call graph if requested (expensive operation)
    if include_call_graph and source.get("element_type") in ("function", "method"):
        raw_code = source.get("raw_code", "")
        func_name = source.get("name", "")

        # Find callers (elements that reference this function)
        callers = []
        if func_name:
            # Search for references to this function in other code
            caller_result = client.search(
                index=INDEX_NAME,
                body={
                    "size": 20,
                    "query": {
                        "bool": {
                            "must": [
                                {"match": {"raw_code": func_name}},
                            ],
                            "must_not": [
                                {"term": {"element_id": element_id}},
                            ],
                            "filter": [
                                {"terms": {"element_type": ["function", "method"]}},
                            ],
                        },
                    },
                    "_source": ["element_id", "hash_id", "name", "element_type", "relative_path", "line_start"],
                },
            )
            for hit in caller_result.get("hits", {}).get("hits", []):
                s = hit["_source"]
                callers.append({
                    "element_id": s.get("element_id"),
                    "hash_id": s.get("hash_id"),
                    "name": s.get("name"),
                    "element_type": s.get("element_type"),
                    "file_path": s.get("relative_path"),
                    "line_start": s.get("line_start"),
                })

        # Find callees (functions called by this code)
        callees = []
        if raw_code:
            # Extract function calls using regex
            call_pattern = re.compile(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(')
            called_names = set(call_pattern.findall(raw_code))
            # Remove common keywords/builtins
            called_names -= {"if", "for", "while", "with", "return", "print", "len", "str", "int", "float", "list", "dict", "set", "range", "enumerate", "zip", "map", "filter", "sorted", "reversed", "any", "all", "sum", "min", "max", "abs", "round", "isinstance", "type", "getattr", "setattr", "hasattr", "super", func_name}

            if called_names:
                # Search for these function names
                callee_result = client.search(
                    index=INDEX_NAME,
                    body={
                        "size": 50,
                        "query": {
                            "bool": {
                                "must": [
                                    {"terms": {"name": list(called_names)}},
                                ],
                                "filter": [
                                    {"terms": {"element_type": ["function", "method", "class"]}},
                                ],
                            },
                        },
                        "_source": ["element_id", "hash_id", "name", "element_type", "relative_path", "line_start"],
                    },
                )
                for hit in callee_result.get("hits", {}).get("hits", []):
                    s = hit["_source"]
                    callees.append({
                        "element_id": s.get("element_id"),
                        "hash_id": s.get("hash_id"),
                        "name": s.get("name"),
                        "element_type": s.get("element_type"),
                        "file_path": s.get("relative_path"),
                        "line_start": s.get("line_start"),
                    })

        result["callers"] = callers
        result["callees"] = callees

    return result
