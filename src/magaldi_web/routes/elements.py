"""Elements API routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Path, Query

from magaldi_web.dependencies import get_es_repository
from magaldi_web.models import (
    ChildInfo,
    ElementContext,
    ElementDetailResponse,
    FileContext,
    ParentContext,
    RepoRef,
    SiblingInfo,
)
from shared.db.elasticsearch import ElasticsearchRepository, INDEX_NAME

router = APIRouter()


@router.get("/elements/{element_id:path}", response_model=ElementDetailResponse)
async def get_element_detail(
    element_id: str = Path(..., description="Element ID"),
    es_repo: ElasticsearchRepository = Depends(get_es_repository),
) -> ElementDetailResponse:
    """Get detailed information about a code element."""
    client = es_repo._get_client()

    # Get the element
    result = client.search(
        index=INDEX_NAME,
        body={
            "size": 1,
            "query": {"term": {"element_id": element_id}},
        },
    )

    hits = result.get("hits", {}).get("hits", [])
    if not hits:
        raise HTTPException(status_code=404, detail="Element not found")

    source = hits[0]["_source"]

    # Get file context
    file_context = None
    if source["element_type"] != "file":
        file_result = client.search(
            index=INDEX_NAME,
            body={
                "size": 1,
                "query": {
                    "bool": {
                        "filter": [
                            {"term": {"scope": source["scope"]}},
                            {"term": {"repository": source["repository"]}},
                            {"term": {"username": source["username"]}},
                            {"term": {"relative_path": source["relative_path"]}},
                            {"term": {"element_type": "file"}},
                        ],
                    },
                },
                "_source": ["element_id", "name", "summary"],
            },
        )
        file_hits = file_result.get("hits", {}).get("hits", [])
        if file_hits:
            fs = file_hits[0]["_source"]
            file_context = FileContext(
                element_id=fs["element_id"],
                name=fs["name"],
                summary=fs.get("summary"),
            )

    # Get parent context
    parent_context = None
    parent_id = source.get("parent_id")
    if parent_id:
        parent_result = client.search(
            index=INDEX_NAME,
            body={
                "size": 1,
                "query": {"term": {"element_id": parent_id}},
                "_source": ["element_id", "name", "element_type", "summary"],
            },
        )
        parent_hits = parent_result.get("hits", {}).get("hits", [])
        if parent_hits:
            ps = parent_hits[0]["_source"]
            parent_context = ParentContext(
                element_id=ps["element_id"],
                name=ps["name"],
                element_type=ps["element_type"],
                summary=ps.get("summary"),
            )

    # Get children
    children = []
    children_result = client.search(
        index=INDEX_NAME,
        body={
            "size": 100,
            "query": {"term": {"parent_id": element_id}},
            "_source": ["element_id", "name", "element_type", "line_start", "summary", "signature"],
            "sort": [{"line_start": "asc"}],
        },
    )
    for hit in children_result.get("hits", {}).get("hits", []):
        cs = hit["_source"]
        children.append(
            ChildInfo(
                element_id=cs["element_id"],
                name=cs["name"],
                element_type=cs["element_type"],
                line=cs["line_start"],
                summary=cs.get("summary"),
                signature=cs.get("signature"),
            )
        )

    # Get siblings (if has parent)
    siblings = []
    if parent_id:
        siblings_result = client.search(
            index=INDEX_NAME,
            body={
                "size": 50,
                "query": {
                    "bool": {
                        "filter": [{"term": {"parent_id": parent_id}}],
                        "must_not": [{"term": {"element_id": element_id}}],
                    },
                },
                "_source": ["element_id", "name", "element_type", "line_start", "summary"],
                "sort": [{"line_start": "asc"}],
            },
        )
        for hit in siblings_result.get("hits", {}).get("hits", []):
            ss = hit["_source"]
            siblings.append(
                SiblingInfo(
                    element_id=ss["element_id"],
                    name=ss["name"],
                    element_type=ss["element_type"],
                    line=ss["line_start"],
                    summary=ss.get("summary"),
                )
            )

    # Parse decorators
    decorators = []
    if source.get("decorators"):
        import json

        try:
            decorators = json.loads(source["decorators"])
        except (json.JSONDecodeError, TypeError):
            decorators = []

    return ElementDetailResponse(
        element_id=source["element_id"],
        name=source["name"],
        element_type=source["element_type"],
        file_path=source["relative_path"],
        line_start=source["line_start"],
        line_end=source.get("line_end"),
        language=source.get("language", "unknown"),
        summary=source.get("summary"),
        signature=source.get("signature"),
        docstring=source.get("docstring"),
        raw_code=source.get("raw_code"),
        decorators=decorators,
        visibility=source.get("visibility"),
        is_async=source.get("is_async", False),
        context=ElementContext(
            file=file_context,
            parent=parent_context,
            children=children,
            siblings=siblings,
        ),
        repository=RepoRef(
            scope=source["scope"],
            name=source["repository"],
        ),
    )


@router.get("/elements/{element_id:path}/similar")
async def get_similar_elements(
    element_id: str = Path(..., description="Element ID"),
    limit: int = Query(default=10, ge=1, le=50),
    es_repo: ElasticsearchRepository = Depends(get_es_repository),
) -> list[dict]:
    """Find similar elements using vector similarity."""
    client = es_repo._get_client()

    # Get the element's embedding
    result = client.search(
        index=INDEX_NAME,
        body={
            "size": 1,
            "query": {"term": {"element_id": element_id}},
            "_source": ["embedding", "scope", "repository", "username"],
        },
    )

    hits = result.get("hits", {}).get("hits", [])
    if not hits:
        raise HTTPException(status_code=404, detail="Element not found")

    source = hits[0]["_source"]
    embedding = source.get("embedding")

    if not embedding:
        raise HTTPException(status_code=400, detail="Element has no embedding")

    # Find similar elements
    similar_result = client.search(
        index=INDEX_NAME,
        body={
            "size": limit + 1,  # +1 to exclude self
            "query": {
                "bool": {
                    "filter": [
                        {"term": {"scope": source["scope"]}},
                        {"term": {"repository": source["repository"]}},
                        {"term": {"username": source["username"]}},
                        {"exists": {"field": "embedding"}},
                    ],
                    "must": {
                        "script_score": {
                            "query": {"match_all": {}},
                            "script": {
                                "source": "cosineSimilarity(params.qv, 'embedding') + 1.0",
                                "params": {"qv": embedding},
                            },
                        },
                    },
                },
            },
            "_source": [
                "element_id",
                "name",
                "element_type",
                "relative_path",
                "line_start",
                "summary",
            ],
        },
    )

    similar = []
    for hit in similar_result.get("hits", {}).get("hits", []):
        s = hit["_source"]
        if s["element_id"] == element_id:
            continue  # Skip self
        similar.append(
            {
                "element_id": s["element_id"],
                "name": s["name"],
                "element_type": s["element_type"],
                "file_path": s["relative_path"],
                "line": s["line_start"],
                "summary": s.get("summary"),
                "similarity": round((hit["_score"] - 1.0), 3),  # Convert back to cosine similarity
            }
        )
        if len(similar) >= limit:
            break

    return similar
