"""Elements API routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Path, Query

from magaldi_web.dependencies import get_es_repository
from magaldi_web.models import (
    ChildInfo,
    ClassAttributeInfo,
    ElementContext,
    ElementDetailResponse,
    FeatureInfo,
    FeatureMember,
    FileContext,
    ImportInfo,
    ParentContext,
    ParentFeatureInfo,
    RepoRef,
    SiblingInfo,
)
from shared.db.elasticsearch import ElasticsearchRepository, INDEX_NAME

router = APIRouter()


@router.get("/elements/similar/{identifier}")
async def get_similar_elements(
    identifier: str = Path(..., description="Element hash_id (64-char SHA256) or element_id"),
    limit: int = Query(default=10, ge=1, le=50),
    es_repo: ElasticsearchRepository = Depends(get_es_repository),
) -> list[dict]:
    """Find similar elements using vector similarity."""
    # Try hash_id first (64 hex characters), then fall back to element_id
    source = None
    if len(identifier) == 64 and all(c in "0123456789abcdef" for c in identifier.lower()):
        source = es_repo.get_document_by_hash_id(identifier)

    if not source:
        source = es_repo.get_document(identifier)

    if not source:
        raise HTTPException(status_code=404, detail="Element not found")

    element_id = source["element_id"]
    embedding = source.get("embedding")

    if not embedding:
        raise HTTPException(status_code=400, detail="Element has no embedding")

    # Find similar elements
    client = es_repo._get_client()
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
                "hash_id",
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
                "hash_id": s.get("hash_id"),
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


@router.get("/elements/{identifier}", response_model=ElementDetailResponse)
async def get_element_detail(
    identifier: str = Path(..., description="Element hash_id (64-char SHA256) or element_id"),
    es_repo: ElasticsearchRepository = Depends(get_es_repository),
) -> ElementDetailResponse:
    """Get detailed information about a code element."""
    # Try hash_id first (64 hex characters), then fall back to element_id
    source = None
    if len(identifier) == 64 and all(c in "0123456789abcdef" for c in identifier.lower()):
        source = es_repo.get_document_by_hash_id(identifier)

    if not source:
        # Try as element_id
        source = es_repo.get_document(identifier)

    if not source:
        raise HTTPException(status_code=404, detail="Element not found")

    element_id = source["element_id"]
    client = es_repo._get_client()

    # Get file context (only for code elements that have a relative_path)
    file_context = None
    if source["element_type"] not in ("file", "feature", "subfeature") and source.get("relative_path"):
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
                "_source": ["element_id", "hash_id", "name", "summary"],
            },
        )
        file_hits = file_result.get("hits", {}).get("hits", [])
        if file_hits:
            fs = file_hits[0]["_source"]
            file_context = FileContext(
                element_id=fs["element_id"],
                hash_id=fs.get("hash_id"),
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
                "_source": ["element_id", "hash_id", "name", "element_type", "summary"],
            },
        )
        parent_hits = parent_result.get("hits", {}).get("hits", [])
        if parent_hits:
            ps = parent_hits[0]["_source"]
            parent_context = ParentContext(
                element_id=ps["element_id"],
                hash_id=ps.get("hash_id"),
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
            "_source": ["element_id", "hash_id", "name", "element_type", "line_start", "summary", "signature"],
            "sort": [{"line_start": "asc"}],
        },
    )
    for hit in children_result.get("hits", {}).get("hits", []):
        cs = hit["_source"]
        children.append(
            ChildInfo(
                element_id=cs["element_id"],
                hash_id=cs.get("hash_id"),
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
                "_source": ["element_id", "hash_id", "name", "element_type", "line_start", "summary"],
                "sort": [{"line_start": "asc"}],
            },
        )
        for hit in siblings_result.get("hits", {}).get("hits", []):
            ss = hit["_source"]
            siblings.append(
                SiblingInfo(
                    element_id=ss["element_id"],
                    hash_id=ss.get("hash_id"),
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

    # Get feature info for features/subfeatures
    feature_info = None
    if source["element_type"] in ("feature", "subfeature"):
        member_ids = source.get("member_ids", [])
        members = []

        # Fetch member element details (limit to first 50 for performance)
        if member_ids:
            members_result = client.mget(
                index=INDEX_NAME,
                ids=member_ids[:50],
                _source=["element_id", "hash_id", "name", "element_type", "relative_path", "line_start", "summary", "signature"],
            )
            for doc in members_result.get("docs", []):
                if doc.get("found") and doc.get("_source"):
                    ms = doc["_source"]
                    members.append(
                        FeatureMember(
                            element_id=ms["element_id"],
                            hash_id=ms.get("hash_id"),
                            name=ms["name"],
                            element_type=ms["element_type"],
                            file_path=ms.get("relative_path", ""),
                            line=ms.get("line_start", 0),
                            summary=ms.get("summary"),
                            signature=ms.get("signature"),
                        )
                    )

        # Get parent feature info for subfeatures
        parent_feature = None
        if source["element_type"] == "subfeature":
            parent_label = source.get("parent_feature_label")
            if parent_label:
                parent_feature = ParentFeatureInfo(
                    label=parent_label,
                    summary=source.get("parent_feature_summary"),
                )

        feature_info = FeatureInfo(
            member_count=source.get("member_count", len(member_ids)),
            members=members,
            parent_feature=parent_feature,
        )

    # Parse class attributes
    class_attributes = []
    raw_class_attrs = source.get("class_attributes", [])
    if raw_class_attrs:
        for attr in raw_class_attrs:
            class_attributes.append(
                ClassAttributeInfo(
                    name=attr.get("name", ""),
                    type=attr.get("type"),
                    line=attr.get("line"),
                )
            )

    # Parse imports (for file elements)
    imports = []
    raw_imports = source.get("imports", [])
    if raw_imports:
        for imp in raw_imports:
            imports.append(
                ImportInfo(
                    name=imp.get("name", ""),
                    module=imp.get("module", ""),
                    alias=imp.get("alias"),
                    line=imp.get("line", 0),
                    is_internal=imp.get("is_internal", False),
                )
            )

    return ElementDetailResponse(
        element_id=source["element_id"],
        hash_id=source.get("hash_id"),
        name=source["name"],
        element_type=source["element_type"],
        file_path=source.get("relative_path", ""),
        line_start=source.get("line_start", 0),
        line_end=source.get("line_end"),
        language=source.get("language", "unknown"),
        summary=source.get("summary"),
        signature=source.get("signature"),
        docstring=source.get("docstring"),
        raw_code=source.get("raw_code"),
        decorators=decorators,
        visibility=source.get("visibility"),
        is_async=source.get("is_async", False),
        is_test=source.get("is_test", False),
        indexed_at=source.get("indexed_at"),
        # Enhanced context for classes
        base_classes=source.get("base_classes", []),
        class_attributes=class_attributes,
        # Enhanced context for functions/methods
        exceptions_raised=source.get("exceptions_raised", []),
        attributes_modified=source.get("attributes_modified", []),
        # For file elements
        imports=imports,
        element_count=source.get("element_count"),
        # Context and relationships
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
        feature_info=feature_info,
    )
