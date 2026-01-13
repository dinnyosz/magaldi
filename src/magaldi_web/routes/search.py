"""Search API routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from magaldi_web.dependencies import get_cached_config, get_es_repository
from magaldi_web.models import SearchRequest, SearchResponse, SearchResult
from shared.db.elasticsearch import ElasticsearchRepository, INDEX_NAME

router = APIRouter()


@router.post("/search", response_model=SearchResponse)
async def search(
    request: SearchRequest,
    es_repo: ElasticsearchRepository = Depends(get_es_repository),
) -> SearchResponse:
    """Perform semantic code search."""
    config = get_cached_config()
    client = es_repo._get_client()

    # Build filters
    filters = []

    # Username filter - include both main and user branch
    usernames = ["main"]
    if request.username and request.username != "main":
        usernames.append(request.username)
    filters.append({"terms": {"username": usernames}})

    if request.scope:
        filters.append({"term": {"scope": request.scope}})
    if request.repository:
        filters.append({"term": {"repository": request.repository}})
    if request.element_types:
        filters.append({"terms": {"element_type": request.element_types}})
    if request.language:
        filters.append({"term": {"language": request.language}})

    # Generate query embedding for semantic search
    query_embedding = None
    try:
        from shared.ai.embedding import CodeEmbeddingClient

        embed_client = CodeEmbeddingClient(
            url=config.llm.url,
            model=config.llm.embed_model,
            provider=config.llm.provider,
            api_key=config.llm.embed_api_key or config.llm.api_key,
        )
        query_embedding = embed_client.embed(request.query)
    except Exception:
        # Fall back to text-only search if embedding fails
        pass

    # Build query
    should_clauses = [
        {"match": {"name": {"query": request.query, "boost": 2.0}}},
        {"match": {"summary": {"query": request.query, "boost": 1.5}}},
        {"match": {"docstring": {"query": request.query, "boost": 1.0}}},
        {"match": {"raw_code": {"query": request.query, "boost": 0.5}}},
    ]

    # Add vector search if we have an embedding
    if query_embedding:
        should_clauses.insert(
            0,
            {
                "script_score": {
                    "query": {"exists": {"field": "embedding"}},
                    "script": {
                        "source": "(cosineSimilarity(params.qv, 'embedding') + 1.0) * 2",
                        "params": {"qv": query_embedding},
                    },
                },
            },
        )

    # Execute search
    result = client.search(
        index=INDEX_NAME,
        body={
            "size": request.limit,
            "from": request.offset,
            "query": {
                "bool": {
                    "filter": filters,
                    "should": should_clauses,
                    "minimum_should_match": 1,
                },
            },
            "highlight": {
                "fields": {
                    "summary": {},
                    "docstring": {},
                    "name": {},
                },
                "pre_tags": ["<mark>"],
                "post_tags": ["</mark>"],
            },
            "_source": [
                "element_id",
                "name",
                "element_type",
                "relative_path",
                "line_start",
                "language",
                "summary",
                "signature",
                "repository",
                "scope",
            ],
        },
    )

    # Process results
    hits = result.get("hits", {})
    total = hits.get("total", {}).get("value", 0)
    took_ms = result.get("took", 0)
    max_score = hits.get("max_score") or 1.0

    results = []
    for hit in hits.get("hits", []):
        source = hit["_source"]
        score = hit.get("_score", 0)
        highlights = hit.get("highlight", {})

        results.append(
            SearchResult(
                element_id=source["element_id"],
                name=source["name"],
                element_type=source["element_type"],
                file_path=source["relative_path"],
                line=source["line_start"],
                language=source["language"],
                summary=source.get("summary"),
                signature=source.get("signature"),
                repository=source["repository"],
                scope=source["scope"],
                score=score,
                relevance_pct=round((score / max_score) * 100, 1) if max_score else 0,
                highlights=highlights,
            )
        )

    return SearchResponse(
        query=request.query,
        total=total,
        took_ms=took_ms,
        results=results,
    )


@router.get("/search/filters")
async def get_search_filters(
    es_repo: ElasticsearchRepository = Depends(get_es_repository),
) -> dict:
    """Get available filter options for search."""
    client = es_repo._get_client()

    result = client.search(
        index=INDEX_NAME,
        body={
            "size": 0,
            "query": {"term": {"username": "main"}},
            "aggs": {
                "scopes": {"terms": {"field": "scope", "size": 50}},
                "repositories": {"terms": {"field": "repository", "size": 100}},
                "element_types": {"terms": {"field": "element_type", "size": 10}},
                "languages": {"terms": {"field": "language", "size": 20}},
            },
        },
    )

    aggs = result.get("aggregations", {})

    return {
        "scopes": [b["key"] for b in aggs.get("scopes", {}).get("buckets", [])],
        "repositories": [b["key"] for b in aggs.get("repositories", {}).get("buckets", [])],
        "element_types": [b["key"] for b in aggs.get("element_types", {}).get("buckets", [])],
        "languages": [b["key"] for b in aggs.get("languages", {}).get("buckets", [])],
    }
