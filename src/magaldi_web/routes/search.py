"""Search API routes."""

from __future__ import annotations

import math

from fastapi import APIRouter, Depends, HTTPException

from magaldi_web.dependencies import get_cached_config, get_es_repository
from magaldi_web.models import SearchRequest, SearchResponse, SearchResult
from shared.db.elasticsearch import ElasticsearchRepository, INDEX_NAME

router = APIRouter()


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)


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
        query_embedding = embed_client.embed_single(request.query)
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
                "hash_id",
                "name",
                "element_type",
                "relative_path",
                "line_start",
                "language",
                "summary",
                "signature",
                "repository",
                "scope",
                "embedding",
            ],
        },
    )

    # Process results
    hits = result.get("hits", {})
    total = hits.get("total", {}).get("value", 0)
    took_ms = result.get("took", 0)
    max_score = hits.get("max_score") or 1.0

    # Find max scores for normalization
    max_vector_score = 0.0
    max_text_score = 0.0

    # First pass: compute individual scores
    processed_hits = []
    for hit in hits.get("hits", []):
        source = hit["_source"]
        score = hit.get("_score", 0)
        highlights = hit.get("highlight", {})
        result_embedding = source.get("embedding")

        # Compute vector similarity (0-1 range)
        vector_score = None
        text_score = None
        if query_embedding and result_embedding:
            vector_score = cosine_similarity(query_embedding, result_embedding)
            # Vector contribution to total: (similarity + 1) * 2, range 0-4
            vector_contribution = (vector_score + 1.0) * 2
            # Text score is the remainder
            text_score = max(0, score - vector_contribution)
            max_vector_score = max(max_vector_score, vector_score)
            max_text_score = max(max_text_score, text_score)
        else:
            # No vector search, all score is from text
            text_score = score
            max_text_score = max(max_text_score, text_score)

        processed_hits.append({
            "source": source,
            "score": score,
            "highlights": highlights,
            "vector_score": vector_score,
            "text_score": text_score,
        })

    # Second pass: normalize and build results
    results = []
    for hit_data in processed_hits:
        source = hit_data["source"]
        vector_score = hit_data["vector_score"]
        text_score = hit_data["text_score"]

        # Normalize to percentages (0-100)
        vector_pct = round(vector_score * 100, 1) if vector_score is not None else None
        text_pct = round((text_score / max_text_score) * 100, 1) if max_text_score and text_score else None

        results.append(
            SearchResult(
                element_id=source["element_id"],
                hash_id=source.get("hash_id"),
                name=source["name"],
                element_type=source["element_type"],
                file_path=source.get("relative_path", ""),
                line=source.get("line_start", 0),
                language=source.get("language", ""),
                summary=source.get("summary"),
                signature=source.get("signature"),
                repository=source["repository"],
                scope=source["scope"],
                score=hit_data["score"],
                relevance_pct=round((hit_data["score"] / max_score) * 100, 1) if max_score else 0,
                text_score=text_pct,
                vector_score=vector_pct,
                highlights=hit_data["highlights"],
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
