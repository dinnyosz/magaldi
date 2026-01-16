"""Search API routes."""

from __future__ import annotations

import logging
import math

from fastapi import APIRouter, Depends, HTTPException

from magaldi_web.dependencies import get_cached_config, get_es_repository

logger = logging.getLogger(__name__)
from magaldi_web.models import (
    SearchRequest,
    SearchResponse,
    SearchResult,
    SearchSummaryRequest,
    SearchSummaryResponse,
)
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


def _generate_search_summary_sync(
    query: str,
    results: list,
    config,
) -> tuple[str | None, str | None]:
    """Generate AI summary of search results (synchronous).

    Args:
        query: The user's search query.
        results: Top search results to summarize (SummaryResultItem objects).
        config: Application config with LLM settings.

    Returns:
        Tuple of (summary, error). One will be None.
    """
    try:
        from shared.ai.llm_client import LLMClient

        # Build context from results
        context_items = []
        for i, r in enumerate(results[:50], 1):
            item = f"{i}. [{r.element_type}] {r.name}"
            if r.file_path:
                item += f" ({r.file_path}:{r.line})"
            if r.summary:
                item += f"\n   Summary: {r.summary}"
            if r.signature:
                item += f"\n   Signature: {r.signature}"
            context_items.append(item)

        context = "\n\n".join(context_items)

        prompt = f"""The user asked: "{query}"

Based on the code elements found below, answer the user's question directly in 2-3 paragraphs.

Code Context:
{context}

Instructions:
- Answer the question directly - explain how this works in the codebase
- Describe the overall system/feature, not individual code elements
- Focus on the big picture: what capability exists, how it works, why it's designed this way
- Never list or enumerate specific functions, classes, or files
- Write as if explaining to someone who wants to understand the system, not the code"""

        # Use the configured model (prefer smaller/faster model for summary)
        llm = LLMClient(
            model=config.llm.get_litellm_model(config.llm.summarize_model),
            api_base=config.llm.url if config.llm.provider == "ollama" else None,
            api_key=config.llm.api_key,
        )

        summary = llm.generate(
            prompt=prompt,
            temperature=0.3,
            max_tokens=500,
            timeout=30,
        )

        return summary.strip(), None

    except Exception as e:
        logger.warning(f"AI summary generation failed: {e}")
        return None, str(e)


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

    # Validate at least one search mode is enabled
    if not request.use_text_search and not request.use_vector_search:
        raise HTTPException(
            status_code=400,
            detail="At least one search mode (text or vector) must be enabled",
        )

    # Generate query embedding for semantic search (only if vector search enabled)
    query_embedding = None
    embedding_error = None
    if request.use_vector_search:
        try:
            from shared.ai.embedding import CodeEmbeddingClient

            embed_client = CodeEmbeddingClient(
                url=config.llm.url,
                model=config.llm.embed_model,
                provider=config.llm.provider,
                api_key=config.llm.embed_api_key or config.llm.api_key,
            )
            query_embedding = await embed_client.embed_single_async(request.query)
            logger.info(f"Query embedding generated successfully (dim={len(query_embedding)})")
        except Exception as e:
            embedding_error = str(e)
            logger.warning(f"Query embedding failed: {e}")
            # If text search is also disabled, this is an error
            if not request.use_text_search:
                raise HTTPException(
                    status_code=500,
                    detail=f"Vector search failed and text search is disabled: {e}",
                )

    # Build query clauses
    should_clauses = []

    # Add text search clauses if enabled
    if request.use_text_search:
        should_clauses.extend([
            {"match": {"name": {"query": request.query, "boost": 2.0}}},
            {"match": {"summary": {"query": request.query, "boost": 1.5}}},
            {"match": {"docstring": {"query": request.query, "boost": 1.0}}},
            {"match": {"raw_code": {"query": request.query, "boost": 0.5}}},
        ])

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

    # Second pass: normalize to 0-1 scale and build results
    results = []
    for hit_data in processed_hits:
        source = hit_data["source"]
        vector_score = hit_data["vector_score"]
        text_score = hit_data["text_score"]

        # Normalize both to 0-1 scale relative to max in result set
        # Vector: divide by max vector score
        vector_normalized = round(vector_score / max_vector_score, 3) if max_vector_score and vector_score is not None else None
        # Text: divide by max text score
        text_normalized = round(text_score / max_text_score, 3) if max_text_score and text_score else None

        # Compute combined score for sorting (sum of enabled scores)
        combined_score = 0.0
        if request.use_text_search and text_normalized is not None:
            combined_score += text_normalized
        if request.use_vector_search and vector_normalized is not None:
            combined_score += vector_normalized

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
                text_score=text_normalized,
                vector_score=vector_normalized,
                highlights=hit_data["highlights"],
                combined_score=round(combined_score, 3),
            )
        )

    # Sort by combined score (sum of enabled search mode scores)
    results.sort(key=lambda r: r.combined_score or 0, reverse=True)

    return SearchResponse(
        query=request.query,
        total=total,
        took_ms=took_ms,
        results=results,
        text_search_used=request.use_text_search,
        vector_search_used=query_embedding is not None,
        embedding_error=embedding_error,
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


@router.post("/search/summary", response_model=SearchSummaryResponse)
async def generate_search_summary(
    request: SearchSummaryRequest,
) -> SearchSummaryResponse:
    """Generate AI summary of search results.

    This endpoint is called separately after search results are displayed,
    allowing results to appear immediately while summary generates.
    """
    if not request.results:
        return SearchSummaryResponse(error="No results provided")

    config = get_cached_config()
    summary, error = _generate_search_summary_sync(
        query=request.query,
        results=request.results,
        config=config,
    )

    return SearchSummaryResponse(summary=summary, error=error)
