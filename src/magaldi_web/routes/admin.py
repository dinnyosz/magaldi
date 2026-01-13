"""Admin API routes."""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends

from magaldi_web.dependencies import (
    check_elasticsearch_health,
    check_llm_health,
    check_redis_health,
    get_cached_config,
    get_es_repository,
)
from magaldi_web.models import (
    AdminOverviewResponse,
    HealthStatus,
    IndexStatsResponse,
    JobStatsResponse,
    QueueStats,
    ServiceHealth,
)
from shared.db.elasticsearch import ElasticsearchRepository, INDEX_NAME

router = APIRouter()


def format_bytes(size: int) -> str:
    """Format bytes to human-readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} PB"


@router.get("/admin/health", response_model=HealthStatus)
async def get_health(
    es_repo: ElasticsearchRepository = Depends(get_es_repository),
) -> HealthStatus:
    """Get detailed health status of all services."""
    config = get_cached_config()

    # Measure ES latency
    es_start = time.time()
    es_health = await check_elasticsearch_health(es_repo)
    es_latency = (time.time() - es_start) * 1000

    # Measure Ollama latency
    llm_start = time.time()
    llm_health = await check_llm_health(config)
    llm_latency = (time.time() - llm_start) * 1000

    # Measure Redis latency
    redis_start = time.time()
    redis_health = await check_redis_health(config)
    redis_latency = (time.time() - redis_start) * 1000

    return HealthStatus(
        elasticsearch=ServiceHealth(
            status=es_health.get("status", "unknown"),
            latency_ms=es_latency,
            details={
                "cluster_status": es_health.get("cluster_status"),
                "nodes": es_health.get("number_of_nodes"),
            },
        ),
        llm=ServiceHealth(
            status=llm_health.get("status", "unknown"),
            latency_ms=llm_latency,
            details={"models": llm_health.get("models_loaded", [])},
        ),
        redis=ServiceHealth(
            status=redis_health.get("status", "unknown"),
            latency_ms=redis_latency,
        ),
    )


@router.get("/admin/jobs", response_model=JobStatsResponse)
async def get_job_stats(
    es_repo: ElasticsearchRepository = Depends(get_es_repository),
) -> JobStatsResponse:
    """Get job queue statistics.

    Note: Currently returns zeros as job tracking is via Redis.
    TODO: Implement Redis-based job stats retrieval.
    """
    # TODO: Get actual job stats from Redis
    return JobStatsResponse(
        summarization=QueueStats(
            pending=0,
            running=0,
            completed=0,
            failed=0,
        ),
        embedding=QueueStats(
            pending=0,
            running=0,
            completed=0,
            failed=0,
        ),
    )


@router.get("/admin/index-stats", response_model=IndexStatsResponse)
async def get_index_stats(
    es_repo: ElasticsearchRepository = Depends(get_es_repository),
) -> IndexStatsResponse:
    """Get Elasticsearch index statistics."""
    client = es_repo._get_client()

    # Get index stats
    try:
        stats = client.indices.stats(index=INDEX_NAME)
        index_stats = stats.get("indices", {}).get(INDEX_NAME, {})
        primaries = index_stats.get("primaries", {})

        doc_count = primaries.get("docs", {}).get("count", 0)
        size_bytes = primaries.get("store", {}).get("size_in_bytes", 0)
    except Exception:
        doc_count = 0
        size_bytes = 0

    # Count documents with embeddings
    try:
        with_vectors_result = client.count(
            index=INDEX_NAME,
            body={"query": {"exists": {"field": "embedding"}}},
        )
        with_vectors = with_vectors_result.get("count", 0)
    except Exception:
        with_vectors = 0

    vector_coverage = (with_vectors / doc_count * 100) if doc_count > 0 else 0.0

    return IndexStatsResponse(
        index_name=INDEX_NAME,
        document_count=doc_count,
        size_bytes=size_bytes,
        size_human=format_bytes(size_bytes),
        with_vectors=with_vectors,
        vector_coverage_pct=round(vector_coverage, 1),
    )


@router.get("/admin/overview", response_model=AdminOverviewResponse)
async def get_admin_overview(
    es_repo: ElasticsearchRepository = Depends(get_es_repository),
) -> AdminOverviewResponse:
    """Get complete admin overview."""
    health = await get_health(es_repo)
    jobs = await get_job_stats(es_repo)
    index_stats = await get_index_stats(es_repo)

    return AdminOverviewResponse(
        health=health,
        jobs=jobs,
        index_stats=index_stats,
        recent_activity=[],  # TODO: Implement activity logging
    )


@router.post("/admin/jobs/retry")
async def retry_failed_jobs(
    job_type: str,
) -> dict:
    """Retry failed jobs.

    Args:
        job_type: Either 'summarization' or 'embedding'

    Note: Currently a placeholder. TODO: Implement Redis-based job retry.
    """
    if job_type not in ("summarization", "embedding"):
        return {"error": "Invalid job type", "jobs_reset": 0}

    # TODO: Implement actual retry logic via Redis
    return {"jobs_reset": 0, "message": "Job retry not yet implemented"}


@router.post("/admin/index/refresh")
async def refresh_index(
    es_repo: ElasticsearchRepository = Depends(get_es_repository),
) -> dict:
    """Refresh the Elasticsearch index."""
    client = es_repo._get_client()
    client.indices.refresh(index=INDEX_NAME)
    return {"status": "refreshed"}
