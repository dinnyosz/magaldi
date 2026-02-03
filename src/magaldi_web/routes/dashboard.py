"""Dashboard API routes."""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends

from magaldi_web.dependencies import (
    check_elasticsearch_health,
    check_llm_health,
    check_redis_health,
    get_cached_config,
    get_es_repository,
    get_redis_queue_stats,
)
from magaldi_web.models import (
    DashboardResponse,
    DashboardStats,
    HealthStatus,
    QueueInfo,
    QueueStatus,
    RepoSummary,
    ServiceHealth,
)
from shared.db.elasticsearch import ElasticsearchRepository, INDEX_NAME

router = APIRouter()


@router.get("/dashboard", response_model=DashboardResponse)
async def get_dashboard(
    es_repo: ElasticsearchRepository = Depends(get_es_repository),
) -> DashboardResponse:
    """Get dashboard overview data."""
    config = get_cached_config()
    client = es_repo._get_client()

    # Get repository stats with element type breakdowns
    repos_agg = client.search(
        index=INDEX_NAME,
        body={
            "size": 0,
            "query": {"term": {"username": "main"}},
            "aggs": {
                "repos": {
                    "composite": {
                        "size": 100,
                        "sources": [
                            {"scope": {"terms": {"field": "scope"}}},
                            {"repository": {"terms": {"field": "repository"}}},
                        ],
                    },
                    "aggs": {
                        "file_count": {"filter": {"term": {"element_type": "file"}}},
                        "class_count": {"filter": {"term": {"element_type": "class"}}},
                        "interface_count": {"filter": {"term": {"element_type": "interface"}}},
                        "trait_count": {"filter": {"term": {"element_type": "trait"}}},
                        "enum_count": {"filter": {"term": {"element_type": "enum"}}},
                        "type_alias_count": {"filter": {"term": {"element_type": "type_alias"}}},
                        "function_count": {"filter": {"term": {"element_type": "function"}}},
                        "method_count": {"filter": {"term": {"element_type": "method"}}},
                        "variable_count": {"filter": {"term": {"element_type": "variable"}}},
                        "constant_count": {"filter": {"term": {"element_type": "constant"}}},
                        "import_count": {"filter": {"term": {"element_type": "import"}}},
                        "feature_count": {"filter": {"term": {"element_type": "feature"}}},
                        "languages": {"terms": {"field": "language", "size": 10}},
                    },
                },
                "total_repos": {"cardinality": {"field": "repository"}},
            },
        },
    )

    # Build repo summaries
    recent_repos = []
    repo_buckets = repos_agg.get("aggregations", {}).get("repos", {}).get("buckets", [])
    for bucket in repo_buckets[:10]:  # Top 10
        languages = [
            lang["key"]
            for lang in bucket.get("languages", {}).get("buckets", [])
        ]
        recent_repos.append(
            RepoSummary(
                scope=bucket["key"]["scope"],
                name=bucket["key"]["repository"],
                file_count=bucket.get("file_count", {}).get("doc_count", 0),
                class_count=bucket.get("class_count", {}).get("doc_count", 0),
                interface_count=bucket.get("interface_count", {}).get("doc_count", 0),
                trait_count=bucket.get("trait_count", {}).get("doc_count", 0),
                enum_count=bucket.get("enum_count", {}).get("doc_count", 0),
                type_alias_count=bucket.get("type_alias_count", {}).get("doc_count", 0),
                function_count=bucket.get("function_count", {}).get("doc_count", 0),
                method_count=bucket.get("method_count", {}).get("doc_count", 0),
                variable_count=bucket.get("variable_count", {}).get("doc_count", 0),
                constant_count=bucket.get("constant_count", {}).get("doc_count", 0),
                import_count=bucket.get("import_count", {}).get("doc_count", 0),
                feature_count=bucket.get("feature_count", {}).get("doc_count", 0),
                element_count=bucket["doc_count"],
                languages=languages,
            )
        )

    # Total counts
    total_repos = repos_agg.get("aggregations", {}).get("total_repos", {}).get("value", 0)

    # Element counts by type for main branch
    type_agg = client.search(
        index=INDEX_NAME,
        body={
            "size": 0,
            "query": {"term": {"username": "main"}},
            "aggs": {
                "by_type": {
                    "terms": {"field": "element_type", "size": 20},
                },
            },
        },
    )

    # Parse type counts
    type_buckets = type_agg.get("aggregations", {}).get("by_type", {}).get("buckets", [])
    type_counts = {bucket["key"]: bucket["doc_count"] for bucket in type_buckets}
    element_count = sum(type_counts.values())

    # Check service health and queue stats
    es_health = await check_elasticsearch_health(es_repo)
    llm_health = await check_llm_health(config)
    redis_health = await check_redis_health(config)
    queue_stats = await get_redis_queue_stats(config)

    # Convert queue stats to QueueInfo objects
    summarization_queues = {
        user: QueueInfo(**info)
        for user, info in queue_stats.get("summarization", {}).items()
    }
    embedding_queues = {
        user: QueueInfo(**info)
        for user, info in queue_stats.get("embedding", {}).items()
    }
    labeling_queues = {
        user: QueueInfo(**info)
        for user, info in queue_stats.get("labeling", {}).items()
    }
    feature_queues = {
        user: QueueInfo(**info)
        for user, info in queue_stats.get("feature", {}).items()
    }
    subfeature_queues = {
        user: QueueInfo(**info)
        for user, info in queue_stats.get("subfeature", {}).items()
    }
    subfeature_labeling_queues = {
        user: QueueInfo(**info)
        for user, info in queue_stats.get("subfeature_labeling", {}).items()
    }

    return DashboardResponse(
        stats=DashboardStats(
            repository_count=total_repos,
            element_count=element_count,
            file_count=type_counts.get("file", 0),
            class_count=type_counts.get("class", 0),
            interface_count=type_counts.get("interface", 0),
            trait_count=type_counts.get("trait", 0),
            enum_count=type_counts.get("enum", 0),
            type_alias_count=type_counts.get("type_alias", 0),
            function_count=type_counts.get("function", 0),
            method_count=type_counts.get("method", 0),
            variable_count=type_counts.get("variable", 0),
            constant_count=type_counts.get("constant", 0),
            import_count=type_counts.get("import", 0),
            feature_count=type_counts.get("feature", 0),
            subfeature_count=type_counts.get("subfeature", 0),
        ),
        recent_repos=recent_repos,
        queue_status=QueueStatus(
            summarization=summarization_queues,
            embedding=embedding_queues,
            labeling=labeling_queues,
            feature=feature_queues,
            subfeature=subfeature_queues,
            subfeature_labeling=subfeature_labeling_queues,
            total_pending=queue_stats.get("total_pending", 0),
            total_running=queue_stats.get("total_running", 0),
        ),
        health=HealthStatus(
            elasticsearch=ServiceHealth(
                status=es_health.get("status", "unknown"),
                details={
                    "cluster_status": es_health.get("cluster_status"),
                    "nodes": es_health.get("number_of_nodes"),
                },
            ),
            llm=ServiceHealth(
                status=llm_health.get("status", "unknown"),
                details={"models": llm_health.get("models_loaded", [])},
            ),
            redis=ServiceHealth(
                status=redis_health.get("status", "unknown"),
            ),
        ),
    )
