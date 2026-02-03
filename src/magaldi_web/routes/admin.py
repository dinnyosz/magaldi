"""Admin API routes."""

from __future__ import annotations

import time
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, Query

from magaldi_web.dependencies import (
    check_elasticsearch_health,
    check_llm_health,
    check_redis_health,
    get_cached_config,
    get_es_repository,
)
from magaldi_web.models import (
    AdminOverviewResponse,
    CausalLinkInfo,
    CausalPairInfo,
    DailyActivityItem,
    HealthStatus,
    IndexStatsResponse,
    JobStatsResponse,
    MCPActivityHistoryResponse,
    MCPAnalyticsResponse,
    MCPCausalStatisticsResponse,
    MCPRecentCallsResponse,
    MCPTransitionDetailsResponse,
    QueueStats,
    RecentToolCallInfo,
    ServiceHealth,
    ToolDurationInfo,
    ToolTransitionInfo,
    ToolUsageInfo,
    TransitionDetailInfo,
)
from shared.db.elasticsearch import INDEX_NAME, ElasticsearchRepository
from shared.db.redis import RedisMCPAnalyticsRepository

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


@router.get("/admin/mcp-analytics", response_model=MCPAnalyticsResponse)
async def get_mcp_analytics() -> MCPAnalyticsResponse:
    """Get MCP tool usage analytics."""
    config = get_cached_config()

    try:
        analytics_repo = RedisMCPAnalyticsRepository(config)

        # Get total counts
        tool_counts = analytics_repo.get_tool_counts()
        total_calls = sum(tool_counts.values())
        unique_tools = len(tool_counts)

        # Get today's counts
        daily_counts = analytics_repo.get_daily_counts()
        today_calls = sum(daily_counts.values())

        # Build tool usage list with percentages
        tool_usage = []
        for tool_name, count in sorted(
            tool_counts.items(), key=lambda x: x[1], reverse=True
        ):
            percentage = (count / total_calls * 100) if total_calls > 0 else 0.0
            tool_usage.append(
                ToolUsageInfo(
                    tool_name=tool_name,
                    call_count=count,
                    percentage=round(percentage, 1),
                )
            )

        # Get top transitions with confirmation data
        raw_transitions = analytics_repo.get_top_transitions(limit=20)
        transition_matrix = analytics_repo.get_tool_transitions()
        confirmed_matrix = analytics_repo.get_confirmed_transitions()

        top_transitions = []
        for from_tool, to_tool, count in raw_transitions:
            # Calculate percentage of transitions from from_tool
            total_from = sum(transition_matrix.get(from_tool, {}).values())
            percentage = (count / total_from * 100) if total_from > 0 else 0.0
            # Get confirmed count for this transition
            confirmed_count = confirmed_matrix.get(from_tool, {}).get(to_tool, 0)
            confirmation_rate = (confirmed_count / count * 100) if count > 0 else 0.0
            top_transitions.append(
                ToolTransitionInfo(
                    from_tool=from_tool,
                    to_tool=to_tool,
                    count=count,
                    percentage=round(percentage, 1),
                    confirmed_count=confirmed_count,
                    confirmation_rate=round(confirmation_rate, 1),
                )
            )

        # Sort by confirmed count (causal) descending, then by total count
        top_transitions.sort(key=lambda t: (t.confirmed_count, t.count), reverse=True)

        # Get tool durations
        duration_stats = analytics_repo.get_tool_durations()
        tool_durations = [
            ToolDurationInfo(
                tool_name=tool_name,
                total_ms=stats["total_ms"],
                call_count=stats["count"],
                avg_ms=stats["avg_ms"],
            )
            for tool_name, stats in sorted(
                duration_stats.items(),
                key=lambda x: x[1]["avg_ms"],
                reverse=True,
            )
        ]

        return MCPAnalyticsResponse(
            total_calls=total_calls,
            unique_tools=unique_tools,
            today_calls=today_calls,
            tool_usage=tool_usage,
            top_transitions=top_transitions,
            transition_matrix=transition_matrix,
            tool_durations=tool_durations,
        )

    except Exception:
        # Return empty response if Redis is unavailable
        return MCPAnalyticsResponse(
            total_calls=0,
            unique_tools=0,
            today_calls=0,
            tool_usage=[],
            top_transitions=[],
            transition_matrix={},
            tool_durations=[],
        )


@router.post("/admin/mcp-analytics/clear")
async def clear_mcp_analytics() -> dict:
    """Clear all MCP analytics data."""
    config = get_cached_config()

    try:
        analytics_repo = RedisMCPAnalyticsRepository(config)
        analytics_repo.clear_analytics()
        return {"status": "cleared"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.get("/admin/mcp-analytics/history", response_model=MCPActivityHistoryResponse)
async def get_mcp_activity_history(
    days: int = Query(default=7, ge=1, le=30, description="Number of days of history (1-30)"),
) -> MCPActivityHistoryResponse:
    """Get MCP tool usage activity history for the last N days.

    Args:
        days: Number of days of history to return (default 7, max 30).

    Returns:
        Daily activity breakdown for the requested period.
    """
    config = get_cached_config()
    max_days = 30  # Matches DAILY_TTL in RedisMCPAnalyticsRepository

    try:
        analytics_repo = RedisMCPAnalyticsRepository(config)

        daily_activity = []
        total_calls = 0

        # Get activity for each day, going backwards from today
        for i in range(days):
            date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
            tool_counts = analytics_repo.get_daily_counts(date)
            day_total = sum(tool_counts.values())
            total_calls += day_total

            daily_activity.append(
                DailyActivityItem(
                    date=date,
                    total_calls=day_total,
                    tool_counts=tool_counts,
                )
            )

        # Reverse so oldest day is first (chronological order)
        daily_activity.reverse()

        return MCPActivityHistoryResponse(
            days=days,
            max_days=max_days,
            daily_activity=daily_activity,
            total_calls=total_calls,
        )

    except Exception:
        # Return empty response if Redis is unavailable
        return MCPActivityHistoryResponse(
            days=days,
            max_days=max_days,
            daily_activity=[],
            total_calls=0,
        )


@router.get("/admin/mcp-analytics/transition-details", response_model=MCPTransitionDetailsResponse)
async def get_mcp_transition_details(
    from_tool: str | None = Query(default=None, description="Filter by source tool"),
    to_tool: str | None = Query(default=None, description="Filter by destination tool"),
    limit: int = Query(default=50, ge=1, le=200, description="Max transitions to return"),
) -> MCPTransitionDetailsResponse:
    """Get detailed transition information including caller output and callee input.

    This endpoint returns tool transitions with the actual data passed between
    consecutive tool calls: the caller's output and the callee's input.

    Args:
        from_tool: Filter by source tool name (optional).
        to_tool: Filter by destination tool name (optional).
        limit: Maximum number of transitions to return.

    Returns:
        List of transition details with caller output and callee input.
    """
    config = get_cached_config()

    try:
        analytics_repo = RedisMCPAnalyticsRepository(config)
        transitions_data = analytics_repo.get_transition_details(
            from_tool=from_tool,
            to_tool=to_tool,
            limit=limit,
        )

        transitions = [
            TransitionDetailInfo(
                caller=t["caller"],
                caller_input=t.get("caller_input"),
                caller_output=t.get("caller_output"),
                callee=t["callee"],
                callee_input=t.get("callee_input"),
                callee_output=t.get("callee_output"),
                elapsed_ms=t.get("elapsed_ms"),
                caller_duration_ms=t.get("caller_duration_ms"),
                callee_duration_ms=t.get("callee_duration_ms"),
                timestamp=t.get("timestamp"),
                is_causal=t.get("is_causal", False),
                causal_match_type=t.get("causal_match_type"),
                causal_matched_value=t.get("causal_matched_value"),
            )
            for t in transitions_data
        ]

        return MCPTransitionDetailsResponse(
            transitions=transitions,
            from_tool=from_tool,
            to_tool=to_tool,
            total=len(transitions),
        )

    except Exception:
        return MCPTransitionDetailsResponse(
            transitions=[],
            from_tool=from_tool,
            to_tool=to_tool,
            total=0,
        )


@router.get("/admin/mcp-analytics/recent-calls", response_model=MCPRecentCallsResponse)
async def get_mcp_recent_calls(
    limit: int = Query(default=100, ge=1, le=500, description="Max calls to return"),
    tool_name: str | None = Query(default=None, description="Filter by tool name"),
    only_causal: bool = Query(default=False, description="Only return calls with causal links"),
) -> MCPRecentCallsResponse:
    """Get recent tool calls with full input/output details.

    Args:
        limit: Maximum number of calls to return.
        tool_name: Optional filter to show only calls for this tool.
        only_causal: If True, only return calls that have causal links.

    Returns:
        List of recent tool calls with their inputs, outputs, and causal info.
    """
    config = get_cached_config()

    try:
        analytics_repo = RedisMCPAnalyticsRepository(config)
        calls_data = analytics_repo.get_calls_with_causal_info(
            limit=limit,
            only_with_links=only_causal,
        )

        # Filter by tool name if specified
        if tool_name:
            calls_data = [c for c in calls_data if c.get("tool_name") == tool_name]

        calls = []
        for c in calls_data:
            # Parse triggered_by into CausalLinkInfo if present
            triggered_by = None
            if c.get("triggered_by"):
                tb = c["triggered_by"]
                triggered_by = CausalLinkInfo(
                    call_id=tb.get("call_id"),
                    tool_name=tb.get("tool_name", "unknown"),
                    matched_value=tb.get("matched_value"),
                    match_type=tb.get("match_type", "unknown"),
                )

            calls.append(
                RecentToolCallInfo(
                    call_id=c.get("call_id", ""),
                    tool_name=c.get("tool_name", ""),
                    session_id=c.get("session_id", ""),
                    input_args=c.get("input_args"),
                    output_result=c.get("output_result"),
                    start_time=c.get("start_time"),
                    end_time=c.get("end_time"),
                    duration_ms=c.get("duration_ms"),
                    triggered_by=triggered_by,
                    suggested_tools=c.get("suggested_tools"),
                )
            )

        return MCPRecentCallsResponse(
            calls=calls,
            total=len(calls),
        )

    except Exception:
        return MCPRecentCallsResponse(
            calls=[],
            total=0,
        )


@router.get("/admin/mcp-analytics/causal", response_model=MCPCausalStatisticsResponse)
async def get_mcp_causal_statistics() -> MCPCausalStatisticsResponse:
    """Get statistics about causal relationships between MCP tool calls.

    Returns statistics about how tools trigger other tools via:
    - Parameter matching (values passed from output to input)
    - Tool suggestions (previous output mentioned the current tool)
    """
    config = get_cached_config()

    try:
        analytics_repo = RedisMCPAnalyticsRepository(config)
        stats = analytics_repo.get_causal_statistics()

        return MCPCausalStatisticsResponse(
            total_causal_links=stats["total_causal_links"],
            total_calls=stats["total_calls"],
            causal_rate=stats["causal_rate"],
            by_match_type=stats["by_match_type"],
            top_causal_pairs=[
                CausalPairInfo(
                    source=p["source"],
                    target=p["target"],
                    count=p["count"],
                )
                for p in stats["top_causal_pairs"]
            ],
        )

    except Exception:
        return MCPCausalStatisticsResponse(
            total_causal_links=0,
            total_calls=0,
            causal_rate=0.0,
            by_match_type={},
            top_causal_pairs=[],
        )
