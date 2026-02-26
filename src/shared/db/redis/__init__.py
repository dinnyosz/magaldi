"""Redis repository implementations for Magaldi.

This package provides Redis-based storage for job queues, summary caching,
and MCP analytics.

Re-exports all classes and functions for backward compatibility with
code that imports from `shared.db.redis`.
"""

from __future__ import annotations

# Base module - connection management and key utilities
from .base import (
    EMBEDDING_JOBS,
    EMBEDDING_QUEUE,
    EMBEDDING_RUNNING,
    FEATURE_JOBS,
    FEATURE_QUEUE,
    FEATURE_RUNNING,
    LABELING_JOBS,
    LABELING_QUEUE,
    LABELING_RUNNING,
    SUBFEATURE_JOBS,
    SUBFEATURE_LABELING_JOBS,
    SUBFEATURE_LABELING_QUEUE,
    SUBFEATURE_LABELING_RUNNING,
    SUBFEATURE_QUEUE,
    SUBFEATURE_RUNNING,
    SUMMARIZATION_JOBS,
    SUMMARIZATION_QUEUE,
    SUMMARIZATION_RUNNING,
    JobType,
    KeyType,
    RedisRepository,
    _key,
    _make_key,
)

# Job repositories
from .job_repositories import (
    RedisEmbeddingJobRepository,
    RedisFeatureJobRepository,
    RedisLabelingJobRepository,
    RedisSubfeatureJobRepository,
    RedisSubfeatureLabelingJobRepository,
    RedisSummarizationJobRepository,
)

# MCP analytics
from .mcp_analytics import (
    MCP_CONFIRMED_TRANSITIONS,
    MCP_DAILY_CALLS_PREFIX,
    MCP_RECENT_CALLS,
    MCP_SESSION_PREFIX,
    MCP_TOOL_CALLS,
    MCP_TOOL_NAMES,
    MCP_TOOL_TRANSITIONS,
    RedisMCPAnalyticsRepository,
)

# Summary store
from .summary_store import RedisSummaryStore

__all__ = [
    # Base
    "RedisRepository",
    "JobType",
    "KeyType",
    "_make_key",
    "_key",
    # Key constants
    "SUMMARIZATION_QUEUE",
    "SUMMARIZATION_RUNNING",
    "SUMMARIZATION_JOBS",
    "EMBEDDING_QUEUE",
    "EMBEDDING_RUNNING",
    "EMBEDDING_JOBS",
    "FEATURE_QUEUE",
    "FEATURE_RUNNING",
    "FEATURE_JOBS",
    "LABELING_QUEUE",
    "LABELING_RUNNING",
    "LABELING_JOBS",
    "SUBFEATURE_QUEUE",
    "SUBFEATURE_RUNNING",
    "SUBFEATURE_JOBS",
    "SUBFEATURE_LABELING_QUEUE",
    "SUBFEATURE_LABELING_RUNNING",
    "SUBFEATURE_LABELING_JOBS",
    # Job repositories
    "RedisSummarizationJobRepository",
    "RedisEmbeddingJobRepository",
    "RedisFeatureJobRepository",
    "RedisLabelingJobRepository",
    "RedisSubfeatureJobRepository",
    "RedisSubfeatureLabelingJobRepository",
    # Summary store
    "RedisSummaryStore",
    # MCP analytics
    "RedisMCPAnalyticsRepository",
    "MCP_TOOL_CALLS",
    "MCP_TOOL_TRANSITIONS",
    "MCP_CONFIRMED_TRANSITIONS",
    "MCP_SESSION_PREFIX",
    "MCP_DAILY_CALLS_PREFIX",
    "MCP_RECENT_CALLS",
    "MCP_TOOL_NAMES",
]
