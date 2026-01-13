"""Pydantic models for the Web API."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


# =============================================================================
# COMMON MODELS
# =============================================================================


class ServiceHealth(BaseModel):
    """Health status for a service."""

    status: str
    latency_ms: float | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class HealthStatus(BaseModel):
    """Overall system health status."""

    elasticsearch: ServiceHealth
    llm: ServiceHealth
    redis: ServiceHealth


class QueueStats(BaseModel):
    """Statistics for a job queue."""

    pending: int = 0
    running: int = 0
    completed: int = 0
    failed: int = 0


# =============================================================================
# DASHBOARD MODELS
# =============================================================================


class RepoSummary(BaseModel):
    """Summary of a repository for dashboard display."""

    scope: str
    name: str
    description: str | None = None
    file_count: int = 0
    class_count: int = 0
    function_count: int = 0
    method_count: int = 0
    variable_count: int = 0
    constant_count: int = 0
    feature_count: int = 0
    element_count: int = 0
    languages: list[str] = Field(default_factory=list)
    last_parsed: datetime | None = None


class DashboardStats(BaseModel):
    """Statistics for the dashboard."""

    repository_count: int = 0
    element_count: int = 0
    file_count: int = 0
    class_count: int = 0
    function_count: int = 0
    method_count: int = 0
    variable_count: int = 0
    constant_count: int = 0
    feature_count: int = 0
    subfeature_count: int = 0


class QueueInfo(BaseModel):
    """Status of a single job queue."""

    pending: int = 0
    running: int = 0


class QueueStatus(BaseModel):
    """Status of all job queues."""

    summarization: dict[str, QueueInfo] = Field(default_factory=dict)
    embedding: dict[str, QueueInfo] = Field(default_factory=dict)
    labeling: dict[str, QueueInfo] = Field(default_factory=dict)
    feature: dict[str, QueueInfo] = Field(default_factory=dict)
    subfeature: dict[str, QueueInfo] = Field(default_factory=dict)
    subfeature_labeling: dict[str, QueueInfo] = Field(default_factory=dict)
    total_pending: int = 0
    total_running: int = 0


class DashboardResponse(BaseModel):
    """Response for the dashboard endpoint."""

    stats: DashboardStats
    recent_repos: list[RepoSummary] = Field(default_factory=list)
    queue_status: QueueStatus
    health: HealthStatus


# =============================================================================
# SEARCH MODELS
# =============================================================================


class SearchRequest(BaseModel):
    """Request for semantic code search."""

    query: str
    scope: str | None = None
    repository: str | None = None
    username: str | None = None
    element_types: list[str] | None = None
    language: str | None = None
    limit: int = Field(default=20, ge=1, le=100)
    offset: int = Field(default=0, ge=0)
    use_text_search: bool = True
    use_vector_search: bool = True


class SearchResult(BaseModel):
    """A single search result."""

    element_id: str
    hash_id: str | None = None
    name: str
    element_type: str
    file_path: str
    line: int
    language: str
    summary: str | None = None
    signature: str | None = None
    repository: str
    scope: str
    score: float
    relevance_pct: float
    text_score: float | None = None
    vector_score: float | None = None
    highlights: dict[str, list[str]] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    """Response for search endpoint."""

    query: str
    total: int
    took_ms: int
    results: list[SearchResult] = Field(default_factory=list)
    text_search_used: bool = True
    vector_search_used: bool = False
    embedding_error: str | None = None


# =============================================================================
# REPOSITORY MODELS
# =============================================================================


class TreeNode(BaseModel):
    """A node in the file tree."""

    name: str
    type: str  # "file" or "directory"
    path: str | None = None
    language: str | None = None
    children: list[TreeNode] = Field(default_factory=list)


class FileTreeResponse(BaseModel):
    """Response for file tree endpoint."""

    scope: str
    repository: str
    tree: list[TreeNode] = Field(default_factory=list)


class ElementInfo(BaseModel):
    """Brief info about a code element."""

    element_id: str
    name: str
    element_type: str
    line_start: int
    line_end: int | None = None
    summary: str | None = None
    signature: str | None = None


class FileInfo(BaseModel):
    """Information about a file."""

    path: str
    language: str
    summary: str | None = None
    line_count: int = 0


class ElementStats(BaseModel):
    """Statistics about elements in a file."""

    classes: int = 0
    functions: int = 0
    methods: int = 0
    variables: int = 0


class Contributor(BaseModel):
    """A contributor working on a file."""

    username: str
    has_changes: bool = False
    last_indexed: datetime | None = None
    expires_at: datetime | None = None


class FileDetailResponse(BaseModel):
    """Response for file detail endpoint."""

    file: FileInfo
    structure: list[ElementInfo] = Field(default_factory=list)
    stats: ElementStats
    contributors: list[Contributor] = Field(default_factory=list)


class ActiveUser(BaseModel):
    """An active user in a repository."""

    username: str
    files_modified: int = 0
    last_activity: datetime | None = None


class RepoDetailResponse(BaseModel):
    """Response for repository detail endpoint."""

    scope: str
    name: str
    description: str | None = None
    file_count: int = 0
    element_count: int = 0
    languages: list[str] = Field(default_factory=list)
    active_users: list[ActiveUser] = Field(default_factory=list)


class RepoListResponse(BaseModel):
    """Response for repository list endpoint."""

    repos: list[RepoSummary] = Field(default_factory=list)
    total: int = 0


# =============================================================================
# ELEMENT MODELS
# =============================================================================


class FileContext(BaseModel):
    """Context about the file containing an element."""

    element_id: str
    name: str
    summary: str | None = None


class ParentContext(BaseModel):
    """Context about the parent of an element."""

    element_id: str
    name: str
    element_type: str
    summary: str | None = None


class ChildInfo(BaseModel):
    """Info about a child element."""

    element_id: str
    hash_id: str | None = None
    name: str
    element_type: str
    line: int
    summary: str | None = None
    signature: str | None = None


class SiblingInfo(BaseModel):
    """Info about a sibling element."""

    element_id: str
    hash_id: str | None = None
    name: str
    element_type: str
    line: int
    summary: str | None = None


class ElementContext(BaseModel):
    """Context surrounding an element."""

    file: FileContext | None = None
    parent: ParentContext | None = None
    children: list[ChildInfo] = Field(default_factory=list)
    siblings: list[SiblingInfo] = Field(default_factory=list)


class RepoRef(BaseModel):
    """Reference to a repository."""

    scope: str
    name: str


class ElementDetailResponse(BaseModel):
    """Response for element detail endpoint."""

    element_id: str
    hash_id: str | None = None
    name: str
    element_type: str
    file_path: str
    line_start: int
    line_end: int | None = None
    language: str
    summary: str | None = None
    signature: str | None = None
    docstring: str | None = None
    raw_code: str | None = None
    decorators: list[str] = Field(default_factory=list)
    visibility: str | None = None
    is_async: bool = False
    context: ElementContext
    repository: RepoRef


# =============================================================================
# ADMIN MODELS
# =============================================================================


class JobStatsResponse(BaseModel):
    """Response for job statistics endpoint."""

    summarization: QueueStats
    embedding: QueueStats


class IndexStatsResponse(BaseModel):
    """Response for index statistics endpoint."""

    index_name: str
    document_count: int = 0
    size_bytes: int = 0
    size_human: str = ""
    with_vectors: int = 0
    vector_coverage_pct: float = 0.0


class RetryResponse(BaseModel):
    """Response for retry jobs endpoint."""

    jobs_reset: int = 0


class RecentActivity(BaseModel):
    """A recent activity log entry."""

    timestamp: datetime
    message: str
    level: str = "info"


class AdminOverviewResponse(BaseModel):
    """Response for admin overview endpoint."""

    health: HealthStatus
    jobs: JobStatsResponse
    index_stats: IndexStatsResponse
    recent_activity: list[RecentActivity] = Field(default_factory=list)


# =============================================================================
# VECTOR MAP MODELS
# =============================================================================


class VectorPoint(BaseModel):
    """A point in the vector space visualization."""

    x: float
    y: float
    z: float | None = None
    element_id: str
    name: str
    element_type: str
    file_path: str
    line: int
    summary: str | None = None


class VectorMapResponse(BaseModel):
    """Response for vector map endpoint."""

    points: list[VectorPoint] = Field(default_factory=list)
    bounds: dict[str, list[float]] = Field(default_factory=dict)
    algorithm: str = "umap"
    dimensions: int = 2
    element_count: int = 0


class ClusterMember(BaseModel):
    """A member of a cluster."""

    element_id: str
    name: str
    element_type: str


class ClusterRepresentative(BaseModel):
    """Representative element of a cluster."""

    name: str
    element_type: str
    file_path: str
    summary: str | None = None


class Subfeature(BaseModel):
    """A subfeature within a larger feature."""

    subfeature_id: str
    label: str
    summary: str | None = None
    member_count: int = 0


class Cluster(BaseModel):
    """A semantic cluster of code elements."""

    cluster_id: int
    size: int
    representative: ClusterRepresentative
    members: list[ClusterMember] = Field(default_factory=list)
    subfeatures: list[Subfeature] = Field(default_factory=list)


class ClustersResponse(BaseModel):
    """Response for clusters endpoint."""

    clusters: list[Cluster] = Field(default_factory=list)
    total_elements: int = 0
