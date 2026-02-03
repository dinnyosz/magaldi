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
    interface_count: int = 0
    trait_count: int = 0
    enum_count: int = 0
    type_alias_count: int = 0
    function_count: int = 0
    method_count: int = 0
    variable_count: int = 0
    constant_count: int = 0
    import_count: int = 0
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
    interface_count: int = 0
    trait_count: int = 0
    enum_count: int = 0
    type_alias_count: int = 0
    function_count: int = 0
    method_count: int = 0
    variable_count: int = 0
    constant_count: int = 0
    import_count: int = 0
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
    include_tests: bool = False


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
    combined_score: float | None = None  # Sum of enabled scores, used for sorting
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


class SummaryResultItem(BaseModel):
    """A simplified result item for summary generation."""

    name: str
    element_type: str
    file_path: str
    line: int
    summary: str | None = None
    signature: str | None = None


class SearchSummaryRequest(BaseModel):
    """Request for AI summary of search results."""

    query: str
    results: list[SummaryResultItem] = Field(default_factory=list, max_length=50)


class SearchSummaryResponse(BaseModel):
    """Response for AI summary endpoint."""

    summary: str | None = None
    error: str | None = None


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
    hash_id: str | None = None
    name: str
    summary: str | None = None


class ParentContext(BaseModel):
    """Context about the parent of an element."""

    element_id: str
    hash_id: str | None = None
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


class FeatureMember(BaseModel):
    """A member element of a feature or subfeature."""

    element_id: str
    hash_id: str | None = None
    name: str
    element_type: str
    file_path: str
    line: int
    summary: str | None = None
    signature: str | None = None


class ParentFeatureInfo(BaseModel):
    """Info about the parent feature of a subfeature."""

    label: str
    summary: str | None = None


class SubfeatureInfo(BaseModel):
    """Info about a subfeature belonging to a feature."""

    element_id: str
    hash_id: str | None = None
    label: str
    summary: str | None = None
    member_count: int = 0


class FeatureInfo(BaseModel):
    """Feature-specific information for feature/subfeature elements."""

    member_count: int = 0
    members: list[FeatureMember] = Field(default_factory=list)
    parent_feature: ParentFeatureInfo | None = None
    subfeatures: list[SubfeatureInfo] = Field(default_factory=list)


class GlossaryFeatureAssociation(BaseModel):
    """A feature association for a glossary term."""

    feature_id: str
    hash_id: str | None = None
    feature_label: str
    summary: str | None = None
    member_count: int = 0


class GlossaryInfo(BaseModel):
    """Glossary-specific information for glossary elements."""

    description: str = ""
    total_count: int = 0
    feature_count: int = 0
    file_paths: list[str] = Field(default_factory=list)
    feature_associations: list[GlossaryFeatureAssociation] = Field(default_factory=list)


class ClassAttributeInfo(BaseModel):
    """Information about a class attribute."""

    name: str
    type: str | None = None
    line: int | None = None


class DecoratorDetailInfo(BaseModel):
    """Detailed information about a decorator including arguments."""

    name: str  # e.g., "router.get", "click.option"
    args: str | None = None  # e.g., '"/users/{id}"', '"--verbose"'
    full: str | None = None  # e.g., 'router.get("/users/{id}")'


class ParameterInfo(BaseModel):
    """Information about a function/method parameter."""

    name: str
    type: str | None = None
    default: str | None = None


class CallInfo(BaseModel):
    """Information about a function call."""

    name: str
    receiver: str | None = None
    line: int
    resolved_id: str | None = None  # Resolved element ID if known


# =============================================================================
# CODE METRICS MODELS (for element detail)
# =============================================================================


class ComplexityInfo(BaseModel):
    """Cyclomatic complexity information."""

    cyclomatic: int = 0
    nesting_depth: int = 0
    branch_count: int = 0


class CodeMetricsInfo(BaseModel):
    """Basic code size metrics."""

    line_count: int = 0
    param_count: int = 0
    char_count: int = 0


class DocstringQualityInfo(BaseModel):
    """Docstring quality metrics."""

    has_docstring: bool = False
    has_params: bool = False
    has_return: bool = False
    coverage: float = 0.0


class SecurityIssueInfo(BaseModel):
    """A security issue detected in the code."""

    kind: str
    pattern: str
    line: int
    severity: str
    message: str | None = None


class EnvVarInfo(BaseModel):
    """An environment variable usage."""

    name: str
    line: int
    access_type: str


class ConcurrencyInfo(BaseModel):
    """Concurrency pattern information."""

    is_async: bool = False
    uses_locks: bool = False
    uses_threads: bool = False
    patterns: list[str] = Field(default_factory=list)


# =============================================================================
# EXTENDED CODE INTELLIGENCE MODELS
# =============================================================================


class TypeAnnotationInfo(BaseModel):
    """Type annotation extracted from source code."""

    name: str
    kind: str
    location: str
    line: int
    generic_args: list[str] | None = None


class TodoInfo(BaseModel):
    """A TODO/FIXME comment extracted from source code."""

    kind: str
    text: str
    line: int
    assignee: str | None = None
    priority: str | None = None
    issue_ref: str | None = None


class SectionMarkerInfo(BaseModel):
    """A section marker comment."""

    label: str
    line: int
    style: str


class CommentInfo(BaseModel):
    """A comment associated with a code element."""

    text: str
    line: int
    kind: str
    position: str


class PurityInfo(BaseModel):
    """Purity analysis result for a function."""

    level: str
    confidence: str
    reasons: list[str] = Field(default_factory=list)


class SideEffectInfo(BaseModel):
    """A side effect detected in a function."""

    kind: str
    target: str | None = None
    line: int


class HttpRouteInfo(BaseModel):
    """An HTTP route extracted from a web framework."""

    method: str
    path: str
    path_params: list[str] = Field(default_factory=list)
    framework: str
    line: int | None = None


class CliCommandInfo(BaseModel):
    """A CLI command extracted from a CLI framework."""

    name: str
    options: list[dict[str, Any]] = Field(default_factory=list)
    framework: str


class MetricsSummaryInfo(BaseModel):
    """Aggregated metrics summary for files and classes."""

    total_elements: int = 0
    total_functions: int = 0
    avg_complexity: float = 0.0
    max_complexity: int = 0
    total_lines: int = 0
    documented_pct: float = 0.0
    async_count: int = 0
    security_issue_count: int = 0
    security_by_severity: dict[str, int] = Field(default_factory=dict)


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
    decorator_details: list[DecoratorDetailInfo] = Field(default_factory=list)
    visibility: str | None = None
    is_async: bool = False
    is_test: bool = False
    indexed_at: str | None = None
    # Enhanced context for classes
    base_classes: list[str] = Field(default_factory=list)
    class_attributes: list[ClassAttributeInfo] = Field(default_factory=list)
    # Enhanced context for functions/methods
    return_type: str | None = None
    parameters: list[ParameterInfo] = Field(default_factory=list)
    exceptions_raised: list[str] = Field(default_factory=list)
    attributes_modified: list[str] = Field(default_factory=list)
    calls: list[CallInfo] = Field(default_factory=list)
    # For file elements
    imports: list[ImportInfo] = Field(default_factory=list)
    element_count: int | None = None
    # Type annotations
    type_annotations: list[TypeAnnotationInfo] = Field(default_factory=list)
    # Pattern detection
    detected_patterns: list[str] = Field(default_factory=list)
    pattern_confidence: dict[str, float] = Field(default_factory=dict)
    # Documentation artifacts
    todos: list[TodoInfo] = Field(default_factory=list)
    section_markers: list[SectionMarkerInfo] = Field(default_factory=list)
    associated_comments: list[CommentInfo] = Field(default_factory=list)
    # API surface
    http_routes: list[HttpRouteInfo] = Field(default_factory=list)
    cli_commands: list[CliCommandInfo] = Field(default_factory=list)
    # Purity and side effects
    purity: PurityInfo | None = None
    side_effects: list[SideEffectInfo] = Field(default_factory=list)
    mutated_state: list[str] = Field(default_factory=list)
    # API surface flag
    is_public_api: bool = False
    # Variable context (how variables are used)
    context_usages: list[str] = Field(default_factory=list)
    # Context and relationships
    context: ElementContext
    repository: RepoRef
    feature_info: FeatureInfo | None = None
    glossary_info: GlossaryInfo | None = None
    # Code metrics (for functions/methods)
    complexity: ComplexityInfo | None = None
    code_metrics: CodeMetricsInfo | None = None
    docstring_quality: DocstringQualityInfo | None = None
    security_issues: list[SecurityIssueInfo] = Field(default_factory=list)
    env_vars: list[EnvVarInfo] = Field(default_factory=list)
    concurrency: ConcurrencyInfo | None = None
    # Aggregated metrics (for files/classes)
    metrics_summary: MetricsSummaryInfo | None = None


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
# MCP ANALYTICS MODELS
# =============================================================================


class ToolUsageInfo(BaseModel):
    """Usage info for a single MCP tool."""

    tool_name: str
    call_count: int
    percentage: float  # Percentage of total calls


class ToolTransitionInfo(BaseModel):
    """Transition from one tool to another."""

    from_tool: str
    to_tool: str
    count: int
    percentage: float  # Percentage of transitions from from_tool


class ToolDurationInfo(BaseModel):
    """Execution duration info for a single MCP tool."""

    tool_name: str
    total_ms: int
    call_count: int
    avg_ms: float


class MCPAnalyticsResponse(BaseModel):
    """Response for MCP analytics endpoint."""

    total_calls: int
    unique_tools: int
    today_calls: int
    tool_usage: list[ToolUsageInfo]
    top_transitions: list[ToolTransitionInfo]
    transition_matrix: dict[str, dict[str, int]]  # from_tool -> to_tool -> count
    tool_durations: list[ToolDurationInfo] = []  # Tool execution times


class DailyActivityItem(BaseModel):
    """Activity data for a single day."""

    date: str  # YYYY-MM-DD
    total_calls: int
    tool_counts: dict[str, int]  # tool_name -> count


class MCPActivityHistoryResponse(BaseModel):
    """Response for MCP activity history endpoint."""

    days: int  # Number of days requested
    max_days: int = 30  # Maximum retention period
    daily_activity: list[DailyActivityItem]
    total_calls: int  # Sum across all days


# =============================================================================
# VECTOR MAP MODELS
# =============================================================================


class VectorPoint(BaseModel):
    """A point in the vector space visualization."""

    x: float
    y: float
    z: float | None = None
    element_id: str
    hash_id: str | None = None
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


# =============================================================================
# GLOSSARY MODELS
# =============================================================================


class GlossaryTermSummary(BaseModel):
    """Summary of a glossary term."""

    term: str
    hash_id: str | None = None
    description: str = ""
    total_count: int
    file_count: int
    feature_count: int


class FeatureAssociationSummary(BaseModel):
    """Summary of a feature association."""

    feature_id: str
    feature_label: str
    frequency: int
    total_members: int
    percentage: float


class GlossaryTermResponse(BaseModel):
    """Detailed glossary term response."""

    term: str
    description: str = ""
    total_count: int
    element_ids: list[str]
    file_paths: list[str]
    feature_associations: list[FeatureAssociationSummary]


class GlossaryListResponse(BaseModel):
    """List of glossary terms response."""

    terms: list[GlossaryTermSummary]
    total: int


# =============================================================================
# CALL ANALYSIS MODELS
# =============================================================================


class CallerInfo(BaseModel):
    """Information about a caller."""

    element_id: str
    hash_id: str | None = None
    name: str
    element_type: str
    file_path: str
    line: int
    summary: str | None = None


class CalleeInfo(BaseModel):
    """Information about a callee."""

    name: str
    receiver: str | None = None
    line: int
    element_id: str | None = None  # None if unresolved
    hash_id: str | None = None
    element_type: str | None = None
    file_path: str | None = None
    target_line: int | None = None
    summary: str | None = None


class CallGraphResponse(BaseModel):
    """Response for call graph endpoint."""

    element: dict
    callers: list[CallerInfo]
    callees: list[CalleeInfo]


class CallChainNode(BaseModel):
    """Node in a call chain tree."""

    element_id: str | None = None
    hash_id: str | None = None
    name: str
    element_type: str | None = None
    file_path: str | None = None
    line: int | None = None
    summary: str | None = None
    callers: list["CallChainNode"] = Field(default_factory=list)
    callees: list["CallChainNode"] = Field(default_factory=list)
    cycle: bool = False
    unresolved: bool = False
    missing: bool = False


class CallChainResponse(BaseModel):
    """Response for call chain endpoint."""

    root: dict
    direction: str
    max_depth: int
    callers: list[CallChainNode] = Field(default_factory=list)
    callees: list[CallChainNode] = Field(default_factory=list)


class DeadCodeItem(BaseModel):
    """A potentially dead code item."""

    element_id: str
    hash_id: str | None = None
    name: str
    element_type: str
    file_path: str
    line: int
    summary: str | None = None
    is_test: bool = False


class DeadCodeResponse(BaseModel):
    """Response for dead code analysis."""

    potentially_dead: list[DeadCodeItem]
    stats: dict


class HttpRouteInfo(BaseModel):
    """HTTP route information."""

    method: str
    path: str
    path_params: list[str] = Field(default_factory=list)
    framework: str | None = None


class CliCommandInfo(BaseModel):
    """CLI command information."""

    name: str
    full_command: str | None = None  # Full command path like "magaldi web serve"
    options: list[dict] = Field(default_factory=list)
    framework: str | None = None


class EntryPointItem(BaseModel):
    """An entry point item."""

    element_id: str
    hash_id: str | None = None
    name: str
    element_type: str
    file_path: str
    line: int
    summary: str | None = None
    decorators: list[str] = Field(default_factory=list)
    http_routes: list[HttpRouteInfo] = Field(default_factory=list)
    cli_commands: list[CliCommandInfo] = Field(default_factory=list)


class EntryPointsResponse(BaseModel):
    """Response for entry points analysis."""

    http: list[EntryPointItem]
    cli: list[EntryPointItem]
    test: list[EntryPointItem]
    main: list[EntryPointItem]
    async_tasks: list[EntryPointItem]
    stats: dict


# =============================================================================
# DEPENDENCY ANALYSIS MODELS
# =============================================================================


class ImportInfo(BaseModel):
    """Information about an import."""

    name: str
    module: str
    alias: str | None = None
    line: int
    is_internal: bool = False


class DependenciesResponse(BaseModel):
    """Response for file dependencies."""

    file_info: dict
    internal_imports: list[ImportInfo]
    external_imports: list[ImportInfo]
    stats: dict


class DependentInfo(BaseModel):
    """Information about a dependent file."""

    element_id: str
    hash_id: str | None = None
    file_path: str
    name: str


class DependentsResponse(BaseModel):
    """Response for module dependents."""

    module: str
    dependents: list[DependentInfo]
    total: int


class DependencyGraphEdge(BaseModel):
    """An edge in the dependency graph."""

    from_path: str = Field(alias="from")
    to_path: str = Field(alias="to")

    model_config = {"populate_by_name": True}


class DependencyGraphResponse(BaseModel):
    """Response for dependency graph."""

    nodes: list[str]
    edges: list[dict]  # {from, to}
    cycles: list[list[str]]
    stats: dict


# =============================================================================
# SIMILARITY MODELS
# =============================================================================


class SimilarCodeItem(BaseModel):
    """A similar code item."""

    element_id: str
    hash_id: str | None = None
    name: str
    element_type: str
    file_path: str
    line: int
    summary: str | None = None
    similarity: float


class SimilarCodeResponse(BaseModel):
    """Response for similar code search."""

    element: dict
    similar_structure: list[SimilarCodeItem]
    similar_intent: list[SimilarCodeItem]


# =============================================================================
# EXPLAIN ELEMENT MODELS
# =============================================================================


class ExplainElementResponse(BaseModel):
    """Comprehensive element explanation response."""

    element: dict
    callers: list[CallerInfo]
    callees: list[CalleeInfo]
    imports: list[ImportInfo]
    similar_code: list[SimilarCodeItem]
    parent: dict | None = None
    embedding_status: dict


# =============================================================================
# METRICS MODELS
# =============================================================================


class ComplexFunctionItem(BaseModel):
    """A function with high complexity."""

    element_id: str
    hash_id: str | None = None
    name: str
    element_type: str
    file_path: str
    line: int
    cyclomatic: int
    nesting_depth: int
    branch_count: int
    line_count: int
    summary: str | None = None
    is_test: bool = False


class ComplexFunctionsResponse(BaseModel):
    """Response for complex functions endpoint."""

    functions: list[ComplexFunctionItem]
    count: int
    min_complexity: int


class SecurityIssueItem(BaseModel):
    """A security issue in code."""

    element_id: str
    hash_id: str | None = None
    function_name: str
    element_type: str
    file_path: str
    function_line: int
    issue_line: int
    severity: str
    kind: str
    message: str | None = None


class SecurityIssuesResponse(BaseModel):
    """Response for security issues endpoint."""

    issues: list[SecurityIssueItem]
    count: int
    by_severity: dict[str, int]
    by_kind: dict[str, int]


class UndocumentedItem(BaseModel):
    """An undocumented function."""

    element_id: str
    hash_id: str | None = None
    name: str
    element_type: str
    file_path: str
    line: int
    visibility: str | None = None
    has_docstring: bool
    has_params: bool
    has_return: bool
    coverage: float
    param_count: int
    summary: str | None = None


class UndocumentedResponse(BaseModel):
    """Response for undocumented functions endpoint."""

    functions: list[UndocumentedItem]
    count: int
    max_coverage: float


class EnvUsageItem(BaseModel):
    """An environment variable usage."""

    element_id: str
    hash_id: str | None = None
    function_name: str
    element_type: str
    file_path: str
    function_line: int
    env_name: str
    env_line: int
    access_type: str


class EnvUsageResponse(BaseModel):
    """Response for environment variable usage endpoint."""

    usages: list[EnvUsageItem]
    count: int
    unique_vars: int
    by_var: dict[str, int]


class AsyncCodeItem(BaseModel):
    """An async/concurrent code element."""

    element_id: str
    hash_id: str | None = None
    name: str
    element_type: str
    file_path: str
    line: int
    is_async: bool
    uses_threads: bool
    uses_locks: bool
    patterns: list[str] = Field(default_factory=list)
    summary: str | None = None


class AsyncCodeResponse(BaseModel):
    """Response for async code endpoint."""

    functions: list[AsyncCodeItem]
    count: int
    pattern_filter: str
    by_pattern: dict[str, int]
