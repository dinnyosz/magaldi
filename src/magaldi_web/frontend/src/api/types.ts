/**
 * API type definitions for Magaldi backend
 */

// =============================================================================
// Dashboard Types
// =============================================================================

export interface DashboardStats {
  stats: {
    repository_count: number
    element_count: number
    file_count: number
    class_count: number
    interface_count: number
    trait_count: number
    enum_count: number
    type_alias_count: number
    function_count: number
    method_count: number
    variable_count: number
    constant_count: number
    import_count: number
    feature_count: number
    subfeature_count: number
  }
  recent_repos: Array<{
    scope: string
    name: string
    description: string | null
    file_count: number
    class_count: number
    interface_count: number
    trait_count: number
    enum_count: number
    type_alias_count: number
    function_count: number
    method_count: number
    variable_count: number
    constant_count: number
    import_count: number
    feature_count: number
    element_count: number
    languages: string[]
    last_parsed: string | null
  }>
  queue_status: {
    summarization: Record<string, { pending: number; running: number }>
    embedding: Record<string, { pending: number; running: number }>
    labeling: Record<string, { pending: number; running: number }>
    feature: Record<string, { pending: number; running: number }>
    subfeature: Record<string, { pending: number; running: number }>
    subfeature_labeling: Record<string, { pending: number; running: number }>
    total_pending: number
    total_running: number
  }
  health: {
    elasticsearch: { status: string; details?: Record<string, unknown> }
    llm: { status: string; details?: Record<string, unknown> }
    redis: { status: string; details?: Record<string, unknown> }
  }
}

// =============================================================================
// Search Types
// =============================================================================

export interface SearchRequest {
  query: string
  scope?: string
  repository?: string
  element_types?: string[]
  username?: string
  limit?: number
  offset?: number
  use_text_search?: boolean
  use_vector_search?: boolean
  include_tests?: boolean
}

export interface SearchResult {
  element_id: string
  hash_id: string | null
  name: string
  element_type: string
  file_path: string
  line: number
  summary: string | null
  score: number
  relevance_pct: number
  text_score: number | null
  vector_score: number | null
  combined_score: number | null
  code_snippet?: string
}

export interface SearchResponse {
  results: SearchResult[]
  total: number
  query: string
  took_ms: number
  text_search_used: boolean
  vector_search_used: boolean
  embedding_error: string | null
}

export interface SummaryResultItem {
  name: string
  element_type: string
  file_path: string
  line: number
  summary: string | null
  signature: string | null
}

export interface SearchSummaryRequest {
  query: string
  results: SummaryResultItem[]
}

export interface SearchSummaryResponse {
  summary: string | null
  error: string | null
}

// =============================================================================
// Repository Types
// =============================================================================

export interface FileTreeNode {
  name: string
  path: string
  type: 'file' | 'directory'
  children?: FileTreeNode[]
  element_count?: number
}

export interface FileTreeResponse {
  tree: FileTreeNode[]
  total_files: number
  total_directories: number
}

export interface FileElement {
  element_id: string
  hash_id: string | null
  name: string
  element_type: string
  line_start: number
  line_end: number
  summary: string | null
}

export interface FileDetailResponse {
  path: string
  content: string
  language: string
  elements: FileElement[]
}

export interface RepoListItem {
  scope: string
  name: string
  file_count: number
  element_count: number
  languages: string[]
}

export interface RepoListResponse {
  repos: RepoListItem[]
  total: number
}

// =============================================================================
// Element Types
// =============================================================================

export interface ElementDetailChild {
  element_id: string
  hash_id: string | null
  name: string
  element_type: string
  line: number
  summary: string | null
  signature: string | null
}

export interface ElementDetailSibling {
  element_id: string
  hash_id: string | null
  name: string
  element_type: string
  line: number
  summary: string | null
}

export interface ElementDetailParent {
  element_id: string
  hash_id: string | null
  name: string
  element_type: string
  summary: string | null
}

export interface ElementDetailFile {
  element_id: string
  hash_id: string | null
  name: string
  summary: string | null
}

export interface FeatureMember {
  element_id: string
  hash_id: string | null
  name: string
  element_type: string
  file_path: string
  line: number
  summary: string | null
  signature: string | null
}

export interface ParentFeatureInfo {
  label: string
  summary: string | null
}

export interface SubfeatureInfo {
  element_id: string
  hash_id: string | null
  label: string
  summary: string | null
  member_count: number
}

export interface ConnectedFeatureInfo {
  feature_id: string
  hash_id: string | null
  label: string
  affinity: number
  shared_member_count: number
  common_glossary_terms: string[]
}

export interface FeatureInfo {
  member_count: number
  members: FeatureMember[]
  parent_feature: ParentFeatureInfo | null
  subfeatures: SubfeatureInfo[]
  connected_features: ConnectedFeatureInfo[]
}

export interface GlossaryFeatureAssociation {
  feature_id: string
  hash_id: string | null
  feature_label: string
  summary: string | null
  member_count: number
}

export interface GlossaryInfo {
  description: string
  total_count: number
  feature_count: number
  file_paths: string[]
  feature_associations: GlossaryFeatureAssociation[]
}

export interface ElementFeatureInfo {
  feature_id: string
  hash_id: string | null
  element_type: string
  label: string
  summary: string | null
  member_count: number
  parent_feature_label: string | null
  parent_feature_summary: string | null
}

export interface ElementFeaturesResponse {
  element_id: string
  features: ElementFeatureInfo[]
}

export interface ClassAttributeInfo {
  name: string
  type: string | null
  line: number | null
}

export interface ElementImportInfo {
  name: string
  module: string
  alias: string | null
  line: number
  is_internal: boolean
}

export interface DecoratorDetailInfo {
  name: string
  args: string | null
  full: string | null
}

export interface ComplexityInfo {
  cyclomatic: number
  nesting_depth: number
  branch_count: number
}

export interface CodeMetricsInfo {
  line_count: number
  param_count: number
  char_count: number
}

export interface DocstringQualityInfo {
  has_docstring: boolean
  has_params: boolean
  has_return: boolean
  coverage: number
}

export interface SecurityIssueInfo {
  kind: string
  pattern: string
  line: number
  severity: string
  message: string | null
}

export interface EnvVarInfo {
  name: string
  line: number
  access_type: string
}

export interface ConcurrencyInfo {
  is_async: boolean
  uses_locks: boolean
  uses_threads: boolean
  patterns: string[]
}

export interface TypeAnnotationInfo {
  name: string
  kind: string
  location: string
  line: number
  generic_args: string[] | null
}

export interface TodoInfo {
  kind: string
  text: string
  line: number
  assignee: string | null
  priority: string | null
  issue_ref: string | null
}

export interface SectionMarkerInfo {
  label: string
  line: number
  style: string
}

export interface CommentInfo {
  text: string
  line: number
  kind: string
  position: string
}

export interface PurityInfo {
  level: string
  confidence: string
  reasons: string[]
}

export interface SideEffectInfo {
  kind: string
  target: string | null
  line: number
}

export interface ParameterInfo {
  name: string
  type: string | null
  default: string | null
}

export interface MetricsSummaryInfo {
  total_elements: number
  total_functions: number
  avg_complexity: number
  max_complexity: number
  total_lines: number
  documented_pct: number
  async_count: number
  security_issue_count: number
  security_by_severity: Record<string, number>
}

export interface HttpRouteInfo {
  method: string
  path: string
  path_params: string[]
  framework: string | null
}

export interface CliCommandInfo {
  name: string
  full_command: string | null
  options: Array<{ name: string; type: string | null; required: boolean }>
  framework: string | null
}

export interface ElementDetail {
  element_id: string
  hash_id: string | null
  name: string
  element_type: string
  file_path: string
  line_start: number
  line_end: number | null
  language: string
  summary: string | null
  signature: string | null
  docstring: string | null
  raw_code: string | null
  decorators: string[]
  decorator_details: DecoratorDetailInfo[]
  visibility: string | null
  is_async: boolean
  is_test: boolean
  indexed_at: string | null
  base_classes: string[]
  class_attributes: ClassAttributeInfo[]
  return_type: string | null
  parameters: ParameterInfo[]
  exceptions_raised: string[]
  attributes_modified: string[]
  imports: ElementImportInfo[]
  element_count: number | null
  type_annotations: TypeAnnotationInfo[]
  detected_patterns: string[]
  pattern_confidence: Record<string, number>
  todos: TodoInfo[]
  section_markers: SectionMarkerInfo[]
  associated_comments: CommentInfo[]
  http_routes: HttpRouteInfo[]
  cli_commands: CliCommandInfo[]
  purity: PurityInfo | null
  side_effects: SideEffectInfo[]
  mutated_state: string[]
  is_public_api: boolean
  context_usages: string[]
  context: {
    file: ElementDetailFile | null
    parent: ElementDetailParent | null
    children: ElementDetailChild[]
    siblings: ElementDetailSibling[]
  }
  repository: {
    scope: string
    name: string
  }
  feature_info: FeatureInfo | null
  glossary_info: GlossaryInfo | null
  complexity: ComplexityInfo | null
  code_metrics: CodeMetricsInfo | null
  docstring_quality: DocstringQualityInfo | null
  security_issues: SecurityIssueInfo[]
  env_vars: EnvVarInfo[]
  concurrency: ConcurrencyInfo | null
  metrics_summary: MetricsSummaryInfo | null
}

export interface SimilarElement {
  element_id: string
  hash_id: string | null
  name: string
  element_type: string
  file_path: string
  line: number
  summary: string | null
  similarity: number
}

// =============================================================================
// Visualization Types
// =============================================================================

export interface VectorPoint {
  x: number
  y: number
  z?: number
  element_id: string
  hash_id: string | null
  name: string
  element_type: string
  file_path: string
  line: number
  summary: string
}

export interface VectorMapResponse {
  points: VectorPoint[]
  bounds: {
    x: [number, number]
    y: [number, number]
    z?: [number, number]
  }
  algorithm: string
  dimensions: number
  element_count: number
}

export interface ClusterMember {
  element_id: string
  name: string
  element_type: string
}

export interface Subfeature {
  subfeature_id: string
  label: string
  summary: string | null
  member_count: number
}

export interface Cluster {
  cluster_id: number
  size: number
  representative: {
    name: string
    element_type: string
    file_path: string
    summary: string | null
  }
  members: ClusterMember[]
  subfeatures: Subfeature[]
}

export interface ClustersResponse {
  clusters: Cluster[]
  total_elements: number
}

export interface FeatureConnection {
  source: string
  target: string
  affinity: number
  shared_member_count: number
}

export interface FeatureGraphNode {
  id: string
  hash_id: string | null
  label: string
  node_type: 'feature' | 'subfeature'
  summary: string | null
  member_count: number
  parent_id: string | null
}

export interface FeatureGraphResponse {
  nodes: FeatureGraphNode[]
  connections: FeatureConnection[]
  feature_count: number
  subfeature_count: number
}

// =============================================================================
// Admin Types
// =============================================================================

export interface ServiceHealth {
  status: string
  latency_ms: number | null
  details?: Record<string, unknown>
}

export interface HealthStatus {
  elasticsearch: ServiceHealth
  llm: ServiceHealth
  redis: ServiceHealth
}

export interface QueueStats {
  pending: number
  running: number
  completed: number
  failed: number
}

export interface JobStatsResponse {
  summarization: QueueStats
  embedding: QueueStats
}

export interface IndexStats {
  index_name: string
  document_count: number
  size_bytes: number
  size_human: string
  with_vectors: number
  vector_coverage_pct: number
}

export interface ToolUsageInfo {
  tool_name: string
  call_count: number
  percentage: number
}

export interface ToolTransitionInfo {
  from_tool: string
  to_tool: string
  count: number
  percentage: number
  confirmed_count: number
  confirmation_rate: number
}

export interface ToolDurationInfo {
  tool_name: string
  total_ms: number
  call_count: number
  avg_ms: number
}

export interface MCPAnalyticsResponse {
  total_calls: number
  unique_tools: number
  today_calls: number
  tool_usage: ToolUsageInfo[]
  top_transitions: ToolTransitionInfo[]
  transition_matrix: Record<string, Record<string, number>>
  tool_durations: ToolDurationInfo[]
}

export interface DailyActivityItem {
  date: string
  total_calls: number
  tool_counts: Record<string, number>
}

export interface MCPActivityHistoryResponse {
  days: number
  max_days: number
  daily_activity: DailyActivityItem[]
  total_calls: number
}

export interface TransitionDetailInfo {
  caller: string
  caller_input: string | null
  caller_output: string | null
  callee: string
  callee_input: string | null
  callee_output: string | null
  elapsed_ms: number | null
  caller_duration_ms: number | null
  callee_duration_ms: number | null
  timestamp: string | null
  is_consecutive: boolean
  is_causal: boolean
  causal_match_type: string | null
  causal_matched_value: string | null
}

export interface MCPTransitionDetailsResponse {
  transitions: TransitionDetailInfo[]
  from_tool: string | null
  to_tool: string | null
  total: number
}

export interface CausalLinkInfo {
  call_id: string | null
  tool_name: string
  matched_value: string | null
  match_type: 'parameter' | 'tool_suggestion'
}

export interface RecentToolCallInfo {
  call_id: string
  tool_name: string
  session_id: string
  input_args: string | null
  output_result: string | null
  start_time: string | null
  end_time: string | null
  duration_ms: number | null
  triggered_by: CausalLinkInfo | null
  suggested_tools: string[] | null
}

export interface MCPRecentCallsResponse {
  calls: RecentToolCallInfo[]
  total: number
}

export interface CausalPairInfo {
  source: string
  target: string
  count: number
}

export interface MCPCausalStatisticsResponse {
  total_causal_links: number
  total_calls: number
  causal_rate: number
  by_match_type: Record<string, number>
  top_causal_pairs: CausalPairInfo[]
}

// =============================================================================
// Browse Types
// =============================================================================

export interface BrowseFilters {
  scopes: string[]
  repositories: Array<{ scope: string; repository: string }>
  element_types: string[]
  languages: string[]
  usernames: string[]
}

export interface ContainerInfo {
  element_id: string
  hash_id: string | null
  name: string
  element_type: string
}

export interface BrowseSubfeature {
  element_id: string
  hash_id: string | null
  label: string
  member_count: number
}

export interface BrowseElement {
  element_id: string
  hash_id: string | null
  name: string
  element_type: string
  file_path: string
  line_start: number
  line_end: number | null
  language: string
  summary: string | null
  signature: string | null
  repository: string
  scope: string
  parent_id: string | null
  container: ContainerInfo | null
  visibility: string | null
  is_async: boolean
  has_docstring: boolean
  decorators: string[]
  level: number
  member_count?: number
  subfeatures?: BrowseSubfeature[]
  total_count?: number
  feature_count?: number
}

export interface BrowseResponse {
  elements: BrowseElement[]
  total: number
  page: number
  limit: number
  total_pages: number
}

export interface BrowseStats {
  type_counts: Record<string, number>
  language_counts: Record<string, number>
  total: number
}

export interface ElementChildren {
  element_id: string
  hash_id: string | null
  children: Record<string, Array<{
    element_id: string
    hash_id: string | null
    name: string
    element_type: string
    line_start: number
    line_end: number | null
    summary: string | null
    signature: string | null
    visibility: string | null
    is_async: boolean
  }>>
  total_children: number
}

export interface CallGraphEntry {
  element_id: string
  hash_id: string | null
  name: string
  element_type: string
  file_path: string
  line_start: number
}

export interface ElementDetailsResponse {
  element_id: string
  hash_id: string | null
  name: string
  element_type: string
  file_path: string
  line_start: number
  line_end: number | null
  language: string
  summary: string | null
  signature: string | null
  docstring: string | null
  visibility: string | null
  is_async: boolean
  decorators: string[]
  level: number
  repository: string
  scope: string
  containers: Array<{
    element_id: string
    hash_id: string | null
    name: string
    element_type: string
    file_path: string
    line_start: number
  }>
  child_count: number
  callers?: CallGraphEntry[]
  callees?: CallGraphEntry[]
  error?: string
}

// =============================================================================
// Glossary Types
// =============================================================================

export interface GlossaryTermSummary {
  term: string
  hash_id: string | null
  description: string
  total_count: number
  file_count: number
  feature_count: number
}

export interface FeatureAssociationSummary {
  feature_id: string
  feature_label: string
  frequency: number
  total_members: number
  percentage: number
}

export interface GlossaryTermDetail {
  term: string
  description: string
  total_count: number
  element_ids: string[]
  file_paths: string[]
  feature_associations: FeatureAssociationSummary[]
}

export interface GlossaryListResponse {
  terms: GlossaryTermSummary[]
  total: number
}

// =============================================================================
// Analysis Types
// =============================================================================

export interface CallerInfo {
  element_id: string
  hash_id: string | null
  name: string
  element_type: string
  file_path: string
  line: number
  summary: string | null
}

export interface CalleeInfo {
  name: string
  receiver: string | null
  line: number
  element_id: string | null
  hash_id: string | null
  element_type: string | null
  file_path: string | null
  target_line: number | null
  summary: string | null
}

export interface CallGraphResponse {
  element: {
    element_id: string
    hash_id: string | null
    name: string
    element_type: string
    file_path: string | null
    line: number | null
  }
  callers: CallerInfo[]
  callees: CalleeInfo[]
}

export interface CallChainNode {
  element_id: string | null
  hash_id: string | null
  name: string
  element_type: string | null
  file_path: string | null
  line: number | null
  summary: string | null
  callers: CallChainNode[]
  callees: CallChainNode[]
  cycle: boolean
  unresolved: boolean
  missing: boolean
}

export interface CallChainRoot {
  element_id: string
  hash_id: string | null
  name: string
  element_type: string
  file_path: string
  line: number
  summary: string | null
}

export interface CallChainResponse {
  root: CallChainRoot
  direction: string
  max_depth: number
  callers: CallChainNode[]
  callees: CallChainNode[]
}

export interface DeadCodeItem {
  element_id: string
  hash_id: string | null
  name: string
  element_type: string
  file_path: string
  line: number
  summary: string | null
  is_test: boolean
}

export interface DeadCodeResponse {
  potentially_dead: DeadCodeItem[]
  stats: {
    total_functions: number
    excluded_entry_points: number
    called: number
    potentially_dead: number
  }
}

export interface EntryPointItem {
  element_id: string
  hash_id: string | null
  name: string
  element_type: string
  file_path: string
  line: number
  summary: string | null
  decorators: string[]
  http_routes: HttpRouteInfo[]
  cli_commands: CliCommandInfo[]
}

export interface EntryPointsResponse {
  http: EntryPointItem[]
  cli: EntryPointItem[]
  test: EntryPointItem[]
  main: EntryPointItem[]
  async_tasks: EntryPointItem[]
  stats: {
    total_http: number
    total_cli: number
    total_test: number
    total_main: number
    total_async: number
    total: number
  }
}

export interface ImportInfo {
  name: string
  module: string
  alias: string | null
  line: number
  is_internal: boolean
}

export interface DependenciesResponse {
  file_info: Record<string, unknown>
  internal_imports: ImportInfo[]
  external_imports: ImportInfo[]
  stats: {
    total: number
    internal: number
    external: number
  }
}

export interface DependentInfo {
  element_id: string
  hash_id: string | null
  file_path: string
  name: string
}

export interface DependentsResponse {
  module: string
  dependents: DependentInfo[]
  total: number
}

export interface DependencyGraphResponse {
  nodes: string[]
  edges: Array<{ from: string; to: string }>
  cycles: string[][]
  stats: {
    node_count: number
    edge_count: number
    cycle_count: number
    has_cycles: boolean
  }
}

export interface SimilarCodeItem {
  element_id: string
  hash_id: string | null
  name: string
  element_type: string
  file_path: string
  line: number
  summary: string | null
  similarity: number
}

export interface ExplainElementResponse {
  element: {
    element_id: string
    name: string
    type: string
    file: string
    line: number
    signature: string | null
    summary: string | null
    docstring: string | null
    decorators: string[]
    is_test: boolean
  }
  callers: CallerInfo[]
  callees: CalleeInfo[]
  imports: ImportInfo[]
  similar_code: SimilarCodeItem[]
  parent: {
    element_id: string
    name: string
    type: string
    file: string
    line: number
    summary: string | null
  } | null
  embedding_status: {
    has_summary: boolean
    has_code: boolean
  }
}

// =============================================================================
// Code Metrics Types
// =============================================================================

export interface ComplexFunctionItem {
  element_id: string
  hash_id: string | null
  name: string
  element_type: string
  file_path: string
  line: number
  cyclomatic: number
  nesting_depth: number
  branch_count: number
  line_count: number
  summary: string | null
  is_test: boolean
}

export interface ComplexFunctionsResponse {
  functions: ComplexFunctionItem[]
  count: number
  min_complexity: number
}

export interface SecurityIssueDetailItem {
  element_id: string
  hash_id: string | null
  function_name: string
  element_type: string
  file_path: string
  function_line: number
  issue_line: number
  severity: string
  kind: string
  message: string | null
}

export interface SecurityIssuesResponse {
  issues: SecurityIssueDetailItem[]
  count: number
  by_severity: Record<string, number>
  by_kind: Record<string, number>
}

export interface UndocumentedItem {
  element_id: string
  hash_id: string | null
  name: string
  element_type: string
  file_path: string
  line: number
  visibility: string | null
  has_docstring: boolean
  has_params: boolean
  has_return: boolean
  coverage: number
  param_count: number
  summary: string | null
}

export interface UndocumentedResponse {
  functions: UndocumentedItem[]
  count: number
  max_coverage: number
}

export interface EnvUsageDetailItem {
  element_id: string
  hash_id: string | null
  function_name: string
  element_type: string
  file_path: string
  function_line: number
  env_name: string
  env_line: number
  access_type: string
}

export interface EnvUsageResponse {
  usages: EnvUsageDetailItem[]
  count: number
  unique_vars: number
  by_var: Record<string, number>
}

export interface AsyncCodeDetailItem {
  element_id: string
  hash_id: string | null
  name: string
  element_type: string
  file_path: string
  line: number
  is_async: boolean
  uses_threads: boolean
  uses_locks: boolean
  patterns: string[]
  summary: string | null
}

export interface AsyncCodeResponse {
  functions: AsyncCodeDetailItem[]
  count: number
  pattern_filter: string
  by_pattern: Record<string, number>
}
