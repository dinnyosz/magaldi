/**
 * API client for Magaldi backend
 */

const API_BASE = '/api/v1'

export interface DashboardStats {
  stats: {
    repository_count: number
    element_count: number
    file_count: number
    class_count: number
    function_count: number
    method_count: number
    variable_count: number
    constant_count: number
    feature_count: number
    subfeature_count: number
  }
  recent_repos: Array<{
    scope: string
    name: string
    description: string | null
    file_count: number
    class_count: number
    function_count: number
    method_count: number
    variable_count: number
    constant_count: number
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
  combined_score: number | null  // Sum of enabled scores, used for sorting
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
  visibility: string | null
  is_async: boolean
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

export interface VectorPoint {
  x: number
  y: number
  z?: number
  element_id: string
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

// API functions

async function fetchJson<T>(url: string, options?: RequestInit): Promise<T> {
  const response = await fetch(url, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options?.headers,
    },
  })
  if (!response.ok) {
    throw new Error(`API error: ${response.status} ${response.statusText}`)
  }
  return response.json()
}

// Dashboard
export async function getDashboard(): Promise<DashboardStats> {
  return fetchJson<DashboardStats>(`${API_BASE}/dashboard`)
}

// Search
export async function search(params: SearchRequest): Promise<SearchResponse> {
  return fetchJson<SearchResponse>(`${API_BASE}/search`, {
    method: 'POST',
    body: JSON.stringify(params),
  })
}

export async function generateSearchSummary(params: SearchSummaryRequest): Promise<SearchSummaryResponse> {
  return fetchJson<SearchSummaryResponse>(`${API_BASE}/search/summary`, {
    method: 'POST',
    body: JSON.stringify(params),
  })
}

// Repositories
export interface RepoListItem {
  scope: string
  name: string
  file_count: number
  element_count: number
  languages: string[]
}

interface RepoListResponse {
  repos: RepoListItem[]
  total: number
}

export async function getRepositories(): Promise<RepoListItem[]> {
  const response = await fetchJson<RepoListResponse>(`${API_BASE}/repos`)
  return response.repos
}

export async function getFileTree(scope: string, repository: string): Promise<FileTreeResponse> {
  return fetchJson(`${API_BASE}/repos/${scope}/${repository}/tree`)
}

export async function getFileDetail(scope: string, repository: string, filePath: string): Promise<FileDetailResponse> {
  const encodedPath = encodeURIComponent(filePath)
  return fetchJson(`${API_BASE}/repos/${scope}/${repository}/files/${encodedPath}`)
}

// Elements
export async function getElement(hashId: string): Promise<ElementDetail> {
  return fetchJson(`${API_BASE}/elements/${hashId}`)
}

export async function getSimilarElements(hashId: string, limit = 10): Promise<SimilarElement[]> {
  return fetchJson(`${API_BASE}/elements/similar/${hashId}?limit=${limit}`)
}

// Vector visualization
export async function getVectorMap(
  scope: string,
  repository: string,
  options?: {
    element_types?: string[]
    dimensions?: number
    algorithm?: 'umap' | 'tsne'
    limit?: number
  }
): Promise<VectorMapResponse> {
  const params = new URLSearchParams()
  if (options?.element_types) {
    options.element_types.forEach(t => params.append('element_types', t))
  }
  if (options?.dimensions) params.set('dimensions', String(options.dimensions))
  if (options?.algorithm) params.set('algorithm', options.algorithm)
  if (options?.limit) params.set('limit', String(options.limit))

  const query = params.toString() ? `?${params.toString()}` : ''
  return fetchJson(`${API_BASE}/repos/${scope}/${repository}/vector-map${query}`)
}

export async function getClusters(scope: string, repository: string, limit = 50): Promise<ClustersResponse> {
  return fetchJson(`${API_BASE}/repos/${scope}/${repository}/clusters?limit=${limit}`)
}

// Admin
export async function getHealth(): Promise<HealthStatus> {
  return fetchJson(`${API_BASE}/admin/health`)
}

export async function getJobs(): Promise<JobStatsResponse> {
  return fetchJson(`${API_BASE}/admin/jobs`)
}

export async function getIndexStats(): Promise<IndexStats> {
  return fetchJson(`${API_BASE}/admin/index-stats`)
}

// Browse API

export interface BrowseFilters {
  scopes: string[]
  repositories: Array<{ scope: string; repository: string }>
  element_types: string[]
  languages: string[]
  usernames: string[]
}

export interface ContainerInfo {
  element_id: string
  name: string
  element_type: string
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

export async function getBrowseFilters(): Promise<BrowseFilters> {
  return fetchJson(`${API_BASE}/browse/filters`)
}

export async function browseElements(params: {
  scope?: string
  repository?: string
  username?: string
  element_type?: string
  parent_id?: string
  language?: string
  page?: number
  limit?: number
}): Promise<BrowseResponse> {
  const searchParams = new URLSearchParams()
  if (params.scope) searchParams.set('scope', params.scope)
  if (params.repository) searchParams.set('repository', params.repository)
  if (params.username) searchParams.set('username', params.username)
  if (params.element_type) searchParams.set('element_type', params.element_type)
  if (params.parent_id) searchParams.set('parent_id', params.parent_id)
  if (params.language) searchParams.set('language', params.language)
  if (params.page) searchParams.set('page', String(params.page))
  if (params.limit) searchParams.set('limit', String(params.limit))

  const query = searchParams.toString() ? `?${searchParams.toString()}` : ''
  return fetchJson(`${API_BASE}/browse/elements${query}`)
}

export async function getBrowseStats(params: {
  scope?: string
  repository?: string
  username?: string
}): Promise<BrowseStats> {
  const searchParams = new URLSearchParams()
  if (params.scope) searchParams.set('scope', params.scope)
  if (params.repository) searchParams.set('repository', params.repository)
  if (params.username) searchParams.set('username', params.username)

  const query = searchParams.toString() ? `?${searchParams.toString()}` : ''
  return fetchJson(`${API_BASE}/browse/stats${query}`)
}

export async function getElementChildren(hashId: string, username?: string): Promise<ElementChildren> {
  const params = username ? `?username=${username}` : ''
  return fetchJson(`${API_BASE}/browse/element/${hashId}/children${params}`)
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

export async function getElementDetails(
  hashId: string,
  options?: { includeCallGraph?: boolean; username?: string }
): Promise<ElementDetailsResponse> {
  const params = new URLSearchParams()
  if (options?.includeCallGraph) params.set('include_call_graph', 'true')
  if (options?.username) params.set('username', options.username)
  const query = params.toString() ? `?${params.toString()}` : ''
  return fetchJson(`${API_BASE}/browse/element/${hashId}/details${query}`)
}
