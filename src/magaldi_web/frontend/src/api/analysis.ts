/**
 * Analysis and Code Metrics API functions
 */

import { API_BASE, fetchJson } from './client'
import type {
  CallGraphResponse,
  CallChainResponse,
  DeadCodeResponse,
  EntryPointsResponse,
  DependenciesResponse,
  DependentsResponse,
  DependencyGraphResponse,
  ExplainElementResponse,
  ComplexFunctionsResponse,
  SecurityIssuesResponse,
  UndocumentedResponse,
  EnvUsageResponse,
  AsyncCodeResponse,
} from './types'

// =============================================================================
// Call Graph Analysis
// =============================================================================

export async function getElementCallers(
  hashId: string,
  options?: { limit?: number; include_tests?: boolean }
): Promise<CallGraphResponse> {
  const params = new URLSearchParams()
  if (options?.limit) params.set('limit', String(options.limit))
  if (options?.include_tests !== undefined) params.set('include_tests', String(options.include_tests))
  const query = params.toString() ? `?${params.toString()}` : ''
  return fetchJson(`${API_BASE}/analysis/callers/${encodeURIComponent(hashId)}${query}`)
}

export async function getCallChain(
  hashId: string,
  options?: { direction?: 'callers' | 'callees' | 'both'; max_depth?: number }
): Promise<CallChainResponse> {
  const params = new URLSearchParams()
  if (options?.direction) params.set('direction', options.direction)
  if (options?.max_depth) params.set('max_depth', String(options.max_depth))
  const query = params.toString() ? `?${params.toString()}` : ''
  return fetchJson(`${API_BASE}/analysis/call-chain/${encodeURIComponent(hashId)}${query}`)
}

// =============================================================================
// Code Structure Analysis
// =============================================================================

export async function getDeadCode(
  scope: string,
  repository: string,
  options?: { username?: string; include_tests?: boolean }
): Promise<DeadCodeResponse> {
  const params = new URLSearchParams()
  params.set('scope', scope)
  params.set('repository', repository)
  if (options?.username) params.set('username', options.username)
  if (options?.include_tests !== undefined) params.set('include_tests', String(options.include_tests))
  return fetchJson(`${API_BASE}/analysis/dead-code?${params.toString()}`)
}

export async function getEntryPoints(
  scope: string,
  repository: string,
  username?: string
): Promise<EntryPointsResponse> {
  const params = new URLSearchParams()
  params.set('scope', scope)
  params.set('repository', repository)
  if (username) params.set('username', username)
  return fetchJson(`${API_BASE}/analysis/entry-points?${params.toString()}`)
}

// =============================================================================
// Dependency Analysis
// =============================================================================

export async function getFileDependencies(hashId: string): Promise<DependenciesResponse> {
  return fetchJson(`${API_BASE}/analysis/dependencies/${encodeURIComponent(hashId)}`)
}

export async function getModuleDependents(
  module: string,
  scope: string,
  repository: string,
  options?: { username?: string; limit?: number }
): Promise<DependentsResponse> {
  const params = new URLSearchParams()
  params.set('module', module)
  params.set('scope', scope)
  params.set('repository', repository)
  if (options?.username) params.set('username', options.username)
  if (options?.limit) params.set('limit', String(options.limit))
  return fetchJson(`${API_BASE}/analysis/dependents?${params.toString()}`)
}

export async function getDependencyGraph(
  scope: string,
  repository: string,
  options?: { username?: string; internal_only?: boolean }
): Promise<DependencyGraphResponse> {
  const params = new URLSearchParams()
  params.set('scope', scope)
  params.set('repository', repository)
  if (options?.username) params.set('username', options.username)
  if (options?.internal_only !== undefined) params.set('internal_only', String(options.internal_only))
  return fetchJson(`${API_BASE}/analysis/dependency-graph?${params.toString()}`)
}

// =============================================================================
// Element Explanation
// =============================================================================

export async function explainElement(hashId: string): Promise<ExplainElementResponse> {
  return fetchJson(`${API_BASE}/analysis/explain/${encodeURIComponent(hashId)}`)
}

// =============================================================================
// Code Metrics
// =============================================================================

export async function getComplexFunctions(
  scope: string,
  repository: string,
  options?: { min_complexity?: number; limit?: number; include_tests?: boolean }
): Promise<ComplexFunctionsResponse> {
  const params = new URLSearchParams()
  params.set('scope', scope)
  params.set('repository', repository)
  if (options?.min_complexity) params.set('min_complexity', String(options.min_complexity))
  if (options?.limit) params.set('limit', String(options.limit))
  if (options?.include_tests !== undefined) params.set('include_tests', String(options.include_tests))
  return fetchJson(`${API_BASE}/analysis/complexity?${params.toString()}`)
}

export async function getSecurityIssues(
  scope: string,
  repository: string,
  options?: { severity?: string; limit?: number }
): Promise<SecurityIssuesResponse> {
  const params = new URLSearchParams()
  params.set('scope', scope)
  params.set('repository', repository)
  if (options?.severity) params.set('severity', options.severity)
  if (options?.limit) params.set('limit', String(options.limit))
  return fetchJson(`${API_BASE}/analysis/security-issues?${params.toString()}`)
}

export async function getUndocumentedFunctions(
  scope: string,
  repository: string,
  options?: { max_coverage?: number; limit?: number; include_tests?: boolean }
): Promise<UndocumentedResponse> {
  const params = new URLSearchParams()
  params.set('scope', scope)
  params.set('repository', repository)
  if (options?.max_coverage !== undefined) params.set('max_coverage', String(options.max_coverage))
  if (options?.limit) params.set('limit', String(options.limit))
  if (options?.include_tests !== undefined) params.set('include_tests', String(options.include_tests))
  return fetchJson(`${API_BASE}/analysis/documentation-coverage?${params.toString()}`)
}

export async function getEnvUsage(
  scope: string,
  repository: string,
  options?: { limit?: number }
): Promise<EnvUsageResponse> {
  const params = new URLSearchParams()
  params.set('scope', scope)
  params.set('repository', repository)
  if (options?.limit) params.set('limit', String(options.limit))
  return fetchJson(`${API_BASE}/analysis/env-usage?${params.toString()}`)
}

export async function getAsyncCode(
  scope: string,
  repository: string,
  options?: { pattern?: string; limit?: number }
): Promise<AsyncCodeResponse> {
  const params = new URLSearchParams()
  params.set('scope', scope)
  params.set('repository', repository)
  if (options?.pattern) params.set('pattern', options.pattern)
  if (options?.limit) params.set('limit', String(options.limit))
  return fetchJson(`${API_BASE}/analysis/concurrency?${params.toString()}`)
}
