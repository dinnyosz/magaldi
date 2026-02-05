/**
 * Admin API functions
 */

import { API_BASE, fetchJson } from './client'
import type {
  HealthStatus,
  JobStatsResponse,
  IndexStats,
  MCPAnalyticsResponse,
  MCPActivityHistoryResponse,
  MCPTransitionDetailsResponse,
  MCPRecentCallsResponse,
  MCPCausalStatisticsResponse,
} from './types'

export async function getHealth(): Promise<HealthStatus> {
  return fetchJson(`${API_BASE}/admin/health`)
}

export async function getJobs(): Promise<JobStatsResponse> {
  return fetchJson(`${API_BASE}/admin/jobs`)
}

export async function getIndexStats(): Promise<IndexStats> {
  return fetchJson(`${API_BASE}/admin/index-stats`)
}

export async function getMCPAnalytics(): Promise<MCPAnalyticsResponse> {
  return fetchJson(`${API_BASE}/admin/mcp-analytics`)
}

export async function getMCPActivityHistory(days: number = 7): Promise<MCPActivityHistoryResponse> {
  return fetchJson(`${API_BASE}/admin/mcp-analytics/history?days=${days}`)
}

export async function clearMCPAnalytics(): Promise<{ status: string; message?: string }> {
  return fetchJson(`${API_BASE}/admin/mcp-analytics/clear`, { method: 'POST' })
}

export async function getMCPTransitionDetails(params?: {
  from_tool?: string
  to_tool?: string
  limit?: number
}): Promise<MCPTransitionDetailsResponse> {
  const searchParams = new URLSearchParams()
  if (params?.from_tool) searchParams.set('from_tool', params.from_tool)
  if (params?.to_tool) searchParams.set('to_tool', params.to_tool)
  if (params?.limit) searchParams.set('limit', String(params.limit))
  const query = searchParams.toString() ? `?${searchParams.toString()}` : ''
  return fetchJson(`${API_BASE}/admin/mcp-analytics/transition-details${query}`)
}

export async function getMCPRecentCalls(params?: {
  limit?: number
  tool_name?: string
  only_causal?: boolean
}): Promise<MCPRecentCallsResponse> {
  const searchParams = new URLSearchParams()
  if (params?.limit) searchParams.set('limit', String(params.limit))
  if (params?.tool_name) searchParams.set('tool_name', params.tool_name)
  if (params?.only_causal) searchParams.set('only_causal', 'true')
  const query = searchParams.toString() ? `?${searchParams.toString()}` : ''
  return fetchJson(`${API_BASE}/admin/mcp-analytics/recent-calls${query}`)
}

export async function getMCPCausalStatistics(): Promise<MCPCausalStatisticsResponse> {
  return fetchJson(`${API_BASE}/admin/mcp-analytics/causal`)
}
