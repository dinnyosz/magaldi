/**
 * Browse API functions
 */

import { API_BASE, fetchJson } from './client'
import type {
  BrowseFilters,
  BrowseResponse,
  BrowseStats,
  ElementChildren,
  ElementDetailsResponse,
} from './types'

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
  return fetchJson(`${API_BASE}/browse/element/${encodeURIComponent(hashId)}/children${params}`)
}

export async function getElementDetails(
  hashId: string,
  options?: { includeCallGraph?: boolean; username?: string }
): Promise<ElementDetailsResponse> {
  const params = new URLSearchParams()
  if (options?.includeCallGraph) params.set('include_call_graph', 'true')
  if (options?.username) params.set('username', options.username)
  const query = params.toString() ? `?${params.toString()}` : ''
  return fetchJson(`${API_BASE}/browse/element/${encodeURIComponent(hashId)}/details${query}`)
}
