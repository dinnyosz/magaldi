/**
 * Search API functions
 */

import { API_BASE, fetchJson } from './client'
import type { SearchRequest, SearchResponse, SearchSummaryRequest, SearchSummaryResponse } from './types'

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
