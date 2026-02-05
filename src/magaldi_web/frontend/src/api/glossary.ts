/**
 * Glossary API functions
 */

import { API_BASE, fetchJson } from './client'
import type { GlossaryListResponse, GlossaryTermDetail } from './types'

export async function getGlossaryTerms(
  scope: string,
  repository: string,
  minCount: number = 1
): Promise<GlossaryListResponse> {
  return fetchJson(`${API_BASE}/repos/${scope}/${repository}/glossary?min_count=${minCount}`)
}

export async function getGlossaryTerm(
  scope: string,
  repository: string,
  term: string
): Promise<GlossaryTermDetail> {
  return fetchJson(`${API_BASE}/repos/${scope}/${repository}/glossary/${encodeURIComponent(term)}`)
}

export async function searchGlossaryTerms(
  scope: string,
  repository: string,
  query: string
): Promise<GlossaryListResponse> {
  return fetchJson(`${API_BASE}/repos/${scope}/${repository}/glossary/search/${encodeURIComponent(query)}`)
}

export async function getGlossaryTermsForFeature(
  featureId: string
): Promise<GlossaryListResponse> {
  return fetchJson(`${API_BASE}/glossary/by-feature/${encodeURIComponent(featureId)}`)
}
