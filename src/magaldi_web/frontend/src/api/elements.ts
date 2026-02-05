/**
 * Element API functions
 */

import { API_BASE, fetchJson } from './client'
import type { ElementDetail, SimilarElement, ElementFeaturesResponse } from './types'

export async function getElement(hashId: string): Promise<ElementDetail> {
  return fetchJson(`${API_BASE}/elements/${encodeURIComponent(hashId)}`)
}

export async function getSimilarElements(hashId: string, limit = 10): Promise<SimilarElement[]> {
  return fetchJson(`${API_BASE}/elements/similar/${encodeURIComponent(hashId)}?limit=${limit}`)
}

export async function getElementFeatures(hashId: string): Promise<ElementFeaturesResponse> {
  return fetchJson(`${API_BASE}/elements/${encodeURIComponent(hashId)}/features`)
}
