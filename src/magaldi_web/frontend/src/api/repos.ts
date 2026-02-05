/**
 * Repository API functions
 */

import { API_BASE, fetchJson } from './client'
import type {
  RepoListItem,
  RepoListResponse,
  FileTreeResponse,
  FileDetailResponse,
  VectorMapResponse,
  ClustersResponse,
  FeatureGraphResponse,
} from './types'

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

export async function getFeatureGraph(scope: string, repository: string): Promise<FeatureGraphResponse> {
  return fetchJson(`${API_BASE}/repos/${scope}/${repository}/feature-graph`)
}
