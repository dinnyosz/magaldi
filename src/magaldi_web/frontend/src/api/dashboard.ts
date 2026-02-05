/**
 * Dashboard API functions
 */

import { API_BASE, fetchJson } from './client'
import type { DashboardStats } from './types'

export async function getDashboard(): Promise<DashboardStats> {
  return fetchJson<DashboardStats>(`${API_BASE}/dashboard`)
}
