import apiFetch from './client'
import type { SystemHealthResponse } from './types'

export function getSystemHealth() {
  return apiFetch<SystemHealthResponse>('/api/system/health')
}

export function upgradeApiTier() {
  return apiFetch<{ ok: boolean }>('/api/system/api-tier/upgrade', { method: 'POST' })
}
