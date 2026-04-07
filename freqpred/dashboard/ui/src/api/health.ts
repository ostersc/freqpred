import apiFetch from './client'
import type { SystemHealthResponse } from './types'

export function getSystemHealth() {
  return apiFetch<SystemHealthResponse>('/api/system/health')
}
