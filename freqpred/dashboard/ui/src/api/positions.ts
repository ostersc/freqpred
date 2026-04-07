import apiFetch from './client'
import type { PositionListResponse, PositionOut } from './types'

export function getPositions(status: 'open' | 'closed' | 'all' = 'all') {
  return apiFetch<PositionListResponse>(`/api/positions?status=${status}`)
}

export function getPosition(id: string) {
  return apiFetch<PositionOut>(`/api/positions/${id}`)
}
