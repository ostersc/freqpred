import apiFetch from './client'
import type { PositionListResponse, PositionOut, PositionDetailOut } from './types'

export function getPositions(status: 'open' | 'closed' | 'all' = 'all') {
  return apiFetch<PositionListResponse>(`/api/positions?status=${status}`)
}

export function getPosition(id: string) {
  return apiFetch<PositionOut>(`/api/positions/${id}`)
}

export function getPositionDetail(id: string) {
  return apiFetch<PositionDetailOut>(`/api/positions/${id}/detail`)
}

export function forceExitPosition(id: string): Promise<PositionOut> {
  return apiFetch<PositionOut>(
    `/api/positions/${encodeURIComponent(id)}/force-exit`,
    { method: 'POST' },
  )
}
