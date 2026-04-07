import apiFetch from './client'
import type { SignalListResponse, SignalDetailOut } from './types'

export function getSignals(params?: { limit?: number; offset?: number; market_id?: string; direction?: string }) {
  const q = new URLSearchParams()
  if (params?.limit !== undefined) q.set('limit', String(params.limit))
  if (params?.offset !== undefined) q.set('offset', String(params.offset))
  if (params?.market_id) q.set('market_id', params.market_id)
  if (params?.direction) q.set('direction', params.direction)
  return apiFetch<SignalListResponse>(`/api/signals?${q}`)
}

export function getSignal(id: string) {
  return apiFetch<SignalDetailOut>(`/api/signals/${id}`)
}
