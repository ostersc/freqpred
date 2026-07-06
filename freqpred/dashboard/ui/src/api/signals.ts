import apiFetch from './client'
import type { SignalListResponse, SignalDetailOut } from './types'

export interface GetSignalsParams {
  limit?: number
  offset?: number
  market_id?: string
  direction?: string
  min_edge?: number
  max_edge?: number
  min_confidence?: number
  max_confidence?: number
  has_factbase?: boolean
  has_docs?: boolean
  trigger?: string
  series_ticker?: string
}

export function getSignals(params?: GetSignalsParams) {
  const q = new URLSearchParams()
  if (params?.limit !== undefined) q.set('limit', String(params.limit))
  if (params?.offset !== undefined) q.set('offset', String(params.offset))
  if (params?.market_id) q.set('market_id', params.market_id)
  if (params?.direction) q.set('direction', params.direction)
  if (params?.min_edge !== undefined) q.set('min_edge', String(params.min_edge))
  if (params?.max_edge !== undefined) q.set('max_edge', String(params.max_edge))
  if (params?.min_confidence !== undefined) q.set('min_confidence', String(params.min_confidence))
  if (params?.max_confidence !== undefined) q.set('max_confidence', String(params.max_confidence))
  if (params?.has_factbase !== undefined) q.set('has_factbase', String(params.has_factbase))
  if (params?.has_docs !== undefined) q.set('has_docs', String(params.has_docs))
  if (params?.trigger) q.set('trigger', params.trigger)
  if (params?.series_ticker) q.set('series_ticker', params.series_ticker)
  return apiFetch<SignalListResponse>(`/api/signals?${q}`)
}

export function getSignal(id: string) {
  return apiFetch<SignalDetailOut>(`/api/signals/${id}`)
}
