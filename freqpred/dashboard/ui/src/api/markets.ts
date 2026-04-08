import apiFetch from './client'
import type { MarketListResponse, MarketDetailOut, AnalyzeResponse } from './types'

export interface GetMarketsParams {
  search?: string
  status?: 'open' | 'closed' | 'all'
  limit?: number
  offset?: number
}

export function getMarkets(params: GetMarketsParams = {}) {
  const q = new URLSearchParams()
  if (params.search) q.set('search', params.search)
  if (params.status) q.set('status', params.status)
  if (params.limit !== undefined) q.set('limit', String(params.limit))
  if (params.offset !== undefined) q.set('offset', String(params.offset))
  const qs = q.toString()
  return apiFetch<MarketListResponse>(`/api/markets${qs ? `?${qs}` : ''}`)
}

export function getMarket(id: string) {
  return apiFetch<MarketDetailOut>(`/api/markets/${encodeURIComponent(id)}`)
}

export function analyzeMarket(id: string) {
  return apiFetch<AnalyzeResponse>(`/api/markets/${encodeURIComponent(id)}/analyze`, {
    method: 'POST',
  })
}
