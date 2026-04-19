import apiFetch from './client'
import type { SourceQualityListResponse } from './types'

export function getSourceQuality(params?: { category?: string; lookback_days?: number }) {
  const q = new URLSearchParams()
  if (params?.category) q.set('category', params.category)
  if (params?.lookback_days !== undefined) q.set('lookback_days', String(params.lookback_days))
  const suffix = q.size > 0 ? `?${q}` : ''
  return apiFetch<SourceQualityListResponse>(`/api/metrics/source-quality${suffix}`)
}
