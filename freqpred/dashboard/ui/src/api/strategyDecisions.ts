import apiFetch from './client'
import type { StrategyDecisionListResponse } from './types'

export interface GetStrategyDecisionsParams {
  strategy?: string
  exitReason?: string
  tickerPrefix?: string
  dateFrom?: string // ISO date
  dateTo?: string   // ISO date
  limit?: number
  offset?: number
}

export function getStrategyDecisions(params: GetStrategyDecisionsParams = {}) {
  const q = new URLSearchParams()
  if (params.strategy) q.set('strategy', params.strategy)
  if (params.exitReason) q.set('exit_reason', params.exitReason)
  if (params.tickerPrefix) q.set('ticker_prefix', params.tickerPrefix)
  if (params.dateFrom) q.set('date_from', params.dateFrom)
  if (params.dateTo) q.set('date_to', params.dateTo)
  if (params.limit !== undefined) q.set('limit', String(params.limit))
  if (params.offset !== undefined) q.set('offset', String(params.offset))
  const qs = q.toString()
  return apiFetch<StrategyDecisionListResponse>(
    `/api/strategy-decisions${qs ? `?${qs}` : ''}`,
  )
}
