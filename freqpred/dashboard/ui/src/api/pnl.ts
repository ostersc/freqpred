import apiFetch from './client'

export interface PnLDayOut {
  date: string
  daily_pnl: number
  cumulative_pnl: number
  trade_count: number
}

export interface LLMSpendDayOut {
  date: string
  daily_spend: number
  cumulative_spend: number
}

export interface PromptVersionStart {
  version: string
  date: string
}

export interface PnLTimeSeriesResponse {
  pnl_series: PnLDayOut[]
  llm_series: LLMSpendDayOut[]
  prompt_version_starts: PromptVersionStart[]
  initial_bankroll: number
  /** initial_bankroll + all-time closed P&L for the active mode. Not window-scoped. */
  net_bankroll_now: number
  total_trades: number
  all_time_pnl: number
  available_strategies: string[]
  available_models: string[]
  available_prompt_versions: string[]
  available_directions: string[]
  available_categories: string[]
  available_series_tickers: string[]
  available_market_ids: string[]
}

export interface PnLFilters {
  lookbackDays?: number
  strategy?: string
  modelUsed?: string
  promptVersion?: string
  direction?: string
  category?: string
  seriesTicker?: string
  marketId?: string
}

export function getPnLTimeSeries(filters: PnLFilters = {}): Promise<PnLTimeSeriesResponse> {
  const q = new URLSearchParams()
  if (filters.lookbackDays != null) q.set('lookback_days', String(filters.lookbackDays))
  if (filters.strategy)      q.set('strategy', filters.strategy)
  if (filters.modelUsed)     q.set('model_used', filters.modelUsed)
  if (filters.promptVersion) q.set('prompt_version', filters.promptVersion)
  if (filters.direction)     q.set('direction', filters.direction)
  if (filters.category)      q.set('category', filters.category)
  if (filters.seriesTicker)  q.set('series_ticker', filters.seriesTicker)
  if (filters.marketId)      q.set('market_id', filters.marketId)
  const qs = q.toString()
  return apiFetch<PnLTimeSeriesResponse>(`/api/pnl/time-series${qs ? `?${qs}` : ''}`)
}
