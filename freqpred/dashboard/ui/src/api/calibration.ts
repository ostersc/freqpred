import apiFetch from './client'
import type {
  CalibrationHeatmapResponse,
  CalibrationResponse,
  CalibrationTimeSeriesResponse,
} from './types'

export interface CalibrationFilters {
  lookbackDays?: number
  category?: string
  tickerPrefix?: string
  direction?: string
  modelUsed?: string
  promptVersion?: string
  seriesTicker?: string
  minConfidence?: number
  maxConfidence?: number
}

function _buildParams(filters: Omit<CalibrationFilters, 'promptVersion'> & { promptVersion?: string }): URLSearchParams {
  const q = new URLSearchParams()
  if (filters.lookbackDays != null) q.set('lookback_days', String(filters.lookbackDays))
  if (filters.category) q.set('category', filters.category)
  if (filters.tickerPrefix) q.set('ticker_prefix', filters.tickerPrefix)
  if (filters.direction) q.set('direction', filters.direction)
  if (filters.modelUsed) q.set('model_used', filters.modelUsed)
  if (filters.promptVersion) q.set('prompt_version', filters.promptVersion)
  if (filters.seriesTicker) q.set('series_ticker', filters.seriesTicker)
  if (filters.minConfidence != null) q.set('min_confidence', String(filters.minConfidence))
  if (filters.maxConfidence != null) q.set('max_confidence', String(filters.maxConfidence))
  return q
}

export function getCalibration(filters: CalibrationFilters = {}) {
  const q = _buildParams(filters)
  const suffix = q.size > 0 ? `?${q}` : ''
  return apiFetch<CalibrationResponse>(`/api/calibration${suffix}`)
}

export function getCalibrationTimeSeries(filters: CalibrationFilters = {}) {
  const q = _buildParams(filters)
  const suffix = q.size > 0 ? `?${q}` : ''
  return apiFetch<CalibrationTimeSeriesResponse>(`/api/calibration/time-series${suffix}`)
}

export type CalibrationHeatmapFilters = Omit<CalibrationFilters, 'promptVersion'>

export function getCalibrationByOption(filters: CalibrationHeatmapFilters = {}) {
  const q = _buildParams(filters)
  const suffix = q.size > 0 ? `?${q}` : ''
  return apiFetch<CalibrationHeatmapResponse>(`/api/calibration/by-option${suffix}`)
}
