import apiFetch from './client'
import type { CalibrationResponse } from './types'

export function getCalibration(lookbackDays?: number, category?: string) {
  const q = new URLSearchParams()
  if (lookbackDays != null) q.set('lookback_days', String(lookbackDays))
  if (category) q.set('category', category)
  const suffix = q.size > 0 ? `?${q}` : ''
  return apiFetch<CalibrationResponse>(`/api/calibration${suffix}`)
}
