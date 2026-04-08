import apiFetch from './client'
import type { CalibrationResponse } from './types'

export function getCalibration(lookbackDays?: number) {
  const params = lookbackDays != null ? `?lookback_days=${lookbackDays}` : ''
  return apiFetch<CalibrationResponse>(`/api/calibration${params}`)
}
