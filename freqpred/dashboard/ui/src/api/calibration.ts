import apiFetch from './client'
import type { CalibrationResponse } from './types'

export function getCalibration() {
  return apiFetch<CalibrationResponse>('/api/calibration')
}
