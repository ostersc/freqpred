import apiFetch from './client'
import type { LedgerResponse } from './types'

export function getLedger() {
  return apiFetch<LedgerResponse>('/api/ledger')
}
