import apiFetch from './client'
import type { StrategyConfigOut } from './types'

export function getStrategyConfig() {
  return apiFetch<StrategyConfigOut>('/api/strategy/config')
}

export function updateStrategyConfig(body: Partial<StrategyConfigOut>) {
  return apiFetch<StrategyConfigOut>('/api/strategy/config', {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
}
