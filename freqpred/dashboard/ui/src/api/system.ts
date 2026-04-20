import apiFetch from './client'

export function setRunState(state: 'running' | 'paused' | 'stopped') {
  return apiFetch<{ state: string }>('/api/system/run-state', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ state }),
  })
}

export function shutdown() {
  return apiFetch<{ ok: boolean }>('/api/system/shutdown', { method: 'POST' })
}

export function getVersion() {
  return apiFetch<{ version: string; git_hash: string }>('/api/system/version')
}
