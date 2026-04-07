import apiFetch from './client'
import type { LLMCostResponse, LLMQueryListResponse, LLMQueryDetailOut } from './types'

export function getLLMCost() {
  return apiFetch<LLMCostResponse>('/api/llm/cost')
}

export function getLLMQueries(params?: { limit?: number; offset?: number }) {
  const q = new URLSearchParams()
  if (params?.limit !== undefined) q.set('limit', String(params.limit))
  if (params?.offset !== undefined) q.set('offset', String(params.offset))
  return apiFetch<LLMQueryListResponse>(`/api/llm/queries?${q}`)
}

export function getLLMQuery(id: number) {
  return apiFetch<LLMQueryDetailOut>(`/api/llm/queries/${id}`)
}
