import { useState } from 'react'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { analyzeMarket } from '../api/markets'

export default function AnalyzeButton({ marketId }: { marketId: string }) {
  const queryClient = useQueryClient()
  const [error, setError] = useState<string | null>(null)
  const [message, setMessage] = useState<string | null>(null)

  const mutation = useMutation({
    mutationFn: () => analyzeMarket(marketId),
    onSuccess: (data) => {
      setError(null)
      setMessage(data.cached ? 'Cached — analyzed within the last 60 s.' : 'New signal generated.')
      queryClient.invalidateQueries({ queryKey: ['signals'] })
      queryClient.invalidateQueries({ queryKey: ['signal'] })
      queryClient.invalidateQueries({ queryKey: ['positions'] })
      queryClient.invalidateQueries({ queryKey: ['position-detail'] })
      queryClient.invalidateQueries({ queryKey: ['market-detail', marketId] })
      queryClient.invalidateQueries({ queryKey: ['markets'] })
    },
    onError: (err: Error) => {
      setMessage(null)
      setError(err.message)
    },
  })

  return (
    <div style={{ display: 'inline-flex', alignItems: 'center', gap: 8 }}>
      <button
        className="btn primary sm"
        onClick={(e) => {
          e.stopPropagation()
          setError(null)
          setMessage(null)
          mutation.mutate()
        }}
        disabled={mutation.isPending}
      >
        {mutation.isPending ? 'Analyzing…' : '⚡ Analyze now'}
      </button>
      {message && <span style={{ fontSize: 11.5, color: 'var(--pos)' }}>{message}</span>}
      {error && <span style={{ fontSize: 11.5, color: 'var(--neg)' }}>{error}</span>}
    </div>
  )
}
