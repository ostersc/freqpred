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
      // Invalidate any queries that might show this market's signals
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
    <div className="inline-flex items-center gap-2">
      <button
        onClick={(e) => {
          e.stopPropagation()
          setError(null)
          setMessage(null)
          mutation.mutate()
        }}
        disabled={mutation.isPending}
        className="px-3 py-1 text-xs rounded border bg-blue-600 text-white hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed inline-flex items-center gap-1.5"
      >
        {mutation.isPending ? (
          <>
            <svg className="animate-spin h-3 w-3" viewBox="0 0 24 24" fill="none">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
            </svg>
            Analyzing…
          </>
        ) : 'Analyze now'}
      </button>
      {message && <span className="text-xs text-green-700">{message}</span>}
      {error && <span className="text-xs text-red-600">{error}</span>}
    </div>
  )
}
