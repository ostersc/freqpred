import { useState, useCallback } from 'react'
import { useQuery } from '@tanstack/react-query'
import { getMarkets, getMarket } from '../api/markets'
import type { MarketOut, MarketDetailOut } from '../api/types'
import LoadingSpinner from '../components/LoadingSpinner'
import ErrorBanner from '../components/ErrorBanner'
import AnalyzeButton from '../components/AnalyzeButton'

type StatusFilter = 'open' | 'closed' | 'all'

function fmt2(v: number) {
  return (v * 100).toFixed(1)
}

function relTime(iso: string) {
  return new Date(iso).toLocaleString()
}

function closeTimeLabel(iso: string) {
  const d = new Date(iso)
  const now = new Date()
  const diffMs = d.getTime() - now.getTime()
  const diffDays = Math.round(diffMs / 86_400_000)
  if (diffDays < 0) return `Closed ${Math.abs(diffDays)}d ago`
  if (diffDays === 0) return 'Closes today'
  return `${diffDays}d`
}

// ---- Market detail panel ------------------------------------------------

function MarketDetail({ marketId }: { marketId: string }) {
  const { data, isLoading, error } = useQuery({
    queryKey: ['market-detail', marketId],
    queryFn: () => getMarket(marketId),
    staleTime: 30_000,
  })

  if (isLoading) return <div className="p-4 text-sm text-gray-500">Loading…</div>
  if (error) return <div className="p-4 text-sm text-red-600">{String(error)}</div>
  if (!data) return null

  const d: MarketDetailOut = data
  const sig = d.current_signal

  return (
    <div className="bg-gray-50 border-t px-4 py-4 space-y-4 text-sm">
      {/* Market question */}
      <div className="font-semibold text-gray-800 text-base leading-snug">{d.question}</div>

      {/* Stat grid */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <div className="bg-white rounded border p-3">
          <div className="text-xs text-gray-400 uppercase tracking-wide mb-1">Mid price</div>
          <div className="font-semibold">{fmt2(d.mid_price)}¢</div>
        </div>
        <div className="bg-white rounded border p-3">
          <div className="text-xs text-gray-400 uppercase tracking-wide mb-1">Bid / Ask</div>
          <div className="font-semibold">{fmt2(d.yes_bid)}¢ / {fmt2(d.yes_ask)}¢</div>
        </div>
        <div className="bg-white rounded border p-3">
          <div className="text-xs text-gray-400 uppercase tracking-wide mb-1">Volume 24h</div>
          <div className="font-semibold">{d.volume_24h.toLocaleString()}</div>
        </div>
        <div className="bg-white rounded border p-3">
          <div className="text-xs text-gray-400 uppercase tracking-wide mb-1">Closes</div>
          <div className="font-semibold">{relTime(d.close_time)}</div>
        </div>
      </div>

      {/* Current signal */}
      <div>
        <div className="flex items-center justify-between mb-2">
          <div className="text-xs font-semibold text-gray-500 uppercase tracking-wide">
            Current signal
          </div>
          <AnalyzeButton marketId={marketId} />
        </div>

        {sig ? (
          <div className="bg-white rounded border p-3 space-y-2">
            <div className="flex flex-wrap gap-4 text-xs text-gray-500">
              <span>Our prob: <span className="font-semibold text-gray-800">{fmt2(sig.estimated_probability)}%</span></span>
              <span>Market mid: <span className="font-semibold text-gray-800">{fmt2(sig.market_mid_at_signal)}%</span></span>
              <span>Edge: <span className={`font-semibold ${sig.edge >= 0 ? 'text-green-700' : 'text-red-700'}`}>
                {sig.edge >= 0 ? '+' : ''}{(sig.edge * 100).toFixed(1)}%
              </span></span>
              <span>Confidence: <span className="font-semibold text-gray-800">{fmt2(sig.confidence)}%</span></span>
              <span className="text-gray-400">{relTime(sig.created_at)}</span>
            </div>
            <div>
              <div className="font-medium text-gray-700 mb-0.5">Reasoning:</div>
              <p className="text-gray-600 whitespace-pre-wrap">{sig.reasoning}</p>
            </div>
            {sig.social_sentiment_summary && (
              <div>
                <div className="font-medium text-gray-700 mb-0.5">Social sentiment:</div>
                <p className="text-gray-600">{sig.social_sentiment_summary}</p>
              </div>
            )}
          </div>
        ) : (
          <div className="text-gray-400 text-xs">No signal yet — click "Analyze now" to run the signal pipeline.</div>
        )}
      </div>

      {/* Full market ID + last fetched */}
      <div className="text-xs text-gray-400 pt-1">
        <span className="font-medium">Market ID:</span> {d.id} &nbsp;·&nbsp;
        <span className="font-medium">Last fetched:</span> {relTime(d.last_fetched_at)}
      </div>
    </div>
  )
}

// ---- Main page -----------------------------------------------------------

export default function Markets() {
  const [status, setStatus] = useState<StatusFilter>('open')
  const [search, setSearch] = useState('')
  const [debouncedSearch, setDebouncedSearch] = useState('')
  const [debounceTimer, setDebounceTimer] = useState<ReturnType<typeof setTimeout> | null>(null)
  const [expandedId, setExpandedId] = useState<string | null>(null)

  const handleSearchChange = useCallback((value: string) => {
    setSearch(value)
    if (debounceTimer) clearTimeout(debounceTimer)
    const t = setTimeout(() => setDebouncedSearch(value), 300)
    setDebounceTimer(t)
  }, [debounceTimer])

  const { data, isLoading, error } = useQuery({
    queryKey: ['markets', status, debouncedSearch],
    queryFn: () => getMarkets({ status, search: debouncedSearch || undefined, limit: 100 }),
    staleTime: 30_000,
  })

  function toggleExpand(id: string) {
    setExpandedId((prev) => (prev === id ? null : id))
  }

  return (
    <div>
      <div className="flex items-center justify-between mb-4 gap-3 flex-wrap">
        <h1 className="text-xl font-bold text-gray-900">Markets</h1>
        <div className="flex gap-2 items-center flex-1 max-w-lg">
          <input
            type="text"
            value={search}
            onChange={(e) => handleSearchChange(e.target.value)}
            placeholder="Search by question or market ID…"
            className="flex-1 border border-gray-300 rounded px-3 py-1.5 text-sm focus:outline-none focus:ring-1 focus:ring-blue-500"
          />
        </div>
        <div className="flex gap-1 text-sm">
          {(['open', 'closed', 'all'] as StatusFilter[]).map((s) => (
            <button
              key={s}
              onClick={() => setStatus(s)}
              className={`px-3 py-1 rounded border capitalize ${status === s ? 'bg-blue-600 text-white border-blue-600' : 'border-gray-300 text-gray-700 hover:bg-gray-50'}`}
            >
              {s}
            </button>
          ))}
        </div>
      </div>

      {isLoading && <LoadingSpinner />}
      {error && <ErrorBanner message={String(error)} />}
      {data && (
        <>
          <div className="text-sm text-gray-500 mb-2">{data.total} markets</div>
          <div className="bg-white rounded shadow overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead className="bg-gray-100 text-xs text-gray-600 uppercase tracking-wide">
                <tr>
                  <th className="px-3 py-2">Question</th>
                  <th className="px-3 py-2 text-center">Mid</th>
                  <th className="px-3 py-2 text-center">Vol 24h</th>
                  <th className="px-3 py-2 text-center">Closes</th>
                  <th className="px-3 py-2 text-center">Signal edge</th>
                  <th className="px-3 py-2 text-center">Status</th>
                  <th className="px-3 py-2" />
                </tr>
              </thead>
              <tbody>
                {data.items.map((m: MarketOut) => (
                  <>
                    <tr
                      key={m.id}
                      className="border-t cursor-pointer hover:bg-blue-50 transition-colors"
                      onClick={() => toggleExpand(m.id)}
                    >
                      <td className="px-3 py-2 max-w-md">
                        <div className="truncate text-gray-800">{m.question}</div>
                        <div className="text-xs text-gray-400 truncate">{m.id}</div>
                      </td>
                      <td className="px-3 py-2 text-center font-mono text-gray-700">
                        {fmt2(m.mid_price)}¢
                      </td>
                      <td className="px-3 py-2 text-center text-gray-600">
                        {m.volume_24h.toLocaleString()}
                      </td>
                      <td className="px-3 py-2 text-center text-gray-600 text-xs whitespace-nowrap">
                        {closeTimeLabel(m.close_time)}
                      </td>
                      <td className="px-3 py-2 text-center text-gray-400 text-xs">
                        {m.current_signal_id ? '—' : <span className="text-gray-300">none</span>}
                      </td>
                      <td className="px-3 py-2 text-center">
                        <span className={`px-1.5 py-0.5 rounded text-xs font-medium ${
                          m.status === 'active' ? 'bg-green-100 text-green-800' :
                          'bg-gray-100 text-gray-600'
                        }`}>
                          {m.status === 'active' ? 'open' : m.status}
                        </span>
                      </td>
                      <td className="px-3 py-2 text-center text-gray-400 text-xs">
                        {expandedId === m.id ? '▲' : '▼'}
                      </td>
                    </tr>
                    {expandedId === m.id && (
                      <tr key={`${m.id}-detail`}>
                        <td colSpan={7} className="p-0">
                          <MarketDetail marketId={m.id} />
                        </td>
                      </tr>
                    )}
                  </>
                ))}
                {data.items.length === 0 && (
                  <tr>
                    <td colSpan={7} className="px-3 py-6 text-center text-gray-400">
                      No markets found
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  )
}
