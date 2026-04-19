import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { getSignals, getSignal } from '../api/signals'
import LoadingSpinner from '../components/LoadingSpinner'
import ErrorBanner from '../components/ErrorBanner'
import AnalyzeButton from '../components/AnalyzeButton'
import { SignalDetail as SharedSignalDetail } from '../components/SignalDetail'
import type { SignalOut } from '../api/types'

const PAGE_SIZE = 20

function pct(v: number) {
  return `${(v * 100).toFixed(1)}%`
}

function age(iso: string) {
  const secs = (Date.now() - new Date(iso).getTime()) / 1000
  if (secs < 60) return `${Math.round(secs)}s`
  if (secs < 3600) return `${Math.round(secs / 60)}m`
  if (secs < 86400) return `${Math.round(secs / 3600)}h`
  return `${Math.round(secs / 86400)}d`
}

function SignalDetail({ id, marketId }: { id: string; marketId: string }) {
  const { data, isLoading } = useQuery({
    queryKey: ['signal', id],
    queryFn: () => getSignal(id),
  })

  if (isLoading) return <div className="px-4 py-2 text-gray-500 text-sm">Loading…</div>
  if (!data) return null

  return (
    <div className="px-4 py-3 bg-gray-50 border-t text-sm space-y-3">
      <div className="flex items-center justify-between">
        <span className="font-medium text-gray-700">Signal detail</span>
        <AnalyzeButton marketId={marketId} />
      </div>
      <SharedSignalDetail signal={data} />
    </div>
  )
}

function SignalRow({ signal }: { signal: SignalOut }) {
  const [expanded, setExpanded] = useState(false)
  const edgeColor = signal.edge >= 0.15 ? 'text-green-700' : signal.edge >= 0.08 ? 'text-yellow-700' : 'text-gray-500'
  const confColor = signal.confidence >= 0.7 ? 'text-green-700 font-semibold' : signal.confidence >= 0.5 ? 'text-yellow-700' : 'text-red-600'
  const dirColor = signal.direction === 'YES' ? 'text-green-700 font-semibold' : signal.direction === 'NO' ? 'text-red-700 font-semibold' : 'text-gray-500'

  return (
    <>
      <tr
        className="border-t cursor-pointer hover:bg-blue-50 transition-colors"
        onClick={() => setExpanded((e) => !e)}
      >
        <td className="px-3 py-2 max-w-xs">
          <div className="truncate text-sm text-gray-800">
            {signal.market_question ?? signal.market_id}
          </div>
          <div className="text-xs text-gray-400">{signal.market_id}</div>
        </td>
        <td className="px-3 py-2 text-sm text-center">{pct(signal.estimated_probability)}</td>
        <td className="px-3 py-2 text-sm text-center">{pct(signal.market_mid_at_signal)}</td>
        <td className={`px-3 py-2 text-sm text-center ${edgeColor}`}>+{pct(signal.edge)}</td>
        <td className={`px-3 py-2 text-sm text-center ${confColor}`}>{pct(signal.confidence)}</td>
        <td className={`px-3 py-2 text-sm text-center ${dirColor}`}>{signal.direction}</td>
        <td className="px-3 py-2 text-sm text-center text-gray-500">{age(signal.created_at)}</td>
        <td className="px-3 py-2 text-sm text-center">
          <span className="text-gray-400">{expanded ? '▲' : '▼'}</span>
        </td>
      </tr>
      {expanded && (
        <tr>
          <td colSpan={8} className="p-0">
            <SignalDetail id={signal.id} marketId={signal.market_id} />
          </td>
        </tr>
      )}
    </>
  )
}

export default function SignalFeed() {
  const [offset, setOffset] = useState(0)

  const { data, isLoading, error } = useQuery({
    queryKey: ['signals', offset],
    queryFn: () => getSignals({ limit: PAGE_SIZE, offset }),
    refetchInterval: 30_000,
  })

  return (
    <div>
      <h1 className="text-xl font-bold text-gray-900 mb-4">Signal Feed</h1>
      {isLoading && <LoadingSpinner />}
      {error && <ErrorBanner message={String(error)} />}
      {data && (
        <>
          <div className="text-sm text-gray-500 mb-2">{data.total} signals total — refreshes every 30s</div>
          <div className="bg-white rounded shadow overflow-x-auto">
            <table className="w-full text-left">
              <thead className="bg-gray-100 text-xs text-gray-600 uppercase tracking-wide">
                <tr>
                  <th className="px-3 py-2">Market</th>
                  <th className="px-3 py-2 text-center">Our Prob</th>
                  <th className="px-3 py-2 text-center">Market Mid</th>
                  <th className="px-3 py-2 text-center">Edge</th>
                  <th className="px-3 py-2 text-center">Confidence</th>
                  <th className="px-3 py-2 text-center">Dir</th>
                  <th className="px-3 py-2 text-center">Age</th>
                  <th className="px-3 py-2" />
                </tr>
              </thead>
              <tbody>
                {data.items.map((s) => <SignalRow key={s.id} signal={s} />)}
              </tbody>
            </table>
          </div>
          <div className="flex items-center gap-3 mt-3 text-sm">
            <button
              className="px-3 py-1 rounded border disabled:opacity-40"
              disabled={offset === 0}
              onClick={() => setOffset(Math.max(0, offset - PAGE_SIZE))}
            >
              Previous
            </button>
            <span className="text-gray-500">{offset + 1}–{Math.min(offset + PAGE_SIZE, data.total)} of {data.total}</span>
            <button
              className="px-3 py-1 rounded border disabled:opacity-40"
              disabled={offset + PAGE_SIZE >= data.total}
              onClick={() => setOffset(offset + PAGE_SIZE)}
            >
              Next
            </button>
          </div>
        </>
      )}
    </div>
  )
}
