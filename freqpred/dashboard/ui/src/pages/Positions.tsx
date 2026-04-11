import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { getPositions } from '../api/positions'
import type { PositionOut } from '../api/types'
import LoadingSpinner from '../components/LoadingSpinner'
import ErrorBanner from '../components/ErrorBanner'
import PositionDetail from '../components/PositionDetail'

type StatusFilter = 'open' | 'closed' | 'all'

function fmt(v: number | null, decimals = 2) {
  if (v === null) return '—'
  return v.toFixed(decimals)
}

function fmtPct(v: number | null) {
  if (v === null) return '—'
  const s = (v * 100).toFixed(1)
  return v >= 0 ? `+${s}%` : `${s}%`
}

function pnlColor(v: number | null) {
  if (v === null) return 'text-gray-500'
  return v > 0 ? 'text-green-700 font-semibold' : v < 0 ? 'text-red-700 font-semibold' : 'text-gray-600'
}

function relTime(iso: string) {
  return new Date(iso).toLocaleString()
}

export default function Positions() {
  const [status, setStatus] = useState<StatusFilter>('open')
  const [expandedId, setExpandedId] = useState<string | null>(null)

  const { data, isLoading, error } = useQuery({
    queryKey: ['positions', status],
    queryFn: () => getPositions(status),
    refetchInterval: status === 'open' ? 60_000 : false,
  })

  function toggleExpand(id: string) {
    setExpandedId((prev) => (prev === id ? null : id))
  }

  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <h1 className="text-xl font-bold text-gray-900">Positions</h1>
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
          <div className="text-sm text-gray-500 mb-2">{data.total} positions{status === 'open' ? ' — refreshes every 60s' : ''}</div>
          <div className="bg-white rounded shadow overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead className="bg-gray-100 text-xs text-gray-600 uppercase tracking-wide">
                <tr>
                  <th className="px-3 py-2">Market</th>
                  <th className="px-3 py-2 text-center">Dir</th>
                  <th className="px-3 py-2 text-center">Contracts</th>
                  <th className="px-3 py-2 text-center">Entry</th>
                  <th className="px-3 py-2 text-center">Exposure</th>
                  <th className="px-3 py-2 text-center">Exit</th>
                  <th className="px-3 py-2 text-center">P&L (unreal.)</th>
                  <th className="px-3 py-2 text-center">P&L %</th>
                  <th className="px-3 py-2 text-center">Status</th>
                  <th className="px-3 py-2 text-center">Strategy</th>
                  <th className="px-3 py-2 text-center">Entered</th>
                  <th className="px-3 py-2" />
                </tr>
              </thead>
              <tbody>
                {data.items.map((p: PositionOut) => (
                  <>
                    <tr
                      key={p.id}
                      className="border-t cursor-pointer hover:bg-blue-50 transition-colors"
                      onClick={() => toggleExpand(p.id)}
                    >
                      <td className="px-3 py-2 max-w-xs">
                        <div className="truncate text-gray-800">{p.market_id}</div>
                      </td>
                      <td className={`px-3 py-2 text-center font-semibold ${p.direction === 'YES' ? 'text-green-700' : 'text-red-700'}`}>
                        {p.direction}
                      </td>
                      <td className="px-3 py-2 text-center">{p.contracts}</td>
                      <td className="px-3 py-2 text-center">${fmt(p.entry_price)}</td>
                      <td className="px-3 py-2 text-center text-gray-700">${fmt(p.entry_price * p.contracts)}</td>
                      <td className="px-3 py-2 text-center">{p.exit_price !== null ? `$${fmt(p.exit_price)}` : '—'}</td>
                      <td className={`px-3 py-2 text-center ${pnlColor(p.status === 'open' ? p.unrealized_pnl : p.pnl)}`}>
                        {p.status === 'open'
                          ? p.unrealized_pnl !== null ? `$${fmt(p.unrealized_pnl)}` : '—'
                          : p.pnl !== null ? `$${fmt(p.pnl)}` : '—'}
                      </td>
                      <td className={`px-3 py-2 text-center ${pnlColor(p.status === 'open' ? p.unrealized_pnl_pct : p.pnl_pct)}`}>
                        {p.status === 'open' ? fmtPct(p.unrealized_pnl_pct) : fmtPct(p.pnl_pct)}
                      </td>
                      <td className="px-3 py-2 text-center">
                        <span className={`px-1.5 py-0.5 rounded text-xs font-medium ${
                          p.status === 'open' ? 'bg-green-100 text-green-800' :
                          p.status === 'closed' ? 'bg-gray-100 text-gray-700' :
                          'bg-yellow-100 text-yellow-800'
                        }`}>{p.status}</span>
                      </td>
                      <td className="px-3 py-2 text-center text-gray-500 text-xs">{p.strategy_name}</td>
                      <td className="px-3 py-2 text-center text-gray-500 text-xs">{relTime(p.entry_time)}</td>
                      <td className="px-3 py-2 text-center text-gray-400 text-xs">
                        {expandedId === p.id ? '▲' : '▼'}
                      </td>
                    </tr>
                    {expandedId === p.id && (
                      <tr key={`${p.id}-detail`}>
                        <td colSpan={12} className="p-0">
                          <PositionDetail positionId={p.id} />
                        </td>
                      </tr>
                    )}
                  </>
                ))}
                {data.items.length === 0 && (
                  <tr>
                    <td colSpan={12} className="px-3 py-6 text-center text-gray-400">No positions</td>
                  </tr>
                )}
                {data.items.length > 0 && (() => {
                  const totalContracts = data.items.reduce((s, p) => s + p.contracts, 0)
                  const totalPnl = data.items.reduce((s, p) => {
                    const v = p.status === 'open' ? p.unrealized_pnl : p.pnl
                    return s + (v ?? 0)
                  }, 0)
                  const totalCostBasis = data.items.reduce((s, p) => s + p.entry_price * p.contracts, 0)
                  const weightedPct = totalCostBasis > 0 ? totalPnl / totalCostBasis : null
                  const weightedAvgEntry = totalContracts > 0 ? totalCostBasis / totalContracts : null
                  return (
                    <tr className="border-t-2 border-gray-300 bg-gray-50 font-semibold text-xs">
                      <td className="px-3 py-2 text-gray-500 uppercase tracking-wide">Total</td>
                      <td />
                      <td className="px-3 py-2 text-center">{totalContracts}</td>
                      <td className="px-3 py-2 text-center text-gray-600">{weightedAvgEntry !== null ? `$${fmt(weightedAvgEntry)}` : '—'}</td>
                      <td className="px-3 py-2 text-center text-gray-700">${fmt(totalCostBasis)}</td>
                      <td />
                      <td className={`px-3 py-2 text-center ${pnlColor(totalPnl)}`}>${fmt(totalPnl)}</td>
                      <td className={`px-3 py-2 text-center ${pnlColor(weightedPct)}`}>{fmtPct(weightedPct)}</td>
                      <td /><td /><td /><td />
                    </tr>
                  )
                })()}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  )
}
