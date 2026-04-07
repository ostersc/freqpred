import { useState, useMemo } from 'react'
import { useQuery } from '@tanstack/react-query'
import { getLedger } from '../api/ledger'
import { getPositions } from '../api/positions'
import LoadingSpinner from '../components/LoadingSpinner'
import ErrorBanner from '../components/ErrorBanner'
import type { PositionOut } from '../api/types'

function fmt(v: number, prefix = '$') {
  return `${prefix}${v.toFixed(2)}`
}

function pnlColor(v: number) {
  return v > 0 ? 'text-green-700' : v < 0 ? 'text-red-700' : 'text-gray-700'
}

function SummaryCard({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="bg-white rounded shadow p-4">
      <div className="text-xs text-gray-500 uppercase tracking-wide mb-1">{label}</div>
      <div className="text-2xl font-bold text-gray-900">{value}</div>
      {sub && <div className="text-xs text-gray-400 mt-0.5">{sub}</div>}
    </div>
  )
}

export default function Ledger() {
  const [exitFilter, setExitFilter] = useState('')
  const [strategyFilter, setStrategyFilter] = useState('')

  const { data: summary, isLoading: summaryLoading, error: summaryError } = useQuery({
    queryKey: ['ledger'],
    queryFn: getLedger,
  })

  const { data: closed, isLoading: closedLoading, error: closedError } = useQuery({
    queryKey: ['positions', 'closed'],
    queryFn: () => getPositions('closed'),
  })

  const exitReasons = useMemo(() => {
    if (!closed) return []
    const reasons = new Set<string>()
    closed.items.forEach((p: PositionOut) => {
      if ((p as PositionOut & { exit_reason?: string }).exit_reason) {
        reasons.add((p as PositionOut & { exit_reason?: string }).exit_reason!)
      }
    })
    return [...reasons].sort()
  }, [closed])

  const strategies = useMemo(() => {
    if (!closed) return []
    const s = new Set(closed.items.map((p) => p.strategy_name))
    return [...s].sort()
  }, [closed])

  const filtered = useMemo(() => {
    if (!closed) return []
    return closed.items.filter((p) => {
      const er = (p as PositionOut & { exit_reason?: string }).exit_reason
      if (exitFilter && er !== exitFilter) return false
      if (strategyFilter && p.strategy_name !== strategyFilter) return false
      return true
    })
  }, [closed, exitFilter, strategyFilter])

  const isLoading = summaryLoading || closedLoading
  const error = summaryError || closedError

  return (
    <div>
      <h1 className="text-xl font-bold text-gray-900 mb-4">Ledger</h1>
      {isLoading && <LoadingSpinner />}
      {error && <ErrorBanner message={String(error)} />}
      {summary && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
          <SummaryCard label="Open Positions" value={String(summary.open_count)} />
          <SummaryCard label="Total Exposure" value={fmt(summary.total_exposure_usd)} />
          <SummaryCard
            label="Daily P&L"
            value={fmt(summary.daily_pnl_usd)}
            sub={summary.daily_pnl_usd >= 0 ? 'today' : 'today'}
          />
          <SummaryCard label="All-time P&L" value={fmt(summary.all_time_pnl_usd)} />
        </div>
      )}
      {closed && (
        <>
          <div className="flex items-center gap-3 mb-3 text-sm">
            <span className="text-gray-500">Filter:</span>
            <select
              className="border rounded px-2 py-1"
              value={exitFilter}
              onChange={(e) => setExitFilter(e.target.value)}
            >
              <option value="">All exit reasons</option>
              {exitReasons.map((r) => <option key={r} value={r}>{r}</option>)}
            </select>
            <select
              className="border rounded px-2 py-1"
              value={strategyFilter}
              onChange={(e) => setStrategyFilter(e.target.value)}
            >
              <option value="">All strategies</option>
              {strategies.map((s) => <option key={s} value={s}>{s}</option>)}
            </select>
            <span className="text-gray-400">{filtered.length} rows</span>
          </div>
          <div className="bg-white rounded shadow overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead className="bg-gray-100 text-xs text-gray-600 uppercase tracking-wide">
                <tr>
                  <th className="px-3 py-2">Market</th>
                  <th className="px-3 py-2 text-center">Dir</th>
                  <th className="px-3 py-2 text-center">Entry</th>
                  <th className="px-3 py-2 text-center">Exit</th>
                  <th className="px-3 py-2 text-center">P&L</th>
                  <th className="px-3 py-2 text-center">P&L %</th>
                  <th className="px-3 py-2 text-center">Resolution</th>
                  <th className="px-3 py-2 text-center">Strategy</th>
                </tr>
              </thead>
              <tbody>
                {filtered.map((p) => (
                  <tr key={p.id} className="border-t hover:bg-gray-50">
                    <td className="px-3 py-2 max-w-xs">
                      <div className="truncate">{p.market_id}</div>
                    </td>
                    <td className={`px-3 py-2 text-center font-semibold ${p.direction === 'YES' ? 'text-green-700' : 'text-red-700'}`}>
                      {p.direction}
                    </td>
                    <td className="px-3 py-2 text-center">${p.entry_price.toFixed(2)}</td>
                    <td className="px-3 py-2 text-center">{p.exit_price !== null ? `$${p.exit_price.toFixed(2)}` : '—'}</td>
                    <td className={`px-3 py-2 text-center font-semibold ${p.pnl !== null ? pnlColor(p.pnl) : 'text-gray-500'}`}>
                      {p.pnl !== null ? fmt(p.pnl) : '—'}
                    </td>
                    <td className={`px-3 py-2 text-center ${p.pnl_pct !== null ? pnlColor(p.pnl_pct) : 'text-gray-500'}`}>
                      {p.pnl_pct !== null ? `${(p.pnl_pct * 100).toFixed(1)}%` : '—'}
                    </td>
                    <td className="px-3 py-2 text-center">
                      {p.resolution === 1 ? <span className="text-green-700 font-semibold">YES</span>
                        : p.resolution === 0 ? <span className="text-red-700 font-semibold">NO</span>
                        : '—'}
                    </td>
                    <td className="px-3 py-2 text-center text-gray-500 text-xs">{p.strategy_name}</td>
                  </tr>
                ))}
                {filtered.length === 0 && (
                  <tr>
                    <td colSpan={8} className="px-3 py-6 text-center text-gray-400">No closed positions</td>
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
