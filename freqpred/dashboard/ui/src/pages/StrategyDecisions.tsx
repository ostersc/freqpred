import { useState, useCallback } from 'react'
import { useQuery } from '@tanstack/react-query'
import { getStrategyDecisions } from '../api/strategyDecisions'
import type { StrategyDecisionOut } from '../api/types'
import LoadingSpinner from '../components/LoadingSpinner'
import ErrorBanner from '../components/ErrorBanner'
import PositionDetail from '../components/PositionDetail'

const PAGE_SIZE = 50

function fmt(v: number | null, decimals = 2) {
  if (v === null || v === undefined) return '—'
  return v.toFixed(decimals)
}

function fmtSigned(v: number | null, decimals = 2) {
  if (v === null || v === undefined) return '—'
  const s = v.toFixed(decimals)
  return v >= 0 ? `+${s}` : s
}

function pnlColor(v: number | null | undefined) {
  if (v === null || v === undefined) return 'text-gray-500'
  return v > 0 ? 'text-green-700 font-semibold' : v < 0 ? 'text-red-700 font-semibold' : 'text-gray-600'
}

function relTime(iso: string) {
  return new Date(iso).toLocaleString()
}

function resolutionLabel(result: string | null): { text: string; cls: string } {
  if (result === 'yes') return { text: '✓ YES', cls: 'text-green-700' }
  if (result === 'no') return { text: '✗ NO', cls: 'text-red-700' }
  return { text: '—', cls: 'text-gray-400' }
}

function exitReasonBadge(reason: string | null): { short: string; cls: string } {
  if (!reason) return { short: '—', cls: 'bg-gray-100 text-gray-600' }
  // Collapse force_exit:xxx / custom_exit:xxx to the parent group for coloring.
  const r = reason.toLowerCase()
  if (r === 'market_resolved') return { short: reason, cls: 'bg-blue-100 text-blue-800' }
  if (r === 'stoploss') return { short: reason, cls: 'bg-red-100 text-red-800' }
  if (r === 'trailing_stop') return { short: reason, cls: 'bg-orange-100 text-orange-800' }
  if (r === 'roi') return { short: reason, cls: 'bg-green-100 text-green-800' }
  if (r.startsWith('force_exit')) return { short: reason, cls: 'bg-yellow-100 text-yellow-800' }
  if (r.startsWith('custom_exit')) return { short: reason, cls: 'bg-purple-100 text-purple-800' }
  if (r === 'signal') return { short: reason, cls: 'bg-indigo-100 text-indigo-800' }
  return { short: reason, cls: 'bg-gray-100 text-gray-700' }
}

// ---- Main page ----------------------------------------------------------

export default function StrategyDecisions() {
  const [strategy, setStrategy] = useState<string>('')
  const [exitReason, setExitReason] = useState<string>('')
  const [tickerPrefixInput, setTickerPrefixInput] = useState<string>('')
  const [tickerPrefix, setTickerPrefix] = useState<string>('')
  const [debounceTimer, setDebounceTimer] = useState<ReturnType<typeof setTimeout> | null>(null)
  const [dateFrom, setDateFrom] = useState<string>('')
  const [dateTo, setDateTo] = useState<string>('')
  const [offset, setOffset] = useState(0)
  const [expandedId, setExpandedId] = useState<string | null>(null)

  const handleTickerChange = useCallback((value: string) => {
    setTickerPrefixInput(value)
    if (debounceTimer) clearTimeout(debounceTimer)
    const t = setTimeout(() => {
      setTickerPrefix(value)
      setOffset(0)
    }, 300)
    setDebounceTimer(t)
  }, [debounceTimer])

  const { data, isLoading, error } = useQuery({
    queryKey: ['strategy-decisions', strategy, exitReason, tickerPrefix, dateFrom, dateTo, offset],
    queryFn: () =>
      getStrategyDecisions({
        strategy: strategy || undefined,
        exitReason: exitReason || undefined,
        tickerPrefix: tickerPrefix || undefined,
        dateFrom: dateFrom || undefined,
        dateTo: dateTo || undefined,
        limit: PAGE_SIZE,
        offset,
      }),
    staleTime: 30_000,
  })

  function toggleExpand(id: string) {
    setExpandedId((prev) => (prev === id ? null : id))
  }

  function resetFilters() {
    setStrategy('')
    setExitReason('')
    setTickerPrefixInput('')
    setTickerPrefix('')
    setDateFrom('')
    setDateTo('')
    setOffset(0)
  }

  // Totals for the currently displayed page.
  const totals = data
    ? data.items.reduce(
        (acc, row) => {
          acc.actual += row.pnl ?? 0
          acc.counterfactual += row.counterfactual_pnl_usd ?? 0
          acc.exitDelta += row.exit_delta_usd ?? 0
          acc.entryEff += row.entry_efficiency_usd ?? 0
          return acc
        },
        { actual: 0, counterfactual: 0, exitDelta: 0, entryEff: 0 },
      )
    : null

  return (
    <div>
      <div className="flex items-center justify-between mb-4 gap-3 flex-wrap">
        <h1 className="text-xl font-bold text-gray-900">Strategy Decisions</h1>
      </div>

      {/* Filter bar */}
      <div className="bg-white rounded shadow p-3 mb-3 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-6 gap-3 text-sm">
        <div>
          <label className="block text-xs text-gray-500 uppercase tracking-wide mb-1">Strategy</label>
          <select
            value={strategy}
            onChange={(e) => { setStrategy(e.target.value); setOffset(0) }}
            className="w-full border border-gray-300 rounded px-2 py-1 focus:outline-none focus:ring-1 focus:ring-blue-500"
          >
            <option value="">All</option>
            {data?.distinct_strategies.map((s) => (
              <option key={s} value={s}>{s}</option>
            ))}
          </select>
        </div>
        <div>
          <label className="block text-xs text-gray-500 uppercase tracking-wide mb-1">Exit reason</label>
          <select
            value={exitReason}
            onChange={(e) => { setExitReason(e.target.value); setOffset(0) }}
            className="w-full border border-gray-300 rounded px-2 py-1 focus:outline-none focus:ring-1 focus:ring-blue-500"
          >
            <option value="">All</option>
            {data?.distinct_exit_reasons.map((r) => (
              <option key={r} value={r}>{r}</option>
            ))}
          </select>
        </div>
        <div className="lg:col-span-2">
          <label className="block text-xs text-gray-500 uppercase tracking-wide mb-1">Ticker prefix</label>
          <input
            type="text"
            value={tickerPrefixInput}
            onChange={(e) => handleTickerChange(e.target.value)}
            placeholder="e.g. KXTRUMPSAY or KXTRUMPSAY-26APR13"
            className="w-full border border-gray-300 rounded px-2 py-1 focus:outline-none focus:ring-1 focus:ring-blue-500"
          />
        </div>
        <div>
          <label className="block text-xs text-gray-500 uppercase tracking-wide mb-1">Exit from</label>
          <input
            type="date"
            value={dateFrom}
            onChange={(e) => { setDateFrom(e.target.value); setOffset(0) }}
            className="w-full border border-gray-300 rounded px-2 py-1 focus:outline-none focus:ring-1 focus:ring-blue-500"
          />
        </div>
        <div>
          <label className="block text-xs text-gray-500 uppercase tracking-wide mb-1">Exit to</label>
          <input
            type="date"
            value={dateTo}
            onChange={(e) => { setDateTo(e.target.value); setOffset(0) }}
            className="w-full border border-gray-300 rounded px-2 py-1 focus:outline-none focus:ring-1 focus:ring-blue-500"
          />
        </div>
        <div className="sm:col-span-2 lg:col-span-6 flex justify-end">
          <button
            onClick={resetFilters}
            className="text-xs text-gray-500 hover:text-blue-600 underline"
          >
            Reset filters
          </button>
        </div>
      </div>

      {isLoading && <LoadingSpinner />}
      {error && <ErrorBanner message={String(error)} />}
      {data && (
        <>
          <div className="text-sm text-gray-500 mb-2">
            {data.total} closed positions{data.total > 0 ? ` — showing ${offset + 1}–${Math.min(offset + PAGE_SIZE, data.total)}` : ''}
          </div>
          <div className="bg-white rounded shadow overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead className="bg-gray-100 text-xs text-gray-600 uppercase tracking-wide">
                <tr>
                  <th className="px-3 py-2">Exited</th>
                  <th className="px-3 py-2">Market</th>
                  <th className="px-3 py-2 text-center">Strategy</th>
                  <th className="px-3 py-2 text-center">Dir</th>
                  <th className="px-3 py-2 text-center">Contracts</th>
                  <th className="px-3 py-2 text-center">Entry</th>
                  <th className="px-3 py-2 text-center">Exit</th>
                  <th className="px-3 py-2 text-center">Result</th>
                  <th className="px-3 py-2 text-center">Exit reason</th>
                  <th className="px-3 py-2 text-center">Actual P&L</th>
                  <th className="px-3 py-2 text-center">Counterfactual</th>
                  <th className="px-3 py-2 text-center">Exit Δ</th>
                  <th className="px-3 py-2 text-center">Best prior ask</th>
                  <th className="px-3 py-2 text-center">Entry efficiency</th>
                  <th className="px-3 py-2" />
                </tr>
              </thead>
              <tbody>
                {data.items.map((row: StrategyDecisionOut) => {
                  const res = resolutionLabel(row.market_result)
                  const badge = exitReasonBadge(row.exit_reason)
                  const isMarketResolved = (row.exit_reason ?? '').toLowerCase() === 'market_resolved'
                  return (
                    <>
                      <tr
                        key={row.id}
                        className="border-t cursor-pointer hover:bg-blue-50 transition-colors"
                        onClick={() => toggleExpand(row.id)}
                      >
                        <td className="px-3 py-2 text-gray-500 text-xs whitespace-nowrap">
                          {row.exit_time ? relTime(row.exit_time) : '—'}
                        </td>
                        <td className="px-3 py-2 max-w-xs">
                          <div className="truncate text-gray-800 text-xs font-mono">{row.market_id}</div>
                          {row.market_question && (
                            <div className="truncate text-xs text-gray-400">{row.market_question}</div>
                          )}
                        </td>
                        <td className="px-3 py-2 text-center text-gray-500 text-xs">{row.strategy_name}</td>
                        <td className={`px-3 py-2 text-center font-semibold ${row.direction === 'YES' ? 'text-green-700' : 'text-red-700'}`}>
                          {row.direction}
                        </td>
                        <td className="px-3 py-2 text-center">{row.contracts}</td>
                        <td className="px-3 py-2 text-center font-mono">${fmt(row.entry_price)}</td>
                        <td className="px-3 py-2 text-center font-mono">
                          {row.exit_price !== null ? `$${fmt(row.exit_price)}` : '—'}
                        </td>
                        <td className={`px-3 py-2 text-center font-semibold ${res.cls}`}>{res.text}</td>
                        <td className="px-3 py-2 text-center">
                          <span className={`px-1.5 py-0.5 rounded text-xs font-medium ${badge.cls}`}>
                            {badge.short}
                          </span>
                        </td>
                        <td className={`px-3 py-2 text-center font-mono ${pnlColor(row.pnl)}`}>
                          {row.pnl !== null ? `$${fmtSigned(row.pnl)}` : '—'}
                        </td>
                        <td className={`px-3 py-2 text-center font-mono ${isMarketResolved ? 'text-gray-400' : pnlColor(row.counterfactual_pnl_usd)}`}>
                          {row.counterfactual_pnl_usd !== null
                            ? `$${fmtSigned(row.counterfactual_pnl_usd)}`
                            : '—'}
                        </td>
                        <td className={`px-3 py-2 text-center font-mono ${pnlColor(row.exit_delta_usd)}`}>
                          {isMarketResolved
                            ? <span className="text-gray-400">—</span>
                            : row.exit_delta_usd !== null
                              ? `$${fmtSigned(row.exit_delta_usd)}`
                              : '—'}
                        </td>
                        <td className="px-3 py-2 text-center font-mono text-gray-600">
                          {row.best_prior_ask !== null ? `$${fmt(row.best_prior_ask)}` : '—'}
                        </td>
                        <td className={`px-3 py-2 text-center font-mono ${pnlColor(row.entry_efficiency_usd)}`}>
                          {row.entry_efficiency_usd !== null ? `$${fmtSigned(row.entry_efficiency_usd)}` : '—'}
                        </td>
                        <td className="px-3 py-2 text-center text-gray-400 text-xs">
                          {expandedId === row.id ? '▲' : '▼'}
                        </td>
                      </tr>
                      {expandedId === row.id && (
                        <tr key={`${row.id}-detail`}>
                          <td colSpan={15} className="p-0">
                            <PositionDetail positionId={row.id} />
                          </td>
                        </tr>
                      )}
                    </>
                  )
                })}
                {data.items.length === 0 && (
                  <tr>
                    <td colSpan={15} className="px-3 py-6 text-center text-gray-400">
                      No closed positions match the current filters
                    </td>
                  </tr>
                )}
                {totals && data.items.length > 0 && (
                  <tr className="border-t-2 border-gray-300 bg-gray-50 font-semibold text-xs">
                    <td className="px-3 py-2 text-gray-500 uppercase tracking-wide" colSpan={9}>
                      Page total
                    </td>
                    <td className={`px-3 py-2 text-center font-mono ${pnlColor(totals.actual)}`}>
                      ${fmtSigned(totals.actual)}
                    </td>
                    <td className={`px-3 py-2 text-center font-mono ${pnlColor(totals.counterfactual)}`}>
                      ${fmtSigned(totals.counterfactual)}
                    </td>
                    <td className={`px-3 py-2 text-center font-mono ${pnlColor(totals.exitDelta)}`}>
                      ${fmtSigned(totals.exitDelta)}
                    </td>
                    <td />
                    <td className={`px-3 py-2 text-center font-mono ${pnlColor(totals.entryEff)}`}>
                      ${fmtSigned(totals.entryEff)}
                    </td>
                    <td />
                  </tr>
                )}
              </tbody>
            </table>
          </div>

          {/* Pagination */}
          {data.total > 0 && (
            <div className="flex items-center gap-3 mt-3 text-sm">
              <button
                className="px-3 py-1 rounded border disabled:opacity-40"
                disabled={offset === 0}
                onClick={() => setOffset(Math.max(0, offset - PAGE_SIZE))}
              >
                Previous
              </button>
              <span className="text-gray-500">
                {offset + 1}–{Math.min(offset + PAGE_SIZE, data.total)} of {data.total}
              </span>
              <button
                className="px-3 py-1 rounded border disabled:opacity-40"
                disabled={offset + PAGE_SIZE >= data.total}
                onClick={() => setOffset(offset + PAGE_SIZE)}
              >
                Next
              </button>
            </div>
          )}
        </>
      )}
    </div>
  )
}
