import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { getPositionDetail } from '../api/positions'
import type { PositionDetailOut } from '../api/types'
import AnalyzeButton from './AnalyzeButton'
import PriceTimeline, { triggerLabel } from './PriceTimeline'
import { SelectedSignalPanel } from './SignalDetail'

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

export default function PositionDetail({ positionId }: { positionId: string }) {
  // null = show entry signal (default)
  const [selectedSignalId, setSelectedSignalId] = useState<string | null>(null)

  const { data, isLoading, error } = useQuery({
    queryKey: ['position-detail', positionId],
    queryFn: () => getPositionDetail(positionId),
    staleTime: 30_000,
  })

  if (isLoading) return <div className="p-4 text-sm text-gray-500">Loading…</div>
  if (error) return <div className="p-4 text-sm text-red-600">{String(error)}</div>
  if (!data) return null

  const d: PositionDetailOut = data
  const activeSignalId = selectedSignalId ?? d.entry_signal.id
  const isEntryActive = activeSignalId === d.entry_signal.id

  // Entry signal always labelled "Entry signal" regardless of its trigger field.
  // For other signals, look up the trigger from market_signals.
  const activeTrigger = isEntryActive
    ? 'entry'
    : (d.market_signals.find((s) => s.id === activeSignalId)?.trigger ?? 'entry')

  return (
    <div className="bg-gray-50 border-t px-4 py-4 space-y-5 text-sm">

      {/* Market question */}
      {d.market_question && (
        <div className="font-semibold text-gray-800 text-base leading-snug">{d.market_question}</div>
      )}

      {/* Stat cards */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <div className="bg-white rounded border p-3">
          <div className="text-xs text-gray-400 uppercase tracking-wide mb-1">Entry price</div>
          <div className="font-semibold">{(d.entry_price * 100).toFixed(1)}¢</div>
        </div>
        <div className="bg-white rounded border p-3">
          <div className="text-xs text-gray-400 uppercase tracking-wide mb-1">
            {d.status === 'open' ? 'Current mid' : 'Exit price'}
          </div>
          <div className="font-semibold">
            {d.status === 'open'
              ? d.current_mid !== null ? `${(d.current_mid * 100).toFixed(1)}¢` : '—'
              : d.exit_price !== null ? `${(d.exit_price * 100).toFixed(1)}¢` : '—'}
          </div>
        </div>
        <div className="bg-white rounded border p-3">
          <div className="text-xs text-gray-400 uppercase tracking-wide mb-1">Contracts</div>
          <div className="font-semibold">{d.contracts}</div>
        </div>
        <div className="bg-white rounded border p-3">
          <div className="text-xs text-gray-400 uppercase tracking-wide mb-1">
            {d.status === 'open' ? 'Unrealized P&L' : 'P&L'}
          </div>
          <div className={`font-semibold ${pnlColor(d.status === 'open' ? d.unrealized_pnl : d.pnl)}`}>
            {d.status === 'open'
              ? d.unrealized_pnl !== null
                ? `$${fmt(d.unrealized_pnl)} (${fmtPct(d.unrealized_pnl_pct)})`
                : '—'
              : d.pnl !== null
                ? `$${fmt(d.pnl)} (${fmtPct(d.pnl_pct)})`
                : '—'}
          </div>
        </div>
      </div>

      {/* Price timeline */}
      <PriceTimeline
        signals={d.market_signals}
        entrySignalId={d.entry_signal.id}
        entryPrice={d.entry_price}
        currentMid={d.status === 'open' ? d.current_mid : null}
        direction={d.direction}
        selectedSignalId={activeSignalId}
        onSignalClick={(id) => setSelectedSignalId(id)}
        exitPrice={d.status === 'closed' ? d.exit_price : null}
        exitTime={d.status === 'closed' ? d.exit_time : null}
        exitReason={d.status === 'closed' ? d.exit_reason : null}
      />

      {/* Selected signal panel — label changes to match clicked signal's trigger */}
      <div>
        <div className="flex items-center gap-3 mb-2">
          <div className="text-xs font-semibold text-gray-500 uppercase tracking-wide">
            {triggerLabel(activeTrigger)}
          </div>
          {!isEntryActive && (
            <button
              onClick={() => setSelectedSignalId(null)}
              className="text-xs text-blue-500 hover:underline"
            >
              ← back to entry signal
            </button>
          )}
          <AnalyzeButton marketId={d.market_id} />
        </div>
        <SelectedSignalPanel signalId={activeSignalId} entrySignal={d.entry_signal} />
      </div>
    </div>
  )
}
