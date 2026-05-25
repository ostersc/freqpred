import React, { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { getPositionDetail, forceExitPosition } from '../api/positions'
import type { PositionDetailOut } from '../api/types'
import AnalyzeButton from './AnalyzeButton'
import PriceTimeline, { triggerLabel } from './PriceTimeline'
import { SelectedSignalPanel } from './SignalDetail'

function fmt(v: number | null, decimals = 2) {
  if (v === null) return '—'
  return v.toFixed(decimals)
}

function fmtSignedPct(v: number | null) {
  if (v === null) return ''
  const s = (v * 100).toFixed(1)
  return v >= 0 ? ` (+${s}%)` : ` (${s}%)`
}

const miniCard: React.CSSProperties = {
  padding: '8px 10px', background: 'var(--bg-1)', border: '1px solid var(--line-soft)', borderRadius: 6,
}
const miniLabel: React.CSSProperties = {
  fontSize: 10.5, textTransform: 'uppercase', letterSpacing: '0.08em', color: 'var(--fg-3)', marginBottom: 4,
}

export default function PositionDetail({ positionId }: { positionId: string }) {
  const [selectedSignalId, setSelectedSignalId] = useState<string | null>(null)
  const queryClient = useQueryClient()

  const { data, isLoading, error } = useQuery({
    queryKey: ['position-detail', positionId],
    queryFn: () => getPositionDetail(positionId),
    staleTime: 30_000,
  })

  const forceExit = useMutation({
    mutationFn: () => forceExitPosition(positionId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['positions'] })
      queryClient.invalidateQueries({ queryKey: ['position-detail', positionId] })
    },
  })

  if (isLoading) return <div style={{ padding: '16px 20px', color: 'var(--fg-3)', fontSize: 12.5 }}>Loading…</div>
  if (error) return <div style={{ padding: '16px 20px', color: 'var(--neg)', fontSize: 12.5 }}>{String(error)}</div>
  if (!data) return null

  const d: PositionDetailOut = data
  const activeSignalId = selectedSignalId ?? d.entry_signal.id
  const isEntryActive = activeSignalId === d.entry_signal.id
  const activeTrigger = isEntryActive
    ? 'entry'
    : (d.market_signals.find((s) => s.id === activeSignalId)?.trigger ?? 'entry')

  const pnlVal = d.status === 'open' ? d.unrealized_pnl : d.pnl
  const pnlPctVal = d.status === 'open' ? d.unrealized_pnl_pct : d.pnl_pct
  const pnlColor = pnlVal === null ? 'var(--fg-2)' : pnlVal > 0 ? 'var(--pos)' : pnlVal < 0 ? 'var(--neg)' : 'var(--fg-2)'

  return (
    <div style={{ background: 'var(--bg-1)', borderTop: '1px solid var(--line-soft)', padding: '16px 20px', display: 'flex', flexDirection: 'column', gap: 16 }}>

      {d.market_question && (
        <div style={{ fontWeight: 500, color: 'var(--fg-0)', fontSize: 13, lineHeight: 1.5 }}>{d.market_question}</div>
      )}

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 8 }}>
        <div style={miniCard}>
          <div style={miniLabel}>Entry price</div>
          <div className="mono" style={{ fontWeight: 600 }}>{(d.entry_price * 100).toFixed(1)}¢</div>
        </div>
        <div style={miniCard}>
          <div style={miniLabel}>{d.status === 'open' ? 'Current mid' : 'Exit price'}</div>
          <div className="mono" style={{ fontWeight: 600 }}>
            {d.status === 'open'
              ? d.current_mid !== null ? `${(d.current_mid * 100).toFixed(1)}¢` : '—'
              : d.exit_price !== null ? `${(d.exit_price * 100).toFixed(1)}¢` : '—'}
          </div>
        </div>
        <div style={miniCard}>
          <div style={miniLabel}>Contracts</div>
          <div className="mono" style={{ fontWeight: 600 }}>
            {d.contracts}
            {d.requested_contracts !== undefined && d.requested_contracts !== null && d.requested_contracts > d.contracts && (
              <span style={{ marginLeft: 6, fontSize: 10.5, color: 'var(--warn)' }}>
                (partial · req {d.requested_contracts})
              </span>
            )}
          </div>
        </div>
        <div style={miniCard}>
          <div style={miniLabel}>{d.status === 'open' ? 'Unrealized P&L' : 'P&L'}</div>
          <div className="mono" style={{ fontWeight: 600, color: pnlColor }}>
            {pnlVal !== null ? `$${fmt(pnlVal)}${fmtSignedPct(pnlPctVal)}` : '—'}
          </div>
        </div>
      </div>

      {d.mode === 'live' && (d.exchange_order_id || d.exchange_order_status) && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 8 }}>
          <div style={miniCard}>
            <div style={miniLabel}>Entry order</div>
            <div className="mono" style={{ fontSize: 11.5 }}>{d.exchange_order_id ?? '—'}</div>
          </div>
          <div style={miniCard}>
            <div style={miniLabel}>Exchange status</div>
            <div className="mono" style={{ fontWeight: 600 }}>{d.exchange_order_status ?? '—'}</div>
          </div>
          <div style={miniCard}>
            <div style={miniLabel}>Last exchange sync</div>
            <div className="mono dim" style={{ fontSize: 11.5 }}>
              {d.last_exchange_sync_at
                ? new Date(d.last_exchange_sync_at).toLocaleString()
                : '—'}
            </div>
          </div>
        </div>
      )}

      {d.mode === 'live' && d.exit_order_id != null && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 8 }}>
          <div style={miniCard}>
            <div style={miniLabel}>Exit order</div>
            <div className="mono" style={{ fontSize: 11.5 }}>{d.exit_order_id}</div>
          </div>
          <div style={miniCard}>
            <div style={miniLabel}>Exit fill</div>
            <div className="mono" style={{ fontWeight: 600, color: (d.exit_filled_contracts ?? 0) < (d.exit_requested_contracts ?? 0) ? 'var(--warn)' : 'inherit' }}>
              {d.exit_filled_contracts ?? 0} / {d.exit_requested_contracts ?? '—'} contracts
              {(d.exit_filled_contracts ?? 0) < (d.exit_requested_contracts ?? 0) && d.status === 'open' && (
                <span style={{ marginLeft: 6, fontSize: 10, color: 'var(--warn)' }}>mid-exit</span>
              )}
            </div>
          </div>
          <div style={miniCard}>
            <div style={miniLabel}>Exit fee</div>
            <div className="mono dim">${((d.exit_fee_usd ?? 0)).toFixed(4)}</div>
          </div>
        </div>
      )}

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

      <div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 8 }}>
          <span style={{ fontSize: 10.5, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.08em', color: 'var(--fg-2)' }}>
            {triggerLabel(activeTrigger)}
          </span>
          {!isEntryActive && (
            <button
              onClick={() => setSelectedSignalId(null)}
              style={{ fontSize: 11.5, color: 'var(--accent)', background: 'none', border: 'none', cursor: 'pointer', padding: 0 }}
            >
              ← back to entry signal
            </button>
          )}
          <AnalyzeButton marketId={d.market_id} />
        </div>
        <SelectedSignalPanel signalId={activeSignalId} entrySignal={d.entry_signal} />
      </div>

      {d.status === 'open' && (
        <div style={{ paddingTop: 8, borderTop: '1px solid var(--line-soft)' }}>
          {forceExit.error && (
            <div style={{ marginBottom: 8, fontSize: 11.5, color: 'var(--neg)' }}>{String(forceExit.error)}</div>
          )}
          <button
            className="btn"
            style={{ borderColor: 'var(--neg)', color: 'var(--neg)' }}
            onClick={() => {
              if (window.confirm(`Force-exit position ${positionId}? This cannot be undone.`)) {
                forceExit.mutate()
              }
            }}
            disabled={forceExit.isPending}
          >
            {forceExit.isPending ? 'Closing…' : 'Force Exit'}
          </button>
        </div>
      )}
    </div>
  )
}
