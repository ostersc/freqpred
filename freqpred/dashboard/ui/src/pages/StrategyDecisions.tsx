import { useState, useCallback } from 'react'
import { useQuery } from '@tanstack/react-query'
import { getStrategyDecisions } from '../api/strategyDecisions'
import type { StrategyDecisionOut } from '../api/types'
import { Badge, Panel, LoadingSpinner, ErrorBanner, fmtSignedMoney } from '../components/ui'
import PositionDetailPanel from '../components/PositionDetail'

const PAGE_SIZE = 50

function fmt(v: number | null, d = 2) {
  if (v === null || v === undefined) return '—'
  return v.toFixed(d)
}

function fmtSigned(v: number | null, d = 2) {
  if (v === null || v === undefined) return '—'
  const s = v.toFixed(d)
  return v >= 0 ? `+${s}` : s
}

function exitReasonKind(reason: string | null): 'info' | 'neg' | 'warn' | 'pos' | 'muted' {
  if (!reason) return 'muted'
  const r = reason.toLowerCase()
  if (r === 'market_resolved') return 'info'
  if (r === 'stoploss' || r === 'trailing_stop') return 'neg'
  if (r.startsWith('force_exit')) return 'warn'
  if (r === 'roi' || r === 'signal') return 'pos'
  return 'muted'
}

export default function StrategyDecisions() {
  const [strategy, setStrategy] = useState('')
  const [exitReason, setExitReason] = useState('')
  const [tickerPrefixInput, setTickerPrefixInput] = useState('')
  const [tickerPrefix, setTickerPrefix] = useState('')
  const [debounceTimer, setDebounceTimer] = useState<ReturnType<typeof setTimeout> | null>(null)
  const [dateFrom, setDateFrom] = useState('')
  const [dateTo, setDateTo] = useState('')
  const [offset, setOffset] = useState(0)
  const [expandedId, setExpandedId] = useState<string | null>(null)

  const handleTickerChange = useCallback((value: string) => {
    setTickerPrefixInput(value)
    if (debounceTimer) clearTimeout(debounceTimer)
    const t = setTimeout(() => { setTickerPrefix(value); setOffset(0) }, 300)
    setDebounceTimer(t)
  }, [debounceTimer])

  const { data, isLoading, error } = useQuery({
    queryKey: ['strategy-decisions', strategy, exitReason, tickerPrefix, dateFrom, dateTo, offset],
    queryFn: () => getStrategyDecisions({
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

  function resetFilters() {
    setStrategy(''); setExitReason('')
    setTickerPrefixInput(''); setTickerPrefix('')
    setDateFrom(''); setDateTo(''); setOffset(0)
  }

  return (
    <div className="page">
      <div className="page-head">
        <div>
          <h1 className="page-title">Strategy Decisions</h1>
          <div className="page-subtitle">
            {data ? <><span className="num">{data.total}</span> closed positions{data.total > 0 ? ` — showing ${offset + 1}–${Math.min(offset + PAGE_SIZE, data.total)}` : ''}</> : 'Closed positions'}
          </div>
        </div>
      </div>

      <Panel style={{ marginBottom: 12 }}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr) auto', gap: 12, alignItems: 'end' }}>
          <div className="labeled-field">
            <label className="field-label">Strategy</label>
            <select className="input select" value={strategy} onChange={(e) => { setStrategy(e.target.value); setOffset(0) }}>
              <option value="">All</option>
              {data?.distinct_strategies.map((s) => <option key={s} value={s}>{s}</option>)}
            </select>
          </div>
          <div className="labeled-field">
            <label className="field-label">Exit reason</label>
            <select className="input select" value={exitReason} onChange={(e) => { setExitReason(e.target.value); setOffset(0) }}>
              <option value="">All</option>
              {data?.distinct_exit_reasons.map((r) => <option key={r} value={r}>{r}</option>)}
            </select>
          </div>
          <div className="labeled-field">
            <label className="field-label">Ticker prefix</label>
            <input className="input" placeholder="e.g. KXTRUMPSAY" value={tickerPrefixInput} onChange={(e) => handleTickerChange(e.target.value)} />
          </div>
          <div className="labeled-field">
            <label className="field-label">Exit from</label>
            <input className="input" type="date" value={dateFrom} onChange={(e) => { setDateFrom(e.target.value); setOffset(0) }} />
          </div>
          <div className="labeled-field">
            <label className="field-label">Exit to</label>
            <input className="input" type="date" value={dateTo} onChange={(e) => { setDateTo(e.target.value); setOffset(0) }} />
          </div>
          <button className="btn ghost" onClick={resetFilters}>Reset</button>
        </div>
      </Panel>

      {isLoading && <LoadingSpinner />}
      {error && <ErrorBanner message={String(error)} />}

      {data && (
        <>
          <Panel flush>
            <table className="tbl">
              <thead>
                <tr>
                  <th>Exited</th>
                  <th>Market</th>
                  <th className="c">Dir</th>
                  <th className="r">Ctr</th>
                  <th className="r">Entry</th>
                  <th className="r">Exit</th>
                  <th className="c">Result</th>
                  <th className="c">Exit Reason</th>
                  <th className="r">Actual P&amp;L</th>
                  <th className="r">Counterfactual</th>
                  <th className="r">Best Prior Ask</th>
                  <th className="r">Entry Eff.</th>
                  <th style={{ width: 36 }}></th>
                </tr>
              </thead>
              <tbody>
                {data.items.map((row: StrategyDecisionOut) => {
                  const isExp = expandedId === row.id
                  return (
                    <>
                      <tr
                        key={row.id}
                        className={isExp ? 'expanded' : ''}
                        onClick={() => setExpandedId(isExp ? null : row.id)}
                        style={{ cursor: 'pointer' }}
                      >
                        <td className="dim" style={{ fontSize: 11, whiteSpace: 'nowrap' }}>
                          {row.exit_time ? new Date(row.exit_time).toLocaleString() : '—'}
                        </td>
                        <td>
                          <div className="ticker-id" style={{ color: 'var(--fg-0)', fontSize: 11.5 }}>{row.market_id}</div>
                          {row.market_question && <div style={{ fontSize: 11, color: 'var(--fg-2)', marginTop: 2 }}>{row.market_question}</div>}
                        </td>
                        <td className="c">
                          <Badge kind={row.direction === 'YES' ? 'pos' : 'neg'}>{row.direction}</Badge>
                        </td>
                        <td className="r">{row.contracts}</td>
                        <td className="r">${fmt(row.entry_price)}</td>
                        <td className="r">{row.exit_price !== null ? `$${fmt(row.exit_price)}` : <span className="muted">—</span>}</td>
                        <td className="c">
                          {row.market_result === 'yes' && <span className="pos" style={{ fontWeight: 500 }}>✓ YES</span>}
                          {row.market_result === 'no' && <span className="neg" style={{ fontWeight: 500 }}>✗ NO</span>}
                          {!row.market_result && <span className="muted">—</span>}
                        </td>
                        <td className="c">
                          <Badge kind={exitReasonKind(row.exit_reason)}>{row.exit_reason ?? '—'}</Badge>
                        </td>
                        <td className={`r ${row.pnl !== null && row.pnl >= 0 ? 'pos' : 'neg'}`} style={{ fontWeight: 500 }}>
                          {row.pnl !== null ? fmtSignedMoney(row.pnl) : '—'}
                        </td>
                        <td className="r dim">
                          {row.counterfactual_pnl_usd !== null ? fmtSignedMoney(row.counterfactual_pnl_usd) : '—'}
                        </td>
                        <td className="r">{row.best_prior_ask !== null ? `$${fmt(row.best_prior_ask)}` : <span className="muted">—</span>}</td>
                        <td className={`r ${row.entry_efficiency_usd !== null && row.entry_efficiency_usd < 0 ? 'neg' : 'pos'}`}>
                          {row.entry_efficiency_usd !== null ? fmtSigned(row.entry_efficiency_usd) : <span className="muted">—</span>}
                        </td>
                        <td className="c"><span className={`caret${isExp ? ' open' : ''}`}>›</span></td>
                      </tr>
                      {isExp && (
                        <tr key={`${row.id}-d`} className="detail-row">
                          <td colSpan={13}>
                            <PositionDetailPanel positionId={row.id} />
                          </td>
                        </tr>
                      )}
                    </>
                  )
                })}
                {data.items.length === 0 && (
                  <tr>
                    <td colSpan={13} style={{ padding: '24px', textAlign: 'center', color: 'var(--fg-3)' }}>
                      No closed positions match the current filters
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </Panel>
          <div className="pagination">
            <button className="btn sm" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - PAGE_SIZE))}>Previous</button>
            <span>{offset + 1}–{Math.min(offset + PAGE_SIZE, data.total)} of {data.total}</span>
            <button className="btn sm" disabled={offset + PAGE_SIZE >= data.total} onClick={() => setOffset(offset + PAGE_SIZE)}>Next</button>
          </div>
        </>
      )}
    </div>
  )
}
