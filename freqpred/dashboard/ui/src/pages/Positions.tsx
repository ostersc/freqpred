import React, { Fragment, useEffect, useRef, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { getPositions, getPositionDetail } from '../api/positions'
import type { PositionOut } from '../api/types'
import { Badge, Panel, Segmented, LoadingSpinner, ErrorBanner, Sparkline, fmtSignedMoney } from '../components/ui'
import PositionDetailPanel from '../components/PositionDetail'

function formatEnteredAt(value: string): string {
  const d = new Date(value)
  const hour24 = d.getHours()
  const hour12 = hour24 % 12 || 12
  const minute = String(d.getMinutes()).padStart(2, '0')
  const suffix = hour24 >= 12 ? 'p' : 'a'
  return `${d.getMonth() + 1}/${d.getDate()} ${hour12}:${minute}${suffix}`
}

function PositionSparkline({ positionId, color, width }: { positionId: string; color: string; width: number }) {
  const { data } = useQuery({
    queryKey: ['position-detail', positionId],
    queryFn: () => getPositionDetail(positionId),
    staleTime: 30_000,
  })

  if (!data || data.market_signals.length < 2) {
    return <span style={{ display: 'inline-block', width, height: 12, opacity: 0.2, background: 'var(--line-soft)', borderRadius: 2 }} />
  }

  const sorted = [...data.market_signals].sort(
    (a, b) => new Date(a.created_at).getTime() - new Date(b.created_at).getTime(),
  )
  const sparkData = sorted.map((s) => s.market_mid_at_signal * 100)
  const sparkTs = sorted.map((s) => new Date(s.created_at).getTime())
  if (data.status === 'open' && data.current_mid !== null) {
    sparkData.push(data.current_mid * 100)
    sparkTs.push(Date.now())
  }
  return <Sparkline data={sparkData} timestamps={sparkTs} w={width} h={12} color={color} />
}

function settlementIntensity(currentMid: number | null): number {
  if (currentMid === null) return 0
  const price = currentMid * 100
  if (price >= 90) return (price - 90) / 10
  if (price <= 10) return (10 - price) / 10
  return 0
}

function pillStyle(bg: string, fg: string, intensity: number): React.CSSProperties {
  const mixPct = Math.round(intensity * 100)
  return {
    display: 'inline-block',
    padding: '1px 7px',
    borderRadius: 4,
    fontSize: 10.5,
    fontWeight: 500,
    letterSpacing: '0.02em',
    background: `color-mix(in oklch, ${bg} ${mixPct}%, transparent)`,
    color: fg,
  }
}

function settlementPillStyle(currentMid: number | null): React.CSSProperties | null {
  if (currentMid === null) return null
  const price = currentMid * 100
  const intensity = settlementIntensity(currentMid)
  if (intensity === 0) return null
  const isHigh = price >= 90
  return pillStyle(
    isHigh ? 'var(--pos-soft)' : 'var(--neg-soft)',
    isHigh ? 'var(--pos)' : 'var(--neg)',
    intensity,
  )
}

function lockInPillStyle(currentMid: number | null, pnl: number | null): React.CSSProperties | null {
  const intensity = settlementIntensity(currentMid)
  if (intensity === 0 || pnl === null) return null
  const profit = pnl >= 0
  return pillStyle(
    profit ? 'var(--pos-soft)' : 'var(--neg-soft)',
    profit ? 'var(--pos)' : 'var(--neg)',
    intensity,
  )
}

type StatusFilter = 'open' | 'closed' | 'all'

export default function Positions() {
  const [status, setStatus] = useState<StatusFilter>('open')
  const [expandedId, setExpandedId] = useState<string | null>(null)
  const marketHistoryHeaderRef = useRef<HTMLSpanElement | null>(null)
  const [marketHistoryWidth, setMarketHistoryWidth] = useState(96)

  const { data, isLoading, error } = useQuery({
    queryKey: ['positions', status],
    queryFn: () => getPositions(status),
    refetchInterval: status === 'open' ? 60_000 : false,
  })

  function toggleExpand(id: string) {
    setExpandedId((prev) => (prev === id ? null : id))
  }

  useEffect(() => {
    const el = marketHistoryHeaderRef.current
    if (!el) return

    const updateWidth = () => {
      const next = Math.ceil(el.getBoundingClientRect().width)
      if (next > 0) setMarketHistoryWidth(next)
    }

    updateWidth()

    const observer = new ResizeObserver(updateWidth)
    observer.observe(el)
    return () => observer.disconnect()
  }, [])

  const totals = data?.items.reduce(
    (acc, p) => {
      const pnl = p.status === 'open' ? (p.unrealized_pnl ?? 0) : (p.pnl ?? 0)
      const exposure = p.entry_price * p.contracts
      return { contracts: acc.contracts + p.contracts, exposure: acc.exposure + exposure, pnl: acc.pnl + pnl, weightedEntry: acc.weightedEntry + p.entry_price * p.contracts }
    },
    { contracts: 0, exposure: 0, pnl: 0, weightedEntry: 0 },
  )

  const totalExposure = totals?.exposure ?? 1

  const maxAbsPnl = data
    ? Math.max(
        Math.abs(totals?.pnl ?? 0),
        ...data.items.map((p) => Math.abs(p.status === 'open' ? (p.unrealized_pnl ?? 0) : (p.pnl ?? 0))),
      )
    : 1

  return (
    <div className="page">
      <div className="page-head">
        <div>
          <h1 className="page-title">Positions</h1>
          <div className="page-subtitle">
            <span className="num">{data?.total ?? '—'}</span> positions
            {status === 'open' && ' · refreshes every 60s'}
          </div>
        </div>
        <div className="row">
          {totals && (
            <>
              <div className="chip">
                Unrealized{' '}
                <b className={`mono ${totals.pnl >= 0 ? 'pos' : 'neg'}`} style={{ marginLeft: 6 }}>
                  {fmtSignedMoney(totals.pnl)}
                </b>
              </div>
              <div className="chip">
                Exposure <b className="mono" style={{ marginLeft: 6 }}>${totals.exposure.toFixed(2)}</b>
              </div>
            </>
          )}
          <Segmented<StatusFilter>
            items={['open', 'closed', 'all']}
            value={status}
            onChange={setStatus}
          />
        </div>
      </div>

      {isLoading && <LoadingSpinner />}
      {error && <ErrorBanner message={String(error)} />}

      {data && (
        <Panel flush>
          <div className="tbl-scroll">
          <table className="tbl tbl-positions">
            <thead>
              <tr>
                <th>Market</th>
                <th className="c">Dir</th>
                <th className="r">Contracts</th>
                <th className="r">Entry</th>
                <th className="r">Current</th>
                <th className="r">Exposure</th>
                <th className="r positions-market-history">
                  <span ref={marketHistoryHeaderRef} className="positions-market-history-content">Market history</span>
                </th>
                <th className="c">P&amp;L</th>
                <th className="r">%</th>
                <th className="c">Status</th>
                <th>Strategy</th>
                <th className="r">Entered</th>
              </tr>
            </thead>
            <tbody>
              {data.items.map((p: PositionOut) => {
                const isExp = expandedId === p.id
                const pnl = p.status === 'open' ? p.unrealized_pnl : p.pnl
                const pnlPct = p.status === 'open' ? p.unrealized_pnl_pct : p.pnl_pct
                const exposure = p.entry_price * p.contracts
                const expoPct = totalExposure > 0 ? (exposure / totalExposure) * 100 : 0
                const displayedMid = p.current_mid !== null
                  ? (p.direction === 'YES' ? p.current_mid : 1 - p.current_mid)
                  : null
                return (
                  <Fragment key={p.id}>
                    <tr
                      className={isExp ? 'expanded' : ''}
                      onClick={() => toggleExpand(p.id)}
                      style={{ cursor: 'pointer' }}
                    >
                      <td>
                        <span className="ticker-id positions-market" style={{ color: 'var(--fg-0)', fontSize: 12 }} title={p.market_id}>{p.market_id}</span>
                        {(p.has_factbase || p.series_ticker) && (
                          <div className="row" style={{ gap: 5, marginTop: 4, flexWrap: 'wrap' }}>
                            {p.has_factbase && <Badge kind="accent" dot>factbase</Badge>}
                            {p.series_ticker && <Badge kind="muted">{p.series_ticker}</Badge>}
                          </div>
                        )}
                      </td>
                      <td className="c">
                        <Badge kind={p.direction === 'YES' ? 'pos' : 'neg'}>{p.direction}</Badge>
                      </td>
                      <td className="r">
                        {p.contracts}
                        {p.requested_contracts !== undefined && p.requested_contracts !== null && p.requested_contracts > p.contracts && (
                          <span title={`Partial fill: filled ${p.contracts} of ${p.requested_contracts} requested`} style={{ marginLeft: 4, fontSize: 10, color: 'var(--warn)' }}>
                            ◐
                          </span>
                        )}
                        {p.exit_requested_contracts != null && p.exit_filled_contracts != null && p.exit_filled_contracts < p.exit_requested_contracts && p.status === 'open' && (
                          <span title={`Mid-exit: closed ${p.exit_filled_contracts} of ${p.exit_requested_contracts} contracts`} style={{ marginLeft: 4, fontSize: 10, color: 'var(--warn)' }}>
                            ↘
                          </span>
                        )}
                      </td>
                      <td className="r">${p.entry_price.toFixed(2)}</td>
                      <td className="r dim">
                        {p.status === 'open'
                          ? (displayedMid !== null ? (() => {
                              const pill = settlementPillStyle(displayedMid)
                              const text = `${(displayedMid * 100).toFixed(1)}¢`
                              return pill ? <span style={pill}>{text}</span> : text
                            })() : '—')
                          : (p.exit_price !== null ? `$${p.exit_price.toFixed(2)}` : '—')}
                      </td>
                      <td className="r">
                        <div className="expo-cell expo-cell-inline">
                          <div className="expo-bar">
                            <div className="expo-fill" style={{ width: `${expoPct}%` }} />
                          </div>
                          <div className="expo-vals">
                            <span>${exposure.toFixed(2)}</span>
                            <span className="dim" style={{ fontSize: 10.5 }}>{expoPct.toFixed(1)}%</span>
                          </div>
                        </div>
                      </td>
                      <td className="r positions-market-history">
                        <span className="positions-market-history-content">
                          <PositionSparkline
                            positionId={p.id}
                            color={pnl !== null && pnl >= 0 ? 'var(--pos)' : 'var(--neg)'}
                            width={marketHistoryWidth}
                          />
                        </span>
                      </td>
                      <td className="c" style={{ position: 'relative', overflow: 'hidden', fontFamily: 'var(--f-mono)', fontVariantNumeric: 'tabular-nums' }}>
                        {pnl !== null && (() => {
                          const barPct = maxAbsPnl > 0 ? (Math.abs(pnl) / maxAbsPnl) * 50 : 0
                          const barLeft = pnl >= 0 ? 50 : 50 - barPct
                          return (
                            <div className="pnl-bar">
                              <div className={`pnl-fill ${pnl >= 0 ? 'pos' : 'neg'}`} style={{ left: `${barLeft}%`, width: `${barPct}%` }} />
                            </div>
                          )
                        })()}
                        {(() => {
                          const pill = lockInPillStyle(p.current_mid, pnl)
                          const text = pnl !== null ? fmtSignedMoney(pnl) : '—'
                          return pill
                            ? <span style={pill}>{text}</span>
                            : <span className={pnl !== null && pnl >= 0 ? 'pos' : 'neg'} style={{ fontWeight: 500 }}>{text}</span>
                        })()}
                      </td>
                      <td className={`r ${pnlPct !== null && pnlPct >= 0 ? 'pos' : 'neg'}`}>
                        {(() => {
                          const pill = lockInPillStyle(p.current_mid, pnl)
                          const text = pnlPct !== null ? `${pnlPct >= 0 ? '+' : ''}${(pnlPct * 100).toFixed(1)}%` : '—'
                          return pill ? <span style={pill}>{text}</span> : text
                        })()}
                      </td>
                      <td className="c">
                        <Badge kind={p.status === 'open' ? 'pos' : p.status === 'closed' ? 'muted' : 'warn'} dot>
                          {p.status === 'open' && p.exit_requested_contracts != null && p.exit_filled_contracts != null && p.exit_filled_contracts < p.exit_requested_contracts
                            ? 'mid-exit'
                            : p.status}
                        </Badge>
                      </td>
                      <td><span className="positions-strategy" style={{ fontSize: 11, color: 'var(--fg-2)' }} title={p.strategy_name}>{p.strategy_name}</span></td>
                      <td className="r dim positions-entered-cell" style={{ fontSize: 11 }}>
                        <span className="positions-entered-wrap" title={new Date(p.entry_time).toLocaleString()}>
                          <span className="positions-entered">{formatEnteredAt(p.entry_time)}</span>
                          <span className={`caret positions-caret${isExp ? ' open' : ''}`}>›</span>
                        </span>
                      </td>
                    </tr>
                    {isExp && (
                      <tr className="detail-row">
                        <td colSpan={12}>
                          <div className="positions-detail-wrap">
                            <PositionDetailPanel positionId={p.id} />
                          </div>
                        </td>
                      </tr>
                    )}
                  </Fragment>
                )
              })}
              {data.items.length === 0 && (
                <tr>
                  <td colSpan={12} style={{ padding: '24px', textAlign: 'center', color: 'var(--fg-3)' }}>No positions</td>
                </tr>
              )}
              {totals && data.items.length > 0 && (
                <tr style={{ background: 'var(--bg-2)', fontWeight: 500 }}>
                  <td><b>TOTAL</b></td>
                  <td></td>
                  <td className="r">{totals.contracts}</td>
                  <td className="r">${totals.contracts > 0 ? (totals.weightedEntry / totals.contracts).toFixed(2) : '—'}</td>
                  <td></td>
                  <td className="r"><b>${totals.exposure.toFixed(2)}</b></td>
                  <td></td>
                  <td className={`c ${totals.pnl >= 0 ? 'pos' : 'neg'}`} style={{ fontFamily: 'var(--f-mono)', fontVariantNumeric: 'tabular-nums' }}><b>{fmtSignedMoney(totals.pnl)}</b></td>
                  <td className={`r ${totals.pnl >= 0 ? 'pos' : 'neg'}`}>
                    <b>{totals.exposure > 0 ? `${totals.pnl >= 0 ? '+' : ''}${(totals.pnl / totals.exposure * 100).toFixed(1)}%` : '—'}</b>
                  </td>
                  <td></td><td></td><td></td>
                </tr>
              )}
            </tbody>
          </table>
          </div>
        </Panel>
      )}
    </div>
  )
}
