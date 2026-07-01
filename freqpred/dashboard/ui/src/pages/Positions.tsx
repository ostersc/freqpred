import React, { Fragment, useEffect, useMemo, useRef, useState } from 'react'
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

interface MarketGroup {
  marketId: string
  positions: PositionOut[]
}

function buildGroups(items: PositionOut[]): MarketGroup[] {
  const order: string[] = []
  const map = new Map<string, PositionOut[]>()
  for (const p of items) {
    if (!map.has(p.market_id)) {
      map.set(p.market_id, [])
      order.push(p.market_id)
    }
    map.get(p.market_id)!.push(p)
  }
  return order.map((marketId) => ({ marketId, positions: map.get(marketId)! }))
}

interface GroupSummary {
  contracts: number
  exposure: number
  pnl: number
  pnlPct: number | null
  weightedEntry: number | null
  weightedCurrent: number | null
  rawCurrentMid: number | null
  directions: string[]
  yesCount: number
  noCount: number
  statuses: string[]
  anyMidExit: boolean
  strategies: string[]
  hasFactbase: boolean
  seriesTicker: string | null
  latestEntry: string
  representativePositionId: string
}

function summarizeGroup(positions: PositionOut[]): GroupSummary {
  let contracts = 0
  let exposure = 0
  let pnl = 0
  let priceWeightSum = 0
  let priceContractSum = 0
  let yesCount = 0
  let noCount = 0
  let anyMidExit = false
  let latestEntry = positions[0].entry_time
  let rawCurrentMid: number | null = null

  for (const p of positions) {
    contracts += p.contracts
    exposure += p.entry_price * p.contracts
    pnl += (p.status === 'open' ? p.unrealized_pnl : p.pnl) ?? 0

    const price = p.status === 'open'
      ? (p.current_mid !== null ? (p.direction === 'YES' ? p.current_mid : 1 - p.current_mid) : null)
      : p.exit_price
    if (price !== null) {
      priceWeightSum += price * p.contracts
      priceContractSum += p.contracts
    }

    if (p.direction === 'YES') yesCount += 1
    else noCount += 1

    if (p.status === 'open' && p.exit_requested_contracts != null && p.exit_filled_contracts != null && p.exit_filled_contracts < p.exit_requested_contracts) {
      anyMidExit = true
    }

    if (p.entry_time > latestEntry) latestEntry = p.entry_time
    if (p.status === 'open' && p.current_mid !== null) rawCurrentMid = p.current_mid
  }

  return {
    contracts,
    exposure,
    pnl,
    pnlPct: exposure > 0 ? pnl / exposure : null,
    weightedEntry: contracts > 0 ? exposure / contracts : null,
    weightedCurrent: priceContractSum > 0 ? priceWeightSum / priceContractSum : null,
    rawCurrentMid,
    directions: Array.from(new Set(positions.map((p) => p.direction))),
    yesCount,
    noCount,
    statuses: Array.from(new Set(positions.map((p) => p.status))),
    anyMidExit,
    strategies: Array.from(new Set(positions.map((p) => p.strategy_name))),
    hasFactbase: positions.some((p) => p.has_factbase),
    seriesTicker: positions.find((p) => p.series_ticker)?.series_ticker ?? null,
    latestEntry,
    representativePositionId: positions[0].id,
  }
}

function PositionRow({
  p,
  isExpanded,
  onToggle,
  totalExposure,
  maxAbsPnl,
  marketHistoryWidth,
  nested = false,
}: {
  p: PositionOut
  isExpanded: boolean
  onToggle: () => void
  totalExposure: number
  maxAbsPnl: number
  marketHistoryWidth: number
  nested?: boolean
}) {
  const pnl = p.status === 'open' ? p.unrealized_pnl : p.pnl
  const pnlPct = p.status === 'open' ? p.unrealized_pnl_pct : p.pnl_pct
  const exposure = p.entry_price * p.contracts
  const expoPct = totalExposure > 0 ? (exposure / totalExposure) * 100 : 0
  const displayedMid = p.current_mid !== null
    ? (p.direction === 'YES' ? p.current_mid : 1 - p.current_mid)
    : null

  return (
    <Fragment>
      <tr
        className={`${isExpanded ? 'expanded' : ''}${nested ? ' positions-nested-row' : ''}`.trim()}
        onClick={onToggle}
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
        <td className="r">{(p.entry_price * 100).toFixed(1)}¢</td>
        <td className="r dim">
          {p.status === 'open'
            ? (displayedMid !== null ? (() => {
                const pill = settlementPillStyle(displayedMid)
                const text = `${(displayedMid * 100).toFixed(1)}¢`
                return pill ? <span style={pill}>{text}</span> : text
              })() : '—')
            : (p.exit_price !== null ? `${(p.exit_price * 100).toFixed(1)}¢` : '—')}
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
            <span className={`caret positions-caret${isExpanded ? ' open' : ''}`}>›</span>
          </span>
        </td>
      </tr>
      {isExpanded && (
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
}

function GroupSummaryRow({
  marketId,
  count,
  summary,
  isExpanded,
  onToggle,
  totalExposure,
  maxAbsPnl,
  marketHistoryWidth,
}: {
  marketId: string
  count: number
  summary: GroupSummary
  isExpanded: boolean
  onToggle: () => void
  totalExposure: number
  maxAbsPnl: number
  marketHistoryWidth: number
}) {
  const expoPct = totalExposure > 0 ? (summary.exposure / totalExposure) * 100 : 0

  return (
    <tr
      className={`positions-group-row${isExpanded ? ' expanded' : ''}`}
      onClick={onToggle}
      style={{ cursor: 'pointer' }}
    >
      <td>
        <span className="ticker-id positions-market" style={{ color: 'var(--fg-0)', fontSize: 12 }} title={marketId}>{marketId}</span>
        <div className="row" style={{ gap: 5, marginTop: 4, flexWrap: 'wrap' }}>
          <Badge kind="info">{count} positions</Badge>
          {summary.hasFactbase && <Badge kind="accent" dot>factbase</Badge>}
          {summary.seriesTicker && <Badge kind="muted">{summary.seriesTicker}</Badge>}
        </div>
      </td>
      <td className="c">
        {summary.directions.length === 1 ? (
          <Badge kind={summary.directions[0] === 'YES' ? 'pos' : 'neg'}>{summary.directions[0]}</Badge>
        ) : (
          <div className="row" style={{ gap: 4, justifyContent: 'center', flexWrap: 'wrap' }}>
            <Badge kind="pos">{summary.yesCount}Y</Badge>
            <Badge kind="neg">{summary.noCount}N</Badge>
          </div>
        )}
      </td>
      <td className="r">{summary.contracts}</td>
      <td className="r">{summary.weightedEntry !== null ? `${(summary.weightedEntry * 100).toFixed(1)}¢` : '—'}</td>
      <td className="r dim">
        {summary.weightedCurrent !== null ? (() => {
          const pill = settlementPillStyle(summary.weightedCurrent)
          const text = `${(summary.weightedCurrent * 100).toFixed(1)}¢`
          return pill ? <span style={pill}>{text}</span> : text
        })() : '—'}
      </td>
      <td className="r">
        <div className="expo-cell expo-cell-inline">
          <div className="expo-bar">
            <div className="expo-fill" style={{ width: `${expoPct}%` }} />
          </div>
          <div className="expo-vals">
            <span>${summary.exposure.toFixed(2)}</span>
            <span className="dim" style={{ fontSize: 10.5 }}>{expoPct.toFixed(1)}%</span>
          </div>
        </div>
      </td>
      <td className="r positions-market-history">
        <span className="positions-market-history-content">
          <PositionSparkline
            positionId={summary.representativePositionId}
            color={summary.pnl >= 0 ? 'var(--pos)' : 'var(--neg)'}
            width={marketHistoryWidth}
          />
        </span>
      </td>
      <td className="c" style={{ position: 'relative', overflow: 'hidden', fontFamily: 'var(--f-mono)', fontVariantNumeric: 'tabular-nums' }}>
        {(() => {
          const barPct = maxAbsPnl > 0 ? (Math.abs(summary.pnl) / maxAbsPnl) * 50 : 0
          const barLeft = summary.pnl >= 0 ? 50 : 50 - barPct
          return (
            <div className="pnl-bar">
              <div className={`pnl-fill ${summary.pnl >= 0 ? 'pos' : 'neg'}`} style={{ left: `${barLeft}%`, width: `${barPct}%` }} />
            </div>
          )
        })()}
        {(() => {
          const pill = lockInPillStyle(summary.rawCurrentMid, summary.pnl)
          const text = fmtSignedMoney(summary.pnl)
          return pill
            ? <span style={pill}>{text}</span>
            : <span className={summary.pnl >= 0 ? 'pos' : 'neg'} style={{ fontWeight: 500 }}>{text}</span>
        })()}
      </td>
      <td className={`r ${summary.pnlPct !== null && summary.pnlPct >= 0 ? 'pos' : 'neg'}`}>
        {(() => {
          const pill = lockInPillStyle(summary.rawCurrentMid, summary.pnl)
          const text = summary.pnlPct !== null ? `${summary.pnlPct >= 0 ? '+' : ''}${(summary.pnlPct * 100).toFixed(1)}%` : '—'
          return pill ? <span style={pill}>{text}</span> : text
        })()}
      </td>
      <td className="c">
        {summary.statuses.length === 1 ? (
          <Badge kind={summary.statuses[0] === 'open' ? 'pos' : summary.statuses[0] === 'closed' ? 'muted' : 'warn'} dot>
            {summary.statuses[0] === 'open' && summary.anyMidExit ? 'mid-exit' : summary.statuses[0]}
          </Badge>
        ) : (
          <span title={`Mixed statuses: ${summary.statuses.join(', ')}`}>
            <Badge kind="warn" dot>mixed</Badge>
          </span>
        )}
      </td>
      <td>
        <span className="positions-strategy" style={{ fontSize: 11, color: 'var(--fg-2)' }} title={summary.strategies.join(', ')}>
          {summary.strategies.length === 1 ? summary.strategies[0] : `${summary.strategies.length} strategies`}
        </span>
      </td>
      <td className="r dim positions-entered-cell" style={{ fontSize: 11 }}>
        <span className="positions-entered-wrap" title={new Date(summary.latestEntry).toLocaleString()}>
          <span className="positions-entered">{formatEnteredAt(summary.latestEntry)}</span>
          <span className={`caret positions-caret${isExpanded ? ' open' : ''}`}>›</span>
        </span>
      </td>
    </tr>
  )
}

export default function Positions() {
  const [status, setStatus] = useState<StatusFilter>('open')
  const [grouped, setGrouped] = useState(true)
  const [expandedId, setExpandedId] = useState<string | null>(null)
  const [expandedGroupId, setExpandedGroupId] = useState<string | null>(null)
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

  function toggleGroup(marketId: string) {
    setExpandedGroupId((prev) => (prev === marketId ? null : marketId))
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

  const groups = useMemo(() => (data ? buildGroups(data.items) : []), [data])
  const groupSummaries = useMemo(
    () => new Map(groups.map((g) => [g.marketId, summarizeGroup(g.positions)])),
    [groups],
  )

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
        ...groups.map((g) => Math.abs(groupSummaries.get(g.marketId)?.pnl ?? 0)),
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
          <label style={{ display: 'inline-flex', alignItems: 'center', gap: 6, cursor: 'pointer', fontSize: 12, color: 'var(--fg-2)' }}>
            <input type="checkbox" checked={grouped} onChange={(e) => setGrouped(e.target.checked)} />
            Group by market
          </label>
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
              {grouped
                ? groups.map((g) => {
                    if (g.positions.length === 1) {
                      const p = g.positions[0]
                      return (
                        <PositionRow
                          key={p.id}
                          p={p}
                          isExpanded={expandedId === p.id}
                          onToggle={() => toggleExpand(p.id)}
                          totalExposure={totalExposure}
                          maxAbsPnl={maxAbsPnl}
                          marketHistoryWidth={marketHistoryWidth}
                        />
                      )
                    }
                    const summary = groupSummaries.get(g.marketId)!
                    const isGroupExpanded = expandedGroupId === g.marketId
                    return (
                      <Fragment key={g.marketId}>
                        <GroupSummaryRow
                          marketId={g.marketId}
                          count={g.positions.length}
                          summary={summary}
                          isExpanded={isGroupExpanded}
                          onToggle={() => toggleGroup(g.marketId)}
                          totalExposure={totalExposure}
                          maxAbsPnl={maxAbsPnl}
                          marketHistoryWidth={marketHistoryWidth}
                        />
                        {isGroupExpanded && g.positions.map((p) => (
                          <PositionRow
                            key={p.id}
                            p={p}
                            isExpanded={expandedId === p.id}
                            onToggle={() => toggleExpand(p.id)}
                            totalExposure={totalExposure}
                            maxAbsPnl={maxAbsPnl}
                            marketHistoryWidth={marketHistoryWidth}
                            nested
                          />
                        ))}
                      </Fragment>
                    )
                  })
                : data.items.map((p: PositionOut) => (
                    <PositionRow
                      key={p.id}
                      p={p}
                      isExpanded={expandedId === p.id}
                      onToggle={() => toggleExpand(p.id)}
                      totalExposure={totalExposure}
                      maxAbsPnl={maxAbsPnl}
                      marketHistoryWidth={marketHistoryWidth}
                    />
                  ))}
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
                  <td className="r">{totals.contracts > 0 ? `${((totals.weightedEntry / totals.contracts) * 100).toFixed(1)}¢` : '—'}</td>
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
