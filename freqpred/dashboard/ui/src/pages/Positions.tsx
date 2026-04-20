import { Fragment, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { getPositions, getPositionDetail } from '../api/positions'
import type { PositionOut } from '../api/types'
import { Badge, Panel, Segmented, LoadingSpinner, ErrorBanner, Sparkline, fmtSignedMoney } from '../components/ui'
import PositionDetailPanel from '../components/PositionDetail'

function PositionSparkline({ positionId, color }: { positionId: string; color: string }) {
  const { data } = useQuery({
    queryKey: ['position-detail', positionId],
    queryFn: () => getPositionDetail(positionId),
    staleTime: 30_000,
  })

  if (!data || data.market_signals.length < 2) {
    return <span style={{ display: 'inline-block', width: 80, height: 20, opacity: 0.2, background: 'var(--line-soft)', borderRadius: 2 }} />
  }

  const sorted = [...data.market_signals].sort(
    (a, b) => new Date(a.created_at).getTime() - new Date(b.created_at).getTime(),
  )
  return <Sparkline data={sorted.map((s) => s.market_mid_at_signal * 100)} w={80} h={20} color={color} />
}

type StatusFilter = 'open' | 'closed' | 'all'

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

  const totals = data?.items.reduce(
    (acc, p) => {
      const pnl = p.status === 'open' ? (p.unrealized_pnl ?? 0) : (p.pnl ?? 0)
      const exposure = p.entry_price * p.contracts
      return { contracts: acc.contracts + p.contracts, exposure: acc.exposure + exposure, pnl: acc.pnl + pnl, weightedEntry: acc.weightedEntry + p.entry_price * p.contracts }
    },
    { contracts: 0, exposure: 0, pnl: 0, weightedEntry: 0 },
  )

  const totalExposure = totals?.exposure ?? 1

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
          <table className="tbl">
            <thead>
              <tr>
                <th>Market</th>
                <th className="c">Dir</th>
                <th className="r">Contracts</th>
                <th className="r">Entry</th>
                <th className="r">Current</th>
                <th className="r" style={{ minWidth: 140 }}>Exposure</th>
                <th className="r" style={{ width: 100 }}>Market history</th>
                <th className="r">P&amp;L</th>
                <th className="r">%</th>
                <th className="c">Status</th>
                <th className="c">Strategy</th>
                <th className="r">Entered</th>
                <th style={{ width: 36 }}></th>
              </tr>
            </thead>
            <tbody>
              {data.items.map((p: PositionOut) => {
                const isExp = expandedId === p.id
                const pnl = p.status === 'open' ? p.unrealized_pnl : p.pnl
                const pnlPct = p.status === 'open' ? p.unrealized_pnl_pct : p.pnl_pct
                const exposure = p.entry_price * p.contracts
                const expoPct = totalExposure > 0 ? (exposure / totalExposure) * 100 : 0
                return (
                  <Fragment key={p.id}>
                    <tr
                      className={isExp ? 'expanded' : ''}
                      onClick={() => toggleExpand(p.id)}
                      style={{ cursor: 'pointer' }}
                    >
                      <td>
                        <span className="ticker-id" style={{ color: 'var(--fg-0)', fontSize: 12 }}>{p.market_id}</span>
                      </td>
                      <td className="c">
                        <Badge kind={p.direction === 'YES' ? 'pos' : 'neg'}>{p.direction}</Badge>
                      </td>
                      <td className="r">{p.contracts}</td>
                      <td className="r">${p.entry_price.toFixed(2)}</td>
                      <td className="r dim">
                        {p.exit_price !== null ? `$${p.exit_price.toFixed(2)}` : '—'}
                      </td>
                      <td className="r">
                        <div className="expo-cell">
                          <div className="expo-bar">
                            <div className="expo-fill" style={{ width: `${expoPct}%` }} />
                          </div>
                          <div className="expo-vals">
                            <span>${exposure.toFixed(2)}</span>
                            <span className="dim" style={{ fontSize: 10.5 }}>{expoPct.toFixed(1)}%</span>
                          </div>
                        </div>
                      </td>
                      <td className="r">
                        <PositionSparkline
                          positionId={p.id}
                          color={pnl !== null && pnl >= 0 ? 'var(--pos)' : 'var(--neg)'}
                        />
                      </td>
                      <td className={`r ${pnl !== null && pnl >= 0 ? 'pos' : 'neg'}`} style={{ fontWeight: 500 }}>
                        {pnl !== null ? fmtSignedMoney(pnl) : '—'}
                      </td>
                      <td className={`r ${pnlPct !== null && pnlPct >= 0 ? 'pos' : 'neg'}`}>
                        {pnlPct !== null ? `${pnlPct >= 0 ? '+' : ''}${(pnlPct * 100).toFixed(1)}%` : '—'}
                      </td>
                      <td className="c">
                        <Badge kind={p.status === 'open' ? 'pos' : p.status === 'closed' ? 'muted' : 'warn'} dot>
                          {p.status}
                        </Badge>
                      </td>
                      <td className="c"><span style={{ fontSize: 11, color: 'var(--fg-2)' }}>{p.strategy_name}</span></td>
                      <td className="r dim" style={{ fontSize: 11, whiteSpace: 'nowrap' }}>
                        {new Date(p.entry_time).toLocaleString()}
                      </td>
                      <td className="c"><span className={`caret${isExp ? ' open' : ''}`}>›</span></td>
                    </tr>
                    {isExp && (
                      <tr className="detail-row">
                        <td colSpan={13}>
                          <PositionDetailPanel positionId={p.id} />
                        </td>
                      </tr>
                    )}
                  </Fragment>
                )
              })}
              {data.items.length === 0 && (
                <tr>
                  <td colSpan={13} style={{ padding: '24px', textAlign: 'center', color: 'var(--fg-3)' }}>No positions</td>
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
                  <td className={`r ${totals.pnl >= 0 ? 'pos' : 'neg'}`}><b>{fmtSignedMoney(totals.pnl)}</b></td>
                  <td className={`r ${totals.pnl >= 0 ? 'pos' : 'neg'}`}>
                    <b>{totals.exposure > 0 ? `${totals.pnl >= 0 ? '+' : ''}${(totals.pnl / totals.exposure * 100).toFixed(1)}%` : '—'}</b>
                  </td>
                  <td></td><td></td><td></td><td></td>
                </tr>
              )}
            </tbody>
          </table>
        </Panel>
      )}
    </div>
  )
}
