import { Fragment, useState, useCallback } from 'react'
import { useQuery } from '@tanstack/react-query'
import { getMarkets, getMarket } from '../api/markets'
import type { MarketOut, MarketDetailOut } from '../api/types'
import { Badge, Panel, MiniStat, Segmented, ProbBar, Icon, LoadingSpinner, ErrorBanner } from '../components/ui'
import AnalyzeButton from '../components/AnalyzeButton'

function MarketEdgeCell({ marketId }: { marketId: string }) {
  const { data } = useQuery({
    queryKey: ['market-detail', marketId],
    queryFn: () => getMarket(marketId),
    staleTime: 30_000,
  })
  const sig = data?.current_signal
  if (!sig) return <span className="muted">—</span>
  return (
    <Badge kind={sig.edge >= 0 ? 'pos' : 'neg'}>
      {sig.edge >= 0 ? '+' : ''}{(sig.edge * 100).toFixed(1)}%
    </Badge>
  )
}

type StatusFilter = 'open' | 'closed' | 'all'

function closeTimeLabel(iso: string) {
  const d = new Date(iso)
  const diffDays = Math.round((d.getTime() - Date.now()) / 86_400_000)
  if (diffDays < 0) return `${Math.abs(diffDays)}d ago`
  if (diffDays === 0) return 'today'
  return `${diffDays}d`
}

function MarketDetail({ market }: { market: MarketOut }) {
  const { data, isLoading, error } = useQuery({
    queryKey: ['market-detail', market.id],
    queryFn: () => getMarket(market.id),
    staleTime: 30_000,
  })

  if (isLoading) return <div style={{ padding: '14px 20px', color: 'var(--fg-2)', fontSize: 12 }}>Loading…</div>
  if (error) return <div style={{ padding: '14px 20px', color: 'var(--neg)', fontSize: 12 }}>{String(error)}</div>
  if (!data) return null

  const d: MarketDetailOut = data
  const sig = d.current_signal

  return (
    <div style={{ padding: '16px 20px', display: 'grid', gridTemplateColumns: '2fr 1fr', gap: 20 }}>
      <div>
        <div style={{ fontSize: 11, textTransform: 'uppercase', letterSpacing: '0.08em', color: 'var(--fg-2)', marginBottom: 8 }}>Market Question</div>
        <div style={{ fontSize: 13, lineHeight: 1.6, color: 'var(--fg-1)', marginBottom: 16 }}>{d.question}</div>

        {sig ? (
          <div style={{ padding: 14, background: 'var(--bg-0)', borderRadius: 8, border: '1px solid var(--line-soft)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 10 }}>
              <div style={{ fontSize: 11, textTransform: 'uppercase', letterSpacing: '0.08em', color: 'var(--fg-2)' }}>Current Signal</div>
              <AnalyzeButton marketId={market.id} />
            </div>
            <div style={{ display: 'flex', gap: 20, marginBottom: 10 }}>
              <div><span className="dim">Our prob:</span> <b className="mono">{(sig.estimated_probability * 100).toFixed(1)}%</b></div>
              <div><span className="dim">Market:</span> <b className="mono">{(sig.market_mid_at_signal * 100).toFixed(1)}%</b></div>
              <div><span className="dim">Edge:</span> <b className={`mono ${sig.edge >= 0 ? 'pos' : 'neg'}`}>{sig.edge >= 0 ? '+' : ''}{(sig.edge * 100).toFixed(1)}%</b></div>
              <div><span className="dim">Confidence:</span> <b className="mono">{(sig.confidence * 100).toFixed(1)}%</b></div>
            </div>
            <ProbBar ours={sig.estimated_probability} market={sig.market_mid_at_signal} />
            {sig.reasoning && (
              <div style={{ marginTop: 10, fontSize: 12, lineHeight: 1.6, color: 'var(--fg-1)' }}>{sig.reasoning}</div>
            )}
          </div>
        ) : (
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <div style={{ color: 'var(--fg-3)', fontSize: 12 }}>No signal yet</div>
            <AnalyzeButton marketId={market.id} />
          </div>
        )}
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
        <div className="grid grid-2">
          <MiniStat label="Mid" value={`${(d.mid_price * 100).toFixed(1)}¢`} />
          <MiniStat label="Bid / Ask" value={`${(d.yes_bid * 100).toFixed(1)}¢ / ${(d.yes_ask * 100).toFixed(1)}¢`} />
          <MiniStat label="Volume 24h" value={d.volume_24h.toLocaleString(undefined, { maximumFractionDigits: 0 })} />
          <MiniStat label="Closes" value={new Date(d.close_time).toLocaleDateString()} />
        </div>
        <div className="ticker-id" style={{ marginTop: 4 }}>{d.id}</div>
      </div>
    </div>
  )
}

export default function Markets() {
  const [status, setStatus] = useState<StatusFilter>('open')
  const [search, setSearch] = useState('')
  const [debouncedSearch, setDebouncedSearch] = useState('')
  const [debounceTimer, setDebounceTimer] = useState<ReturnType<typeof setTimeout> | null>(null)
  const [expandedId, setExpandedId] = useState<string | null>(null)

  const handleSearchChange = useCallback((value: string) => {
    setSearch(value)
    if (debounceTimer) clearTimeout(debounceTimer)
    const t = setTimeout(() => setDebouncedSearch(value), 300)
    setDebounceTimer(t)
  }, [debounceTimer])

  const { data, isLoading, error } = useQuery({
    queryKey: ['markets', status, debouncedSearch],
    queryFn: () => getMarkets({ status, search: debouncedSearch || undefined, limit: 100 }),
    staleTime: 30_000,
  })

  function toggleExpand(id: string) {
    setExpandedId((prev) => (prev === id ? null : id))
  }

  return (
    <div className="page">
      <div className="page-head">
        <div>
          <h1 className="page-title">Markets</h1>
          <div className="page-subtitle">
            <span className="num">{data?.total ?? '—'}</span> markets total
          </div>
        </div>
        <div className="row">
          <div style={{ position: 'relative', width: 360 }}>
            <div style={{ position: 'absolute', left: 10, top: '50%', transform: 'translateY(-50%)', color: 'var(--fg-2)' }}>
              <Icon name="search" />
            </div>
            <input
              className="input"
              style={{ paddingLeft: 30 }}
              placeholder="Search by question or market ID…"
              value={search}
              onChange={(e) => handleSearchChange(e.target.value)}
            />
          </div>
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
                <th style={{ width: '50%' }}>Question</th>
                <th className="r">Mid</th>
                <th className="r">Vol 24h</th>
                <th className="r">Closes</th>
                <th className="c">Edge</th>
                <th className="c" style={{ width: 80 }}>Status</th>
                <th style={{ width: 36 }}></th>
              </tr>
            </thead>
            <tbody>
              {data.items.map((m: MarketOut) => {
                const isExp = expandedId === m.id
                return (
                  <Fragment key={m.id}>
                    <tr
                      className={isExp ? 'expanded' : ''}
                      onClick={() => toggleExpand(m.id)}
                      style={{ cursor: 'pointer' }}
                    >
                      <td>
                        <div style={{ fontWeight: 500, marginBottom: 2 }}>{m.question}</div>
                        <div className="ticker-id">{m.id}</div>
                      </td>
                      <td className="r">{(m.mid_price * 100).toFixed(1)}¢</td>
                      <td className="r">{m.volume_24h.toLocaleString(undefined, { maximumFractionDigits: 0 })}</td>
                      <td className="r dim">{closeTimeLabel(m.close_time)}</td>
                      <td className="c">
                        {m.current_signal_id
                          ? <MarketEdgeCell marketId={m.id} />
                          : <span className="muted">—</span>}
                      </td>
                      <td className="c">
                        <Badge kind={m.status === 'active' ? 'pos' : 'muted'} dot>
                          {m.status === 'active' ? 'open' : m.status}
                        </Badge>
                      </td>
                      <td className="c"><span className={`caret${isExp ? ' open' : ''}`}>›</span></td>
                    </tr>
                    {isExp && (
                      <tr className="detail-row">
                        <td colSpan={7}>
                          <MarketDetail market={m} />
                        </td>
                      </tr>
                    )}
                  </Fragment>
                )
              })}
              {data.items.length === 0 && (
                <tr>
                  <td colSpan={7} style={{ padding: '24px', textAlign: 'center', color: 'var(--fg-3)' }}>No markets found</td>
                </tr>
              )}
            </tbody>
          </table>
        </Panel>
      )}
    </div>
  )
}
