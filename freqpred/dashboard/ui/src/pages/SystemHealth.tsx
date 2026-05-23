import React from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { getSystemHealth } from '../api/health'
import { setRunState, shutdown } from '../api/system'
import { Badge, Panel, Stat, LoadingSpinner, ErrorBanner, fmtUptime } from '../components/ui'

function statusKind(s: string): 'pos' | 'neg' | 'warn' | 'info' | 'muted' {
  if (s === 'running' || s === 'connected' || s === 'ok') return 'pos'
  if (s === 'error' || s === 'halted' || s === 'stale') return 'neg'
  if (s === 'warn' || s === 'degraded') return 'warn'
  if (s === 'paper' || s === 'signal-only') return 'info'
  return 'muted'
}

function fmtAgeSecs(s: number | null): string {
  if (s === null) return '—'
  return fmtUptime(s)
}

// Interpolates: red (fresh) → yellow (mid) → grey (near 24h expiry)
function errorTextColor(ageSeconds: number): string {
  const t = Math.min(1, ageSeconds / 86400)
  if (t < 0.5) {
    const pct = (1 - t / 0.5) * 100
    return `color-mix(in oklch, var(--neg) ${pct.toFixed(1)}%, var(--warn))`
  }
  const pct = (1 - (t - 0.5) / 0.5) * 100
  return `color-mix(in oklch, var(--warn) ${pct.toFixed(1)}%, var(--fg-3))`
}

export default function SystemHealth() {
  const queryClient = useQueryClient()

  const { data, isLoading, error } = useQuery({
    queryKey: ['systemHealth'],
    queryFn: getSystemHealth,
    refetchInterval: 15_000,
  })

  const stateMutation = useMutation({
    mutationFn: (state: 'running' | 'paused' | 'stopped') => setRunState(state),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['systemHealth'] }),
  })

  const shutdownMutation = useMutation({
    mutationFn: shutdown,
  })

  return (
    <div className="page">
      <div className="page-head">
        <div>
          <h1 className="page-title">System Health</h1>
          <div className="page-subtitle">Heartbeat across every subsystem of the trading bot.</div>
        </div>
        <div className="chip">
          <span style={{ width: 6, height: 6, borderRadius: '50%', background: 'var(--pos)', boxShadow: '0 0 8px var(--pos)', display: 'inline-block' }} />
          {' '}refreshes every 15s
        </div>
      </div>

      {isLoading && <LoadingSpinner />}
      {error && <ErrorBanner message={String(error)} />}

      {data && (
        <>
          {data.circuit_breakers.trading_halted && (
            <div className="error-banner" style={{ marginBottom: 12 }}>
              Circuit breaker active — trading halted. Reason: {data.circuit_breakers.reason ?? 'unknown'}
            </div>
          )}

          <div className="grid" style={{ gridTemplateColumns: '2fr 1fr 1fr 1fr 1fr 1fr 1fr', marginBottom: 8 }}>
            <div className="stat">
              <div className="stat-label">Run state</div>
              <div style={{ marginBottom: 10 }}>
                <Badge kind={statusKind(data.run_state)} dot>{data.run_state}</Badge>
              </div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                <button
                  className="btn sm"
                  disabled={data.run_state === 'running' || stateMutation.isPending}
                  onClick={() => stateMutation.mutate('running')}
                >Start</button>
                <button
                  className="btn sm"
                  disabled={data.run_state === 'paused' || stateMutation.isPending}
                  onClick={() => stateMutation.mutate('paused')}
                >Pause</button>
                <button
                  className="btn sm"
                  disabled={data.run_state === 'stopped' || stateMutation.isPending}
                  onClick={() => stateMutation.mutate('stopped')}
                >Stop</button>
                <button
                  className="btn sm"
                  style={{ borderColor: 'var(--neg)', color: 'var(--neg)' }}
                  disabled={shutdownMutation.isPending}
                  onClick={() => {
                    if (window.confirm('Shut down freqpred?\n\nThis sends SIGTERM to the process. The dashboard and all loops will exit. You cannot restart from the dashboard — you must restart the process manually.')) {
                      shutdownMutation.mutate()
                    }
                  }}
                >Shutdown</button>
              </div>
              {stateMutation.isError && <div className="neg" style={{ fontSize: 11, marginTop: 6 }}>{String(stateMutation.error)}</div>}
            </div>
            <Stat label="Mode" value={<Badge kind={statusKind(data.mode)}>{data.mode}</Badge>} sub={data.mode === 'paper' ? 'no real orders sent' : 'live trading'} />
            <Stat label="Database" value={<Badge kind={data.db_ok ? 'pos' : 'neg'} dot>{data.db_ok ? 'connected' : 'error'}</Badge>} sub="primary" />
            <Stat label="Uptime" value={fmtUptime(data.uptime_seconds)} />
            <Stat label="Open positions" value={String(data.open_positions)} />
            <Stat label="Pending orders" value={String(data.pending_orders)} sub={`Oldest age: ${fmtAgeSecs(data.oldest_pending_order_age_seconds)}`} />
            {data.pending_orders_detail && data.pending_orders_detail.length > 0 && (
              <div style={{ gridColumn: '1 / -1' }}>
                <Panel title="Pending order detail">
                  <table className="tbl" style={{ fontSize: 12 }}>
                    <thead>
                      <tr>
                        <th style={{ textAlign: 'left' }}>Market</th>
                        <th style={{ textAlign: 'right' }}>Age</th>
                        <th style={{ textAlign: 'left' }}>Exchange status</th>
                        <th style={{ textAlign: 'right' }}>Filled / Requested</th>
                        <th style={{ textAlign: 'right' }}>Last sync</th>
                      </tr>
                    </thead>
                    <tbody>
                      {data.pending_orders_detail.map((p) => (
                        <tr key={p.position_id}>
                          <td className="mono">{p.market_id}</td>
                          <td className="mono" style={{ textAlign: 'right' }}>{fmtUptime(p.age_seconds)}</td>
                          <td className="mono">{p.exchange_order_status ?? '—'}</td>
                          <td className="mono" style={{ textAlign: 'right' }}>
                            {p.filled_contracts} / {p.requested_contracts ?? '—'}
                          </td>
                          <td className="mono dim" style={{ textAlign: 'right' }}>
                            {p.last_exchange_sync_at
                              ? new Date(p.last_exchange_sync_at).toLocaleTimeString()
                              : '—'}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </Panel>
              </div>
            )}
            <div className="stat">
              <div className="stat-label">API errors (last hour)</div>
              <div style={{ display: 'flex', gap: 20, marginTop: 6 }}>
                <div>
                  <div className="dim" style={{ fontSize: 11 }}>Kalshi</div>
                  <div className={`mono ${data.api_errors.kalshi_errors_last_hour > 0 ? 'neg' : 'pos'}`} style={{ fontSize: 20, fontWeight: 600 }}>{data.api_errors.kalshi_errors_last_hour}</div>
                </div>
                <div>
                  <div className="dim" style={{ fontSize: 11 }}>LLM</div>
                  <div className={`mono ${data.api_errors.llm_errors_last_hour > 0 ? 'neg' : 'pos'}`} style={{ fontSize: 20, fontWeight: 600 }}>{data.api_errors.llm_errors_last_hour}</div>
                </div>
              </div>
            </div>
          </div>

          <div className="grid grid-3" style={{ marginBottom: 8 }}>
            <Panel title="Circuit breaker" action={<Badge kind={data.circuit_breakers.trading_halted ? 'neg' : 'pos'} dot>{data.circuit_breakers.trading_halted ? 'halted' : 'ok'}</Badge>}>
              <table className="tbl" style={{ fontSize: 12.5 }}>
                <tbody>
                  <tr>
                    <td style={{ padding: '8px 0', border: 'none' }}>Trading halted</td>
                    <td style={{ padding: '8px 0', border: 'none', textAlign: 'right' }} className="mono">{String(data.circuit_breakers.trading_halted)}</td>
                  </tr>
                  <tr>
                    <td style={{ padding: '8px 0', border: 'none' }}>Daily loss <span className="dim">(since {new Date(data.circuit_breakers.daily_loss_window_start).toLocaleTimeString()})</span></td>
                    <td style={{ padding: '8px 0', border: 'none', textAlign: 'right' }} className={`mono ${data.circuit_breakers.daily_loss_pct > 0 ? 'neg' : ''}`}>
                      {(data.circuit_breakers.daily_loss_pct * 100).toFixed(2)}% / {(data.circuit_breakers.daily_loss_limit_pct * 100).toFixed(0)}%
                    </td>
                  </tr>
                  {data.circuit_breakers.daily_loss_ack_at && (
                    <tr>
                      <td style={{ padding: '8px 0', border: 'none' }}>CB last acknowledged</td>
                      <td style={{ padding: '8px 0', border: 'none', textAlign: 'right' }} className="mono dim">{new Date(data.circuit_breakers.daily_loss_ack_at).toLocaleString()}</td>
                    </tr>
                  )}
                  <tr>
                    <td style={{ padding: '8px 0', border: 'none' }}>LLM budget used</td>
                    <td style={{ padding: '8px 0', border: 'none', textAlign: 'right' }} className="mono">
                      ${data.circuit_breakers.llm_budget_used_usd.toFixed(4)} / ${data.circuit_breakers.llm_budget_cap_usd.toFixed(2)}
                    </td>
                  </tr>
                  {data.circuit_breakers.reason && (
                    <tr>
                      <td colSpan={2} style={{ padding: '8px 0', border: 'none' }}>
                        <span className="neg" style={{ fontSize: 11.5 }}>{data.circuit_breakers.reason}</span>
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </Panel>

            <Panel title="Websocket" action={<Badge kind={data.websocket.connected ? 'pos' : 'neg'} dot>{data.websocket.connected ? 'connected' : data.websocket.status}</Badge>}>
              <table className="tbl" style={{ fontSize: 12.5 }}>
                <tbody>
                  <tr>
                    <td style={{ padding: '5px 0', border: 'none' }}>Feed</td>
                    <td style={{ padding: '5px 0', border: 'none', textAlign: 'right' }}>
                      <span style={{ display: 'flex', gap: 4, justifyContent: 'flex-end' }}>
                        <Badge kind={statusKind(data.websocket.status)} dot>{data.websocket.status}</Badge>
                        {data.websocket.connected !== null && (
                          <Badge kind={data.websocket.connected ? 'pos' : 'neg'} dot>{data.websocket.connected ? 'connected' : 'disconnected'}</Badge>
                        )}
                      </span>
                    </td>
                  </tr>
                  <tr>
                    <td style={{ padding: '5px 0', border: 'none' }}>Subscribed markets</td>
                    <td style={{ padding: '5px 0', border: 'none', textAlign: 'right' }} className="mono">{data.websocket.subscribed_markets ?? '—'}</td>
                  </tr>
                  <tr>
                    <td style={{ padding: '5px 0', border: 'none' }}>Last message</td>
                    <td style={{ padding: '5px 0', border: 'none', textAlign: 'right', fontSize: 11 }} className="mono dim">
                      {data.websocket.last_message_at ? new Date(data.websocket.last_message_at).toLocaleString() : '—'}
                    </td>
                  </tr>
                  <tr>
                    <td style={{ padding: '5px 0', border: 'none' }}>Last reconcile</td>
                    <td style={{ padding: '5px 0', border: 'none', textAlign: 'right', fontSize: 11 }} className="mono dim">
                      {data.websocket.last_reconcile_at ? new Date(data.websocket.last_reconcile_at).toLocaleString() : '—'}
                    </td>
                  </tr>
                </tbody>
              </table>
            </Panel>

            {(() => {
              const ex = data.exchange
              const cl = data.changelog
              const overallOk = ex.exchange_active === true && ex.trading_active === true
              const unavailable = ex.exchange_active === null
              const exchangeBadge = unavailable
                ? <Badge kind="muted">unavailable</Badge>
                : <Badge kind={overallOk ? 'pos' : 'neg'} dot>{overallOk ? 'ok' : 'degraded'}</Badge>

              const clBadge = cl == null ? null
                : cl.has_unreviewed_breaking_change
                  ? <Badge kind="neg">⚠ {cl.unreviewed_count} breaking</Badge>
                  : cl.unreviewed_count > 0
                    ? <Badge kind="warn">{cl.unreviewed_count} unreviewed</Badge>
                    : <Badge kind="pos">up to date</Badge>

              const R = ({ label, children }: { label: string; children: React.ReactNode }) => (
                <tr>
                  <td style={{ padding: '5px 0', border: 'none', fontSize: 12.5 }}>{label}</td>
                  <td style={{ padding: '5px 0', border: 'none', textAlign: 'right' }}>{children}</td>
                </tr>
              )

              return (
                <Panel
                  title="Kalshi exchange"
                  action={<a href="https://kalshistatus.com/" target="_blank" rel="noreferrer" className="dim" style={{ fontSize: 11 }}>kalshistatus.com ↗</a>}
                >
                  <table className="tbl" style={{ fontSize: 12.5, marginTop: 2 }}>
                    <tbody>
                      <R label="Status">{exchangeBadge}</R>
                      <R label="Exchange / Trading">
                        <span style={{ display: 'flex', gap: 4, justifyContent: 'flex-end' }}>
                          {ex.exchange_active === null
                            ? <span className="muted">—</span>
                            : <><Badge kind={ex.exchange_active ? 'pos' : 'neg'} dot>exchange</Badge><Badge kind={ex.trading_active ? 'pos' : 'neg'} dot>trading</Badge></>}
                        </span>
                      </R>
                      <R label="Checked">
                        <span className="mono dim" style={{ fontSize: 11 }}>
                          {ex.fetched_at ? new Date(ex.fetched_at).toLocaleTimeString() : '—'}
                        </span>
                      </R>
                      {cl != null && (
                        <R label="API changelog">
                          <span style={{ display: 'flex', gap: 6, alignItems: 'center', justifyContent: 'flex-end' }}>
                            {clBadge}
                            {cl.unreviewed_count > 0
                              ? <a href="https://docs.kalshi.com/changelog" target="_blank" rel="noreferrer" style={{ fontSize: 11 }}>view ↗</a>
                              : <span className="mono dim" style={{ fontSize: 11 }}>{cl.last_reviewed_at ?? '—'}</span>}
                          </span>
                        </R>
                      )}
                    </tbody>
                  </table>
                </Panel>
              )
            })()}
          </div>

          <Panel title="Service heartbeats" flush action={<span className="dim" style={{ fontSize: 11 }}>one row per service · from service_heartbeats</span>}>
            <table className="tbl">
              <thead>
                <tr>
                  <th>Service</th>
                  <th>Last success</th>
                  <th className="r">Age</th>
                  <th style={{ width: 200 }}>Staleness</th>
                  <th>Last error</th>
                  <th className="c">Status</th>
                </tr>
              </thead>
              <tbody>
                {data.services.map((s) => {
                  const ageSec = s.age_seconds ?? 0
                  const ok = s.status === 'ok'
                  // Approximate scheduled interval as half the stale threshold
                  // (matches build_freshness_specs which sets stale_after = interval * 2)
                  const intervalSec = s.stale_after_seconds / 2
                  const totalRange = s.stale_after_seconds * 2
                  const fillPct = Math.min(100, (ageSec / totalRange) * 100)
                  const intervalBp = (intervalSec / totalRange) * 100   // 25%
                  const staleBp = (s.stale_after_seconds / totalRange) * 100  // 50%
                  // Blend over ±20% of each boundary point for soft transitions
                  const blend = intervalBp * 0.20  // ~5 percentage points
                  const gradientBg = `linear-gradient(to right, var(--pos) 0%, var(--pos) ${intervalBp - blend}%, var(--warn) ${intervalBp + blend}%, var(--warn) ${staleBp - blend * 1.2}%, var(--neg) ${staleBp + blend * 1.2}%, var(--neg) 100%)`
                  const displayPct = (ageSec / s.stale_after_seconds) * 100
                  return (
                    <React.Fragment key={s.service_name}>
                      <tr>
                        <td>
                          <div className="mono" style={{ fontWeight: 500, fontSize: 12 }}>{s.service_name}</div>
                          <div style={{ fontSize: 10.5, color: 'var(--fg-3)', marginTop: 2 }}>stale after {fmtUptime(s.stale_after_seconds)}</div>
                        </td>
                        <td className="dim mono" style={{ fontSize: 11.5 }}>
                          {s.last_success_at ? new Date(s.last_success_at).toLocaleString() : <span className="muted">—</span>}
                        </td>
                        <td className="r mono">{fmtAgeSecs(s.age_seconds)}</td>
                        <td>
                          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                            <div style={{ position: 'relative', flex: 1, height: 4, borderRadius: 2, overflow: 'hidden' }}>
                              <div style={{ position: 'absolute', inset: 0, background: gradientBg }} />
                              <div style={{ position: 'absolute', top: 0, right: 0, bottom: 0, width: `${100 - fillPct}%`, background: 'var(--bg-3)', transition: 'width 0.3s' }} />
                            </div>
                            <span className="mono" style={{ fontSize: 10.5, color: 'var(--fg-2)', minWidth: 36, textAlign: 'right' }}>{displayPct.toFixed(0)}%</span>
                          </div>
                        </td>
                        <td>
                          {(() => {
                            if (!s.last_error_message || !s.last_error_at) return <span className="muted">—</span>
                            const errorAge = (Date.now() - new Date(s.last_error_at).getTime()) / 1000
                            if (errorAge > 86400) return <span className="muted">—</span>
                            return (
                              <div>
                                <div className="mono" style={{ fontSize: 11.5, color: errorTextColor(errorAge) }}>{s.last_error_message}</div>
                                <div className="dim mono" style={{ fontSize: 10.5, marginTop: 2 }}>{new Date(s.last_error_at).toLocaleString()}</div>
                              </div>
                            )
                          })()}
                        </td>
                        <td className="c">
                          <Badge kind={ok ? 'pos' : 'neg'} dot>{ok ? 'ok' : s.status}</Badge>
                        </td>
                      </tr>
                    </React.Fragment>
                  )
                })}
              </tbody>
            </table>
          </Panel>
        </>
      )}
    </div>
  )
}
