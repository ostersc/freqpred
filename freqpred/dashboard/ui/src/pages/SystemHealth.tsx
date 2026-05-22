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

          <div className="grid grid-4" style={{ marginBottom: 12 }}>
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
          </div>

          <div className="grid grid-3" style={{ marginBottom: 12 }}>
            <Stat label="Open positions" value={String(data.open_positions)} />
            <Stat label="Pending orders" value={String(data.pending_orders)} sub={`Oldest age: ${fmtAgeSecs(data.oldest_pending_order_age_seconds)}`} />
            <div className="stat">
              <div className="stat-label">API errors (last hour)</div>
              <div style={{ display: 'flex', gap: 24, marginTop: 6 }}>
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

          <div className="grid grid-3" style={{ marginBottom: 12 }}>
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
                    <td style={{ padding: '8px 0', border: 'none' }}>Feed status</td>
                    <td style={{ padding: '8px 0', border: 'none', textAlign: 'right' }}>
                      <Badge kind={statusKind(data.websocket.status)} dot>{data.websocket.status}</Badge>
                    </td>
                  </tr>
                  <tr>
                    <td style={{ padding: '8px 0', border: 'none' }}>Connected</td>
                    <td style={{ padding: '8px 0', border: 'none', textAlign: 'right' }}>
                      {data.websocket.connected === null
                        ? <span className="muted">n/a</span>
                        : <Badge kind={data.websocket.connected ? 'pos' : 'neg'} dot>{data.websocket.connected ? 'connected' : 'disconnected'}</Badge>}
                    </td>
                  </tr>
                  <tr>
                    <td style={{ padding: '8px 0', border: 'none' }}>Subscribed markets</td>
                    <td style={{ padding: '8px 0', border: 'none', textAlign: 'right' }} className="mono">{data.websocket.subscribed_markets ?? '—'}</td>
                  </tr>
                  <tr>
                    <td style={{ padding: '8px 0', border: 'none' }}>Last message</td>
                    <td style={{ padding: '8px 0', border: 'none', textAlign: 'right', fontSize: 11 }} className="mono dim">
                      {data.websocket.last_message_at ? new Date(data.websocket.last_message_at).toLocaleString() : '—'}
                    </td>
                  </tr>
                  <tr>
                    <td style={{ padding: '8px 0', border: 'none' }}>Last reconcile</td>
                    <td style={{ padding: '8px 0', border: 'none', textAlign: 'right', fontSize: 11 }} className="mono dim">
                      {data.websocket.last_reconcile_at ? new Date(data.websocket.last_reconcile_at).toLocaleString() : '—'}
                    </td>
                  </tr>
                </tbody>
              </table>
            </Panel>

            {(() => {
              const ex = data.exchange
              const overallOk = ex.exchange_active === true && ex.trading_active === true
              const unavailable = ex.exchange_active === null
              const badge = unavailable
                ? <Badge kind="muted">unavailable</Badge>
                : <Badge kind={overallOk ? 'pos' : 'neg'} dot>{overallOk ? 'ok' : 'degraded'}</Badge>
              return (
                <Panel
                  title="Kalshi exchange"
                  action={<a href="https://kalshistatus.com/" target="_blank" rel="noreferrer" className="dim" style={{ fontSize: 11 }}>kalshistatus.com ↗</a>}
                >
                  <div style={{ marginBottom: 8 }}>{badge}</div>
                  <table className="tbl" style={{ fontSize: 12.5 }}>
                    <tbody>
                      <tr>
                        <td style={{ padding: '8px 0', border: 'none' }}>Exchange active</td>
                        <td style={{ padding: '8px 0', border: 'none', textAlign: 'right' }}>
                          {ex.exchange_active === null
                            ? <span className="muted">—</span>
                            : <Badge kind={ex.exchange_active ? 'pos' : 'neg'} dot>{String(ex.exchange_active)}</Badge>}
                        </td>
                      </tr>
                      <tr>
                        <td style={{ padding: '8px 0', border: 'none' }}>Trading active</td>
                        <td style={{ padding: '8px 0', border: 'none', textAlign: 'right' }}>
                          {ex.trading_active === null
                            ? <span className="muted">—</span>
                            : <Badge kind={ex.trading_active ? 'pos' : 'neg'} dot>{String(ex.trading_active)}</Badge>}
                        </td>
                      </tr>
                      <tr>
                        <td style={{ padding: '8px 0', border: 'none' }}>Checked at</td>
                        <td style={{ padding: '8px 0', border: 'none', textAlign: 'right', fontSize: 11 }} className="mono dim">
                          {ex.fetched_at ? new Date(ex.fetched_at).toLocaleString() : '—'}
                        </td>
                      </tr>
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
                  <th style={{ width: 200 }}>Freshness</th>
                  <th>Last error</th>
                  <th className="c">Status</th>
                </tr>
              </thead>
              <tbody>
                {data.services.map((s) => {
                  const ageSec = s.age_seconds ?? 0
                  const pct = Math.min(100, (ageSec / s.stale_after_seconds) * 100)
                  const ok = s.status === 'ok'
                  const barColor = pct > 90 ? 'var(--neg)' : pct > 60 ? 'var(--warn)' : 'var(--pos)'
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
                            <div style={{ flex: 1, height: 4, background: 'var(--bg-3)', borderRadius: 2, overflow: 'hidden' }}>
                              <div style={{ width: `${pct}%`, height: '100%', background: barColor, transition: 'width 0.3s' }} />
                            </div>
                            <span className="mono" style={{ fontSize: 10.5, color: 'var(--fg-2)', minWidth: 32, textAlign: 'right' }}>{pct.toFixed(0)}%</span>
                          </div>
                        </td>
                        <td>
                          {(() => {
                            if (!s.last_error_message || !s.last_error_at) return <span className="muted">—</span>
                            const errorAge = (Date.now() - new Date(s.last_error_at).getTime()) / 1000
                            if (errorAge > 86400) return <span className="muted">—</span>
                            return (
                              <div>
                                <div className="neg mono" style={{ fontSize: 11.5 }}>{s.last_error_message}</div>
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
