import { useQuery } from '@tanstack/react-query'
import { getSystemHealth } from '../api/health'
import LoadingSpinner from '../components/LoadingSpinner'
import ErrorBanner from '../components/ErrorBanner'
import StatusBadge from '../components/StatusBadge'

function Card({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="bg-white rounded shadow p-4">
      <div className="text-xs text-gray-500 uppercase tracking-wide mb-2">{label}</div>
      {children}
    </div>
  )
}

function formatUptime(secs: number) {
  const h = Math.floor(secs / 3600)
  const m = Math.floor((secs % 3600) / 60)
  const s = secs % 60
  if (h > 0) return `${h}h ${m}m`
  if (m > 0) return `${m}m ${s}s`
  return `${s}s`
}

function formatAge(secs: number | null) {
  if (secs === null) return '—'
  return formatUptime(secs)
}

export default function SystemHealth() {
  const { data, isLoading, error } = useQuery({
    queryKey: ['systemHealth'],
    queryFn: getSystemHealth,
    refetchInterval: 15_000,
  })

  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <h1 className="text-xl font-bold text-gray-900">System Health</h1>
        <span className="text-xs text-gray-400">refreshes every 15s</span>
      </div>
      {isLoading && <LoadingSpinner />}
      {error && <ErrorBanner message={String(error)} />}
      {data && (
        <>
          {data.circuit_breakers.trading_halted && (
            <div className="mb-4 px-4 py-3 bg-red-50 border border-red-300 rounded text-red-800 text-sm font-medium">
              Circuit breaker active — trading halted. Reason: {data.circuit_breakers.reason ?? 'unknown'}
            </div>
          )}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
            <Card label="Run state">
              <StatusBadge status={data.run_state} />
            </Card>
            <Card label="Mode">
              <StatusBadge status={data.mode} />
            </Card>
            <Card label="Database">
              <StatusBadge status={data.db_ok ? 'connected' : 'error'} />
            </Card>
            <Card label="Uptime">
              <span className="text-lg font-bold text-gray-900">{formatUptime(data.uptime_seconds)}</span>
            </Card>
          </div>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-4 mb-4">
            <Card label="Open positions">
              <span className="text-2xl font-bold text-gray-900">{data.open_positions}</span>
            </Card>
            <Card label="Pending orders">
              <div>
                <div className="text-2xl font-bold text-gray-900">{data.pending_orders}</div>
                <div className="text-xs text-gray-500 mt-1">
                  Oldest age: {formatAge(data.oldest_pending_order_age_seconds)}
                </div>
              </div>
            </Card>
            <Card label="API errors (last hour)">
              <div className="space-y-1">
                <div className={`text-lg font-bold ${data.api_errors.kalshi_errors_last_hour > 0 ? 'text-red-700' : 'text-gray-900'}`}>
                  Kalshi: {data.api_errors.kalshi_errors_last_hour}
                </div>
                <div className={`text-lg font-bold ${data.api_errors.llm_errors_last_hour > 0 ? 'text-red-700' : 'text-gray-900'}`}>
                  LLM: {data.api_errors.llm_errors_last_hour}
                </div>
              </div>
            </Card>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
            <Card label="Circuit breaker">
              <div className="space-y-1 text-sm">
                <div className="flex items-center justify-between">
                  <span className="text-gray-600">Trading halted</span>
                  <StatusBadge status={data.circuit_breakers.trading_halted ? 'halted' : 'ok'} />
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-gray-600">
                    Daily loss{' '}
                    <span className="text-xs text-gray-400">
                      (since {new Date(data.circuit_breakers.daily_loss_window_start).toLocaleTimeString()})
                    </span>
                  </span>
                  <span className={data.circuit_breakers.daily_loss_pct > 0 ? 'text-red-700 font-semibold' : 'text-gray-700'}>
                    {(data.circuit_breakers.daily_loss_pct * 100).toFixed(2)}% / {(data.circuit_breakers.daily_loss_limit_pct * 100).toFixed(0)}%
                  </span>
                </div>
                {data.circuit_breakers.daily_loss_ack_at && (
                  <div className="flex items-center justify-between text-xs text-gray-500">
                    <span>CB last acknowledged</span>
                    <span>{new Date(data.circuit_breakers.daily_loss_ack_at).toLocaleString()}</span>
                  </div>
                )}
                <div className="flex items-center justify-between">
                  <span className="text-gray-600">LLM budget used</span>
                  <span className="text-gray-700">
                    ${data.circuit_breakers.llm_budget_used_usd.toFixed(4)} / ${data.circuit_breakers.llm_budget_cap_usd.toFixed(2)}
                  </span>
                </div>
                {data.circuit_breakers.reason && (
                  <div className="mt-1 text-xs text-red-600 bg-red-50 rounded px-2 py-1">
                    {data.circuit_breakers.reason}
                  </div>
                )}
              </div>
            </Card>
            <Card label="WebSocket">
              <div className="space-y-1 text-sm">
                <div className="flex items-center justify-between">
                  <span className="text-gray-600">Feed status</span>
                  <StatusBadge status={data.websocket.status} />
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-gray-600">Connected</span>
                  <StatusBadge status={
                    data.websocket.connected === null ? 'n/a'
                    : data.websocket.connected ? 'connected' : 'error'
                  } />
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-gray-600">Subscribed markets</span>
                  <span className="text-gray-700">{data.websocket.subscribed_markets ?? '—'}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-gray-600">Last message</span>
                  <span className="text-gray-700 text-xs">
                    {data.websocket.last_message_at
                      ? new Date(data.websocket.last_message_at).toLocaleString()
                      : '—'}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-gray-600">Last reconcile</span>
                  <span className="text-gray-700 text-xs">
                    {data.websocket.last_reconcile_at
                      ? new Date(data.websocket.last_reconcile_at).toLocaleString()
                      : '—'}
                  </span>
                </div>
              </div>
            </Card>
          </div>
          <Card label="Service Freshness">
            <div className="space-y-2">
              {data.services.map((service) => (
                <div key={service.service_name} className="border border-gray-200 rounded px-3 py-2">
                  <div className="flex items-center justify-between gap-4">
                    <div>
                      <div className="font-medium text-gray-900">{service.label}</div>
                      <div className="text-xs text-gray-500">
                        stale after {formatUptime(service.stale_after_seconds)}
                      </div>
                    </div>
                    <StatusBadge status={service.status} />
                  </div>
                  <div className="mt-2 grid grid-cols-1 md:grid-cols-3 gap-2 text-xs text-gray-600">
                    <div>
                      Last success: {service.last_success_at ? new Date(service.last_success_at).toLocaleString() : '—'}
                    </div>
                    <div>
                      Age: {formatAge(service.age_seconds)}
                    </div>
                    <div>
                      Last error: {service.last_error_at ? new Date(service.last_error_at).toLocaleString() : '—'}
                    </div>
                  </div>
                  {service.last_error_message && (
                    <div className="mt-2 text-xs text-red-700 bg-red-50 rounded px-2 py-1">
                      {service.last_error_message}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </Card>
        </>
      )}
    </div>
  )
}
