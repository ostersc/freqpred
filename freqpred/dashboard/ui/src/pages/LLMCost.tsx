import { useState, useMemo } from 'react'
import { useQuery } from '@tanstack/react-query'
import { useSearchParams } from 'react-router-dom'
import { getLLMCost, getLLMQueries, getLLMQuery } from '../api/llm'
import { Badge, Panel, Stat, Donut, LoadingSpinner, ErrorBanner, Icon } from '../components/ui'
import type { LLMQueryOut } from '../api/types'

const PAGE = 50

const DONUT_COLORS = [
  'oklch(0.68 0.16 265)',
  'oklch(0.82 0.14 80)',
  'oklch(0.72 0.17 25)',
  'oklch(0.80 0.15 160)',
  'oklch(0.78 0.10 230)',
  'oklch(0.75 0.14 310)',
]

function queryTypeKind(type: string): 'accent' | 'warn' | 'info' | 'muted' {
  if (type === 'catalyst_generation') return 'accent'
  if (type === 'signal_assessment') return 'warn'
  if (type === 'market_analysis') return 'info'
  return 'muted'
}

function QueryDetailModal({ id, onClose }: { id: number; onClose: () => void }) {
  const { data, isLoading } = useQuery({
    queryKey: ['llmQuery', id],
    queryFn: () => getLLMQuery(id),
  })

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-head">
          <span>LLM Query #{id}</span>
          <button className="btn ghost sm" onClick={onClose}><Icon name="x" /></button>
        </div>
        <div style={{ padding: 18 }}>
          {isLoading && <LoadingSpinner />}
          {data && (
            <>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 10, marginBottom: 16, fontSize: 12 }}>
                <div><span className="dim">Model:</span> <span className="mono">{data.model_used}</span></div>
                <div><span className="dim">Type:</span> <span>{data.query_type}</span></div>
                <div><span className="dim">Cost:</span> <span className="mono">${data.cost_usd.toFixed(5)}</span></div>
                <div><span className="dim">Tokens:</span> <span className="mono">{data.tokens_total.toLocaleString()}</span></div>
                <div><span className="dim">Latency:</span> <span className="mono">{data.latency_ms}ms</span></div>
                <div><span className="dim">Success:</span> <Badge kind={data.success ? 'pos' : 'neg'}>{data.success ? 'yes' : 'no'}</Badge></div>
              </div>
              {data.error_message && (
                <div className="error-banner" style={{ marginBottom: 12 }}>{data.error_message}</div>
              )}
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 14 }}>
                <div>
                  <div style={{ fontSize: 10.5, textTransform: 'uppercase', letterSpacing: '0.08em', color: 'var(--fg-2)', marginBottom: 6 }}>Prompt</div>
                  <pre className="mono" style={{ fontSize: 11, background: 'var(--bg-0)', padding: 12, borderRadius: 6, border: '1px solid var(--line-soft)', lineHeight: 1.55, maxHeight: 240, overflow: 'auto', whiteSpace: 'pre-wrap', margin: 0 }}>{data.prompt}</pre>
                </div>
                <div>
                  <div style={{ fontSize: 10.5, textTransform: 'uppercase', letterSpacing: '0.08em', color: 'var(--fg-2)', marginBottom: 6 }}>Response</div>
                  <pre className="mono" style={{ fontSize: 11, background: 'var(--bg-0)', padding: 12, borderRadius: 6, border: '1px solid var(--line-soft)', lineHeight: 1.55, maxHeight: 240, overflow: 'auto', whiteSpace: 'pre-wrap', margin: 0 }}>{data.response}</pre>
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  )
}

export default function LLMCost() {
  const [offset, setOffset] = useState(0)
  const [searchParams, setSearchParams] = useSearchParams()
  const selectedIdParam = searchParams.get('queryId')
  const activeQueryId = selectedIdParam !== null && Number.isFinite(Number(selectedIdParam)) ? Number(selectedIdParam) : null

  const { data: cost, isLoading: costLoading, error: costError } = useQuery({
    queryKey: ['llmCost'],
    queryFn: getLLMCost,
  })

  const { data: queries, isLoading: queriesLoading, error: queriesError } = useQuery({
    queryKey: ['llmQueries', offset],
    queryFn: () => getLLMQueries({ limit: PAGE, offset }),
  })

  const donutData = useMemo(() => {
    if (!cost) return []
    const total = Object.values(cost.by_query_type).reduce((a, b) => a + b, 0)
    if (total === 0) return []
    return Object.entries(cost.by_query_type).map(([label, value], i) => ({
      label,
      pct: (value / total) * 100,
      color: DONUT_COLORS[i % DONUT_COLORS.length],
    }))
  }, [cost])

  const isLoading = costLoading || queriesLoading
  const error = costError || queriesError

  return (
    <div className="page">
      {activeQueryId !== null && (
        <QueryDetailModal
          id={activeQueryId}
          onClose={() => {
            const next = new URLSearchParams(searchParams)
            next.delete('queryId')
            setSearchParams(next)
          }}
        />
      )}

      <div className="page-head">
        <div>
          <h1 className="page-title">LLM Cost & Audit</h1>
          <div className="page-subtitle">Budget tracking, query volume, and per-call audit trail</div>
        </div>
      </div>

      {isLoading && !cost && <LoadingSpinner />}
      {error && <ErrorBanner message={String(error)} />}

      {cost && (
        <div className="grid grid-3" style={{ marginBottom: 12 }}>
          <div className="stat">
            <div className="stat-label">Today</div>
            <div style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between' }}>
              <div className="stat-value">${cost.today_usd.toFixed(4)}</div>
              <div style={{ fontSize: 11, color: 'var(--fg-2)' }}>{cost.pct_used.toFixed(1)}% used</div>
            </div>
            <div style={{ fontSize: 11, color: 'var(--fg-2)', marginTop: 6, marginBottom: 8 }}>Cap: ${cost.daily_cap_usd.toFixed(2)}</div>
            <div className="progress">
              <span style={{ width: `${Math.min(100, cost.pct_used)}%`, background: cost.pct_used >= 90 ? 'var(--neg)' : cost.pct_used >= 70 ? 'var(--warn)' : 'var(--accent)' }} />
            </div>
          </div>
          <Stat label="This week" value={`$${cost.weekly_usd.toFixed(4)}`} sub={`${queries?.total ?? 0} queries total`} />
          {donutData.length > 0 && (
            <div className="stat">
              <div className="stat-label">By query type (today)</div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 14, marginTop: 4 }}>
                <Donut data={donutData} size={76} />
                <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 3 }}>
                  {donutData.map((b) => (
                    <div key={b.label} style={{ display: 'flex', alignItems: 'center', gap: 7, fontSize: 11 }}>
                      <span style={{ width: 8, height: 8, borderRadius: 2, background: b.color, flexShrink: 0 }} />
                      <span style={{ flex: 1, color: 'var(--fg-1)' }}>{b.label}</span>
                      <span className="mono dim">{b.pct.toFixed(0)}%</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {queries && (
        <Panel title="Recent queries — click for full prompt & response" flush>
          <table className="tbl">
            <thead>
              <tr>
                <th>Time</th>
                <th>Type</th>
                <th>Model</th>
                <th className="r">Tokens</th>
                <th className="r">Cost</th>
                <th className="r">Latency</th>
                <th className="c">OK</th>
              </tr>
            </thead>
            <tbody>
              {queries.items.map((q: LLMQueryOut) => (
                <tr
                  key={q.id}
                  style={{ cursor: 'pointer' }}
                  onClick={() => {
                    const next = new URLSearchParams(searchParams)
                    next.set('queryId', String(q.id))
                    setSearchParams(next)
                  }}
                >
                  <td className="dim" style={{ fontSize: 11.5, whiteSpace: 'nowrap' }}>{new Date(q.timestamp).toLocaleString()}</td>
                  <td><Badge kind={queryTypeKind(q.query_type)}>{q.query_type}</Badge></td>
                  <td className="mono" style={{ fontSize: 11.5, color: 'var(--fg-1)' }}>{q.model_used}</td>
                  <td className="r">{q.tokens_total.toLocaleString()}</td>
                  <td className="r">${q.cost_usd.toFixed(5)}</td>
                  <td className="r">
                    <span className={q.latency_ms > 5000 ? 'warn' : ''}>{q.latency_ms}ms</span>
                  </td>
                  <td className="c">
                    <span className={q.success ? 'pos' : 'neg'} style={{ display: 'inline-flex', width: 16, height: 16, alignItems: 'center', justifyContent: 'center' }}>
                      <Icon name={q.success ? 'check' : 'x'} size={12} />
                    </span>
                  </td>
                </tr>
              ))}
              {queries.items.length === 0 && (
                <tr>
                  <td colSpan={7} style={{ padding: '24px', textAlign: 'center', color: 'var(--fg-3)' }}>No LLM queries recorded</td>
                </tr>
              )}
            </tbody>
          </table>
        </Panel>
      )}
      {queries && queries.total > PAGE && (
        <div className="pagination">
          <button className="btn sm" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - PAGE))}>Previous</button>
          <span>{offset + 1}–{Math.min(offset + PAGE, queries.total)} of {queries.total}</span>
          <button className="btn sm" disabled={offset + PAGE >= queries.total} onClick={() => setOffset(offset + PAGE)}>Next</button>
        </div>
      )}
    </div>
  )
}
