import { useState, useMemo } from 'react'
import { useQuery } from '@tanstack/react-query'
import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer } from 'recharts'
import { getLLMCost, getLLMQueries, getLLMQuery } from '../api/llm'
import LoadingSpinner from '../components/LoadingSpinner'
import ErrorBanner from '../components/ErrorBanner'
import type { LLMQueryOut } from '../api/types'

const COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899']

function QueryModal({ id, onClose }: { id: number; onClose: () => void }) {
  const { data, isLoading } = useQuery({
    queryKey: ['llmQuery', id],
    queryFn: () => getLLMQuery(id),
  })

  return (
    <div
      className="fixed inset-0 bg-black/40 flex items-center justify-center z-50 p-4"
      onClick={onClose}
    >
      <div
        className="bg-white rounded shadow-lg max-w-3xl w-full max-h-[80vh] overflow-y-auto p-6 text-sm"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between mb-4">
          <h2 className="font-semibold text-gray-900">LLM Query #{id}</h2>
          <button onClick={onClose} className="text-gray-500 hover:text-gray-700 text-lg leading-none">✕</button>
        </div>
        {isLoading && <LoadingSpinner />}
        {data && (
          <div className="space-y-4">
            <div className="grid grid-cols-3 gap-3 text-xs text-gray-600">
              <div><span className="font-medium">Model:</span> {data.model_used}</div>
              <div><span className="font-medium">Type:</span> {data.query_type}</div>
              <div><span className="font-medium">Cost:</span> ${data.cost_usd.toFixed(5)}</div>
              <div><span className="font-medium">Tokens:</span> {data.tokens_total}</div>
              <div><span className="font-medium">Latency:</span> {data.latency_ms}ms</div>
              <div><span className="font-medium">Success:</span> {data.success ? 'Yes' : 'No'}</div>
            </div>
            {data.error_message && (
              <div className="bg-red-50 border border-red-200 rounded p-2 text-red-700 text-xs">
                {data.error_message}
              </div>
            )}
            <div>
              <div className="font-medium text-gray-700 mb-1">Prompt</div>
              <pre className="bg-gray-50 rounded p-3 text-xs overflow-x-auto whitespace-pre-wrap">{data.prompt}</pre>
            </div>
            <div>
              <div className="font-medium text-gray-700 mb-1">Response</div>
              <pre className="bg-gray-50 rounded p-3 text-xs overflow-x-auto whitespace-pre-wrap">{data.response}</pre>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default function LLMCost() {
  const [selectedId, setSelectedId] = useState<number | null>(null)
  const [offset, setOffset] = useState(0)
  const PAGE = 50

  const { data: cost, isLoading: costLoading, error: costError } = useQuery({
    queryKey: ['llmCost'],
    queryFn: getLLMCost,
  })

  const { data: queries, isLoading: queriesLoading, error: queriesError } = useQuery({
    queryKey: ['llmQueries', offset],
    queryFn: () => getLLMQueries({ limit: PAGE, offset }),
  })

  const pieData = useMemo(() => {
    if (!cost) return []
    return Object.entries(cost.by_query_type).map(([name, value]) => ({ name, value }))
  }, [cost])

  const isLoading = costLoading || queriesLoading
  const error = costError || queriesError

  return (
    <div>
      {selectedId !== null && (
        <QueryModal id={selectedId} onClose={() => setSelectedId(null)} />
      )}
      <h1 className="text-xl font-bold text-gray-900 mb-4">LLM Cost & Audit</h1>
      {isLoading && !cost && <LoadingSpinner />}
      {error && <ErrorBanner message={String(error)} />}
      {cost && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
          <div className="bg-white rounded shadow p-4">
            <div className="text-xs text-gray-500 uppercase tracking-wide mb-1">Today</div>
            <div className="text-2xl font-bold text-gray-900">${cost.today_usd.toFixed(4)}</div>
            <div className="mt-2">
              <div className="flex justify-between text-xs text-gray-500 mb-1">
                <span>Cap: ${cost.daily_cap_usd.toFixed(2)}</span>
                <span>{cost.pct_used.toFixed(1)}% used</span>
              </div>
              <div className="bg-gray-100 rounded-full h-2">
                <div
                  className={`h-2 rounded-full ${cost.pct_used >= 90 ? 'bg-red-500' : cost.pct_used >= 70 ? 'bg-yellow-500' : 'bg-blue-500'}`}
                  style={{ width: `${Math.min(100, cost.pct_used)}%` }}
                />
              </div>
            </div>
          </div>
          <div className="bg-white rounded shadow p-4">
            <div className="text-xs text-gray-500 uppercase tracking-wide mb-1">This week</div>
            <div className="text-2xl font-bold text-gray-900">${cost.weekly_usd.toFixed(4)}</div>
          </div>
          {pieData.length > 0 && (
            <div className="bg-white rounded shadow p-4">
              <div className="text-xs text-gray-500 uppercase tracking-wide mb-2">By query type (today)</div>
              <div className="flex items-center gap-3">
                <ResponsiveContainer width={100} height={100}>
                  <PieChart>
                    <Pie data={pieData} dataKey="value" cx="50%" cy="50%" outerRadius={45} innerRadius={25}>
                      {pieData.map((_, i) => (
                        <Cell key={i} fill={COLORS[i % COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip formatter={(v: number) => `$${v.toFixed(5)}`} />
                  </PieChart>
                </ResponsiveContainer>
                <ul className="text-xs text-gray-600 space-y-1">
                  {pieData.map((d, i) => (
                    <li key={d.name} className="flex items-center gap-1">
                      <span className="inline-block w-2 h-2 rounded-full" style={{ background: COLORS[i % COLORS.length] }} />
                      {d.name}
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          )}
        </div>
      )}
      {queries && (
        <>
          <h2 className="text-sm font-semibold text-gray-700 mb-2">Recent queries — click for full prompt &amp; response</h2>
          <div className="bg-white rounded shadow overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead className="bg-gray-100 text-xs text-gray-600 uppercase tracking-wide">
                <tr>
                  <th className="px-3 py-2">Time</th>
                  <th className="px-3 py-2">Type</th>
                  <th className="px-3 py-2">Model</th>
                  <th className="px-3 py-2 text-right">Tokens</th>
                  <th className="px-3 py-2 text-right">Cost</th>
                  <th className="px-3 py-2 text-right">Latency</th>
                  <th className="px-3 py-2 text-center">OK</th>
                </tr>
              </thead>
              <tbody>
                {queries.items.map((q: LLMQueryOut) => (
                  <tr
                    key={q.id}
                    className="border-t hover:bg-blue-50 cursor-pointer transition-colors"
                    onClick={() => setSelectedId(q.id)}
                  >
                    <td className="px-3 py-2 text-xs text-gray-500">{new Date(q.timestamp).toLocaleString()}</td>
                    <td className="px-3 py-2">{q.query_type}</td>
                    <td className="px-3 py-2 text-xs text-gray-500">{q.model_used}</td>
                    <td className="px-3 py-2 text-right">{q.tokens_total.toLocaleString()}</td>
                    <td className="px-3 py-2 text-right">${q.cost_usd.toFixed(5)}</td>
                    <td className="px-3 py-2 text-right">{q.latency_ms}ms</td>
                    <td className="px-3 py-2 text-center">
                      {q.success
                        ? <span className="text-green-600">✓</span>
                        : <span className="text-red-600">✗</span>}
                    </td>
                  </tr>
                ))}
                {queries.items.length === 0 && (
                  <tr>
                    <td colSpan={7} className="px-3 py-6 text-center text-gray-400">No LLM queries recorded</td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
          <div className="flex items-center gap-3 mt-3 text-sm">
            <button
              className="px-3 py-1 rounded border disabled:opacity-40"
              disabled={offset === 0}
              onClick={() => setOffset(Math.max(0, offset - PAGE))}
            >
              Previous
            </button>
            <span className="text-gray-500">{offset + 1}–{Math.min(offset + PAGE, queries.total)} of {queries.total}</span>
            <button
              className="px-3 py-1 rounded border disabled:opacity-40"
              disabled={offset + PAGE >= queries.total}
              onClick={() => setOffset(offset + PAGE)}
            >
              Next
            </button>
          </div>
        </>
      )}
    </div>
  )
}
