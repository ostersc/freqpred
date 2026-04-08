import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import {
  ComposedChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ReferenceArea,
  ResponsiveContainer,
  Legend,
} from 'recharts'
import { getPositions, getPositionDetail } from '../api/positions'
import type { PositionOut, PositionDetailOut, SignalOut } from '../api/types'
import LoadingSpinner from '../components/LoadingSpinner'
import ErrorBanner from '../components/ErrorBanner'

type StatusFilter = 'open' | 'closed' | 'all'

function fmt(v: number | null, decimals = 2) {
  if (v === null) return '—'
  return v.toFixed(decimals)
}

function fmtPct(v: number | null) {
  if (v === null) return '—'
  const s = (v * 100).toFixed(1)
  return v >= 0 ? `+${s}%` : `${s}%`
}

function pnlColor(v: number | null) {
  if (v === null) return 'text-gray-500'
  return v > 0 ? 'text-green-700 font-semibold' : v < 0 ? 'text-red-700 font-semibold' : 'text-gray-600'
}

function relTime(iso: string) {
  return new Date(iso).toLocaleString()
}

function fmtTs(ts: number) {
  const d = new Date(ts)
  return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' }) +
    ' ' + d.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' })
}

// ---- Signal price timeline chart ----------------------------------------

type ChartPoint = {
  ts: number
  our_prob: number
  market_mid: number
  isEntry: boolean
}

function PriceTimeline({
  signals,
  entrySignalId,
  entryPrice,
  currentMid,
  direction,
}: {
  signals: SignalOut[]
  entrySignalId: string
  entryPrice: number
  currentMid: number | null
  direction: string
}) {
  if (signals.length === 0) return null

  const data: ChartPoint[] = signals.map((s) => ({
    ts: new Date(s.created_at).getTime(),
    our_prob: s.estimated_probability,
    market_mid: s.market_mid_at_signal,
    isEntry: s.id === entrySignalId,
  }))

  // For NO trades: entry_price is the NO price paid; chart Y-axis is in YES terms
  const chartEntryPrice = direction === 'NO' ? 1 - entryPrice : entryPrice

  // ReferenceArea fill between our_prob and market_mid for each segment.
  // Green = we have edge: YES trade → our_prob > market_mid; NO trade → market_mid > our_prob
  const fills = data.slice(0, -1).map((d, i) => {
    const next = data[i + 1]
    const hasEdge = direction === 'YES' ? d.our_prob > d.market_mid : d.market_mid > d.our_prob
    return {
      x1: d.ts, x2: next.ts,
      y1: Math.min(d.our_prob, d.market_mid),
      y2: Math.max(d.our_prob, d.market_mid),
      color: hasEdge ? '#16a34a' : '#dc2626',
    }
  })

  // Custom dot: larger blue circle on entry signal
  const renderDot = (props: { cx?: number; cy?: number; payload?: ChartPoint }) => {
    const { cx = 0, cy = 0, payload } = props
    if (!payload?.isEntry) return <circle key={`${cx}-${cy}`} />
    return <circle key={`${cx}-${cy}-entry`} cx={cx} cy={cy} r={5} fill="#3b82f6" stroke="#fff" strokeWidth={1.5} />
  }

  return (
    <div>
      <div className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2">
        Signal history — estimated probability vs. market mid
      </div>
      <ResponsiveContainer width="100%" height={220}>
        <ComposedChart data={data} margin={{ top: 8, right: 16, bottom: 5, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
          <XAxis
            dataKey="ts"
            type="number"
            scale="time"
            domain={['dataMin', 'dataMax']}
            tickFormatter={fmtTs}
            tick={{ fontSize: 10 }}
          />
          <YAxis
            domain={[0, 1]}
            tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
            tick={{ fontSize: 10 }}
            width={36}
          />
          <Tooltip
            content={({ payload }) => {
              if (!payload?.length) return null
              const d = payload[0]?.payload as ChartPoint
              const edge = direction === 'NO' ? d.market_mid - d.our_prob : d.our_prob - d.market_mid
              return (
                <div className="bg-white border rounded p-2 text-xs shadow space-y-0.5">
                  <div className="text-gray-400">{fmtTs(d.ts)}</div>
                  <div>Our prob: <span className="font-semibold text-blue-600">{(d.our_prob * 100).toFixed(1)}%</span></div>
                  <div>Market mid: <span className="font-semibold text-orange-500">{(d.market_mid * 100).toFixed(1)}%</span></div>
                  <div className={edge >= 0 ? 'text-green-700 font-semibold' : 'text-red-600 font-semibold'}>
                    Edge: {edge >= 0 ? '+' : ''}{(edge * 100).toFixed(1)}%
                  </div>
                  {d.isEntry && <div className="text-blue-600 font-semibold pt-0.5">← entry signal</div>}
                </div>
              )
            }}
          />
          <Legend
            verticalAlign="top"
            align="right"
            iconSize={10}
            wrapperStyle={{ fontSize: 11 }}
            payload={[
              { value: 'Our prob', type: 'line', color: '#3b82f6' },
              { value: 'Market mid', type: 'line', color: '#f97316' },
            ]}
          />

          {/* Colored fill between the two lines */}
          {fills.map((f, i) => (
            <ReferenceArea
              key={i}
              x1={f.x1} x2={f.x2}
              y1={f.y1} y2={f.y2}
              fill={f.color}
              fillOpacity={0.15}
              strokeOpacity={0}
            />
          ))}

          {/* Entry price dashed line — convert to YES-equivalent for NO trades */}
          <ReferenceLine
            y={chartEntryPrice}
            stroke="#3b82f6"
            strokeDasharray="4 2"
            strokeOpacity={0.5}
            label={{ value: `Entry ${(chartEntryPrice * 100).toFixed(0)}%`, position: 'insideTopRight', fontSize: 9, fill: '#3b82f6' }}
          />

          {/* Current mid dashed line (open only) */}
          {currentMid !== null && (
            <ReferenceLine
              y={currentMid}
              stroke={direction === 'YES' ? '#16a34a' : '#dc2626'}
              strokeDasharray="4 2"
              strokeOpacity={0.6}
              label={{ value: `Now ${(currentMid * 100).toFixed(0)}%`, position: 'insideBottomRight', fontSize: 9, fill: direction === 'YES' ? '#16a34a' : '#dc2626' }}
            />
          )}

          <Line
            type="linear"
            dataKey="our_prob"
            stroke="#3b82f6"
            strokeWidth={2}
            dot={renderDot}
            isAnimationActive={false}
            name="Our prob"
          />
          <Line
            type="linear"
            dataKey="market_mid"
            stroke="#f97316"
            strokeWidth={1.5}
            dot={false}
            isAnimationActive={false}
            name="Market mid"
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  )
}

// ---- Position detail panel -----------------------------------------------

function PositionDetail({ positionId }: { positionId: string }) {
  const [showOtherSignals, setShowOtherSignals] = useState(false)

  const { data, isLoading, error } = useQuery({
    queryKey: ['position-detail', positionId],
    queryFn: () => getPositionDetail(positionId),
    staleTime: 30_000,
  })

  if (isLoading) return <div className="p-4 text-sm text-gray-500">Loading…</div>
  if (error) return <div className="p-4 text-sm text-red-600">{String(error)}</div>
  if (!data) return null

  const d: PositionDetailOut = data
  const otherSignals = d.market_signals.filter((s) => s.id !== d.entry_signal.id)

  return (
    <div className="bg-gray-50 border-t px-4 py-4 space-y-5 text-sm">

      {/* Market question */}
      {d.market_question && (
        <div className="font-semibold text-gray-800 text-base leading-snug">{d.market_question}</div>
      )}

      {/* Stat cards */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <div className="bg-white rounded border p-3">
          <div className="text-xs text-gray-400 uppercase tracking-wide mb-1">Entry price</div>
          <div className="font-semibold">{(d.entry_price * 100).toFixed(1)}¢</div>
        </div>
        <div className="bg-white rounded border p-3">
          <div className="text-xs text-gray-400 uppercase tracking-wide mb-1">
            {d.status === 'open' ? 'Current mid' : 'Exit price'}
          </div>
          <div className="font-semibold">
            {d.status === 'open'
              ? d.current_mid !== null ? `${(d.current_mid * 100).toFixed(1)}¢` : '—'
              : d.exit_price !== null ? `${(d.exit_price * 100).toFixed(1)}¢` : '—'}
          </div>
        </div>
        <div className="bg-white rounded border p-3">
          <div className="text-xs text-gray-400 uppercase tracking-wide mb-1">Contracts</div>
          <div className="font-semibold">{d.contracts}</div>
        </div>
        <div className="bg-white rounded border p-3">
          <div className="text-xs text-gray-400 uppercase tracking-wide mb-1">
            {d.status === 'open' ? 'Unrealized P&L' : 'P&L'}
          </div>
          <div className={`font-semibold ${pnlColor(d.status === 'open' ? d.unrealized_pnl : d.pnl)}`}>
            {d.status === 'open'
              ? d.unrealized_pnl !== null
                ? `$${fmt(d.unrealized_pnl)} (${fmtPct(d.unrealized_pnl_pct)})`
                : '—'
              : d.pnl !== null
                ? `$${fmt(d.pnl)} (${fmtPct(d.pnl_pct)})`
                : '—'}
          </div>
        </div>
      </div>

      {/* Price timeline */}
      <PriceTimeline
        signals={d.market_signals}
        entrySignalId={d.entry_signal.id}
        entryPrice={d.entry_price}
        currentMid={d.status === 'open' ? d.current_mid : null}
        direction={d.direction}
      />

      {/* Entry signal */}
      <div>
        <div className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2">Entry signal</div>
        <div className="bg-white rounded border p-3 space-y-3">
          <div className="flex flex-wrap gap-4 text-xs text-gray-500">
            <span>Our prob: <span className="font-semibold text-gray-800">{(d.entry_signal.estimated_probability * 100).toFixed(1)}%</span></span>
            <span>Market mid: <span className="font-semibold text-gray-800">{(d.entry_signal.market_mid_at_signal * 100).toFixed(1)}%</span></span>
            <span>Edge: <span className={`font-semibold ${d.entry_signal.edge >= 0 ? 'text-green-700' : 'text-red-700'}`}>{fmtPct(d.entry_signal.edge)}</span></span>
            <span>Confidence: <span className="font-semibold text-gray-800">{(d.entry_signal.confidence * 100).toFixed(1)}%</span></span>
            <span className="text-gray-400">{relTime(d.entry_signal.created_at)}</span>
          </div>
          <div>
            <div className="font-medium text-gray-700 mb-1">Reasoning:</div>
            <p className="text-gray-600 whitespace-pre-wrap">{d.entry_signal.reasoning}</p>
          </div>
          {d.entry_signal.social_sentiment_summary && (
            <div>
              <div className="font-medium text-gray-700 mb-1">Social sentiment:</div>
              <p className="text-gray-600">{d.entry_signal.social_sentiment_summary}</p>
            </div>
          )}
          {d.entry_signal.document_links.length > 0 && (
            <div>
              <div className="font-medium text-gray-700 mb-1">Evidence documents:</div>
              <ul className="space-y-1">
                {d.entry_signal.document_links.map((doc) => (
                  <li key={doc.document_id} className="flex items-center gap-2">
                    <span className="text-xs text-gray-400">{doc.relevance_score.toFixed(3)}</span>
                    <a href={doc.source_url} target="_blank" rel="noreferrer" className="text-blue-600 hover:underline truncate max-w-xl">
                      {doc.title || doc.source_url}
                    </a>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      </div>

      {/* Other signals */}
      {otherSignals.length > 0 && (
        <div>
          <button
            className="text-xs font-semibold text-gray-500 uppercase tracking-wide hover:text-gray-700"
            onClick={() => setShowOtherSignals((v) => !v)}
          >
            {showOtherSignals ? '▲' : '▼'} Other signals for this market ({otherSignals.length})
          </button>
          {showOtherSignals && (
            <div className="mt-2 space-y-1">
              {otherSignals.map((s) => (
                <div key={s.id} className="bg-white rounded border px-3 py-2 flex flex-wrap gap-4 text-xs text-gray-500">
                  <span className="text-gray-400">{relTime(s.created_at)}</span>
                  <span>Our prob: <span className="font-semibold text-gray-800">{(s.estimated_probability * 100).toFixed(1)}%</span></span>
                  <span>Market mid: <span className="font-semibold text-gray-800">{(s.market_mid_at_signal * 100).toFixed(1)}%</span></span>
                  <span>Edge: <span className={`font-semibold ${s.edge >= 0 ? 'text-green-700' : 'text-red-700'}`}>{fmtPct(s.edge)}</span></span>
                  <span className={s.direction === 'YES' ? 'text-green-700 font-semibold' : 'text-red-700 font-semibold'}>{s.direction}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

// ---- Main page -----------------------------------------------------------

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

  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <h1 className="text-xl font-bold text-gray-900">Positions</h1>
        <div className="flex gap-1 text-sm">
          {(['open', 'closed', 'all'] as StatusFilter[]).map((s) => (
            <button
              key={s}
              onClick={() => setStatus(s)}
              className={`px-3 py-1 rounded border capitalize ${status === s ? 'bg-blue-600 text-white border-blue-600' : 'border-gray-300 text-gray-700 hover:bg-gray-50'}`}
            >
              {s}
            </button>
          ))}
        </div>
      </div>
      {isLoading && <LoadingSpinner />}
      {error && <ErrorBanner message={String(error)} />}
      {data && (
        <>
          <div className="text-sm text-gray-500 mb-2">{data.total} positions{status === 'open' ? ' — refreshes every 60s' : ''}</div>
          <div className="bg-white rounded shadow overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead className="bg-gray-100 text-xs text-gray-600 uppercase tracking-wide">
                <tr>
                  <th className="px-3 py-2">Market</th>
                  <th className="px-3 py-2 text-center">Dir</th>
                  <th className="px-3 py-2 text-center">Contracts</th>
                  <th className="px-3 py-2 text-center">Entry</th>
                  <th className="px-3 py-2 text-center">Exit</th>
                  <th className="px-3 py-2 text-center">P&L (unreal.)</th>
                  <th className="px-3 py-2 text-center">P&L %</th>
                  <th className="px-3 py-2 text-center">Status</th>
                  <th className="px-3 py-2 text-center">Strategy</th>
                  <th className="px-3 py-2 text-center">Entered</th>
                  <th className="px-3 py-2" />
                </tr>
              </thead>
              <tbody>
                {data.items.map((p: PositionOut) => (
                  <>
                    <tr
                      key={p.id}
                      className="border-t cursor-pointer hover:bg-blue-50 transition-colors"
                      onClick={() => toggleExpand(p.id)}
                    >
                      <td className="px-3 py-2 max-w-xs">
                        <div className="truncate text-gray-800">{p.market_id}</div>
                      </td>
                      <td className={`px-3 py-2 text-center font-semibold ${p.direction === 'YES' ? 'text-green-700' : 'text-red-700'}`}>
                        {p.direction}
                      </td>
                      <td className="px-3 py-2 text-center">{p.contracts}</td>
                      <td className="px-3 py-2 text-center">${fmt(p.entry_price)}</td>
                      <td className="px-3 py-2 text-center">{p.exit_price !== null ? `$${fmt(p.exit_price)}` : '—'}</td>
                      <td className={`px-3 py-2 text-center ${pnlColor(p.status === 'open' ? p.unrealized_pnl : p.pnl)}`}>
                        {p.status === 'open'
                          ? p.unrealized_pnl !== null ? `$${fmt(p.unrealized_pnl)}` : '—'
                          : p.pnl !== null ? `$${fmt(p.pnl)}` : '—'}
                      </td>
                      <td className={`px-3 py-2 text-center ${pnlColor(p.status === 'open' ? p.unrealized_pnl_pct : p.pnl_pct)}`}>
                        {p.status === 'open' ? fmtPct(p.unrealized_pnl_pct) : fmtPct(p.pnl_pct)}
                      </td>
                      <td className="px-3 py-2 text-center">
                        <span className={`px-1.5 py-0.5 rounded text-xs font-medium ${
                          p.status === 'open' ? 'bg-green-100 text-green-800' :
                          p.status === 'closed' ? 'bg-gray-100 text-gray-700' :
                          'bg-yellow-100 text-yellow-800'
                        }`}>{p.status}</span>
                      </td>
                      <td className="px-3 py-2 text-center text-gray-500 text-xs">{p.strategy_name}</td>
                      <td className="px-3 py-2 text-center text-gray-500 text-xs">{relTime(p.entry_time)}</td>
                      <td className="px-3 py-2 text-center text-gray-400 text-xs">
                        {expandedId === p.id ? '▲' : '▼'}
                      </td>
                    </tr>
                    {expandedId === p.id && (
                      <tr key={`${p.id}-detail`}>
                        <td colSpan={11} className="p-0">
                          <PositionDetail positionId={p.id} />
                        </td>
                      </tr>
                    )}
                  </>
                ))}
                {data.items.length === 0 && (
                  <tr>
                    <td colSpan={11} className="px-3 py-6 text-center text-gray-400">No positions</td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  )
}
