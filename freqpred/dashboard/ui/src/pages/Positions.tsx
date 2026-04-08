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
import { getSignal } from '../api/signals'
import type { PositionOut, PositionDetailOut, SignalOut, SignalDetailOut } from '../api/types'
import LoadingSpinner from '../components/LoadingSpinner'
import ErrorBanner from '../components/ErrorBanner'
import { DocLinkItem } from '../components/DocLinkItem'

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

function triggerLabel(trigger: string): string {
  const t = trigger.toLowerCase()
  switch (t) {
    case 'entry': return 'Entry signal'
    case 'scheduled': return 'Scheduled signal'
    case 'price_moved': return 'Price-moved signal'
    case 'market_update': return 'Market update signal'
    case 'manual': return 'Manual signal'
    default: return trigger.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase()) + ' signal'
  }
}

// ---- Signal price timeline chart ----------------------------------------

type ChartPoint = {
  ts: number
  our_prob: number
  market_mid: number
  isEntry: boolean
  signalId: string
  trigger: string
}

function renderDotShape(
  trigger: string,
  cx: number,
  cy: number,
  r: number,
  fill: string,
  stroke: string,
  strokeWidth: number,
  key: string,
) {
  const t = trigger.toLowerCase()
  if (t === 'scheduled') {
    // Diamond
    return (
      <polygon
        key={key}
        points={`${cx},${cy - r} ${cx + r},${cy} ${cx},${cy + r} ${cx - r},${cy}`}
        fill={fill} stroke={stroke} strokeWidth={strokeWidth}
      />
    )
  }
  if (t === 'price_moved') {
    // Triangle (pointing up)
    return (
      <polygon
        key={key}
        points={`${cx},${cy - r} ${cx + r},${cy + r} ${cx - r},${cy + r}`}
        fill={fill} stroke={stroke} strokeWidth={strokeWidth}
      />
    )
  }
  if (t === 'market_update') {
    // Square
    return (
      <rect
        key={key}
        x={cx - r} y={cy - r} width={r * 2} height={r * 2}
        fill={fill} stroke={stroke} strokeWidth={strokeWidth}
      />
    )
  }
  // Default: circle (entry, manual, unknown)
  return (
    <circle
      key={key}
      cx={cx} cy={cy} r={r}
      fill={fill} stroke={stroke} strokeWidth={strokeWidth}
    />
  )
}

function PriceTimeline({
  signals,
  entrySignalId,
  entryPrice,
  currentMid,
  direction,
  selectedSignalId,
  onSignalClick,
}: {
  signals: SignalOut[]
  entrySignalId: string
  entryPrice: number
  currentMid: number | null
  direction: string
  selectedSignalId: string | null
  onSignalClick: (id: string) => void
}) {
  if (signals.length === 0) return null

  const data: ChartPoint[] = signals.map((s) => ({
    ts: new Date(s.created_at).getTime(),
    our_prob: s.estimated_probability,
    market_mid: s.market_mid_at_signal,
    isEntry: s.id === entrySignalId,
    signalId: s.id,
    trigger: s.trigger,
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

  const renderDot = (props: { cx?: number; cy?: number; payload?: ChartPoint }) => {
    const { cx = 0, cy = 0, payload } = props
    if (!payload) return <g key={`${cx}-${cy}`} />

    const isEntry = payload.isEntry
    const isSelected = payload.signalId === selectedSignalId
    const r = isEntry ? 6 : 4
    const fill = isEntry ? '#3b82f6' : '#6366f1'
    const stroke = isSelected ? '#1e1b4b' : '#fff'
    const strokeWidth = isSelected ? 2.5 : 1.5

    return renderDotShape(
      payload.trigger, cx, cy, r, fill, stroke, strokeWidth,
      `dot-${payload.signalId}`,
    )
  }

  // Recharts intercepts pointer events before custom dot onClick fires.
  // Handle clicks at the chart level instead — activePayload gives nearest point.
  // Entry signal takes precedence: if the clicked point shares a timestamp with the
  // entry signal, always select the entry signal instead.
  function handleChartClick(chartData: { activePayload?: Array<{ dataKey?: string; payload?: ChartPoint }> } | null) {
    if (!chartData?.activePayload?.length) return
    const point = chartData.activePayload.find((p) => p.dataKey === 'our_prob')
    if (!point?.payload?.signalId) return

    const clickedId = point.payload.signalId
    const clickedTs = point.payload.ts
    const entryPoint = data.find((p) => p.signalId === entrySignalId)

    if (entryPoint && entryPoint.ts === clickedTs && clickedId !== entrySignalId) {
      onSignalClick(entrySignalId)
    } else {
      onSignalClick(clickedId)
    }
  }

  return (
    <div>
      <div className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-1">
        Signal history — estimated probability vs. market mid
      </div>
      <div className="text-xs text-gray-400 mb-2">
        Click any point to view that signal's detail below.
        Shapes: <span className="font-medium">●</span> entry/manual &nbsp;
        <span className="font-medium">◆</span> scheduled &nbsp;
        <span className="font-medium">▲</span> price moved &nbsp;
        <span className="font-medium">■</span> market update
      </div>
      <ResponsiveContainer width="100%" height={220}>
        <ComposedChart data={data} margin={{ top: 8, right: 16, bottom: 5, left: 0 }} onClick={handleChartClick} style={{ cursor: 'pointer' }}>
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
                  <div className="text-gray-400 capitalize pt-0.5">
                    {d.trigger.replace(/_/g, ' ')}{d.isEntry ? ' · entry' : ''}
                  </div>
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

// ---- Signal detail (shared renderer) ------------------------------------

function SignalDetail({ signal }: { signal: SignalDetailOut }) {
  return (
    <div className="bg-white rounded border p-3 space-y-3">
      <div className="flex flex-wrap gap-4 text-xs text-gray-500">
        <span>Our prob: <span className="font-semibold text-gray-800">{(signal.estimated_probability * 100).toFixed(1)}%</span></span>
        <span>Market mid: <span className="font-semibold text-gray-800">{(signal.market_mid_at_signal * 100).toFixed(1)}%</span></span>
        <span>Edge: <span className={`font-semibold ${signal.edge >= 0 ? 'text-green-700' : 'text-red-700'}`}>{fmtPct(signal.edge)}</span></span>
        <span>Confidence: <span className="font-semibold text-gray-800">{(signal.confidence * 100).toFixed(1)}%</span></span>
        <span className="text-gray-400">{relTime(signal.created_at)}</span>
      </div>
      <div>
        <div className="font-medium text-gray-700 mb-1">Reasoning:</div>
        <p className="text-gray-600 whitespace-pre-wrap">{signal.reasoning}</p>
      </div>
      {signal.social_sentiment_summary && (
        <div>
          <div className="font-medium text-gray-700 mb-1">Social sentiment:</div>
          <p className="text-gray-600">{signal.social_sentiment_summary}</p>
        </div>
      )}
      {signal.document_links.length > 0 && (
        <div>
          <div className="font-medium text-gray-700 mb-1">Evidence documents:</div>
          <ul className="space-y-1.5">
            {signal.document_links.map((doc) => (
              <DocLinkItem key={doc.document_id} doc={doc} />
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}

// Fetches signal detail on demand (for non-entry signals clicked in chart)
function SelectedSignalPanel({
  signalId,
  entrySignal,
}: {
  signalId: string
  entrySignal: SignalDetailOut
}) {
  const isEntry = signalId === entrySignal.id

  const { data, isLoading } = useQuery({
    queryKey: ['signal', signalId],
    queryFn: () => getSignal(signalId),
    staleTime: 60_000,
    enabled: !isEntry,
  })

  if (isEntry) return <SignalDetail signal={entrySignal} />
  if (isLoading) return <div className="p-3 text-sm text-gray-400">Loading signal…</div>
  if (!data) return null
  return <SignalDetail signal={data} />
}

// ---- Position detail panel -----------------------------------------------

function PositionDetail({ positionId }: { positionId: string }) {
  // null = show entry signal (default)
  const [selectedSignalId, setSelectedSignalId] = useState<string | null>(null)

  const { data, isLoading, error } = useQuery({
    queryKey: ['position-detail', positionId],
    queryFn: () => getPositionDetail(positionId),
    staleTime: 30_000,
  })

  if (isLoading) return <div className="p-4 text-sm text-gray-500">Loading…</div>
  if (error) return <div className="p-4 text-sm text-red-600">{String(error)}</div>
  if (!data) return null

  const d: PositionDetailOut = data
  const activeSignalId = selectedSignalId ?? d.entry_signal.id
  const isEntryActive = activeSignalId === d.entry_signal.id

  // Entry signal always labelled "Entry signal" regardless of its trigger field.
  // For other signals, look up the trigger from market_signals.
  const activeTrigger = isEntryActive
    ? 'entry'
    : (d.market_signals.find((s) => s.id === activeSignalId)?.trigger ?? 'entry')

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
        selectedSignalId={activeSignalId}
        onSignalClick={(id) => setSelectedSignalId(id)}
      />

      {/* Selected signal panel — label changes to match clicked signal's trigger */}
      <div>
        <div className="flex items-center gap-3 mb-2">
          <div className="text-xs font-semibold text-gray-500 uppercase tracking-wide">
            {triggerLabel(activeTrigger)}
          </div>
          {!isEntryActive && (
            <button
              onClick={() => setSelectedSignalId(null)}
              className="text-xs text-blue-500 hover:underline"
            >
              ← back to entry signal
            </button>
          )}
        </div>
        <SelectedSignalPanel signalId={activeSignalId} entrySignal={d.entry_signal} />
      </div>
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
                  <th className="px-3 py-2 text-center">Exposure</th>
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
                      <td className="px-3 py-2 text-center text-gray-700">${fmt(p.entry_price * p.contracts)}</td>
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
                        <td colSpan={12} className="p-0">
                          <PositionDetail positionId={p.id} />
                        </td>
                      </tr>
                    )}
                  </>
                ))}
                {data.items.length === 0 && (
                  <tr>
                    <td colSpan={12} className="px-3 py-6 text-center text-gray-400">No positions</td>
                  </tr>
                )}
                {data.items.length > 0 && (() => {
                  const totalContracts = data.items.reduce((s, p) => s + p.contracts, 0)
                  const totalPnl = data.items.reduce((s, p) => {
                    const v = p.status === 'open' ? p.unrealized_pnl : p.pnl
                    return s + (v ?? 0)
                  }, 0)
                  const totalCostBasis = data.items.reduce((s, p) => s + p.entry_price * p.contracts, 0)
                  const weightedPct = totalCostBasis > 0 ? totalPnl / totalCostBasis : null
                  const weightedAvgEntry = totalContracts > 0 ? totalCostBasis / totalContracts : null
                  return (
                    <tr className="border-t-2 border-gray-300 bg-gray-50 font-semibold text-xs">
                      <td className="px-3 py-2 text-gray-500 uppercase tracking-wide">Total</td>
                      <td />
                      <td className="px-3 py-2 text-center">{totalContracts}</td>
                      <td className="px-3 py-2 text-center text-gray-600">{weightedAvgEntry !== null ? `$${fmt(weightedAvgEntry)}` : '—'}</td>
                      <td className="px-3 py-2 text-center text-gray-700">${fmt(totalCostBasis)}</td>
                      <td />
                      <td className={`px-3 py-2 text-center ${pnlColor(totalPnl)}`}>${fmt(totalPnl)}</td>
                      <td className={`px-3 py-2 text-center ${pnlColor(weightedPct)}`}>{fmtPct(weightedPct)}</td>
                      <td /><td /><td /><td />
                    </tr>
                  )
                })()}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  )
}
