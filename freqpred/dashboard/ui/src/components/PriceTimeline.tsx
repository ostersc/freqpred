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
import type { SignalOut } from '../api/types'

// ---- Formatting helpers -------------------------------------------------

export function fmtTs(ts: number) {
  const d = new Date(ts)
  return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' }) +
    ' ' + d.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' })
}

export function triggerLabel(trigger: string): string {
  const t = trigger.toLowerCase()
  switch (t) {
    case 'entry': return 'Entry signal'
    case 'scheduled': return 'Scheduled signal'
    case 'price_moved': return 'Price-moved signal'
    case 'market_update': return 'Market update signal'
    case 'manual': return 'Manual signal'
    default:
      return trigger.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase()) + ' signal'
  }
}

// ---- Chart -------------------------------------------------------------

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

export interface PriceTimelineProps {
  signals: SignalOut[]
  entrySignalId: string
  entryPrice: number
  currentMid: number | null
  direction: string
  selectedSignalId: string | null
  onSignalClick: (id: string) => void
  // Exit event — optional. When both exitPrice and exitTime are provided
  // (closed positions), the chart draws a vertical red "Exit" line at exit_time
  // and a horizontal dashed line at the exit price. NO trades flip the horizontal
  // exit line to the YES-equivalent (1 − exit_price) so it is comparable to the
  // YES-axis chart.
  exitPrice?: number | null
  exitTime?: string | null
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  exitReason?: string | null
}

export default function PriceTimeline({
  signals,
  entrySignalId,
  entryPrice,
  currentMid,
  direction,
  selectedSignalId,
  onSignalClick,
  exitPrice = null,
  exitTime = null,
}: PriceTimelineProps) {
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
  const chartExitPrice =
    exitPrice !== null && exitPrice !== undefined
      ? (direction === 'NO' ? 1 - exitPrice : exitPrice)
      : null
  const exitTs = exitTime ? new Date(exitTime).getTime() : null

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
      <div style={{ fontSize: 10.5, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.08em', color: 'var(--fg-2)', marginBottom: 4 }}>
        Signal history — estimated probability vs. market mid
      </div>
      <div style={{ fontSize: 11, color: 'var(--fg-3)', marginBottom: 8 }}>
        Click any point to view that signal's detail below.
        Shapes: <b>●</b> entry/manual &nbsp;
        <b>◆</b> scheduled &nbsp;
        <b>▲</b> price moved &nbsp;
        <b>■</b> market update
      </div>
      <ResponsiveContainer width="100%" height={220}>
        <ComposedChart data={data} margin={{ top: 8, right: 16, bottom: 5, left: 0 }} onClick={handleChartClick} style={{ cursor: 'pointer' }}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--line-soft)" />
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
                <div style={{ background: 'var(--bg-1)', border: '1px solid var(--line)', borderRadius: 6, padding: '8px 10px', fontSize: 11.5, display: 'flex', flexDirection: 'column', gap: 3, boxShadow: 'var(--shadow-panel)' }}>
                  <div style={{ color: 'var(--fg-3)' }}>{fmtTs(d.ts)}</div>
                  <div>Our prob: <b className="mono" style={{ color: 'var(--accent)' }}>{(d.our_prob * 100).toFixed(1)}%</b></div>
                  <div>Market mid: <b className="mono" style={{ color: 'var(--warn)' }}>{(d.market_mid * 100).toFixed(1)}%</b></div>
                  <div style={{ fontWeight: 600, color: edge >= 0 ? 'var(--pos)' : 'var(--neg)' }}>
                    Edge: {edge >= 0 ? '+' : ''}{(edge * 100).toFixed(1)}%
                  </div>
                  <div style={{ color: 'var(--fg-3)', textTransform: 'capitalize', paddingTop: 2 }}>
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

          {/* Exit price horizontal dashed line + timestamp vertical line */}
          {chartExitPrice !== null && (
            <ReferenceLine
              y={chartExitPrice}
              stroke="#dc2626"
              strokeDasharray="4 2"
              strokeOpacity={0.7}
              label={{ value: `Exit ${(chartExitPrice * 100).toFixed(0)}%`, position: 'insideBottomRight', fontSize: 9, fill: '#dc2626' }}
            />
          )}
          {exitTs !== null && (
            <ReferenceLine
              x={exitTs}
              stroke="#dc2626"
              strokeDasharray="4 2"
              strokeOpacity={0.7}
              label={{ value: 'Exit', position: 'top', fontSize: 9, fill: '#dc2626' }}
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
