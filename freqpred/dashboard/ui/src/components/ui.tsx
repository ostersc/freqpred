import React, { useMemo, useState } from 'react'

// ---- Badge ----
type BadgeKind = 'pos' | 'neg' | 'warn' | 'info' | 'accent' | 'muted'

export function Badge({ kind = 'muted', children, dot }: {
  kind?: BadgeKind
  children: React.ReactNode
  dot?: boolean
}) {
  return (
    <span className={`badge ${kind}`}>
      {dot && <span className="d" />}
      {children}
    </span>
  )
}

// ---- Panel ----
export function Panel({ title, children, action, style, className = '', flush = false }: {
  title?: string
  children: React.ReactNode
  action?: React.ReactNode
  style?: React.CSSProperties
  className?: string
  flush?: boolean
}) {
  return (
    <div className={`panel ${className}`} style={style}>
      {title && (
        <div className="panel-head">
          <div className="panel-head-title">{title}</div>
          {action}
        </div>
      )}
      <div className={`panel-body ${flush ? 'flush' : ''}`}>{children}</div>
    </div>
  )
}

// ---- Stat ----
export function Stat({ label, value, sub, delta, deltaKind, spark, accent, children }: {
  label: string
  value: React.ReactNode
  sub?: string
  delta?: string
  deltaKind?: string
  spark?: React.ReactNode
  accent?: string
  children?: React.ReactNode
}) {
  return (
    <div className="stat">
      <div className="stat-label">
        {label}
        {accent && <span style={{ width: 5, height: 5, borderRadius: '50%', background: accent, display: 'inline-block', marginLeft: 5 }} />}
      </div>
      <div style={{ display: 'flex', alignItems: 'flex-end', justifyContent: 'space-between', gap: 12 }}>
        <div>
          <div className="stat-value">{value}</div>
          {sub && <div className="stat-sub">{sub}</div>}
          {delta != null && <div className={`stat-delta ${deltaKind || ''}`}>{delta}</div>}
        </div>
        {spark}
      </div>
      {children}
    </div>
  )
}

// ---- Segmented control ----
export function Segmented<T extends string>({ items, value, onChange }: {
  items: { v: T; label: string }[] | T[]
  value: T
  onChange: (v: T) => void
}) {
  return (
    <div className="seg">
      {items.map((item) => {
        const v = typeof item === 'string' ? item as T : (item as { v: T }).v
        const label = typeof item === 'string' ? item : (item as { label: string }).label
        return (
          <button
            key={v}
            className={`seg-item${v === value ? ' active' : ''}`}
            onClick={() => onChange(v)}
          >
            {label}
          </button>
        )
      })}
    </div>
  )
}

// ---- ProbBar ----
export function ProbBar({ ours, market }: { ours: number; market: number }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8, minWidth: 140 }}>
      <div style={{ flex: 1, height: 14, position: 'relative', background: 'var(--bg-3)', borderRadius: 3, overflow: 'hidden' }}>
        <div style={{ position: 'absolute', left: 0, top: 0, bottom: 0, width: `${market * 100}%`, background: 'var(--warn)', opacity: 0.55 }} />
        <div style={{ position: 'absolute', left: `${ours * 100}%`, top: -2, bottom: -2, width: 2, background: 'var(--accent)' }} />
      </div>
      <span className="mono" style={{ fontSize: 10.5, color: 'var(--fg-2)', whiteSpace: 'nowrap' }}>
        {(ours * 100).toFixed(0)}/{(market * 100).toFixed(0)}
      </span>
    </div>
  )
}

// ---- Icon ----
type IconName = 'search' | 'chev' | 'chevR' | 'filter' | 'refresh' | 'check' | 'x' | 'info' | 'bolt' | 'gear'

export function Icon({ name, size = 14 }: { name: IconName; size?: number }) {
  const paths: Record<IconName, React.ReactNode> = {
    search:  <path d="M11 11l3 3M7 12.5A5.5 5.5 0 117 1.5a5.5 5.5 0 010 11z" stroke="currentColor" strokeWidth="1.5" fill="none" strokeLinecap="round"/>,
    chev:    <path d="M4 6l4 4 4-4" stroke="currentColor" strokeWidth="1.5" fill="none" strokeLinecap="round" strokeLinejoin="round"/>,
    chevR:   <path d="M6 4l4 4-4 4" stroke="currentColor" strokeWidth="1.5" fill="none" strokeLinecap="round" strokeLinejoin="round"/>,
    filter:  <path d="M2 3h12l-4.5 6v4l-3 1V9L2 3z" stroke="currentColor" strokeWidth="1.3" fill="none" strokeLinejoin="round"/>,
    refresh: <path d="M13 8A5 5 0 013 8m0 0V5m0 3h3M3 8a5 5 0 0110 0m0 0v3m0-3h-3" stroke="currentColor" strokeWidth="1.3" fill="none" strokeLinecap="round" strokeLinejoin="round"/>,
    check:   <path d="M3 8l3 3 7-7" stroke="currentColor" strokeWidth="1.8" fill="none" strokeLinecap="round" strokeLinejoin="round"/>,
    x:       <path d="M4 4l8 8M12 4l-8 8" stroke="currentColor" strokeWidth="1.5" fill="none" strokeLinecap="round"/>,
    info:    <path d="M8 1a7 7 0 100 14A7 7 0 008 1zm0 6v4M8 5.5v.01" stroke="currentColor" strokeWidth="1.3" fill="none" strokeLinecap="round"/>,
    bolt:    <path d="M9 1L3 9h4l-1 6 6-8H8l1-6z" stroke="currentColor" strokeWidth="1.3" fill="none" strokeLinejoin="round"/>,
    gear:    <path d="M8 5.5a2.5 2.5 0 110 5 2.5 2.5 0 010-5zM8 1v2M8 13v2M14 8h-2M4 8H2M12.24 3.76l-1.41 1.41M5.17 10.83l-1.41 1.41M12.24 12.24l-1.41-1.41M5.17 5.17L3.76 3.76" stroke="currentColor" strokeWidth="1.3" fill="none" strokeLinecap="round"/>,
  }
  return <svg width={size} height={size} viewBox="0 0 16 16">{paths[name]}</svg>
}

// ---- Sparkline ----
export function Sparkline({ data, timestamps, w = 80, h = 22, color = 'var(--accent)', fill = true }: {
  data: number[]
  timestamps?: number[]
  w?: number
  h?: number
  color?: string
  fill?: boolean
}) {
  if (!data || data.length < 2) return null
  const min = Math.min(...data), max = Math.max(...data)
  const range = max - min || 1
  const useTime = timestamps && timestamps.length === data.length
  const tMin = useTime ? Math.min(...timestamps!) : 0
  const tMax = useTime ? Math.max(...timestamps!) : 0
  const tRange = tMax - tMin || 1
  const pts = data.map((v, i) => [
    useTime ? ((timestamps![i] - tMin) / tRange) * w : (i / (data.length - 1)) * w,
    h - 2 - ((v - min) / range) * (h - 4),
  ])
  const d = pts.map((p, i) => `${i ? 'L' : 'M'}${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(' ')
  const dFill = `${d} L${w},${h} L0,${h} Z`
  return (
    <svg className="spark" width={w} height={h} viewBox={`0 0 ${w} ${h}`}>
      {fill && <path d={dFill} fill={color} opacity="0.12" />}
      <path d={d} fill="none" stroke={color} strokeWidth="1.25" strokeLinejoin="round" strokeLinecap="round" />
    </svg>
  )
}

// ---- MiniStat ----
export function MiniStat({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="mini-stat">
      <div className="mini-stat-label">{label}</div>
      <div className="mini-stat-value">{value}</div>
    </div>
  )
}

// ---- LoadingSpinner ----
export function LoadingSpinner() {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '40px', color: 'var(--fg-2)', fontSize: 12 }}>
      <span className="spinner-ring" />
      Loading…
    </div>
  )
}

// ---- ErrorBanner ----
export function ErrorBanner({ message }: { message: string }) {
  return <div className="error-banner">{message}</div>
}

// ---- WarnBanner ----
export function WarnBanner({ message }: { message: string }) {
  return <div className="warn-banner">{message}</div>
}

// ---- Labeled helpers ----
export function LabeledSelect({ label, value, onChange, options }: {
  label: string
  value: string
  onChange: (v: string) => void
  options: { value: string; label: string }[]
}) {
  return (
    <div className="labeled-field">
      <label className="field-label">{label}</label>
      <select className="input select" value={value} onChange={(e) => onChange(e.target.value)}>
        {options.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
      </select>
    </div>
  )
}

// ---- RangeSlider (dual-thumb, two overlaid native range inputs) ----
export function RangeSlider({ min, max, step = 0.01, valueMin, valueMax, onChange }: {
  min: number
  max: number
  step?: number
  valueMin: number
  valueMax: number
  onChange: (lo: number, hi: number) => void
}) {
  const span = max - min || 1
  const loPct = ((valueMin - min) / span) * 100
  const hiPct = ((valueMax - min) / span) * 100

  return (
    <div className="range-slider">
      <div className="range-slider-track">
        <div className="range-slider-fill" style={{ left: `${loPct}%`, width: `${hiPct - loPct}%` }} />
      </div>
      <input
        type="range"
        className="range-slider-input"
        min={min}
        max={max}
        step={step}
        value={valueMin}
        onChange={(e) => onChange(Math.min(Number(e.target.value), valueMax), valueMax)}
      />
      <input
        type="range"
        className="range-slider-input"
        min={min}
        max={max}
        step={step}
        value={valueMax}
        onChange={(e) => onChange(valueMin, Math.max(Number(e.target.value), valueMin))}
      />
    </div>
  )
}

export function LabeledInput({ label, placeholder, type = 'text', value, onChange }: {
  label: string
  placeholder?: string
  type?: string
  value?: string
  onChange?: (v: string) => void
}) {
  return (
    <div className="labeled-field">
      <label className="field-label">{label}</label>
      <input
        className="input"
        placeholder={placeholder}
        type={type}
        value={value}
        onChange={(e) => onChange?.(e.target.value)}
      />
    </div>
  )
}

// ---- Donut chart (SVG) ----
export function Donut({ data, size = 80 }: {
  data: { label: string; pct: number; color: string }[]
  size?: number
}) {
  const total = data.reduce((a, b) => a + b.pct, 0)
  let angle = -Math.PI / 2
  const r = size / 2 - 2, cx = size / 2, cy = size / 2, ir = r * 0.62
  const parts = data.map((d) => {
    const sweep = (d.pct / total) * Math.PI * 2
    const x1 = cx + Math.cos(angle) * r, y1 = cy + Math.sin(angle) * r
    const x2 = cx + Math.cos(angle + sweep) * r, y2 = cy + Math.sin(angle + sweep) * r
    const x3 = cx + Math.cos(angle + sweep) * ir, y3 = cy + Math.sin(angle + sweep) * ir
    const x4 = cx + Math.cos(angle) * ir, y4 = cy + Math.sin(angle) * ir
    const large = sweep > Math.PI ? 1 : 0
    const path = `M${x1},${y1} A${r},${r} 0 ${large} 1 ${x2},${y2} L${x3},${y3} A${ir},${ir} 0 ${large} 0 ${x4},${y4} Z`
    angle += sweep
    return <path key={d.label} d={path} fill={d.color} />
  })
  return <svg width={size} height={size}>{parts}</svg>
}

// ---- Seeded RNG for sparklines ----
export function mulberry32(a: number) {
  return function () {
    let t = (a += 0x6d2b79f5)
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

export function walk(seed: number, n = 24, start = 50, vol = 6): number[] {
  let s = seed
  const out = [start]
  for (let i = 0; i < n - 1; i++) {
    s = (s * 9301 + 49297) % 233280
    const r = s / 233280
    out.push(Math.max(0, Math.min(100, out[out.length - 1] + (r - 0.5) * vol * 2)))
  }
  return out
}

// ---- Format helpers ----
export const fmtMoney = (v: number, digits = 2) =>
  (v < 0 ? '-' : '') + '$' + Math.abs(v).toFixed(digits)

export const fmtSignedMoney = (v: number) =>
  (v >= 0 ? '+' : '-') + '$' + Math.abs(v).toFixed(2)

export const fmtPct = (v: number, d = 1) =>
  (v * 100).toFixed(d) + '%'

export const fmtSignedPct = (v: number, d = 1) =>
  (v >= 0 ? '+' : '') + (v * 100).toFixed(d) + '%'

export const fmtCents = (v: number) => (v * 100).toFixed(1) + '¢'

export function fmtAge(iso: string): string {
  const secs = (Date.now() - new Date(iso).getTime()) / 1000
  if (secs < 60) return `${Math.round(secs)}s`
  if (secs < 3600) return `${Math.round(secs / 60)}m`
  if (secs < 86400) return `${Math.round(secs / 3600)}h`
  return `${Math.round(secs / 86400)}d`
}

export function fmtUptime(secs: number): string {
  const h = Math.floor(secs / 3600)
  const m = Math.floor((secs % 3600) / 60)
  if (h > 0) return `${h}h ${m}m`
  if (m > 0) return `${m}m ${secs % 60}s`
  return `${secs}s`
}

// ---- Signal history chart ----
const SIG_KINDS = [
  { key: 'entry',         color: 'var(--pos)',    shape: 'triangle' },
  { key: 'manual',        color: 'var(--warn)',   shape: 'triangle' },
  { key: 'scheduled',     color: 'var(--accent)', shape: 'diamond' },
  { key: 'price_moved',   color: 'var(--fg-2)',   shape: 'triangleU' },
  { key: 'market_update', color: 'var(--fg-3)',   shape: 'square' },
]

interface ChartSignal {
  idx: number
  prob: number
  market: number
  edge: number
  time: Date
  kind: string
  reasoning?: string
}

function ShapeMark({ kind, cx, cy, r = 5 }: { kind: string; cx: number; cy: number; r?: number }) {
  const s = SIG_KINDS.find((k) => k.key === kind) || SIG_KINDS[2]
  const fill = s.color
  if (s.shape === 'triangle') return <polygon points={`${cx},${cy - r} ${cx - r},${cy + r} ${cx + r},${cy + r}`} fill={fill} />
  if (s.shape === 'triangleU') return <polygon points={`${cx},${cy + r} ${cx - r},${cy - r} ${cx + r},${cy - r}`} fill={fill} />
  if (s.shape === 'diamond') return <polygon points={`${cx},${cy - r} ${cx + r},${cy} ${cx},${cy + r} ${cx - r},${cy}`} fill={fill} />
  return <rect x={cx - r} y={cy - r} width={r * 2} height={r * 2} fill={fill} />
}

export interface SignalPoint {
  estimated_probability: number
  market_mid_at_signal: number
  created_at: string
  trigger: string
  reasoning?: string
}

export function SignalHistoryChart({ signals }: { signals: SignalPoint[] }) {
  const W = 960, H = 220, padL = 44, padR = 16, padT = 14, padB = 36
  const [selIdx, setSelIdx] = useState<number | null>(null)
  const [hover, setHover] = useState<number | null>(null)

  const sorted = useMemo(
    () => [...signals].sort((a, b) => new Date(a.created_at).getTime() - new Date(b.created_at).getTime()),
    [signals],
  )

  if (sorted.length < 2) {
    return <div style={{ padding: 16, color: 'var(--fg-3)', fontSize: 12 }}>Not enough signal history to display chart.</div>
  }

  const N = sorted.length
  const probs = sorted.map((s) => s.estimated_probability)
  const mkts = sorted.map((s) => s.market_mid_at_signal)

  const toY = (v: number) => padT + (1 - v) * (H - padT - padB)
  const toX = (i: number) => padL + (i / (N - 1)) * (W - padL - padR)

  const fmtT = (iso: string) =>
    new Date(iso).toLocaleString(undefined, { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' })

  const toPath = (arr: number[]) =>
    arr.map((v, i) => `${i ? 'L' : 'M'}${toX(i).toFixed(1)},${toY(v).toFixed(1)}`).join(' ')

  const segs = []
  for (let i = 0; i < N - 1; i++) {
    const e0 = probs[i] - mkts[i]
    const e1 = probs[i + 1] - mkts[i + 1]
    const positive = (e0 + e1) / 2 > 0
    segs.push({ x0: toX(i), x1: toX(i + 1), positive, y0o: toY(probs[i]), y1o: toY(probs[i + 1]), y0m: toY(mkts[i]), y1m: toY(mkts[i + 1]) })
  }

  const events: ChartSignal[] = sorted.map((s, idx) => ({
    idx,
    prob: s.estimated_probability,
    market: s.market_mid_at_signal,
    edge: s.estimated_probability - s.market_mid_at_signal,
    time: new Date(s.created_at),
    kind: s.trigger,
    reasoning: s.reasoning,
  }))

  const tickIdxs = N <= 5 ? events.map((_, i) => i) : [0, Math.floor(N / 4), Math.floor(N / 2), Math.floor(N * 3 / 4), N - 1]

  const hovered = hover !== null ? events[hover] : null

  return (
    <div className="sig-chart-wrap">
      <svg width="100%" viewBox={`0 0 ${W} ${H}`} style={{ display: 'block' }}>
        {[0, 0.25, 0.5, 0.75, 1].map((t) => (
          <g key={t}>
            <line x1={padL} x2={W - padR} y1={toY(t)} y2={toY(t)} stroke="var(--line-soft)" strokeDasharray="2 4" />
            <text x={padL - 6} y={toY(t) + 3} fontSize="10" fill="var(--fg-3)" textAnchor="end" fontFamily="var(--f-mono)">{(t * 100).toFixed(0)}%</text>
          </g>
        ))}
        {segs.map((s, i) => (
          <polygon key={i}
            points={`${s.x0},${s.y0o} ${s.x1},${s.y1o} ${s.x1},${s.y1m} ${s.x0},${s.y0m}`}
            fill={s.positive ? 'var(--pos)' : 'var(--neg)'} opacity={s.positive ? 0.14 : 0.18}
          />
        ))}
        {tickIdxs.map((i) => (
          <g key={i}>
            <line x1={toX(i)} x2={toX(i)} y1={H - padB} y2={H - padB + 4} stroke="var(--fg-3)" />
            <text x={toX(i)} y={H - padB + 16} fontSize="10" fill="var(--fg-3)" textAnchor="middle" fontFamily="var(--f-mono)">
              {fmtT(sorted[i].created_at)}
            </text>
          </g>
        ))}
        <path d={toPath(mkts)} fill="none" stroke="var(--warn)" strokeWidth="1.4" />
        <path d={toPath(probs)} fill="none" stroke="var(--accent)" strokeWidth="1.6" />
        {events.map((e, i) => {
          const selected = i === selIdx
          return (
            <g key={i} style={{ cursor: 'pointer' }}
              onClick={() => setSelIdx(i === selIdx ? null : i)}
              onMouseEnter={() => setHover(i)}
              onMouseLeave={() => setHover(null)}
            >
              {selected && <circle cx={toX(e.idx)} cy={toY(e.prob)} r={10} fill="var(--accent)" opacity={0.2} />}
              <ShapeMark kind={e.kind} cx={toX(e.idx)} cy={toY(e.prob)} r={selected ? 6 : 4.5} />
            </g>
          )
        })}
        <g transform={`translate(${W - padR - 180},${padT})`}>
          <rect x={-8} y={-10} width={188} height={22} fill="var(--bg-1)" stroke="var(--line)" rx={4} />
          <circle cx={0} cy={0} r={3} fill="var(--accent)" />
          <text x={8} y={3} fontSize="10" fill="var(--fg-1)">Our prob</text>
          <circle cx={72} cy={0} r={3} fill="var(--warn)" />
          <text x={80} y={3} fontSize="10" fill="var(--fg-1)">Market mid</text>
        </g>
      </svg>
      {hovered && (
        <div className="sig-tooltip" style={{
          left: `${(toX(hovered.idx) / W) * 100}%`,
          top: `${(toY(hovered.prob) / H) * 100}%`,
          transform: 'translate(-50%, -110%)',
        }}>
          <div className="dim" style={{ fontSize: 10, marginBottom: 4 }}>{fmtT(hovered.time.toISOString())}</div>
          <div className="mono"><span className="dim">Our prob:</span> <b>{(hovered.prob * 100).toFixed(1)}%</b></div>
          <div className="mono"><span className="dim">Market mid:</span> <b>{(hovered.market * 100).toFixed(1)}%</b></div>
          <div className="mono"><span className="dim">Edge:</span> <b className={hovered.edge >= 0 ? 'pos' : 'neg'}>{hovered.edge >= 0 ? '+' : ''}{(hovered.edge * 100).toFixed(1)}%</b></div>
          <div className="dim" style={{ fontSize: 10, marginTop: 4, textTransform: 'capitalize' }}>{hovered.kind.replace('_', ' ')}</div>
        </div>
      )}
      {selIdx !== null && events[selIdx]?.reasoning && (
        <div style={{ padding: '10px 16px', background: 'var(--bg-0)', borderRadius: 6, border: '1px solid var(--line-soft)', marginTop: 10, fontSize: 12, lineHeight: 1.6, color: 'var(--fg-1)' }}>
          <div style={{ fontSize: 10, textTransform: 'uppercase', letterSpacing: '0.08em', color: 'var(--fg-2)', marginBottom: 4 }}>Signal reasoning</div>
          {events[selIdx].reasoning}
        </div>
      )}
    </div>
  )
}
