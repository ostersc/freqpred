import { useMemo, useState } from 'react'

type TooltipState = {
  x: number; y: number
  series: 'Model' | 'Market'
  meanProb: number; actualRate: number; count: number
} | null
import { useQuery } from '@tanstack/react-query'
import {
  Bar,
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import {
  getCalibration,
  getCalibrationTimeSeries,
  getCalibrationByOption,
} from '../api/calibration'
import type { CalibrationFilters, CalibrationHeatmapFilters } from '../api/calibration'
import type { CalibrationHeatmapRow } from '../api/types'
import { getStrategyConfig } from '../api/strategy'
import { Stat, Panel, Segmented, LoadingSpinner, ErrorBanner } from '../components/ui'

// ---------------------------------------------------------------------------
// Constants & helpers
// ---------------------------------------------------------------------------

const PRESETS = [
  { v: '7' as const,   label: '7d' },
  { v: '30' as const,  label: '30d' },
  { v: '90' as const,  label: '90d' },
  { v: 'all' as const, label: 'All time' },
]
type Preset = '7' | '30' | '90' | 'all'
type TabId = 'distribution' | 'over-time' | 'by-option'
type EMAPeriod = '7' | '14' | '30'

const TABS: { v: TabId; label: string }[] = [
  { v: 'distribution', label: 'Distribution' },
  { v: 'over-time',    label: 'Over Time' },
  { v: 'by-option',    label: 'By Option' },
]
const EMA_PERIODS: { v: EMAPeriod; label: string }[] = [
  { v: '7',  label: '7d EMA' },
  { v: '14', label: '14d EMA' },
  { v: '30', label: '30d EMA' },
]

function presetDays(p: Preset): number | undefined {
  if (p === 'all') return undefined
  return Number(p)
}

function fmtDate(ts: number) {
  return new Date(ts).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
}

function fmtDateFull(ts: number) {
  return new Date(ts).toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: '2-digit' })
}

// ---------------------------------------------------------------------------
// Over Time chart types & helpers
// ---------------------------------------------------------------------------

interface TimeChartPoint {
  ts: number
  date: string
  brier: number | null
  market_brier: number | null
  ema: number | null
  control: number
}

function TimeTooltip({ active, payload, label }: {
  active?: boolean
  payload?: { name: string; value: number | null; color: string }[]
  label?: number
}) {
  if (!active || !payload?.length) return null
  return (
    <div className="tooltip-box">
      <div className="tooltip-title">{label != null ? fmtDateFull(label) : ''}</div>
      {payload.map((p) =>
        p.value != null ? (
          <div key={p.name} style={{ color: p.color, fontSize: 12 }}>
            {p.name}: {p.value.toFixed(4)}
          </div>
        ) : null
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Heatmap helpers
// ---------------------------------------------------------------------------

function cellStyle(delta: number | null): { bg: string; fg: string } {
  if (delta == null) return { bg: 'transparent', fg: 'var(--fg-1)' }
  const abs = Math.abs(delta)
  if (abs < 0.003) return { bg: 'transparent', fg: 'var(--fg-1)' }
  const t = Math.sqrt(Math.min(abs / 0.15, 1))
  const alpha = (t * 0.65 + 0.05).toFixed(2)
  if (delta > 0) {
    return { bg: `rgba(34,197,94,${alpha})`, fg: 'var(--fg-0)' }
  }
  return { bg: `rgba(239,68,68,${alpha})`, fg: 'var(--fg-0)' }
}

type HmTooltip = { x: number; y: number; lines: string[] } | null

function HeatmapTable({
  data,
  onNavigate,
}: {
  data: { rows: CalibrationHeatmapRow[]; prompt_versions: string[] }
  onNavigate: (filters: { promptVersion?: string; seriesTicker?: string }) => void
}) {
  const [tip, setTip] = useState<HmTooltip>(null)

  const { rows, prompt_versions: pvs } = data
  const allRow = rows[0]

  // Column max n (data rows only) for n-bar scaling per column
  const colMaxN: Record<string, number> = {}
  for (const pv of [...pvs, 'All']) {
    colMaxN[pv] = Math.max(1, ...rows.slice(1).map((r) => r.cells[pv]?.n_samples ?? 0))
  }

  // Max n across all prompt-version columns of the All Options row — denominator for the aggregate n-bar
  const allRowMaxN = Math.max(1, ...[...pvs, 'All'].map((pv) => allRow?.cells[pv]?.n_samples ?? 0))

  // Global average market Brier (allRow "All") — reference tick on difficulty bar
  const globalMarketBrier = allRow?.cells['All']?.market_brier_score ?? null

  // Max observed market Brier across data rows — scale denominator for difficulty bar
  const dataRows = rows.slice(1)
  const mktBarMax = Math.max(
    0.001,
    globalMarketBrier ?? 0,
    ...dataRows.map((r) => r.cells['All']?.market_brier_score ?? 0),
  )

  // Group data rows by series_ticker for sub-headers
  const groups: { series: string; rows: CalibrationHeatmapRow[] }[] = []
  for (const row of dataRows) {
    const last = groups[groups.length - 1]
    if (!last || last.series !== row.series_ticker) {
      groups.push({ series: row.series_ticker, rows: [row] })
    } else {
      last.rows.push(row)
    }
  }

  function showTip(e: React.MouseEvent, lines: string[]) {
    setTip({ x: e.clientX, y: e.clientY, lines })
  }
  function moveTip(e: React.MouseEvent) {
    if (tip) setTip((t) => t && { ...t, x: e.clientX, y: e.clientY })
  }
  function hideTip() { setTip(null) }

  const thStyle: React.CSSProperties = {
    padding: '3px 5px', fontSize: 11, fontWeight: 600, textAlign: 'center',
    color: 'var(--fg-2)', borderBottom: '1px solid var(--line)', whiteSpace: 'nowrap',
  }
  const tdStyle: React.CSSProperties = {
    padding: '2px 4px', fontSize: 11, textAlign: 'center',
    borderBottom: '1px solid var(--line-soft)', verticalAlign: 'middle',
  }

  function CellContent({
    row, pv, isAllRow,
  }: {
    row: CalibrationHeatmapRow; pv: string; isAllRow: boolean
  }) {
    const cell = row.cells[pv]
    if (!cell || cell.n_samples === 0) {
      return <span style={{ color: 'var(--fg-3)' }}>—</span>
    }
    const avgCell = allRow?.cells[pv]
    const avgBrier = avgCell?.brier_score ?? null
    const vsDiff = !isAllRow && avgBrier != null && cell.brier_score != null
      ? avgBrier - cell.brier_score
      : null
    const nBarPct = (isAllRow && pv === 'All')
      ? null
      : isAllRow
        ? (cell.n_samples / allRowMaxN) * 100
        : (cell.n_samples / (colMaxN[pv] || 1)) * 100
    const { fg } = cellStyle(cell.delta)

    return (
      <div style={{ minWidth: 72, color: fg }}>
        <div style={{ fontFamily: 'var(--f-mono)', fontWeight: 600, fontSize: 12, lineHeight: 1.2 }}>
          {cell.brier_score?.toFixed(3) ?? '—'}
        </div>
        {cell.delta != null && (
          <div style={{ fontSize: 10, lineHeight: 1.2, opacity: 0.8 }}>
            Δ{cell.delta >= 0 ? '+' : ''}{cell.delta.toFixed(3)}
          </div>
        )}
        {vsDiff != null && (
          <div style={{ fontSize: 9, opacity: 0.75, lineHeight: 1.2 }}>
            {vsDiff >= 0 ? '+' : ''}{vsDiff.toFixed(3)} vs avg
          </div>
        )}
        {nBarPct != null && (
          <div style={{ marginTop: 2, height: 2, background: 'rgba(255,255,255,0.07)', borderRadius: 1, overflow: 'hidden' }}>
            <div style={{ width: `${nBarPct}%`, height: '100%', background: 'rgba(99,102,241,0.5)', borderRadius: 1 }} />
          </div>
        )}
      </div>
    )
  }

  function TrendSparkline({ row }: { row: CalibrationHeatmapRow }) {
    const allVals = pvs.map((pv) => row.cells[pv]?.brier_score ?? null)
    const valid = allVals.filter((v): v is number => v != null)
    if (valid.length < 2) return <span style={{ color: 'var(--fg-3)', fontSize: 10 }}>—</span>
    const lo = Math.min(...valid), hi = Math.max(...valid)
    const span = Math.max(hi - lo, 0.02)
    const rMin = lo - span * 0.15, rMax = hi + span * 0.15
    const W = 52, H = 20, pad = 2
    const toX = (i: number) => pad + (i / (pvs.length - 1)) * (W - 2 * pad)
    const toY = (v: number) => H - pad - ((v - rMin) / (rMax - rMin)) * (H - 2 * pad)
    let d = '', dots = ''
    allVals.forEach((v, i) => {
      if (v == null) return
      const x = toX(i).toFixed(1), y = toY(v).toFixed(1)
      d += d === '' ? `M${x},${y}` : `L${x},${y}`
      dots += `<circle cx="${x}" cy="${y}" r="2"/>`
    })
    const first = allVals.find((v) => v != null)!
    const last = [...allVals].reverse().find((v) => v != null)!
    const color = last <= first ? '#4ade80' : '#f87171'
    const tipLines = [
      `Trend: ${pvs[0]} → ${pvs[pvs.length - 1]}`,
      `First: ${first.toFixed(3)}  Last: ${last.toFixed(3)}`,
      last < first ? `▼ ${(first - last).toFixed(3)} improvement` : `▲ ${(last - first).toFixed(3)} regression`,
    ]
    return (
      <svg
        width={W} height={H} viewBox={`0 0 ${W} ${H}`}
        style={{ display: 'block', margin: '0 auto', cursor: 'default', flexShrink: 0 }}
        onMouseEnter={(e) => showTip(e, tipLines)}
        onMouseMove={moveTip}
        onMouseLeave={hideTip}
      >
        <path d={d} stroke={color} strokeWidth={1.5} fill="none" strokeLinejoin="round" />
        <g fill={color} dangerouslySetInnerHTML={{ __html: dots }} />
      </svg>
    )
  }

  function cellTipLines(row: CalibrationHeatmapRow, pv: string): string[] {
    const cell = row.cells[pv]
    if (!cell || cell.n_samples === 0) return ['No data']
    const isAll = row.option_code === 'All'
    const avgBrier = allRow?.cells[pv]?.brier_score ?? null
    const vsDiff = !isAll && avgBrier != null && cell.brier_score != null
      ? avgBrier - cell.brier_score : null
    const lines = [
      `${row.option_label}  ×  ${pv === 'All' ? 'All versions' : `v${pv}`}`,
      `Model Brier: ${cell.brier_score?.toFixed(4) ?? '—'}`,
      `Market Brier: ${cell.market_brier_score?.toFixed(4) ?? '—'}`,
      `Δ (mkt−model): ${cell.delta != null ? (cell.delta >= 0 ? '+' : '') + cell.delta.toFixed(4) : '—'}`,
    ]
    if (vsDiff != null) {
      lines.push(`vs avg (${avgBrier?.toFixed(3)}): ${vsDiff >= 0 ? '+' : ''}${vsDiff.toFixed(4)}`)
    }
    lines.push(`n = ${cell.n_samples}`)
    return lines
  }

  function mktDiffTipLines(row: CalibrationHeatmapRow): string[] {
    const mktBrier = row.cells['All']?.market_brier_score
    if (mktBrier == null) return []
    if (row.option_code === 'All') {
      return [
        'All Options — global average',
        `Market Brier: ${mktBrier.toFixed(4)}`,
        'Reference bar (all data rows scaled to this max)',
      ]
    }
    if (globalMarketBrier == null) return []
    const diff = mktBrier - globalMarketBrier
    return [
      `Market difficulty: ${row.option_label}`,
      `Market Brier: ${mktBrier.toFixed(4)}`,
      `Global avg: ${globalMarketBrier.toFixed(4)}`,
      `${diff >= 0 ? '+' : ''}${diff.toFixed(4)} vs avg (${diff > 0.005 ? 'harder' : diff < -0.005 ? 'easier' : 'similar'})`,
    ]
  }

  return (
    <div style={{ overflowX: 'auto', position: 'relative' }}>
      {/* Floating tooltip */}
      {tip && (
        <div style={{
          position: 'fixed',
          left: tip.x + 14,
          top: tip.y - 10,
          background: 'var(--bg-1, #1e2440)',
          border: '1px solid var(--line)',
          borderRadius: 6,
          padding: '7px 10px',
          fontSize: 11,
          lineHeight: 1.7,
          pointerEvents: 'none',
          zIndex: 100,
          maxWidth: 240,
          whiteSpace: 'pre-wrap',
        }}>
          {tip.lines.map((l, i) => (
            <div key={i} style={{ fontWeight: i === 0 ? 600 : 400, color: i === 0 ? 'var(--fg-0)' : 'var(--fg-2)' }}>{l}</div>
          ))}
        </div>
      )}

      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
        <thead>
          <tr>
            <th style={{ ...thStyle, textAlign: 'left', minWidth: 160 }}>Option</th>
            {pvs.map((pv) => (
              <th
                key={pv}
                style={{ ...thStyle, cursor: 'pointer', color: 'var(--accent)' }}
                onClick={() => onNavigate({ promptVersion: pv })}
                onMouseEnter={(e) => showTip(e, [`Prompt version: ${pv}`, 'Click → filter Distribution to this version'])}
                onMouseMove={moveTip}
                onMouseLeave={hideTip}
              >
                {pv}
              </th>
            ))}
            <th style={{ ...thStyle }}>All</th>
            <th style={{ ...thStyle, minWidth: 72 }}>Trend</th>
          </tr>
        </thead>
        <tbody>
          {/* All Options aggregate row */}
          <tr style={{ background: 'var(--bg-2)', borderBottom: '2px solid var(--line)' }}>
            <td
              style={{ ...tdStyle, textAlign: 'left', fontWeight: 700, color: 'var(--fg-0)' }}
              onMouseEnter={(e) => showTip(e, mktDiffTipLines(allRow))}
              onMouseMove={moveTip}
              onMouseLeave={hideTip}
            >
              <div>All Options</div>
              {globalMarketBrier != null && (() => {
                const fillPct = Math.min(100, (globalMarketBrier / mktBarMax) * 100)
                return (
                  <div style={{ display: 'flex', alignItems: 'center', gap: 5, marginTop: 4 }}>
                    <div style={{ position: 'relative', width: 56, height: 4, background: 'rgba(255,255,255,0.13)', borderRadius: 2, flexShrink: 0 }}>
                      <div style={{ width: `${fillPct}%`, height: '100%', background: 'rgba(245,158,11,0.85)', borderRadius: 2 }} />
                      <div style={{ position: 'absolute', right: 0, top: -1, bottom: -1, width: 1.5, background: 'rgba(255,255,255,0.25)', borderRadius: 1 }} />
                    </div>
                    <span style={{ fontSize: 9, color: 'var(--fg-2)', fontVariantNumeric: 'tabular-nums', whiteSpace: 'nowrap' }}>
                      avg {globalMarketBrier.toFixed(3)}
                    </span>
                  </div>
                )
              })()}
            </td>
            {pvs.map((pv) => {
              const { bg } = cellStyle(allRow?.cells[pv]?.delta ?? null)
              return (
                <td
                  key={pv}
                  style={{ ...tdStyle, background: bg, cursor: 'pointer' }}
                  onClick={() => onNavigate({ promptVersion: pv })}
                  onMouseEnter={(e) => showTip(e, cellTipLines(allRow, pv))}
                  onMouseMove={moveTip}
                  onMouseLeave={hideTip}
                >
                  <CellContent row={allRow} pv={pv} isAllRow />
                </td>
              )
            })}
            <td style={{ ...tdStyle, background: cellStyle(allRow?.cells['All']?.delta ?? null).bg }}
              onMouseEnter={(e) => showTip(e, cellTipLines(allRow, 'All'))}
              onMouseMove={moveTip}
              onMouseLeave={hideTip}
            >
              <CellContent row={allRow} pv="All" isAllRow />
            </td>
            <td style={{ ...tdStyle }}>
              <TrendSparkline row={allRow} />
            </td>
          </tr>

          {groups.map(({ series, rows: gRows }) => (
            <>
              <tr
                key={`hdr-${series}`}
                style={{ background: 'var(--bg-1)', cursor: 'pointer' }}
                onClick={() => onNavigate({ seriesTicker: series })}
                onMouseEnter={(e) => showTip(e, [`Series: ${series}`, 'Click → filter Distribution to this series'])}
                onMouseMove={moveTip}
                onMouseLeave={hideTip}
              >
                <td
                  colSpan={pvs.length + 3}
                  style={{
                    ...tdStyle,
                    textAlign: 'left', fontWeight: 700, color: 'var(--accent)',
                    fontSize: 10, textTransform: 'uppercase', letterSpacing: '0.06em', paddingLeft: 8,
                  }}
                >
                  {series}
                </td>
              </tr>
              {gRows.map((row) => {
                const mktBrier = row.cells['All']?.market_brier_score ?? null
                const fillPct = mktBrier != null ? Math.min(100, (mktBrier / mktBarMax) * 100) : 0
                const refPct = globalMarketBrier != null ? Math.min(100, (globalMarketBrier / mktBarMax) * 100) : 50
                const mktDelta = mktBrier != null && globalMarketBrier != null ? mktBrier - globalMarketBrier : null
                const mktDeltaColor = mktDelta != null
                  ? (mktDelta > 0.005 ? '#f59e0b' : mktDelta < -0.005 ? '#94a3b8' : 'var(--fg-3)')
                  : 'var(--fg-3)'

                return (
                  <tr key={`${row.series_ticker}-${row.option_code}`}>
                    <td
                      style={{ ...tdStyle, textAlign: 'left', cursor: 'pointer' }}
                      onClick={() => onNavigate({ seriesTicker: row.series_ticker })}
                      onMouseEnter={(e) => showTip(e, mktDiffTipLines(row))}
                      onMouseMove={moveTip}
                      onMouseLeave={hideTip}
                    >
                      <div style={{ fontWeight: 500, overflow: 'hidden', textOverflow: 'ellipsis', maxWidth: 180, lineHeight: 1.3 }}>
                        {row.option_label}
                      </div>
                      <div style={{ color: 'var(--fg-3)', fontFamily: 'var(--f-mono)', fontSize: 10, lineHeight: 1.2 }}>
                        {row.option_code}
                      </div>
                      {mktBrier != null && globalMarketBrier != null && (
                        <div style={{ display: 'flex', alignItems: 'center', gap: 4, marginTop: 2 }}>
                          <div style={{ position: 'relative', width: 56, height: 3, background: 'rgba(255,255,255,0.07)', borderRadius: 2, flexShrink: 0 }}>
                            <div style={{ width: `${fillPct}%`, height: '100%', background: 'rgba(245,158,11,0.55)', borderRadius: 2 }} />
                            <div style={{ position: 'absolute', left: `${refPct}%`, top: -1, bottom: -1, width: 1, background: 'rgba(245,158,11,0.9)' }} />
                          </div>
                          <span style={{ fontSize: 9, color: mktDeltaColor, fontVariantNumeric: 'tabular-nums', whiteSpace: 'nowrap' }}>
                            mkt {mktBrier.toFixed(3)}
                            {mktDelta != null && (
                              <span style={{ opacity: 0.7 }}> ({mktDelta >= 0 ? '+' : ''}{mktDelta.toFixed(3)})</span>
                            )}
                          </span>
                        </div>
                      )}
                    </td>
                    {pvs.map((pv) => {
                      const { bg } = cellStyle(row.cells[pv]?.delta ?? null)
                      return (
                        <td
                          key={pv}
                          style={{ ...tdStyle, background: bg, cursor: 'pointer' }}
                          onClick={() => onNavigate({ promptVersion: pv, seriesTicker: row.series_ticker })}
                          onMouseEnter={(e) => showTip(e, cellTipLines(row, pv))}
                          onMouseMove={moveTip}
                          onMouseLeave={hideTip}
                        >
                          <CellContent row={row} pv={pv} isAllRow={false} />
                        </td>
                      )
                    })}
                    <td
                      style={{ ...tdStyle, background: cellStyle(row.cells['All']?.delta ?? null).bg }}
                      onMouseEnter={(e) => showTip(e, cellTipLines(row, 'All'))}
                      onMouseMove={moveTip}
                      onMouseLeave={hideTip}
                    >
                      <CellContent row={row} pv="All" isAllRow={false} />
                    </td>
                    <td style={{ ...tdStyle }}>
                      <TrendSparkline row={row} />
                    </td>
                  </tr>
                )
              })}
            </>
          ))}
        </tbody>
      </table>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Main page
// ---------------------------------------------------------------------------

export default function Calibration() {
  const [activeTab, setActiveTab] = useState<TabId>('distribution')
  const [preset, setPreset] = useState<Preset>('all')
  const [emaPeriod, setEmaPeriod] = useState<EMAPeriod>('7')

  // Shared filter state
  const [category, setCategory] = useState('all')
  const [tickerPrefix, setTickerPrefix] = useState('')
  const [direction, setDirection] = useState('all')
  const [modelUsed, setModelUsed] = useState('all')
  const [promptVersion, setPromptVersion] = useState('all')
  const [seriesTicker, setSeriesTicker] = useState('all')
  const [minConf, setMinConf] = useState<string | null>(null)
  const [maxConf, setMaxConf] = useState('')

  const { data: strategyConfig } = useQuery({
    queryKey: ['strategy-config'],
    queryFn: getStrategyConfig,
    staleTime: 60_000,
  })
  const strategyMinConf = strategyConfig?.min_confidence

  const effectiveMinConf = minConf === null
    ? strategyMinConf
    : (minConf !== '' ? Number(minConf) / 100 : undefined)
  const minConfDisplay = minConf === null
    ? (strategyMinConf != null ? String(Math.round(strategyMinConf * 100)) : '')
    : minConf

  const filters: CalibrationFilters = {
    lookbackDays: presetDays(preset),
    category: category === 'all' ? undefined : category,
    tickerPrefix: tickerPrefix.trim() || undefined,
    direction: direction === 'all' ? undefined : direction,
    modelUsed: modelUsed === 'all' ? undefined : modelUsed,
    promptVersion: promptVersion === 'all' ? undefined : promptVersion,
    seriesTicker: seriesTicker === 'all' ? undefined : seriesTicker,
    minConfidence: effectiveMinConf,
    maxConfidence: maxConf !== '' ? Number(maxConf) / 100 : undefined,
  }

  const heatmapFilters: CalibrationHeatmapFilters = {
    lookbackDays: filters.lookbackDays,
    category: filters.category,
    tickerPrefix: filters.tickerPrefix,
    direction: filters.direction,
    modelUsed: filters.modelUsed,
    seriesTicker: filters.seriesTicker,
    minConfidence: filters.minConfidence,
    maxConfidence: filters.maxConfidence,
  }

  const { data, isLoading, error } = useQuery({
    queryKey: ['calibration', filters],
    queryFn: () => getCalibration(filters),
    enabled: activeTab === 'distribution',
  })

  const { data: tsData, isLoading: tsLoading, error: tsError } = useQuery({
    queryKey: ['calibration-time-series', filters],
    queryFn: () => getCalibrationTimeSeries(filters),
    enabled: activeTab === 'over-time',
  })

  const { data: hmData, isLoading: hmLoading, error: hmError } = useQuery({
    queryKey: ['calibration-by-option', heatmapFilters],
    queryFn: () => getCalibrationByOption(heatmapFilters),
    enabled: activeTab === 'by-option',
  })

  // Derive available filter options across all active queries
  const availableCategories = data?.available_categories ?? tsData?.available_categories ?? hmData?.available_categories ?? []
  const availableModels = data?.available_models ?? tsData?.available_models ?? hmData?.available_models ?? []
  const availablePromptVersions = data?.available_prompt_versions ?? tsData?.available_prompt_versions ?? []
  const availableDirections = data?.available_directions ?? tsData?.available_directions ?? hmData?.available_directions ?? []
  const availableSeriesTickers = data?.available_series_tickers ?? tsData?.available_series_tickers ?? hmData?.available_series_tickers ?? []

  // ---------------------------------------------------------------------------
  // Over Time chart data with EMA
  // ---------------------------------------------------------------------------
  const timeChartData = useMemo((): TimeChartPoint[] => {
    if (!tsData) return []
    const N = Number(emaPeriod)
    const alpha = 2 / (N + 1)
    let emaVal: number | null = null
    return tsData.series.map((pt) => {
      const b = pt.brier_score ?? 0
      emaVal = emaVal === null ? b : alpha * b + (1 - alpha) * emaVal
      return {
        ts: new Date(pt.date).getTime(),
        date: pt.date,
        brier: pt.brier_score,
        market_brier: pt.market_brier_score,
        ema: emaVal,
        control: 0.25,
      }
    })
  }, [tsData, emaPeriod])

  // ---------------------------------------------------------------------------
  // Tooltip for distribution
  // ---------------------------------------------------------------------------
  const [tooltip, setTooltip] = useState<TooltipState>(null)
  const W = 1200, H = 480, pad = 60
  const toX = (v: number) => pad + v * (W - pad * 2)
  const toY = (v: number) => H - pad - v * (H - pad * 2)

  // ---------------------------------------------------------------------------
  // Heatmap navigation callback
  // ---------------------------------------------------------------------------
  function navigateToDistribution(overrides: { promptVersion?: string; seriesTicker?: string }) {
    if (overrides.promptVersion) setPromptVersion(overrides.promptVersion)
    if (overrides.seriesTicker) setSeriesTicker(overrides.seriesTicker)
    setActiveTab('distribution')
  }

  const isAnyLoading = isLoading || tsLoading || hmLoading
  const anyError = error || tsError || hmError

  return (
    <div className="page">
      <div className="page-head">
        <div>
          <h1 className="page-title">Calibration</h1>
          <div className="page-subtitle">How well our signals track the world. Lower Brier score → better.</div>
        </div>
        <div style={{ display: 'flex', alignItems: 'flex-end', gap: 10, flexWrap: 'wrap' }}>
          <div style={{ display: 'flex', flexDirection: 'column' }}>
            <label style={{ fontSize: 10.5, textTransform: 'uppercase', letterSpacing: '0.08em', color: 'transparent', marginBottom: 5, userSelect: 'none' }}>Tabs</label>
            <Segmented items={TABS} value={activeTab} onChange={setActiveTab} />
          </div>
          <div style={{ display: 'flex', flexDirection: 'column' }}>
            <label style={{ fontSize: 10.5, textTransform: 'uppercase', letterSpacing: '0.08em', color: 'transparent', marginBottom: 5, userSelect: 'none' }}>Range</label>
            <Segmented items={PRESETS} value={preset} onChange={setPreset} />
          </div>
        </div>
      </div>

      {/* Filter panel */}
      <Panel style={{ marginBottom: 12 }}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr) auto', gap: 12, alignItems: 'end' }}>
          <div className="labeled-field">
            <label className="field-label">Category</label>
            <select className="input select" value={category} onChange={(e) => setCategory(e.target.value)}>
              <option value="all">All</option>
              {availableCategories.map((c) => <option key={c} value={c}>{c}</option>)}
            </select>
          </div>
          <div className="labeled-field">
            <label className="field-label">Series ticker</label>
            <select className="input select" value={seriesTicker} onChange={(e) => setSeriesTicker(e.target.value)}>
              <option value="all">All</option>
              {availableSeriesTickers.map((s) => <option key={s} value={s}>{s}</option>)}
            </select>
          </div>
          <div className="labeled-field">
            <label className="field-label">Ticker prefix</label>
            <input className="input" placeholder="e.g. KXBTC" value={tickerPrefix} onChange={(e) => setTickerPrefix(e.target.value)} />
          </div>
          <div className="labeled-field">
            <label className="field-label">Direction</label>
            <select className="input select" value={direction} onChange={(e) => setDirection(e.target.value)}>
              <option value="all">All</option>
              {availableDirections.map((d) => <option key={d} value={d}>{d}</option>)}
            </select>
          </div>
          <div />
          <div className="labeled-field">
            <label className="field-label">Model</label>
            <select className="input select" value={modelUsed} onChange={(e) => setModelUsed(e.target.value)}>
              <option value="all">All</option>
              {availableModels.map((m) => <option key={m} value={m}>{m}</option>)}
            </select>
          </div>
          {activeTab !== 'by-option' && (
            <div className="labeled-field">
              <label className="field-label">Prompt version</label>
              <select className="input select" value={promptVersion} onChange={(e) => setPromptVersion(e.target.value)}>
                <option value="all">All</option>
                {availablePromptVersions.map((p) => <option key={p} value={p}>{p}</option>)}
              </select>
            </div>
          )}
          <div className="labeled-field">
            <label className="field-label">Min confidence %</label>
            <input className="input" type="number" placeholder="0" min={0} max={100} value={minConfDisplay} onChange={(e) => setMinConf(e.target.value)} />
          </div>
          <div className="labeled-field">
            <label className="field-label">Max confidence %</label>
            <input className="input" type="number" placeholder="100" min={0} max={100} value={maxConf} onChange={(e) => setMaxConf(e.target.value)} />
          </div>
          <button className="btn ghost" onClick={() => {
            setCategory('all'); setSeriesTicker('all'); setTickerPrefix('')
            setDirection('all'); setModelUsed('all'); setPromptVersion('all')
            setMinConf(null); setMaxConf('')
          }}>Reset</button>
        </div>
      </Panel>

      {isAnyLoading && <LoadingSpinner />}
      {anyError && <ErrorBanner message={String(anyError)} />}

      {/* ------------------------------------------------------------------ */}
      {/* Distribution tab                                                    */}
      {/* ------------------------------------------------------------------ */}
      {activeTab === 'distribution' && data && (
        <>
          <div className="grid grid-3" style={{ marginBottom: 12 }}>
            <Stat label="Brier score (ours)" value={data.brier_score.toFixed(4)} sub="lower is better" />
            <Stat label="Brier score (market)" value={data.market_brier_score.toFixed(4)} sub="baseline: market mid at signal time" />
            <Stat label="Samples" value={data.n_samples.toLocaleString()} sub="resolved signals" />
          </div>

          {data.n_samples > 0 ? (
            <Panel title="Calibration curve — estimated probability vs. actual resolution rate">
              <svg width="100%" viewBox={`0 0 ${W} ${H}`} style={{ display: 'block' }}>
                {[0, 0.25, 0.5, 0.75, 1].map((t) => (
                  <g key={t}>
                    <line x1={toX(t)} x2={toX(t)} y1={pad} y2={H - pad} stroke="var(--line-soft)" strokeDasharray="2 4" />
                    <line x1={pad} x2={W - pad} y1={toY(t)} y2={toY(t)} stroke="var(--line-soft)" strokeDasharray="2 4" />
                    <text x={toX(t)} y={H - pad + 18} fontSize="11" fill="var(--fg-2)" textAnchor="middle" fontFamily="var(--f-mono)">{(t * 100).toFixed(0)}%</text>
                    <text x={pad - 10} y={toY(t) + 4} fontSize="11" fill="var(--fg-2)" textAnchor="end" fontFamily="var(--f-mono)">{(t * 100).toFixed(0)}%</text>
                  </g>
                ))}
                <line x1={toX(0)} y1={toY(0)} x2={toX(1)} y2={toY(1)} stroke="var(--fg-3)" strokeDasharray="4 4" strokeWidth="1" />
                {data.market_buckets.filter((b) => b.count > 0).map((b, i) => (
                  <circle key={'m' + i} cx={toX(b.mean_estimated_prob)} cy={toY(b.actual_resolution_rate)}
                    r={4 + Math.sqrt(b.count) * 0.8} fill="var(--warn)" opacity="0.55"
                    style={{ cursor: 'pointer' }}
                    onMouseEnter={() => setTooltip({ x: toX(b.mean_estimated_prob), y: toY(b.actual_resolution_rate), series: 'Market', meanProb: b.mean_estimated_prob, actualRate: b.actual_resolution_rate, count: b.count })}
                    onMouseLeave={() => setTooltip(null)} />
                ))}
                {data.buckets.filter((b) => b.count > 0).map((b, i) => (
                  <circle key={'o' + i} cx={toX(b.mean_estimated_prob)} cy={toY(b.actual_resolution_rate)}
                    r={4 + Math.sqrt(b.count) * 0.8} fill="var(--accent)" opacity="0.8"
                    style={{ cursor: 'pointer' }}
                    onMouseEnter={() => setTooltip({ x: toX(b.mean_estimated_prob), y: toY(b.actual_resolution_rate), series: 'Model', meanProb: b.mean_estimated_prob, actualRate: b.actual_resolution_rate, count: b.count })}
                    onMouseLeave={() => setTooltip(null)} />
                ))}
                {tooltip && (() => {
                  const tw = 210, th = 72, tx = Math.min(tooltip.x + 12, W - tw - 8), ty = Math.max(tooltip.y - th - 8, pad)
                  return (
                    <g pointerEvents="none">
                      <rect x={tx} y={ty} width={tw} height={th} rx={6} fill="var(--bg-1)" stroke="var(--line)" strokeWidth={1} />
                      <text x={tx + 10} y={ty + 18} fontSize="11" fontWeight="600" fill={tooltip.series === 'Model' ? 'var(--accent)' : 'var(--warn)'} fontFamily="var(--f-sans)">{tooltip.series}</text>
                      <text x={tx + 10} y={ty + 34} fontSize="11" fill="var(--fg-2)" fontFamily="var(--f-mono)">Est. prob: <tspan fill="var(--fg-0)">{(tooltip.meanProb * 100).toFixed(1)}%</tspan></text>
                      <text x={tx + 10} y={ty + 50} fontSize="11" fill="var(--fg-2)" fontFamily="var(--f-mono)">Actual rate: <tspan fill="var(--fg-0)">{(tooltip.actualRate * 100).toFixed(1)}%</tspan></text>
                      <text x={tx + 10} y={ty + 66} fontSize="11" fill="var(--fg-2)" fontFamily="var(--f-mono)">Samples: <tspan fill="var(--fg-0)">{tooltip.count}</tspan></text>
                    </g>
                  )
                })()}
                <text x={W / 2} y={H - 16} fontSize="12" fill="var(--fg-2)" textAnchor="middle">Estimated probability</text>
                <text x={18} y={H / 2} fontSize="12" fill="var(--fg-2)" textAnchor="middle" transform={`rotate(-90 18 ${H / 2})`}>Actual resolution rate</text>
                <g transform={`translate(${W - pad - 244},${H - pad - 21})`}>
                  <rect x={-12} y={-16} width={254} height={35} fill="var(--bg-2)" stroke="var(--line)" rx={6} />
                  <line x1={0} y1={0} x2={18} y2={0} stroke="var(--fg-3)" strokeDasharray="4 4" />
                  <text x={24} y={4} fontSize="11" fill="var(--fg-2)">Perfect calibration</text>
                  <circle cx={134} cy={0} r={4} fill="var(--accent)" />
                  <text x={144} y={4} fontSize="11" fill="var(--fg-1)">Model</text>
                  <circle cx={184} cy={0} r={4} fill="var(--warn)" />
                  <text x={194} y={4} fontSize="11" fill="var(--fg-1)">Market</text>
                </g>
              </svg>
              <div style={{ textAlign: 'center', color: 'var(--fg-3)', fontSize: 11, marginTop: 8 }}>
                Bubble size proportional to sample count
              </div>
            </Panel>
          ) : (
            <div className="panel" style={{ padding: '40px', textAlign: 'center', color: 'var(--fg-3)' }}>
              No resolved signals yet — calibration chart will appear once markets resolve.
            </div>
          )}
        </>
      )}

      {/* ------------------------------------------------------------------ */}
      {/* Over Time tab                                                        */}
      {/* ------------------------------------------------------------------ */}
      {activeTab === 'over-time' && tsData && (
        <>
          <div style={{ display: 'flex', justifyContent: 'flex-end', marginBottom: 8 }}>
            <Segmented items={EMA_PERIODS} value={emaPeriod} onChange={setEmaPeriod} />
          </div>

          {timeChartData.length === 0 ? (
            <Panel>
              <div style={{ padding: 32, textAlign: 'center', color: 'var(--fg-3)' }}>
                No resolved signals in this window.
              </div>
            </Panel>
          ) : (
            <Panel flush>
              <ResponsiveContainer width="100%" height={380}>
                <ComposedChart data={timeChartData} margin={{ top: 8, right: 16, bottom: 8, left: 8 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--line-soft, #e5e7eb)" />
                  <XAxis
                    dataKey="ts"
                    type="number"
                    scale="time"
                    domain={['dataMin', 'dataMax']}
                    tickFormatter={fmtDate}
                    tick={{ fontSize: 11 }}
                  />
                  <YAxis
                    tickFormatter={(v) => v.toFixed(3)}
                    tick={{ fontSize: 11 }}
                    width={60}
                    domain={[0, 'auto']}
                    reversed
                  />
                  <Tooltip content={<TimeTooltip />} />
                  <Legend wrapperStyle={{ fontSize: 12 }} />

                  {tsData.prompt_version_starts.map((pv, i) => (
                    <ReferenceLine
                      key={pv.version}
                      x={new Date(pv.date).getTime()}
                      stroke="var(--fg-0)"
                      strokeWidth={1}
                      strokeDasharray="3 3"
                      label={{
                        value: `v${pv.version}`,
                        position: 'insideTopLeft',
                        fontSize: 10,
                        fill: 'var(--fg-0)',
                        dy: i % 2 === 0 ? 2 : 14,
                      }}
                    />
                  ))}

                  <Bar dataKey="brier" name="Daily Brier (model)" maxBarSize={16} fill="var(--accent, #6366f1)" fillOpacity={0.65} />
                  <Line
                    dataKey="market_brier"
                    name="Daily Brier (market)"
                    stroke="var(--warn, #f59e0b)"
                    strokeWidth={1.5}
                    dot={false}
                    connectNulls
                  />
                  <Line
                    dataKey="ema"
                    name={`${emaPeriod}d EMA`}
                    stroke="var(--accent, #6366f1)"
                    strokeWidth={2}
                    strokeDasharray="4 2"
                    dot={false}
                    connectNulls
                  />
                  <Line
                    dataKey="control"
                    name="Control (0.25)"
                    stroke="var(--fg-3, #6b7280)"
                    strokeWidth={1}
                    strokeDasharray="4 2"
                    dot={false}
                  />
                </ComposedChart>
              </ResponsiveContainer>
            </Panel>
          )}
        </>
      )}

      {/* ------------------------------------------------------------------ */}
      {/* By Option tab                                                        */}
      {/* ------------------------------------------------------------------ */}
      {activeTab === 'by-option' && hmData && (
        <>
          {hmData.rows.length <= 1 && hmData.prompt_versions.length === 0 ? (
            <Panel>
              <div style={{ padding: 32, textAlign: 'center', color: 'var(--fg-3)' }}>
                No data — heatmap requires resolved signals from markets with a series ticker.
              </div>
            </Panel>
          ) : (
            <Panel title="Brier score by option × prompt version  (green = model beats market)">
              <HeatmapTable data={hmData} onNavigate={navigateToDistribution} />
            </Panel>
          )}
        </>
      )}
    </div>
  )
}
