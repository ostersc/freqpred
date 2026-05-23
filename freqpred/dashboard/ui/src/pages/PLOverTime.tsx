import React, { useMemo, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import {
  Area,
  Bar,
  CartesianGrid,
  Cell,
  ComposedChart,
  Legend,
  Line,
  ReferenceArea,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { getPnLTimeSeries } from '../api/pnl'
import type { PnLFilters, PromptVersionStart } from '../api/pnl'
import {
  ErrorBanner,
  LabeledInput,
  LabeledSelect,
  LoadingSpinner,
  Panel,
  Segmented,
  Stat,
  fmtMoney,
  fmtSignedMoney,
  fmtSignedPct,
} from '../components/ui'

// ---------------------------------------------------------------------------
// Types & helpers
// ---------------------------------------------------------------------------

type Preset = '7' | '30' | '90' | 'all'
type TabId = 'chart' | 'projection'
type EMAPeriod = '7' | '14' | '30'

const PRESETS: { v: Preset; label: string }[] = [
  { v: '7',   label: '7d' },
  { v: '30',  label: '30d' },
  { v: '90',  label: '90d' },
  { v: 'all', label: 'All time' },
]
const EMA_PERIODS: { v: EMAPeriod; label: string }[] = [
  { v: '7',  label: '7d EMA' },
  { v: '14', label: '14d EMA' },
  { v: '30', label: '30d EMA' },
]
const TABS: { v: TabId; label: string }[] = [
  { v: 'chart',      label: 'History' },
  { v: 'projection', label: 'Projection' },
]

function presetDays(p: Preset): number | undefined {
  return p === 'all' ? undefined : Number(p)
}

function fmtDate(ts: number) {
  return new Date(ts).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
}

function fmtDateFull(ts: number) {
  return new Date(ts).toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: '2-digit' })
}

// ---------------------------------------------------------------------------
// Projection math (pure functions, no side-effects)
// ---------------------------------------------------------------------------

function linearTrend(values: number[]): { slope: number; intercept: number } {
  const n = values.length
  if (n < 2) return { slope: 0, intercept: values[0] ?? 0 }
  const xBar = (n - 1) / 2
  const yBar = values.reduce((a, b) => a + b, 0) / n
  let num = 0
  let den = 0
  for (let i = 0; i < n; i++) {
    num += (i - xBar) * (values[i] - yBar)
    den += (i - xBar) ** 2
  }
  const slope = den === 0 ? 0 : num / den
  return { slope, intercept: yBar - slope * xBar }
}

function computeGBMParams(bankrollSeries: number[]): { mu: number; sigma: number } | null {
  const logReturns: number[] = []
  for (let i = 1; i < bankrollSeries.length; i++) {
    if (bankrollSeries[i] > 0 && bankrollSeries[i - 1] > 0) {
      logReturns.push(Math.log(bankrollSeries[i] / bankrollSeries[i - 1]))
    }
  }
  if (logReturns.length < 2) return null
  const mu = logReturns.reduce((a, b) => a + b, 0) / logReturns.length
  const variance = logReturns.reduce((a, r) => a + (r - mu) ** 2, 0) / (logReturns.length - 1)
  return { mu, sigma: Math.sqrt(variance) }
}

// ---------------------------------------------------------------------------
// Chart data types
// ---------------------------------------------------------------------------

interface ChartPoint {
  ts: number
  date: string
  daily_pnl: number | null
  cumulative_pnl: number | null
  daily_spend: number | null
  cumulative_spend: number | null
  ema: number | null
}

interface ProjPoint {
  ts: number
  date: string
  hist_bankroll: number | null
  hist_llm_spend: number | null
  proj_central: number | null
  proj_1s_lo: number | null
  proj_1s_spread: number | null
  proj_2s_lo: number | null
  proj_2s_spread: number | null
  proj_llm_spend: number | null
}

// ---------------------------------------------------------------------------
// Custom tooltip
// ---------------------------------------------------------------------------

function HistoryTooltip({ active, payload, label }: {
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
            {p.name}: {p.name.includes('Spend') ? fmtMoney(p.value, 4) : fmtSignedMoney(p.value)}
          </div>
        ) : null
      )}
    </div>
  )
}

function ProjTooltip({ active, payload, label }: {
  active?: boolean
  payload?: { name: string; value: number | null; color: string; payload: ProjPoint }[]
  label?: number
}) {
  if (!active || !payload?.length) return null
  const pt = payload[0].payload
  return (
    <div className="tooltip-box">
      <div className="tooltip-title">{label != null ? fmtDateFull(label) : ''}</div>
      {pt.hist_bankroll != null && (
        <div style={{ fontSize: 12 }}>Bankroll: {fmtSignedMoney(pt.hist_bankroll)}</div>
      )}
      {pt.proj_central != null && (
        <div style={{ fontSize: 12 }}>Projection: {fmtSignedMoney(pt.proj_central)}</div>
      )}
      {pt.proj_1s_lo != null && pt.proj_1s_spread != null && (
        <div style={{ fontSize: 12, opacity: 0.75 }}>
          ±1σ: {fmtSignedMoney(pt.proj_1s_lo)} – {fmtSignedMoney(pt.proj_1s_lo + pt.proj_1s_spread)}
        </div>
      )}
      {pt.proj_2s_lo != null && pt.proj_2s_spread != null && (
        <div style={{ fontSize: 12, opacity: 0.55 }}>
          ±2σ: {fmtSignedMoney(pt.proj_2s_lo)} – {fmtSignedMoney(pt.proj_2s_lo + pt.proj_2s_spread)}
        </div>
      )}
      {pt.hist_llm_spend != null && (
        <div style={{ color: '#dc2626', fontSize: 12 }}>LLM spend: {fmtMoney(pt.hist_llm_spend, 4)}</div>
      )}
      {pt.proj_llm_spend != null && (
        <div style={{ color: '#dc2626', fontSize: 12, opacity: 0.7 }}>LLM spend (proj): {fmtMoney(pt.proj_llm_spend, 4)}</div>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Main page
// ---------------------------------------------------------------------------

export default function PLOverTime() {
  const [preset, setPreset] = useState<Preset>('all')
  const [activeTab, setActiveTab] = useState<TabId>('chart')
  const [emaPeriod, setEmaPeriod] = useState<EMAPeriod>('7')

  // Filters
  const [strategy, setStrategy] = useState('')
  const [modelUsed, setModelUsed] = useState('')
  const [promptVersion, setPromptVersion] = useState('')
  const [direction, setDirection] = useState('')
  const [category, setCategory] = useState('')
  const [seriesTicker, setSeriesTicker] = useState('')
  const [marketId, setMarketId] = useState('')

  const filters: PnLFilters = {
    lookbackDays: presetDays(preset),
    strategy: strategy || undefined,
    modelUsed: modelUsed || undefined,
    promptVersion: promptVersion || undefined,
    direction: direction || undefined,
    category: category || undefined,
    seriesTicker: seriesTicker || undefined,
    marketId: marketId || undefined,
  }

  const { data, isLoading, error } = useQuery({
    queryKey: ['pnl-time-series', filters],
    queryFn: () => getPnLTimeSeries(filters),
    staleTime: 60_000,
  })

  // ---------------------------------------------------------------------------
  // Merge pnl_series + llm_series into unified ChartPoint[] with EMA
  // ---------------------------------------------------------------------------
  const chartData = useMemo((): ChartPoint[] => {
    if (!data) return []
    const byDate = new Map<string, Partial<ChartPoint>>()

    for (const d of data.pnl_series) {
      byDate.set(d.date, {
        date: d.date,
        ts: new Date(d.date).getTime(),
        daily_pnl: d.daily_pnl,
        cumulative_pnl: d.cumulative_pnl,
      })
    }
    for (const d of data.llm_series) {
      const existing = byDate.get(d.date) ?? { date: d.date, ts: new Date(d.date).getTime() }
      byDate.set(d.date, {
        ...existing,
        daily_spend: d.daily_spend,
        cumulative_spend: d.cumulative_spend,
      })
    }

    const sorted = Array.from(byDate.values())
      .sort((a, b) => (a.ts ?? 0) - (b.ts ?? 0)) as ChartPoint[]

    // EMA over daily_pnl (null gaps treated as 0 — a day with no closed trades)
    const N = Number(emaPeriod)
    const alpha = 2 / (N + 1)
    let emaVal: number | null = null
    for (const pt of sorted) {
      const pnl = pt.daily_pnl ?? 0
      emaVal = emaVal === null ? pnl : alpha * pnl + (1 - alpha) * emaVal
      pt.ema = emaVal
    }

    // Fill null fields
    for (const pt of sorted) {
      if (pt.daily_pnl === undefined) pt.daily_pnl = null
      if (pt.cumulative_pnl === undefined) pt.cumulative_pnl = null
      if (pt.daily_spend === undefined) pt.daily_spend = null
      if (pt.cumulative_spend === undefined) pt.cumulative_spend = null
    }

    return sorted
  }, [data, emaPeriod])

  // Right axis for LLM spend: 0 to 2× the max absolute P&L value, matching the left axis magnitude.
  const llmSpendDomain = useMemo((): [number, number] => {
    const maxAbs = Math.max(
      1,
      ...chartData.flatMap(d => [Math.abs(d.daily_pnl ?? 0), Math.abs(d.cumulative_pnl ?? 0)]),
    )
    return [0, maxAbs * 2]
  }, [chartData])

  // Where y=0 falls as % from the TOP of the Area's bounding box.
  // SVG gradient top-to-bottom (0%=peak, 100%=trough). Green above 0, red below.
  const gradientZeroPct = useMemo(() => {
    const vals = chartData.map(d => d.cumulative_pnl).filter((v): v is number => v != null)
    if (vals.length === 0) return '0%'
    const yMax = Math.max(0, ...vals)
    const yMin = Math.min(0, ...vals)
    if (yMax === 0) return '0%'    // all at or below zero: fully red
    if (yMin === 0) return '100%'  // all at or above zero: fully green
    return `${((yMax / (yMax - yMin)) * 100).toFixed(2)}%`
  }, [chartData])

  // ---------------------------------------------------------------------------
  // Projection math
  // ---------------------------------------------------------------------------
  const projection = useMemo(() => {
    if (!data || data.pnl_series.length === 0) return null

    const dailyLlmValues = data.llm_series.map((d) => d.daily_spend)
    const llmTrend = linearTrend(dailyLlmValues.length > 0 ? dailyLlmValues : [0])

    const lastPnlPoint = data.pnl_series[data.pnl_series.length - 1]
    const lastLlmPoint = data.llm_series[data.llm_series.length - 1]
    const lastCumPnl = lastPnlPoint.cumulative_pnl
    const lastCumLlm = lastLlmPoint?.cumulative_spend ?? 0

    const initialBankroll = data.initial_bankroll
    const netBankrollNow = initialBankroll + lastCumPnl - lastCumLlm

    // GBM parameters estimated from the P&L bankroll series (log-returns)
    const bankrollSeries = data.pnl_series.map((d) => initialBankroll + d.cumulative_pnl)
    const gbm = computeGBMParams(bankrollSeries)

    // CAGR = geometric mean daily log-return annualized (e^(μ·365) − 1)
    const cagr = gbm ? Math.exp(gbm.mu * 365) - 1 : null

    // Days until broke: GBM central path minus projected LLM spend
    const daysUntilBroke = (() => {
      if (netBankrollNow <= 0) return 0
      if (!gbm) return null
      let addlLlm = 0
      for (let d = 1; d <= 3650; d++) {
        const projBankroll = netBankrollNow * Math.exp(gbm.mu * d)
        addlLlm += Math.max(0, llmTrend.intercept + llmTrend.slope * (dailyLlmValues.length - 1 + d))
        if (projBankroll - addlLlm <= 0) return d
      }
      return null
    })()

    // Projected 30-day LLM burn (linear trend, unchanged)
    let proj30LlmBurn = 0
    for (let d = 1; d <= 30; d++) {
      proj30LlmBurn += Math.max(0, llmTrend.intercept + llmTrend.slope * (dailyLlmValues.length - 1 + d))
    }

    // Build projection chart points: history + forward window sized to the selected preset
    const HISTORY_DAYS = 90
    const FORWARD_DAYS = preset === 'all' ? data.pnl_series.length : Number(preset)
    const today = new Date()
    today.setHours(0, 0, 0, 0)
    const todayTs = today.getTime()
    const msPerDay = 24 * 3600 * 1000

    const projPoints: ProjPoint[] = []

    // Historical bankroll curve (last HISTORY_DAYS of data).
    const histSlice = chartData.slice(-HISTORY_DAYS)
    for (const pt of histSlice) {
      projPoints.push({
        ts: pt.ts,
        date: pt.date,
        hist_bankroll: pt.cumulative_pnl != null
          ? Math.max(0, initialBankroll + pt.cumulative_pnl)
          : null,
        hist_llm_spend: pt.cumulative_spend ?? null,
        proj_central: null,
        proj_1s_lo: null,
        proj_1s_spread: null,
        proj_2s_lo: null,
        proj_2s_spread: null,
        proj_llm_spend: null,
      })
    }

    // GBM fan projection forward
    if (gbm && netBankrollNow > 0) {
      let fwdCumLlm = lastCumLlm
      for (let d = 1; d <= FORWARD_DAYS; d++) {
        const sqrtD = Math.sqrt(d)
        const central = netBankrollNow * Math.exp(gbm.mu * d)
        const lo1 = netBankrollNow * Math.exp(gbm.mu * d - gbm.sigma * sqrtD)
        const hi1 = netBankrollNow * Math.exp(gbm.mu * d + gbm.sigma * sqrtD)
        const lo2 = netBankrollNow * Math.exp(gbm.mu * d - 2 * gbm.sigma * sqrtD)
        const hi2 = netBankrollNow * Math.exp(gbm.mu * d + 2 * gbm.sigma * sqrtD)
        fwdCumLlm += Math.max(0, llmTrend.intercept + llmTrend.slope * (dailyLlmValues.length - 1 + d))
        const lo2c = Math.max(0, lo2)
        const lo1c = Math.max(0, lo1)
        const ts = todayTs + d * msPerDay
        projPoints.push({
          ts,
          date: new Date(ts).toISOString().slice(0, 10),
          hist_bankroll: null,
          hist_llm_spend: null,
          proj_central: Math.max(0, central),
          proj_1s_lo: lo1c,
          proj_1s_spread: Math.max(0, Math.max(0, hi1) - lo1c),
          proj_2s_lo: lo2c,
          proj_2s_spread: Math.max(0, Math.max(0, hi2) - lo2c),
          proj_llm_spend: fwdCumLlm,
        })
      }
    }

    // Bridge history→projection: pin the fan to the last known bankroll value.
    const lastHistWithData = [...projPoints].reverse().find(p => p.hist_bankroll != null)
    if (lastHistWithData && gbm && netBankrollNow > 0) {
      const b = lastHistWithData.hist_bankroll!
      lastHistWithData.proj_central = b
      lastHistWithData.proj_1s_lo = b
      lastHistWithData.proj_1s_spread = 0
      lastHistWithData.proj_2s_lo = b
      lastHistWithData.proj_2s_spread = 0
      lastHistWithData.proj_llm_spend = lastHistWithData.hist_llm_spend
    }

    // Gradient stop from TOP (0%=hi, 100%=lo)
    function bankrollGradStop(vals: (number | null)[], ref: number): string {
      const nums = vals.filter((v): v is number => v != null)
      if (nums.length === 0) return '0%'
      const hi = Math.max(ref, ...nums)
      const lo = Math.min(ref, ...nums)
      if (hi === lo) return '50%'
      return `${(((hi - ref) / (hi - lo)) * 100).toFixed(2)}%`
    }
    const histBankrollGradPct = bankrollGradStop(projPoints.map(p => p.hist_bankroll), initialBankroll)
    const projBankrollGradPct = bankrollGradStop(projPoints.map(p => p.proj_central), initialBankroll)

    return {
      netBankrollNow,
      cagr,
      gbm,
      proj30LlmBurn,
      daysUntilBroke,
      projPoints,
      todayTs,
      histBankrollGradPct,
      projBankrollGradPct,
      FORWARD_DAYS,
    }
  }, [data, chartData, preset])

  // ---------------------------------------------------------------------------
  // Filter option builders
  // ---------------------------------------------------------------------------
  const strategyOpts = [
    { value: '', label: 'All' },
    ...(data?.available_strategies ?? []).map((s) => ({ value: s, label: s })),
  ]
  const modelOpts = [
    { value: '', label: 'All' },
    ...(data?.available_models ?? []).map((m) => ({ value: m, label: m })),
  ]
  const promptOpts = [
    { value: '', label: 'All' },
    ...(data?.available_prompt_versions ?? []).map((v) => ({ value: v, label: v })),
  ]
  const directionOpts = [
    { value: '', label: 'All' },
    ...(data?.available_directions ?? []).map((d) => ({ value: d, label: d })),
  ]
  const categoryOpts = [
    { value: '', label: 'All' },
    ...(data?.available_categories ?? []).map((c) => ({ value: c, label: c })),
  ]
  const seriesOpts = [
    { value: '', label: 'All' },
    ...(data?.available_series_tickers ?? []).map((s) => ({ value: s, label: s })),
  ]

  function resetFilters() {
    setStrategy('')
    setModelUsed('')
    setPromptVersion('')
    setDirection('')
    setCategory('')
    setSeriesTicker('')
    setMarketId('')
  }

  // ---------------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------------
  const hasFilters = !!(strategy || modelUsed || promptVersion || direction || category || seriesTicker || marketId)

  return (
    <div className="page">
      <div className="page-head">
        <h1 className="page-title">P&amp;L History</h1>
        <div style={{ display: 'flex', alignItems: 'flex-end', gap: 10 }}>
          <Segmented items={TABS} value={activeTab} onChange={setActiveTab} />
          <Segmented items={PRESETS} value={preset} onChange={setPreset} />
        </div>
      </div>

      {/* Filter panel */}
      <Panel style={{ marginBottom: 12 }}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr) auto', gap: 12, alignItems: 'end' }}>
          <LabeledSelect label="Strategy" value={strategy} onChange={setStrategy} options={strategyOpts} />
          <LabeledSelect label="Direction" value={direction} onChange={setDirection} options={directionOpts} />
          <LabeledSelect label="Category" value={category} onChange={setCategory} options={categoryOpts} />
          <LabeledSelect label="Series" value={seriesTicker} onChange={setSeriesTicker} options={seriesOpts} />
          <div />
          <LabeledSelect label="Model" value={modelUsed} onChange={setModelUsed} options={modelOpts} />
          <LabeledSelect label="Prompt version" value={promptVersion} onChange={setPromptVersion} options={promptOpts} />
          <LabeledInput label="Market ID" placeholder="e.g. KXBTC-25JUN-T30000" value={marketId} onChange={setMarketId} />
          <div />
          <button className="btn ghost" onClick={resetFilters} disabled={!hasFilters} style={{ alignSelf: 'end' }}>
            Reset
          </button>
        </div>
      </Panel>

      {isLoading && <LoadingSpinner />}
      {error && <ErrorBanner message={String(error)} />}

      {data && activeTab === 'chart' && (
        <>
          {/* Summary stats */}
          <div className="grid grid-4" style={{ marginBottom: 12 }}>
            <Stat
              label="Total trades"
              value={
                <>
                  {data.total_trades}
                  {data.total_trades > 0 && (
                    <span style={{ fontSize: '0.65em', marginLeft: 7, opacity: 0.75 }}
                          className={data.all_time_pnl >= 0 ? 'pos' : 'neg'}>
                      ({fmtSignedMoney(data.all_time_pnl / data.total_trades)} avg)
                    </span>
                  )}
                </>
              }
            />
            <Stat
              label="All-time P&L"
              value={
                <span className={data.all_time_pnl >= 0 ? 'pos' : 'neg'}>
                  {fmtSignedMoney(data.all_time_pnl)}
                  {data.initial_bankroll > 0 && (
                    <span style={{ fontSize: '0.65em', marginLeft: 7, opacity: 0.75 }}>
                      ({fmtSignedPct(data.all_time_pnl / data.initial_bankroll)})
                    </span>
                  )}
                </span>
              }
            />
            <Stat label="Initial bankroll" value={fmtMoney(data.initial_bankroll)} />
            <Stat
              label="Net bankroll"
              value={
                <span className={data.all_time_pnl >= 0 ? 'pos' : 'neg'}>
                  {fmtMoney(data.initial_bankroll + data.all_time_pnl)}
                  {data.initial_bankroll > 0 && (
                    <span style={{ fontSize: '0.65em', marginLeft: 7, opacity: 0.75 }}>
                      ({fmtSignedPct(data.all_time_pnl / data.initial_bankroll)})
                    </span>
                  )}
                </span>
              }
            />
          </div>

          {/* EMA period toggle */}
          <div style={{ display: 'flex', justifyContent: 'flex-end', marginBottom: 8 }}>
            <Segmented items={EMA_PERIODS} value={emaPeriod} onChange={setEmaPeriod} />
          </div>

          {chartData.length === 0 ? (
            <Panel><div style={{ padding: 32, textAlign: 'center', color: 'var(--text-3)' }}>No closed positions in this window.</div></Panel>
          ) : (
            <Panel flush>
              <ResponsiveContainer width="100%" height={380}>
                <ComposedChart data={chartData} margin={{ top: 8, right: 60, bottom: 8, left: 8 }}>
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
                    yAxisId="pnl"
                    orientation="left"
                    tickFormatter={(v) => fmtSignedMoney(v)}
                    tick={{ fontSize: 11 }}
                    width={72}
                  />
                  <YAxis
                    yAxisId="spend"
                    orientation="right"
                    tickFormatter={(v) => fmtMoney(v)}
                    tick={{ fontSize: 11 }}
                    width={72}
                    domain={llmSpendDomain}
                  />
                  <Tooltip content={<HistoryTooltip />} />
                  <Legend wrapperStyle={{ fontSize: 12 }} />
                  {((<defs>
                    <linearGradient id="pnlAreaGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%"             stopColor="#22c55e" stopOpacity={0.6} />
                      <stop offset={gradientZeroPct} stopColor="#22c55e" stopOpacity={0.04} />
                      <stop offset={gradientZeroPct} stopColor="#ef4444" stopOpacity={0.04} />
                      <stop offset="100%"            stopColor="#ef4444" stopOpacity={0.6} />
                    </linearGradient>
                    <linearGradient id="pnlLineGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset={gradientZeroPct} stopColor="#22c55e" />
                      <stop offset={gradientZeroPct} stopColor="#ef4444" />
                    </linearGradient>
                  </defs>) as React.ReactNode)}
                  <ReferenceLine yAxisId="pnl" y={0} stroke="var(--line, #9ca3af)" strokeWidth={1} />

                  {data.prompt_version_starts.map((pv: PromptVersionStart, i: number) => (
                    <ReferenceLine
                      key={pv.version}
                      yAxisId="pnl"
                      x={new Date(pv.date).getTime()}
                      stroke="var(--accent, #6366f1)"
                      strokeWidth={1}
                      strokeDasharray="3 3"
                      label={{
                        value: `v${pv.version}`,
                        position: 'insideTopLeft',
                        fontSize: 10,
                        fill: 'var(--accent, #6366f1)',
                        dy: i % 2 === 0 ? 2 : 14,
                      }}
                    />
                  ))}

                  <Bar yAxisId="pnl" dataKey="daily_pnl" name="Daily P&L" maxBarSize={16} fill="#22c55e" fillOpacity={0.7}>
                    {chartData.map((pt, i) => (
                      <Cell key={i} fill={(pt.daily_pnl ?? 0) >= 0 ? '#22c55e' : '#ef4444'} fillOpacity={0.7} />
                    ))}
                  </Bar>
                  <Area
                    yAxisId="pnl"
                    dataKey="cumulative_pnl"
                    name="Cumulative P&L"
                    baseValue={0}
                    fill="url(#pnlAreaGrad)"
                    stroke="url(#pnlLineGrad)"
                    strokeWidth={1.5}
                    dot={false}
                    connectNulls
                  />
                  <Line
                    yAxisId="pnl"
                    dataKey="ema"
                    name={`${emaPeriod}d EMA`}
                    stroke="#f97316"
                    strokeWidth={1.5}
                    strokeDasharray="4 2"
                    dot={false}
                    connectNulls
                  />
                  <Line
                    yAxisId="spend"
                    dataKey="cumulative_spend"
                    name="Cumulative LLM Spend"
                    stroke="#dc2626"
                    strokeWidth={1.5}
                    dot={false}
                    connectNulls
                  />
                </ComposedChart>
              </ResponsiveContainer>
            </Panel>
          )}
        </>
      )}

      {data && activeTab === 'projection' && projection && (
        <>
          {/* Projection summary stats */}
          <div className="grid grid-4" style={{ marginBottom: 12 }}>
            <Stat
              label="Net bankroll now"
              value={(() => {
                const change = projection.netBankrollNow - data.initial_bankroll
                const isPos = change >= 0
                return (
                  <span className={isPos ? 'pos' : 'neg'}>
                    {fmtMoney(projection.netBankrollNow)}
                    {data.initial_bankroll > 0 && (
                      <span style={{ fontSize: '0.65em', marginLeft: 7, opacity: 0.75 }}>
                        ({fmtSignedPct(change / data.initial_bankroll)})
                      </span>
                    )}
                  </span>
                )
              })()}
            />
            <Stat
              label="Proj. CAGR"
              value={
                projection.cagr != null
                  ? <span className={projection.cagr >= 0 ? 'pos' : 'neg'}>{fmtSignedPct(projection.cagr)}</span>
                  : 'N/A'
              }
              sub={
                projection.cagr == null
                  ? 'insufficient data'
                  : `GBM μ=${(projection.gbm!.mu * 100).toFixed(3)}% σ=${(projection.gbm!.sigma * 100).toFixed(3)}%/d`
              }
            />
            <Stat
              label="Proj. 30d LLM burn"
              value={fmtMoney(projection.proj30LlmBurn, 4)}
            />
            <Stat
              label="Days until broke"
              value={projection.daysUntilBroke == null ? '∞' : String(projection.daysUntilBroke)}
              sub={projection.daysUntilBroke == null ? 'GBM central path positive' : 'GBM central path'}
              deltaKind={projection.daysUntilBroke == null ? 'pos' : projection.daysUntilBroke < 90 ? 'neg' : 'warn'}
            />
          </div>

          {/* Projection chart */}
          <Panel flush>
            <ResponsiveContainer width="100%" height={380}>
              <ComposedChart data={projection.projPoints} margin={{ top: 28, right: 16, bottom: 8, left: 8 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--line-soft, #e5e7eb)" />
                <XAxis
                  dataKey="ts"
                  type="number"
                  scale="time"
                  domain={['dataMin', 'dataMax']}
                  tickFormatter={fmtDate}
                  tick={{ fontSize: 11 }}
                />
                <YAxis tickFormatter={(v) => fmtSignedMoney(v)} tick={{ fontSize: 11 }} width={72} domain={[0, 'auto']} />
                <Tooltip content={<ProjTooltip />} />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                {((<defs>
                  <linearGradient id="histBankrollAreaGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%"                            stopColor="#22c55e" stopOpacity={0.6} />
                    <stop offset={projection.histBankrollGradPct} stopColor="#22c55e" stopOpacity={0.04} />
                    <stop offset={projection.histBankrollGradPct} stopColor="#ef4444" stopOpacity={0.04} />
                    <stop offset="100%"                          stopColor="#ef4444" stopOpacity={0.6} />
                  </linearGradient>
                  <linearGradient id="histBankrollLineGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset={projection.histBankrollGradPct} stopColor="#22c55e" />
                    <stop offset={projection.histBankrollGradPct} stopColor="#ef4444" />
                  </linearGradient>
                  <linearGradient id="projBankrollLineGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset={projection.projBankrollGradPct} stopColor="#22c55e" />
                    <stop offset={projection.projBankrollGradPct} stopColor="#ef4444" />
                  </linearGradient>
                </defs>) as React.ReactNode)}

                {/* Shaded projection zone */}
                <ReferenceArea
                  x1={projection.todayTs}
                  x2={projection.projPoints[projection.projPoints.length - 1]?.ts}
                  fill="var(--accent, #3b82f6)"
                  fillOpacity={0.05}
                />
                <ReferenceLine
                  x={projection.todayTs}
                  stroke="var(--text-3, #9ca3af)"
                  strokeDasharray="4 2"
                  label={{ value: 'Today', position: 'top', fontSize: 11 }}
                />
                <ReferenceLine y={0} stroke="var(--neg, #ef4444)" strokeWidth={1} />

                <Area
                  dataKey="hist_bankroll"
                  name="Bankroll (history)"
                  baseValue={data.initial_bankroll}
                  fill="url(#histBankrollAreaGrad)"
                  stroke="url(#histBankrollLineGrad)"
                  strokeWidth={2}
                  dot={false}
                  connectNulls
                />
                {/* ±2σ fan band — stacked: transparent base + visible spread */}
                <Area type="monotone" stackId="band2" dataKey="proj_2s_lo"     fill="transparent" stroke="none" legendType="none" dot={false} />
                <Area type="monotone" stackId="band2" dataKey="proj_2s_spread" fill="#3b82f6" fillOpacity={0.08} stroke="none" name="±2σ band" dot={false} />
                {/* ±1σ fan band */}
                <Area type="monotone" stackId="band1" dataKey="proj_1s_lo"     fill="transparent" stroke="none" legendType="none" dot={false} />
                <Area type="monotone" stackId="band1" dataKey="proj_1s_spread" fill="#3b82f6" fillOpacity={0.16} stroke="none" name="±1σ band" dot={false} />
                {/* GBM central path */}
                <Line
                  type="monotone"
                  dataKey="proj_central"
                  name="Projection"
                  stroke="url(#projBankrollLineGrad)"
                  strokeWidth={1.5}
                  strokeDasharray="5 3"
                  dot={false}
                  connectNulls
                />
                <Line
                  dataKey="hist_llm_spend"
                  name="LLM spend (history)"
                  stroke="#dc2626"
                  strokeWidth={2}
                  dot={false}
                  connectNulls
                />
                <Line
                  dataKey="proj_llm_spend"
                  name="LLM spend (projected)"
                  stroke="#dc2626"
                  strokeWidth={1.5}
                  strokeDasharray="5 3"
                  dot={false}
                  connectNulls
                />
              </ComposedChart>
            </ResponsiveContainer>
          </Panel>
        </>
      )}

      {data && activeTab === 'projection' && !projection && (
        <Panel>
          <div style={{ padding: 32, textAlign: 'center', color: 'var(--text-3)' }}>
            No closed positions yet — projection requires at least one trade.
          </div>
        </Panel>
      )}
    </div>
  )
}

