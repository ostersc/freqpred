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

export const MS_PER_DAY = 24 * 3600 * 1000

/**
 * Least-squares fit of `value` against elapsed CALENDAR days since the first point.
 *
 * Regressing on the array index instead would treat a gap day (no LLM rows at
 * all) as one step, compressing the x-axis and biasing the extrapolated slope.
 */
export function linearTrendByDay(
  points: { ts: number; value: number }[],
): { slope: number; intercept: number } {
  const n = points.length
  if (n === 0) return { slope: 0, intercept: 0 }
  if (n === 1) return { slope: 0, intercept: points[0].value }
  const t0 = points[0].ts
  const xs = points.map((p) => (p.ts - t0) / MS_PER_DAY)
  const xBar = xs.reduce((a, b) => a + b, 0) / n
  const yBar = points.reduce((a, p) => a + p.value, 0) / n
  let num = 0
  let den = 0
  for (let i = 0; i < n; i++) {
    num += (xs[i] - xBar) * (points[i].value - yBar)
    den += (xs[i] - xBar) ** 2
  }
  const slope = den === 0 ? 0 : num / den
  return { slope, intercept: yBar - slope * xBar }
}

/**
 * GBM drift and volatility expressed per CALENDAR day.
 *
 * `pnl_series` only carries a row on days that closed a position, so the raw
 * per-observation moments are on a "per closing day" clock while the projection
 * advances one calendar day at a time — which overstated both drift and vol by
 * the ratio between the two. Rescale by observations-per-calendar-day: drift is
 * linear in time, volatility scales with its square root.
 */
export function computeGBMParams(
  points: { ts: number; bankroll: number }[],
): { mu: number; sigma: number } | null {
  const logReturns: number[] = []
  for (let i = 1; i < points.length; i++) {
    const prev = points[i - 1].bankroll
    const cur = points[i].bankroll
    if (cur > 0 && prev > 0) logReturns.push(Math.log(cur / prev))
  }
  if (logReturns.length < 2) return null

  const muPerObs = logReturns.reduce((a, b) => a + b, 0) / logReturns.length
  const variance =
    logReturns.reduce((a, r) => a + (r - muPerObs) ** 2, 0) / (logReturns.length - 1)
  const sigmaPerObs = Math.sqrt(variance)

  const spanDays = (points[points.length - 1].ts - points[0].ts) / MS_PER_DAY
  // A sub-day span would blow the rescale up; leave the moments unscaled.
  const obsPerDay = spanDays >= 1 ? logReturns.length / spanDays : 1

  return { mu: muPerObs * obsPerDay, sigma: sigmaPerObs * Math.sqrt(obsPerDay) }
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
  /** Cumulative spend carried forward from the window's running total. */
  proj_llm_cum: number | null
  /** Burn accumulated from $0 at Today — the runway line `daysUntilBroke` uses. */
  proj_llm_fwd: number | null
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
        <div style={{ color: 'var(--viz-llm)', fontSize: 12 }}>
          LLM spend: {fmtMoney(pt.hist_llm_spend, 4)}
        </div>
      )}
      {pt.proj_llm_cum != null && (
        <div style={{ color: 'var(--viz-llm)', fontSize: 12, opacity: 0.8 }}>
          LLM spend (proj): {fmtMoney(pt.proj_llm_cum, 4)}
        </div>
      )}
      {pt.proj_llm_fwd != null && (
        <div style={{ color: 'var(--viz-llm)', fontSize: 12, opacity: 0.8 }}>
          LLM burn since today: {fmtMoney(pt.proj_llm_fwd, 4)}
        </div>
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

    const today = new Date()
    today.setHours(0, 0, 0, 0)
    const todayTs = today.getTime()

    // The current day is still accruing spend, so its partial total drags the
    // burn-rate fit down. Fit the trend on completed days only.
    const todayIso = new Date().toISOString().slice(0, 10)
    const completedLlm = data.llm_series.filter((d) => d.date < todayIso)
    const llmFitRows = completedLlm.length >= 2 ? completedLlm : data.llm_series
    const llmTrend = linearTrendByDay(
      llmFitRows.map((d) => ({ ts: new Date(d.date).getTime(), value: d.daily_spend })),
    )
    // Today's position on the fitted line, so forward days extrapolate from now.
    const llmFitT0 = llmFitRows.length > 0 ? new Date(llmFitRows[0].date).getTime() : todayTs
    const llmX0 = (todayTs - llmFitT0) / MS_PER_DAY
    const burnOnDay = (d: number) =>
      Math.max(0, llmTrend.intercept + llmTrend.slope * (llmX0 + d))

    const lastPnlPoint = data.pnl_series[data.pnl_series.length - 1]
    const lastLlmPoint = data.llm_series[data.llm_series.length - 1]
    const lastCumPnl = lastPnlPoint.cumulative_pnl
    const lastCumLlm = lastLlmPoint?.cumulative_spend ?? 0

    const initialBankroll = data.initial_bankroll
    // All-time and mode-scoped — the same figure risk sizing uses. Previously this
    // was `initialBankroll + windowPnl − windowLlmSpend`, which (a) moved with the
    // 7d/30d/90d toggle because both cumulatives restart at the lookback edge, and
    // (b) charged a month of already-spent LLM cost as a single overnight drop at
    // the history→projection seam. Past spend belongs in history; only FUTURE burn
    // may erode the forward path.
    const netBankrollNow = data.net_bankroll_now

    // GBM parameters estimated from the bankroll series (log-returns). The series
    // is anchored the same way as the history curve below so both agree.
    const gbm = computeGBMParams(
      data.pnl_series.map((d) => ({
        ts: new Date(d.date).getTime(),
        bankroll: netBankrollNow - (lastCumPnl - d.cumulative_pnl),
      })),
    )

    // CAGR = geometric mean daily log-return annualized (e^(μ·365) − 1)
    const cagr = gbm ? Math.exp(gbm.mu * 365) - 1 : null

    // Days until broke = runway: where the projected bankroll crosses LLM burn
    // accumulated *from today*, i.e. how many more days of API spend the account
    // can cover. LLM spend is paid from outside the trading account, so it is
    // never subtracted from the bankroll path — the two are compared, not netted.
    //
    // Forward burn deliberately restarts at zero rather than continuing from the
    // window's cumulative spend: that cumulative restarts at the lookback edge, so
    // carrying it forward made the crossing an artifact of the chosen preset
    // (2 days on 90d, 1 on all-time) instead of a forecast.
    //
    // Computed against the same series the chart plots, so the stat and the
    // visible crossing cannot disagree.
    const daysUntilBroke = (() => {
      if (netBankrollNow <= 0) return 0
      if (!gbm) return null
      let fwdBurn = 0
      for (let d = 1; d <= 3650; d++) {
        fwdBurn += burnOnDay(d)
        if (netBankrollNow * Math.exp(gbm.mu * d) <= fwdBurn) return d
      }
      return null
    })()

    // Projected 30-day LLM burn (linear trend, unchanged)
    let proj30LlmBurn = 0
    for (let d = 1; d <= 30; d++) proj30LlmBurn += burnOnDay(d)

    // Build projection chart points: history + forward window sized to the selected preset.
    //
    // No display-side truncation. This used to slice to the last 90 rows, which
    // dropped rows off the FRONT without re-basing the running totals that were
    // computed over the full set — so on "All time" the spend line began partway
    // up its own accumulation (~$85 on the first plotted day) instead of at $0,
    // and the axis silently started ~40 days after the real first row. The API's
    // `lookback_days` already scopes the window; "All time" means all of it.
    const histSlice = chartData
    const histSpanDays =
      histSlice.length > 1
        ? (histSlice[histSlice.length - 1].ts - histSlice[0].ts) / MS_PER_DAY
        : 30
    const FORWARD_DAYS =
      preset === 'all' ? Math.max(30, Math.round(histSpanDays)) : Number(preset)

    const projPoints: ProjPoint[] = []

    // Historical bankroll curve, anchored on the CURRENT bankroll and walked
    // backwards through the in-window P&L deltas. Plotting
    // `initialBankroll + cumulative_pnl` instead planted the configured bankroll at
    // the window's LEFT edge, so the whole curve shifted whenever the lookback
    // preset — or the configured bankroll — changed.
    // `cumulative_pnl` is null on rows contributed only by llm_series; carry the
    // last known value forward, and treat rows before the window's first close as
    // zero in-window P&L.
    let runningCumPnl = 0
    for (const pt of histSlice) {
      if (pt.cumulative_pnl != null) runningCumPnl = pt.cumulative_pnl
      projPoints.push({
        ts: pt.ts,
        date: pt.date,
        hist_bankroll: Math.max(0, netBankrollNow - (lastCumPnl - runningCumPnl)),
        hist_llm_spend: pt.cumulative_spend ?? null,
        proj_central: null,
        proj_1s_lo: null,
        proj_1s_spread: null,
        proj_2s_lo: null,
        proj_2s_spread: null,
        proj_llm_cum: null,
        proj_llm_fwd: null,
      })
    }

    // GBM fan projection forward — trading P&L only. LLM spend is paid from
    // outside the trading account and is absent from the history curve, so
    // subtracting it here would make the forward line mean something the trailing
    // line does not. It stays a separate series; `daysUntilBroke` compares them.
    if (gbm && netBankrollNow > 0) {
      // Two views of the same burn rate, both plotted:
      //   cum — carries the window's running total forward (continuous with the
      //         history line, so the spend curve doesn't appear to reset)
      //   fwd — accumulates from $0 at Today (the runway `daysUntilBroke` uses)
      // They differ only by the constant `lastCumLlm`, so the gap between them is
      // exactly what has already been spent inside the window.
      let cumLlm = lastCumLlm
      let fwdCumLlm = 0
      for (let d = 1; d <= FORWARD_DAYS; d++) {
        const sqrtD = Math.sqrt(d)
        const central = netBankrollNow * Math.exp(gbm.mu * d)
        const lo1 = netBankrollNow * Math.exp(gbm.mu * d - gbm.sigma * sqrtD)
        const hi1 = netBankrollNow * Math.exp(gbm.mu * d + gbm.sigma * sqrtD)
        const lo2 = netBankrollNow * Math.exp(gbm.mu * d - 2 * gbm.sigma * sqrtD)
        const hi2 = netBankrollNow * Math.exp(gbm.mu * d + 2 * gbm.sigma * sqrtD)
        const burn = burnOnDay(d)
        cumLlm += burn
        fwdCumLlm += burn
        const lo2c = Math.max(0, lo2)
        const lo1c = Math.max(0, lo1)
        const ts = todayTs + d * MS_PER_DAY
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
          proj_llm_cum: cumLlm,
          proj_llm_fwd: fwdCumLlm,
        })
      }
    }

    // Bridge history→projection: pin the fan to the last known bankroll value.
    // With the anchoring above this is already `netBankrollNow`, so the seam is
    // continuous by construction rather than by patching over a jump.
    const lastHistWithData = [...projPoints].reverse().find(p => p.hist_bankroll != null)
    if (lastHistWithData && gbm && netBankrollNow > 0) {
      const b = lastHistWithData.hist_bankroll!
      lastHistWithData.proj_central = b
      lastHistWithData.proj_1s_lo = b
      lastHistWithData.proj_1s_spread = 0
      lastHistWithData.proj_2s_lo = b
      lastHistWithData.proj_2s_spread = 0
      // The cumulative line continues the history line unbroken; the forward line
      // is a different quantity and starts from zero at the seam.
      lastHistWithData.proj_llm_cum = lastHistWithData.hist_llm_spend
      lastHistWithData.proj_llm_fwd = 0
    }

    // Where the initial-bankroll line falls as % from the TOP of the plotted
    // range — the gradient flips green→red there, so the bankroll line reads
    // profit vs loss against where it started.
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

    // The σ fans get the same treatment, computed over each band's own extent
    // (lo .. lo+spread) — a stacked Area's bounding box spans exactly that, and
    // the gradient is in objectBoundingBox units. Where a band straddles the
    // starting bankroll the break flips mid-band, so the red portion is literally
    // the share of the distribution that has lost money.
    function bandExtent(lo: (p: ProjPoint) => number | null, spread: (p: ProjPoint) => number | null) {
      return projPoints.flatMap((p) => {
        const l = lo(p)
        const s = spread(p)
        return l != null && s != null ? [l, l + s] : []
      })
    }
    const band1GradPct = bankrollGradStop(
      bandExtent(p => p.proj_1s_lo, p => p.proj_1s_spread), initialBankroll,
    )
    const band2GradPct = bankrollGradStop(
      bandExtent(p => p.proj_2s_lo, p => p.proj_2s_spread), initialBankroll,
    )

    return {
      initialBankroll,
      band1GradPct,
      band2GradPct,
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
              label="Bankroll now"
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
              sub="realized P&L only — LLM spend is paid outside the account"
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
              sub={
                projection.daysUntilBroke == null
                  ? 'bankroll outpaces forward LLM burn'
                  : 'bankroll crosses forward LLM burn'
              }
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
                {/* Bankroll keeps the pos/neg polarity gradient — green above the
                    starting bankroll, red below. LLM cost is a separate entity in
                    its own hue (blue); dash encodes actual vs projected. */}
                {((<defs>
                  <linearGradient id="histBankrollAreaGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%"                             stopColor="var(--pos)" stopOpacity={0.55} />
                    <stop offset={projection.histBankrollGradPct} stopColor="var(--pos)" stopOpacity={0.04} />
                    <stop offset={projection.histBankrollGradPct} stopColor="var(--neg)" stopOpacity={0.04} />
                    <stop offset="100%"                           stopColor="var(--neg)" stopOpacity={0.55} />
                  </linearGradient>
                  <linearGradient id="histBankrollLineGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset={projection.histBankrollGradPct} stopColor="var(--pos)" />
                    <stop offset={projection.histBankrollGradPct} stopColor="var(--neg)" />
                  </linearGradient>
                  <linearGradient id="projBankrollLineGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset={projection.projBankrollGradPct} stopColor="var(--pos)" />
                    <stop offset={projection.projBankrollGradPct} stopColor="var(--neg)" />
                  </linearGradient>
                  {/* σ fans: green above the starting bankroll, red below, with a
                      hard break at break-even — a soft fade would blur exactly the
                      threshold the fan exists to show. */}
                  <linearGradient id="band1Grad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset={projection.band1GradPct} stopColor="var(--pos)" stopOpacity={0.24} />
                    <stop offset={projection.band1GradPct} stopColor="var(--neg)" stopOpacity={0.24} />
                  </linearGradient>
                  <linearGradient id="band2Grad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset={projection.band2GradPct} stopColor="var(--pos)" stopOpacity={0.13} />
                    <stop offset={projection.band2GradPct} stopColor="var(--neg)" stopOpacity={0.13} />
                  </linearGradient>
                </defs>) as React.ReactNode)}

                {/* Shaded projection zone */}
                <ReferenceArea
                  x1={projection.todayTs}
                  x2={projection.projPoints[projection.projPoints.length - 1]?.ts}
                  fill="var(--fg-3)"
                  fillOpacity={0.06}
                />
                <ReferenceLine
                  x={projection.todayTs}
                  stroke="var(--text-3, #9ca3af)"
                  strokeDasharray="4 2"
                  label={{ value: 'Today', position: 'top', fontSize: 11 }}
                />
                <ReferenceLine y={0} stroke="var(--neg)" strokeWidth={1} />
                {/* Where every green/red break on this chart is anchored. */}
                <ReferenceLine
                  y={projection.initialBankroll}
                  stroke="var(--fg-3)"
                  strokeDasharray="2 4"
                  strokeWidth={1}
                  label={{ value: 'break-even', position: 'insideBottomLeft', fontSize: 10, fill: 'var(--fg-3)' }}
                />

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
                <Area type="monotone" stackId="band2" dataKey="proj_2s_spread" fill="url(#band2Grad)" stroke="none" name="±2σ band" dot={false} />
                {/* ±1σ fan band */}
                <Area type="monotone" stackId="band1" dataKey="proj_1s_lo"     fill="transparent" stroke="none" legendType="none" dot={false} />
                <Area type="monotone" stackId="band1" dataKey="proj_1s_spread" fill="url(#band1Grad)" stroke="none" name="±1σ band" dot={false} />
                {/* GBM central path */}
                <Line
                  type="monotone"
                  dataKey="proj_central"
                  name="Bankroll (projected)"
                  stroke="url(#projBankrollLineGrad)"
                  strokeWidth={1.5}
                  strokeDasharray="5 3"
                  dot={false}
                  connectNulls
                />
                <Line
                  dataKey="hist_llm_spend"
                  name="LLM spend (history)"
                  stroke="var(--viz-llm)"
                  strokeWidth={2}
                  dot={false}
                  connectNulls
                />
                <Line
                  dataKey="proj_llm_cum"
                  name="LLM spend (projected)"
                  stroke="var(--viz-llm)"
                  strokeWidth={1.5}
                  strokeDasharray="5 3"
                  dot={false}
                  connectNulls
                />
                <Line
                  dataKey="proj_llm_fwd"
                  name="LLM burn (from today)"
                  stroke="var(--viz-llm)"
                  strokeWidth={1.5}
                  strokeDasharray="1 4"
                  strokeOpacity={0.85}
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

