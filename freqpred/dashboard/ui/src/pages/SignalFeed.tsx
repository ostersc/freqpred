import { useEffect, useState } from 'react'
import { useQuery, keepPreviousData } from '@tanstack/react-query'
import { getSignals, getSignal } from '../api/signals'
import { getStrategyConfig } from '../api/strategy'
import { Badge, Panel, LoadingSpinner, ErrorBanner, ProbBar, LabeledSelect, RangeSlider, fmtAge } from '../components/ui'
import AnalyzeButton from '../components/AnalyzeButton'
import { SignalDetail as SharedSignalDetail } from '../components/SignalDetail'
import type { SignalOut } from '../api/types'

function confColor(conf: number, minConf: number): string {
  if (conf >= minConf) {
    const t = minConf < 1 ? (conf - minConf) / (1 - minConf) : 1
    const L = (0.88 - t * 0.26).toFixed(2)
    const C = (0.06 + t * 0.14).toFixed(2)
    return `oklch(${L} ${C} 160)`
  } else {
    const t = minConf > 0 ? (minConf - conf) / minConf : 1
    const L = (0.92 - t * 0.37).toFixed(2)
    const C = (0.06 + t * 0.14).toFixed(2)
    const H = (80 - t * 55).toFixed(0)
    return `oklch(${L} ${C} ${H})`
  }
}

const PAGE_SIZE = 20

function firstSentence(text: string): string {
  const m = text.match(/^[^.!?]*[.!?]/)
  return m ? m[0] : text
}

function triggerBadge(trigger: string) {
  const map: Record<string, 'accent' | 'info' | 'pos' | 'muted'> = {
    price_moved: 'accent',
    scheduled: 'info',
    manual: 'pos',
    demo_harness: 'muted',
  }
  return map[trigger] ?? 'muted'
}

function ragBadgeKind(count: number): 'warn' | 'info' | 'accent' {
  if (count === 0) return 'warn'
  if (count <= 3) return 'info'
  return 'accent'
}

function SignalDataBadges({ signal }: { signal: SignalOut }) {
  return (
    <div className="row" style={{ gap: 5, marginTop: 4, flexWrap: 'wrap' }}>
      <Badge kind={ragBadgeKind(signal.rag_hit_count)}>
        {signal.rag_hit_count === 0 ? 'no docs' : `${signal.rag_hit_count} doc${signal.rag_hit_count === 1 ? '' : 's'}`}
      </Badge>
      {signal.has_factbase && <Badge kind="accent" dot>factbase</Badge>}
      {signal.series_ticker && <Badge kind="muted">{signal.series_ticker}</Badge>}
      {signal.has_assessment && <Badge kind="pos">assessed</Badge>}
      {signal.social_sentiment_summary && <Badge kind="info">social</Badge>}
    </div>
  )
}

function SignalDetailRow({ id, marketId }: { id: string; marketId: string }) {
  const { data, isLoading } = useQuery({
    queryKey: ['signal', id],
    queryFn: () => getSignal(id),
  })

  if (isLoading) return <div style={{ padding: '14px 20px', color: 'var(--fg-2)', fontSize: 12 }}>Loading…</div>
  if (!data) return null

  return (
    <div style={{ padding: '14px 20px' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 12 }}>
        <span style={{ fontWeight: 500, fontSize: 13 }}>{data.market_question ?? 'Signal detail'}</span>
        <AnalyzeButton marketId={marketId} />
      </div>
      <SharedSignalDetail signal={data} />
    </div>
  )
}

function SignalRow({ signal, minConf }: { signal: SignalOut; minConf: number }) {
  const [expanded, setExpanded] = useState(false)
  return (
    <>
      <tr
        className={expanded ? 'expanded' : ''}
        onClick={() => setExpanded((e) => !e)}
        style={{ cursor: 'pointer' }}
      >
        <td>
          <div style={{ fontWeight: 500, marginBottom: 3 }}>{firstSentence(signal.market_question ?? signal.market_id)}</div>
          <div className="row" style={{ gap: 8, flexWrap: 'wrap' }}>
            <span className="ticker-id">{signal.market_id}</span>
            <Badge kind={triggerBadge(signal.trigger)} dot>{signal.trigger.replace('_', ' ')}</Badge>
            {signal.has_open_position && <Badge kind="pos" dot>open position</Badge>}
          </div>
          <SignalDataBadges signal={signal} />
        </td>
        <td style={{ width: 200 }}>
          <div style={{ fontSize: 10.5, color: 'var(--fg-2)', marginBottom: 4, textTransform: 'uppercase', letterSpacing: '0.06em' }}>Our · Market</div>
          <ProbBar ours={signal.estimated_probability} market={signal.market_mid_at_signal} />
        </td>
        <td className="r">
          <div style={{ fontSize: 10, color: 'var(--fg-2)', letterSpacing: '0.06em', textTransform: 'uppercase' }}>Edge</div>
          <div className={`mono ${signal.edge >= 0 ? 'pos' : 'neg'}`} style={{ fontSize: 14, fontWeight: 500 }}>
            {signal.edge >= 0 ? '+' : ''}{(signal.edge * 100).toFixed(1)}%
          </div>
        </td>
        <td className="r">
          <div style={{ fontSize: 10, color: 'var(--fg-2)', letterSpacing: '0.06em', textTransform: 'uppercase' }}>Conf</div>
          <div className="mono" style={{ fontSize: 14, fontWeight: 500, color: confColor(signal.confidence, minConf) }}>{(signal.confidence * 100).toFixed(0)}%</div>
        </td>
        <td className="c">
          <Badge kind={signal.direction === 'YES' ? 'pos' : signal.direction === 'NO' ? 'neg' : 'muted'}>
            {signal.direction}
          </Badge>
        </td>
        <td className="r dim" style={{ fontSize: 11 }}>{fmtAge(signal.created_at)}</td>
        <td className="c"><span className={`caret${expanded ? ' open' : ''}`}>›</span></td>
      </tr>
      {expanded && (
        <tr className="detail-row">
          <td colSpan={7}>
            <SignalDetailRow id={signal.id} marketId={signal.market_id} />
          </td>
        </tr>
      )}
    </>
  )
}

type TriState = '' | 'yes' | 'no'

const DEFAULT_TRIGGER = 'scheduled'

export default function SignalFeed() {
  const [offset, setOffset] = useState(0)
  const [direction, setDirection] = useState('')
  const [trigger, setTrigger] = useState(DEFAULT_TRIGGER)
  const [seriesTicker, setSeriesTicker] = useState('')
  const [hasFactbase, setHasFactbase] = useState<TriState>('')
  const [hasDocs, setHasDocs] = useState<TriState>('')
  const [edgeRange, setEdgeRange] = useState<[number, number] | null>(null)
  const [confRange, setConfRange] = useState<[number, number] | null>(null)

  const { data: config } = useQuery({
    queryKey: ['strategyConfig'],
    queryFn: getStrategyConfig,
    staleTime: 60_000,
  })
  const minConf = config?.min_confidence ?? 0.5

  // Initialize the range filters from the live strategy config exactly once —
  // re-running this on every config refetch would clobber in-progress user edits.
  useEffect(() => {
    if (config && edgeRange === null) setEdgeRange([config.min_edge, config.max_edge ?? 1])
    if (config && confRange === null) setConfRange([config.min_confidence, 1])
  }, [config, edgeRange, confRange])

  const { data, isLoading, error } = useQuery({
    queryKey: ['signals', offset, direction, trigger, seriesTicker, hasFactbase, hasDocs, edgeRange, confRange],
    queryFn: () => getSignals({
      limit: PAGE_SIZE,
      offset,
      direction: direction || undefined,
      trigger: trigger || undefined,
      series_ticker: seriesTicker || undefined,
      has_factbase: hasFactbase === '' ? undefined : hasFactbase === 'yes',
      has_docs: hasDocs === '' ? undefined : hasDocs === 'yes',
      min_edge: edgeRange?.[0],
      max_edge: edgeRange?.[1],
      min_confidence: confRange?.[0],
      max_confidence: confRange?.[1],
    }),
    enabled: edgeRange !== null && confRange !== null,
    placeholderData: keepPreviousData,
    refetchInterval: 30_000,
  })

  function resetFilters() {
    setDirection('')
    setTrigger(DEFAULT_TRIGGER)
    setSeriesTicker('')
    setHasFactbase('')
    setHasDocs('')
    setEdgeRange(null)
    setConfRange(null)
    setOffset(0)
  }

  const triggerOptions = Array.from(new Set([DEFAULT_TRIGGER, ...(data?.distinct_triggers ?? [])]))
  const seriesOptions = data?.distinct_series_tickers ?? []

  return (
    <div className="page">
      <div className="page-head">
        <div>
          <h1 className="page-title">Signal Feed</h1>
          <div className="page-subtitle">Live stream of every probability estimate from the signal loop</div>
        </div>
        <div className="row">
          <div className="chip">
            <span className="d" style={{ width: 6, height: 6, borderRadius: '50%', background: 'var(--pos)', display: 'inline-block', boxShadow: '0 0 8px var(--pos)' }} />
            Live · refreshes 30s
          </div>
          {data && <div className="chip mono">{data.total} signals total</div>}
        </div>
      </div>

      <Panel style={{ marginBottom: 12 }}>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 16, alignItems: 'end' }}>
          <LabeledSelect
            label="Direction"
            value={direction}
            onChange={(v) => { setDirection(v); setOffset(0) }}
            options={[
              { value: '', label: 'All' },
              { value: 'YES', label: 'YES' },
              { value: 'NO', label: 'NO' },
              { value: 'SKIP', label: 'SKIP' },
            ]}
          />
          <LabeledSelect
            label="Type"
            value={trigger}
            onChange={(v) => { setTrigger(v); setOffset(0) }}
            options={[
              { value: '', label: 'All' },
              ...triggerOptions.map((t) => ({ value: t, label: t.replace('_', ' ') })),
            ]}
          />
          <LabeledSelect
            label="Series"
            value={seriesTicker}
            onChange={(v) => { setSeriesTicker(v); setOffset(0) }}
            options={[
              { value: '', label: 'All' },
              ...seriesOptions.map((s) => ({ value: s, label: s })),
            ]}
          />
          <LabeledSelect
            label="Factbase"
            value={hasFactbase}
            onChange={(v) => { setHasFactbase(v as TriState); setOffset(0) }}
            options={[
              { value: '', label: 'Any' },
              { value: 'yes', label: 'Has factbase' },
              { value: 'no', label: 'No factbase' },
            ]}
          />
          <LabeledSelect
            label="Docs"
            value={hasDocs}
            onChange={(v) => { setHasDocs(v as TriState); setOffset(0) }}
            options={[
              { value: '', label: 'Any' },
              { value: 'yes', label: 'Has docs' },
              { value: 'no', label: 'No docs' },
            ]}
          />
          <div className="labeled-field" style={{ minWidth: 200 }}>
            <label className="field-label">
              Edge {edgeRange ? `${(edgeRange[0] * 100).toFixed(0)}%–${(edgeRange[1] * 100).toFixed(0)}%` : ''}
            </label>
            {edgeRange && (
              <RangeSlider
                min={-1} max={1} step={0.01}
                valueMin={edgeRange[0]} valueMax={edgeRange[1]}
                onChange={(lo, hi) => { setEdgeRange([lo, hi]); setOffset(0) }}
              />
            )}
          </div>
          <div className="labeled-field" style={{ minWidth: 200 }}>
            <label className="field-label">
              Confidence {confRange ? `${(confRange[0] * 100).toFixed(0)}%–${(confRange[1] * 100).toFixed(0)}%` : ''}
            </label>
            {confRange && (
              <RangeSlider
                min={0} max={1} step={0.01}
                valueMin={confRange[0]} valueMax={confRange[1]}
                onChange={(lo, hi) => { setConfRange([lo, hi]); setOffset(0) }}
              />
            )}
          </div>
          <button className="btn ghost" onClick={resetFilters}>Reset</button>
        </div>
      </Panel>

      {isLoading && <LoadingSpinner />}
      {error && <ErrorBanner message={String(error)} />}

      {data && (
        <>
          <Panel flush>
            <table className="tbl">
              <thead>
                <tr>
                  <th>Market</th>
                  <th style={{ width: 200 }}>Our · Market</th>
                  <th className="r">Edge</th>
                  <th className="r">Confidence</th>
                  <th className="c">Dir</th>
                  <th className="r">Age</th>
                  <th style={{ width: 36 }}></th>
                </tr>
              </thead>
              <tbody>
                {data.items.map((s) => <SignalRow key={s.id} signal={s} minConf={minConf} />)}
                {data.items.length === 0 && (
                  <tr>
                    <td colSpan={7} style={{ padding: '24px', textAlign: 'center', color: 'var(--fg-3)' }}>
                      No signals match the current filters
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </Panel>
          <div className="pagination">
            <button className="btn sm" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - PAGE_SIZE))}>Previous</button>
            <span>{offset + 1}–{Math.min(offset + PAGE_SIZE, data.total)} of {data.total}</span>
            <button className="btn sm" disabled={offset + PAGE_SIZE >= data.total} onClick={() => setOffset(offset + PAGE_SIZE)}>Next</button>
          </div>
        </>
      )}
    </div>
  )
}
