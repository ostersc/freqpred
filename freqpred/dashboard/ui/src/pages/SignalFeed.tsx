import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { getSignals, getSignal } from '../api/signals'
import { getStrategyConfig } from '../api/strategy'
import { Badge, Panel, LoadingSpinner, ErrorBanner, ProbBar, fmtAge } from '../components/ui'
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

type TriggerKind = 'scheduled' | 'price_moved' | 'entry_manual' | 'market_update' | string

function triggerBadge(trigger: TriggerKind) {
  const map: Record<string, 'accent' | 'info' | 'pos' | 'muted'> = {
    price_moved: 'accent',
    scheduled: 'info',
    entry_manual: 'pos',
    market_update: 'muted',
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
          <div className="row" style={{ gap: 8 }}>
            <span className="ticker-id">{signal.market_id}</span>
            <Badge kind={triggerBadge(signal.trigger)} dot>{signal.trigger.replace('_', ' ')}</Badge>
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

export default function SignalFeed() {
  const [offset, setOffset] = useState(0)

  const { data: config } = useQuery({
    queryKey: ['strategyConfig'],
    queryFn: getStrategyConfig,
    staleTime: 60_000,
  })
  const minConf = config?.min_confidence ?? 0.5

  const { data, isLoading, error } = useQuery({
    queryKey: ['signals', offset],
    queryFn: () => getSignals({ limit: PAGE_SIZE, offset }),
    refetchInterval: 30_000,
  })

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
