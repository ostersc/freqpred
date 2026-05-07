import React from 'react'
import { Link } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { getSignal } from '../api/signals'
import type { SignalDetailOut } from '../api/types'
import AssessmentCard from './AssessmentCard'
import { DocLinkItem } from './DocLinkItem'

const sectionLabel: React.CSSProperties = {
  fontSize: 10.5, textTransform: 'uppercase', letterSpacing: '0.08em',
  color: 'var(--fg-2)', marginBottom: 6, fontWeight: 600,
}

export function SignalDetail({ signal }: { signal: SignalDetailOut }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12, padding: '12px 0' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 16, fontSize: 12 }}>
          <div><span className="dim">Our prob:</span> <b className="mono">{(signal.estimated_probability * 100).toFixed(1)}%</b></div>
          <div><span className="dim">Market mid:</span> <b className="mono">{(signal.market_mid_at_signal * 100).toFixed(1)}%</b></div>
          <div>
            <span className="dim">Edge:</span>{' '}
            <b className={`mono ${signal.edge >= 0 ? 'pos' : 'neg'}`}>
              {signal.edge >= 0 ? '+' : ''}{(signal.edge * 100).toFixed(1)}%
            </b>
          </div>
          <div><span className="dim">Confidence:</span> <b className="mono">{(signal.confidence * 100).toFixed(1)}%</b></div>
          <div className="dim" style={{ fontSize: 11 }}>{new Date(signal.created_at).toLocaleString()}</div>
        </div>
        {signal.llm_query_id !== null && (
          <Link to={`/llm?queryId=${signal.llm_query_id}`} style={{ fontSize: 11.5, color: 'var(--accent)', whiteSpace: 'nowrap' }}>
            Open LLM audit →
          </Link>
        )}
      </div>

      <div>
        <div style={sectionLabel}>Reasoning</div>
        <p style={{ margin: 0, fontSize: 12.5, color: 'var(--fg-1)', lineHeight: 1.6, whiteSpace: 'pre-wrap' }}>{signal.reasoning}</p>
      </div>

      {signal.social_sentiment_summary && (
        <div>
          <div style={sectionLabel}>Social sentiment</div>
          <p style={{ margin: 0, fontSize: 12.5, color: 'var(--fg-1)', lineHeight: 1.6 }}>{signal.social_sentiment_summary}</p>
        </div>
      )}

      <AssessmentCard assessment={signal.assessment} />

      {signal.document_links.length > 0 && (
        <div>
          <div style={sectionLabel}>Evidence documents</div>
          <ul style={{ margin: 0, padding: 0, display: 'flex', flexDirection: 'column', gap: 6 }}>
            {signal.document_links.map((doc) => (
              <DocLinkItem key={doc.document_id} doc={doc} />
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}

export function SelectedSignalPanel({
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
  if (isLoading) return <div style={{ padding: '12px 0', color: 'var(--fg-3)', fontSize: 12.5 }}>Loading signal…</div>
  if (!data) return null
  return <SignalDetail signal={data} />
}
