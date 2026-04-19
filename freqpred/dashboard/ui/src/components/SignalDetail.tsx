import { useQuery } from '@tanstack/react-query'
import { getSignal } from '../api/signals'
import type { SignalDetailOut } from '../api/types'
import AssessmentCard from './AssessmentCard'
import { DocLinkItem } from './DocLinkItem'

// ---- Formatting helpers -------------------------------------------------

function fmtPct(v: number | null) {
  if (v === null) return '—'
  const s = (v * 100).toFixed(1)
  return v >= 0 ? `+${s}%` : `${s}%`
}

function relTime(iso: string) {
  return new Date(iso).toLocaleString()
}

// ---- SignalDetail --------------------------------------------------------

export function SignalDetail({ signal }: { signal: SignalDetailOut }) {
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
      <AssessmentCard assessment={signal.assessment} />
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

// ---- SelectedSignalPanel -------------------------------------------------
// Fetches signal detail on demand (for non-entry signals clicked in chart)

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
  if (isLoading) return <div className="p-3 text-sm text-gray-400">Loading signal…</div>
  if (!data) return null
  return <SignalDetail signal={data} />
}
