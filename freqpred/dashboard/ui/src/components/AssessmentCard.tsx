import { Link } from 'react-router-dom'
import type { SignalAssessmentOut } from '../api/types'

function fmtPct(value: number, digits = 1) {
  return `${(value * 100).toFixed(digits)}%`
}

function fmtSignedPct(value: number) {
  const rendered = (value * 100).toFixed(1)
  return value > 0 ? `+${rendered}%` : `${rendered}%`
}

function verdictLabel(verdict: string) {
  if (verdict === 'size_up') return 'Size up'
  if (verdict === 'size_down') return 'Size down'
  return 'Neutral'
}

function verdictTone(verdict: string) {
  if (verdict === 'size_up') return 'bg-green-100 text-green-800'
  if (verdict === 'size_down') return 'bg-red-100 text-red-800'
  return 'bg-gray-100 text-gray-700'
}

function trendTone(delta: number | null) {
  if (delta === null) return 'bg-gray-100 text-gray-700'
  if (delta < 0) return 'bg-green-100 text-green-800'
  if (delta > 0) return 'bg-red-100 text-red-800'
  return 'bg-gray-100 text-gray-700'
}

function trendLabel(delta: number | null) {
  if (delta === null) return 'No trend'
  if (delta < 0) return 'Improving'
  if (delta > 0) return 'Harming'
  return 'Flat'
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
    ? value as Record<string, unknown>
    : null
}

function asNumber(value: unknown): number | null {
  return typeof value === 'number' ? value : null
}

function asString(value: unknown): string | null {
  return typeof value === 'string' ? value : null
}

function buildSimilarMarketLines(summary: Record<string, unknown>): string[] {
  const lines: string[] = []
  const available = summary.available === true
  if (!available) {
    const reason = asString(summary.reason)
    if (reason === 'missing_series_ticker') return ['No similar-market family is available for this market.']
    if (reason === 'insufficient_matched_history') return ['Matched market history exists, but the sample is still too thin to trust.']
    return ['No similar-market trust data is available yet.']
  }

  const family = asRecord(summary.family_match)
  const exactSubset = asRecord(summary.exact_question_subset)
  const strategyHistory = asRecord(summary.strategy_trade_history)

  const familySignals = asNumber(family?.resolved_signals)
  const familyDelta = asNumber(family?.family_signal_delta_vs_overall)
  if (familySignals !== null) {
    let line = `${familySignals} resolved family signals`
    if (familyDelta !== null) {
      line += familyDelta < 0
        ? ` with better-than-overall calibration (${fmtSignedPct(familyDelta)} vs overall Brier).`
        : ` with weaker-than-overall calibration (${fmtSignedPct(familyDelta)} vs overall Brier).`
    } else {
      line += '.'
    }
    lines.push(line)
  }

  const exactSignals = asNumber(exactSubset?.resolved_signals)
  if (exactSignals !== null) {
    const smallSample = exactSubset?.small_sample === true
    lines.push(
      smallSample
        ? `${exactSignals} exact-question matches, but this is still a small sample.`
        : `${exactSignals} exact-question matches are available.`,
    )
  }

  const trades = asNumber(strategyHistory?.closed_trades)
  const winRate = asNumber(strategyHistory?.win_rate)
  if (trades !== null && winRate !== null) {
    lines.push(`Strategy history in this family: ${trades} closed trades, ${fmtPct(winRate)} win rate.`)
  }

  return lines.length > 0 ? lines : ['Similar-market history is available.']
}

export default function AssessmentCard({ assessment }: { assessment: SignalAssessmentOut | null }) {
  if (assessment === null) {
    return (
      <div className="rounded border border-dashed border-gray-300 bg-gray-50 p-3">
        <div className="text-xs font-semibold uppercase tracking-wide text-gray-500">Assessment</div>
        <p className="mt-2 text-sm text-gray-500">No persisted assessment is available for this signal yet.</p>
      </div>
    )
  }

  const sourceBreakdown = assessment.source_breakdown
    .map((value) => asRecord(value))
    .filter((value): value is Record<string, unknown> => value !== null)
  const similarSummary = asRecord(assessment.similar_market_summary) ?? {}

  return (
    <div className="rounded border bg-slate-50 p-3 space-y-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="text-xs font-semibold uppercase tracking-wide text-slate-600">Assessment</div>
        {assessment.llm_query_id !== null && (
          <Link
            to={`/llm?queryId=${assessment.llm_query_id}`}
            className="text-xs font-medium text-blue-600 hover:underline"
          >
            Open LLM audit
          </Link>
        )}
      </div>

      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <div className="rounded border bg-white p-3">
          <div className="text-xs uppercase tracking-wide text-gray-400">Trust score</div>
          <div className="mt-1 text-lg font-semibold text-gray-900">{fmtPct(assessment.trust_score)}</div>
        </div>
        <div className="rounded border bg-white p-3">
          <div className="text-xs uppercase tracking-wide text-gray-400">Size effect</div>
          <div className="mt-1 text-lg font-semibold text-gray-900">{assessment.size_multiplier.toFixed(2)}x</div>
        </div>
        <div className="rounded border bg-white p-3">
          <div className="text-xs uppercase tracking-wide text-gray-400">Verdict</div>
          <div className="mt-1">
            <span className={`rounded-full px-2 py-1 text-xs font-semibold ${verdictTone(assessment.verdict)}`}>
              {verdictLabel(assessment.verdict)}
            </span>
          </div>
        </div>
        <div className="rounded border bg-white p-3">
          <div className="text-xs uppercase tracking-wide text-gray-400">Assessed</div>
          <div className="mt-1 text-sm font-medium text-gray-800">{new Date(assessment.created_at).toLocaleString()}</div>
        </div>
      </div>

      <div>
        <div className="mb-1 text-xs font-semibold uppercase tracking-wide text-gray-500">Reasoning</div>
        <p className="text-sm text-gray-700 whitespace-pre-wrap">{assessment.reasoning}</p>
      </div>

      {assessment.key_factors.length > 0 && (
        <div>
          <div className="mb-1 text-xs font-semibold uppercase tracking-wide text-gray-500">Key factors</div>
          <div className="flex flex-wrap gap-2">
            {assessment.key_factors.map((factor) => (
              <span key={factor} className="rounded-full bg-blue-100 px-2 py-1 text-xs text-blue-800">
                {factor}
              </span>
            ))}
          </div>
        </div>
      )}

      <div>
        <div className="mb-1 text-xs font-semibold uppercase tracking-wide text-gray-500">Source quality summary</div>
        {sourceBreakdown.length === 0 ? (
          <p className="text-sm text-gray-500">No scored source-quality history was attached to this evidence set.</p>
        ) : (
          <div className="space-y-2">
            {sourceBreakdown.slice(0, 4).map((source) => {
              const sourceName = asString(source.source_name) ?? 'Unknown source'
              const documentShare = asNumber(source.document_share)
              const delta = asNumber(source.delta_vs_overall)
              return (
                <div key={sourceName} className="rounded border bg-white p-2 text-sm">
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <div className="font-medium text-gray-800">{sourceName}</div>
                    <span
                      className={`rounded-full px-2 py-1 text-xs font-semibold ${trendTone(delta)}`}
                      title="Recent performance is better/worse than historical average."
                    >
                      {trendLabel(delta)}
                    </span>
                  </div>
                  <div className="mt-1 text-gray-600">
                    {documentShare !== null ? `${fmtPct(documentShare, 0)} of retrieved docs` : 'Share unavailable'}
                    {delta !== null ? `, ${fmtSignedPct(delta)} vs overall Brier.` : '.'}
                  </div>
                </div>
              )
            })}
          </div>
        )}
      </div>

      <div>
        <div className="mb-1 text-xs font-semibold uppercase tracking-wide text-gray-500">Similar-market summary</div>
        <div className="space-y-1 text-sm text-gray-700">
          {buildSimilarMarketLines(similarSummary).map((line) => (
            <p key={line}>{line}</p>
          ))}
        </div>
      </div>

      {assessment.warnings.length > 0 && (
        <div className="rounded border border-amber-200 bg-amber-50 p-3">
          <div className="mb-1 text-xs font-semibold uppercase tracking-wide text-amber-700">Warnings</div>
          <div className="space-y-1 text-sm text-amber-900">
            {assessment.warnings.map((warning) => (
              <p key={warning}>{warning}</p>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
