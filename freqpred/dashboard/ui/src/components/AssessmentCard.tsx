import React from 'react'
import { Link } from 'react-router-dom'
import type { SignalAssessmentOut } from '../api/types'
import { Badge } from './ui'

function fmtPct(value: number, digits = 1) {
  return `${(value * 100).toFixed(digits)}%`
}

function fmtSignedPct(value: number) {
  const rendered = (value * 100).toFixed(1)
  return value > 0 ? `+${rendered}%` : `${rendered}%`
}

function verdictKind(verdict: string): 'pos' | 'neg' | 'muted' {
  if (verdict === 'size_up') return 'pos'
  if (verdict === 'size_down') return 'neg'
  return 'muted'
}

function verdictLabel(verdict: string) {
  if (verdict === 'size_up') return 'Size up'
  if (verdict === 'size_down') return 'Size down'
  return 'Neutral'
}

function trendKind(delta: number | null): 'pos' | 'neg' | 'muted' {
  if (delta === null) return 'muted'
  if (delta < 0) return 'pos'
  if (delta > 0) return 'neg'
  return 'muted'
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

const sectionLabel: React.CSSProperties = {
  fontSize: 10.5, textTransform: 'uppercase', letterSpacing: '0.08em',
  color: 'var(--fg-2)', marginBottom: 6, fontWeight: 600,
}

export default function AssessmentCard({ assessment }: { assessment: SignalAssessmentOut | null }) {
  if (assessment === null) {
    return (
      <div style={{ padding: 12, border: '1px dashed var(--line)', borderRadius: 6, background: 'var(--bg-1)' }}>
        <div style={sectionLabel}>Assessment</div>
        <p style={{ margin: 0, fontSize: 12.5, color: 'var(--fg-3)' }}>No persisted assessment is available for this signal yet.</p>
      </div>
    )
  }

  const sourceBreakdown = assessment.source_breakdown
    .map((value) => asRecord(value))
    .filter((value): value is Record<string, unknown> => value !== null)
  const similarSummary = asRecord(assessment.similar_market_summary) ?? {}

  return (
    <div style={{ padding: 12, border: '1px solid var(--line-soft)', borderRadius: 6, background: 'var(--bg-1)' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
        <div style={sectionLabel}>Assessment</div>
        {assessment.llm_query_id !== null && (
          <Link to={`/llm?queryId=${assessment.llm_query_id}`} style={{ fontSize: 11.5, color: 'var(--accent)' }}>
            Open LLM audit →
          </Link>
        )}
      </div>

      <div className="grid grid-4" style={{ marginBottom: 12, gap: 8 }}>
        <div style={{ padding: '8px 10px', background: 'var(--bg-0)', border: '1px solid var(--line-soft)', borderRadius: 6 }}>
          <div style={{ fontSize: 10.5, color: 'var(--fg-3)', marginBottom: 4 }}>Trust score</div>
          <div className="mono" style={{ fontSize: 16, fontWeight: 600 }}>{fmtPct(assessment.trust_score)}</div>
        </div>
        <div style={{ padding: '8px 10px', background: 'var(--bg-0)', border: '1px solid var(--line-soft)', borderRadius: 6 }}>
          <div style={{ fontSize: 10.5, color: 'var(--fg-3)', marginBottom: 4 }}>Size effect</div>
          <div className="mono" style={{ fontSize: 16, fontWeight: 600 }}>{assessment.size_multiplier.toFixed(2)}x</div>
        </div>
        <div style={{ padding: '8px 10px', background: 'var(--bg-0)', border: '1px solid var(--line-soft)', borderRadius: 6 }}>
          <div style={{ fontSize: 10.5, color: 'var(--fg-3)', marginBottom: 4 }}>Verdict</div>
          <Badge kind={verdictKind(assessment.verdict)}>{verdictLabel(assessment.verdict)}</Badge>
        </div>
        <div style={{ padding: '8px 10px', background: 'var(--bg-0)', border: '1px solid var(--line-soft)', borderRadius: 6 }}>
          <div style={{ fontSize: 10.5, color: 'var(--fg-3)', marginBottom: 4 }}>Assessed</div>
          <div style={{ fontSize: 11.5, color: 'var(--fg-1)' }}>{new Date(assessment.created_at).toLocaleString()}</div>
        </div>
      </div>

      <div style={{ marginBottom: 10 }}>
        <div style={sectionLabel}>Reasoning</div>
        <p style={{ margin: 0, fontSize: 12.5, color: 'var(--fg-1)', lineHeight: 1.6, whiteSpace: 'pre-wrap' }}>{assessment.reasoning}</p>
      </div>

      {assessment.key_factors.length > 0 && (
        <div style={{ marginBottom: 10 }}>
          <div style={sectionLabel}>Key factors</div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
            {assessment.key_factors.map((factor) => (
              <span key={factor} className="mono" style={{ padding: '2px 8px', background: 'var(--bg-2)', border: '1px solid var(--line-soft)', borderRadius: 4, fontSize: 11.5, color: 'var(--fg-1)' }}>
                {factor}
              </span>
            ))}
          </div>
        </div>
      )}

      <div style={{ marginBottom: 10 }}>
        <div style={sectionLabel}>Source quality summary</div>
        {sourceBreakdown.length === 0 ? (
          <p style={{ margin: 0, fontSize: 12.5, color: 'var(--fg-3)' }}>No scored source-quality history was attached to this evidence set.</p>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            {sourceBreakdown.slice(0, 4).map((source) => {
              const sourceName = asString(source.source_name) ?? 'Unknown source'
              const documentShare = asNumber(source.document_share)
              const delta = asNumber(source.delta_vs_overall)
              return (
                <div key={sourceName} style={{ padding: '8px 10px', background: 'var(--bg-0)', border: '1px solid var(--line-soft)', borderRadius: 6, fontSize: 12 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
                    <span style={{ fontWeight: 500, color: 'var(--fg-0)' }}>{sourceName}</span>
                    <Badge kind={trendKind(delta)}>{trendLabel(delta)}</Badge>
                  </div>
                  <div style={{ color: 'var(--fg-2)' }}>
                    {documentShare !== null ? `${fmtPct(documentShare, 0)} of retrieved docs` : 'Share unavailable'}
                    {delta !== null ? `, ${fmtSignedPct(delta)} vs overall Brier.` : '.'}
                  </div>
                </div>
              )
            })}
          </div>
        )}
      </div>

      <div style={{ marginBottom: assessment.warnings.length > 0 ? 10 : 0 }}>
        <div style={sectionLabel}>Similar-market summary</div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
          {buildSimilarMarketLines(similarSummary).map((line) => (
            <p key={line} style={{ margin: 0, fontSize: 12.5, color: 'var(--fg-1)' }}>{line}</p>
          ))}
        </div>
      </div>

      {assessment.warnings.length > 0 && (
        <div style={{ padding: '10px 12px', border: '1px solid var(--warn)', borderRadius: 6, background: 'oklch(0.82 0.14 80 / 0.08)' }}>
          <div style={{ ...sectionLabel, color: 'var(--warn)', marginBottom: 6 }}>Warnings</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
            {assessment.warnings.map((warning) => (
              <p key={warning} style={{ margin: 0, fontSize: 12.5, color: 'var(--warn)' }}>{warning}</p>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
