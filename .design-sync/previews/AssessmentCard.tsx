import { AssessmentCard } from 'freqpred-dashboard'

const base = {
  trust_score: 0.82,
  size_multiplier: 1.35,
  verdict: 'size_up',
  reasoning: 'Source mix skews toward high-reliability outlets (Reuters, AP) with no conflicting reporting in the retrieval window. Family calibration is strong.',
  key_factors: ['high_source_diversity', 'strong_family_calibration', 'no_conflicting_reports'],
  warnings: [] as string[],
  source_breakdown: [
    { source_name: 'Reuters', document_share: 0.42, delta_vs_overall: -0.03 },
    { source_name: 'Associated Press', document_share: 0.31, delta_vs_overall: -0.02 },
    { source_name: 'Local blog aggregator', document_share: 0.12, delta_vs_overall: 0.05 },
  ],
  similar_market_summary: {
    available: true,
    family_match: { resolved_signals: 46, family_signal_delta_vs_overall: -0.018 },
    exact_question_subset: { resolved_signals: 9, small_sample: true },
    strategy_trade_history: { closed_trades: 21, win_rate: 0.62 },
  },
  llm_query_id: 4821,
  created_at: '2026-07-05T14:22:00Z',
}

export function SizeUp() {
  return <AssessmentCard assessment={base} />
}

export function SizeDownWithWarnings() {
  return (
    <AssessmentCard
      assessment={{
        ...base,
        verdict: 'size_down',
        trust_score: 0.41,
        size_multiplier: 0.6,
        reasoning: 'Retrieval window is dominated by a single low-reliability aggregator; no corroborating high-trust sources were found.',
        key_factors: ['low_source_diversity', 'single_source_dominance'],
        warnings: [
          'Trust score is below the 0.5 sizing floor for this strategy.',
          'Similar-market family has fewer than 10 resolved signals.',
        ],
        similar_market_summary: { available: false, reason: 'insufficient_matched_history' },
        llm_query_id: null,
      }}
    />
  )
}

export function NoAssessment() {
  return <AssessmentCard assessment={null} />
}
