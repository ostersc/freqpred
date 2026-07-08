import { SignalHistoryChart } from 'freqpred-dashboard'

const signals = [
  { estimated_probability: 0.42, market_mid_at_signal: 0.38, created_at: '2026-06-30T09:00:00Z', trigger: 'scheduled' },
  { estimated_probability: 0.46, market_mid_at_signal: 0.40, created_at: '2026-07-01T09:00:00Z', trigger: 'price_moved' },
  { estimated_probability: 0.55, market_mid_at_signal: 0.44, created_at: '2026-07-02T14:00:00Z', trigger: 'entry', reasoning: 'Strong source consensus after committee vote leak.' },
  { estimated_probability: 0.58, market_mid_at_signal: 0.50, created_at: '2026-07-03T09:00:00Z', trigger: 'manual', reasoning: 'Manual re-check after a market-moving headline.' },
  { estimated_probability: 0.61, market_mid_at_signal: 0.55, created_at: '2026-07-04T09:00:00Z', trigger: 'market_update' },
  { estimated_probability: 0.64, market_mid_at_signal: 0.60, created_at: '2026-07-05T09:00:00Z', trigger: 'scheduled' },
]

export function Default() {
  return <SignalHistoryChart signals={signals} />
}

export function ShortHistory() {
  return <SignalHistoryChart signals={[signals[0]]} />
}
