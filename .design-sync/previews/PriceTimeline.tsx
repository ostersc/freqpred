import { useState, type ComponentProps } from 'react'
import { PriceTimeline } from 'freqpred-dashboard'

const signals = [
  { id: 's1', created_at: '2026-06-28T09:00:00Z', estimated_probability: 0.40, market_mid_at_signal: 0.35, trigger: 'entry' },
  { id: 's2', created_at: '2026-06-29T09:00:00Z', estimated_probability: 0.44, market_mid_at_signal: 0.39, trigger: 'scheduled' },
  { id: 's3', created_at: '2026-06-30T09:00:00Z', estimated_probability: 0.50, market_mid_at_signal: 0.46, trigger: 'price_moved' },
  { id: 's4', created_at: '2026-07-01T09:00:00Z', estimated_probability: 0.57, market_mid_at_signal: 0.52, trigger: 'scheduled' },
  { id: 's5', created_at: '2026-07-02T09:00:00Z', estimated_probability: 0.63, market_mid_at_signal: 0.58, trigger: 'scheduled' },
]

function Demo(overrides: Partial<ComponentProps<typeof PriceTimeline>> = {}) {
  const [selected, setSelected] = useState<string | null>(null)
  return (
    <PriceTimeline
      signals={signals}
      entrySignalId="s1"
      entryPrice={0.35}
      currentMid={0.58}
      direction="YES"
      selectedSignalId={selected}
      onSignalClick={setSelected}
      {...overrides}
    />
  )
}

export function OpenPosition() {
  return <Demo />
}

export function ClosedPosition() {
  return <Demo exitPrice={0.61} exitTime="2026-07-03T09:00:00Z" exitReason="target_hit" />
}

export function NoDirection() {
  return <Demo direction="NO" entryPrice={0.65} currentMid={0.42} />
}
