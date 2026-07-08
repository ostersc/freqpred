import { Donut } from 'freqpred-dashboard'

export function Allocation() {
  return (
    <Donut
      data={[
        { label: 'Politics', pct: 45, color: 'var(--accent)' },
        { label: 'Economics', pct: 30, color: 'var(--pos)' },
        { label: 'Other', pct: 25, color: 'var(--warn)' },
      ]}
    />
  )
}

export function Small() {
  return (
    <Donut
      size={48}
      data={[
        { label: 'YES', pct: 68, color: 'var(--pos)' },
        { label: 'NO', pct: 32, color: 'var(--neg)' },
      ]}
    />
  )
}
