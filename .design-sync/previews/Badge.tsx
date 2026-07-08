import { Badge } from 'freqpred-dashboard'

export function Kinds() {
  return (
    <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
      <Badge kind="pos">Size up</Badge>
      <Badge kind="neg">Size down</Badge>
      <Badge kind="warn">Wide spread</Badge>
      <Badge kind="info">Scheduled</Badge>
      <Badge kind="accent">New</Badge>
      <Badge kind="muted">Neutral</Badge>
    </div>
  )
}

export function WithDot() {
  return (
    <div style={{ display: 'flex', gap: 8 }}>
      <Badge kind="pos" dot>Live</Badge>
      <Badge kind="neg" dot>Halted</Badge>
    </div>
  )
}
