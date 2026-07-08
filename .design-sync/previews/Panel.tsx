import { Panel, Badge } from 'freqpred-dashboard'

export function Basic() {
  return (
    <Panel title="Open positions">
      <p style={{ margin: 0, fontSize: 12.5, color: 'var(--fg-2)' }}>
        3 positions open across 2 markets.
      </p>
    </Panel>
  )
}

export function WithAction() {
  return (
    <Panel title="Signal feed" action={<Badge kind="accent">Live</Badge>}>
      <p style={{ margin: 0, fontSize: 12.5, color: 'var(--fg-2)' }}>
        Latest signal 4m ago — KXFED-26JUL edge +6.2%.
      </p>
    </Panel>
  )
}

export function Flush() {
  return (
    <Panel title="Recent trades" flush>
      <div style={{ padding: '10px 16px', fontSize: 12, color: 'var(--fg-1)' }}>
        Flush removes the panel body's default padding — used when the child
        (e.g. a table) manages its own edge spacing.
      </div>
    </Panel>
  )
}
