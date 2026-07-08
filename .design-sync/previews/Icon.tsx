import { Icon } from 'freqpred-dashboard'

const names = ['search', 'chev', 'chevR', 'filter', 'refresh', 'check', 'x', 'info', 'bolt', 'gear'] as const

export function AllIcons() {
  return (
    <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', color: 'var(--fg-1)' }}>
      {names.map((n) => (
        <div key={n} style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 4 }}>
          <Icon name={n} size={18} />
          <span className="mono" style={{ fontSize: 9, color: 'var(--fg-3)' }}>{n}</span>
        </div>
      ))}
    </div>
  )
}
