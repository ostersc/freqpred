import { MiniStat } from 'freqpred-dashboard'

export function Basic() {
  return (
    <div style={{ display: 'flex', gap: 16 }}>
      <MiniStat label="Contracts" value={120} />
      <MiniStat label="Avg entry" value="42.5¢" />
      <MiniStat label="Edge" value="+6.1%" />
    </div>
  )
}
