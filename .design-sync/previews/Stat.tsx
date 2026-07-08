import { Stat, Sparkline } from 'freqpred-dashboard'

export function Basic() {
  return <Stat label="Open P&L" value="+$482.10" />
}

export function WithDelta() {
  return <Stat label="Win rate" value="61.4%" delta="+2.1%" deltaKind="pos" sub="last 30 signals" />
}

export function NegativeDelta() {
  return <Stat label="Daily drawdown" value="-$120.40" delta="-3.2%" deltaKind="neg" />
}

export function WithSparkline() {
  return (
    <Stat
      label="Equity curve"
      value="$14,382"
      sub="7d"
      spark={<Sparkline data={[50, 52, 49, 55, 58, 56, 61, 64]} />}
      accent="var(--accent)"
    />
  )
}
