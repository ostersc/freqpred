import { Sparkline } from 'freqpred-dashboard'

export function Uptrend() {
  return <Sparkline data={[40, 42, 41, 45, 48, 47, 52, 55, 58]} />
}

export function Downtrend() {
  return <Sparkline data={[60, 58, 55, 56, 50, 48, 44, 41, 38]} color="var(--neg)" />
}

export function NoFill() {
  return <Sparkline data={[50, 53, 49, 51, 55, 52]} fill={false} w={120} h={30} />
}
