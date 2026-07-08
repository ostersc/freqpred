import { ProbBar } from 'freqpred-dashboard'

export function OurEdgeHigher() {
  return <ProbBar ours={0.68} market={0.54} />
}

export function OurEdgeLower() {
  return <ProbBar ours={0.32} market={0.47} />
}
