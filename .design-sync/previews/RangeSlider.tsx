import { useState } from 'react'
import { RangeSlider } from 'freqpred-dashboard'

function Demo({ min, max, step, initialMin, initialMax }: {
  min: number
  max: number
  step?: number
  initialMin: number
  initialMax: number
}) {
  const [range, setRange] = useState<[number, number]>([initialMin, initialMax])
  return (
    <div style={{ maxWidth: 320 }}>
      <RangeSlider
        min={min}
        max={max}
        step={step}
        valueMin={range[0]}
        valueMax={range[1]}
        onChange={(lo, hi) => setRange([lo, hi])}
      />
      <div className="mono" style={{ marginTop: 8, fontSize: 11, color: 'var(--fg-2)' }}>
        {range[0].toFixed(2)} – {range[1].toFixed(2)}
      </div>
    </div>
  )
}

export function ProbabilityRange() {
  return <Demo min={0} max={1} step={0.01} initialMin={0.15} initialMax={0.85} />
}

export function EdgeRange() {
  return <Demo min={-0.2} max={0.2} step={0.01} initialMin={-0.05} initialMax={0.1} />
}
