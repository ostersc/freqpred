import { useState } from 'react'
import { LabeledInput } from 'freqpred-dashboard'

export function Text() {
  const [value, setValue] = useState('')
  return <LabeledInput label="Search markets" placeholder="e.g. KXFED" value={value} onChange={setValue} />
}

export function Filled() {
  const [value, setValue] = useState('KXTRUMPSAY-26JUL06')
  return <LabeledInput label="Market ticker" value={value} onChange={setValue} />
}
