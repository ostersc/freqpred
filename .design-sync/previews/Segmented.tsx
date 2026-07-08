import { useState } from 'react'
import { Segmented } from 'freqpred-dashboard'

export function StringItems() {
  const [value, setValue] = useState('24h')
  return <Segmented items={['1h', '24h', '7d', '30d']} value={value} onChange={setValue} />
}

export function LabeledItems() {
  const [value, setValue] = useState('yes')
  return (
    <Segmented
      items={[{ v: 'yes', label: 'YES' }, { v: 'no', label: 'NO' }]}
      value={value}
      onChange={setValue}
    />
  )
}
