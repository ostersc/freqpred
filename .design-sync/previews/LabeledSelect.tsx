import { useState } from 'react'
import { LabeledSelect } from 'freqpred-dashboard'

export function Basic() {
  const [value, setValue] = useState('open')
  return (
    <LabeledSelect
      label="Status"
      value={value}
      onChange={setValue}
      options={[
        { value: 'open', label: 'Open' },
        { value: 'closed', label: 'Closed' },
        { value: 'all', label: 'All' },
      ]}
    />
  )
}
