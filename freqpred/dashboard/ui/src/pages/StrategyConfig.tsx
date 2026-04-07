import { useState, useEffect } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { getStrategyConfig, updateStrategyConfig } from '../api/strategy'
import LoadingSpinner from '../components/LoadingSpinner'
import ErrorBanner from '../components/ErrorBanner'
import type { StrategyConfigOut } from '../api/types'

type NumericKey = keyof {
  [K in keyof StrategyConfigOut as StrategyConfigOut[K] extends number | null ? K : never]: true
}
type BoolKey = keyof {
  [K in keyof StrategyConfigOut as StrategyConfigOut[K] extends boolean ? K : never]: true
}

function Field({ label, children, immutable }: { label: string; children: React.ReactNode; immutable?: boolean }) {
  return (
    <div className="flex items-center border-b last:border-0 py-2 gap-4">
      <div className="w-56 text-sm text-gray-600 flex-shrink-0">
        {label}
        {immutable && <span className="ml-1 text-xs text-gray-400">(requires restart)</span>}
      </div>
      <div className="flex-1">{children}</div>
    </div>
  )
}

export default function StrategyConfig() {
  const queryClient = useQueryClient()
  const { data, isLoading, error } = useQuery({
    queryKey: ['strategyConfig'],
    queryFn: getStrategyConfig,
  })

  const [edits, setEdits] = useState<Partial<StrategyConfigOut>>({})
  const [saveStatus, setSaveStatus] = useState<'idle' | 'saved' | 'error'>('idle')

  useEffect(() => {
    if (data) setEdits({})
  }, [data])

  const mutation = useMutation({
    mutationFn: updateStrategyConfig,
    onSuccess: (updated) => {
      queryClient.setQueryData(['strategyConfig'], updated)
      setEdits({})
      setSaveStatus('saved')
      setTimeout(() => setSaveStatus('idle'), 3000)
    },
    onError: () => setSaveStatus('error'),
  })

  function numEdit(key: NumericKey, val: string) {
    const n = parseFloat(val)
    if (!isNaN(n)) setEdits((e) => ({ ...e, [key]: n }))
  }

  function boolEdit(key: BoolKey, val: boolean) {
    setEdits((e) => ({ ...e, [key]: val }))
  }

  function numVal(key: NumericKey): string {
    const v = key in edits ? (edits as Record<string, number | null>)[key] : data?.[key]
    return v !== null && v !== undefined ? String(v) : ''
  }

  function boolVal(key: BoolKey): boolean {
    return key in edits ? Boolean((edits as Record<string, boolean>)[key]) : Boolean(data?.[key])
  }

  const hasChanges = Object.keys(edits).length > 0

  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <h1 className="text-xl font-bold text-gray-900">Strategy Config</h1>
        {hasChanges && (
          <button
            className="px-4 py-1.5 bg-blue-600 text-white text-sm rounded hover:bg-blue-700 disabled:opacity-50"
            disabled={mutation.isPending}
            onClick={() => mutation.mutate(edits)}
          >
            {mutation.isPending ? 'Saving…' : 'Save changes'}
          </button>
        )}
      </div>
      {saveStatus === 'saved' && (
        <div className="mb-3 px-3 py-2 bg-green-50 border border-green-200 text-green-800 text-sm rounded">
          Configuration saved successfully.
        </div>
      )}
      {saveStatus === 'error' && (
        <div className="mb-3 px-3 py-2 bg-red-50 border border-red-200 text-red-800 text-sm rounded">
          Save failed. Check that the run loop is active and fields are valid.
        </div>
      )}
      {isLoading && <LoadingSpinner />}
      {error && (
        <ErrorBanner message={(error as Error).message.includes('503')
          ? 'No active strategy — start freqpred run first.'
          : String(error)} />
      )}
      {data && (
        <div className="bg-white rounded shadow divide-y px-4">
          <Field label="Strategy name" immutable>
            <span className="text-sm font-mono text-gray-800">{data.name}</span>
          </Field>
          <Field label="Categories" immutable>
            <span className="text-sm text-gray-800">{data.categories.join(', ')}</span>
          </Field>

          {(
            [
              ['min_edge', 'Min edge'],
              ['min_confidence', 'Min confidence'],
              ['kelly_fraction', 'Kelly fraction'],
              ['max_exposure_per_market', 'Max exposure per market'],
              ['min_volume_24h', 'Min 24h volume ($)'],
              ['max_days_to_close', 'Max days to close'],
              ['min_days_to_close', 'Min days to close'],
              ['stoploss', 'Stoploss'],
              ['trailing_stop_positive', 'Trailing stop positive'],
              ['trailing_stop_positive_offset', 'Trailing stop positive offset'],
              ['min_mid_price', 'Min mid price'],
              ['max_mid_price', 'Max mid price'],
              ['max_spread', 'Max spread'],
              ['stoploss_cooldown_hours', 'Stoploss cooldown (hours)'],
            ] as [NumericKey, string][]
          ).map(([key, label]) => (
            <Field key={key} label={label}>
              <input
                type="number"
                step="any"
                value={numVal(key)}
                onChange={(e) => numEdit(key, e.target.value)}
                className="border rounded px-2 py-1 text-sm w-36 focus:outline-none focus:ring-2 focus:ring-blue-300"
              />
            </Field>
          ))}

          {([
            ['trailing_stop', 'Trailing stop'],
            ['block_reentry_after_stoploss', 'Block re-entry after stoploss'],
          ] as [BoolKey, string][]).map(([key, label]) => (
            <Field key={key} label={label}>
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={boolVal(key)}
                  onChange={(e) => boolEdit(key, e.target.checked)}
                  className="h-4 w-4 rounded"
                />
                <span className="text-sm text-gray-700">{boolVal(key) ? 'Enabled' : 'Disabled'}</span>
              </label>
            </Field>
          ))}
        </div>
      )}
    </div>
  )
}
