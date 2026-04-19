import { useState, useEffect } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { getStrategyConfig, updateStrategyConfig } from '../api/strategy'
import { Panel, LoadingSpinner, ErrorBanner } from '../components/ui'
import type { StrategyConfigOut } from '../api/types'

type NumericKey = keyof {
  [K in keyof StrategyConfigOut as StrategyConfigOut[K] extends number | null ? K : never]: true
}
type BoolKey = keyof {
  [K in keyof StrategyConfigOut as StrategyConfigOut[K] extends boolean ? K : never]: true
}

const NUM_FIELDS: [NumericKey, string, string?][] = [
  ['min_edge', 'Min edge', 'Minimum edge vs market mid to enter.'],
  ['min_confidence', 'Min confidence', 'Minimum confidence score from signal assessment.'],
  ['kelly_fraction', 'Kelly fraction', 'Fraction of full-Kelly sizing.'],
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
  ['stoploss_cooldown_hours', 'Stoploss cooldown (hours)', 'Prevents re-entry for N hours after stoploss.'],
  ['assessment_scale_min', 'Assessment scale min'],
  ['assessment_scale_max', 'Assessment scale max'],
  ['similar_market_min_signals', 'Similar-market min signals'],
  ['similar_market_min_trades', 'Similar-market min trades'],
]

const BOOL_FIELDS: [BoolKey, string][] = [
  ['trailing_stop', 'Trailing stop'],
  ['block_reentry_after_stoploss', 'Block re-entry after stoploss'],
]

export default function StrategyConfig() {
  const queryClient = useQueryClient()
  const { data, isLoading, error } = useQuery({
    queryKey: ['strategyConfig'],
    queryFn: getStrategyConfig,
  })

  const [edits, setEdits] = useState<Partial<StrategyConfigOut>>({})
  const [saveStatus, setSaveStatus] = useState<'idle' | 'saved' | 'error'>('idle')

  useEffect(() => { if (data) setEdits({}) }, [data])

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
  const rowStyle = (i: number): React.CSSProperties => ({
    padding: '12px 16px',
    borderTop: i > 0 ? '1px solid var(--line-soft)' : 'none',
  })

  return (
    <div className="page">
      <div className="page-head">
        <div>
          <h1 className="page-title">Strategy Config</h1>
          <div className="page-subtitle">
            Parameters for <b className="mono">{data?.name ?? '…'}</b> — changes marked <em>requires restart</em> take effect on reboot.
          </div>
        </div>
        <div className="row">
          {hasChanges && (
            <button className="btn" onClick={() => setEdits({})}>Revert</button>
          )}
          <button
            className="btn primary"
            disabled={!hasChanges || mutation.isPending}
            onClick={() => mutation.mutate(edits)}
          >
            {mutation.isPending ? 'Saving…' : 'Apply changes'}
          </button>
        </div>
      </div>

      {saveStatus === 'saved' && (
        <div className="panel" style={{ padding: '10px 16px', marginBottom: 12, background: 'var(--pos-soft)', border: '1px solid var(--pos)', borderRadius: 'var(--r-sm)', color: 'var(--pos)', fontSize: 12.5 }}>
          Configuration saved successfully.
        </div>
      )}
      {saveStatus === 'error' && (
        <div className="error-banner">Save failed. Check that the run loop is active and fields are valid.</div>
      )}

      {isLoading && <LoadingSpinner />}
      {error && (
        <ErrorBanner message={
          (error as Error).message.includes('503')
            ? 'No active strategy — start freqpred run first.'
            : String(error)
        } />
      )}

      {data && (
        <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: 12, alignItems: 'flex-start' }}>
          <Panel>
            <div style={{ display: 'grid', gridTemplateColumns: '220px 1fr' }}>
              <div style={rowStyle(0)}>
                <span>Strategy name</span>
                <div style={{ fontSize: 10, color: 'var(--fg-3)' }}>(requires restart)</div>
              </div>
              <div style={rowStyle(0)}>
                <span className="mono" style={{ fontSize: 12, color: 'var(--fg-1)' }}>{data.name}</span>
              </div>

              <div style={rowStyle(1)}>
                <span>Categories</span>
                <div style={{ fontSize: 10, color: 'var(--fg-3)' }}>(requires restart)</div>
              </div>
              <div style={rowStyle(1)}>
                <span style={{ fontSize: 12, color: 'var(--fg-1)' }}>{data.categories.join(', ')}</span>
              </div>

              {NUM_FIELDS.map(([key, label], i) => (
                <>
                  <div key={`l-${key}`} style={rowStyle(i + 2)}>
                    <span>{label}</span>
                  </div>
                  <div key={`v-${key}`} style={rowStyle(i + 2)}>
                    <input
                      className="input mono"
                      style={{ maxWidth: 180 }}
                      type="number"
                      step="any"
                      value={numVal(key)}
                      onChange={(e) => numEdit(key, e.target.value)}
                    />
                  </div>
                </>
              ))}

              {BOOL_FIELDS.map(([key, label], i) => (
                <>
                  <div key={`l-${key}`} style={rowStyle(NUM_FIELDS.length + i + 2)}>
                    <span>{label}</span>
                  </div>
                  <div key={`v-${key}`} style={rowStyle(NUM_FIELDS.length + i + 2)}>
                    <label style={{ display: 'inline-flex', alignItems: 'center', gap: 8, cursor: 'pointer', fontSize: 12.5 }}>
                      <input
                        type="checkbox"
                        checked={boolVal(key)}
                        onChange={(e) => boolEdit(key, e.target.checked)}
                      />
                      {boolVal(key) ? 'Enabled' : 'Disabled'}
                    </label>
                  </div>
                </>
              ))}
            </div>
          </Panel>

          <div className="col">
            <Panel title="Help">
              <div style={{ fontSize: 12, color: 'var(--fg-1)', lineHeight: 1.7 }}>
                <p style={{ margin: '0 0 10px 0' }}>Parameters here drive the live strategy loop.</p>
                <p style={{ margin: '0 0 10px 0' }}><b>Min edge</b> and <b>Min confidence</b> gate entries.</p>
                <p style={{ margin: '0 0 10px 0' }}><b>Kelly fraction</b> scales sizing (0.25 = quarter-Kelly).</p>
                <p style={{ margin: 0 }}><b>Stoploss cooldown</b> prevents re-entry for N hours after stoploss.</p>
              </div>
            </Panel>
          </div>
        </div>
      )}
    </div>
  )
}
