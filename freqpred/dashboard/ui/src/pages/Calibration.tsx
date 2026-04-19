import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { getCalibration } from '../api/calibration'
import { Stat, Panel, Segmented, LoadingSpinner, ErrorBanner, LabeledSelect } from '../components/ui'

const PRESETS = [
  { v: '7' as const,   label: '7d' },
  { v: '30' as const,  label: '30d' },
  { v: '90' as const,  label: '90d' },
  { v: 'all' as const, label: 'All time' },
]
type Preset = '7' | '30' | '90' | 'all'

function presetDays(p: Preset): number | undefined {
  if (p === 'all') return undefined
  return Number(p)
}

export default function Calibration() {
  const [preset, setPreset] = useState<Preset>('all')
  const [category, setCategory] = useState('all')

  const { data, isLoading, error } = useQuery({
    queryKey: ['calibration', preset, category],
    queryFn: () => getCalibration(presetDays(preset), category === 'all' ? undefined : category),
  })

  const W = 1200, H = 480, pad = 60
  const toX = (v: number) => pad + v * (W - pad * 2)
  const toY = (v: number) => H - pad - v * (H - pad * 2)

  return (
    <div className="page">
      <div className="page-head">
        <div>
          <h1 className="page-title">Calibration</h1>
          <div className="page-subtitle">How well our signals track the world. Lower Brier score → better.</div>
        </div>
        <div className="row" style={{ alignItems: 'flex-end' }}>
          <LabeledSelect
            label="Category"
            value={category}
            onChange={setCategory}
            options={[
              { value: 'all', label: 'All' },
              ...(data?.available_categories ?? []).map((c) => ({ value: c, label: c })),
            ]}
          />
          <div style={{ display: 'flex', flexDirection: 'column' }}>
            <label style={{ fontSize: 10.5, textTransform: 'uppercase', letterSpacing: '0.08em', color: 'transparent', marginBottom: 5, userSelect: 'none' }}>Range</label>
            <Segmented items={PRESETS} value={preset} onChange={setPreset} />
          </div>
        </div>
      </div>

      {isLoading && <LoadingSpinner />}
      {error && <ErrorBanner message={String(error)} />}

      {data && (
        <>
          <div className="grid grid-3" style={{ marginBottom: 12 }}>
            <Stat label="Brier score (ours)" value={data.brier_score.toFixed(4)} sub="lower is better" />
            <Stat label="Brier score (market)" value={data.market_brier_score.toFixed(4)} sub="baseline: market mid at signal time" />
            <Stat label="Samples" value={data.n_samples.toLocaleString()} sub="resolved signals" />
          </div>

          {data.n_samples > 0 ? (
            <Panel title="Calibration curve — estimated probability vs. actual resolution rate">
              <svg width="100%" viewBox={`0 0 ${W} ${H}`} style={{ display: 'block' }}>
                {[0, 0.25, 0.5, 0.75, 1].map((t) => (
                  <g key={t}>
                    <line x1={toX(t)} x2={toX(t)} y1={pad} y2={H - pad} stroke="var(--line-soft)" strokeDasharray="2 4" />
                    <line x1={pad} x2={W - pad} y1={toY(t)} y2={toY(t)} stroke="var(--line-soft)" strokeDasharray="2 4" />
                    <text x={toX(t)} y={H - pad + 18} fontSize="11" fill="var(--fg-2)" textAnchor="middle" fontFamily="var(--f-mono)">{(t * 100).toFixed(0)}%</text>
                    <text x={pad - 10} y={toY(t) + 4} fontSize="11" fill="var(--fg-2)" textAnchor="end" fontFamily="var(--f-mono)">{(t * 100).toFixed(0)}%</text>
                  </g>
                ))}
                <line x1={toX(0)} y1={toY(0)} x2={toX(1)} y2={toY(1)} stroke="var(--fg-3)" strokeDasharray="4 4" strokeWidth="1" />
                {data.market_buckets.filter((b) => b.count > 0).map((b, i) => (
                  <circle key={'m' + i} cx={toX(b.mean_estimated_prob)} cy={toY(b.actual_resolution_rate)}
                    r={4 + Math.sqrt(b.count) * 0.8} fill="var(--warn)" opacity="0.55" />
                ))}
                {data.buckets.filter((b) => b.count > 0).map((b, i) => (
                  <circle key={'o' + i} cx={toX(b.mean_estimated_prob)} cy={toY(b.actual_resolution_rate)}
                    r={4 + Math.sqrt(b.count) * 0.8} fill="var(--accent)" opacity="0.8" />
                ))}
                <text x={W / 2} y={H - 16} fontSize="12" fill="var(--fg-2)" textAnchor="middle">Estimated probability</text>
                <text x={18} y={H / 2} fontSize="12" fill="var(--fg-2)" textAnchor="middle" transform={`rotate(-90 18 ${H / 2})`}>Actual resolution rate</text>
                <g transform={`translate(${W - pad - 220},${H - pad - 20})`}>
                  <rect x={-10} y={-16} width={230} height={36} fill="var(--bg-1)" stroke="var(--line)" rx={6} />
                  <line x1={0} y1={0} x2={18} y2={0} stroke="var(--fg-3)" strokeDasharray="4 4" />
                  <text x={24} y={4} fontSize="11" fill="var(--fg-2)">Perfect calibration</text>
                  <circle cx={130} cy={0} r={4} fill="var(--accent)" />
                  <text x={140} y={4} fontSize="11" fill="var(--fg-1)">Model</text>
                  <circle cx={178} cy={0} r={4} fill="var(--warn)" />
                  <text x={188} y={4} fontSize="11" fill="var(--fg-1)">Market</text>
                </g>
              </svg>
              <div style={{ textAlign: 'center', color: 'var(--fg-3)', fontSize: 11, marginTop: 8 }}>
                Bubble size proportional to sample count
              </div>
            </Panel>
          ) : (
            <div className="panel" style={{ padding: '40px', textAlign: 'center', color: 'var(--fg-3)' }}>
              No resolved signals yet — calibration chart will appear once markets resolve.
            </div>
          )}
        </>
      )}
    </div>
  )
}
