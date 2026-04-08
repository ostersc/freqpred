import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
  ZAxis,
  Legend,
} from 'recharts'
import { getCalibration } from '../api/calibration'
import type { CalibrationBucketOut } from '../api/types'
import LoadingSpinner from '../components/LoadingSpinner'
import ErrorBanner from '../components/ErrorBanner'

const PRESETS = [
  { label: '7d', days: 7 },
  { label: '30d', days: 30 },
  { label: '90d', days: 90 },
  { label: 'All time', days: undefined },
] as const

type BucketPoint = CalibrationBucketOut & { _series: 'model' | 'market' }

export default function Calibration() {
  const [lookbackDays, setLookbackDays] = useState<number | undefined>(undefined)

  const { data, isLoading, error } = useQuery({
    queryKey: ['calibration', lookbackDays],
    queryFn: () => getCalibration(lookbackDays),
  })

  const modelPoints: BucketPoint[] = (data?.buckets ?? [])
    .filter((b) => b.count > 0)
    .map((b) => ({ ...b, _series: 'model' }))

  const marketPoints: BucketPoint[] = (data?.market_buckets ?? [])
    .filter((b) => b.count > 0)
    .map((b) => ({ ...b, _series: 'market' }))

  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <h1 className="text-xl font-bold text-gray-900">Calibration</h1>
        <div className="flex gap-1">
          {PRESETS.map((p) => (
            <button
              key={p.label}
              onClick={() => setLookbackDays(p.days)}
              className={`px-3 py-1 text-sm rounded border transition-colors ${
                lookbackDays === p.days
                  ? 'bg-blue-600 text-white border-blue-600'
                  : 'bg-white text-gray-600 border-gray-300 hover:border-blue-400 hover:text-blue-600'
              }`}
            >
              {p.label}
            </button>
          ))}
        </div>
      </div>

      {isLoading && <LoadingSpinner />}
      {error && <ErrorBanner message={String(error)} />}
      {data && (
        <>
          <div className="grid grid-cols-3 gap-4 mb-6">
            <div className="bg-white rounded shadow p-4">
              <div className="text-xs text-gray-500 uppercase tracking-wide mb-1">Brier Score (ours)</div>
              <div className="text-2xl font-bold text-gray-900">{data.brier_score.toFixed(4)}</div>
              <div className="text-xs text-gray-400 mt-0.5">lower is better</div>
            </div>
            <div className="bg-white rounded shadow p-4">
              <div className="text-xs text-gray-500 uppercase tracking-wide mb-1">Brier Score (market)</div>
              <div className="text-2xl font-bold text-gray-900">{data.market_brier_score.toFixed(4)}</div>
              <div className="text-xs text-gray-400 mt-0.5">baseline: market mid at signal time</div>
            </div>
            <div className="bg-white rounded shadow p-4">
              <div className="text-xs text-gray-500 uppercase tracking-wide mb-1">Samples</div>
              <div className="text-2xl font-bold text-gray-900">{data.n_samples}</div>
              <div className="text-xs text-gray-400 mt-0.5">resolved signals</div>
            </div>
          </div>

          {data.n_samples > 0 ? (
            <div className="bg-white rounded shadow p-4">
              <h2 className="text-sm font-semibold text-gray-700 mb-3">
                Calibration curve — estimated probability vs. actual resolution rate
              </h2>
              <ResponsiveContainer width="100%" height={340}>
                <ScatterChart margin={{ top: 10, right: 30, bottom: 30, left: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis
                    dataKey="mean_estimated_prob"
                    type="number"
                    domain={[0, 1]}
                    tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
                    label={{ value: 'Estimated probability', position: 'insideBottom', offset: -15, fontSize: 12 }}
                  />
                  <YAxis
                    dataKey="actual_resolution_rate"
                    type="number"
                    domain={[0, 1]}
                    tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
                    label={{ value: 'Actual resolution rate', angle: -90, position: 'insideLeft', offset: 10, fontSize: 12 }}
                  />
                  <ZAxis dataKey="count" range={[40, 400]} />
                  <Tooltip
                    content={({ payload }) => {
                      if (!payload?.length) return null
                      const d = payload[0]?.payload as BucketPoint
                      return (
                        <div className="bg-white border rounded p-2 text-xs shadow">
                          <div className="font-semibold mb-1" style={{ color: d._series === 'model' ? '#3b82f6' : '#f97316' }}>
                            {d._series === 'model' ? 'Model' : 'Market'}
                          </div>
                          <div>Est. prob: {(d.mean_estimated_prob * 100).toFixed(1)}%</div>
                          <div>Resolution rate: {(d.actual_resolution_rate * 100).toFixed(1)}%</div>
                          <div>Count: {d.count}</div>
                        </div>
                      )
                    }}
                  />
                  <Legend
                    verticalAlign="top"
                    align="right"
                    payload={[
                      { value: 'Model', type: 'circle', color: '#3b82f6' },
                      { value: 'Market', type: 'circle', color: '#f97316' },
                    ]}
                  />
                  <ReferenceLine
                    segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]}
                    stroke="#94a3b8"
                    strokeDasharray="6 3"
                    label={{ value: 'Perfect calibration', position: 'insideTopLeft', fontSize: 11, fill: '#94a3b8' }}
                  />
                  <Scatter
                    name="Model"
                    data={modelPoints}
                    fill="#3b82f6"
                    fillOpacity={0.7}
                  />
                  <Scatter
                    name="Market"
                    data={marketPoints}
                    fill="#f97316"
                    fillOpacity={0.7}
                  />
                </ScatterChart>
              </ResponsiveContainer>
              <div className="text-xs text-gray-400 mt-2 text-center">Bubble size proportional to sample count</div>
            </div>
          ) : (
            <div className="bg-white rounded shadow p-8 text-center text-gray-400">
              No resolved signals yet — calibration chart will appear once markets resolve.
            </div>
          )}
        </>
      )}
    </div>
  )
}
