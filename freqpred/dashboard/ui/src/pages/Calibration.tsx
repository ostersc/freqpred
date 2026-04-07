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
} from 'recharts'
import { getCalibration } from '../api/calibration'
import LoadingSpinner from '../components/LoadingSpinner'
import ErrorBanner from '../components/ErrorBanner'

export default function Calibration() {
  const { data, isLoading, error } = useQuery({
    queryKey: ['calibration'],
    queryFn: getCalibration,
  })

  return (
    <div>
      <h1 className="text-xl font-bold text-gray-900 mb-4">Calibration</h1>
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
              <h2 className="text-sm font-semibold text-gray-700 mb-3">Calibration curve — estimated probability vs. actual resolution rate</h2>
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
                    formatter={(value: number, name: string) => [
                      `${(value * 100).toFixed(1)}%`,
                      name === 'actual_resolution_rate' ? 'Resolution rate' : name,
                    ]}
                    labelFormatter={() => ''}
                    content={({ payload }) => {
                      if (!payload?.length) return null
                      const d = payload[0]?.payload as { mean_estimated_prob: number; actual_resolution_rate: number; count: number }
                      return (
                        <div className="bg-white border rounded p-2 text-xs shadow">
                          <div>Est. prob: {(d.mean_estimated_prob * 100).toFixed(1)}%</div>
                          <div>Resolution rate: {(d.actual_resolution_rate * 100).toFixed(1)}%</div>
                          <div>Count: {d.count}</div>
                        </div>
                      )
                    }}
                  />
                  <ReferenceLine
                    segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]}
                    stroke="#94a3b8"
                    strokeDasharray="6 3"
                    label={{ value: 'Perfect calibration', position: 'insideTopLeft', fontSize: 11, fill: '#94a3b8' }}
                  />
                  <Scatter
                    data={data.buckets.filter((b) => b.count > 0)}
                    fill="#3b82f6"
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
