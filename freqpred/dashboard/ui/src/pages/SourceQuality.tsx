import { useMemo, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { getSourceQuality } from '../api/sourceQuality'
import type { SourceQualityScoreOut } from '../api/types'
import ErrorBanner from '../components/ErrorBanner'
import LoadingSpinner from '../components/LoadingSpinner'

const PRESETS = [
  { label: '7d', days: 7 },
  { label: '30d', days: 30 },
  { label: '90d', days: 90 },
  { label: 'All time', days: undefined },
] as const

type CategoryFilter = 'all' | '__global__' | string
type SortColumn =
  | 'source_name'
  | 'market_category'
  | 'weighted_brier'
  | 'overall_brier'
  | 'delta_vs_overall'
  | 'n_signals'
  | 'total_doc_uses'
  | 'computed_at'
type SortDirection = 'asc' | 'desc'

function fmtBrier(value: number) {
  return value.toFixed(3)
}

function fmtSignedBrierDelta(value: number) {
  const rendered = value.toFixed(3)
  return value > 0 ? `+${rendered}` : rendered
}

function fmtAge(iso: string) {
  const ms = Date.now() - new Date(iso).getTime()
  const hours = Math.floor(ms / 3_600_000)
  if (hours < 1) return 'just now'
  if (hours < 24) return `${hours}h ago`
  const days = Math.floor(hours / 24)
  return `${days}d ago`
}

function categoryLabel(category: string | null) {
  return category ?? 'All markets'
}

function trendDelta(row: SourceQualityScoreOut) {
  return row.weighted_brier - row.overall_brier
}

function deltaTone(delta: number) {
  const absDelta = Math.abs(delta)
  if (delta < 0) {
    if (absDelta >= 0.1) return 'bg-green-300 text-green-950'
    if (absDelta >= 0.05) return 'bg-green-200 text-green-900'
    return 'bg-green-100 text-green-800'
  }
  if (delta > 0) {
    if (absDelta >= 0.1) return 'bg-red-300 text-red-950'
    if (absDelta >= 0.05) return 'bg-red-200 text-red-900'
    return 'bg-red-100 text-red-800'
  }
  return 'bg-gray-100 text-gray-700'
}

function DeltaBadge({ row }: { row: SourceQualityScoreOut }) {
  const delta = trendDelta(row)
  return (
    <span
      className={`inline-flex min-w-20 justify-center rounded-full px-2 py-1 text-xs font-semibold ${deltaTone(delta)}`}
      title="Weighted Brier minus overall Brier. Negative helps; positive harms."
    >
      {fmtSignedBrierDelta(delta)}
    </span>
  )
}

function compareRows(a: SourceQualityScoreOut, b: SourceQualityScoreOut, column: SortColumn) {
  switch (column) {
    case 'source_name':
      return a.source_name.localeCompare(b.source_name)
    case 'market_category':
      return categoryLabel(a.market_category).localeCompare(categoryLabel(b.market_category))
    case 'weighted_brier':
      return a.weighted_brier - b.weighted_brier
    case 'overall_brier':
      return a.overall_brier - b.overall_brier
    case 'delta_vs_overall':
      return trendDelta(a) - trendDelta(b)
    case 'n_signals':
      return a.n_signals - b.n_signals
    case 'total_doc_uses':
      return a.total_doc_uses - b.total_doc_uses
    case 'computed_at':
      return new Date(a.computed_at).getTime() - new Date(b.computed_at).getTime()
  }
}

export default function SourceQuality() {
  const [category, setCategory] = useState<CategoryFilter>('all')
  const [lookbackDays, setLookbackDays] = useState<number | undefined>(90)
  const [sortColumn, setSortColumn] = useState<SortColumn>('weighted_brier')
  const [sortDirection, setSortDirection] = useState<SortDirection>('asc')
  const { data, isLoading, error } = useQuery({
    queryKey: ['sourceQuality', lookbackDays],
    queryFn: () => getSourceQuality({ lookback_days: lookbackDays }),
    refetchInterval: 300_000,
  })

  const rows = data?.items ?? []
  const categories = useMemo(() => {
    const values = new Set<string>()
    let hasGlobal = false
    for (const row of rows) {
      if (row.market_category === null) {
        hasGlobal = true
      } else {
        values.add(row.market_category)
      }
    }
    return {
      categories: Array.from(values).sort(),
      hasGlobal,
    }
  }, [rows])

  const filteredRows = useMemo(() => {
    const filtered = category === 'all'
      ? rows
      : category === '__global__'
        ? rows.filter((row) => row.market_category === null)
        : rows.filter((row) => row.market_category === category)

    const direction = sortDirection === 'asc' ? 1 : -1
    return [...filtered].sort((a, b) => {
      const primary = compareRows(a, b, sortColumn)
      if (primary !== 0) return primary * direction

      const bySource = a.source_name.localeCompare(b.source_name)
      if (bySource !== 0) return bySource

      return categoryLabel(a.market_category).localeCompare(categoryLabel(b.market_category))
    })
  }, [category, rows, sortColumn, sortDirection])

  const latestComputedAt = rows.reduce<string | null>((latest, row) => {
    if (latest === null) return row.computed_at
    return new Date(row.computed_at).getTime() > new Date(latest).getTime()
      ? row.computed_at
      : latest
  }, null)
  const hoursStale = latestComputedAt !== null
    ? (Date.now() - new Date(latestComputedAt).getTime()) / 3_600_000
    : null

  function toggleSort(column: SortColumn) {
    if (sortColumn === column) {
      setSortDirection((current) => current === 'asc' ? 'desc' : 'asc')
      return
    }
    setSortColumn(column)
    setSortDirection(column === 'weighted_brier' ? 'asc' : 'desc')
  }

  function sortIndicator(column: SortColumn) {
    if (sortColumn !== column) return '↕'
    return sortDirection === 'asc' ? '↑' : '↓'
  }

  function SortHeader({
    column,
    label,
    align = 'left',
  }: {
    column: SortColumn
    label: string
    align?: 'left' | 'center' | 'right'
  }) {
    const alignmentClass = align === 'right'
      ? 'justify-end text-right'
      : align === 'center'
        ? 'justify-center text-center'
        : 'justify-start text-left'

    return (
      <th className={`px-3 py-2 ${align === 'right' ? 'text-right' : align === 'center' ? 'text-center' : ''}`}>
        <button
          type="button"
          onClick={() => toggleSort(column)}
          className={`inline-flex w-full items-center gap-1 hover:text-gray-900 ${alignmentClass}`}
        >
          <span>{label}</span>
          <span className="text-[10px] text-gray-400">{sortIndicator(column)}</span>
        </button>
      </th>
    )
  }

  return (
    <div>
      <div className="flex flex-wrap items-end justify-between gap-3 mb-4">
        <div>
          <h1 className="text-xl font-bold text-gray-900">Source Quality</h1>
          <p className="text-sm text-gray-500">Lower weighted Brier is better. Scores refresh from the metrics scheduler.</p>
        </div>
        <div className="flex flex-wrap items-end gap-3">
          <label className="text-sm text-gray-600">
            <span className="mr-2">Category</span>
            <select
              value={category}
              onChange={(e) => setCategory(e.target.value)}
              className="rounded border border-gray-300 bg-white px-3 py-2"
            >
              <option value="all">All</option>
              {categories.hasGlobal && <option value="__global__">All markets</option>}
              {categories.categories.map((value) => (
                <option key={value} value={value}>
                  {value}
                </option>
              ))}
            </select>
          </label>
          <div className="flex gap-1">
            {PRESETS.map((preset) => (
              <button
                key={preset.label}
                type="button"
                onClick={() => setLookbackDays(preset.days)}
                className={`px-3 py-1 text-sm rounded border transition-colors ${
                  lookbackDays === preset.days
                    ? 'bg-blue-600 text-white border-blue-600'
                    : 'bg-white text-gray-600 border-gray-300 hover:border-blue-400 hover:text-blue-600'
                }`}
              >
                {preset.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      {isLoading && <LoadingSpinner />}
      {error && <ErrorBanner message={String(error)} />}

      {hoursStale !== null && hoursStale > 24 && (
        <div className="mb-4 rounded border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-900">
          Scores last refreshed {Math.floor(hoursStale)} hours ago. Run the metrics scheduler to update.
        </div>
      )}

      {data && (
        <div className="bg-white rounded shadow overflow-x-auto">
          <table className="w-full text-left text-sm">
            <thead className="bg-gray-100 text-xs uppercase tracking-wide text-gray-600">
              <tr>
                <SortHeader column="source_name" label="Source" />
                <SortHeader column="market_category" label="Category" />
                <SortHeader column="weighted_brier" label="Weighted Brier" align="right" />
                <SortHeader column="overall_brier" label="Overall Brier" align="right" />
                <SortHeader column="delta_vs_overall" label="Delta vs Overall" align="center" />
                <SortHeader column="n_signals" label="Signals" align="right" />
                <SortHeader column="total_doc_uses" label="Doc Uses" align="right" />
                <SortHeader column="computed_at" label="Snapshot Age" align="right" />
              </tr>
            </thead>
            <tbody>
              {filteredRows.map((row) => (
                <tr key={`${row.source_name}:${row.market_category ?? 'all'}`} className="border-t">
                  <td className="px-3 py-2 font-medium text-gray-800">{row.source_name}</td>
                  <td className="px-3 py-2 text-gray-600">{categoryLabel(row.market_category)}</td>
                  <td className="px-3 py-2 text-right">{fmtBrier(row.weighted_brier)}</td>
                  <td className="px-3 py-2 text-right text-gray-600">{fmtBrier(row.overall_brier)}</td>
                  <td className="px-3 py-2 text-center"><DeltaBadge row={row} /></td>
                  <td className="px-3 py-2 text-right">{row.n_signals}</td>
                  <td className="px-3 py-2 text-right">{row.total_doc_uses}</td>
                  <td className="px-3 py-2 text-right text-gray-500">{fmtAge(row.computed_at)}</td>
                </tr>
              ))}
              {filteredRows.length === 0 && (
                <tr>
                  <td colSpan={8} className="px-3 py-8 text-center text-gray-400">
                    No source-quality snapshots are available for this filter yet.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
