import { useMemo, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { getSourceQuality } from '../api/sourceQuality'
import type { SourceQualityScoreOut } from '../api/types'
import { Badge, Panel, Segmented, LoadingSpinner, ErrorBanner, WarnBanner, LabeledSelect } from '../components/ui'

const PRESETS = [
  { v: '7' as const,   label: '7d' },
  { v: '30' as const,  label: '30d' },
  { v: '90' as const,  label: '90d' },
  { v: 'all' as const, label: 'All time' },
]
type Preset = '7' | '30' | '90' | 'all'

type SortCol = 'source_name' | 'market_category' | 'weighted_brier' | 'overall_brier' | 'delta' | 'n_signals' | 'total_doc_uses' | 'computed_at'
type SortDir = 'asc' | 'desc'

function delta(row: SourceQualityScoreOut) {
  return row.weighted_brier - row.overall_brier
}

function deltaKind(d: number): 'pos' | 'neg' | 'muted' {
  if (d < -0.01) return 'pos'
  if (d > 0.001) return 'neg'
  return 'muted'
}

function fmtAge(iso: string): string {
  const hours = (Date.now() - new Date(iso).getTime()) / 3_600_000
  if (hours < 1) return 'just now'
  if (hours < 24) return `${Math.floor(hours)}h ago`
  return `${Math.floor(hours / 24)}d ago`
}

export default function SourceQuality() {
  const [preset, setPreset] = useState<Preset>('90')
  const [category, setCategory] = useState('all')
  const [sortCol, setSortCol] = useState<SortCol>('weighted_brier')
  const [sortDir, setSortDir] = useState<SortDir>('asc')

  const { data, isLoading, error } = useQuery({
    queryKey: ['sourceQuality', preset],
    queryFn: () => getSourceQuality({ lookback_days: preset === 'all' ? undefined : Number(preset) }),
    refetchInterval: 300_000,
  })

  const rows = data?.items ?? []

  const categories = useMemo(() => {
    const vals = new Set<string>()
    let hasGlobal = false
    for (const r of rows) {
      if (r.market_category === null) hasGlobal = true
      else vals.add(r.market_category)
    }
    return { list: Array.from(vals).sort(), hasGlobal }
  }, [rows])

  const filtered = useMemo(() => {
    const f = category === 'all' ? rows
      : category === '__global__' ? rows.filter((r) => r.market_category === null)
        : rows.filter((r) => r.market_category === category)
    const dir = sortDir === 'asc' ? 1 : -1
    return [...f].sort((a, b) => {
      const map: Record<SortCol, number> = {
        source_name: a.source_name.localeCompare(b.source_name),
        market_category: (a.market_category ?? 'All').localeCompare(b.market_category ?? 'All'),
        weighted_brier: a.weighted_brier - b.weighted_brier,
        overall_brier: a.overall_brier - b.overall_brier,
        delta: delta(a) - delta(b),
        n_signals: a.n_signals - b.n_signals,
        total_doc_uses: a.total_doc_uses - b.total_doc_uses,
        computed_at: new Date(a.computed_at).getTime() - new Date(b.computed_at).getTime(),
      }
      return map[sortCol] * dir
    })
  }, [category, rows, sortCol, sortDir])

  const latestAt = rows.reduce<string | null>((l, r) => l === null ? r.computed_at : (new Date(r.computed_at) > new Date(l) ? r.computed_at : l), null)
  const hoursStale = latestAt ? (Date.now() - new Date(latestAt).getTime()) / 3_600_000 : null

  function hdr(col: SortCol, label: string, right = false) {
    const indicator = sortCol === col ? (sortDir === 'asc' ? ' ↑' : ' ↓') : ''
    return (
      <th className={right ? 'r' : ''}>
        <button onClick={() => {
          if (sortCol === col) setSortDir((d) => d === 'asc' ? 'desc' : 'asc')
          else { setSortCol(col); setSortDir('asc') }
        }}>
          {label}{indicator}
        </button>
      </th>
    )
  }

  const catOptions = [
    { value: 'all', label: 'All' },
    ...(categories.hasGlobal ? [{ value: '__global__', label: 'All markets' }] : []),
    ...categories.list.map((c) => ({ value: c, label: c })),
  ]

  return (
    <div className="page">
      <div className="page-head">
        <div>
          <h1 className="page-title">Source Quality</h1>
          <div className="page-subtitle">Lower weighted Brier is better. Scores refresh from the metrics scheduler.</div>
        </div>
        <div className="row" style={{ alignItems: 'flex-end' }}>
          <LabeledSelect label="Category" value={category} onChange={setCategory} options={catOptions} />
          <div style={{ display: 'flex', flexDirection: 'column' }}>
            <label style={{ fontSize: 10.5, textTransform: 'uppercase', letterSpacing: '0.08em', color: 'transparent', marginBottom: 5, userSelect: 'none' }}>Range</label>
            <Segmented items={PRESETS} value={preset} onChange={setPreset} />
          </div>
        </div>
      </div>

      {isLoading && <LoadingSpinner />}
      {error && <ErrorBanner message={String(error)} />}
      {hoursStale !== null && hoursStale > 24 && (
        <WarnBanner message={`Scores last refreshed ${Math.floor(hoursStale)} hours ago. Run the metrics scheduler to update.`} />
      )}

      {data && (
        <Panel flush>
          <table className="tbl">
            <thead>
              <tr>
                {hdr('source_name', 'Source')}
                {hdr('market_category', 'Category')}
                {hdr('weighted_brier', 'Weighted Brier', true)}
                {hdr('overall_brier', 'Overall Brier', true)}
                {hdr('delta', 'Delta vs Overall', true)}
                {hdr('n_signals', 'Signals', true)}
                {hdr('total_doc_uses', 'Doc Uses', true)}
                {hdr('computed_at', 'Snapshot Age', true)}
              </tr>
            </thead>
            <tbody>
              {filtered.map((row) => {
                const d = delta(row)
                return (
                  <tr key={`${row.source_name}:${row.market_category ?? 'all'}`}>
                    <td style={{ fontWeight: 500 }}>{row.source_name}</td>
                    <td className="dim">{row.market_category ?? 'All markets'}</td>
                    <td className="r">{row.weighted_brier.toFixed(3)}</td>
                    <td className="r">{row.overall_brier.toFixed(3)}</td>
                    <td className="r">
                      <Badge kind={deltaKind(d)}>{d >= 0 ? '+' : ''}{d.toFixed(3)}</Badge>
                    </td>
                    <td className="r">{row.n_signals}</td>
                    <td className="r">{row.total_doc_uses}</td>
                    <td className="r dim" style={{ fontSize: 11 }}>{fmtAge(row.computed_at)}</td>
                  </tr>
                )
              })}
              {filtered.length === 0 && (
                <tr>
                  <td colSpan={8} style={{ padding: '24px', textAlign: 'center', color: 'var(--fg-3)' }}>
                    No source-quality snapshots available for this filter.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </Panel>
      )}
    </div>
  )
}
