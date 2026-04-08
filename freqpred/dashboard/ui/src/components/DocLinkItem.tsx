import { useState } from 'react'
import type { DocumentLinkOut } from '../api/types'

function fmtDate(iso: string | null): string {
  if (!iso) return 'unknown'
  return new Date(iso).toLocaleString(undefined, {
    month: 'short', day: 'numeric', year: 'numeric',
    hour: '2-digit', minute: '2-digit',
  })
}

export function DocLinkItem({ doc }: { doc: DocumentLinkOut }) {
  const [open, setOpen] = useState(false)
  const excerpt = doc.summary || (doc.body_excerpt ? doc.body_excerpt + (doc.body_excerpt.length >= 400 ? '…' : '') : null)

  return (
    <li>
      <div className="flex items-center gap-2 min-w-0">
        <span className="text-xs text-gray-400 shrink-0 tabular-nums">{doc.relevance_score.toFixed(3)}</span>
        <a
          href={doc.source_url}
          target="_blank"
          rel="noreferrer"
          className="text-blue-600 hover:underline truncate min-w-0"
        >
          {doc.title || doc.source_url}
        </a>
        <button
          onClick={() => setOpen((v) => !v)}
          className={`shrink-0 w-4 h-4 rounded-full text-xs font-bold leading-none border transition-colors ${
            open
              ? 'bg-blue-600 text-white border-blue-600'
              : 'text-gray-400 border-gray-300 hover:text-blue-600 hover:border-blue-400'
          }`}
          title="Show document details"
          aria-label="Show document details"
        >
          i
        </button>
      </div>
      {open && (
        <div className="mt-1.5 ml-14 bg-white border rounded p-2.5 text-xs space-y-1.5 shadow-sm">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="px-1.5 py-0.5 rounded bg-gray-100 text-gray-600 font-medium capitalize">
              {doc.source_type}
            </span>
            <span className="text-gray-700 font-medium">{doc.source_name}</span>
          </div>
          <div className="flex gap-4 text-gray-400">
            {doc.published_at && (
              <span>Published: <span className="text-gray-600">{fmtDate(doc.published_at)}</span></span>
            )}
            <span>Fetched: <span className="text-gray-600">{fmtDate(doc.fetched_at)}</span></span>
          </div>
          {excerpt && (
            <p className="text-gray-600 whitespace-pre-wrap leading-relaxed border-t pt-1.5 mt-1">{excerpt}</p>
          )}
        </div>
      )}
    </li>
  )
}
