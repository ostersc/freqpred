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
    <li style={{ listStyle: 'none' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, minWidth: 0 }}>
        <span className="mono dim" style={{ fontSize: 10.5, flexShrink: 0 }}>{doc.relevance_score.toFixed(3)}</span>
        <button
          onClick={() => setOpen((v) => !v)}
          style={{
            flexShrink: 0, width: 16, height: 16, borderRadius: '50%', fontSize: 10, fontWeight: 700,
            border: `1px solid ${open ? 'var(--accent)' : 'var(--line)'}`,
            background: open ? 'var(--accent)' : 'transparent',
            color: open ? 'var(--bg-0)' : 'var(--fg-3)',
            cursor: 'pointer', lineHeight: 1,
          }}
          title="Show document details"
        >
          i
        </button>
        <a
          href={doc.source_url}
          target="_blank"
          rel="noreferrer"
          style={{ color: 'var(--accent)', fontSize: 12, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', minWidth: 0, flex: 1 }}
        >
          {doc.title || doc.source_url}
        </a>
      </div>
      {open && (
        <div style={{ marginTop: 6, marginLeft: 56, background: 'var(--bg-1)', border: '1px solid var(--line-soft)', borderRadius: 6, padding: '10px 12px', fontSize: 11.5 }}>
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 6 }}>
            <span style={{ padding: '2px 6px', background: 'var(--bg-2)', border: '1px solid var(--line-soft)', borderRadius: 4, color: 'var(--fg-1)', textTransform: 'capitalize' }}>{doc.source_type}</span>
            <span style={{ color: 'var(--fg-1)', fontWeight: 500 }}>{doc.source_name}</span>
          </div>
          <div style={{ display: 'flex', gap: 16, color: 'var(--fg-3)', marginBottom: excerpt ? 6 : 0 }}>
            {doc.published_at && (
              <span>Published: <span style={{ color: 'var(--fg-2)' }}>{fmtDate(doc.published_at)}</span></span>
            )}
            <span>Fetched: <span style={{ color: 'var(--fg-2)' }}>{fmtDate(doc.fetched_at)}</span></span>
          </div>
          {excerpt && (
            <p style={{ margin: 0, color: 'var(--fg-1)', lineHeight: 1.6, borderTop: '1px solid var(--line-soft)', paddingTop: 6, whiteSpace: 'pre-wrap' }}>{excerpt}</p>
          )}
        </div>
      )}
    </li>
  )
}
