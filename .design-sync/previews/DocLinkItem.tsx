import { DocLinkItem } from 'freqpred-dashboard'

const docs = [
  {
    document_id: 'd1',
    source_url: 'https://www.reuters.com/world/us/fed-signals-pause-2026-07-01/',
    title: 'Fed signals pause in rate hikes amid cooling inflation',
    relevance_score: 0.912,
    source_type: 'news',
    source_name: 'Reuters',
    published_at: '2026-07-01T14:00:00Z',
    fetched_at: '2026-07-01T14:32:00Z',
    summary: 'The Federal Reserve indicated it may pause further rate increases after inflation data came in below expectations for a third consecutive month.',
    body_excerpt: 'The Federal Reserve indicated...',
  },
  {
    document_id: 'd2',
    source_url: 'https://apnews.com/article/fed-rates-inflation-2026',
    title: 'AP explains: what the Fed pause means for markets',
    relevance_score: 0.847,
    source_type: 'news',
    source_name: 'Associated Press',
    published_at: null,
    fetched_at: '2026-07-01T15:10:00Z',
    summary: null,
    body_excerpt: 'Markets rallied on the news, with the S&P 500 climbing 1.2% intraday as traders priced in a longer pause than previously expected.',
  },
]

export function List() {
  return (
    <ul style={{ margin: 0, padding: 0, display: 'flex', flexDirection: 'column', gap: 8, minWidth: 360 }}>
      {docs.map((doc) => <DocLinkItem key={doc.document_id} doc={doc} />)}
    </ul>
  )
}
