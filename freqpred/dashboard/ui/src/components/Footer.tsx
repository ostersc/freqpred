import { useQuery } from '@tanstack/react-query'
import { getVersion } from '../api/system'

export default function Footer() {
  const { data } = useQuery({
    queryKey: ['version'],
    queryFn: getVersion,
    staleTime: Infinity,
  })

  return (
    <footer style={{
      borderTop: '1px solid var(--line-soft)',
      padding: '10px 24px',
      fontSize: 11,
      color: 'var(--fg-3)',
      textAlign: 'center',
      marginTop: 24,
    }}>
      {data
        ? <span className="mono">freqpred {data.version} · git {data.git_hash}</span>
        : <span className="mono" style={{ opacity: 0.4 }}>freqpred</span>}
    </footer>
  )
}
