interface Props {
  status: string
}

const colors: Record<string, string> = {
  ok: 'bg-green-100 text-green-800',
  running: 'bg-green-100 text-green-800',
  connected: 'bg-green-100 text-green-800',
  degraded: 'bg-yellow-100 text-yellow-800',
  paused: 'bg-yellow-100 text-yellow-800',
  stopped: 'bg-red-100 text-red-800',
  halted: 'bg-red-100 text-red-800',
  error: 'bg-red-100 text-red-800',
  paper: 'bg-blue-100 text-blue-800',
  live: 'bg-purple-100 text-purple-800',
}

export default function StatusBadge({ status }: Props) {
  const cls = colors[status.toLowerCase()] ?? 'bg-gray-100 text-gray-800'
  return (
    <span className={`inline-flex px-2 py-0.5 rounded text-xs font-medium ${cls}`}>
      {status}
    </span>
  )
}
