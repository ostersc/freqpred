import { NavLink } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { getSystemHealth } from '../api/health'
import { getLedger } from '../api/ledger'
import { useTheme } from '../hooks/useTheme'

const links = [
  { to: '/markets',        label: 'Markets' },
  { to: '/signals',        label: 'Signal Feed' },
  { to: '/positions',      label: 'Positions' },
  { to: '/decisions',      label: 'Decisions' },
  { to: '/pnl',            label: 'P&L History' },
  { to: '/calibration',    label: 'Calibration' },
  { to: '/source-quality', label: 'Source Quality' },
  { to: '/llm',            label: 'LLM Cost' },
  { to: '/strategy',       label: 'Strategy Config' },
  { to: '/health',         label: 'System Health' },
]

function fmtSignedMoney(v: number) {
  const abs = Math.abs(v).toFixed(2)
  return v >= 0 ? `+$${abs}` : `-$${abs}`
}

export default function NavBar() {
  const { data: health } = useQuery({
    queryKey: ['systemHealth'],
    queryFn: getSystemHealth,
    refetchInterval: 15_000,
    staleTime: 10_000,
  })

  const { data: ledger } = useQuery({
    queryKey: ['ledger'],
    queryFn: getLedger,
    refetchInterval: 30_000,
    staleTime: 20_000,
  })

  const runState = health?.run_state ?? '—'
  const mode = health?.mode ?? '—'
  const isRunning = runState === 'running'
  const { light, dense, toggleLight, toggleDense } = useTheme()

  return (
    <nav className="topnav">
      <div className="topnav-inner">
        <NavLink to="/" className="brand" style={{ textDecoration: 'none' }}>
          <div className="brand-mark" />
          <span className="brand-name">freqpred<span className="brand-dot">.</span></span>
        </NavLink>
        <div className="nav-tabs">
          {links.map(({ to, label }) => (
            <NavLink
              key={to}
              to={to}
              className={({ isActive }) => `nav-tab${isActive ? ' active' : ''}`}
            >
              {label}
            </NavLink>
          ))}
        </div>
        <div className="nav-right">
          <span className="nav-status" style={isRunning ? {} : { background: 'var(--bg-2)', color: 'var(--fg-2)' }}>
            {isRunning && <span className="dot" />}
            {runState.toUpperCase()} · {mode.toUpperCase()}
          </span>
          {ledger && (
            <>
              <span>p&amp;l <b className={`mono ${ledger.daily_pnl_usd >= 0 ? 'pos' : 'neg'}`}>{fmtSignedMoney(ledger.daily_pnl_usd)}</b></span>
              <span className="dim">all-time <b className={`mono ${ledger.all_time_pnl_usd >= 0 ? 'pos' : 'neg'}`}>{fmtSignedMoney(ledger.all_time_pnl_usd)}</b></span>
            </>
          )}
          <button className="btn ghost sm" onClick={toggleLight} title={light ? 'Switch to dark' : 'Switch to light'} style={{ fontSize: 13 }}>
            {light ? '☽' : '☀︎'}
          </button>
          <button className="btn ghost sm" onClick={toggleDense} title={dense ? 'Normal density' : 'Dense mode'} style={{ opacity: dense ? 1 : 0.5, padding: '3px 6px' }}>
            <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
              <rect x="1" y="1" width="12" height="2" rx="1" fill="currentColor"/>
              <rect x="1" y="4.5" width="12" height="2" rx="1" fill="currentColor"/>
              <rect x="1" y="8" width="12" height="2" rx="1" fill="currentColor"/>
              <rect x="1" y="11" width="12" height="2" rx="1" fill="currentColor"/>
            </svg>
          </button>
        </div>
      </div>
    </nav>
  )
}
