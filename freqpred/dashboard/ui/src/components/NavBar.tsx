import { NavLink } from 'react-router-dom'

const links = [
  { to: '/markets',       label: 'Markets' },
  { to: '/signals',       label: 'Signal Feed' },
  { to: '/positions',     label: 'Positions' },
  { to: '/decisions',     label: 'Decisions' },
  { to: '/calibration',   label: 'Calibration' },
  { to: '/source-quality',label: 'Source Quality' },
  { to: '/llm',           label: 'LLM Cost' },
  { to: '/strategy',      label: 'Strategy Config' },
  { to: '/health',        label: 'System Health' },
]

export default function NavBar() {
  return (
    <nav className="topnav">
      <div className="topnav-inner">
        <div className="brand">
          <div className="brand-mark" />
          <span className="brand-name">freqpred<span className="brand-dot">.</span></span>
        </div>
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
          <span className="nav-status"><span className="dot" /> PAPER</span>
        </div>
      </div>
    </nav>
  )
}
