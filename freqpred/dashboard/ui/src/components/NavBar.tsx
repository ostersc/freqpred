import { NavLink } from 'react-router-dom'

const links = [
  { to: '/signals', label: 'Signal Feed' },
  { to: '/positions', label: 'Positions' },
  { to: '/ledger', label: 'Ledger' },
  { to: '/calibration', label: 'Calibration' },
  { to: '/llm', label: 'LLM Cost' },
  { to: '/strategy', label: 'Strategy Config' },
  { to: '/health', label: 'System Health' },
]

export default function NavBar() {
  return (
    <nav className="bg-gray-900 text-white shadow">
      <div className="max-w-7xl mx-auto px-4 flex items-center h-12 gap-1">
        <span className="font-bold text-sm mr-4 text-gray-300">freqpred</span>
        {links.map(({ to, label }) => (
          <NavLink
            key={to}
            to={to}
            className={({ isActive }) =>
              `px-3 py-1.5 rounded text-sm transition-colors ${
                isActive
                  ? 'bg-blue-600 text-white'
                  : 'text-gray-300 hover:bg-gray-700'
              }`
            }
          >
            {label}
          </NavLink>
        ))}
      </div>
    </nav>
  )
}
