import { Routes, Route, Navigate } from 'react-router-dom'
import NavBar from './components/NavBar'
import SignalFeed from './pages/SignalFeed'
import Positions from './pages/Positions'
import Calibration from './pages/Calibration'
import LLMCost from './pages/LLMCost'
import StrategyConfig from './pages/StrategyConfig'
import SystemHealth from './pages/SystemHealth'

export default function App() {
  return (
    <div className="min-h-screen bg-gray-50">
      <NavBar />
      <main className="max-w-7xl mx-auto px-4 py-6">
        <Routes>
          <Route path="/" element={<Navigate to="/signals" replace />} />
          <Route path="/signals" element={<SignalFeed />} />
          <Route path="/positions" element={<Positions />} />
          <Route path="/calibration" element={<Calibration />} />
          <Route path="/llm" element={<LLMCost />} />
          <Route path="/strategy" element={<StrategyConfig />} />
          <Route path="/health" element={<SystemHealth />} />
        </Routes>
      </main>
    </div>
  )
}
