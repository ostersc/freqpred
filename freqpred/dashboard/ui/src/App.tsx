import { Routes, Route, Navigate } from 'react-router-dom'
import NavBar from './components/NavBar'
import SignalFeed from './pages/SignalFeed'
import Positions from './pages/Positions'
import StrategyDecisions from './pages/StrategyDecisions'
import Markets from './pages/Markets'
import Calibration from './pages/Calibration'
import LLMCost from './pages/LLMCost'
import SourceQuality from './pages/SourceQuality'
import StrategyConfig from './pages/StrategyConfig'
import SystemHealth from './pages/SystemHealth'

export default function App() {
  return (
    <div style={{ minHeight: '100vh' }}>
      <NavBar />
      <Routes>
        <Route path="/" element={<Navigate to="/markets" replace />} />
        <Route path="/markets" element={<Markets />} />
        <Route path="/signals" element={<SignalFeed />} />
        <Route path="/positions" element={<Positions />} />
        <Route path="/decisions" element={<StrategyDecisions />} />
        <Route path="/calibration" element={<Calibration />} />
        <Route path="/source-quality" element={<SourceQuality />} />
        <Route path="/llm" element={<LLMCost />} />
        <Route path="/strategy" element={<StrategyConfig />} />
        <Route path="/health" element={<SystemHealth />} />
      </Routes>
    </div>
  )
}
