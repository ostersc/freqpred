export interface SignalOut {
  id: string
  market_id: string
  market_question: string | null
  estimated_probability: number
  confidence: number
  edge: number
  market_mid_at_signal: number
  direction: string
  reasoning: string
  sources: string[]
  retrieval_hash: string
  model_used: string
  prompt_version: string
  trigger: string
  created_at: string
  social_sentiment_summary: string | null
}

export interface DocumentLinkOut {
  document_id: string
  source_url: string
  title: string
  relevance_score: number
  source_type: string
  source_name: string
  published_at: string | null
  fetched_at: string
  summary: string | null
  body_excerpt: string
}

export interface SignalDetailOut extends SignalOut {
  document_links: DocumentLinkOut[]
}

export interface SignalListResponse {
  items: SignalOut[]
  total: number
  limit: number
  offset: number
}

export interface PositionOut {
  id: string
  market_id: string
  signal_id: string
  strategy_name: string
  strategy_version: string
  signal_confidence: number
  signal_edge: number
  signal_estimated_prob: number
  direction: string
  contracts: number
  entry_price: number
  entry_time: string
  mode: string
  status: string
  exit_price: number | null
  exit_time: string | null
  resolution: number | null
  pnl: number | null
  pnl_pct: number | null
  unrealized_pnl: number | null
  unrealized_pnl_pct: number | null
  created_at: string
}

export interface PositionDetailOut extends PositionOut {
  market_question: string | null
  current_mid: number | null
  unrealized_pnl: number | null
  unrealized_pnl_pct: number | null
  entry_signal: SignalDetailOut
  market_signals: SignalOut[]
}

export interface PositionListResponse {
  items: PositionOut[]
  total: number
}

export interface CalibrationBucketOut {
  lower: number
  upper: number
  count: number
  mean_estimated_prob: number
  actual_resolution_rate: number
}

export interface CalibrationResponse {
  brier_score: number
  market_brier_score: number
  n_samples: number
  buckets: CalibrationBucketOut[]
  market_buckets: CalibrationBucketOut[]
}

export interface LLMCostResponse {
  today_usd: number
  weekly_usd: number
  daily_cap_usd: number
  pct_used: number
  by_query_type: Record<string, number>
}

export interface LLMQueryOut {
  id: number
  timestamp: string
  query_type: string
  market_id: string | null
  model_used: string
  tokens_total: number
  cost_usd: number
  latency_ms: number
  success: boolean
}

export interface LLMQueryListResponse {
  items: LLMQueryOut[]
  total: number
  limit: number
  offset: number
}

export interface LLMQueryDetailOut extends LLMQueryOut {
  prompt: string
  response: string
  error_message: string | null
}

export interface StrategyConfigOut {
  name: string
  min_edge: number
  min_confidence: number
  kelly_fraction: number
  max_exposure_per_market: number
  categories: string[]
  min_volume_24h: number
  max_days_to_close: number
  min_days_to_close: number
  stoploss: number
  trailing_stop: boolean
  trailing_stop_positive: number | null
  trailing_stop_positive_offset: number
  min_mid_price: number | null
  max_mid_price: number | null
  max_spread: number | null
  block_reentry_after_stoploss: boolean
  stoploss_cooldown_hours: number
}

export interface CircuitBreakerStateOut {
  trading_halted: boolean
  reason: string | null
  daily_loss_pct: number
  daily_loss_limit_pct: number
  llm_budget_used_usd: number
  llm_budget_cap_usd: number
}

export interface WebSocketStateOut {
  connected: boolean | null
  subscribed_markets: number | null
  last_message_at: string | null
}

export interface ApiErrorStateOut {
  kalshi_errors_last_hour: number
  llm_errors_last_hour: number
  consecutive_llm_errors: number | null
}

export interface SystemHealthResponse {
  run_state: string
  mode: string
  circuit_breakers: CircuitBreakerStateOut
  websocket: WebSocketStateOut
  api_errors: ApiErrorStateOut
  pending_orders: number
  open_positions: number
  db_ok: boolean
  uptime_seconds: number
}
