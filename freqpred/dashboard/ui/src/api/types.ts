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
  llm_query_id: number | null
  rag_hit_count: number
  has_factbase: boolean
  series_ticker: string | null
  has_assessment: boolean
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

export interface SignalAssessmentOut {
  trust_score: number
  size_multiplier: number
  verdict: string
  reasoning: string
  key_factors: string[]
  warnings: string[]
  source_breakdown: Array<Record<string, unknown>>
  similar_market_summary: Record<string, unknown>
  llm_query_id: number | null
  created_at: string
}

export interface SignalDetailOut extends SignalOut {
  document_links: DocumentLinkOut[]
  assessment: SignalAssessmentOut | null
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
  exit_reason: string | null
  resolution: number | null
  pnl: number | null
  pnl_pct: number | null
  unrealized_pnl: number | null
  unrealized_pnl_pct: number | null
  current_mid: number | null
  created_at: string
  has_factbase: boolean
  series_ticker: string | null
  exchange_order_id?: string | null
  requested_contracts?: number | null
  exchange_order_status?: string | null
  last_exchange_sync_at?: string | null
  // Exit-side order state (live mode only)
  exit_order_id?: string | null
  exit_fee_usd?: number
  exit_requested_contracts?: number | null
  exit_filled_contracts?: number | null
}

export interface StrategyDecisionOut extends PositionOut {
  market_question: string | null
  market_result: string | null
  counterfactual_pnl_per_contract: number | null
  counterfactual_pnl_usd: number | null
  exit_delta_per_contract: number | null
  exit_delta_usd: number | null
  best_prior_ask: number | null
  entry_efficiency_per_contract: number | null
  entry_efficiency_usd: number | null
}

export interface StrategyDecisionListResponse {
  items: StrategyDecisionOut[]
  total: number
  limit: number
  offset: number
  distinct_strategies: string[]
  distinct_exit_reasons: string[]
}

export interface PositionDetailOut extends PositionOut {
  market_question: string | null
  entry_signal: SignalDetailOut
  market_signals: SignalOut[]
}

export interface PositionListResponse {
  items: PositionOut[]
  total: number
}

export interface MarketOut {
  id: string
  question: string
  status: string
  yes_bid: number
  yes_ask: number
  mid_price: number
  volume_24h: number
  close_time: string
  last_fetched_at: string
  current_signal_id: string | null
}

export interface MarketDetailOut extends MarketOut {
  current_signal: SignalOut | null
}

export interface MarketListResponse {
  items: MarketOut[]
  total: number
  limit: number
  offset: number
}

export interface AnalyzeResponse {
  signal: SignalOut
  cached: boolean
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
  available_categories: string[]
  available_models: string[]
  available_prompt_versions: string[]
  available_directions: string[]
  available_series_tickers: string[]
}

export interface CalibrationTimeSeriesPoint {
  date: string
  brier_score: number | null
  market_brier_score: number | null
  n_samples: number
}

export interface CalibrationTimeSeriesResponse {
  series: CalibrationTimeSeriesPoint[]
  prompt_version_starts: { version: string; date: string }[]
  available_categories: string[]
  available_models: string[]
  available_prompt_versions: string[]
  available_directions: string[]
  available_series_tickers: string[]
}

export interface CalibrationHeatmapCell {
  brier_score: number | null
  market_brier_score: number | null
  n_samples: number
  delta: number | null
}

export interface CalibrationHeatmapRow {
  series_ticker: string
  option_code: string
  option_label: string
  cells: Record<string, CalibrationHeatmapCell>
}

export interface CalibrationHeatmapResponse {
  rows: CalibrationHeatmapRow[]
  prompt_versions: string[]
  available_categories: string[]
  available_models: string[]
  available_directions: string[]
  available_series_tickers: string[]
}

export interface SourceQualityScoreOut {
  source_name: string
  market_category: string | null
  weighted_brier: number
  overall_brier: number
  n_signals: number
  total_doc_uses: number
  computed_at: string
}

export interface SourceQualityListResponse {
  items: SourceQualityScoreOut[]
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
  assessment_scale_min: number
  assessment_scale_max: number
  similar_market_min_signals: number
  similar_market_min_trades: number
}

export interface CircuitBreakerStateOut {
  trading_halted: boolean
  reason: string | null
  daily_loss_pct: number
  daily_loss_limit_pct: number
  daily_loss_window_start: string
  daily_loss_ack_at: string | null
  llm_budget_used_usd: number
  llm_budget_cap_usd: number
}

export interface WebSocketStateOut {
  status: string
  connected: boolean | null
  subscribed_markets: number | null
  last_message_at: string | null
  last_reconcile_at: string | null
}

export interface ServiceFreshnessOut {
  service_name: string
  label: string
  status: string
  last_success_at: string | null
  last_error_at: string | null
  last_error_message: string | null
  stale_after_seconds: number
  age_seconds: number | null
}

export interface ApiErrorStateOut {
  kalshi_errors_last_hour: number
  llm_errors_last_hour: number
  consecutive_llm_errors: number | null
}

export interface ExchangeStatusOut {
  exchange_active: boolean | null
  trading_active: boolean | null
  fetched_at: string | null
}

export interface ChangelogStatusOut {
  unreviewed_count: number
  has_unreviewed_breaking_change: boolean
  last_reviewed_at: string | null   // YYYY-MM-DD
  last_checked_at: string | null
}

export interface PendingOrderSummary {
  position_id: string
  market_id: string
  requested_contracts: number | null
  filled_contracts: number
  exchange_order_status: string | null
  age_seconds: number
  last_exchange_sync_at: string | null
}

export interface KalshiApiTierOut {
  api_usage_level: string | null
  can_upgrade: boolean
  fetched_at: string
}

export interface SystemHealthResponse {
  run_state: string
  mode: string
  circuit_breakers: CircuitBreakerStateOut
  websocket: WebSocketStateOut
  api_errors: ApiErrorStateOut
  services: ServiceFreshnessOut[]
  exchange: ExchangeStatusOut
  changelog: ChangelogStatusOut
  pending_orders: number
  oldest_pending_order_age_seconds: number | null
  pending_orders_detail?: PendingOrderSummary[]
  open_positions: number
  db_ok: boolean
  uptime_seconds: number
  api_tier?: KalshiApiTierOut | null
}

export interface LedgerResponse {
  open_count: number
  total_exposure_usd: number
  daily_pnl_usd: number
  all_time_pnl_usd: number
}
