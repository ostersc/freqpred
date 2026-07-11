"""StrategyConfig dataclass."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class OrderTypes:
    """Per-strategy order type configuration.

    entry="limit" posts a resting bid at estimated_probability - min_edge rather
    than crossing the spread immediately. The position opens as status="pending"
    and is promoted to "open" by PositionMonitor when the ask crosses the limit.
    All fields default to "market" so existing strategies are unaffected.
    """

    entry: str = "market"                        # "limit" | "market"
    exit: str = "market"                         # "limit" | "market"
    emergency_exit: str = "market"               # always market — not overridable
    stoploss: str = "market"                     # "limit" | "market"
    stoploss_on_exchange: bool = False
    stoploss_on_exchange_interval: int = 60
    stoploss_on_exchange_limit_ratio: float = 0.99

    def __post_init__(self) -> None:
        if self.stoploss_on_exchange and self.stoploss != "limit":
            raise ValueError(
                "stoploss_on_exchange=True requires stoploss='limit', "
                f"got stoploss={self.stoploss!r}"
            )


@dataclass
class StrategyConfig:
    name: str
    min_confidence: float
    max_exposure_per_market: float
    kelly_fraction: float
    categories: list[str]
    min_volume_24h: float
    max_days_to_close: float
    min_days_to_close: float

    # Exit management — all thresholds are absolute 0-1 price-scale dollars.
    # e.g. stoploss=-0.10 means "exit if price drops 10 cents from entry".
    stoploss: float = -0.15
    trailing_stop: bool = False
    trailing_stop_positive: float | None = None      # switch to tight trail once up this many cents
    trailing_stop_positive_offset: float = 0.02      # tight trail distance (cents) below peak

    # Edge band filter: only enter when the signal's estimated edge falls within
    # [min_edge, max_edge]. min_edge rejects weak signals where the model barely
    # disagrees with the market. max_edge rejects overconfident signals — empirically,
    # very high edge means the market is right and the model is wrong. None = no cap.
    min_edge: float = 0.10
    max_edge: float | None = None

    # Price range filter: skip markets the market has already decided.
    # Markets trading below min_mid_price or above max_mid_price are excluded
    # from ingestion and signal generation. None = no filter on that bound.
    # At entry time (should_trade) the bounds apply to the entry side's own
    # cost — the YES mid for YES signals, 1 - mid for NO signals — so a NO
    # entry on a market at 0.93 (own cost 0.07) is blocked as a longshot.
    min_mid_price: float | None = 0.05
    max_mid_price: float | None = 0.95

    # Liquidity filter: reject entry if yes_ask - yes_bid exceeds this threshold.
    # None = auto-compute as min_edge / 2 (spread must consume < half your edge).
    max_spread: float | None = None

    # Re-entry guards after a stoploss or trailing_stop exit.
    # block_reentry_after_stoploss takes precedence: if True, the market is
    # permanently blocked from re-entry once any stoploss/trailing_stop has fired,
    # regardless of stoploss_cooldown_hours.
    # If False and stoploss_cooldown_hours > 0, re-entry is blocked for that many
    # hours after the most recent stoploss/trailing_stop exit on this market.
    block_reentry_after_stoploss: bool = False
    stoploss_cooldown_hours: float = 4.0  # set to 0.0 to disable cooldown

    # Pre-signal risk gate: skip LLM analysis for new-entry markets where risk
    # would block the resulting trade anyway (global exposure caps reached, spread
    # too wide, or stoploss re-entry blocked). Set to False to always generate
    # fresh signals regardless of risk state — useful when signals are needed for
    # calibration or analytics even when trading is constrained.
    # Has no effect in signal-only mode (order manager is not active).
    pre_signal_risk_gate: bool = True

    # Assessment-based sizing controls. The Opus judgment model outputs a
    # trust_score, which the framework maps to this multiplier range.
    # 0.80 (through assessment-v4) compressed all assessor discrimination into
    # a <=20% stake haircut; T94's live audit showed re-mapping the same
    # trust scores at a wider floor cut sample losses 58% vs no assessor.
    # 0.25 is deferred until assessment-v5 accrues live history.
    assessment_scale_min: float = 0.50
    assessment_scale_max: float = 1.20
    similar_market_min_signals: int = 10
    similar_market_min_trades: int = 5

    # FactBase phrase frequency gate. Markets whose series_ticker appears in
    # this list are held as not-interesting until phrase frequency data is
    # cached in DB. Haiku extracts the search terms once per market lifetime.
    # Only meaningful for KXTRUMPSAY-style "will he say X" markets.
    factbase_series_allowlist: list[str] = field(default_factory=list)

    # Order type configuration. Defaults to all market orders — existing strategies
    # are unaffected. Set entry="limit" to post resting bids at
    # estimated_probability - min_edge instead of crossing the spread immediately.
    order_types: OrderTypes = field(default_factory=OrderTypes)

    # Paper-mode only. Cancel unfilled resting limit entries after this many hours.
    # Live-mode resting orders use pending_order_timeout_seconds (exchange-side).
    limit_order_timeout_hours: float = 4.0

    # Live-mode only. After this many seconds in 'pending', cancel_order is called.
    # Reconcile sweeps (startup, periodic, WS reconnect) check the age and cancel
    # any pending row whose created_at exceeds the cutoff.
    pending_order_timeout_seconds: float = 900.0
