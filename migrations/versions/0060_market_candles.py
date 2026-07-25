"""Add market_candles and candle_fetch_cursors.

The system persists only point price observations (``signals.market_*_at_signal``,
median ~22 per live hold over a median 72h) and censored extremes
(``positions.mae``, which stops updating at the exit). Neither supports an honest
"what if we had run a different stoploss" counterfactual: the point series can
miss a threshold touched and recovered inside a 3-hour gap, and MAE cannot say
whether a position that really stopped at -0.15 would also have breached -0.30.

Kalshi's candlestick endpoint supplies the missing intra-hold path, including
yes_bid and yes_ask OHLC — both sides, which is what modelling a stop fill needs.

**Retention is a rolling window, measured at ~67 days on 2026-07-25**: candles
were available for markets closing 2026-05-20 and later, and 404 for 2026-05-18
and earlier. Expired data does not come back, so it is captured into this table
rather than fetched on demand.

Both tables use natural composite primary keys, so re-fetching an overlapping
window is an idempotent upsert. Neither has a FK to ``markets`` — the backfill
tool must work for tickers this system never ingested.

Revision ID: 0060
Revises: 0059
"""
import sqlalchemy as sa
from alembic import op

revision = "0060"
down_revision = "0059"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "market_candles",
        sa.Column("market_id", sa.VARCHAR(255), nullable=False),
        sa.Column("period_interval", sa.SmallInteger(), nullable=False),
        sa.Column("end_period_ts", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("series_ticker", sa.Text(), nullable=False),
        # Nullable: a period with no trades has no price OHLC at all.
        sa.Column("price_open", sa.Float(), nullable=True),
        sa.Column("price_high", sa.Float(), nullable=True),
        sa.Column("price_low", sa.Float(), nullable=True),
        sa.Column("price_close", sa.Float(), nullable=True),
        sa.Column("price_mean", sa.Float(), nullable=True),
        sa.Column("yes_bid_open", sa.Float(), nullable=True),
        sa.Column("yes_bid_high", sa.Float(), nullable=True),
        sa.Column("yes_bid_low", sa.Float(), nullable=True),
        sa.Column("yes_bid_close", sa.Float(), nullable=True),
        sa.Column("yes_ask_open", sa.Float(), nullable=True),
        sa.Column("yes_ask_high", sa.Float(), nullable=True),
        sa.Column("yes_ask_low", sa.Float(), nullable=True),
        sa.Column("yes_ask_close", sa.Float(), nullable=True),
        sa.Column("volume", sa.Float(), nullable=False, server_default="0"),
        sa.Column("open_interest", sa.Float(), nullable=False, server_default="0"),
        sa.Column("fetched_at", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.PrimaryKeyConstraint(
            "market_id", "period_interval", "end_period_ts", name="market_candles_pkey"
        ),
    )
    # Family-scoped scans ("every KXTRUMPSAY candle in this window").
    op.create_index(
        "ix_market_candles_series_ts",
        "market_candles",
        ["series_ticker", "period_interval", "end_period_ts"],
    )

    op.create_table(
        "candle_fetch_cursors",
        sa.Column("market_id", sa.VARCHAR(255), nullable=False),
        sa.Column("period_interval", sa.SmallInteger(), nullable=False),
        sa.Column("covered_from", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("covered_to", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("candle_count", sa.Integer(), nullable=False, server_default="0"),
        # True once the API has returned 404 for this market: it settled before
        # the retention cutoff. Permanent — never retry, the data is gone.
        sa.Column(
            "expired", sa.Boolean(), nullable=False, server_default=sa.text("false")
        ),
        sa.Column("last_fetched_at", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.PrimaryKeyConstraint(
            "market_id", "period_interval", name="candle_fetch_cursors_pkey"
        ),
    )


def downgrade() -> None:
    op.drop_table("candle_fetch_cursors")
    op.drop_index("ix_market_candles_series_ts", table_name="market_candles")
    op.drop_table("market_candles")
