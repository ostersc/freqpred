"""Alert dispatcher — fans out to all configured senders."""
from __future__ import annotations

import structlog

from freqpred.alerts.base import AlertSender
from freqpred.markets.models import Market, Position
from freqpred.signal.models import Signal

log = structlog.get_logger(__name__)


class AlertDispatcher:
    """Fans out alert messages to all configured senders.

    A sender failure is logged but never propagates — the run loop must not
    crash because of a network blip or bad credentials.
    """

    def __init__(self, senders: list[AlertSender]) -> None:
        self._senders = senders

    async def send(self, message: str) -> None:
        for sender in self._senders:
            try:
                await sender.send(message)
            except Exception:
                log.exception("alert_sender_failed", sender=type(sender).__name__)

    async def signal_alert(self, signal: Signal, market: Market) -> None:
        prob_pct = signal.estimated_probability * 100
        mid_pct = market.mid_price * 100
        edge_pct = signal.edge * 100
        msg = (
            f"NEW SIGNAL: {market.question}\n"
            f"Prob: {prob_pct:.1f}%  |  Market: {mid_pct:.1f}%  |  Edge: {edge_pct:+.1f}%  |  "
            f"Direction: {signal.direction}  |  Confidence: {signal.confidence:.2f}"
        )
        await self.send(msg)

    async def trade_alert(self, position: Position, market: Market) -> None:
        label = "LIVE TRADE" if position.mode == "live" else "PAPER TRADE"
        msg = (
            f"{label}: {position.direction} on {market.question}\n"
            f"@ ${position.entry_price:.4f}  |  {position.contracts} contracts  |  "
            f"Edge: {position.signal_edge:+.3f}"
        )
        await self.send(msg)

    async def resolution_alert(self, position: Position, market: Market) -> None:
        pnl = position.pnl or 0.0
        prefix = "WIN" if pnl >= 0 else "LOSS"
        resolution_label = "YES" if position.resolution == 1 else "NO"
        msg = (
            f"{prefix}: {market.question} resolved {resolution_label}\n"
            f"P&L: {pnl:+.4f}  |  Direction: {position.direction}  |  "
            f"Entry: {position.entry_price:.4f}  Exit: {position.exit_price:.4f}"
        )
        await self.send(msg)

    async def exit_alert(self, position: Position, exit_reason: str) -> None:
        pnl = position.pnl or 0.0
        prefix = "WIN" if pnl >= 0 else "LOSS"
        msg = (
            f"{prefix} EXIT ({exit_reason}): {position.direction} position closed\n"
            f"Market: {position.market_id}\n"
            f"P&L: {pnl:+.4f}  |  Entry: {position.entry_price:.4f}  "
            f"Exit: {position.exit_price:.4f}  |  {position.contracts} contracts"
        )
        await self.send(msg)

    async def circuit_breaker_alert(self, reason: str) -> None:
        msg = f"CIRCUIT BREAKER: {reason}. Trading halted."
        await self.send(msg)

    async def digest_alert(self, digest_text: str) -> None:
        await self.send(digest_text)

    async def startup_alert(self, strategy_name: str, mode: str, run_state: str) -> None:
        msg = f"freqpred started | strategy={strategy_name} | mode={mode} | state={run_state}"
        if run_state != "running":
            msg += (
                f"\n*** WARNING: run_state='{run_state}' — "
                "signal loop is INACTIVE. Use /start to resume. ***"
            )
        await self.send(msg)
