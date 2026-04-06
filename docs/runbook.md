# freqpred Incident Runbook

This runbook describes each circuit breaker alert, what it means, and how to respond.

---

## Circuit Breaker: Daily Loss Limit

**Alert:**
```
🚨 CIRCUIT BREAKER TRIPPED
Type: daily_loss
Reason: Circuit breaker: daily loss X.XX exceeds 15% of bankroll (Y.YY)
Action required: freqpred will not enter new positions until manually resumed.
Resume: /start (Telegram) or freqpred run (restart)
```

**Meaning:** Realized P&L on live closed positions today has exceeded `max_daily_loss_pct` (default 15%) of bankroll. No new orders will be submitted for the rest of this run.

**Immediate actions:**
1. Run `/profit` in Telegram to see which positions closed at a loss and why.
2. Run `/status` to check if any positions are still open and at risk.
3. Run `/signals 20` to review signal quality for affected markets.
4. Determine whether the losses reflect a systematic signal error or isolated bad luck.

**To resume:** Send `/start` in Telegram after investigating. freqpred does not resume automatically — manual restart is required. If restarting the process, the circuit breaker resets and trading resumes from the next signal cycle.

---

## Circuit Breaker: Total Drawdown

**Alert:**
```
🚨 CIRCUIT BREAKER TRIPPED
Type: drawdown
Reason: Circuit breaker: drawdown X.X% exceeds 30% (baseline: A.AA, current: B.BB)
Action required: freqpred will not enter new positions until manually resumed.
Resume: /start (Telegram) or freqpred run (restart)
```

**Meaning:** The current net bankroll has fallen more than 30% from the drawdown window baseline (set when `/reset_drawdown` was last called). This fires in both live and paper modes.

**Immediate actions:**
1. Run `/profit` to see the full P&L history since the last drawdown reset.
2. Review signal quality with `/signals 20` — look for a systematic overestimation of probability.
3. Check if market conditions have changed significantly (regime shift, news shock, etc.).
4. Consider reducing position size (`max_position_pct`) or increasing `min_edge` before resuming.

**To resume:** After investigating, send `/reset_drawdown` in Telegram to establish a new baseline, then send `/start`. Do not reset without understanding why the drawdown occurred.

---

## Circuit Breaker: LLM Consecutive Errors

**Alert:**
```
🚨 CIRCUIT BREAKER TRIPPED
Type: llm_errors
Reason: LLM API failed N consecutive times
Action required: freqpred will not enter new positions until manually resumed.
Resume: /start (Telegram) or freqpred run (restart)
```

**Meaning:** The Anthropic API has failed `max_consecutive_llm_errors` times in a row (default: 3) without a successful call. The signal pipeline cannot produce new signals. This circuit breaker is in-memory and resets on service restart.

**Immediate actions:**
1. Check the Anthropic status page for outages.
2. Verify the `ANTHROPIC_API_KEY` environment variable is set correctly.
3. Check the freqpred logs for the specific API error message (`signal_loop.llm_consecutive_errors`).
4. Check for network connectivity issues between the freqpred host and `api.anthropic.com`.

**To resume:** Once the underlying API issue is resolved, restart the freqpred process. The consecutive-error counter resets on restart. Alternatively, wait for the next cycle — if the API recovers, the counter resets automatically after any successful call.

---

## Circuit Breaker: LLM Budget Cap

**Alert:**
```
🚨 CIRCUIT BREAKER TRIPPED
Type: llm_budget
Reason: Daily LLM spend cap of $X.XX reached (spent $Y.YYYY today)
Action required: freqpred will not enter new positions until manually resumed.
Resume: /start (Telegram) or freqpred run (restart)
```

**Meaning:** Daily LLM API spend has reached `max_daily_llm_spend_usd` (default: $10.00). No further LLM calls will be made today. This resets automatically at UTC midnight.

**Immediate actions:**
1. Check the LLM Cost & Audit dashboard page to see which query types consumed the budget.
2. Look for unusually long prompts or unexpectedly high call volume from the ingestion pipeline.
3. Consider increasing `max_daily_llm_spend_usd` in `config.yaml` if the budget is consistently too low.
4. If this is an anomaly (e.g., a loop bug), investigate before resuming.

**To resume:** The budget resets at UTC midnight — freqpred will resume signal generation automatically on the next cycle after midnight. To resume earlier, increase `max_daily_llm_spend_usd` and restart the process.
