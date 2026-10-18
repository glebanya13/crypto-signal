"""Walk-forward backtest of the Long/Short scenario on closed candles (no look-ahead).

Candle-based factors only (trend, VWAP, SMC, oscillators, levels): the trade tape is not available historically,
so Delta factors are not used here. Orders: limit at the entry price, valid for ENTRY_VALID bars; the stop is
checked before the target when both fall in one bar (pessimistic). The result is statistics of this rule set on
past data, not a promise about the future.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from app.analytics.levels import find_levels
from app.analytics.signal import compute_signal
from app.analytics.smc import analyze_smc
from app.analytics.technical import analyze_technical
from app.analytics.volume import volume_summary

WARMUP = 250        # bars needed before the first decision (EMA/SMC history)
STEP = 4            # evaluate every N bars to keep runtime reasonable
ENTRY_VALID = 12    # bars a limit order waits for a fill
HOLD_MAX = 48       # bars a position may stay open before it is closed at market
FEE_R = 0.05        # round-trip cost, in R units, deducted from every filled trade


@dataclass
class Trade:
    index: int
    direction: str
    strength: float
    r: float
    outcome: str      # "target" | "stop" | "timeout"
    risk_reward: float = 0.0
    factors: dict = field(default_factory=dict)


@dataclass
class BacktestResult:
    symbol: str
    timeframe: str
    bars: int
    decisions: int
    signals: int
    filled: int
    trades: list[Trade] = field(default_factory=list)
    low_rr: int = 0     # directional signals skipped by the R/R filter

    def stats(self, trades: list[Trade] | None = None) -> dict:
        t = self.trades if trades is None else trades
        if not t:
            return {"n": 0, "win_rate": float("nan"), "avg_r": float("nan"), "total_r": 0.0, "max_dd": 0.0, "profit_factor": float("nan")}
        r = np.array([x.r for x in t])
        equity = np.cumsum(r)
        drawdown = float((np.maximum.accumulate(equity) - equity).max())
        wins, losses = r[r > 0].sum(), -r[r < 0].sum()
        return {"n": len(t), "win_rate": float((r > 0).mean()), "avg_r": float(r.mean()), "total_r": float(r.sum()),
                "max_dd": drawdown, "profit_factor": float(wins / losses) if losses > 0 else float("inf")}


def _simulate(frame: pd.DataFrame, start: int, direction: str, entry: float, stop: float, target: float) -> tuple[float, str, int] | None:
    """Fill a limit order, then walk to stop/target. Returns (R, outcome, bars_used) or None when never filled."""
    long = direction == "long"
    high, low, close = frame.high.to_numpy(), frame.low.to_numpy(), frame.close.to_numpy()
    n = len(frame)
    fill = None
    for j in range(start, min(start + ENTRY_VALID, n)):
        if (low[j] <= entry) if long else (high[j] >= entry):
            fill = j
            break
    if fill is None:
        return None
    risk = abs(entry - stop)
    if risk <= 0:
        return None
    for j in range(fill, min(fill + HOLD_MAX, n)):
        hit_stop = low[j] <= stop if long else high[j] >= stop
        hit_target = high[j] >= target if long else low[j] <= target
        if hit_stop:  # pessimistic: stop wins a same-bar tie (and the bar that fills the order may also hit it)
            return -1.0 - FEE_R, "stop", j - start
        if hit_target and j > fill:  # on the fill bar the order of events is unknown, so no target credit
            reward = abs(target - entry) / risk
            return reward - FEE_R, "target", j - start
    end = min(fill + HOLD_MAX, n) - 1
    pnl = (close[end] - entry) if long else (entry - close[end])
    return pnl / risk - FEE_R, "timeout", end - start


def run_backtest(candles: pd.DataFrame, symbol: str, timeframe: str) -> BacktestResult:
    frame = candles.sort_values("timestamp").reset_index(drop=True)
    result = BacktestResult(symbol, timeframe, len(frame), 0, 0, 0)
    busy_until = -1
    for i in range(WARMUP, len(frame) - 2, STEP):
        if i <= busy_until:
            continue
        window = frame.iloc[: i + 1]   # only data known at the close of bar i
        price = float(window.close.iloc[-1])
        support, resistance = find_levels(window)
        vwap = volume_summary(window, price)["vwap_24h"]
        signal = compute_signal(window, price, support, resistance, None, vwap, analyze_smc(window), analyze_technical(window))
        result.decisions += 1
        if signal.direction == "neutral" or signal.entry is None or not signal.targets:
            continue
        result.signals += 1
        if not signal.actionable:
            result.low_rr += 1
            continue
        outcome = _simulate(frame, i + 1, signal.direction, signal.entry, signal.stop, signal.targets[0])
        if outcome is None:
            continue
        r, kind, used = outcome
        result.filled += 1
        result.trades.append(Trade(i, signal.direction, abs(signal.score), r, kind, signal.risk_reward or 0.0,
                                  {f.name: f.value * (1 if signal.direction == "long" else -1) for f in signal.factors}))
        busy_until = i + used + 1  # one position at a time
    return result
