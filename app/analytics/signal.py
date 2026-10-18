"""Transparent long/short bias from observable factors. A weighted score, not a probability."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.analytics.flow import FlowReport
from app.analytics.levels import PriceLevel
from app.analytics.smc import SmcAnalysis
from app.analytics.technical import TechnicalAnalysis

NEUTRAL_BAND = 25  # |score| below this = no edge
MIN_RR = 1.5  # backtest: trades below this reward/risk lost money on average, so they are not offered
MAX_ENTRY_DISTANCE_ATR = 4.0  # farther pullback zones are not offered as an entry


@dataclass
class Factor:
    name: str
    value: float   # -1 (bearish) .. +1 (bullish)
    weight: float
    note: str


@dataclass
class Signal:
    direction: str            # "long" | "short" | "neutral"
    score: float              # -100 .. +100
    factors: list[Factor]
    entry: float | None = None
    stop: float | None = None
    targets: list[float] | None = None
    risk_reward: float | None = None
    entry_zone: tuple[float, float] | None = None
    entry_source: str = ""
    entry_now: bool = True    # False: wait for a pullback into entry_zone

    @property
    def actionable(self) -> bool:
        return self.entry is not None and self.risk_reward is not None and self.risk_reward >= MIN_RR


def _atr(candles: pd.DataFrame, n: int = 14) -> float:
    prev = candles.close.shift().fillna(candles.close)
    tr = np.maximum(candles.high - candles.low, np.maximum((candles.high - prev).abs(), (candles.low - prev).abs()))
    return float(tr.tail(n).mean())


def _score_factors(candles: pd.DataFrame, price: float, support: list[PriceLevel], resistance: list[PriceLevel],
                   flow: FlowReport | None, vwap: float, smc: SmcAnalysis | None,
                   tech: TechnicalAnalysis | None) -> list[Factor]:
    factors: list[Factor] = []
    close = candles.close
    ema_fast, ema_slow = close.ewm(span=20).mean().iloc[-1], close.ewm(span=50).mean().iloc[-1]
    trend = (1 if ema_fast > ema_slow else -1) * 0.6 + (1 if price > ema_slow else -1) * 0.4
    factors.append(Factor("Тренд", trend, 2.0, f"EMA20 {'выше' if ema_fast > ema_slow else 'ниже'} EMA50, цена {'выше' if price > ema_slow else 'ниже'} EMA50"))
    if pd.notna(vwap):
        factors.append(Factor("VWAP", 1.0 if price > vwap else -1.0, 1.0, f"цена {'выше' if price > vwap else 'ниже'} VWAP 24ч"))
    if flow:
        for minutes, weight, label in ((60, 2.0, "Delta 1ч"), (5, 1.0, "Delta 5м")):
            w = flow.windows[minutes]
            if w.covered and w.quote:
                share = w.delta / w.quote
                factors.append(Factor(label, float(np.clip(share / 0.2, -1, 1)), weight,
                                      f"покупки {w.buy_quote / w.quote:.0%} / продажи {w.sell_quote / w.quote:.0%}"))
    if smc:
        last = smc.events[-1] if smc.events else None
        struct = {"bull": 1.0, "bear": -1.0, "range": 0.0}[smc.trend]
        note = f"структура {'бычья' if struct > 0 else 'медвежья' if struct < 0 else 'не определена'}"
        if last:
            note += f", последнее событие {last.kind} {'вверх' if last.direction == 'bull' else 'вниз'}"
        factors.append(Factor("SMC структура", struct, 2.0, note))
        if smc.zone_label in ("premium", "discount"):
            val = -0.5 if smc.zone_label == "premium" else 0.5
            factors.append(Factor("Premium/Discount", val, 1.0,
                                  f"цена в {'premium (дорогая часть диапазона)' if val < 0 else 'discount (дешевая часть диапазона)'}"))
    if tech:  # momentum only: EMA trend is already counted above
        osc = tech.osc_score
        factors.append(Factor("Осцилляторы", float(np.clip(osc / 60, -1, 1)), 1.5,
                              f"RSI, MACD и Bollinger в сумме {'за рост' if osc > 0 else 'за падение'} ({abs(osc):.0f} из 100)"))
    loc = 0.0
    notes = []
    if resistance and resistance[0].distance_pct <= 1.0:
        loc -= 0.5
        notes.append(f"сопротивление в {resistance[0].distance_pct:.2f}%")
    if support and support[0].distance_pct <= 1.0:
        loc += 0.5
        notes.append(f"поддержка в {support[0].distance_pct:.2f}%")
    factors.append(Factor("Расположение", loc, 1.5, ", ".join(notes) or "рядом нет сильных уровней"))
    return factors


def _entry_zones(direction: str, price: float, support: list[PriceLevel], resistance: list[PriceLevel],
                 smc: SmcAnalysis | None) -> list[tuple[float, float, str]]:
    """Candidate pullback zones on the entry side: demand for long, supply for short."""
    zones: list[tuple[float, float, str]] = []
    if direction == "long":
        zones += [(x.low, x.high, "поддержка") for x in support]
        if smc:
            zones += [(z.low, z.high, "бычий Order Block") for z in smc.order_blocks if z.direction == "bull"]
            zones += [(z.low, z.high, "бычий FVG") for z in smc.fvgs if z.direction == "bull"]
        return [z for z in zones if z[0] < price]
    zones += [(x.low, x.high, "сопротивление") for x in resistance]
    if smc:
        zones += [(z.low, z.high, "медвежий Order Block") for z in smc.order_blocks if z.direction == "bear"]
        zones += [(z.low, z.high, "медвежий FVG") for z in smc.fvgs if z.direction == "bear"]
    return [z for z in zones if z[1] > price]


def compute_signal(candles: pd.DataFrame, price: float, support: list[PriceLevel], resistance: list[PriceLevel],
                   flow: FlowReport | None, vwap: float,
                   smc: SmcAnalysis | None = None, tech: TechnicalAnalysis | None = None) -> Signal:
    factors = _score_factors(candles, price, support, resistance, flow, vwap, smc, tech)
    score = float(sum(f.value * f.weight for f in factors) / sum(f.weight for f in factors) * 100)
    direction = "long" if score >= NEUTRAL_BAND else "short" if score <= -NEUTRAL_BAND else "neutral"
    signal = Signal(direction, score, factors)
    if direction == "neutral":
        return signal
    atr = _atr(candles)
    long = direction == "long"
    # nearest pullback zone on the entry side; price inside a zone means entry now
    zones = _entry_zones(direction, price, support, resistance, smc)
    inside = [z for z in zones if z[0] <= price <= z[1]]
    if inside:
        low, high, source = inside[0]
        entry, now = price, True
    elif zones:
        low, high, source = min(zones, key=lambda z: abs((z[1] if long else z[0]) - price))
        entry = high if long else low
        now = abs(entry - price) <= atr * 0.15
        if abs(entry - price) > atr * MAX_ENTRY_DISTANCE_ATR:
            return Signal(direction, score, factors)  # direction exists, but no sensible zone to enter from
    else:
        return Signal(direction, score, factors)
    stop = low - atr * 0.3 if long else high + atr * 0.3
    if long:
        targets = [x.low for x in resistance if x.low > entry][:2]
        risk, reward = entry - stop, (targets[0] - entry) if targets else 0.0
    else:
        targets = [x.high for x in support if x.high < entry][:2]
        risk, reward = stop - entry, (entry - targets[0]) if targets else 0.0
    if not targets:
        targets = [entry + atr * 2] if long else [entry - atr * 2]
        reward = atr * 2
    signal.entry, signal.stop, signal.targets = float(entry), float(stop), [float(t) for t in targets]
    signal.risk_reward = float(reward / risk) if risk > 0 and reward > 0 else None
    signal.entry_zone, signal.entry_source, signal.entry_now = (float(low), float(high)), source, now
    return signal
