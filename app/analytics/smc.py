"""Smart Money Concepts on OHLCV: market structure, order blocks, fair value gaps, liquidity.

All definitions are rule-based and mechanical (documented below); they describe price action, they do not
identify who traded.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

SWING_N = 3  # fractal width: a swing needs N lower highs (higher lows) on each side


@dataclass
class Swing:
    idx: int
    kind: str      # "H" | "L"
    price: float


@dataclass
class StructureEvent:
    idx: int
    kind: str      # "BOS" | "CHoCH"
    direction: str  # "bull" | "bear"
    level: float
    ts: pd.Timestamp


@dataclass
class Zone:
    kind: str       # "OB" | "FVG"
    direction: str  # "bull" | "bear"
    low: float
    high: float
    idx: int
    ts: pd.Timestamp
    tested: bool = False


@dataclass
class LiquidityPool:
    side: str       # "buy" (equal highs, stops above) | "sell" (equal lows, stops below)
    price: float
    touches: int


@dataclass
class Sweep:
    side: str       # "buy" = highs swept then rejected | "sell" = lows swept then rejected
    level: float
    idx: int
    ts: pd.Timestamp


@dataclass
class SmcAnalysis:
    trend: str                                  # "bull" | "bear" | "range"
    swings: list[Swing]
    events: list[StructureEvent]
    order_blocks: list[Zone]                    # unmitigated
    fvgs: list[Zone]                            # not fully filled
    pools: list[LiquidityPool]
    sweep: Sweep | None
    range_high: float | None = None
    range_low: float | None = None
    position_pct: float | None = None           # 0 = range low, 100 = range high
    zone_label: str = "н/д"                     # premium / discount / equilibrium
    notes: list[str] = field(default_factory=list)


def _atr(frame: pd.DataFrame, n: int = 14) -> float:
    prev = frame.close.shift().fillna(frame.close)
    tr = np.maximum(frame.high - frame.low, np.maximum((frame.high - prev).abs(), (frame.low - prev).abs()))
    return float(tr.tail(n).mean())


def find_swings(frame: pd.DataFrame, n: int = SWING_N) -> list[Swing]:
    high, low = frame.high.to_numpy(), frame.low.to_numpy()
    raw: list[Swing] = []
    for i in range(n, len(frame) - n):
        if high[i] == high[i - n:i + n + 1].max() and high[i] > high[i - n:i].max():
            raw.append(Swing(i, "H", float(high[i])))
        if low[i] == low[i - n:i + n + 1].min() and low[i] < low[i - n:i].min():
            raw.append(Swing(i, "L", float(low[i])))
    raw.sort(key=lambda s: s.idx)
    swings: list[Swing] = []
    for s in raw:  # collapse consecutive same-type swings to the more extreme one
        if swings and swings[-1].kind == s.kind:
            if (s.kind == "H" and s.price > swings[-1].price) or (s.kind == "L" and s.price < swings[-1].price):
                swings[-1] = s
        else:
            swings.append(s)
    return swings


def _order_block(frame: pd.DataFrame, start: int, end: int, bullish: bool) -> Zone | None:
    """Last opposite-colour candle at the origin of the impulse that broke structure."""
    seg = frame.iloc[start:end + 1]
    if seg.empty:
        return None
    origin = int(seg.low.values.argmin() if bullish else seg.high.values.argmax()) + start
    for j in range(origin, max(start - 1, origin - 4), -1):
        row = frame.iloc[j]
        if (row.close < row.open) if bullish else (row.close > row.open):
            return Zone("OB", "bull" if bullish else "bear", float(row.low), float(row.high), j, row.timestamp)
    return None


def analyze_smc(candles: pd.DataFrame, n: int = SWING_N, max_zones: int = 3) -> SmcAnalysis:
    frame = candles.sort_values("timestamp").reset_index(drop=True)
    close = frame.close.to_numpy()
    swings = find_swings(frame, n)
    atr = _atr(frame)
    price = float(close[-1])

    # --- structure: BOS / CHoCH on closes through the latest confirmed, unbroken swing ---
    events: list[StructureEvent] = []
    trend_state = 0
    last_high: Swing | None = None
    last_low: Swing | None = None
    obs: list[Zone] = []
    pointer = 0
    for i in range(len(frame)):
        while pointer < len(swings) and swings[pointer].idx + n <= i:
            s = swings[pointer]
            if s.kind == "H":
                last_high = s
            else:
                last_low = s
            pointer += 1
        if last_high and close[i] > last_high.price:
            kind = "CHoCH" if trend_state == -1 else "BOS"
            events.append(StructureEvent(i, kind, "bull", last_high.price, frame.timestamp.iloc[i]))
            ob = _order_block(frame, last_high.idx, i, bullish=True)
            if ob:
                obs.append(ob)
            trend_state, last_high = 1, None
        elif last_low and close[i] < last_low.price:
            kind = "CHoCH" if trend_state == 1 else "BOS"
            events.append(StructureEvent(i, kind, "bear", last_low.price, frame.timestamp.iloc[i]))
            ob = _order_block(frame, last_low.idx, i, bullish=False)
            if ob:
                obs.append(ob)
            trend_state, last_low = -1, None
    trend = {1: "bull", -1: "bear", 0: "range"}[trend_state]

    # --- order blocks: keep unmitigated (no later close through the far side) ---
    live_obs = []
    seen: set[int] = set()
    for ob in obs:
        if ob.idx in seen:
            continue
        seen.add(ob.idx)
        after = frame.iloc[ob.idx + 1:]
        if ob.direction == "bull":
            mitigated = bool((after.close < ob.low).any())
            ob.tested = bool((after.low <= ob.high).any())
        else:
            mitigated = bool((after.close > ob.high).any())
            ob.tested = bool((after.high >= ob.low).any())
        if not mitigated:
            live_obs.append(ob)

    # --- fair value gaps: 3-candle imbalance not yet fully filled ---
    high, low = frame.high.to_numpy(), frame.low.to_numpy()
    fvgs: list[Zone] = []
    min_gap = atr * 0.25
    for i in range(2, len(frame)):
        if low[i] - high[i - 2] > min_gap:
            lo, hi, direction = float(high[i - 2]), float(low[i]), "bull"
            filled = bool((low[i + 1:] <= lo).any())
            tested = bool((low[i + 1:] <= hi).any())
        elif low[i - 2] - high[i] > min_gap:
            lo, hi, direction = float(high[i]), float(low[i - 2]), "bear"
            filled = bool((high[i + 1:] >= hi).any())
            tested = bool((high[i + 1:] >= lo).any())
        else:
            continue
        if not filled:
            fvgs.append(Zone("FVG", direction, lo, hi, i - 1, frame.timestamp.iloc[i - 1], tested))

    # --- liquidity: equal highs / equal lows (resting stops) and recent sweeps ---
    tol = max(atr * 0.15, price * 0.0005)
    pools: list[LiquidityPool] = []
    for kind, side in (("H", "buy"), ("L", "sell")):
        pts = sorted((s.price, s.idx) for s in swings if s.kind == kind)
        group: list[tuple[float, int]] = []
        for p, idx in [*pts, (float("inf"), 0)]:
            if group and p - group[-1][0] > tol:
                if len(group) >= 2:
                    prices = [g[0] for g in group]
                    level = max(prices) if kind == "H" else min(prices)
                    later = slice(max(g[1] for g in group) + 1, None)
                    resting = (high[later].max(initial=-np.inf) < level) if kind == "H" else (low[later].min(initial=np.inf) > level)
                    on_side = (price < level) if kind == "H" else (price > level)
                    if resting and on_side:
                        pools.append(LiquidityPool(side, float(level), len(group)))
                group = []
            group.append((p, idx))
    pools.sort(key=lambda x: abs(x.price - price))

    sweep: Sweep | None = None
    confirmed = [s for s in swings if s.idx + n <= len(frame) - 1]
    for i in range(len(frame) - 1, max(len(frame) - 21, n), -1):
        prior_h = [s for s in confirmed if s.kind == "H" and s.idx + n <= i]
        prior_l = [s for s in confirmed if s.kind == "L" and s.idx + n <= i]
        if prior_h and high[i] > prior_h[-1].price > close[i]:
            sweep = Sweep("buy", prior_h[-1].price, i, frame.timestamp.iloc[i])
            break
        if prior_l and low[i] < prior_l[-1].price < close[i]:
            sweep = Sweep("sell", prior_l[-1].price, i, frame.timestamp.iloc[i])
            break

    # --- premium / discount of the current dealing range ---
    sh = next((s for s in reversed(swings) if s.kind == "H"), None)
    sl = next((s for s in reversed(swings) if s.kind == "L"), None)
    result = SmcAnalysis(trend, swings, events, sorted(live_obs, key=lambda z: -z.idx)[:max_zones * 2],
                         sorted(fvgs, key=lambda z: -z.idx)[:max_zones * 2], pools[:4], sweep)
    if sh and sl and sh.price > sl.price:
        result.range_high, result.range_low = sh.price, sl.price
        pos = (price - sl.price) / (sh.price - sl.price) * 100
        result.position_pct = float(pos)
        result.zone_label = "premium" if pos > 55 else "discount" if pos < 45 else "equilibrium"
    return result
