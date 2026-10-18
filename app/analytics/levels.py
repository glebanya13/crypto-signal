from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import find_peaks


@dataclass(frozen=True)
class PriceLevel:
    low: float
    high: float
    touches: int
    volume: float
    strength: float
    distance_pct: float
    last_touch: pd.Timestamp | None = None
    age_hours: float | None = None


def _make_zones(prices: np.ndarray, volumes: np.ndarray, timestamps: np.ndarray,
                current: float, atr: float, now: pd.Timestamp) -> list[PriceLevel]:
    if not len(prices):
        return []
    tolerance = max(atr * 0.45, current * 0.0015)
    ordered = sorted(zip((float(p) for p in prices), volumes, timestamps), key=lambda item: item[0])
    groups: list[list[tuple[float, float, pd.Timestamp]]] = []
    for price, volume, timestamp in ordered:
        if not groups or price - np.mean([item[0] for item in groups[-1]]) > tolerance:
            groups.append([(price, float(volume), pd.Timestamp(timestamp))])
        else:
            groups[-1].append((price, float(volume), pd.Timestamp(timestamp)))
    result = []
    for group in groups:
        vals = np.array([item[0] for item in group])
        volume = float(sum(item[1] for item in group))
        touches = len(group)
        age_days = max(0.0, (now - max(item[2] for item in group)).total_seconds() / 86400)
        recency = float(np.exp(-age_days / 30))
        volume_score = min(1.0, volume / max(float(np.median(volumes)) * 5, 1e-12))
        # Heuristic score, explicitly not a probability; bounded, inputs are observable.
        strength = min(100.0, 20 + 12 * min(touches, 4) + 25 * volume_score + 15 * recency)
        center = float(np.mean(vals))
        last_touch = max(item[2] for item in group)
        result.append(PriceLevel(float(vals.min()), float(vals.max()), touches, volume, strength,
                                 abs(center - current) / current * 100, last_touch, age_days * 24))
    return result


def level_zones(frame: pd.DataFrame) -> tuple[list[PriceLevel], list[PriceLevel]]:
    """All clustered swing zones, not filtered by side: (zones from swing lows, zones from swing highs)."""
    high, low = frame.high.to_numpy(), frame.low.to_numpy()
    current = float(frame.close.iloc[-1])
    prev_close = frame.close.shift().fillna(frame.close).to_numpy()
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    atr = float(np.mean(tr[-14:]))
    distance = max(3, min(8, len(frame) // 60))
    prominence = max(atr * 0.3, current * 0.001)
    highs, _ = find_peaks(high, distance=distance, prominence=prominence)
    lows, _ = find_peaks(-low, distance=distance, prominence=prominence)
    now = pd.Timestamp(frame.timestamp.iloc[-1])
    timestamps = frame.timestamp.to_numpy()
    resist = _make_zones(high[highs], frame.volume.iloc[highs].to_numpy(), timestamps[highs], current, atr, now)
    support = _make_zones(low[lows], frame.volume.iloc[lows].to_numpy(), timestamps[lows], current, atr, now)
    return support, resist


def find_levels(candles: pd.DataFrame, max_each_side: int = 3) -> tuple[list[PriceLevel], list[PriceLevel]]:
    """Cluster swing extrema; strength is a transparent heuristic, not breakout probability."""
    frame = candles.sort_values("timestamp")
    current = float(frame.close.iloc[-1])
    support, resist = level_zones(frame)
    support = sorted((x for x in support if x.high < current), key=lambda x: x.distance_pct)[:max_each_side]
    resist = sorted((x for x in resist if x.low > current), key=lambda x: x.distance_pct)[:max_each_side]
    return support, resist


@dataclass(frozen=True)
class PivotPoints:
    pivot: float
    r1: float
    r2: float
    s1: float
    s2: float


def pivot_points(daily: pd.DataFrame) -> PivotPoints | None:
    """Classic floor pivots from the last closed daily candle."""
    if daily.empty:
        return None
    last = daily.sort_values("timestamp").iloc[-1]
    h, l, c = float(last.high), float(last.low), float(last.close)
    p = (h + l + c) / 3
    return PivotPoints(p, 2 * p - l, p + (h - l), 2 * p - h, p - (h - l))


def profile_nodes(candles: pd.DataFrame, bins: int = 48) -> tuple[list[float], list[float]]:
    """High/low volume nodes (HVN/LVN) from the candle-approximated profile: peaks and troughs of the histogram."""
    typical = (candles.high + candles.low + candles.close) / 3
    weights, edges = np.histogram(typical, bins=bins, weights=candles.volume)
    centers = (edges[:-1] + edges[1:]) / 2
    smooth = np.convolve(weights, np.ones(3) / 3, mode="same")
    peaks, _ = find_peaks(smooth, prominence=float(smooth.max()) * 0.12)
    troughs, _ = find_peaks(-smooth, prominence=float(smooth.max()) * 0.12)
    return [float(centers[i]) for i in peaks], [float(centers[i]) for i in troughs]


@dataclass(frozen=True)
class Breakout:
    direction: str          # "up" through resistance | "down" through support
    zone: PriceLevel
    bars_ago: int
    status: str             # "ретест удержан" | "ждет ретеста" | "ложный пробой"
    volume_ratio: float     # breakout bar volume vs previous 20 bars


def detect_breakouts(candles: pd.DataFrame, lookback: int = 40) -> list[Breakout]:
    """Closes through a zone in the last `lookback` bars, and whether the retest held."""
    frame = candles.sort_values("timestamp").reset_index(drop=True)
    support, resist = level_zones(frame)
    close, high, low = frame.close.to_numpy(), frame.high.to_numpy(), frame.low.to_numpy()
    n = len(frame)
    out: list[Breakout] = []
    for zones, up in ((resist, True), (support, False)):
        for zone in zones:
            for k in range(n - 1, max(n - lookback, 1), -1):
                crossed = close[k] > zone.high and close[k - 1] <= zone.high if up else close[k] < zone.low and close[k - 1] >= zone.low
                if not crossed:
                    continue
                after = slice(k + 1, n)
                now_beyond = close[-1] > zone.high if up else close[-1] < zone.low
                if up:
                    retested = bool((low[after] <= zone.high * 1.001).any())
                    held = retested and close[-1] > zone.high
                else:
                    retested = bool((high[after] >= zone.low * 0.999).any())
                    held = retested and close[-1] < zone.low
                status = "ретест удержан" if held else "ложный пробой, цена вернулась" if not now_beyond else "ждет ретеста"
                base = frame.volume.iloc[max(0, k - 20):k].mean()
                out.append(Breakout("up" if up else "down", zone, n - 1 - k, status,
                                    float(frame.volume.iloc[k] / base) if base else float("nan")))
                break
    return sorted(out, key=lambda b: b.bars_ago)[:3]
