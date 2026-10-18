"""Classic technical indicators computed from OHLCV (pure pandas/numpy)."""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from app.analytics.smc import find_swings


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(100.0)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[pd.Series, pd.Series, pd.Series]:
    line = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
    sig = line.ewm(span=signal, adjust=False).mean()
    return line, sig, line - sig


def atr(frame: pd.DataFrame, n: int = 14) -> pd.Series:
    prev = frame.close.shift().fillna(frame.close)
    tr = np.maximum(frame.high - frame.low, np.maximum((frame.high - prev).abs(), (frame.low - prev).abs()))
    return tr.ewm(alpha=1 / n, adjust=False).mean()


@dataclass
class Indicator:
    name: str
    text: str
    score: float  # -1 bearish .. +1 bullish


@dataclass
class TechnicalAnalysis:
    price: float
    indicators: list[Indicator]
    score: float                    # -100..100, mean of indicator scores
    atr_pct: float
    divergence: str | None
    notes: list[str] = field(default_factory=list)

    @property
    def osc_score(self) -> float:
        """Momentum-only score (RSI, MACD, Bollinger, divergence); excludes EMA trend to avoid double counting."""
        scored = [i.score for i in self.indicators if i.name not in ("Волатильность", "EMA")]
        return float(np.mean(scored) * 100) if scored else 0.0

    @property
    def bias(self) -> str:
        return "bull" if self.score >= 25 else "bear" if self.score <= -25 else "neutral"


def _divergence(frame: pd.DataFrame, rsi_s: pd.Series) -> str | None:
    """Regular divergence between the last two swing lows/highs and RSI at those bars."""
    swings = find_swings(frame)
    lows = [s for s in swings if s.kind == "L"][-2:]
    highs = [s for s in swings if s.kind == "H"][-2:]
    last = len(frame) - 1
    if len(lows) == 2 and last - lows[1].idx < 40 and lows[1].price < lows[0].price and rsi_s.iloc[lows[1].idx] > rsi_s.iloc[lows[0].idx] + 2:
        return "бычья дивергенция: цена обновила минимум, RSI - нет"
    if len(highs) == 2 and last - highs[1].idx < 40 and highs[1].price > highs[0].price and rsi_s.iloc[highs[1].idx] < rsi_s.iloc[highs[0].idx] - 2:
        return "медвежья дивергенция: цена обновила максимум, RSI - нет"
    return None


def analyze_technical(candles: pd.DataFrame) -> TechnicalAnalysis:
    frame = candles.sort_values("timestamp").reset_index(drop=True)
    close = frame.close
    price = float(close.iloc[-1])
    out: list[Indicator] = []

    emas = {n: close.ewm(span=n, adjust=False).mean() for n in (20, 50, 200)}
    have200 = len(frame) >= 200
    e20, e50 = float(emas[20].iloc[-1]), float(emas[50].iloc[-1])
    if have200:
        e200 = float(emas[200].iloc[-1])
        stack = (price > e20) + (e20 > e50) + (e50 > e200)
        score = (stack - 1.5) / 1.5
        text = f"цена {price:,.4g} · EMA20 {e20:,.4g} / EMA50 {e50:,.4g} / EMA200 {e200:,.4g}"
    else:
        score = ((price > e20) + (e20 > e50) - 1.0)
        text = f"цена {price:,.4g} · EMA20 {e20:,.4g} / EMA50 {e50:,.4g} (EMA200 н/д: мало истории)"
    out.append(Indicator("EMA", text, float(score)))

    r = rsi(close)
    rv = float(r.iloc[-1])
    rsi_score = -1.0 if rv >= 70 else 1.0 if rv <= 30 else float(np.clip((rv - 50) / 20, -1, 1)) * 0.6
    zone = "перекупленность" if rv >= 70 else "перепроданность" if rv <= 30 else "нейтральная зона"
    out.append(Indicator("RSI(14)", f"{rv:.1f} - {zone}", rsi_score))

    line, sig, hist = macd(close)
    h, h_prev = float(hist.iloc[-1]), float(hist.iloc[-2])
    cross = ""
    if line.iloc[-2] <= sig.iloc[-2] and line.iloc[-1] > sig.iloc[-1]:
        cross = ", свежее бычье пересечение"
    elif line.iloc[-2] >= sig.iloc[-2] and line.iloc[-1] < sig.iloc[-1]:
        cross = ", свежее медвежье пересечение"
    macd_score = (0.6 if h > 0 else -0.6) + (0.4 if h > h_prev else -0.4)
    out.append(Indicator("MACD", f"гистограмма {h:+.4g} ({'растет' if h > h_prev else 'падает'}){cross}", macd_score))

    mid = close.rolling(20).mean()
    std = close.rolling(20).std()
    upper, lower = mid + 2 * std, mid - 2 * std
    pb = float((price - lower.iloc[-1]) / (upper.iloc[-1] - lower.iloc[-1]))
    width = (upper - lower) / mid
    squeeze = bool(width.iloc[-1] <= width.tail(120).quantile(0.15))
    bb_score = float(np.clip((pb - 0.5) * 2, -1, 1)) * 0.5
    out.append(Indicator("Bollinger", f"%B {pb:.2f}" + (" · сжатие полос (низкая волатильность, часто перед импульсом)" if squeeze else ""), bb_score))

    atr_pct = float(atr(frame).iloc[-1] / price * 100)
    vol_ratio = float(frame.volume.iloc[-1] / frame.volume.tail(21).iloc[:-1].mean()) if frame.volume.tail(21).iloc[:-1].mean() else float("nan")
    out.append(Indicator("Волатильность", f"ATR {atr_pct:.2f}% от цены", 0.0))

    divergence = _divergence(frame, r)
    if divergence:
        out.append(Indicator("Дивергенция", divergence, 0.7 if divergence.startswith("бычья") else -0.7))
    scored = [i for i in out if i.name != "Волатильность"]
    total = float(np.mean([i.score for i in scored]) * 100)
    notes = []
    if pd.notna(vol_ratio):
        notes.append(f"объем последней свечи {vol_ratio:.2f}x от среднего (свеча может быть не закрыта)")
    return TechnicalAnalysis(price, out, total, atr_pct, divergence, notes)


def trend_label(candles: pd.DataFrame) -> tuple[str, float]:
    """Trend by EMA20/50 and price vs EMA50; returns (label, rsi)."""
    close = candles.sort_values("timestamp").close
    e20, e50 = close.ewm(span=20, adjust=False).mean().iloc[-1], close.ewm(span=50, adjust=False).mean().iloc[-1]
    price = close.iloc[-1]
    label = "восходящий" if e20 > e50 and price > e50 else "нисходящий" if e20 < e50 and price < e50 else "боковой/смешанный"
    return label, float(rsi(close).iloc[-1])
