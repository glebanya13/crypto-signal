import itertools

import numpy as np
import pandas as pd

from app.analytics.smc import analyze_smc, find_swings
from app.analytics.technical import analyze_technical, macd, rsi


def frame_from(closes, spread=0.5):
    closes = np.asarray(closes, dtype=float)
    opens = np.concatenate([[closes[0]], closes[:-1]])
    return pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=len(closes), freq="h", tz="UTC"),
        "open": opens, "high": np.maximum(opens, closes) + spread, "low": np.minimum(opens, closes) - spread,
        "close": closes, "volume": 1.0})


def zigzag(points, step=6):
    out = [points[0]]
    for a, b in itertools.pairwise(points):
        out += list(np.linspace(a, b, step + 1)[1:])
    return out


def test_swings_alternate():
    swings = find_swings(frame_from(zigzag([100, 110, 102, 112, 104, 114, 106])))
    kinds = [s.kind for s in swings]
    assert all(a != b for a, b in itertools.pairwise(kinds))


def test_uptrend_gives_bullish_structure_with_bos():
    smc = analyze_smc(frame_from(zigzag([100, 110, 104, 116, 109, 122, 115, 128, 121, 134])))
    assert smc.trend == "bull"
    assert any(e.kind == "BOS" and e.direction == "bull" for e in smc.events)


def test_choch_when_trend_flips():
    smc = analyze_smc(frame_from(zigzag([100, 110, 104, 116, 109, 122, 112, 104, 108, 95, 100, 90])))
    assert smc.trend == "bear"
    assert any(e.kind == "CHoCH" and e.direction == "bear" for e in smc.events)


def test_bullish_fvg_detected_and_filled_gap_removed():
    closes = [100] * 10 + [100, 108, 112] + [112] * 5
    frame = frame_from(closes, spread=0.2)
    gaps = analyze_smc(frame).fvgs
    assert any(z.direction == "bull" for z in gaps)
    refill = frame_from(closes + [95] * 3, spread=0.2)
    assert not any(z.direction == "bull" and z.low > 100 for z in analyze_smc(refill).fvgs)


def test_order_block_is_unmitigated_opposite_candle():
    smc = analyze_smc(frame_from(zigzag([100, 110, 104, 116, 109, 122, 115, 128])))
    bull = [z for z in smc.order_blocks if z.direction == "bull"]
    assert bull and all(z.low < z.high for z in bull)


def test_rsi_bounds_and_trend_direction():
    up = pd.Series(np.linspace(100, 150, 80))
    assert rsi(up).iloc[-1] > 70
    assert rsi(up[::-1].reset_index(drop=True)).iloc[-1] < 30
    line, sig, hist = macd(up)
    assert hist.iloc[-1] == line.iloc[-1] - sig.iloc[-1]


def test_technical_bias_matches_direction():
    up = analyze_technical(frame_from(np.linspace(100, 160, 260)))
    down = analyze_technical(frame_from(np.linspace(160, 100, 260)))
    assert up.score > down.score
    assert any(i.name == "EMA" and i.score > 0 for i in up.indicators)
    assert any(i.name == "EMA" and i.score < 0 for i in down.indicators)
