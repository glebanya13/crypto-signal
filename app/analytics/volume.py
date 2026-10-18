import numpy as np
import pandas as pd


def _quote_volume(frame: pd.DataFrame) -> pd.Series:
    return ((frame.high + frame.low + frame.close) / 3) * frame.volume


def _quote_sum(frame: pd.DataFrame) -> float:
    return float(_quote_volume(frame).sum())


def volume_summary(candles: pd.DataFrame, price: float, ticker: dict | None = None) -> dict[str, float]:
    """Candlestick-derived volumes; UTC windows are time-based, never candle-count based."""
    frame = candles.sort_values("timestamp").copy()
    frame["timestamp"] = pd.to_datetime(frame.timestamp, utc=True)
    frame["quote_volume"] = _quote_volume(frame)
    last = frame.iloc[-1]
    previous = frame.iloc[-2]
    rvol_window = frame.volume.iloc[-21:-1]
    rvol = float(last.volume / rvol_window.mean()) if len(rvol_window) and rvol_window.mean() else float("nan")
    # OHLCV may end with an in-progress candle. Aggregate only candles whose full interval is closed.
    duration = frame.timestamp.diff().median()
    if pd.isna(duration) or duration <= pd.Timedelta(0):
        duration = pd.Timedelta(0)
    end = frame.timestamp.iloc[-1] - duration if duration > pd.Timedelta(0) else frame.timestamp.iloc[-1]
    start_1h = end - pd.Timedelta(hours=1)
    start_24h = end - pd.Timedelta(hours=24)
    last_hour = frame.loc[(frame.timestamp > start_1h) & (frame.timestamp <= end)]
    last_day = frame.loc[(frame.timestamp > start_24h) & (frame.timestamp <= end)]
    prev_hour = frame.loc[(frame.timestamp > start_1h - pd.Timedelta(hours=1)) & (frame.timestamp <= start_1h)]
    expected_bars_hour = max(1, int(pd.Timedelta(hours=1) / duration)) if duration > pd.Timedelta(0) else 1
    expected_bars_day = max(1, int(pd.Timedelta(hours=24) / duration)) if duration > pd.Timedelta(0) else 1
    enough_hour = len(last_hour) >= expected_bars_hour
    enough_day = len(last_day) >= expected_bars_day
    enough_prev_hour = len(prev_hour) >= expected_bars_hour
    vwap_period = frame.loc[(frame.timestamp > start_24h) & (frame.timestamp <= end)]
    vwap = float((vwap_period.quote_volume.sum() / vwap_period.volume.sum()) if vwap_period.volume.sum() else float("nan"))
    ticker = ticker or {}
    # CCXT Binance ticker.quoteVolume is the exchange's rolling 24h quote volume.
    ticker_24h = ticker.get("quoteVolume")
    return {
        "volume_base_last": float(last.volume), "volume_quote_last": float(last.quote_volume),
        "volume_change_pct": float((last.volume / previous.volume - 1) * 100) if previous.volume else float("nan"),
        "rvol": rvol, "vwap_24h": vwap,
        "volume_1h_quote": _quote_sum(last_hour) if enough_hour else float("nan"),
        "volume_1h_change_pct": (_quote_sum(last_hour) / _quote_sum(prev_hour) - 1) * 100 if enough_hour and enough_prev_hour and _quote_sum(prev_hour) else float("nan"),
        "volume_24h_quote": float(ticker_24h) if ticker_24h is not None else (_quote_sum(last_day) if enough_day else float("nan")),
        "current_price": price,
    }


def volume_profile(candles: pd.DataFrame, bins: int = 32) -> dict[str, float]:
    """Approximate profile: allocate each candle volume to its typical price; contiguous 70% VA."""
    typical = (candles.high + candles.low + candles.close) / 3
    weights, edges = np.histogram(typical, bins=bins, weights=candles.volume)
    idx = int(np.argmax(weights))
    poc = float((edges[idx] + edges[idx + 1]) / 2)
    low = high = idx
    accumulated = float(weights[idx])
    target = float(weights.sum()) * 0.70
    while accumulated < target and (low > 0 or high < len(weights) - 1):
        left = float(weights[low - 1]) if low > 0 else -1.0
        right = float(weights[high + 1]) if high < len(weights) - 1 else -1.0
        if right > left:
            high += 1
            accumulated += right
        else:
            low -= 1
            accumulated += left
    return {"poc": poc, "val": float(edges[low]), "vah": float(edges[high + 1])}
