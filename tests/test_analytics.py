import numpy as np
import pandas as pd

from app.analytics.levels import find_levels
from app.analytics.volume import volume_profile, volume_summary


def sample_frame(n=160, freq="h"):
    x = np.arange(n)
    close = 100 + np.sin(x / 5) * 4 + x * 0.01
    return pd.DataFrame({"timestamp": pd.date_range("2025-01-01", periods=n, freq=freq, tz="UTC"),
                         "open": close - .3, "high": close + 1, "low": close - 1,
                         "close": close, "volume": np.full(n, 10.0)})


def test_rvol_and_time_window_volumes():
    frame = sample_frame()
    result = volume_summary(frame, 100, {"quoteVolume": 12345})
    assert np.isfinite(result["rvol"]) and result["rvol"] > 0
    assert result["volume_1h_quote"] == 10 * ((frame.high.iloc[-2] + frame.low.iloc[-2] + frame.close.iloc[-2]) / 3)
    assert result["volume_24h_quote"] == 12345
    assert np.isfinite(result["vwap_24h"])


def test_time_windows_do_not_depend_on_bar_count():
    # 5m data spanning 160 bars is under 24h, so missing day volume must remain unknown.
    result = volume_summary(sample_frame(freq="5min"), 100)
    assert np.isnan(result["volume_24h_quote"])


def test_profile_has_contiguous_ordered_value_area():
    profile = volume_profile(sample_frame())
    assert profile["val"] <= profile["poc"] <= profile["vah"]


def test_levels_have_real_touch_volume_and_bounded_strength():
    support, resistance = find_levels(sample_frame())
    price = sample_frame().close.iloc[-1]
    assert len(support) <= 3 and len(resistance) <= 3
    assert all(level.high < price for level in support)
    assert all(level.low > price for level in resistance)
    assert all(level.volume > 0 and 20 <= level.strength <= 100 for level in support + resistance)


def test_value_area_covers_70_percent_of_volume():
    typical = np.array([100.0] * 10 + [101.0] * 4 + [99.0] * 4 + [105.0] * 2 + [95.0] * 2)
    frame = pd.DataFrame({"high": typical, "low": typical, "close": typical, "volume": np.ones(len(typical))})
    profile = volume_profile(frame, bins=11)
    inside = frame[(frame.close >= profile["val"]) & (frame.close <= profile["vah"])]
    assert len(inside) / len(frame) >= 0.7
    assert profile["vah"] - profile["val"] < 8  # tails 95/105 stay outside


def test_signal_direction_and_pullback_entry():
    from app.analytics.flow import FlowReport, WindowStats
    from app.analytics.levels import PriceLevel
    from app.analytics.signal import compute_signal
    n = 120
    up = pd.DataFrame({"timestamp": pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC"),
                       "open": np.linspace(100, 130, n), "high": np.linspace(101, 131, n),
                       "low": np.linspace(99, 129, n), "close": np.linspace(100, 130, n), "volume": 1.0})

    def flow(buy, sell):
        w = {m: WindowStats(m, True, 10, 1, buy + sell, buy, sell) for m in (5, 60)}
        return FlowReport(60, 100, w, 1, [], [], 1.0, None)
    sup = [PriceLevel(127.0, 128.0, 3, 10.0, 70.0, 1.5)]
    res = [PriceLevel(135.0, 136.0, 3, 10.0, 70.0, 4.0)]
    bull = compute_signal(up, 130.0, sup, res, flow(80, 20), 125)
    assert bull.direction == "long" and bull.entry_zone == (127.0, 128.0)
    assert bull.entry == 128.0 and not bull.entry_now          # limit at the zone edge, not at market
    assert bull.stop < 127.0 and bull.targets[0] == 135.0 and bull.risk_reward > 1
    down = up.assign(close=up.close[::-1].to_numpy(), high=up.high[::-1].to_numpy(), low=up.low[::-1].to_numpy())
    bear = compute_signal(down, 100.0, [PriceLevel(97.0, 98.0, 3, 10.0, 70.0, 2.0)],
                          [PriceLevel(102.0, 103.0, 3, 10.0, 70.0, 2.5)], flow(20, 80), 105)
    assert bear.direction == "short" and bear.entry == 102.0 and bear.stop > 103.0 and bear.targets[0] == 98.0
    no_zone = compute_signal(up, 130.0, [], [], flow(80, 20), 125)
    assert no_zone.direction == "long" and no_zone.entry is None   # direction without a sensible entry zone


def test_telegram_html_has_only_allowed_tags():
    import re

    from app.analytics.report import signal_lines
    from app.analytics.signal import Signal
    sig = Signal("short", -60, [], 100.0, 102.0, [99.9], 0.2, (100.0, 101.0), "сопротивление", True)
    text = "\n".join(signal_lines(sig))
    assert "<" not in re.sub(r"</?b>", "", text)
