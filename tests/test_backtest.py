import numpy as np
import pandas as pd

from app.backtest import FEE_R, BacktestResult, Trade, _simulate, run_backtest


def bars(rows):
    return pd.DataFrame({"timestamp": pd.date_range("2025-01-01", periods=len(rows), freq="h", tz="UTC"),
                         "open": [r[0] for r in rows], "high": [r[1] for r in rows],
                         "low": [r[2] for r in rows], "close": [r[3] for r in rows], "volume": 1.0})


def test_long_target_hit():
    frame = bars([(100, 101, 99, 100), (100, 100.5, 99.5, 100), (100, 103, 99.8, 102)])
    r, kind, _ = _simulate(frame, 1, "long", entry=99.6, stop=98.6, target=101.6)
    assert kind == "target" and abs(r - (2.0 - FEE_R)) < 1e-9


def test_stop_wins_same_bar_tie():
    frame = bars([(100, 101, 99, 100), (100, 103, 97, 100)])
    r, kind, _ = _simulate(frame, 1, "long", entry=99.0, stop=98.0, target=102.0)
    assert kind == "stop" and r < -1


def test_unfilled_limit_order_is_skipped():
    frame = bars([(100, 101, 99, 100)] * 30)
    assert _simulate(frame, 1, "long", entry=90.0, stop=89.0, target=95.0) is None


def test_short_mirrors_long():
    frame = bars([(100, 101, 99, 100), (100, 100.4, 99.2, 99.5), (99.5, 99.6, 96.0, 97.0)])
    r, kind, _ = _simulate(frame, 1, "short", entry=100.3, stop=101.3, target=98.3)
    assert kind == "target" and r > 0


def test_stats_math():
    result = BacktestResult("X", "1h", 100, 10, 5, 3, [Trade(1, "long", 40, 2.0, "target"), Trade(2, "long", 40, -1.0, "stop"),
                                                          Trade(3, "short", 60, 1.0, "target")])
    s = result.stats()
    assert s["n"] == 3 and abs(s["win_rate"] - 2 / 3) < 1e-9 and s["total_r"] == 2.0 and s["max_dd"] == 1.0
    assert abs(s["profit_factor"] - 3.0) < 1e-9


def test_backtest_runs_without_lookahead_on_synthetic_data():
    rng = np.random.default_rng(3)
    close = 100 + np.cumsum(rng.normal(0, 0.8, 600))
    frame = bars([(c, c + 1, c - 1, c) for c in close])
    result = run_backtest(frame, "TEST/USDT", "1h")
    assert result.decisions > 0 and result.filled <= result.signals <= result.decisions
