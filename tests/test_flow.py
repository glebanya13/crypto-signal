import numpy as np
import pandas as pd

from app.analytics.flow import analyze_flow, detect_spike, find_clusters, to_frame, window_stats


def make_trades(n=3000, seconds=3600, seed=1):
    rng = np.random.default_rng(seed)
    ts = np.sort(rng.integers(0, seconds * 1000, n)) + 1_700_000_000_000
    qty = rng.lognormal(-3, 0.8, n)
    return [{"a": i, "T": int(t), "p": "100", "q": f"{q:.6f}", "m": bool(rng.integers(0, 2))}
            for i, (t, q) in enumerate(zip(ts, qty))]


def test_aggressor_side_and_cvd():
    frame = to_frame([{"T": 1000, "p": "100", "q": "2", "m": False}, {"T": 2000, "p": "100", "q": "1", "m": True}])
    w = window_stats(frame, 5)
    assert w.buy_quote == 200 and w.sell_quote == 100 and w.delta == 100


def test_window_not_covered_when_tape_is_short():
    frame = to_frame(make_trades(seconds=120))
    assert not window_stats(frame, 5).covered
    assert not window_stats(frame, 60).covered


def test_window_covered_and_sums_match():
    frame = to_frame(make_trades())
    w = window_stats(frame, 5)
    assert w.covered and abs(w.quote - (w.buy_quote + w.sell_quote)) < 1e-6
    assert w.count == (frame.ts > frame.ts.iloc[-1] - 300_000).sum()


def test_dynamic_whale_threshold_and_ordering():
    trades = make_trades()
    trades[1500]["q"] = "500"  # one very large print
    report = analyze_flow(trades)
    assert report.whales[0].quote == 50_000
    assert all(a.quote >= b.quote for a, b in zip(report.whales, report.whales[1:]))
    assert all(w.quote >= report.threshold for w in report.whales)


def test_cluster_requires_same_side_and_small_gaps():
    rows = [{"ts": 1000 * i, "sell": False, "quote": 1.0} for i in range(5)]
    rows += [{"ts": 5000 + 1000 * i, "sell": True, "quote": 1.0} for i in range(2)]
    clusters = find_clusters(pd.DataFrame(rows))
    assert len(clusters) == 1 and clusters[0].side == "buy" and clusters[0].count == 5
    assert find_clusters(pd.DataFrame([{"ts": 0, "sell": False, "quote": 1.0}, {"ts": 90_000, "sell": False, "quote": 1.0}] * 3)) == []


def test_spike_detected_only_when_anomalous():
    trades = make_trades(seconds=3600)
    frame = to_frame(trades)
    assert detect_spike(frame) is None
    base = frame.ts.iloc[-1]
    spike = [{"a": 10_000 + i, "T": int((base // 60_000) * 60_000 - 30_000 + i), "p": "100", "q": "50", "m": False} for i in range(100)]
    assert detect_spike(to_frame(trades + spike)) is not None


def test_spike_needs_enough_history():
    assert detect_spike(to_frame(make_trades(seconds=300))) is None
