"""Trade-tape analytics from Binance Spot aggTrades. Side = aggressor side, not money in/out."""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class WindowStats:
    minutes: int
    covered: bool          # tape reaches back far enough to fill the window
    count: int = 0
    base: float = 0.0
    quote: float = 0.0
    buy_quote: float = 0.0
    sell_quote: float = 0.0

    @property
    def delta(self) -> float:
        return self.buy_quote - self.sell_quote


@dataclass
class WhaleTrade:
    ts_ms: int
    price: float
    quote: float
    side: str              # "buy" | "sell" (aggressor)
    x_median: float        # size / median trade
    pct_of_minute: float   # size / average 1-minute turnover, %


@dataclass
class Cluster:
    side: str
    start_ms: int
    end_ms: int
    count: int
    quote: float


@dataclass
class FlowReport:
    span_minutes: float
    trades: int
    windows: dict[int, WindowStats]
    threshold: float
    whales: list[WhaleTrade]
    clusters: list[Cluster]
    rvol_5m: float
    spike: dict | None
    all_large: list[WhaleTrade] = field(default_factory=list)


def to_frame(trades: list[dict]) -> pd.DataFrame:
    """aggTrade dicts -> frame; m=True means buyer is maker, so the seller was the aggressor."""
    frame = pd.DataFrame({
        "ts": [int(t["T"]) for t in trades],
        "price": [float(t["p"]) for t in trades],
        "qty": [float(t["q"]) for t in trades],
        "sell": [bool(t["m"]) for t in trades],
    })
    frame["quote"] = frame.price * frame.qty
    return frame.sort_values("ts").reset_index(drop=True)


def window_stats(frame: pd.DataFrame, minutes: int) -> WindowStats:
    if frame.empty:
        return WindowStats(minutes, False)
    end = int(frame.ts.iloc[-1])
    start = end - minutes * 60_000
    part = frame[frame.ts > start]
    sells = float(part.loc[part.sell, "quote"].sum())
    buys = float(part.loc[~part.sell, "quote"].sum())
    covered = int(frame.ts.iloc[0]) <= start + 60_000  # tape begins within a minute of the window start
    return WindowStats(minutes, covered, len(part), float(part.qty.sum()), buys + sells, buys, sells)


def whale_threshold(frame: pd.DataFrame) -> float:
    """Dynamic: above the 99.9th percentile of trade size and at least 30x the median trade."""
    return float(max(np.percentile(frame.quote, 99.9), 30 * frame.quote.median()))


def cluster_threshold(frame: pd.DataFrame) -> float:
    """Looser tier (p99, >=10x median) used only to find runs of same-side big prints."""
    return float(max(np.percentile(frame.quote, 99), 10 * frame.quote.median()))


def find_clusters(large: pd.DataFrame, gap_ms: int = 20_000, min_count: int = 5) -> list[Cluster]:
    """Runs of same-side large trades with no gap longer than gap_ms."""
    clusters: list[Cluster] = []
    run: list = []

    def flush() -> None:
        if len(run) >= min_count:
            clusters.append(Cluster("buy" if not run[0].sell else "sell", int(run[0].ts), int(run[-1].ts),
                                    len(run), float(sum(r.quote for r in run))))
        run.clear()

    for row in large.itertuples():
        if run and (row.sell != run[-1].sell or row.ts - run[-1].ts > gap_ms):
            flush()
        run.append(row)
    flush()
    return clusters


def detect_spike(frame: pd.DataFrame, z_min: float = 3.5, ratio_min: float = 2.0) -> dict | None:
    """Robust (median/MAD) check of the last full minute's turnover against earlier minutes."""
    if frame.empty:
        return None
    minute = (frame.ts // 60_000) * 60_000
    per_min = frame.groupby(minute).quote.sum()
    if len(per_min) < 12:
        return None
    history, last = per_min.iloc[:-2], float(per_min.iloc[-2])  # last bucket is still filling
    median = float(history.median())
    mad = float((history - median).abs().median()) * 1.4826
    if median <= 0 or mad <= 0:
        return None
    z = (last - median) / mad
    if z >= z_min and last / median >= ratio_min:
        return {"minute_ms": int(per_min.index[-2]), "quote": last, "x_median": last / median, "z": z}
    return None


def analyze_flow(trades: list[dict], top: int = 10) -> FlowReport | None:
    if len(trades) < 50:
        return None
    frame = to_frame(trades)
    span = (int(frame.ts.iloc[-1]) - int(frame.ts.iloc[0])) / 60_000
    windows = {m: window_stats(frame, m) for m in (5, 60)}
    threshold = whale_threshold(frame)
    large = frame[frame.quote >= threshold]
    median = float(frame.quote.median())
    per_minute = float(frame.quote.sum() / max(span, 1 / 60))
    all_large = [WhaleTrade(int(r.ts), float(r.price), float(r.quote), "sell" if r.sell else "buy",
                            float(r.quote / median), float(r.quote / per_minute * 100)) for r in large.itertuples()]
    top_trades = sorted(all_large, key=lambda w: w.quote, reverse=True)[:top]
    # RVOL on the tape: last 5 min against the average 5-min turnover of the earlier tape.
    last5 = windows[5].quote
    earlier = frame[frame.ts <= int(frame.ts.iloc[-1]) - 5 * 60_000]
    earlier_span = (int(earlier.ts.iloc[-1]) - int(earlier.ts.iloc[0])) / 300_000 if len(earlier) > 1 else 0
    rvol = last5 / (earlier.quote.sum() / earlier_span) if earlier_span >= 2 and earlier.quote.sum() else float("nan")
    return FlowReport(span, len(frame), windows, threshold, top_trades, find_clusters(frame[frame.quote >= cluster_threshold(frame)]), rvol,
                      detect_spike(frame), all_large)
