"""Shared process-wide services: market data clients, tape cache, backtest cache, per-user state."""
import logging
import time

from app.analytics.flow import FlowReport, analyze_flow
from app.backtest import BacktestResult
from app.market.binance import BinanceSpotProvider
from app.market.binance_stream import BinanceSpotStream

logger = logging.getLogger(__name__)
provider = BinanceSpotProvider()
stream = BinanceSpotStream()

BACKTEST_TTL = 1800.0
_backtests: dict[tuple, tuple[float, BacktestResult]] = {}
# last shown snapshot per (chat, symbol, timeframe, action) - powers the "what's new" block on refresh
snapshots: dict[tuple, dict] = {}


async def get_flow(symbol: str) -> FlowReport | None:
    """Trade-tape analytics; None (shown as н/д) when the tape cannot be fetched."""
    try:
        return analyze_flow(await stream.recent_trades(symbol))
    except RuntimeError:
        logger.warning("aggTrades unavailable for %s", symbol)
        return None


def cached_backtest(symbol: str, timeframe: str) -> BacktestResult | None:
    hit = _backtests.get((symbol, timeframe))
    return hit[1] if hit and time.monotonic() - hit[0] < BACKTEST_TTL else None


def store_backtest(symbol: str, timeframe: str, result: BacktestResult) -> None:
    _backtests[(symbol, timeframe)] = (time.monotonic(), result)
