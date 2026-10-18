"""Short-lived Binance Spot WebSocket analytics. No credentials or trading endpoints."""
import asyncio
import json
import logging
import time
from collections import deque
from dataclasses import dataclass

import httpx
import websockets

logger = logging.getLogger(__name__)
WS_ROOT = "wss://data-stream.binance.vision/ws"
REST_ROOT = "https://data-api.binance.vision"


@dataclass
class TradeFlow:
    count: int
    buy_quote: float
    sell_quote: float
    cvd_quote: float
    large_trades: list[dict]
    observed_seconds: float


@dataclass
class BookSummary:
    bids: list[tuple[float, float]]
    asks: list[tuple[float, float]]
    imbalance: float
    snapshot_time_ms: int
    last_update_id: int


def summarize_trades(trades: list[dict]) -> TradeFlow:
    """Binance aggTrade m=true means buyer is maker, hence seller was aggressor."""
    if not trades:
        return TradeFlow(0, 0.0, 0.0, 0.0, [], 0.0)
    notionals = [float(t["p"]) * float(t["q"]) for t in trades]
    buys = sells = 0.0
    large = []
    for trade, quote in zip(trades, notionals):
        is_sell_aggressor = bool(trade["m"])
        if is_sell_aggressor:
            sells += quote
        else:
            buys += quote
        large.append({"timestamp_ms": int(trade["T"]), "price": float(trade["p"]),
                          "quote": quote, "side": "sell" if is_sell_aggressor else "buy"})
    elapsed = max(0.0, (int(trades[-1]["T"]) - int(trades[0]["T"])) / 1000)
    large.sort(key=lambda t: t["quote"], reverse=True)
    return TradeFlow(len(trades), buys, sells, buys - sells, large[:10], elapsed)


class BinanceSpotStream:
    """Collect a short real-time aggTrade sample and synchronized depth snapshot via WS+REST."""
    def __init__(self, sample_seconds: float = 4.0, depth_limit: int = 100):
        self.sample_seconds = sample_seconds
        self.depth_limit = depth_limit
        self.cache_ttl = 20.0
        self._cache: dict[tuple, tuple[float, list[dict]]] = {}
        self._sem = asyncio.Semaphore(8)

    @staticmethod
    def _stream_symbol(symbol: str) -> str:
        return symbol.replace("/", "").replace(":", "").lower()

    async def recent_trades(self, symbol: str, minutes: int = 60, max_pages: int = 150) -> list[dict]:
        """aggTrades of the last `minutes` (a fixed time window, comparable across coins) from public REST.

        aggTrade ids are contiguous, so after one probe request the pages are fetched in parallel. If the coin is so
        active that the cap is hit first, the tape is shorter and the window flags (covered) report that honestly.
        """
        key = (self._stream_symbol(symbol).upper(), minutes)
        cached = self._cache.get(key)
        if cached and time.monotonic() - cached[0] < self.cache_ttl:
            return cached[1]
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                async def get(params: dict) -> list[dict]:
                    async with self._sem:
                        response = await client.get(f"{REST_ROOT}/api/v3/aggTrades", params={"symbol": key[0], **params})
                    response.raise_for_status()
                    return response.json()

                latest = await get({"limit": 1000})
                if not latest:
                    return []
                cutoff = int(latest[-1]["T"]) - minutes * 60_000
                collected: dict[int, dict] = {int(t["a"]): t for t in latest}
                oldest_id, oldest_ts = int(latest[0]["a"]), int(latest[0]["T"])
                span_min = max((int(latest[-1]["T"]) - oldest_ts) / 60_000, 1 / 600)
                fetched, batch_pages = 1, min(max_pages - 1, int(minutes / span_min * 1.3) + 2)
                while oldest_ts > cutoff and oldest_id > 0 and fetched < max_pages:
                    starts = [oldest_id - 1000 * k for k in range(1, batch_pages + 1) if oldest_id - 1000 * k >= -999]
                    batches = await asyncio.gather(*(get({"fromId": max(0, f), "limit": 1000}) for f in starts))
                    for batch in batches:
                        collected.update({int(t["a"]): t for t in batch})
                    fetched += len(starts)
                    oldest_id = min(collected)
                    oldest_ts = int(collected[oldest_id]["T"])
                    batch_pages = min(max_pages - fetched, 20)   # top up in smaller rounds if the estimate was short
                    if batch_pages <= 0:
                        break
        except Exception as exc:
            raise RuntimeError(f"Binance aggTrades недоступны: {exc}") from exc
        trades = [collected[k] for k in sorted(collected) if int(collected[k]["T"]) >= cutoff]
        self._cache[key] = (time.monotonic(), trades)
        return trades

    async def trades(self, symbol: str, minutes: int = 60) -> TradeFlow:
        return summarize_trades(await self.recent_trades(symbol, minutes))

    async def order_book(self, symbol: str) -> BookSummary:
        """Synchronize local depth from REST snapshot and WS deltas per Binance sequence rules."""
        stream_symbol = self._stream_symbol(symbol)
        url = f"{WS_ROOT}/{stream_symbol}@depth@100ms"
        buffered = deque(maxlen=1000)
        snapshot = None
        try:
            async with websockets.connect(url, open_timeout=5, close_timeout=2, ping_interval=20) as ws:
                # Buffer updates first to avoid losing events while obtaining snapshot.
                async def receive_until_snapshot():
                    while snapshot is None:
                        buffered.append(json.loads(await asyncio.wait_for(ws.recv(), timeout=5)))
                receiver = asyncio.create_task(receive_until_snapshot())
                async with httpx.AsyncClient(timeout=5) as client:
                    response = await client.get(f"{REST_ROOT}/api/v3/depth", params={"symbol": stream_symbol.upper(), "limit": self.depth_limit})
                    response.raise_for_status()
                    snapshot = response.json()
                receiver.cancel()
                try:
                    await receiver
                except asyncio.CancelledError:
                    pass
                last_id = int(snapshot["lastUpdateId"])
                bids = {float(p): float(q) for p, q in snapshot["bids"]}
                asks = {float(p): float(q) for p, q in snapshot["asks"]}

                def apply(event: dict) -> None:
                    for side, book in (("b", bids), ("a", asks)):
                        for price, qty in event[side]:
                            if float(qty) == 0:
                                book.pop(float(price), None)
                            else:
                                book[float(price)] = float(qty)

                # Binance Spot: drop events with u <= lastUpdateId, first must bridge, then U == prev u + 1.
                synced = False
                deadline = time.monotonic() + 3
                pending = deque(buffered)
                live_until = None
                while True:
                    if pending:
                        event = pending.popleft()
                    else:
                        if synced and live_until is None:
                            live_until = time.monotonic() + .5
                        limit = (live_until or deadline) - time.monotonic()
                        if limit <= 0:
                            break
                        try:
                            event = json.loads(await asyncio.wait_for(ws.recv(), timeout=limit))
                        except TimeoutError:
                            break
                    if int(event["u"]) <= last_id:
                        continue
                    if not synced:
                        if not int(event["U"]) <= last_id + 1 <= int(event["u"]):
                            raise RuntimeError("Стакан Binance рассинхронизирован; повторите запрос.")
                        synced = True
                    elif int(event["U"]) != last_id + 1:
                        raise RuntimeError("Пропущено обновление стакана Binance; повторите запрос.")
                    apply(event)
                    last_id = int(event["u"])
                if not synced:
                    raise RuntimeError("Стакан Binance рассинхронизирован; повторите запрос.")
                top_bids = sorted(bids.items(), reverse=True)[:self.depth_limit]
                top_asks = sorted(asks.items())[:self.depth_limit]
                bid_quote = sum(price * qty for price, qty in top_bids)
                ask_quote = sum(price * qty for price, qty in top_asks)
                denom = bid_quote + ask_quote
                return BookSummary(top_bids, top_asks, (bid_quote - ask_quote) / denom if denom else 0.0,
                                   int(time.time() * 1000), last_id)
        except Exception as exc:
            if isinstance(exc, RuntimeError):
                raise
            raise RuntimeError(f"Binance WebSocket стакан недоступен: {exc}") from exc
