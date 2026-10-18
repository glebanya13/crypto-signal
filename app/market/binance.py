import asyncio
import time
from datetime import UTC, datetime

import ccxt.async_support as ccxt
import pandas as pd

REST_ROOT = "https://data-api.binance.vision"
TIMEFRAMES = {"5m": 300, "15m": 900, "1h": 3600, "4h": 14400, "1d": 86400, "1w": 604800}
CACHE_TTL = 15.0


class MarketDataError(RuntimeError):
    """Market data could not be fetched or validated."""


def drop_unclosed(frame: pd.DataFrame, timeframe: str, now: datetime | None = None) -> pd.DataFrame:
    """OHLCV from exchanges ends with the still-forming candle; analytics must use closed candles only."""
    if frame.empty:
        return frame
    now = now or datetime.now(UTC)
    close_time = frame.timestamp.iloc[-1] + pd.Timedelta(seconds=TIMEFRAMES[timeframe])
    return frame.iloc[:-1].reset_index(drop=True) if close_time > pd.Timestamp(now) else frame


def to_frame(raw: list) -> pd.DataFrame:
    frame = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
    return frame.astype({c: float for c in ["open", "high", "low", "close", "volume"]})


class BinanceSpotProvider:
    """Public Binance Spot endpoints through CCXT; no keys and no trading access."""

    def __init__(self) -> None:
        # data-api.binance.vision - публичное зеркало рыночных данных, не блокируется по региону
        self.exchange = ccxt.binance({
            "enableRateLimit": True,
            "timeout": 30000,
            "options": {"defaultType": "spot", "fetchCurrencies": False, "fetchMarkets": ["spot"]},
            "urls": {"api": {"public": f"{REST_ROOT}/api/v3", "private": f"{REST_ROOT}/api/v3"}},
        })
        self._cache: dict[tuple, tuple[float, object]] = {}
        self._inflight: dict[tuple, asyncio.Future] = {}

    async def close(self) -> None:
        await self.exchange.close()

    @staticmethod
    def normalize_symbol(text: str, quote: str = "USDT") -> str:
        raw = text.strip().upper().replace(" ", "")
        if "/" in raw:
            base, provided_quote = raw.split("/", 1)
            if not base or not provided_quote:
                raise ValueError("Используйте формат BTC/USDT.")
            return f"{base}/{provided_quote}"
        if raw.endswith(quote) and len(raw) > len(quote):
            raw = raw[: -len(quote)]
        if not raw.isalnum():
            raise ValueError("Укажите тикер, например SOL или SOL/USDT.")
        return f"{raw}/{quote}"

    async def _retry(self, call, attempts: int = 3):
        """Retry transient network errors; geo-blocks (451) and exchange errors are not retried."""
        for attempt in range(attempts):
            try:
                return await call()
            except ccxt.NetworkError as exc:
                if "451" in str(exc) or attempt == attempts - 1:
                    raise
                await asyncio.sleep(0.7 * 2 ** attempt)

    async def _cached(self, key: tuple, factory):
        """TTL cache that also merges concurrent identical requests into one."""
        hit = self._cache.get(key)
        if hit and time.monotonic() - hit[0] < CACHE_TTL:
            return hit[1]
        if key in self._inflight:
            return await self._inflight[key]
        future = asyncio.get_running_loop().create_future()
        self._inflight[key] = future
        try:
            value = await factory()
            self._cache[key] = (time.monotonic(), value)
            future.set_result(value)
            return value
        except BaseException as exc:
            future.set_exception(exc)
            future.exception()  # mark retrieved when nobody else awaits it
            raise
        finally:
            self._inflight.pop(key, None)

    async def _check_pair(self, symbol: str) -> None:
        await self._retry(self.exchange.load_markets)
        if symbol not in self.exchange.markets or not self.exchange.markets[symbol].get("spot"):
            raise MarketDataError(f"Пара {symbol} недоступна на Binance Spot.")

    async def fetch(self, symbol: str, timeframe: str = "1h") -> tuple[pd.DataFrame, dict, datetime]:
        """(closed candles, live ticker, fetch time). Cached for a few seconds."""
        if timeframe not in TIMEFRAMES:
            raise ValueError("Таймфрейм должен быть 5m, 15m, 1h, 4h, 1d или 1w.")
        return await self._cached(("fetch", symbol, timeframe), lambda: self._fetch(symbol, timeframe))

    async def _fetch(self, symbol: str, timeframe: str) -> tuple[pd.DataFrame, dict, datetime]:
        try:
            await self._check_pair(symbol)
            raw, ticker = await asyncio.gather(
                self._retry(lambda: self.exchange.fetch_ohlcv(symbol, timeframe, limit=500)),
                self._retry(lambda: self.exchange.fetch_ticker(symbol)))
            frame = drop_unclosed(to_frame(raw), timeframe)
            if len(frame) < 30:
                raise MarketDataError("Биржа вернула недостаточно свечей для анализа.")
            return frame, ticker, datetime.now(UTC)
        except MarketDataError:
            raise
        except (ccxt.NetworkError, ccxt.ExchangeError, ValueError) as exc:
            raise MarketDataError(f"Не удалось получить актуальные данные Binance: {exc}") from exc

    async def fetch_candles(self, symbol: str, timeframe: str, limit: int = 300) -> pd.DataFrame:
        """Closed OHLCV only (no ticker) - used for multi-timeframe views."""
        async def load() -> pd.DataFrame:
            try:
                await self._check_pair(symbol)
                raw = await self._retry(lambda: self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit))
            except (ccxt.NetworkError, ccxt.ExchangeError) as exc:
                raise MarketDataError(f"Не удалось получить свечи {timeframe}: {exc}") from exc
            return drop_unclosed(to_frame(raw), timeframe)
        return await self._cached(("candles", symbol, timeframe, limit), load)

    async def fetch_history(self, symbol: str, timeframe: str, bars: int) -> pd.DataFrame:
        """Page backwards-in-time history (1000 candles per request) for backtests."""
        try:
            await self._check_pair(symbol)
            step_ms = TIMEFRAMES[timeframe] * 1000
            since = int(time.time() * 1000) - bars * step_ms
            rows: list = []
            while len(rows) < bars:
                batch = await self._retry(lambda s=since: self.exchange.fetch_ohlcv(symbol, timeframe, since=s, limit=1000))
                if not batch:
                    break
                rows += batch
                since = int(batch[-1][0]) + step_ms
                if len(batch) < 1000:
                    break
        except (ccxt.NetworkError, ccxt.ExchangeError) as exc:
            raise MarketDataError(f"Не удалось получить историю: {exc}") from exc
        frame = to_frame(rows).drop_duplicates("timestamp").reset_index(drop=True)
        return drop_unclosed(frame, timeframe)
