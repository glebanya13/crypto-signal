"""Builds the content of each screen (text or chart) independent of Telegram transport."""
import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from io import BytesIO

from app.analytics.chart import render_chart
from app.analytics.levels import find_levels, pivot_points
from app.analytics.report import (
    build_backtest_report,
    build_levels_report,
    build_orderbook_report,
    build_report,
    build_signal_report,
    build_smc_report,
    build_ta_report,
    build_volume_report,
    build_whales_report,
)
from app.analytics.signal import compute_signal
from app.analytics.smc import analyze_smc
from app.analytics.technical import analyze_technical
from app.analytics.volume import volume_summary
from app.backtest import run_backtest
from app.telegram.services import cached_backtest, get_flow, provider, store_backtest, stream

MTF = ["15m", "1h", "4h", "1d"]
NEEDS_FLOW = {"analyze", "signal", "volume", "whales", "chart"}
BACKTEST_BARS = {"5m": 5000, "15m": 5000, "1h": 4000, "4h": 3000, "1d": 1500, "1w": 400}


@dataclass
class View:
    text: str | None = None
    photo: BytesIO | None = None
    caption: str = ""
    snapshot: dict | None = None


def _snapshot(candles, ticker, flow) -> dict:
    """Compact state used to tell the user what changed between two refreshes."""
    price = float(ticker.get("last") or candles.close.iloc[-1])
    support, resistance = find_levels(candles)
    smc = analyze_smc(candles)
    sig = compute_signal(candles, price, support, resistance, flow, volume_summary(candles, price, ticker)["vwap_24h"],
                         smc, analyze_technical(candles))
    last_event = smc.events[-1] if smc.events else None
    w1h = flow.windows[60] if flow else None
    return {
        "time": datetime.now(UTC), "price": price, "direction": sig.direction, "score": sig.score,
        "actionable": sig.actionable, "smc_trend": smc.trend,
        "last_event": (last_event.kind, last_event.direction, str(last_event.ts)) if last_event else None,
        "delta_1h": w1h.delta if w1h and w1h.covered else None,
        "top_whale_ms": flow.whales[0].ts_ms if flow and flow.whales else None,
        "spike": bool(flow and flow.spike),
    }


async def build_view(action: str, symbol: str, timeframe: str) -> View:
    if action == "whales":
        flow = await get_flow(symbol)
        return View(build_whales_report(symbol, flow, datetime.now(UTC)))
    if action == "orderbook":
        return View(build_orderbook_report(symbol, await stream.order_book(symbol), stream.depth_limit))
    if action == "backtest":
        result = cached_backtest(symbol, timeframe)
        if result is None:
            history = await provider.fetch_history(symbol, timeframe, BACKTEST_BARS[timeframe])
            if len(history) < 400:
                raise ValueError("Для бэктеста на этом таймфрейме у пары слишком мало истории.")
            result = await asyncio.to_thread(run_backtest, history, symbol, timeframe)
            store_backtest(symbol, timeframe, result)
        return View(build_backtest_report(result, datetime.now(UTC)))

    flow_task = get_flow(symbol) if action in NEEDS_FLOW else asyncio.sleep(0)
    (candles, ticker, fetched_at), flow = await asyncio.gather(provider.fetch(symbol, timeframe), flow_task)
    snapshot = await asyncio.to_thread(_snapshot, candles, ticker, flow)
    if action == "chart":
        image = await asyncio.to_thread(render_chart, candles, symbol, timeframe, flow)
        return View(photo=image, caption=f"{symbol} · {timeframe} · закрытые свечи · Binance Spot", snapshot=snapshot)
    if action == "levels":
        daily = await provider.fetch_candles(symbol, "1d", 5)
        return View(build_levels_report(symbol, candles, ticker, fetched_at, timeframe, pivot_points(daily)), snapshot=snapshot)
    if action == "smc":
        return View(build_smc_report(symbol, candles, ticker, fetched_at, timeframe), snapshot=snapshot)
    if action == "ta":
        frames = await asyncio.gather(*(provider.fetch_candles(symbol, tf) for tf in MTF))
        return View(build_ta_report(symbol, candles, ticker, fetched_at, timeframe, dict(zip(MTF, frames, strict=True))),
                    snapshot=snapshot)
    if action == "signal":
        return View(build_signal_report(symbol, candles, ticker, fetched_at, timeframe, flow), snapshot=snapshot)
    if action == "volume":
        return View(build_volume_report(symbol, candles, ticker, fetched_at, timeframe, flow), snapshot=snapshot)
    return View(build_report(symbol, candles, ticker, fetched_at, timeframe, flow), snapshot=snapshot)


def whats_new(prev: dict | None, cur: dict | None) -> str:
    """Plain-language diff between two snapshots of the same screen."""
    if not prev or not cur:
        return ""
    minutes = max(1, int((cur["time"] - prev["time"]).total_seconds() // 60))
    ago = f"{minutes} мин назад" if minutes < 90 else f"{minutes // 60} ч назад"
    items = []
    change = (cur["price"] / prev["price"] - 1) * 100
    items.append(f"Цена: {prev['price']:,.6g} → {cur['price']:,.6g} ({change:+.2f}%)")
    names = {"long": "LONG", "short": "SHORT", "neutral": "нет перевеса"}
    if prev["direction"] != cur["direction"]:
        items.append(f"🔔 Направление изменилось: {names[prev['direction']]} → {names[cur['direction']]}")
    elif abs(cur["score"] - prev["score"]) >= 10:
        items.append(f"Сила сигнала {names[cur['direction']]}: {abs(prev['score']):.0f} → {abs(cur['score']):.0f} из 100")
    if prev["actionable"] != cur["actionable"]:
        items.append("✅ Сделка теперь проходит фильтр риск/доходность" if cur["actionable"]
                     else "⚠️ Сделка перестала проходить фильтр риск/доходность")
    if prev["smc_trend"] != cur["smc_trend"]:
        label = {"bull": "бычья", "bear": "медвежья", "range": "не определена"}
        items.append(f"🔔 Структура рынка: {label[prev['smc_trend']]} → {label[cur['smc_trend']]}")
    if cur["last_event"] and cur["last_event"] != prev["last_event"]:
        kind, direction, _ = cur["last_event"]
        items.append(f"🔔 Новое событие структуры: {kind} {'вверх' if direction == 'bull' else 'вниз'}")
    if cur["top_whale_ms"] and cur["top_whale_ms"] != prev["top_whale_ms"]:
        items.append("🐋 Появилась новая крупнейшая сделка в окне (смотрите «Крупные сделки»)")
    if cur["spike"] and not prev["spike"]:
        items.append("⚡ Появился аномальный всплеск оборота")
    if cur["delta_1h"] is not None and prev["delta_1h"] is not None and (cur["delta_1h"] > 0) != (prev["delta_1h"] > 0):
        items.append("🔔 Delta за час сменила знак: " + ("покупатели перехватили инициативу" if cur["delta_1h"] > 0
                                                         else "продавцы перехватили инициативу"))
    quiet = len(items) == 1
    head = f"🆕 <b>Что нового с прошлого обновления ({ago}):</b>"
    body = "\n".join(f"• {x}" for x in items)
    return head + "\n" + body + ("\nСущественных изменений нет." if quiet else "") + "\n\n"
